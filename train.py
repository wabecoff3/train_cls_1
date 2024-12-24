import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import wandb
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
import random
import numpy as np

path_to_core = "/home/wis/training/train_cls_1"
data_dir = "dataset"
batch_print_freq = 10
num_epochs = 5000
batch_size = 16
learning_rate = 2e-5
feature_cache_dir = Path(f"{path_to_core}/feature_cache")

class ImageDataset(Dataset):
    def __init__(self, root_dir, training=True):
        self.root_dir = Path(root_dir)
        self.classes = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
        self.training = training
        
        # Collect all image paths and their labels
        self.images = []
        self.labels = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_dir.name])
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.training and (random.random() > 0.4):
            scale_factor = np.random.uniform(0.7, 1.2)
            new_width = int(image.size[0] * scale_factor)
            new_height = int(image.size[1] * scale_factor)
            image = image.resize((new_width, new_height), Image.BILINEAR)
        
        image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label

    def calculate_class_weights(self):
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total = len(self.labels)
        weights = torch.FloatTensor([total / (len(self.classes) * count) 
                                   for count in class_counts.values()])
        return weights / weights.sum()

class FeatureDataset(Dataset):
    def __init__(self, features, labels, training=True):
        self.features = features  # Shape: [N, num_tiles, feature_dim]
        self.labels = labels
        self.training = training
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TiledFeatureClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=7, num_heads=4, class_names=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.bn1 = nn.LayerNorm(self.feature_dim)
        
        n_additional = self.feature_dim // 2
        self.learned_features = nn.Sequential(
            nn.Linear(self.feature_dim, n_additional),
            nn.GELU(),
            nn.LayerNorm(n_additional),
            nn.Linear(n_additional, n_additional),
            nn.GELU(),
            nn.LayerNorm(n_additional)
        )

        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]
        self.num_heads = num_heads
        feature_per_head = (self.feature_dim + n_additional) // num_heads
        
        self.tile_attention_heads = nn.ModuleList([
            nn.Linear(feature_per_head, 1) for _ in range(num_heads)
        ])
        
        self.classifier = nn.Linear(self.feature_dim + n_additional, num_classes)

    def forward(self, features):
        B, N, D = features.shape
        features = features.view(B * N, D)
        features = self.bn1(features)
        learned_features = self.learned_features(features)
        features = torch.cat([features, learned_features], dim=1)
        
        logits = self.classifier(features)
        
        H = self.num_heads
        D = features.shape[1] // H
        head_features = features.view(B * N, H, D)
        head_features = head_features.transpose(0, 1)
        
        tile_weights = []
        for h in range(H):
            head_weight = self.tile_attention_heads[h](head_features[h])
            tile_weights.append(head_weight)
        tile_weights = torch.cat(tile_weights, dim=1)
        tile_weights = tile_weights.view(B, N, H)
        tile_weights = tile_weights.permute(0, 2, 1)
        tile_weights = F.softmax(tile_weights, dim=2)
        
        logits = logits.view(B, N, -1)
        head_outputs = torch.bmm(tile_weights, logits)
        final_logits = torch.mean(head_outputs, dim=1)
        
        return final_logits

def tile_image(images, tile_size=(64, 64), max_tiles=40):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    
    B, C, H, W = images.shape
    tile_h, tile_w = tile_size
    
    num_tiles_h = (H + tile_h - 1) // tile_h
    num_tiles_w = (W + tile_w - 1) // tile_w
    total_tiles = num_tiles_h * num_tiles_w
    
    h_positions = torch.arange(num_tiles_h).view(-1, 1).repeat(1, num_tiles_w).flatten()
    w_positions = torch.arange(num_tiles_w).repeat(num_tiles_h)
    
    if total_tiles > max_tiles:
        indices = torch.stack([
            torch.randperm(total_tiles)[:max_tiles] 
            for _ in range(B)
        ])
        h_positions = h_positions[indices]
        w_positions = w_positions[indices]
    else:
        h_positions = h_positions.unsqueeze(0).repeat(B, 1)
        w_positions = w_positions.unsqueeze(0).repeat(B, 1)
    
    y_coords = h_positions * tile_h
    x_coords = w_positions * tile_w
    
    N = h_positions.size(1)
    tiles = torch.ones(B, N, C, tile_h, tile_w, device=images.device)
    
    for b in range(B):
        for n in range(N):
            y, x = y_coords[b, n], x_coords[b, n]
            bottom = min(y + tile_h, H)
            right = min(x + tile_w, W)
            tiles[b, n, :, :(bottom-y), :(right-x)] = images[b, :, y:bottom, x:right]
    
    return tiles

def extract_and_cache_features(dataset, vae, batch_size=32, device='cuda', force_regenerate=False, split='train'):
    """Extract VAE features and cache them to disk"""
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = feature_cache_dir / f"features_{split}.pt"
    
    if cache_file.exists() and not force_regenerate:
        print(f"Loading cached {split} features...")
        cache_data = torch.load(cache_file)
        return cache_data['features'], cache_data['labels']
    
    if cache_file.exists():
        cache_file.unlink()  # Delete existing cache file
    
    print(f"Extracting {split} features...")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,
        collate_fn=custom_collate
    )
    all_features = []
    all_labels = []
    
    vae = vae.to(device)
    vae.eval()
    
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # Stack images into a batch with padding
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            
            batch_images = torch.zeros(len(images), 3, max_h, max_w, device=device)
            for i, img in enumerate(images):
                batch_images[i, :, :img.shape[1], :img.shape[2]] = img
            
            batch_images = transform(batch_images)
            
            tiles = tile_image(batch_images)
            B, N = tiles.shape[:2]
            tiles = tiles.view(B * N, *tiles.shape[2:])
            
            features = vae.encode(tiles).latent_dist.sample()
            features = features.view(B, N, -1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    # Cache the features
    torch.save({
        'features': features,
        'labels': labels
    }, cache_file)
    
    return features, labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
def custom_collate(batch):
    """Custom collate function to handle different-sized images"""
    images = []
    labels = []
    
    for image, label in batch:
        images.append(image)
        labels.append(label)
    
    # Stack labels normally since they're all the same size
    labels = torch.stack(labels)
    
    # Don't stack images - keep them as a list
    return images, labels

def train_model(data_dir, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    weights_dir = Path(f"{path_to_core}/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="floor-plan-classification", name="efficient_vae_attention")

    # Create dataset and load VAE
    full_dataset = ImageDataset(data_dir)
    
    # Create train-val split indices once
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), 
        [train_size, val_size]
    )

    # Create train and val datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
    
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    
    # Initialize model
    feature_dim = vae.config.latent_channels * 64
    num_classes = len(full_dataset.classes)
    model = TiledFeatureClassifier(feature_dim, num_classes, class_names=full_dataset.classes).to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=3e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Extract initial validation features (these won't change)
    val_features, val_labels = extract_and_cache_features(
        val_dataset, vae, batch_size=batch_size, device=device, force_regenerate=True, split='val'
    )
    val_feature_dataset = FeatureDataset(val_features, val_labels, training=False)
    val_loader = DataLoader(val_feature_dataset, batch_size=batch_size, num_workers=4)

    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        # Regenerate training features every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch}: Regenerating training features...")
            train_features, train_labels = extract_and_cache_features(
                train_dataset, vae, batch_size=batch_size, device=device, force_regenerate=True, split='train'
            )
            train_feature_dataset = FeatureDataset(train_features, train_labels, training=True)
            train_loader = DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader)):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            train_loss += loss.item()

            if (batch_idx + 1) % batch_print_freq == 0:
                wandb.log({
                    "train/step": epoch * len(train_loader) + batch_idx,
                    "train/batch_loss": loss.item(),
                    "train/batch_accuracy": 100. * train_correct / train_total,
                })

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader):
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_loss += loss.item()

        # Calculate epoch metrics
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = 100. * train_correct / train_total
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total

        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_epoch_loss,
            "train/epoch_accuracy": train_epoch_acc,
            "val/epoch_loss": val_epoch_loss,
            "val/epoch_accuracy": val_epoch_acc
        })

        # Save checkpoint if validation accuracy improves
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_epoch_acc,
                'class_names': model.class_names,
                'num_classes': num_classes,
                'feature_dim': feature_dim,
            }, f"{weights_dir}/efficient_vae_attention_best.pth")

        # Save periodic checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_epoch_acc,
                'class_names': model.class_names,
                'num_classes': num_classes,
                'feature_dim': feature_dim,
            }, f"{weights_dir}/efficient_vae_attention_epoch_{epoch+1}.pth")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}%")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

if __name__ == "__main__":
    train_model(data_dir, num_epochs, batch_size, learning_rate)