import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import wandb
from PIL import Image
from tqdm import tqdm
#from features import extract_features
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
import random
import numpy as np
import subprocess

#path_to_core = "/Users/ryuma/Desktop/RebarApp/"
path_to_core = "/Users/wis/Desktop/code/"
data_dir = f"{path_to_core}core/diagram_classification/data"
batch_print_freq = 10
num_epochs=1000
batch_size=4
learning_rate=3e-5

class FloorPlanDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # Get class names from folder names
        self.classes = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.transform = transforms.ToTensor()  # Just converts PIL to tensor (0-1) and CHW

        # Collect all image paths and their labels
        self.images = []
        self.labels = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.png'):  # Assuming PNG format
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_dir.name])

        self.class_image_counts = {cls: 0 for cls in self.classes}
        self.calculate_image_counts()
        self.class_weights = self.calculate_class_weights()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Random scale factor between 0.7 and 1.2
        scale_factor = np.random.uniform(0.7, 1.2)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Resize with bilinear interpolation
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Convert to tensor
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    def calculate_image_counts(self):
        """Calculate the total number of tiles for each class"""
        for _, label in zip(self.images, self.labels):
            self.class_image_counts[self.classes[label]] += 1
            
    def calculate_class_weights(self):
        """Calculate inverse frequency weights for each class"""
        total_tiles = sum(self.class_image_counts.values())
        weights = {}
        for cls in self.classes:
            image_freq = self.class_image_counts[cls] / total_tiles
            weights[cls] = 1.0 / (image_freq + 1e-6)  # add epsilon to prevent division by zero
        
        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        # Convert to tensor
        weight_tensor = torch.FloatTensor([weights[cls] for cls in self.classes])
        return weight_tensor
    
def tile_image(images, tile_size=(64, 64), max_tiles=10):
    """Split batch of tensor images into tiles and sample at most max_tiles per image"""
    # Handle both single image and batch inputs
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    
    B, C, H, W = images.shape
    tile_h, tile_w = tile_size
    
    # Calculate number of tiles needed
    num_tiles_h = (H + tile_h - 1) // tile_h
    num_tiles_w = (W + tile_w - 1) // tile_w
    total_tiles = num_tiles_h * num_tiles_w
    
    # Create all possible tile positions for each image in batch
    h_positions = torch.arange(num_tiles_h).view(-1, 1).repeat(1, num_tiles_w).flatten()
    w_positions = torch.arange(num_tiles_w).repeat(num_tiles_h)
    
    # Randomly select positions if needed
    if total_tiles > max_tiles:
        # Create random indices for each image in batch
        indices = torch.stack([
            torch.randperm(total_tiles)[:max_tiles] 
            for _ in range(B)
        ])
        h_positions = h_positions[indices]  # Shape: [B, max_tiles]
        w_positions = w_positions[indices]  # Shape: [B, max_tiles]
    else:
        # Expand positions for batch
        h_positions = h_positions.unsqueeze(0).repeat(B, 1)  # Shape: [B, total_tiles]
        w_positions = w_positions.unsqueeze(0).repeat(B, 1)  # Shape: [B, total_tiles]
    
    # Convert positions to pixel coordinates
    y_coords = h_positions * tile_h  # Shape: [B, N]
    x_coords = w_positions * tile_w  # Shape: [B, N]
    
    # Create empty tensor for tiles
    N = h_positions.size(1)  # number of tiles per image
    tiles = torch.ones(B, N, C, tile_h, tile_w, device=images.device)
    
    # Extract and pad tiles in one operation
    for b in range(B):
        for n in range(N):
            y, x = y_coords[b, n], x_coords[b, n]
            bottom = min(y + tile_h, H)
            right = min(x + tile_w, W)
            tiles[b, n, :, :(bottom-y), :(right-x)] = images[b, :, y:bottom, x:right]
    
    return tiles  # Shape: [B, N, C, H, W]

def custom_collate(batch):
    """Custom collate function to handle images of different sizes"""
    # Find the maximum dimensions in the batch
    max_h = max(img.shape[1] for img, _ in batch)
    max_w = max(img.shape[2] for img, _ in batch)
    
    # Pad all images to the maximum size
    padded_images = []
    labels = []
    
    for img, label in batch:
        # Calculate padding
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # Pad the image with white (1.0)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
        padded_images.append(padded_img)
        labels.append(label)
    
    # Stack all images and labels
    images = torch.stack(padded_images)
    labels = torch.tensor(labels)
    
    return images, labels
    
class TiledImageClassifier(nn.Module):
    def __init__(self, num_classes = 7, batch_size=64, num_heads=4, class_names=None):
        super(TiledImageClassifier, self).__init__()

        self.VAE = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
        self.max_tiles = 10

        self.VAE.requires_grad_(False)
        self.VAE.eval()
        self.transform = transforms.Compose([ # Convert numpy to tensor
            transforms.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 3 else x), 
            transforms.Lambda(lambda x: x[:3] if x.shape[1] == 4 else x),  # Take only RGB channels if RGBA
            transforms.Normalize([0.5], [0.5]),
        ]) 
  
        self.batch_size = batch_size
        vae_latent_channels = self.VAE.config.latent_channels  # Usually 4 for SDXL
        self.feature_dim = vae_latent_channels * 64  # Adjust based on your tile size
        print(f"Feature dim from VAE: {self.feature_dim}")
        self.bn1 = nn.LayerNorm(self.feature_dim)

        n_additional = self.feature_dim // 2
        self.learned_features = nn.Sequential(
            nn.Linear(self.feature_dim, n_additional),
            nn.GELU(),
            nn.LayerNorm(n_additional),
        )

        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]

        self.num_heads = num_heads
        feature_per_head = (self.feature_dim + n_additional) // num_heads
        
        # Single linear layer with grouped parameters for all heads
        self.tile_attention_heads = nn.ModuleList([
            nn.Linear(feature_per_head, 1) for _ in range(num_heads)
        ])
        
        self.classifier = nn.Linear(self.feature_dim + n_additional, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def forward(self, images):
        B = images.shape[0]
        # Split images into tiles
        tiles = tile_image(images, max_tiles=self.max_tiles)  # [B, N, C, H, W]
        N = tiles.shape[1]  # number of tiles per image
        
        # Reshape tiles to process all tiles from all images at once
        tiles = tiles.view(B * N, *tiles.shape[2:])  # [B*N, C, H, W]
        
        tiles = self.transform(tiles)
        with torch.no_grad():
            features = self.VAE.encode(tiles).latent_dist.sample()
        
        # Reshape features
        features = features.view(features.shape[0], -1)  # [B*N, -1]
        features = self.bn1(features)
        learned_features = self.learned_features(features)
        features = torch.cat([features, learned_features], dim=1)
        
        # Get logits and attention weights for all tiles
        logits = self.classifier(features)  # [B*N, num_classes]
        # Split features into heads
        H = self.num_heads
        D = features.shape[1] // H
        head_features = features.view(B * N, H, D)  # [B*N, H, D]
        head_features = head_features.transpose(0, 1)  # [H, B*N, D]
        
        # Calculate attention weights for each head
        tile_weights = []
        for h in range(H):
            head_weight = self.tile_attention_heads[h](head_features[h])  # [B*N, 1]
            tile_weights.append(head_weight)
        tile_weights = torch.cat(tile_weights, dim=1)  # [B*N, H]
        tile_weights = tile_weights.view(B, N, H)  # [B, N, H]
        tile_weights = tile_weights.permute(0, 2, 1)  # [B, H, N]
        tile_weights = F.softmax(tile_weights, dim=2)  # [B, H, N]
        
        # Reshape logits
        logits = logits.view(B, N, -1)  # [B, N, num_classes]
        
        # Compute weighted logits for all heads at once
        head_outputs = torch.bmm(tile_weights, logits)  # [B, H, num_classes]
        
        # Average across heads
        final_logits = torch.mean(head_outputs, dim=1)  # [B, num_classes]
        
        return final_logits
    
    def predict(self, image):
        """Separate method for inference"""
        self.eval()
        self.max_tiles = 200
        with torch.no_grad():
            weighted_logits = self.forward(image)
            predicted_class = weighted_logits.argmax().item()
            confidence = F.softmax(weighted_logits, dim=0)[predicted_class].item()
            
            return {
                'class_id': predicted_class,
                'class_name': self.model.config.id2label[predicted_class],
                'confidence': confidence,
                'logits': weighted_logits,
            }

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
    

def train_model(data_dir, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize wandb
    wandb.init(project="floor-plan-classification", name="vae_attention_linear")

    # Create weights directory
    weights_dir = f"{path_to_core}core/diagram_classification/weights"
    Path(weights_dir).mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = FloorPlanDataset(data_dir)
    class_weights = dataset.class_weights.to(device)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=custom_collate
    )
    # Initialize model
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")

    model = TiledImageClassifier(batch_size=batch_size).to(device)
    criterion = FocalLoss(gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        model.train()
        model.max_tiles = 20
        train_loss = 0
        correct = 0
        total = 0
        max_grad_norm = 10.0

        # for batch_idx, (images, targets) in enumerate(train_loader):
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):

            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss = loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()       # Update model parameters


            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
            train_loss += loss.item()

            # Log metrics only after updates
            wandb.log({
                "train/step": epoch * (len(train_loader)) + (batch_idx + 1),
                "train/batch_loss": loss.item(),  # Scale back loss
                "train/batch_accuracy": accuracy,
            })
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 200 == 0:
            checkpoint_path = f"{path_to_core}core/diagram_classification/weights/vae_attention_linear_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': model.class_names,  # Save class names
                'num_classes': len(dataset.classes),
                'batch_size': batch_size,
                'num_heads': model.num_heads,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Optional: still save the latest checkpoint each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{path_to_core}core/diagram_classification/weights/vae_attention_linear_last.pth")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Add per-class tracking
        class_correct = {cls: 0 for cls in dataset.classes}
        class_total = {cls: 0 for cls in dataset.classes}

        model.max_tiles = 20
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Calculate validation metrics
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_accuracy = 100. * val_correct / val_total
                val_loss += loss.item()

                # Track per-class accuracy
                for pred, target in zip(predicted, targets):
                    class_name = dataset.classes[target]
                    class_total[class_name] += 1
                    if pred == target:
                        class_correct[class_name] += 1

                # Log every validation step
                wandb.log({
                    "val/step": epoch * len(val_loader) + batch_idx,
                    "val/batch_loss": loss.item(),
                    "val/batch_accuracy": val_accuracy,
                })
        
        # Calculate and log per-class accuracies
        for class_name in ['table']:
            if class_total[class_name] > 0:  # Avoid division by zero
                class_accuracy = 100. * class_correct[class_name] / class_total[class_name]
                wandb.log({
                    f"val/accuracy_{class_name}": class_accuracy,
                })

        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss / len(train_loader),
            "train/epoch_accuracy": 100. * correct / total,
            "val/epoch_loss": val_loss / len(val_loader),
            "val/epoch_accuracy": 100. * val_correct / val_total
        })

    # Save final model
    final_path = f"{path_to_core}core/diagram_classification/weights/vae_attention_linear.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': model.class_names,  # Save class names
        'num_classes': num_classes,
        'batch_size': batch_size,
        'num_heads': model.num_heads,
    }, final_path)

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with saved parameters
    model = TiledImageClassifier(
        num_classes=checkpoint['num_classes'],
        batch_size=checkpoint['batch_size'],
        num_heads=checkpoint['num_heads'],
        class_names=checkpoint['class_names']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

"""
model = load_model('path/to/weights/vae_attention_linear.pth')
image = ... # Load and preprocess your image
result = model.predict(image)
print(f"Predicted class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2f}")
"""

if __name__ == "__main__":
    caffeine_process = subprocess.Popen(['caffeinate', '-i'])
    try:
        train_model(data_dir, num_epochs, batch_size, learning_rate)
    finally:
        caffeine_process.terminate()