#train_densenet_detector.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time

class DenseNetDetector(nn.Module):
    def __init__(self, num_classes=9):
        super(DenseNetDetector, self).__init__()
        
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(weights='DEFAULT')
        
        # Modify first conv layer for the input size
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the classifier
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        
        # Add detection head with improved architecture
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        features = self.densenet(x)
        return self.detection_head(features)

class BoneAbnormalityDataset(Dataset):
    def __init__(self, base_path, csv_file, split_type='train', image_size=416, transform=None):
        """Reduced image size and optimized data loading"""
        self.base_path = Path(base_path)
        self.images_path = self.base_path / 'yolov5' / 'images' / split_type
        self.labels_path = self.base_path / 'yolov5' / 'labels' / split_type
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.transform = transform
        self.num_classes = 9
        
        # Pre-filter valid samples
        self.valid_samples = self._get_valid_samples()
        print(f"Found {len(self.valid_samples)} valid samples")
        
    def _get_valid_samples(self):
        """Pre-filter valid samples to avoid checking during training"""
        valid_samples = []
        for idx, row in self.df.iterrows():
            img_path = self.images_path / f"{row['filestem']}.png"
            label_path = self.labels_path / f"{row['filestem']}.txt"
            if img_path.exists() and label_path.exists():
                valid_samples.append({
                    'idx': idx,
                    'img_path': str(img_path),
                    'label_path': str(label_path)
                })
        return valid_samples
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Read image
        image = cv2.imread(sample['img_path'])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read labels
        try:
            with open(sample['label_path'], 'r') as f:
                labels = [list(map(float, line.strip().split())) for line in f]
        except Exception as e:
            print(f"Error reading labels from {sample['label_path']}: {e}")
            labels = []
        
        labels = np.array(labels) if labels else np.zeros((0, 5))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert labels to target format
        target = self.prepare_target(labels)
        
        return image, target
    
    def prepare_target(self, labels):
        """Optimized target preparation"""
        grid_size = self.image_size // 32
        target = torch.zeros((self.num_classes, grid_size, grid_size))
        
        if len(labels) > 0:
            for label in labels:
                class_id, x, y = int(label[0]), int(label[1] * grid_size), int(label[2] * grid_size)
                if 0 <= class_id < self.num_classes and 0 <= x < grid_size and 0 <= y < grid_size:
                    target[class_id, y, x] = 1
        
        return target

def train_model():
    # Paths
    base_path = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset")
    train_csv = str(base_path / "train_data.csv")
    valid_csv = str(base_path / "valid_data.csv")
    
    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reduced parameters
    num_epochs = 20
    batch_size = 32  # Increased batch size
    learning_rate = 0.001
    image_size = 416  # Reduced image size
    
    # Transforms - simplified and optimized
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BoneAbnormalityDataset(
        base_path, train_csv, 'train', 
        image_size=image_size, 
        transform=train_transform
    )
    valid_dataset = BoneAbnormalityDataset(
        base_path, valid_csv, 'valid', 
        image_size=image_size, 
        transform=valid_transform
    )
    
    # Create dataloaders with optimized settings
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Adjust based on CPU cores
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model
    print("Initializing model...")
    model = DenseNetDetector(num_classes=9).to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Training loop with progress bars and timing
    best_valid_loss = float('inf')
    early_stopping_counter = 0
    patience = 3
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, targets in train_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_bar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
        
        with torch.no_grad():
            for images, targets in valid_bar:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                valid_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        valid_loss /= len(valid_loader)
        
        # Learning rate scheduling and early stopping
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_densenet_detector.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
        # Early stopping
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Fatal error: {e}")