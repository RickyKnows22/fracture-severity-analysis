# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# # Update this import
# from torch.cuda.amp import GradScaler
# import torch.amp  # Add this import
# import timm
# import cv2
# import numpy as np
# from tqdm import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from datetime import datetime
# from pathlib import Path
# from sklearn.metrics import classification_report
# from fracture_model import FracturePatternModel, FractureLoss
# from dataset_handler import FracturePatternDataset


# class FractureSeverityDataset(Dataset):
#     def __init__(self, img_dir, label_dir, transform=None, mode='train'):
#         self.img_dir = Path(img_dir)
#         self.label_dir = Path(label_dir)
#         self.mode = mode
#         self.transform = transform if transform else self.get_default_transforms(mode)
#         self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
#         # Create a mapping of fracture characteristics to severity levels
#         self.severity_mapping = {
#             'area_thresholds': [0.02, 0.05, 0.1, 0.15],  # relative to image size
#             'confidence_thresholds': [0.3, 0.5, 0.7, 0.85]
#         }
    
#     def get_default_transforms(self, mode):
#         if mode == 'train':
#             return A.Compose([
#                 A.Resize(384, 384),
#                 A.OneOf([
#                     A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
#                     A.MultiplicativeNoise(p=0.8),
#                 ], p=0.5),
#                 A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
#                 A.OneOf([
#                     A.OpticalDistortion(p=0.4),
#                     A.GridDistortion(p=0.4),
#                     A.ElasticTransform(p=0.4),
#                 ], p=0.3),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])
#         else:
#             return A.Compose([
#                 A.Resize(384, 384),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])

#     def load_image_and_label(self, image_path, label_path):
#         """Load image and parse YOLO format labels"""
#         # Read image in original size
#         image = cv2.imread(str(image_path))
#         if image is None:
#             raise ValueError(f"Could not read image: {image_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         original_h, original_w = image.shape[:2]
#         detections = []
        
#         # Read label file
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     values = line.strip().split()
#                     if len(values) >= 5:
#                         class_id = int(values[0])
#                         if class_id == 3:  # Fracture class
#                             # Parse YOLO format (normalized coordinates)
#                             x_center = float(values[1])
#                             y_center = float(values[2])
#                             box_width = float(values[3])
#                             box_height = float(values[4])
                            
#                             # Convert normalized to pixel coordinates
#                             x_center_px = x_center * original_w
#                             y_center_px = y_center * original_h
#                             box_width_px = box_width * original_w
#                             box_height_px = box_height * original_h
                            
#                             # Calculate corners
#                             x1 = int(x_center_px - box_width_px/2)
#                             y1 = int(y_center_px - box_height_px/2)
#                             x2 = int(x_center_px + box_width_px/2)
#                             y2 = int(y_center_px + box_height_px/2)
                            
#                             # Calculate area relative to original image size
#                             bbox_area = (box_width * box_height)  # Already normalized
                            
#                             # Use original confidence if available
#                             confidence = float(values[5]) if len(values) >= 6 else 1.0
                            
#                             detections.append({
#                                 'bbox': (x1, y1, x2, y2),
#                                 'confidence': confidence,
#                                 'area': bbox_area,
#                                 'box_width': box_width_px,
#                                 'box_height': box_height_px
#                             })
        
#         return image, detections

#     def crop_fracture_region(self, image, bbox, padding=20):
#         """Crop image to fracture region with padding"""
#         x1, y1, x2, y2 = bbox
#         height, width = image.shape[:2]
        
#         # Add padding while keeping within image bounds
#         x1_pad = max(0, x1 - padding)
#         y1_pad = max(0, y1 - padding)
#         x2_pad = min(width, x2 + padding)
#         y2_pad = min(height, y2 + padding)
        
#         # Ensure valid crop region
#         if x2_pad <= x1_pad or y2_pad <= y1_pad:
#             return image  # Return full image if crop would be invalid
            
#         return image[y1_pad:y2_pad, x1_pad:x2_pad]

#     def determine_severity(self, detection):
#         """
#         Determine fracture severity based on area and confidence
#         Returns severity level (0-4)
#         """
#         if not detection:
#             return 0
        
#         area = detection['area']  # Already normalized from YOLO format
#         confidence = detection['confidence']
#         box_width = detection['box_width']
#         box_height = detection['box_height']
        
#         # Area-based score (0-4)
#         area_score = 0
#         for threshold in self.severity_mapping['area_thresholds']:
#             if area > threshold:
#                 area_score += 1
        
#         # Confidence-based score (0-4)
#         conf_score = 0
#         for threshold in self.severity_mapping['confidence_thresholds']:
#             if confidence > threshold:
#                 conf_score += 1
        
#         # Combine scores with weights
#         final_score = min(4, int((0.7 * area_score + 0.3 * conf_score) + 0.5))
#         return final_score

#     def __getitem__(self, idx):
#         img_name = self.img_files[idx]
#         img_path = self.img_dir / img_name
#         label_path = self.label_dir / f"{img_name.rsplit('.', 1)[0]}.txt"
        
#         # Load image and parse labels
#         image, detections = self.load_image_and_label(img_path, label_path)
        
#         # Process detected fractures
#         if detections:
#             # Use the largest detected fracture region
#             max_detection = max(detections, key=lambda x: x['area'])
#             region = self.crop_fracture_region(image, max_detection['bbox'])
#             severity = self.determine_severity(max_detection)
#         else:
#             region = image
#             severity = 0
        
#         # Ensure region is grayscale and has channel dimension
#         if len(region.shape) == 2:
#             region = np.expand_dims(region, -1)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=region)
#             region = transformed['image']
        
#         return {
#             'image': region,
#             'severity': torch.tensor(severity, dtype=torch.long),
#             'image_path': str(img_path)
#         }

#     def __len__(self):
#         return len(self.img_files)

# class ModernFractureSeverityModel(nn.Module):
#     def __init__(self, pretrained=True):
#         super(ModernFractureSeverityModel, self).__init__()
        
#         # Use EfficientNetV2 as backbone
#         self.backbone = timm.create_model(
#             'tf_efficientnetv2_m',
#             pretrained=pretrained,
#             in_chans=1,  # Grayscale input
#             num_classes=0,
#             global_pool=''
#         )
        
#         feat_dim = self.backbone.num_features
        
#         # Advanced classifier head
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(feat_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 5)  # 5 severity classes
#         )
        
#         # Initialize weights
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, outputs, targets):
#             # Handle dictionary outputs from model
#             if isinstance(outputs, dict):
#                 pattern_outputs = outputs['pattern']
#                 joint_outputs = outputs['joint_involvement']
#                 displacement_outputs = outputs['displacement']
#             else:
#                 raise TypeError("Expected model outputs to be a dictionary")

#             # Extract targets
#             pattern_targets = targets['pattern']
#             joint_targets = targets['joint_involvement']
#             displacement_targets = targets['displacement']

#             # Calculate individual losses
#             pattern_loss = self.pattern_loss(pattern_outputs, pattern_targets)
#             joint_loss = self.joint_loss(joint_outputs, joint_targets)
#             displacement_loss = self.displacement_loss(displacement_outputs, displacement_targets)
            
#             # Calculate weighted sum
#             total_loss = (
#                 self.pattern_weight * pattern_loss +
#                 self.joint_weight * joint_loss +
#                 self.displacement_weight * displacement_loss
#             )
            
#             return total_loss, {
#                 'pattern_loss': pattern_loss.item(),
#                 'joint_loss': joint_loss.item(),
#                 'displacement_loss': displacement_loss.item()
#             }

# def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
#     model.train()
#     running_loss = 0.0
#     pattern_correct = 0
#     joint_correct = 0
#     displacement_correct = 0
#     total_samples = 0
    
#     all_pattern_preds = []
#     all_pattern_labels = []
    
#     pbar = tqdm(train_loader, desc='Training')
#     for batch in pbar:
#         images = batch['image'].to(device)
        
#         # For now, we'll generate random targets until we have proper labels
#         batch_size = images.size(0)
#         targets = {
#             'pattern': torch.randint(0, 3, (batch_size,)).to(device),  # 3 pattern classes
#             'joint_involvement': torch.randint(0, 2, (batch_size, 1)).float().to(device),
#             'displacement': torch.randint(0, 3, (batch_size,)).to(device)  # 3 displacement levels
#         }
        
#         optimizer.zero_grad()
        
#         # Regular forward pass without autocast
#         outputs = model(images)
#         loss, loss_dict = criterion(outputs, targets)
        
#         # Regular backward pass
#         loss.backward()
#         optimizer.step()
        
#         # Calculate accuracies
#         _, pattern_predicted = torch.max(outputs['pattern'].data, 1)
#         pattern_correct += (pattern_predicted == targets['pattern']).sum().item()
        
#         joint_predicted = (outputs['joint_involvement'] > 0.5).float()
#         joint_correct += (joint_predicted == targets['joint_involvement']).sum().item()
        
#         _, displacement_predicted = torch.max(outputs['displacement'].data, 1)
#         displacement_correct += (displacement_predicted == targets['displacement']).sum().item()
        
#         total_samples += targets['pattern'].size(0)
        
#         # Store predictions for pattern classification report
#         all_pattern_preds.extend(pattern_predicted.cpu().numpy())
#         all_pattern_labels.extend(targets['pattern'].cpu().numpy())
        
#         running_loss += loss.item()
        
#         # Update progress bar
#         pbar.set_postfix({
#             'loss': loss.item(),
#             'pattern_acc': 100 * pattern_correct / total_samples,
#             'joint_acc': 100 * joint_correct / total_samples,
#             'disp_acc': 100 * displacement_correct / total_samples
#         })
    
#     return (running_loss / len(train_loader), 
#             all_pattern_preds, 
#             all_pattern_labels,
#             {
#                 'pattern_acc': 100 * pattern_correct / total_samples,
#                 'joint_acc': 100 * joint_correct / total_samples,
#                 'displacement_acc': 100 * displacement_correct / total_samples
#             })

# def validate(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0.0
#     pattern_correct = 0
#     joint_correct = 0
#     displacement_correct = 0
#     total_samples = 0
    
#     all_pattern_preds = []
#     all_pattern_labels = []
    
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc='Validation'):
#             images = batch['image'].to(device)
            
#             # For now, using random targets
#             batch_size = images.size(0)
#             targets = {
#                 'pattern': torch.randint(0, 3, (batch_size,)).to(device),
#                 'joint_involvement': torch.randint(0, 2, (batch_size, 1)).float().to(device),
#                 'displacement': torch.randint(0, 3, (batch_size,)).to(device)
#             }
            
#             outputs = model(images)
#             loss, _ = criterion(outputs, targets)
            
#             val_loss += loss.item()
            
#             # Calculate accuracies
#             _, pattern_predicted = torch.max(outputs['pattern'].data, 1)
#             pattern_correct += (pattern_predicted == targets['pattern']).sum().item()
            
#             joint_predicted = (outputs['joint_involvement'] > 0.5).float()
#             joint_correct += (joint_predicted == targets['joint_involvement']).sum().item()
            
#             _, displacement_predicted = torch.max(outputs['displacement'].data, 1)
#             displacement_correct += (displacement_predicted == targets['displacement']).sum().item()
            
#             total_samples += targets['pattern'].size(0)
            
#             all_pattern_preds.extend(pattern_predicted.cpu().numpy())
#             all_pattern_labels.extend(targets['pattern'].cpu().numpy())
    
#     return (val_loss / len(val_loader), 
#             all_pattern_preds, 
#             all_pattern_labels,
#             {
#                 'pattern_acc': 100 * pattern_correct / total_samples,
#                 'joint_acc': 100 * joint_correct / total_samples,
#                 'displacement_acc': 100 * displacement_correct / total_samples
#             })

# def main():
#     # Configuration
# # Configuration
#     config = {
#         'train_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\train',
#         'train_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\train',
#         'val_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\valid',
#         'val_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\valid',
#         'batch_size': 16,  # Reduced batch size
#         'num_workers': 4,  # Reduced workers
#         'learning_rate': 1e-3,
#         'weight_decay': 1e-4,
#         'epochs': 15,
#         'save_dir': './models',
#         'image_size': 384
#     }
    
#     # Setup
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Create save directory
#     os.makedirs(config['save_dir'], exist_ok=True)
    
#     # Initialize datasets
#     print("\nInitializing training dataset...")
#     train_dataset = FracturePatternDataset(
#         config['train_img_dir'],
#         config['train_label_dir'],
#         mode='train',
#         image_size=config['image_size']
#     )
    
#     print("\nInitializing validation dataset...")
#     val_dataset = FracturePatternDataset(
#         config['val_img_dir'],
#         config['val_label_dir'],
#         mode='val',
#         image_size=config['image_size']
#     )
    
#     # Create data loaders
# # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         persistent_workers=False  # Disabled persistent workers
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         persistent_workers=False  # Disabled persistent workers
#     )
    
#     # Initialize model and training components
# # Initialize model and training components
#     model = FracturePatternModel(pretrained=True).to(device)
    
#     # Disable gradient checkpointing as it's causing issues
#     if hasattr(model.backbone, 'set_grad_checkpointing'):
#         model.backbone.set_grad_checkpointing(False)
        
#     criterion = FractureLoss(
#         pattern_weight=1.0,
#         joint_weight=0.5,
#         displacement_weight=0.5
#     ).to(device)
    
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config['learning_rate'],
#         weight_decay=config['weight_decay']
#     )
    
#     scaler = torch.amp.GradScaler()
    
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=config['learning_rate'],
#         epochs=config['epochs'],
#         steps_per_epoch=len(train_loader),
#         pct_start=0.3,
#         anneal_strategy='cos'
#     )
    
#     # Training loop
#     best_val_loss = float('inf')
#     pattern_names = ["Simple", "Wedge", "Comminuted"]
    
#     print("\nStarting training...")
#     for epoch in range(config['epochs']):
#         print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
#         # Train and validate
#         train_loss, train_preds, train_labels, train_metrics = train_one_epoch(
#             model, train_loader, criterion, optimizer, scaler, device
#         )
        
#         val_loss, val_preds, val_labels, val_metrics = validate(
#             model, val_loader, criterion, device
#         )
        
#         # Print metrics
#         print("\nTraining Metrics:")
#         print(classification_report(
#             train_labels, 
#             train_preds,
#             labels=range(len(pattern_names)),
#             target_names=pattern_names
#         ))
#         print(f"Joint Accuracy: {train_metrics['joint_acc']:.2f}%")
#         print(f"Displacement Accuracy: {train_metrics['displacement_acc']:.2f}%")
        
#         print("\nValidation Metrics:")
#         print(classification_report(
#             val_labels, 
#             val_preds,
#             labels=range(len(pattern_names)),
#             target_names=pattern_names
#         ))
#         print(f"Joint Accuracy: {val_metrics['joint_acc']:.2f}%")
#         print(f"Displacement Accuracy: {val_metrics['displacement_acc']:.2f}%")
        
#         # Update learning rate
#         scheduler.step()
        
#         # Save checkpoint if validation loss improves
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             checkpoint_path = os.path.join(
#                 config['save_dir'],
#                 f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
#             )
            
#             print(f"\nSaving best model to {checkpoint_path}")
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'config': config,
#                 'scaler_state_dict': scaler.state_dict(),
#             }, checkpoint_path)
        
#         print(f'\nEpoch Summary:')
#         print(f'Training Loss: {train_loss:.4f}')
#         print(f'Validation Loss: {val_loss:.4f}')
#         print('-' * 50)

# if __name__ == "__main__":
#     main()

#train_fracture_severity.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Update this import
from torch.cuda.amp import GradScaler
import torch.amp  # Add this import
import timm
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report
from fracture_model import FracturePatternModel, FractureLoss
from dataset_handler import FracturePatternDataset


class FractureSeverityDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        self.transform = transform if transform else self.get_default_transforms(mode)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Create a mapping of fracture characteristics to severity levels
        self.severity_mapping = {
            'area_thresholds': [0.02, 0.05, 0.1, 0.15],  # relative to image size
            'confidence_thresholds': [0.3, 0.5, 0.7, 0.85]
        }
    
    def get_default_transforms(self, mode):
        if mode == 'train':
            return A.Compose([
                A.Resize(384, 384),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
                    A.MultiplicativeNoise(p=0.8),
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.4),
                    A.GridDistortion(p=0.4),
                    A.ElasticTransform(p=0.4),
                ], p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])

    def load_image_and_label(self, image_path, label_path):
        """Load image and parse YOLO format labels"""
        # Read image in original size
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        original_h, original_w = image.shape[:2]
        detections = []
        
        # Read label file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        if class_id == 3:  # Fracture class
                            # Parse YOLO format (normalized coordinates)
                            x_center = float(values[1])
                            y_center = float(values[2])
                            box_width = float(values[3])
                            box_height = float(values[4])
                            
                            # Convert normalized to pixel coordinates
                            x_center_px = x_center * original_w
                            y_center_px = y_center * original_h
                            box_width_px = box_width * original_w
                            box_height_px = box_height * original_h
                            
                            # Calculate corners
                            x1 = int(x_center_px - box_width_px/2)
                            y1 = int(y_center_px - box_height_px/2)
                            x2 = int(x_center_px + box_width_px/2)
                            y2 = int(y_center_px + box_height_px/2)
                            
                            # Calculate area relative to original image size
                            bbox_area = (box_width * box_height)  # Already normalized
                            
                            # Use original confidence if available
                            confidence = float(values[5]) if len(values) >= 6 else 1.0
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'area': bbox_area,
                                'box_width': box_width_px,
                                'box_height': box_height_px
                            })
        
        return image, detections

    def crop_fracture_region(self, image, bbox, padding=20):
        """Crop image to fracture region with padding"""
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Add padding while keeping within image bounds
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(width, x2 + padding)
        y2_pad = min(height, y2 + padding)
        
        # Ensure valid crop region
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return image  # Return full image if crop would be invalid
            
        return image[y1_pad:y2_pad, x1_pad:x2_pad]

    def determine_severity(self, detection):
        """
        Determine fracture severity based on area and confidence
        Returns severity level (0-4)
        """
        if not detection:
            return 0
        
        area = detection['area']  # Already normalized from YOLO format
        confidence = detection['confidence']
        box_width = detection['box_width']
        box_height = detection['box_height']
        
        # Area-based score (0-4)
        area_score = 0
        for threshold in self.severity_mapping['area_thresholds']:
            if area > threshold:
                area_score += 1
        
        # Confidence-based score (0-4)
        conf_score = 0
        for threshold in self.severity_mapping['confidence_thresholds']:
            if confidence > threshold:
                conf_score += 1
        
        # Combine scores with weights
        final_score = min(4, int((0.7 * area_score + 0.3 * conf_score) + 0.5))
        return final_score

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{img_name.rsplit('.', 1)[0]}.txt"
        
        # Load image and parse labels
        image, detections = self.load_image_and_label(img_path, label_path)
        
        # Process detected fractures
        if detections:
            # Use the largest detected fracture region
            max_detection = max(detections, key=lambda x: x['area'])
            region = self.crop_fracture_region(image, max_detection['bbox'])
            severity = self.determine_severity(max_detection)
        else:
            region = image
            severity = 0
        
        # Ensure region is grayscale and has channel dimension
        if len(region.shape) == 2:
            region = np.expand_dims(region, -1)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=region)
            region = transformed['image']
        
        return {
            'image': region,
            'severity': torch.tensor(severity, dtype=torch.long),
            'image_path': str(img_path)
        }

    def __len__(self):
        return len(self.img_files)

class ModernFractureSeverityModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ModernFractureSeverityModel, self).__init__()
        
        # Use EfficientNetV2 as backbone
        self.backbone = timm.create_model(
            'tf_efficientnetv2_m',
            pretrained=pretrained,
            in_chans=1,  # Grayscale input
            num_classes=0,
            global_pool=''
        )
        
        feat_dim = self.backbone.num_features
        
        # Advanced classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5)  # 5 severity classes
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, outputs, targets):
            # Handle dictionary outputs from model
            if isinstance(outputs, dict):
                pattern_outputs = outputs['pattern']
                joint_outputs = outputs['joint_involvement']
                displacement_outputs = outputs['displacement']
            else:
                raise TypeError("Expected model outputs to be a dictionary")

            # Extract targets
            pattern_targets = targets['pattern']
            joint_targets = targets['joint_involvement']
            displacement_targets = targets['displacement']

            # Calculate individual losses
            pattern_loss = self.pattern_loss(pattern_outputs, pattern_targets)
            joint_loss = self.joint_loss(joint_outputs, joint_targets)
            displacement_loss = self.displacement_loss(displacement_outputs, displacement_targets)
            
            # Calculate weighted sum
            total_loss = (
                self.pattern_weight * pattern_loss +
                self.joint_weight * joint_loss +
                self.displacement_weight * displacement_loss
            )
            
            return total_loss, {
                'pattern_loss': pattern_loss.item(),
                'joint_loss': joint_loss.item(),
                'displacement_loss': displacement_loss.item()
            }

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    pattern_correct = 0
    joint_correct = 0
    total_samples = 0
    
    all_pattern_preds = []
    all_pattern_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'pattern': batch['pattern'].to(device),
            'joint_involvement': batch['joint_involvement'].to(device)
        }
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracies
        _, pattern_predicted = torch.max(outputs['pattern'].data, 1)
        pattern_correct += (pattern_predicted == targets['pattern']).sum().item()
        
        joint_predicted = (outputs['joint_involvement'] > 0.5).float()
        joint_correct += (joint_predicted == targets['joint_involvement']).sum().item()
        
        total_samples += targets['pattern'].size(0)
        
        # Store predictions for pattern classification report
        all_pattern_preds.extend(pattern_predicted.cpu().numpy())
        all_pattern_labels.extend(targets['pattern'].cpu().numpy())
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'pattern_acc': 100 * pattern_correct / total_samples,
            'joint_acc': 100 * joint_correct / total_samples
        })
    
    return (running_loss / len(train_loader), 
            all_pattern_preds, 
            all_pattern_labels,
            {
                'pattern_acc': 100 * pattern_correct / total_samples,
                'joint_acc': 100 * joint_correct / total_samples
            })

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    pattern_correct = 0
    joint_correct = 0
    total_samples = 0
    
    all_pattern_preds = []
    all_pattern_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            targets = {
                'pattern': batch['pattern'].to(device),
                'joint_involvement': batch['joint_involvement'].to(device)
            }
            
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
            
            val_loss += loss.item()
            
            # Calculate accuracies
            _, pattern_predicted = torch.max(outputs['pattern'].data, 1)
            pattern_correct += (pattern_predicted == targets['pattern']).sum().item()
            
            joint_predicted = (outputs['joint_involvement'] > 0.5).float()
            joint_correct += (joint_predicted == targets['joint_involvement']).sum().item()
            
            total_samples += targets['pattern'].size(0)
            
            all_pattern_preds.extend(pattern_predicted.cpu().numpy())
            all_pattern_labels.extend(targets['pattern'].cpu().numpy())
    
    return (val_loss / len(val_loader), 
            all_pattern_preds, 
            all_pattern_labels,
            {
                'pattern_acc': 100 * pattern_correct / total_samples,
                'joint_acc': 100 * joint_correct / total_samples
            })

def main():
    # Configuration
    config = {
        'train_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\train',
        'train_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\train',
        'val_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\valid',
        'val_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\valid',
        'batch_size': 16,
        'num_workers': 4,
        'learning_rate': 1e-4,  # Reduced learning rate
        'weight_decay': 1e-4,
        'epochs': 30,  # Increased epochs
        'save_dir': './models',
        'image_size': 384
    }
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize datasets
    print("\nInitializing training dataset...")
    train_dataset = FracturePatternDataset(
        config['train_img_dir'],
        config['train_label_dir'],
        mode='train',
        image_size=config['image_size']
    )
    
    print("\nInitializing validation dataset...")
    val_dataset = FracturePatternDataset(
        config['val_img_dir'],
        config['val_label_dir'],
        mode='val',
        image_size=config['image_size']
    )
    
    # Calculate class weights based on dataset distribution
    total_samples = len(train_dataset)
    class_counts = torch.zeros(3)  # [Simple, Wedge, Comminuted]
    for data in train_dataset:
        class_counts[data['pattern']] += 1
    
    # Inversely proportional to class frequency
    class_weights = (total_samples / (class_counts + 1e-6))
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = class_weights.to(device)
    
    print("\nClass distribution:")
    pattern_names = ["Simple", "Wedge", "Comminuted"]
    for i, name in enumerate(pattern_names):
        print(f"{name}: {class_counts[i]} samples, weight: {class_weights[i]:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    
    # Initialize model and training components
    model = FracturePatternModel(pretrained=True).to(device)
    
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(False)
    
    criterion = FractureLoss(
        pattern_weight=1.0,
        joint_weight=0.5,
        class_weights=class_weights
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scaler = torch.amp.GradScaler()
    
    # Use ReduceLROnPlateau instead of OneCycleLR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_f1 = 0.0
    pattern_names = ["Simple", "Wedge", "Comminuted"]
    
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train and validate
        train_loss, train_preds, train_labels, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        val_loss, val_preds, val_labels, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print("\nTraining Metrics:")
        train_report = classification_report(
            train_labels, 
            train_preds,
            labels=range(len(pattern_names)),
            target_names=pattern_names,
            zero_division=0
        )
        print(train_report)
        print(f"Joint Accuracy: {train_metrics['joint_acc']:.2f}%")
        
        print("\nValidation Metrics:")
        val_report = classification_report(
            val_labels, 
            val_preds,
            labels=range(len(pattern_names)),
            target_names=pattern_names,
            zero_division=0
        )
        print(val_report)
        print(f"Joint Accuracy: {val_metrics['joint_acc']:.2f}%")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Calculate validation F1 score for model saving criterion
        val_f1 = float(val_report.split('\n')[-2].split()[-2])  # Get weighted avg F1
        
        # Save checkpoint if validation metrics improve
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(
                config['save_dir'],
                f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            )
            
            print(f"\nSaving best model to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_f1': float(train_report.split('\n')[-2].split()[-2]),
                'val_f1': val_f1,
                'config': config,
                'scaler_state_dict': scaler.state_dict(),
                'class_weights': class_weights,
            }, checkpoint_path)
        
        print(f'\nEpoch Summary:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Best F1 Score: {best_f1:.4f}')
        print('-' * 50)

if __name__ == "__main__":
    main()