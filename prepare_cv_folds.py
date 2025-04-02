#prepare_cv_folds.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_cv_folds():
    # Base paths
    base_path = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset")
    
    # Load the complete dataset
    df = pd.read_csv(str(base_path / "dataset.csv"))
    
    # Initialize 5-fold CV splitter
    group_kfold = GroupKFold(n_splits=5)
    
    # Create directory for CV results
    cv_dir = base_path / "cv_folds"
    os.makedirs(cv_dir, exist_ok=True)
    
    # Get all splits from the group k-fold
    folds = list(group_kfold.split(df, groups=df['patient_id']))
    
    # Process each fold
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n===== Preparing Fold {fold+1}/5 =====")
        
        # Split data for this fold
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Create fold directory structure
        fold_dir = cv_dir / f"fold_{fold}"
        os.makedirs(fold_dir / "yolov5" / "images" / "train", exist_ok=True)
        os.makedirs(fold_dir / "yolov5" / "images" / "valid", exist_ok=True)  
        os.makedirs(fold_dir / "yolov5" / "labels" / "train", exist_ok=True)
        os.makedirs(fold_dir / "yolov5" / "labels" / "valid", exist_ok=True)
        
        # Save split data
        train_df.to_csv(fold_dir / "train_data.csv", index=False)
        val_df.to_csv(fold_dir / "val_data.csv", index=False)
        
        # Source directories
        # Check both train and test folders for images and labels
        source_dirs = {
            'images': {
                'train': base_path / "yolov5" / "images" / "train",
                'valid': base_path / "yolov5" / "images" / "valid",
                'test': base_path / "yolov5" / "images" / "test"
            },
            'labels': {
                'train': base_path / "yolov5" / "labels" / "train",
                'valid': base_path / "yolov5" / "labels" / "valid", 
                'test': base_path / "yolov5" / "labels" / "test"
            }
        }
        
        # Copy train files
        print(f"Copying training files for fold {fold+1}...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            filestem = row['filestem']
            
            # Try to find the image in any of the source directories
            found = False
            for split in ['train', 'valid', 'test']:
                img_path = source_dirs['images'][split] / f"{filestem}.png"
                label_path = source_dirs['labels'][split] / f"{filestem}.txt"
                
                if img_path.exists() and label_path.exists():
                    # Copy to fold train directory
                    shutil.copy2(img_path, fold_dir / "yolov5" / "images" / "train" / f"{filestem}.png")
                    shutil.copy2(label_path, fold_dir / "yolov5" / "labels" / "train" / f"{filestem}.txt")
                    found = True
                    break
            
            if not found:
                print(f"Warning: Files for {filestem} not found in any source directory")
        
        # Copy validation files
        print(f"Copying validation files for fold {fold+1}...")
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
            filestem = row['filestem']
            
            # Try to find the image in any of the source directories
            found = False
            for split in ['train', 'valid', 'test']:
                img_path = source_dirs['images'][split] / f"{filestem}.png"
                label_path = source_dirs['labels'][split] / f"{filestem}.txt"
                
                if img_path.exists() and label_path.exists():
                    # Copy to fold validation directory
                    shutil.copy2(img_path, fold_dir / "yolov5" / "images" / "valid" / f"{filestem}.png")
                    shutil.copy2(label_path, fold_dir / "yolov5" / "labels" / "valid" / f"{filestem}.txt")
                    found = True
                    break
            
            if not found:
                print(f"Warning: Files for {filestem} not found in any source directory")
    
    # Calculate fold statistics
    fold_stats = []
    for fold in range(5):
        train_count = len(list((cv_dir / f"fold_{fold}" / "yolov5" / "images" / "train").glob("*.png")))
        val_count = len(list((cv_dir / f"fold_{fold}" / "yolov5" / "images" / "valid").glob("*.png")))
        
        fold_stats.append({
            'Fold': fold + 1,
            'Training Samples': train_count,
            'Validation Samples': val_count,
            'Total': train_count + val_count
        })
    
    stats_df = pd.DataFrame(fold_stats)
    stats_df.to_csv(cv_dir / "fold_statistics.csv", index=False)
    print("\nFold statistics:")
    print(stats_df)
    
    return cv_dir

if __name__ == "__main__":
    prepare_cv_folds()