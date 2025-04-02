#train_densenet_cv.py

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from densenet_inference import DenseNetInference

def evaluate_densenet_on_fold(model_path, fold_dir, output_dir):
    """Evaluate DenseNet on a specific fold"""
    # Initialize DenseNet
    detector = DenseNetInference(model_path, "cpu")
    
    # Get validation data for this fold
    val_df = pd.read_csv(fold_dir / "val_data.csv")
    
    # Create output directory
    fold_output = output_dir / f"fold_{fold_dir.name.split('_')[1]}"
    os.makedirs(fold_output, exist_ok=True)
    
    # Classes
    detection_classes = ["boneanomaly", "bonelesion", "foreignbody", 
                       "fracture", "metal", "periostealreaction", 
                       "pronatorsign", "softtissue", "text"]
    
    # Collection for results
    all_true = []
    all_pred = []
    all_confidences = []
    file_paths = []
    
    # Process each image
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        try:
            img_path = fold_dir / "yolov5" / "images" / "valid" / f"{row['filestem']}.png"
            label_path = fold_dir / "yolov5" / "labels" / "valid" / f"{row['filestem']}.txt"
            
            # Skip if files don't exist
            if not img_path.exists() or not label_path.exists():
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Get predictions
            results, _ = detector(img)
            
            # Process predictions
            predictions = np.zeros(len(detection_classes))
            confidences = np.zeros(len(detection_classes))
            
            for i, cls in enumerate(detection_classes):
                predictions[i] = 1 if results[cls]['detected'] else 0
                confidences[i] = results[cls]['confidence']
            
            # Load ground truth
            true_labels = np.zeros(len(detection_classes))
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    true_labels[class_id] = 1
            
            all_true.append(true_labels)
            all_pred.append(predictions)
            all_confidences.append(confidences)
            file_paths.append(str(img_path))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Skip calculation if no samples were processed
    if not all_true:
        print(f"No valid samples processed in this fold")
        return None
    
    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_confidences = np.array(all_confidences)
    
    # Calculate overall metrics
    metrics = {
        'accuracy': accuracy_score(all_true.flatten(), all_pred.flatten()),
        'precision': precision_score(all_true, all_pred, average='weighted', zero_division=0),
        'recall': recall_score(all_true, all_pred, average='weighted', zero_division=0),
        'f1': f1_score(all_true, all_pred, average='weighted', zero_division=0)
    }
    
    # Calculate metrics for fracture class specifically
    fracture_idx = detection_classes.index("fracture")
    metrics['fracture_precision'] = precision_score(all_true[:, fracture_idx], all_pred[:, fracture_idx], zero_division=0)
    metrics['fracture_recall'] = recall_score(all_true[:, fracture_idx], all_pred[:, fracture_idx], zero_division=0)
    metrics['fracture_f1'] = f1_score(all_true[:, fracture_idx], all_pred[:, fracture_idx], zero_division=0)
    
    # Generate confusion matrix for fracture class
    cm = confusion_matrix(all_true[:, fracture_idx], all_pred[:, fracture_idx])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'DenseNet Fracture Detection - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(fold_output / 'densenet_fracture_confusion_matrix.png')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_fracture': all_true[:, fracture_idx],
        'pred_fracture': all_pred[:, fracture_idx],
        'confidence': all_confidences[:, fracture_idx]
    })
    results_df.to_csv(fold_output / 'densenet_results.csv', index=False)
    
    return metrics

def evaluate_densenet_cv():
    """Run cross-validation for DenseNet"""
    # Base paths
    base_path = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset")
    model_path = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\best_densenet_detector.pth")
    cv_path = base_path / "cv_folds"
    results_path = base_path / "cv_results"
    os.makedirs(results_path, exist_ok=True)
    
    # Results collection
    fold_results = []
    
    # Process each fold
    for fold in range(5):
        print(f"\n===== Evaluating DenseNet on Fold {fold+1}/5 =====")
        fold_dir = cv_path / f"fold_{fold}"
        
        # Check if fold data exists
        if not (fold_dir / "val_data.csv").exists():
            print(f"Fold data not found for fold {fold+1}. Skipping...")
            continue
        
        # Evaluate model on this fold
        metrics = evaluate_densenet_on_fold(model_path, fold_dir, results_path)
        
        if metrics:
            metrics['fold'] = fold
            fold_results.append(metrics)
            print(f"Fold {fold+1} results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Fracture F1: {metrics['fracture_f1']:.4f}")
    
    # Compile and save overall results
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(results_path / "densenet_cv_results.csv", index=False)
        
        # Calculate and display average metrics
        print("\n===== DenseNet Cross-Validation Results =====")
        for metric in ['precision', 'recall', 'f1', 'fracture_precision', 'fracture_recall', 'fracture_f1']:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(x='fold', y='precision', data=results_df)
        plt.title('Precision by Fold')
        plt.axhline(y=results_df['precision'].mean(), color='r', linestyle='--')
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='fold', y='recall', data=results_df)
        plt.title('Recall by Fold')
        plt.axhline(y=results_df['recall'].mean(), color='r', linestyle='--')
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='fold', y='f1', data=results_df)
        plt.title('F1 Score by Fold')
        plt.axhline(y=results_df['f1'].mean(), color='r', linestyle='--')
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='fold', y='fracture_f1', data=results_df)
        plt.title('Fracture F1 Score by Fold')
        plt.axhline(y=results_df['fracture_f1'].mean(), color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(results_path / "densenet_cv_summary.png")
        
        return results_df
    else:
        print("No results were collected.")
        return None

if __name__ == "__main__":
    evaluate_densenet_cv()