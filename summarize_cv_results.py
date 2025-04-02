import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_cv_results():
    base_path = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset")
    
    # Load results
    yolo_results = pd.read_csv(base_path / "cv_results" / "yolo_cv_results.csv")
    densenet_results = pd.read_csv(base_path / "cv_results" /"densenet_cv_results.csv")
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1/mAP@0.5'],
        'YOLOv7 (Mean ± Std)': [
            f"{yolo_results['precision'].mean():.4f} ± {yolo_results['precision'].std():.4f}",
            f"{yolo_results['recall'].mean():.4f} ± {yolo_results['recall'].std():.4f}",
            f"{yolo_results['mAP_50'].mean():.4f} ± {yolo_results['mAP_50'].std():.4f}"
        ],
        'DenseNet (Mean ± Std)': [
            f"{densenet_results['precision'].mean():.4f} ± {densenet_results['precision'].std():.4f}",
            f"{densenet_results['recall'].mean():.4f} ± {densenet_results['recall'].std():.4f}",
            f"{densenet_results['f1'].mean():.4f} ± {densenet_results['f1'].std():.4f}"
        ]
    })
    
    # Save comparison table
    comparison.to_csv(base_path / "cv_comparison.csv", index=False)
    print(comparison)
    
    # Create comparative visualization
    plt.figure(figsize=(15, 12))
    
    # Prepare data for side-by-side comparison
    metrics = ['precision', 'recall']
    yolo_data = []
    densenet_data = []
    
    for metric in metrics:
        for fold, value in enumerate(yolo_results[metric]):
            yolo_data.append({'Fold': fold+1, 'Metric': metric.capitalize(), 'Value': value, 'Model': 'YOLOv7'})
        
        for fold, value in enumerate(densenet_results[metric]):
            densenet_data.append({'Fold': fold+1, 'Metric': metric.capitalize(), 'Value': value, 'Model': 'DenseNet'})
    
    # Add mAP/F1 comparison
    for fold, value in enumerate(yolo_results['mAP_50']):
        yolo_data.append({'Fold': fold+1, 'Metric': 'F1/mAP@0.5', 'Value': value, 'Model': 'YOLOv7'})
    
    for fold, value in enumerate(densenet_results['f1']):
        densenet_data.append({'Fold': fold+1, 'Metric': 'F1/mAP@0.5', 'Value': value, 'Model': 'DenseNet'})
    
    # Combine data
    plot_data = pd.DataFrame(yolo_data + densenet_data)
    
    # Create plots
    for i, metric in enumerate(['Precision', 'Recall', 'F1/mAP@0.5']):
        plt.subplot(3, 1, i+1)
        sns.barplot(x='Fold', y='Value', hue='Model', data=plot_data[plot_data['Metric'] == metric])
        plt.title(f'Comparison of {metric} Across Folds')
        plt.ylabel(metric)
        plt.xlabel('Fold')
        plt.ylim(0, 1)
        
        # Add mean lines
        if metric == 'F1/mAP@0.5':
            plt.axhline(y=yolo_results['mAP_50'].mean(), color='blue', linestyle='--', 
                        label=f'YOLOv7 Mean: {yolo_results["mAP_50"].mean():.4f}')
            plt.axhline(y=densenet_results['f1'].mean(), color='orange', linestyle='--', 
                        label=f'DenseNet Mean: {densenet_results["f1"].mean():.4f}')
        else:
            plt.axhline(y=yolo_results[metric.lower()].mean(), color='blue', linestyle='--', 
                        label=f'YOLOv7 Mean: {yolo_results[metric.lower()].mean():.4f}')
            plt.axhline(y=densenet_results[metric.lower()].mean(), color='orange', linestyle='--', 
                        label=f'DenseNet Mean: {densenet_results[metric.lower()].mean():.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(base_path / "cv_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    analyze_cv_results()