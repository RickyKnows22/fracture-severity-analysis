# import os
# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
# import matplotlib.pyplot as plt
# import seaborn as sns
# from train_fracture_severity import FractureSeverityDataset, ModernFractureSeverityModel
# from torch.utils.data import DataLoader
# import pandas as pd
# from tqdm import tqdm

# class PerformanceAnalyzer:
#     def __init__(self, model_path):
#         print("Initializing Performance Analyzer...")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
        
#         # Load model
#         print("Loading model...")
#         self.model = ModernFractureSeverityModel(pretrained=False)
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model = self.model.to(self.device)
#         self.model.eval()
        
#         self.severity_grades = ["Normal/Healed", "Minimal", "Moderate", "Severe", "Critical"]
#         print("Initialization complete.")

#     def analyze_dataset_distribution(self, dataset_loader, dataset_name):
#         """Analyze the distribution of severity grades in a dataset"""
#         severity_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
#         print(f"\nAnalyzing {dataset_name} dataset distribution...")
        
#         for batch in tqdm(dataset_loader, desc=f"Analyzing {dataset_name}"):
#             labels = batch['severity'].cpu().numpy()
#             for label in labels:
#                 severity_counts[label] = severity_counts.get(label, 0) + 1
        
#         total = sum(severity_counts.values())
#         print(f"\n{dataset_name} Dataset Distribution:")
#         for grade, count in severity_counts.items():
#             percentage = (count / total) * 100
#             print(f"Grade {self.severity_grades[grade]}: {count} images ({percentage:.2f}%)")
        
#         return severity_counts

#     def evaluate_dataset(self, dataset_loader, dataset_name):
#         true_labels = []
#         predicted_labels = []
#         confidences = []
        
#         print(f"\nEvaluating {dataset_name} dataset...")
#         with torch.no_grad():
#             for batch in tqdm(dataset_loader, desc=f"Processing {dataset_name}", unit="batch"):
#                 images = batch['image'].to(self.device)
#                 labels = batch['severity'].to(self.device)
                
#                 outputs = self.model(images)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 predictions = torch.argmax(outputs, dim=1)
                
#                 true_labels.extend(labels.cpu().numpy())
#                 predicted_labels.extend(predictions.cpu().numpy())
#                 confidences.extend(torch.max(probabilities, dim=1)[0].cpu().numpy())
        
#         return true_labels, predicted_labels, confidences

#     def calculate_metrics(self, true_labels, predicted_labels, confidences):
#         print("Calculating metrics...")
        
#         # Get unique classes present in the data
#         unique_classes = sorted(np.unique(np.concatenate([true_labels, predicted_labels])))
#         present_grades = [self.severity_grades[i] for i in unique_classes]
        
#         print(f"Classes found in predictions: {unique_classes}")
#         print(f"Corresponding grades: {present_grades}")
        
#         accuracy = accuracy_score(true_labels, predicted_labels)
#         precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
#         conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
#         class_report = classification_report(
#             true_labels, 
#             predicted_labels,
#             labels=unique_classes,  # Use only the classes that are present
#             target_names=present_grades,
#             output_dict=True
#         )
        
#         avg_confidence = np.mean(confidences)
        
#         # Per-class metrics
#         per_class_metrics = {}
#         for i, grade in enumerate(self.severity_grades):
#             if i in unique_classes:
#                 class_mask = np.array(true_labels) == i
#                 if np.any(class_mask):
#                     class_conf = np.array(confidences)[np.array(predicted_labels) == i]
#                     per_class_metrics[grade] = {
#                         'precision': class_report[grade]['precision'],
#                         'recall': class_report[grade]['recall'],
#                         'f1-score': class_report[grade]['f1-score'],
#                         'support': class_report[grade]['support'],
#                         'avg_confidence': np.mean(class_conf) if len(class_conf) > 0 else 0
#                     }
#             else:
#                 # Add empty metrics for missing classes
#                 per_class_metrics[grade] = {
#                     'precision': 0.0,
#                     'recall': 0.0,
#                     'f1-score': 0.0,
#                     'support': 0,
#                     'avg_confidence': 0.0
#                 }
        
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'confusion_matrix': conf_matrix,
#             'per_class_metrics': per_class_metrics,
#             'avg_confidence': avg_confidence,
#             'classes_found': unique_classes,
#             'grades_found': present_grades
#         }
    
#     def plot_confusion_matrix(self, conf_matrix, dataset_name, save_dir):
#         print(f"Plotting confusion matrix for {dataset_name}...")
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=self.severity_grades,
#                 yticklabels=self.severity_grades)
#         plt.title(f'Confusion Matrix - {dataset_name} Dataset')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dataset_name}.png'))
#         plt.close()

#     def plot_class_metrics(self, metrics, dataset_name, save_dir):
#         print(f"Plotting class metrics for {dataset_name}...")
#         metrics_df = pd.DataFrame(metrics['per_class_metrics']).T
        
#         # Plot metrics
#         plt.figure(figsize=(12, 6))
#         metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
#         plt.title(f'Performance Metrics by Class - {dataset_name} Dataset')
#         plt.xlabel('Severity Grade')
#         plt.ylabel('Score')
#         plt.legend(loc='best')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'class_metrics_{dataset_name}.png'))
#         plt.close()
        
#         # Plot confidence
#         plt.figure(figsize=(8, 6))
#         metrics_df['avg_confidence'].plot(kind='bar')
#         plt.title(f'Average Confidence by Class - {dataset_name} Dataset')
#         plt.xlabel('Severity Grade')
#         plt.ylabel('Confidence')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'confidence_{dataset_name}.png'))
#         plt.close()

#     def save_metrics_to_excel(self, all_metrics, save_path):
#         print("\nSaving metrics to Excel...")
#         with pd.ExcelWriter(save_path) as writer:
#             # Overall metrics
#             overall_df = pd.DataFrame({
#                 'Dataset': list(all_metrics.keys()),
#                 'Accuracy': [m['accuracy'] for m in all_metrics.values()],
#                 'Precision': [m['precision'] for m in all_metrics.values()],
#                 'Recall': [m['recall'] for m in all_metrics.values()],
#                 'F1 Score': [m['f1_score'] for m in all_metrics.values()],
#                 'Average Confidence': [m['avg_confidence'] for m in all_metrics.values()]
#             })
#             overall_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
            
#             # Per-class metrics for each dataset
#             for dataset_name, metrics in all_metrics.items():
#                 df = pd.DataFrame(metrics['per_class_metrics']).T
#                 df.to_excel(writer, sheet_name=f'{dataset_name}_Class_Metrics')
#         print(f"Metrics saved to {save_path}")

# def main():

#     # Analyze distribution


#     print("Starting performance analysis...")
    
#     # Configuration
#     config = {
#         'train_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\train',
#         'train_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\train',
#         'val_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\valid',
#         'val_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\valid',
#         'test_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\test',
#         'test_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\test',
#         'model_path': './models/best_model_20250216_133526.pth',
#         'results_dir': './performance_results',
#         'batch_size': 32
#     }
    
#     # Create results directory
#     os.makedirs(config['results_dir'], exist_ok=True)
#     print(f"Results will be saved to: {config['results_dir']}")
    
#     # Initialize analyzer
#     analyzer = PerformanceAnalyzer(config['model_path'])
    
#     # Initialize datasets
#     print("\nLoading datasets...")
#     datasets = {
#         'train': FractureSeverityDataset(config['train_img_dir'], config['train_label_dir'], mode='train'),
#         'validation': FractureSeverityDataset(config['val_img_dir'], config['val_label_dir'], mode='val'),
#         'test': FractureSeverityDataset(config['test_img_dir'], config['test_label_dir'], mode='val')
#     }
    
#     # Create dataloaders
#     dataloaders = {
#         name: DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
#         for name, dataset in datasets.items()
#     }
    
#     # Evaluate each dataset
#     all_metrics = {}
#     for dataset_name, dataloader in dataloaders.items():
#         # Get predictions
#         true_labels, predicted_labels, confidences = analyzer.evaluate_dataset(dataloader, dataset_name)
        
#         # Calculate metrics
#         metrics = analyzer.calculate_metrics(true_labels, predicted_labels, confidences)
#         all_metrics[dataset_name] = metrics
        
#         # Plot confusion matrix
#         analyzer.plot_confusion_matrix(
#             metrics['confusion_matrix'],
#             dataset_name,
#             config['results_dir']
#         )
        
#         # Plot class metrics
#         analyzer.plot_class_metrics(
#             metrics,
#             dataset_name,
#             config['results_dir']
#         )
        
#         # Print results
#         print(f"\nResults for {dataset_name} dataset:")
#         print(f"Accuracy: {metrics['accuracy']:.4f}")
#         print(f"Precision: {metrics['precision']:.4f}")
#         print(f"Recall: {metrics['recall']:.4f}")
#         print(f"F1 Score: {metrics['f1_score']:.4f}")
#         print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        
#         print("\nPer-class metrics:")
#         for grade, class_metrics in metrics['per_class_metrics'].items():
#             print(f"\n{grade}:")
#             for metric_name, value in class_metrics.items():
#                 if isinstance(value, (int, float)):
#                     print(f"  {metric_name}: {value:.4f}")
#                 else:
#                     print(f"  {metric_name}: {value}")
    
#     # Save all metrics to Excel
#     analyzer.save_metrics_to_excel(
#         all_metrics,
#         os.path.join(config['results_dir'], 'severity_metrics.xlsx')
#     )
    
#     print("\nPerformance analysis complete!")

#     print("\nAnalyzing dataset distributions...")
#     distributions = {}
#     for dataset_name, dataloader in dataloaders.items():
#         distributions[dataset_name] = analyzer.analyze_dataset_distribution(dataloader, dataset_name)


# if __name__ == "__main__":
#     main()



import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from fracture_model import FracturePatternModel
from dataset_handler import FracturePatternDataset

def evaluate_model(model, data_loader, device, pattern_names):
    model.eval()
    all_pattern_preds = []
    all_pattern_labels = []
    all_joint_preds = []
    all_joint_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            images = batch['image'].to(device)
            pattern_labels = batch['pattern'].to(device)
            joint_labels = batch['joint_involvement'].to(device)

            outputs = model(images)
            pattern_out = outputs['pattern']
            joint_out = outputs['joint_involvement']

            # Compute loss
            loss = criterion(pattern_out, pattern_labels)
            total_loss += loss.item()

            # Pattern predictions
            _, pattern_predicted = torch.max(pattern_out.data, 1)
            all_pattern_preds.extend(pattern_predicted.cpu().numpy())
            all_pattern_labels.extend(pattern_labels.cpu().numpy())

            # Joint involvement predictions
            joint_predicted = (joint_out > 0.5).float()
            all_joint_preds.extend(joint_predicted.cpu().numpy())
            all_joint_labels.extend(joint_labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    pattern_accuracy = (np.array(all_pattern_preds) == np.array(all_pattern_labels)).mean() * 100
    joint_accuracy = (np.array(all_joint_preds) == np.array(all_joint_labels)).mean() * 100

    # Classification report for patterns
    print("\nClassification Report for Fracture Patterns:")
    print(classification_report(all_pattern_labels, all_pattern_preds, target_names=pattern_names))

    # Confusion matrix for patterns
    cm = confusion_matrix(all_pattern_labels, all_pattern_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pattern_names, yticklabels=pattern_names)
    plt.title("Confusion Matrix - Fracture Patterns")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.close()

    return {
        'avg_loss': avg_loss,
        'pattern_accuracy': pattern_accuracy,
        'joint_accuracy': joint_accuracy,
        'pattern_preds': all_pattern_preds,
        'pattern_labels': all_pattern_labels,
        'joint_preds': all_joint_preds,
        'joint_labels': all_joint_labels
    }

def main():
    # Configuration
    config = {
        'val_img_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\images\test',
        'val_label_dir': r'C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5\labels\test',
        'batch_size': 16,
        'num_workers': 4,
        'image_size': 640,
        'checkpoint_path': './models/best_model_20250221_151300.pth',  # Path to your trained model
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pattern_names = ["Simple", "Wedge", "Comminuted"]

    # Load dataset
    val_dataset = FracturePatternDataset(
        config['val_img_dir'],
        config['val_label_dir'],
        mode='val',
        image_size=config['image_size']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Load model
    model = FracturePatternModel(pretrained=False).to(device)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {config['checkpoint_path']}")

    # Evaluate
    metrics = evaluate_model(model, val_loader, device, pattern_names)

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Loss: {metrics['avg_loss']:.4f}")
    print(f"Pattern Classification Accuracy: {metrics['pattern_accuracy']:.2f}%")
    print(f"Joint Involvement Accuracy: {metrics['joint_accuracy']:.2f}%")
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()