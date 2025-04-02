import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import onnxruntime as ort
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from densenet_inference import DenseNetInference
import os
from collections import defaultdict

class ModelEvaluator:
    def __init__(self):
        # Base paths
        self.resnet_base = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\Dataset")
        self.detection_base = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset\yolov5")
        
        # Model paths
        self.resnet_path = "weights/ResNet50_BodyParts.h5"
        self.yolo_path = "yolov7-p6-bonefracture.onnx"
        self.densenet_path = "best_densenet_detector.pth"
        
        # Create output directory
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mappings
        self.body_parts = ["Elbow", "Hand", "Shoulder"]
        self.detection_classes = ["boneanomaly", "bonelesion", "foreignbody", 
                                "fracture", "metal", "periostealreaction", 
                                "pronatorsign", "softtissue", "text"]
        
        # Initialize results storage
        self.results = {}
        
    def load_models(self):
        """Load all three models"""
        print("\nLoading models...")
        # ResNet50
        print("Loading ResNet50...")
        self.resnet_model = tf.keras.models.load_model(self.resnet_path)
        
        # YOLOv7
        print("Loading YOLOv7...")
        self.yolo_session = ort.InferenceSession(self.yolo_path, providers=['CPUExecutionProvider'])
        
        # DenseNet
        print("Loading DenseNet...")
        self.densenet = DenseNetInference(self.densenet_path, "cpu")
        
        print("All models loaded successfully!")
    
    def evaluate_resnet(self):
        print("\nEvaluating ResNet50...")
        y_true = []
        y_pred = []
        confidences = []
        file_paths = []
        dataset_info = defaultdict(lambda: defaultdict(int))
        
        test_path = self.resnet_base / "test"
        print(f"\nTest path: {test_path}")
        
        # Print full directory structure for validation
        print("\nDirectory Structure:")
        for body_part in self.body_parts:
            print(f"\n{body_part}:")
            part_path = test_path / body_part
            
            if not part_path.exists():
                print(f"Warning: Path does not exist - {part_path}")
                continue
                
            for patient_dir in part_path.glob("patient*"):
                print(f"  {patient_dir.name}:")
                for study_dir in patient_dir.glob("study1_*"):
                    image_count = len(list(study_dir.glob("*.png")))
                    print(f"    {study_dir.name}/: {image_count} images")
                    dataset_info[body_part]['total_images'] += image_count
                    if "positive" in study_dir.name:
                        dataset_info[body_part]['positive_images'] += image_count
                    else:
                        dataset_info[body_part]['negative_images'] += image_count
        
        # Process images with correct folder structure
        for class_idx, body_part in enumerate(self.body_parts):
            part_path = test_path / body_part
            if not part_path.exists():
                continue
                
            print(f"\nProcessing {body_part}...")
            
            for patient_dir in part_path.glob("patient*"):
                for study_dir in patient_dir.glob("study1_*"):
                    is_positive = "positive" in study_dir.name
                    
                    for img_path in study_dir.glob("*.png"):
                        try:
                            # Load and preprocess image
                            img = image.load_img(img_path, target_size=(224, 224))
                            img_array = image.img_to_array(img)
                            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                            img_array = np.expand_dims(img_array, axis=0)
                            
                            # Make prediction
                            prediction = self.resnet_model.predict(img_array, verbose=0)
                            pred_class = np.argmax(prediction[0])
                            confidence = np.max(prediction[0])
                            
                            # Store prediction info
                            y_true.append(class_idx)
                            y_pred.append(pred_class)
                            confidences.append(confidence)
                            file_paths.append(str(img_path))
                            
                            # Log incorrect predictions
                            if pred_class != class_idx:
                                print(f"\nIncorrect Prediction:")
                                print(f"Image: {img_path}")
                                print(f"Patient: {patient_dir.name}")
                                print(f"Study: {study_dir.name}")
                                print(f"True class: {body_part}")
                                print(f"Predicted as: {self.body_parts[pred_class]}")
                                print(f"Confidence: {confidence:.4f}")
                                
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        for body_part in self.body_parts:
            print(f"\n{body_part}:")
            print(f"  Total images: {dataset_info[body_part]['total_images']}")
            print(f"  Positive studies: {dataset_info[body_part]['positive_images']}")
            print(f"  Negative studies: {dataset_info[body_part]['negative_images']}")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClass Order:", self.body_parts)
        
        return metrics, dataset_info, file_paths

    def evaluate_yolo(self):
        """Evaluate YOLOv7 detector"""
        print("\nEvaluating YOLOv7...")
        test_df = pd.read_csv(self.detection_base.parent / "test_data.csv")
        
        all_true = []
        all_pred = []
        all_confidences = []
        file_paths = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                img_path = self.detection_base / 'images' / 'test' / f"{row['filestem']}.png"
                label_path = self.detection_base / 'labels' / 'test' / f"{row['filestem']}.txt"
                
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_input = cv2.resize(img, (640, 640))
                img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
                img_input = np.expand_dims(img_input, axis=0)
                
                # Run inference
                outputs = self.yolo_session.run(None, 
                    {self.yolo_session.get_inputs()[0].name: img_input})[0]
                
                # Process predictions
                predictions = np.zeros(len(self.detection_classes))
                confidences = np.zeros(len(self.detection_classes))
                
                for detection in outputs:
                    confidence = detection[4]
                    class_id = int(detection[5])
                    if confidence > confidences[class_id]:
                        confidences[class_id] = confidence
                        predictions[class_id] = 1
                
                # Load ground truth
                true_labels = np.zeros(len(self.detection_classes))
                if label_path.exists():
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
        
        # Convert to numpy arrays
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_true.flatten(), all_pred.flatten()),
            'precision': precision_score(all_true, all_pred, average='weighted', zero_division=0),
            'recall': recall_score(all_true, all_pred, average='weighted', zero_division=0),
            'f1': f1_score(all_true, all_pred, average='weighted', zero_division=0)
        }
        
        # Calculate per-class metrics
        for i, cls in enumerate(self.detection_classes):
            metrics[f'{cls}_accuracy'] = accuracy_score(all_true[:, i], all_pred[:, i])
            metrics[f'{cls}_precision'] = precision_score(all_true[:, i], all_pred[:, i], zero_division=0)
            metrics[f'{cls}_recall'] = recall_score(all_true[:, i], all_pred[:, i], zero_division=0)
            metrics[f'{cls}_f1'] = f1_score(all_true[:, i], all_pred[:, i], zero_division=0)
            
            # Generate per-class confusion matrix
            cm = confusion_matrix(all_true[:, i], all_pred[:, i])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'YOLOv7 Confusion Matrix - {cls}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.output_dir / f'yolov7_confusion_matrix_{cls}.png')
            plt.close()
        
        # Plot confidence distributions
        plt.figure(figsize=(12, 6))
        plt.boxplot([all_confidences[:, i] for i in range(len(self.detection_classes))],
                   labels=self.detection_classes)
        plt.xticks(rotation=45)
        plt.title('YOLOv7 Confidence Distribution by Class')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yolov7_confidence_distribution.png')
        plt.close()
        
        # Save detailed results
        results_df = pd.DataFrame(all_pred, columns=self.detection_classes)
        results_df['file_path'] = file_paths
        results_df.to_csv(self.output_dir / 'yolov7_results.csv', index=False)
        
        return metrics

    def evaluate_densenet(self):
        """Evaluate DenseNet detector"""
        print("\nEvaluating DenseNet...")
        test_df = pd.read_csv(self.detection_base.parent / "test_data.csv")
        
        all_true = []
        all_pred = []
        all_confidences = []
        file_paths = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                img_path = self.detection_base / 'images' / 'test' / f"{row['filestem']}.png"
                label_path = self.detection_base / 'labels' / 'test' / f"{row['filestem']}.txt"
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Get predictions
                results, _ = self.densenet(img)
                
                # Process predictions
                predictions = np.zeros(len(self.detection_classes))
                confidences = np.zeros(len(self.detection_classes))
                
                for i, cls in enumerate(self.detection_classes):
                    predictions[i] = 1 if results[cls]['detected'] else 0
                    confidences[i] = results[cls]['confidence']
                
                # Load ground truth
                true_labels = np.zeros(len(self.detection_classes))
                if label_path.exists():
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
        
        # Convert to numpy arrays
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_true.flatten(), all_pred.flatten()),
            'precision': precision_score(all_true, all_pred, average='weighted', zero_division=0),
            'recall': recall_score(all_true, all_pred, average='weighted', zero_division=0),
            'f1': f1_score(all_true, all_pred, average='weighted', zero_division=0)
        }
        
        # Calculate per-class metrics
        for i, cls in enumerate(self.detection_classes):
            metrics[f'{cls}_accuracy'] = accuracy_score(all_true[:, i], all_pred[:, i])
            metrics[f'{cls}_precision'] = precision_score(all_true[:, i], all_pred[:, i], zero_division=0)
            metrics[f'{cls}_recall'] = recall_score(all_true[:, i], all_pred[:, i], zero_division=0)
            metrics[f'{cls}_f1'] = f1_score(all_true[:, i], all_pred[:, i], zero_division=0)
            
            # Generate per-class confusion matrix
            cm = confusion_matrix(all_true[:, i], all_pred[:, i])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'DenseNet Confusion Matrix - {cls}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.output_dir / f'densenet_confusion_matrix_{cls}.png')
            plt.close()
        
        # Plot confidence distributions
        plt.figure(figsize=(12, 6))
        plt.boxplot([all_confidences[:, i] for i in range(len(self.detection_classes))],
                   labels=self.detection_classes)
        plt.xticks(rotation=45)
        plt.title('DenseNet Confidence Distribution by Class')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'densenet_confidence_distribution.png')
        plt.close()
        
        # Save detailed results
        results_df = pd.DataFrame(all_pred, columns=self.detection_classes)
        results_df['file_path'] = file_paths
        results_df.to_csv(self.output_dir / 'densenet_results.csv', index=False)
        
        return metrics

    def plot_comparative_metrics(self):
        """Generate comparative visualizations between models"""
        print("\nGenerating comparative visualizations...")
        
        # Compare overall metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = ['ResNet50', 'YOLOv7', 'DenseNet']
        
        # Create bar plot for overall metrics
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [self.results[model][metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_overall.png')
        plt.close()
        
        # Compare detection models for each class
        if 'YOLOv7' in self.results and 'DenseNet' in self.results:
            for cls in self.detection_classes:
                plt.figure(figsize=(10, 5))
                metrics_cls = [f'{cls}_accuracy', f'{cls}_precision', f'{cls}_recall', f'{cls}_f1']
                x = np.arange(len(metrics_cls))
                width = 0.35
                
                yolo_values = [self.results['YOLOv7'][metric] for metric in metrics_cls]
                densenet_values = [self.results['DenseNet'][metric] for metric in metrics_cls]
                
                plt.bar(x - width/2, yolo_values, width, label='YOLOv7')
                plt.bar(x + width/2, densenet_values, width, label='DenseNet')
                
                plt.xlabel('Metrics')
                plt.ylabel('Score')
                plt.title(f'Model Comparison for {cls}')
                plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1'])
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.output_dir / f'model_comparison_{cls}.png')
                plt.close()

    def save_results(self):
        """Save all results to CSV files"""
        print("\nSaving results...")
        
        # Save overall results
        overall_metrics = {}
        for model, metrics in self.results.items():
            # Skip non-metric results
            if model.endswith('_info') or model.endswith('_files'):
                continue
                
            # Handle metrics differently based on type
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if not isinstance(value, (list, np.ndarray)):
                        overall_metrics[f"{model}_{metric}"] = value
        
        pd.DataFrame([overall_metrics]).to_csv(self.output_dir / 'overall_metrics.csv', index=False)
        
        # Save detailed per-model results
        for model, metrics in self.results.items():
            # Skip non-metric results
            if model.endswith('_info') or model.endswith('_files'):
                continue
                
            if isinstance(metrics, dict):
                pd.DataFrame([metrics]).to_csv(self.output_dir / f'{model.lower()}_metrics.csv', index=False)
        
        # Save ResNet50 additional info if available
        if 'ResNet50_info' in self.results:
            info_df = pd.DataFrame(self.results['ResNet50_info'])
            info_df.to_csv(self.output_dir / 'resnet50_dataset_info.csv')
        
        if 'ResNet50_files' in self.results:
            files_df = pd.DataFrame({'file_path': self.results['ResNet50_files']})
            files_df.to_csv(self.output_dir / 'resnet50_processed_files.csv', index=False)


    def evaluate_all(self):
        """Run complete evaluation pipeline"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load models
            self.load_models()
            
            # Evaluate each model
            resnet_metrics, resnet_info, resnet_files = self.evaluate_resnet()
            self.results['ResNet50'] = resnet_metrics
            self.results['ResNet50_info'] = resnet_info
            self.results['ResNet50_files'] = resnet_files
            
            self.results['YOLOv7'] = self.evaluate_yolo()
            self.results['DenseNet'] = self.evaluate_densenet()
            
            # Generate comparative visualizations
            self.plot_comparative_metrics()
            
            # Save all results
            self.save_results()
            
            print("\nEvaluation complete! Results saved in 'evaluation_results' directory.")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()