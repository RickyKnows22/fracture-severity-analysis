#densenet_inference.py

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class DenseNetDetector(nn.Module):
    def __init__(self, num_classes=9):
        super(DenseNetDetector, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        features = self.densenet(x)
        return self.detection_head(features)

class DenseNetInference:
    def __init__(self, model_path, device=None):
        """Initialize inference"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 416  # Same as training
        
        # Initialize model
        self.model = DenseNetDetector().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Class names matching YOLOv7
        self.classes = [
            "boneanomaly", "bonelesion", "foreignbody", 
            "fracture", "metal", "periostealreaction", 
            "pronatorsign", "softtissue", "text"
        ]
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        return image_tensor.unsqueeze(0).to(self.device)
    
    def process_predictions(self, predictions, confidence_threshold=0.25):
        """Process model predictions into detections"""
        predictions = torch.sigmoid(predictions[0])  # Remove batch dimension and apply sigmoid
        
        results = {}
        vis_results = []
        H, W = self.image_size, self.image_size
        
        for class_idx, class_name in enumerate(self.classes):
            class_pred = predictions[class_idx]
            confidence = float(torch.max(class_pred))
            detected = confidence > confidence_threshold
            
            results[class_name] = {
                'confidence': confidence,
                'detected': detected
            }
            
            if detected:
                # Find location of highest confidence
                max_idx = torch.argmax(class_pred)
                y, x = max_idx // class_pred.shape[1], max_idx % class_pred.shape[1]
                
                # Convert to image coordinates
                x_center = (x.item() + 0.5) * (W / class_pred.shape[1])
                y_center = (y.item() + 0.5) * (H / class_pred.shape[0])
                width = W / class_pred.shape[1]
                height = H / class_pred.shape[0]
                
                vis_results.append({
                    'class_id': class_idx,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x_center/W, y_center/H, width/W, height/H]  # Normalized coordinates
                })
        
        return results, vis_results
    
    def draw_detections(self, image, detections):
        """Draw detections on image"""
        image = image.copy()
        H, W = image.shape[:2]
        
        colors = [
            (0,255,0), (255,0,0), (0,0,255), (255,255,0),
            (0,255,255), (255,0,255), (128,128,0), (0,128,128),
            (128,0,128)
        ]
        
        for det in detections:
            # Get detection info
            class_name = det['class_name']
            confidence = det['confidence']
            bbox = det['bbox']
            color = colors[det['class_id'] % len(colors)]
            
            # Convert normalized coordinates to pixel coordinates
            x_center, y_center, width, height = bbox
            x1 = int((x_center - width/2) * W)
            y1 = int((y_center - height/2) * H)
            x2 = int((x_center + width/2) * W)
            y2 = int((y_center + height/2) * H)
            
            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        return image
    
    def __call__(self, image, confidence_threshold=0.25):
        """Run inference on an image"""
        # Ensure image is in correct format
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        original_image = image.copy()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        results, vis_results = self.process_predictions(predictions, confidence_threshold)
        
        # Draw detections
        output_image = self.draw_detections(original_image, vis_results)
        
        return results, output_image

def densenet_inference(image, device="cpu"):
    """Wrapper function for main application"""
    # Initialize detector
    detector = DenseNetInference("best_densenet_detector.pth", device)
    
    # Run inference
    results, annotated_image = detector(image)
    
    return results, annotated_image

# Example usage
if __name__ == "__main__":
    # Initialize detector
    model_path = "best_densenet_detector.pth"
    detector = DenseNetInference(model_path)
    
    # Load and process image
    image_path = "path/to/test/image.png"
    image = cv2.imread(image_path)
    
    # Run inference
    results, output_image = detector(image)
    
    # Print results
    print("\nDetections:")
    for class_name, result in results.items():
        if result['detected']:
            print(f"{class_name}: {result['confidence']:.2f}")
    
    # Save output image
    cv2.imwrite("output.png", output_image)