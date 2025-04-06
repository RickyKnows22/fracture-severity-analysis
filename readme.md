# Advanced Bone Fracture Analysis System

## Overview
The Advanced Bone Fracture Analysis System is a comprehensive AI-powered platform designed to assist medical professionals in detecting, analyzing, and classifying bone fractures from X-ray images. The system combines multiple state-of-the-art deep learning models to provide a multi-layered analysis pipeline, from initial fracture detection to treatment recommendations.

## Problem Statement
In emergency departments and orthopedic clinics worldwide, bone fracture diagnosis remains a critical challenge with significant time pressure. For patients, every minute waiting for diagnosis means prolonged pain, anxiety, and potential complications. For medical professionals, the challenge is equally pressing as they face increasing workloads with limited time to analyze each X-ray thoroughly. Our system addresses these immediate challenges by providing rapid, accurate fracture detection and classification, helping both patients receive timely care and doctors make confident treatment decisions.

## Features
- **Multi-level Detection & Analysis**:
  - Body part classification (Elbow, Hand, Shoulder)
  - Fracture detection with dual-model verification (YOLOv7 and DenseNet)
  - Fracture pattern classification (Simple, Wedge, Comminuted)
  - Joint involvement assessment
  - Displacement severity evaluation
  
- **Treatment Recommendation**:
  - Evidence-based treatment suggestions based on fracture characteristics
  - Recommendations tailored to fracture type, displacement, and joint involvement
  
- **User-Friendly Interface**:
  - Secure login system with Google OAuth integration
  - Simple upload interface for X-ray images
  - Clear visual results presentation
  - Detailed analysis visualization

## Technology Stack
- **Deep Learning Models**:
  - ResNet50 for body part classification
  - YOLOv7 for primary fracture detection
  - DenseNet121 for verification detection
  - EfficientNetV2 for advanced fracture pattern analysis
  
- **Backend**:
  - Python
  - PyTorch/TensorFlow
  - ONNX Runtime
  - OpenCV
  - SQLite
  
- **Frontend**:
  - Streamlit

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bone-fracture-analysis.git
   cd bone-fracture-analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

4. Download the pre-trained models:
   ```
   python download_models.py
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run mainapp.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Login using your credentials or sign up for a new account

4. Upload an X-ray image for analysis

5. View the results including:
   - Body part classification
   - Fracture detection
   - Fracture pattern analysis
   - Treatment recommendations

## Model Performance

Our severity analysis model demonstrates strong performance on normal and severe fracture cases, with high true positive rates across test, training, and validation sets. The model shows clinically reasonable "neighbor confusion," primarily misclassifying between adjacent severity categories (minimal/moderate and moderate/severe), which reflects the continuous nature of fracture severity in real-world diagnostics.

## Project Structure
```
├── mainapp.py              # Main Streamlit application
├── predictions.py          # Body part and initial fracture prediction functions
├── fracture_model.py       # Advanced fracture pattern model architecture
├── dataset_handler.py      # Dataset processing for model training
├── train_fracture_severity.py  # Training pipeline for fracture severity model
├── database_handler.py     # Treatment recommendation database interface
├── densenet_inference.py   # DenseNet model inference functions
├── models/                 # Pre-trained model files
│   ├── best_model_20250221_151300.pth  # EfficientNetV2 model
│   ├── ResNet50_BodyParts.h5          # Body part classification model
│   └── yolov7-p6-bonefracture.onnx    # YOLO fracture detection model
└── weights/                # Additional model weights
```

## Future Work
- Collaboration with hospitals to expand the dataset with more diverse and balanced severity examples
- Improved labeling through expert consensus panels of radiologists
- Integration of clinical outcome data to correlate fracture classifications with treatment success
- Exploration of confidence-calibrated predictions to help clinicians better interpret model uncertainty
- Development of specialized models for specific bone types or demographic groups

## License
[MIT License](LICENSE)

## Acknowledgements
- MURA Dataset
- GRAZPEDWRI-DX Dataset
- PyTorch, TensorFlow, and Streamlit communities
