#mainapp.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
import sqlite3
import hashlib
import torch
import albumentations as A
import smtplib
import random
import sqlite3
import hashlib
import json
import requests

from matplotlib.colors import TABLEAU_COLORS 
from pathlib import Path
from PIL import Image
from predictions import predict
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from densenet_inference import DenseNetInference  # Import the new DenseNet inference
from train_fracture_severity import ModernFractureSeverityModel
from fracture_model import FracturePatternModel
from database_handler import get_treatment_recommendation, create_database
from urllib.parse import urlencode



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize pattern predictor globally
fracture_pattern_model = FracturePatternModel(pretrained=False)
# Load checkpoint
checkpoint = torch.load('./models/best_model_20250221_151300.pth', 
                       map_location=device,
                       weights_only=True)
fracture_pattern_model.load_state_dict(checkpoint['model_state_dict'])
fracture_pattern_model = fracture_pattern_model.to(device)
fracture_pattern_model.eval()

st.set_page_config(page_title="Advanced Bone Fracture Detection", page_icon="🦴", layout="wide")

# Fix for OpenMP error

# Global configurations
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}
parent_root = Path(__file__).parent.parent.absolute().__str__()
h, w = 640, 640
model_onnx_path = os.path.join(parent_root, "SkeletalScan-main/yolov7-p6-bonefracture.onnx")


def initialize_session_state():
    """Initialize all required session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = ""
        
    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    
    # OTP related
    if 'otp_code' not in st.session_state:
        st.session_state['otp_code'] = {}


def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def load_img(uploaded_file):
    if isinstance(uploaded_file, bytes):
        file_bytes = np.asarray(bytearray(uploaded_file), dtype=np.uint8)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image[..., ::-1]

def preproc(img):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32).transpose(2, 0, 1)/255
    return np.expand_dims(img, axis=0)

def xyxy2xywhn(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def xywhn2xyxy(bbox, H, W):
    x, y, w, h = bbox
    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def model_inference(model_path, image_np, device="cpu"):
    providers = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})
    return output[0][:, :6]

def densenet_inference(image, device="cpu"):
    # Initialize the new DenseNet detector
    try:
        detector = DenseNetInference("best_densenet_detector.pth", device)
        results, vis_img = detector(image)
        return results, vis_img
    except Exception as e:
        st.error(f"Error in DenseNet inference: {e}")
        return {}, image.copy()

def post_process(img, output, score_threshold=0.3):
    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7: "softtissue", 8: "text"
    }

    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    h, w = 640, 640
    label_txt = ""
    found_fracture = False
    detection_stats = {}

    # Initialize detection stats
    for class_name in id2names.values():
        detection_stats[class_name] = {
            'count': 0,
            'confidences': [],
            'max_confidence': 0,
            'avg_confidence': 0
        }

    for idx in range(len(det_bboxes)):
        if det_scores[idx] > score_threshold:
            bbox = det_bboxes[idx]
            label = det_labels[idx]
            score = det_scores[idx]
            class_name = id2names[int(label)]
            
            # Update detection stats
            stats = detection_stats[class_name]
            stats['count'] += 1
            stats['confidences'].append(score)
            stats['max_confidence'] = max(stats['max_confidence'], score)
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            
            if class_name == "fracture":
                found_fracture = True
            
            # Scale bbox coordinates back to original image dimensions
            bbox = bbox @ np.array([[W/w, 0, 0, 0],
                                  [0, H/h, 0, 0],
                                  [0, 0, W/w, 0],
                                  [0, 0, 0, H/h]])
            
            bbox_int = [int(x) for x in bbox]
            x1, y1, x2, y2 = bbox_int
            color_map = colors[int(label)]
            txt = f"{class_name} {score:.2f}"
            
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
            cv2.rectangle(img, (x1-2, y1-text_height-10), (x1 + text_width+2, y1), color_map, -1)
            cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, label_txt, found_fracture, detection_stats

def format_detection_markdown(detection_stats):
    if not detection_stats:
        return "No anomalies detected"
    
    markdown = "### Detected Classes:\n"
    for class_name, stats in detection_stats.items():
        markdown += f"#### {class_name.title()}\n"
        markdown += f"- Count: {stats['count']}\n"
        markdown += f"- Average Confidence: {stats['avg_confidence']:.2%}\n"
        markdown += f"- Highest Confidence: {stats['max_confidence']:.2%}\n"
    return markdown




def analyze_fracture_severity(image, yolo_output, conf_thres=0.3, device='cuda'):
    """
    Analyze fracture pattern using medical classification criteria
    """
    det_bboxes = yolo_output[:, 0:4]
    det_scores = yolo_output[:, 4]
    det_labels = yolo_output[:, 5]
    
    H, W = image.shape[:2]
    
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    analysis_results = []
    
    for idx in range(len(det_bboxes)):
        if det_scores[idx] > conf_thres and det_labels[idx] == 3:  # Fracture class
            bbox = det_bboxes[idx]
            
            # Scale coordinates to original image dimensions
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det_scores[idx]
            
            # Add padding for context
            pad = 20
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(W, x2 + pad)
            y2_pad = min(H, y2 + pad)
            
            region = image_gray[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if region.size == 0:
                continue
                
            # Add channel dimension
            region = np.expand_dims(region, -1)
            
            try:
                transformed = transform(image=region)
                image_tensor = transformed['image'].unsqueeze(0).to(device)
                
                # Get pattern prediction
                with torch.no_grad():
                    outputs = fracture_pattern_model(image_tensor)
                    
                    pattern_probs = torch.softmax(outputs['pattern'], dim=1)
                    pattern_class = torch.argmax(outputs['pattern'], dim=1).item()
                    pattern_conf = pattern_probs[0][pattern_class].item()
                    
                    joint_prob = outputs['joint_involvement'][0].item()
                    
                    disp_probs = torch.softmax(outputs['displacement'], dim=1)
                    disp_class = torch.argmax(outputs['displacement'], dim=1).item()
                    disp_conf = disp_probs[0][disp_class].item()
                    
                    pattern_names = ['Simple', 'Wedge', 'Comminuted']
                    displacement_names = ['Minimal', 'Moderate', 'Severe']
                    
                    analysis_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'bbox_padded': (x1_pad, y1_pad, x2_pad, y2_pad),
                        'detection_confidence': confidence,
                        'pattern': {
                            'class': pattern_names[pattern_class],
                            'confidence': pattern_conf,
                            'probabilities': pattern_probs[0].cpu().numpy()
                        },
                        'joint_involvement': {
                            'present': joint_prob > 0.5,
                            'confidence': joint_prob
                        },
                        'displacement': {
                            'class': displacement_names[disp_class],
                            'confidence': disp_conf,
                            'probabilities': disp_probs[0].cpu().numpy()
                        },
                        'region': region.squeeze()
                    })
            except Exception as e:
                print(f"Error processing region: {e}")
                continue
    
    return analysis_results

def analyze_fracture_pattern(image, yolo_output, conf_thres=0.3, device='cuda'):
    """
    Analyze fracture pattern using medical classification system
    """
    det_bboxes = yolo_output[:, 0:4]
    det_scores = yolo_output[:, 4]
    det_labels = yolo_output[:, 5]
    
    H, W = image.shape[:2]
    h, w = 640, 640  # YOLO input size
    
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    analysis_results = []
    
    for idx in range(len(det_bboxes)):
        if det_scores[idx] > conf_thres and det_labels[idx] == 3:  # Fracture class
            bbox = det_bboxes[idx]
            
            # Scale coordinates back to original image dimensions
            bbox = bbox @ np.array([[W/w, 0, 0, 0],
                                  [0, H/h, 0, 0],
                                  [0, 0, W/w, 0],
                                  [0, 0, 0, H/h]])
            
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det_scores[idx]
            
            # Add padding for context
            pad = 20
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(W, x2 + pad)
            y2_pad = min(H, y2 + pad)
            
            region = image_gray[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if region.size == 0:
                continue
                
            # Add channel dimension
            region = np.expand_dims(region, -1)
            
            try:
                transformed = transform(image=region)
                image_tensor = transformed['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = fracture_pattern_model(image_tensor)
                    
                    pattern_probs = torch.softmax(outputs['pattern'], dim=1)
                    pattern_class = torch.argmax(outputs['pattern'], dim=1).item()
                    pattern_conf = pattern_probs[0][pattern_class].item()
                    
                    joint_prob = torch.sigmoid(outputs['joint_involvement']).squeeze().item()
                    
                    disp_probs = torch.softmax(outputs['displacement'], dim=1)
                    disp_class = torch.argmax(outputs['displacement'], dim=1).item()
                    disp_conf = disp_probs[0][disp_class].item()
                    
                    pattern_names = ['Simple', 'Wedge', 'Comminuted']
                    displacement_names = ['Minimal', 'Moderate', 'Severe']
                    
                    analysis_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'bbox_padded': (x1_pad, y1_pad, x2_pad, y2_pad),
                        'detection_confidence': confidence,
                        'pattern': {
                            'class': pattern_names[pattern_class],
                            'confidence': pattern_conf,
                            'probabilities': pattern_probs[0].cpu().numpy()
                        },
                        'joint_involvement': {
                            'present': joint_prob > 0.5,
                            'confidence': joint_prob
                        },
                        'displacement': {
                            'class': displacement_names[disp_class],
                            'confidence': disp_conf,
                            'probabilities': disp_probs[0].cpu().numpy()
                        },
                        'region': region.squeeze()
                    })
            except Exception as e:
                print(f"Error processing region: {e}")
                continue
    
    return analysis_results

def visualize_pattern(analysis_results, original_image):
    """Create visualization with medical classification results"""
    vis_img = original_image.copy()
    pattern_colors = {
        'Simple': (0, 255, 0),     # Green
        'Wedge': (255, 165, 0),    # Orange
        'Comminuted': (255, 0, 0)  # Red
    }
    
    for result in analysis_results:
        x1, y1, x2, y2 = result['bbox']
        pattern = result['pattern']['class']
        pattern_conf = result['pattern']['confidence']
        joint_involved = result['joint_involvement']['present']
        displacement = result['displacement']['class']
        
        color = pattern_colors[pattern]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        text_lines = [
            f"Pattern: {pattern} ({pattern_conf:.2f})",
            f"Displacement: {displacement}",
            f"Joint: {'Yes' if joint_involved else 'No'}"
        ]
        
        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        padding = 5
        
        # Calculate text dimensions
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
        max_width = max(size[0] for size in text_sizes)
        line_height = max(size[1] for size in text_sizes)
        total_height = (line_height + padding) * len(text_lines)
        
        # Draw background
        cv2.rectangle(vis_img, 
                     (x1, y1 - total_height - padding * 2),
                     (x1 + max_width + padding * 2, y1),
                     color, -1)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = y1 - total_height + (i * (line_height + padding)) + line_height
            cv2.putText(vis_img, line, 
                       (x1 + padding, y_pos),
                       font, font_scale, (0, 0, 0), thickness)
    
    return vis_img

def visualize_severity(analysis_results, original_image):
    """Create visualization with medical classification results"""
    vis_img = original_image.copy()
    pattern_colors = {
        'Simple': (0, 255, 0),     # Green
        'Wedge': (255, 165, 0),    # Orange
        'Comminuted': (255, 0, 0)  # Red
    }
    
    for result in analysis_results:
        x1, y1, x2, y2 = result['bbox']
        pattern = result['pattern']['class']
        pattern_conf = result['pattern']['confidence']
        joint_involved = result['joint_involvement']['present']
        displacement = result['displacement']['class']
        
        color = pattern_colors[pattern]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Create detailed label text
        text_lines = [
            f"Pattern: {pattern} ({pattern_conf:.2f})",
            f"Displacement: {displacement}",
            f"Joint: {'Yes' if joint_involved else 'No'}"
        ]
        
        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        padding = 5
        
        # Calculate text dimensions
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
        max_width = max(size[0] for size in text_sizes)
        line_height = max(size[1] for size in text_sizes)
        total_height = (line_height + padding) * len(text_lines)
        
        # Draw background
        cv2.rectangle(vis_img, 
                     (x1, y1 - total_height - padding * 2),
                     (x1 + max_width + padding * 2, y1),
                     color, -1)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = y1 - total_height + (i * (line_height + padding)) + line_height
            cv2.putText(vis_img, line, 
                       (x1 + padding, y_pos),
                       font, font_scale, (0, 0, 0), thickness)
    
    return vis_img


# Store OTPs temporarily
otp_storage = {}

# Hash Passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Create Users Table
def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  email TEXT UNIQUE, 
                  password TEXT)''')
    conn.commit()
    conn.close()

# Send OTP Email
def send_otp_email(email, otp):
    subject = "Your OTP Code for Signup"
    body = f"Your OTP code is {otp}. Please enter this to complete your signup."
    
    message = f"Subject: {subject}\n\n{body}"
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, email, message)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Signup Process with OTP Verification
# Debug OTP validation issue
def signup(username, email, password, otp_entered):
    """
    Fixed signup function that properly validates OTP
    """
    # Debug print
    print(f"Signup attempt - Email: {email}, OTP entered: {otp_entered}")
    print(f"Stored OTPs: {otp_storage}")
    
    # Convert OTP to string for comparison (in case it comes in as a different type)
    str_otp_entered = str(otp_entered)
    
    if email not in otp_storage:
        print(f"Email {email} not found in OTP storage")
        return "Invalid or expired OTP"
        
    stored_otp = str(otp_storage[email])
    print(f"Stored OTP: {stored_otp}, OTP entered: {str_otp_entered}")
    
    if stored_otp != str_otp_entered:
        print(f"OTP mismatch - stored: {stored_otp}, entered: {str_otp_entered}")
        return "Invalid or expired OTP"

    # OTP is valid, proceed with signup
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, hashed_password))
        conn.commit()
        # Remove OTP after successful signup
        del otp_storage[email]
        print(f"Signup successful for {email}")
        return "Success"
    except sqlite3.IntegrityError:
        print(f"IntegrityError for {email} - user may already exist")
        return "Username or email already exists"
    finally:
        conn.close()



# Login Function
def login(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Google OAuth Login
def google_login():
    """Create Google OAuth login URL"""
    auth_params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': hashlib.sha256(os.urandom(32)).hexdigest()  # For security
    }
    
    auth_url = f"{GOOGLE_AUTHORIZATION_BASE_URL}?{urlencode(auth_params)}"
    st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <div style="display: inline-flex; align-items: center; 
                background-color: #4285F4; color: white; 
                padding: 8px 16px; border-radius: 4px; 
                cursor: pointer; text-decoration: none;">
                <img src="https://static.vecteezy.com/system/resources/previews/022/613/027/non_2x/google-icon-logo-symbol-free-png.png" 
                     style="height: 18px; margin-right: 8px;">
                Login with Google
            </div>
        </a>
    """, unsafe_allow_html=True)

def handle_google_callback():
    """Handle the OAuth callback from Google"""
    if "code" not in st.query_params:
        return
    
    if "code" in st.query_params:
        try:
            code = st.query_params["code"]
            token_data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            
            # Exchange auth code for token
            response = requests.post(GOOGLE_TOKEN_URL, data=token_data)
            response.raise_for_status()  # Raise exception for HTTP errors
            tokens = response.json()
            
            if "access_token" not in tokens:
                st.error("Failed to get access token from Google")
                print(f"Token response error: {tokens}")
                return
                
            # Get user info with access token
            user_info_response = requests.get(
                GOOGLE_USER_INFO_URL, 
                headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()
            
            if "email" not in user_info:
                st.error("Failed to get user email from Google")
                return
                
            # Set session state
            st.session_state['authenticated'] = True
            st.session_state['user_email'] = user_info['email']
            st.session_state['username'] = user_info.get('name', 'Google User')
            
            # Save to database if first time
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (user_info['email'],))
            if not cursor.fetchone():
                # Generate a random secure password for Google users
                random_password = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
                hashed_password = hash_password(random_password)
                
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, email, password) VALUES (?, ?, ?)",
                    (user_info.get('name', 'Google User'), user_info['email'], hashed_password)
                )
                conn.commit()
            conn.close()
            
            st.success(f"Logged in as {user_info.get('name', user_info['email'])}")
            st.rerun()
            
        except requests.RequestException as e:
            st.error(f"Google login request failed: {e}")
            print(f"OAuth error: {e}")
        except Exception as e:
            st.error(f"Google login failed: {e}")
            print(f"General error during Google auth: {e}")

# Login Page
def login_page():
    st.title("Bone Fracture Detection System")
    
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        st.header("Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login(login_email, login_password):
                st.session_state['authenticated'] = True
                st.session_state['user_email'] = login_email
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid email or password")

        google_login()
        handle_google_callback()

    with tab2:
        st.header("Signup")
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

        # OTP Section
        otp_col1, otp_col2 = st.columns([2, 1])
        
        with otp_col1:
            otp_entered = st.text_input("Enter OTP", key="otp_input")
        
        with otp_col2:
            if st.button("Send OTP"):
                if not new_email:
                    st.error("Please enter an email address")
                else:
                    # Generate OTP and store in session state with the email as key
                    otp = str(random.randint(100000, 999999))
                    st.session_state['otp_code'][new_email] = otp
                    
                    if send_otp_email(new_email, otp):
                        st.success(f"OTP sent to {new_email}!")
                        st.info(f"DEBUG - Your OTP is: {otp}")  # Remove in production
                    else:
                        st.error("Failed to send OTP")

            # In the "Signup" button click handler:
        if st.button("Signup"):
            if not new_username or not new_email or not new_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif not otp_entered:
                st.error("Please enter the OTP sent to your email")
            elif new_email not in st.session_state['otp_code']:
                st.error("Please request an OTP first")
            elif st.session_state['otp_code'].get(new_email) != otp_entered:
                st.error("Invalid OTP")
            else:
                # OTP is valid, proceed with user creation
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                try:
                    hashed_password = hash_password(new_password)
                    c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                            (new_username, new_email, hashed_password))
                    conn.commit()
                    # Clear the OTP after successful signup
                    del st.session_state['otp_code'][new_email]
                    st.success("Signup successful! Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username or email already exists")
                finally:
                    conn.close()


# Logout Button
def logout():
    st.session_state.clear()
    st.experimental_rerun()

def main():

    if 'otp_code' not in st.session_state:
        st.session_state['otp_code'] = {}

    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        login_page()
        return
    

    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.title("Advanced Bone Fracture Analysis System")
    
    st.sidebar.header("Analysis Settings")
    conf_thres = st.sidebar.slider("Detection confidence threshold", 0.2, 1.0, 0.3, step=0.05)
    device = st.sidebar.selectbox("Inference Device", ["cuda", "cpu"], 
                                help="Select CUDA for faster processing if you have a compatible GPU")
    
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

    if not os.path.exists("fracture_treatment.db"):
        create_database()
    
    if uploaded_file is not None:
        img = load_img(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, channels="RGB")
            bone_type = predict(uploaded_file, "Parts")
            st.info(f"**Body Part Classification:** {bone_type}")
        
        with col2:
            st.subheader("YOLOv7 Detection Results")
            img_pp = preproc(img)
            out = model_inference(model_onnx_path, img_pp, device)
            out_img, out_txt, found_fracture, detection_stats = post_process(img, out, conf_thres)
            st.image(out_img, channels="RGB")
            
            if found_fracture:
                st.error("🚨 **YOLOv7 Status:** Fracture Detected")
            else:
                st.success("✅ **YOLOv7 Status:** No Fracture Detected")
        
        with col3:
            st.subheader("DenseNet Detection Results")
            densenet_results, densenet_vis = densenet_inference(img, device)
            st.image(densenet_vis, channels="RGB")
            
            if densenet_results['fracture']['detected']:
                st.error("🚨 **DenseNet Status:** Fracture Detected")
            else:
                st.success("✅ **DenseNet Status:** No Fracture Detected")
        
        st.markdown("---")
        
        # Model Consensus Analysis
        st.markdown("### 🤖 Model Consensus Analysis")
        yolo_fracture = found_fracture
        densenet_fracture = densenet_results['fracture']['detected']
        
        yolo_confidence = detection_stats['fracture']['max_confidence']
        densenet_confidence = densenet_results['fracture']['confidence']
        
        def get_confidence_level(confidence):
            if confidence > 0.8:
                return "High"
            elif confidence > 0.6:
                return "Moderate"
            else:
                return "Low"
        
        if yolo_fracture and densenet_fracture:
            st.error("⚠️ Both detection models found fractures")
            st.markdown(f"""
            - YOLOv7 Confidence: **{get_confidence_level(yolo_confidence)}** ({yolo_confidence:.2%})
            - DenseNet Confidence: **{get_confidence_level(densenet_confidence)}** ({densenet_confidence:.2%})
            """)
        elif yolo_fracture != densenet_fracture:
            st.warning("⚠️ Models disagree on fracture detection - Further review recommended")
        else:
            st.success("✅ Both detection models agree: No fractures detected")
        
        # EfficientNet Pattern Analysis
        if yolo_fracture or densenet_fracture:
            st.markdown("### 🔬 EfficientNetV2 Severity Analysis")
            pattern_results = analyze_fracture_pattern(img, out, conf_thres, device)
            
            if pattern_results:
                vis_img = visualize_pattern(pattern_results, img.copy())
                st.image(vis_img, channels="RGB", caption="Pattern Analysis Visualization")
                
                # Display detailed analysis for each fracture
                for idx, result in enumerate(pattern_results, 1):
                    st.markdown(f"#### Fracture Region {idx}")
                    
                    # Image and basic info in one row
                    img_col, info_col = st.columns([1, 2])
                    
                    with img_col:
                        region_display = result['region']
                        st.image(region_display, caption=f"Region {idx}", use_column_width=True)
                    
                    with info_col:
                        pattern = result['pattern']
                        st.markdown(f"""
                        ##### Pattern Classification
                        - **Type**: {pattern['class']}
                        - **Confidence**: {pattern['confidence']:.2%}
                        """)
                        
                        # Create DataFrame for pattern probabilities
                        pattern_data = pd.DataFrame({
                            'Pattern': ['Simple', 'Wedge', 'Comminuted'],
                            'Probability': pattern['probabilities']
                        })
                        st.bar_chart(pattern_data.set_index('Pattern'))
                    
                    # Joint and Displacement info
                    st.markdown("##### Additional Analysis")
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        joint = result['joint_involvement']
                        st.markdown(f"""
                        **Joint Involvement**
                        - Status: {'Present' if joint['present'] else 'Not Present'}
                        - Confidence: {joint['confidence']:.2%}
                        """)
                    
                    with metrics_col2:
                        displacement = result['displacement']
                        st.markdown(f"""
                        **Displacement**
                        - Severity: {displacement['class']}
                        - Confidence: {displacement['confidence']:.2%}
                        """)
                    
                    # Clinical Assessment
                    st.markdown("##### Clinical Assessment")
                    if pattern['class'] == 'Simple':
                        st.success("🟢 Simple fracture: Single clean break")
                    elif pattern['class'] == 'Wedge':
                        st.warning("🟡 Wedge fracture: Angular component")
                    else:  # Comminuted
                        st.error("🔴 Comminuted fracture: Multiple fragments")
                    
                    if joint['present']:
                        st.warning("⚠️ Joint involvement detected")
                    
                    if displacement['class'] == 'Severe':
                        st.error("⚠️ Significant displacement noted")
                    
                    st.markdown("---")
            
                    try:
                        # Get classification results with safety checks
                        fracture_pattern = result.get('pattern', {}).get('class', 'Any')
                        displacement_severity = result.get('displacement', {}).get('class', 'Any')
                        joint_involvement = "Yes" if result.get('joint_involvement', {}).get('present', False) else "No"
                        
                        # Log the values for debugging
                        print(f"Looking up treatment for: Pattern={fracture_pattern}, Displacement={displacement_severity}, Joint={joint_involvement}")
                        
                        # Fetch treatment recommendation
                        treatment_recommendation = get_treatment_recommendation(
                            fracture_pattern=fracture_pattern,
                            displacement_severity=displacement_severity,
                            joint_involvement=joint_involvement
                        )
                        
                        st.markdown("### 🏥 **Treatment Recommendation**")
                        st.info(f"**Recommended Treatment:** {treatment_recommendation}")
                        
                        # Add additional details based on fracture type
                        if "ORIF" in treatment_recommendation:
                            st.warning("⚠️ **Note:** Surgical intervention required. Consult orthopedic specialist.")
                        elif "Conservative" in treatment_recommendation:
                            st.success("✅ **Note:** Conservative treatment appropriate. Regular follow-up recommended.")
                        
                    except Exception as e:
                        # Fallback recommendation if anything fails
                        st.markdown("### 🏥 **Treatment Recommendation**")
                        st.error(f"**Error retrieving recommendation:** {str(e)}")
                        st.info("**Recommended Treatment:** Consult with orthopedic specialist for personalized treatment plan.")
                        print(f"Error in treatment recommendation: {str(e)}")



if __name__ == "__main__":
    create_users_table()  # Ensure the users table exists
    main()