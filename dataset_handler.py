# # # dataset_handler.py

# # import os
# # import cv2
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # from pathlib import Path
# # import albumentations as A
# # from albumentations.pytorch import ToTensorV2
# # from tqdm import tqdm
# # import random
        
# # class FracturePatternDataset(Dataset):
# #     def __init__(self, img_dir, label_dir, mode='train', image_size=256):
# #         self.img_dir = Path(img_dir)
# #         self.label_dir = Path(label_dir)
# #         self.mode = mode
# #         self.image_size = image_size
# #         self.transform = self.get_default_transforms(mode)
        
# #         # Verify directories exist
# #         if not self.img_dir.exists():
# #             raise ValueError(f"Image directory does not exist: {self.img_dir}")
# #         if not self.label_dir.exists():
# #             raise ValueError(f"Label directory does not exist: {self.label_dir}")
            
# #         # Check for image files (prioritize PNG)
# #         self.image_files = list(self.img_dir.glob('*.png'))
# #         if not self.image_files:
# #             self.image_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.jpeg'))
            
# #         if not self.image_files:
# #             raise ValueError(f"No image files (PNG/JPG) found in {self.img_dir}")
            
# #         print(f"\nFound {len(self.image_files)} images in {self.img_dir}")
        
# #         # Pattern classification mapping
# #         self.pattern_mapping = {
# #             'simple': 0,
# #             'wedge': 1,
# #             'comminuted': 2
# #         }
        
# #         # Analyze dataset
# #         print(f"\nInitializing {mode} dataset...")
# #         self.analyzed_data = self._analyze_dataset()
        
# #         if not self.analyzed_data:
# #             raise ValueError("No valid fracture data found in the dataset")
        
# #         # Add balancing and sample crops
# #         if mode == 'train':
# #             self.balanced_indices = self._balance_dataset()
# #             # Save sample crops for verification
# #             self.save_sample_crops(num_samples=5)
# #         else:
# #             self.balanced_indices = list(range(len(self.analyzed_data)))
    
# #     def get_default_transforms(self, mode):
# #         if mode == 'train':
# #             return A.Compose([
# #                 A.Resize(self.image_size, self.image_size),
# #                 A.OneOf([
# #                     A.GaussNoise(p=0.8),
# #                     A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.8),
# #                 ], p=0.5),
# #                 A.Affine(
# #                     scale=(0.9, 1.1),
# #                     translate_percent=0.0625,
# #                     rotate=(-15, 15),
# #                     p=0.5
# #                 ),
# #                 A.OneOf([
# #                     A.OpticalDistortion(distort_limit=0.05, p=0.4),
# #                     A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
# #                     A.ElasticTransform(alpha=1, sigma=50, p=0.4),
# #                 ], p=0.3),
# #                 A.Normalize(mean=[0.485], std=[0.229]),
# #                 ToTensorV2(),
# #             ])
# #         else:
# #             return A.Compose([
# #                 A.Resize(self.image_size, self.image_size),
# #                 A.Normalize(mean=[0.485], std=[0.229]),
# #                 ToTensorV2(),
# #             ])
        
# #     def analyze_fracture_pattern(self, region):
# #         """
# #         Analyze fracture pattern using medically relevant criteria:
# #         - Simple: Single clean break with two main fragments
# #         - Wedge: Angular fracture with triangular fragment
# #         - Comminuted: Multiple (>3) distinct fragments with clear separation
# #         """
# #         if len(region.shape) == 3:
# #             gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
# #         else:
# #             gray = region
        
# #         # Enhanced preprocessing
# #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #         enhanced = clahe.apply(gray)
        
# #         # Multi-scale edge detection
# #         edges_fine = cv2.Canny(enhanced, 30, 100)
# #         edges_coarse = cv2.Canny(enhanced, 100, 200)
# #         edges = cv2.bitwise_or(edges_fine, edges_coarse)
        
# #         # Apply morphological operations to connect nearby edges
# #         kernel = np.ones((3,3), np.uint8)
# #         edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
# #         # Find contours with hierarchy to detect nested fragments
# #         contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
# #         # Filter out very small contours (noise)
# #         min_contour_area = gray.shape[0] * gray.shape[1] * 0.001
# #         valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
# #         if len(valid_contours) == 0:
# #             return 'simple', 0.5  # Default to simple if no clear fracture lines
        
# #         # Extract fracture line characteristics
# #         fracture_lines = []
# #         for contour in valid_contours:
# #             # Fit minimum area rectangle
# #             rect = cv2.minAreaRect(contour)
# #             _, (width, height), angle = rect
            
# #             if width > 0 and height > 0:
# #                 aspect_ratio = min(width, height) / max(width, height)
# #                 if aspect_ratio < 0.3:  # Long, thin shape indicating a fracture line
# #                     fracture_lines.append({
# #                         'angle': angle,
# #                         'length': max(width, height),
# #                         'width': min(width, height),
# #                         'aspect_ratio': aspect_ratio
# #                     })
        
# #         # Sort fracture lines by length
# #         fracture_lines.sort(key=lambda x: x['length'], reverse=True)
        
# #         # Count significant fragments
# #         significant_lines = [line for line in fracture_lines 
# #                             if line['length'] > min_contour_area ** 0.5]
# #         num_fragments = len(significant_lines)
        
# #         # Analyze angular relationships
# #         angles = [line['angle'] for line in significant_lines]
# #         angle_diffs = []
# #         for i in range(len(angles)):
# #             for j in range(i + 1, len(angles)):
# #                 diff = abs(angles[i] - angles[j])
# #                 angle_diffs.append(min(diff, 180 - diff))
        
# #         # Decision logic based on medical criteria
# #         if num_fragments <= 2:
# #             # Simple fracture - single clean break
# #             return 'simple', 0.9
# #         elif len(angle_diffs) > 0 and any(25 <= diff <= 65 for diff in angle_diffs):
# #             # Wedge fracture - characteristic angular pattern
# #             if num_fragments <= 3:
# #                 return 'wedge', 0.85
        
# #         # Only classify as comminuted if there's clear evidence
# #         if num_fragments >= 4 and len([c for c in valid_contours 
# #                                     if cv2.contourArea(c) > min_contour_area * 2]) >= 3:
# #             return 'comminuted', 0.8
        
# #         # Default to simple fracture if pattern is unclear
# #         return 'simple', 0.7
    
# #     def _process_yolo_annotation(self, annotation_line):
# #         """Process YOLO format annotation"""
# #         values = annotation_line.strip().split()
# #         class_id = int(values[0])
        
# #         if class_id == 3:  # Fracture class
# #             x_center = float(values[1])
# #             y_center = float(values[2])
# #             width = float(values[3])
# #             height = float(values[4])
# #             confidence = float(values[5]) if len(values) > 5 else 1.0
            
# #             return {
# #                 'bbox': (x_center, y_center, width, height),
# #                 'confidence': confidence
# #             }
# #         return None
    
# #     def _analyze_dataset(self):
# #         """Analyze all images and classify fracture patterns"""
# #         analyzed_data = []
        
# #         for img_file in tqdm(list(self.img_dir.glob('*.png')), 
# #                             desc=f"Analyzing {self.mode} dataset"):
# #             label_file = self.label_dir / f"{img_file.stem}.txt"
            
# #             if label_file.exists():
# #                 # Read the image
# #                 image = cv2.imread(str(img_file))
# #                 if image is None:
# #                     continue
                    
# #                 # Get annotations
# #                 with open(label_file, 'r') as f:
# #                     annotations = f.readlines()
                
# #                 fracture_regions = []
# #                 for ann in annotations:
# #                     fracture_info = self._process_yolo_annotation(ann)
# #                     if fracture_info:
# #                         H, W = image.shape[:2]
# #                         x, y, w, h = fracture_info['bbox']
                        
# #                         # Convert normalized coordinates to pixel values
# #                         x1 = int((x - w/2) * W)
# #                         y1 = int((y - h/2) * H)
# #                         x2 = int((x + w/2) * W)
# #                         y2 = int((y + h/2) * H)
                        
# #                         # Add padding
# #                         pad = 20
# #                         x1 = max(0, x1 - pad)
# #                         y1 = max(0, y1 - pad)
# #                         x2 = min(W, x2 + pad)
# #                         y2 = min(H, y2 + pad)
                        
# #                         region = image[y1:y2, x1:x2]
# #                         if region.size > 0:
# #                             pattern, conf = self.analyze_fracture_pattern(region)
# #                             joint_involved = random.random() < 0.3  # Placeholder
# #                             displacement = random.randint(0, 2)     # Placeholder
                            
# #                             fracture_regions.append({
# #                                 'pattern': pattern,
# #                                 'confidence': conf,
# #                                 'bbox': (x1, y1, x2, y2),
# #                                 'joint_involvement': joint_involved,
# #                                 'displacement': displacement
# #                             })
                
# #                 if fracture_regions:
# #                     # Take the most confident fracture pattern
# #                     best_region = max(fracture_regions, key=lambda x: x['confidence'])
# #                     analyzed_data.append({
# #                         'image_path': img_file,
# #                         'label_path': label_file,
# #                         'pattern': best_region['pattern'],
# #                         'confidence': best_region['confidence'],
# #                         'bbox': best_region['bbox'],
# #                         'joint_involvement': best_region['joint_involvement'],
# #                         'displacement': best_region['displacement']
# #                     })
        
# #         # Print statistics
# #         pattern_counts = {}
# #         for data in analyzed_data:
# #             pattern = data['pattern']
# #             pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
# #         print("\nFracture Pattern Distribution:")
# #         total = len(analyzed_data)
# #         for pattern, count in pattern_counts.items():
# #             percentage = (count / total) * 100
# #             print(f"{pattern.capitalize()}: {count} ({percentage:.1f}%)")
        
# #         return analyzed_data
    
# #     def _balance_dataset(self):
# #         """Balance dataset through oversampling"""
# #         pattern_counts = {}
# #         for data in self.analyzed_data:
# #             pattern = data['pattern']
# #             pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
# #         max_samples = max(pattern_counts.values())
# #         balanced_indices = []
        
# #         for pattern, count in pattern_counts.items():
# #             pattern_indices = [i for i, data in enumerate(self.analyzed_data) 
# #                              if data['pattern'] == pattern]
            
# #             if count < max_samples:
# #                 multiplier = max_samples // count
# #                 remainder = max_samples % count
# #                 balanced_indices.extend(pattern_indices * multiplier)
# #                 balanced_indices.extend(random.sample(pattern_indices, remainder))
# #             else:
# #                 balanced_indices.extend(pattern_indices)
        
# #         random.shuffle(balanced_indices)
# #         return balanced_indices
    
# #     def __len__(self):
# #         return len(self.balanced_indices)
    
# #     def __getitem__(self, idx):
# #         real_idx = self.balanced_indices[idx]
# #         data = self.analyzed_data[real_idx]
        
# #         # Load and process image
# #         image = cv2.imread(str(data['image_path']))
# #         if image is None:
# #             raise ValueError(f"Could not read image: {data['image_path']}")
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# #         # Apply transformations
# #         if self.transform:
# #             transformed = self.transform(image=image)
# #             image = transformed['image']
        
# #         return {
# #             'image': image,
# #             'pattern': torch.tensor(self.pattern_mapping[data['pattern']], dtype=torch.long),
# #             'joint_involvement': torch.tensor([data['joint_involvement']], dtype=torch.float32),
# #             'displacement': torch.tensor(data['displacement'], dtype=torch.long),
# #             'image_path': str(data['image_path'])
# #         }
    
# #     def save_sample_crops(self, num_samples=5):
# #         """Save sample cropped regions with analysis visualization"""
# #         print("\nSaving sample cropped regions for verification...")
        
# #         os.makedirs('sample_crops', exist_ok=True)
# #         samples_saved = 0
        
# #         # Try to get samples of each type
# #         pattern_types = set()
        
# #         for data in self.analyzed_data:
# #             if samples_saved >= num_samples and len(pattern_types) >= 3:
# #                 break
                
# #             try:
# #                 # Load original image
# #                 image = cv2.imread(str(data['image_path']))
# #                 if image is None:
# #                     continue
                
# #                 # Get bounding box and crop
# #                 x1, y1, x2, y2 = data['bbox']
# #                 cropped = image[y1:y2, x1:x2]
                
# #                 # Create visualization
# #                 pattern = data['pattern']
# #                 if pattern in pattern_types and len(pattern_types) < 3:
# #                     continue
                    
# #                 pattern_types.add(pattern)
                
# #                 # Convert to grayscale for analysis
# #                 gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
# #                 # Get edge analysis
# #                 clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #                 enhanced = clahe.apply(gray)
# #                 edges_fine = cv2.Canny(enhanced, 30, 100)
# #                 edges_coarse = cv2.Canny(enhanced, 100, 200)
# #                 edges = cv2.bitwise_or(edges_fine, edges_coarse)
                
# #                 # Create color visualization
# #                 vis_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
# #                 # Find and draw contours
# #                 contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #                 cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
                
# #                 # Add text for pattern type
# #                 text = f"Pattern: {pattern}"
# #                 cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
# #                 # Create side-by-side visualization
# #                 combined = np.hstack([cropped, vis_img])
                
# #                 # Save the visualization
# #                 output_path = f'sample_crops/sample_{pattern}_{samples_saved+1}.png'
# #                 cv2.imwrite(output_path, combined)
# #                 print(f"Saved {pattern} sample: {output_path}")
                
# #                 samples_saved += 1
                
# #             except Exception as e:
# #                 print(f"Error saving sample crop: {str(e)}")
# #                 continue
            
# #         print(f"Saved {samples_saved} sample crops with {len(pattern_types)} different patterns")


# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm

# class FracturePatternDataset(Dataset):
#     def __init__(self, img_dir, label_dir, mode='train', image_size=640):
#         self.img_dir = Path(img_dir)
#         self.label_dir = Path(label_dir)
#         self.mode = mode
#         self.image_size = image_size  # Set to 640 to match YOLO model
#         self.transform = self.get_default_transforms(mode)
        
#         # Check for image files
#         self.image_files = list(self.img_dir.glob('*.png'))
#         if not self.image_files:
#             raise ValueError(f"No PNG files found in {self.img_dir}")
            
#         print(f"\nFound {len(self.image_files)} images in {self.img_dir}")
#         self.processed_data = self._process_dataset()
    
#     def get_default_transforms(self, mode):
#         if mode == 'train':
#             return A.Compose([
#                 A.Resize(self.image_size, self.image_size),
#                 A.OneOf([
#                     A.RandomBrightnessContrast(p=0.8),
#                     A.RandomGamma(p=0.8),
#                 ], p=0.5),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])
#         else:
#             return A.Compose([
#                 A.Resize(self.image_size, self.image_size),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])

#     def _process_yolo_coordinates(self, x_center, y_center, width, height, original_h, original_w):
#         """Convert YOLO format to pixel coordinates with proper scaling"""
#         # Convert normalized to pixel coordinates
#         x_center_px = x_center * original_w
#         y_center_px = y_center * original_h
#         width_px = width * original_w
#         height_px = height * original_h
        
#         # Calculate corners
#         x1 = int(x_center_px - width_px/2)
#         y1 = int(y_center_px - height_px/2)
#         x2 = int(x_center_px + width_px/2)
#         y2 = int(y_center_px + height_px/2)
        
#         return x1, y1, x2, y2

#     def _process_dataset(self):
#         """Process dataset with proper coordinate scaling"""
#         processed_data = []
        
#         for img_file in tqdm(self.image_files, desc=f"Processing {self.mode} dataset"):
#             label_file = self.label_dir / f"{img_file.stem}.txt"
            
#             if not label_file.exists():
#                 continue
                
#             try:
#                 # Read original image for proper scaling
#                 original_img = cv2.imread(str(img_file))
#                 if original_img is None:
#                     continue
                
#                 # Get original dimensions
#                 original_h, original_w = original_img.shape[:2]
                
#                 # Convert to grayscale
#                 original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                
#                 # Resize image to model input size
#                 resized_img = cv2.resize(original_img, (self.image_size, self.image_size), 
#                                        interpolation=cv2.INTER_LINEAR)
                
#                 # Read YOLO format annotations
#                 with open(label_file, 'r') as f:
#                     for line in f:
#                         values = line.strip().split()
#                         if len(values) >= 5 and int(values[0]) == 3:  # Fracture class
#                             # Parse YOLO format
#                             x_center, y_center = float(values[1]), float(values[2])
#                             width, height = float(values[3]), float(values[4])
#                             confidence = float(values[5]) if len(values) >= 6 else 1.0
                            
#                             # Get coordinates in original image space
#                             x1, y1, x2, y2 = self._process_yolo_coordinates(
#                                 x_center, y_center, width, height, 
#                                 original_h, original_w
#                             )
                            
#                             # Add padding (scaled relative to image size)
#                             pad = int(min(original_h, original_w) * 0.05)  # 5% padding
#                             x1 = max(0, x1 - pad)
#                             y1 = max(0, y1 - pad)
#                             x2 = min(original_w, x2 + pad)
#                             y2 = min(original_h, y2 + pad)
                            
#                             # Get scaling factors
#                             scale_x = self.image_size / original_w
#                             scale_y = self.image_size / original_h
                            
#                             # Scale coordinates to resized image space
#                             x1_scaled = int(x1 * scale_x)
#                             y1_scaled = int(y1 * scale_y)
#                             x2_scaled = int(x2 * scale_x)
#                             y2_scaled = int(y2 * scale_y)
                            
#                             # Extract region from resized image
#                             region = resized_img[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                            
#                             if region.size == 0:
#                                 continue
                            
#                             # Save both original and scaled coordinates
#                             processed_data.append({
#                                 'image_path': img_file,
#                                 'original_bbox': (x1, y1, x2, y2),
#                                 'scaled_bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
#                                 'region': region,
#                                 'confidence': confidence
#                             })
                            
#                             # Save visualization of cropping if in train mode and it's the first few images
#                             if self.mode == 'train' and len(processed_data) <= 5:
#                                 vis_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
#                                 cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
#                                 # Save both original image with bbox and cropped region
#                                 out_dir = Path('sample_crops')
#                                 out_dir.mkdir(exist_ok=True)
                                
#                                 cv2.imwrite(
#                                     str(out_dir / f'sample_{len(processed_data)}_full.png'),
#                                     vis_img
#                                 )
#                                 cv2.imwrite(
#                                     str(out_dir / f'sample_{len(processed_data)}_crop.png'),
#                                     region
#                                 )
                            
#             except Exception as e:
#                 print(f"Error processing {img_file.name}: {str(e)}")
#                 continue
        
#         print(f"\nProcessed {len(processed_data)} fracture regions")
#         return processed_data
    
#     def __len__(self):
#         return len(self.processed_data)
    
#     def __getitem__(self, idx):
#         data = self.processed_data[idx]
#         region = data['region']
        
#         # Add channel dimension if needed
#         if len(region.shape) == 2:
#             region = np.expand_dims(region, -1)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=region)
#             region = transformed['image']
        
#         return {
#             'image': region,
#             'image_path': str(data['image_path']),
#             'original_bbox': data['original_bbox'],
#             'scaled_bbox': data['scaled_bbox'],
#             'confidence': data['confidence']
#         }



# # dataset_handler.py

# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm
# import random
        
# class FracturePatternDataset(Dataset):
#     def __init__(self, img_dir, label_dir, mode='train', image_size=256):
#         self.img_dir = Path(img_dir)
#         self.label_dir = Path(label_dir)
#         self.mode = mode
#         self.image_size = image_size
#         self.transform = self.get_default_transforms(mode)
        
#         # Verify directories exist
#         if not self.img_dir.exists():
#             raise ValueError(f"Image directory does not exist: {self.img_dir}")
#         if not self.label_dir.exists():
#             raise ValueError(f"Label directory does not exist: {self.label_dir}")
            
#         # Check for image files (prioritize PNG)
#         self.image_files = list(self.img_dir.glob('*.png'))
#         if not self.image_files:
#             self.image_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.jpeg'))
            
#         if not self.image_files:
#             raise ValueError(f"No image files (PNG/JPG) found in {self.img_dir}")
            
#         print(f"\nFound {len(self.image_files)} images in {self.img_dir}")
        
#         # Pattern classification mapping
#         self.pattern_mapping = {
#             'simple': 0,
#             'wedge': 1,
#             'comminuted': 2
#         }
        
#         # Analyze dataset
#         print(f"\nInitializing {mode} dataset...")
#         self.analyzed_data = self._analyze_dataset()
        
#         if not self.analyzed_data:
#             raise ValueError("No valid fracture data found in the dataset")
        
#         # Add balancing and sample crops
#         if mode == 'train':
#             self.balanced_indices = self._balance_dataset()
#             # Save sample crops for verification
#             self.save_sample_crops(num_samples=5)
#         else:
#             self.balanced_indices = list(range(len(self.analyzed_data)))
    
#     def get_default_transforms(self, mode):
#         if mode == 'train':
#             return A.Compose([
#                 A.Resize(self.image_size, self.image_size),
#                 A.OneOf([
#                     A.GaussNoise(p=0.8),
#                     A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.8),
#                 ], p=0.5),
#                 A.Affine(
#                     scale=(0.9, 1.1),
#                     translate_percent=0.0625,
#                     rotate=(-15, 15),
#                     p=0.5
#                 ),
#                 A.OneOf([
#                     A.OpticalDistortion(distort_limit=0.05, p=0.4),
#                     A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
#                     A.ElasticTransform(alpha=1, sigma=50, p=0.4),
#                 ], p=0.3),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])
#         else:
#             return A.Compose([
#                 A.Resize(self.image_size, self.image_size),
#                 A.Normalize(mean=[0.485], std=[0.229]),
#                 ToTensorV2(),
#             ])
        
#     def analyze_fracture_pattern(self, region):
#         """
#         Analyze fracture pattern using medically relevant criteria:
#         - Simple: Single clean break with two main fragments
#         - Wedge: Angular fracture with triangular fragment
#         - Comminuted: Multiple (>3) distinct fragments with clear separation
#         """
#         if len(region.shape) == 3:
#             gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = region
        
#         # Enhanced preprocessing
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
        
#         # Multi-scale edge detection
#         edges_fine = cv2.Canny(enhanced, 30, 100)
#         edges_coarse = cv2.Canny(enhanced, 100, 200)
#         edges = cv2.bitwise_or(edges_fine, edges_coarse)
        
#         # Apply morphological operations to connect nearby edges
#         kernel = np.ones((3,3), np.uint8)
#         edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
#         # Find contours with hierarchy to detect nested fragments
#         contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Filter out very small contours (noise)
#         min_contour_area = gray.shape[0] * gray.shape[1] * 0.001
#         valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
#         if len(valid_contours) == 0:
#             return 'simple', 0.5  # Default to simple if no clear fracture lines
        
#         # Extract fracture line characteristics
#         fracture_lines = []
#         for contour in valid_contours:
#             # Fit minimum area rectangle
#             rect = cv2.minAreaRect(contour)
#             _, (width, height), angle = rect
            
#             if width > 0 and height > 0:
#                 aspect_ratio = min(width, height) / max(width, height)
#                 if aspect_ratio < 0.3:  # Long, thin shape indicating a fracture line
#                     fracture_lines.append({
#                         'angle': angle,
#                         'length': max(width, height),
#                         'width': min(width, height),
#                         'aspect_ratio': aspect_ratio
#                     })
        
#         # Sort fracture lines by length
#         fracture_lines.sort(key=lambda x: x['length'], reverse=True)
        
#         # Count significant fragments
#         significant_lines = [line for line in fracture_lines 
#                             if line['length'] > min_contour_area ** 0.5]
#         num_fragments = len(significant_lines)
        
#         # Analyze angular relationships
#         angles = [line['angle'] for line in significant_lines]
#         angle_diffs = []
#         for i in range(len(angles)):
#             for j in range(i + 1, len(angles)):
#                 diff = abs(angles[i] - angles[j])
#                 angle_diffs.append(min(diff, 180 - diff))
        
#         # Decision logic based on medical criteria
#         if num_fragments <= 2:
#             # Simple fracture - single clean break
#             return 'simple', 0.9
#         elif len(angle_diffs) > 0 and any(25 <= diff <= 65 for diff in angle_diffs):
#             # Wedge fracture - characteristic angular pattern
#             if num_fragments <= 3:
#                 return 'wedge', 0.85
        
#         # Only classify as comminuted if there's clear evidence
#         if num_fragments >= 4 and len([c for c in valid_contours 
#                                     if cv2.contourArea(c) > min_contour_area * 2]) >= 3:
#             return 'comminuted', 0.8
        
#         # Default to simple fracture if pattern is unclear
#         return 'simple', 0.7
    
#     def _process_yolo_annotation(self, annotation_line):
#         """Process YOLO format annotation"""
#         values = annotation_line.strip().split()
#         class_id = int(values[0])
        
#         if class_id == 3:  # Fracture class
#             x_center = float(values[1])
#             y_center = float(values[2])
#             width = float(values[3])
#             height = float(values[4])
#             confidence = float(values[5]) if len(values) > 5 else 1.0
            
#             return {
#                 'bbox': (x_center, y_center, width, height),
#                 'confidence': confidence
#             }
#         return None
    
#     def _analyze_dataset(self):
#         """Analyze all images and classify fracture patterns"""
#         analyzed_data = []
        
#         for img_file in tqdm(list(self.img_dir.glob('*.png')), 
#                             desc=f"Analyzing {self.mode} dataset"):
#             label_file = self.label_dir / f"{img_file.stem}.txt"
            
#             if label_file.exists():
#                 # Read the image
#                 image = cv2.imread(str(img_file))
#                 if image is None:
#                     continue
                    
#                 # Get annotations
#                 with open(label_file, 'r') as f:
#                     annotations = f.readlines()
                
#                 fracture_regions = []
#                 for ann in annotations:
#                     fracture_info = self._process_yolo_annotation(ann)
#                     if fracture_info:
#                         H, W = image.shape[:2]
#                         x, y, w, h = fracture_info['bbox']
                        
#                         # Convert normalized coordinates to pixel values
#                         x1 = int((x - w/2) * W)
#                         y1 = int((y - h/2) * H)
#                         x2 = int((x + w/2) * W)
#                         y2 = int((y + h/2) * H)
                        
#                         # Add padding
#                         pad = 20
#                         x1 = max(0, x1 - pad)
#                         y1 = max(0, y1 - pad)
#                         x2 = min(W, x2 + pad)
#                         y2 = min(H, y2 + pad)
                        
#                         region = image[y1:y2, x1:x2]
#                         if region.size > 0:
#                             pattern, conf = self.analyze_fracture_pattern(region)
#                             joint_involved = random.random() < 0.3  # Placeholder
#                             displacement = random.randint(0, 2)     # Placeholder
                            
#                             fracture_regions.append({
#                                 'pattern': pattern,
#                                 'confidence': conf,
#                                 'bbox': (x1, y1, x2, y2),
#                                 'joint_involvement': joint_involved,
#                                 'displacement': displacement
#                             })
                
#                 if fracture_regions:
#                     # Take the most confident fracture pattern
#                     best_region = max(fracture_regions, key=lambda x: x['confidence'])
#                     analyzed_data.append({
#                         'image_path': img_file,
#                         'label_path': label_file,
#                         'pattern': best_region['pattern'],
#                         'confidence': best_region['confidence'],
#                         'bbox': best_region['bbox'],
#                         'joint_involvement': best_region['joint_involvement'],
#                         'displacement': best_region['displacement']
#                     })
        
#         # Print statistics
#         pattern_counts = {}
#         for data in analyzed_data:
#             pattern = data['pattern']
#             pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
#         print("\nFracture Pattern Distribution:")
#         total = len(analyzed_data)
#         for pattern, count in pattern_counts.items():
#             percentage = (count / total) * 100
#             print(f"{pattern.capitalize()}: {count} ({percentage:.1f}%)")
        
#         return analyzed_data
    
#     def _balance_dataset(self):
#         """Balance dataset through oversampling"""
#         pattern_counts = {}
#         for data in self.analyzed_data:
#             pattern = data['pattern']
#             pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
#         max_samples = max(pattern_counts.values())
#         balanced_indices = []
        
#         for pattern, count in pattern_counts.items():
#             pattern_indices = [i for i, data in enumerate(self.analyzed_data) 
#                              if data['pattern'] == pattern]
            
#             if count < max_samples:
#                 multiplier = max_samples // count
#                 remainder = max_samples % count
#                 balanced_indices.extend(pattern_indices * multiplier)
#                 balanced_indices.extend(random.sample(pattern_indices, remainder))
#             else:
#                 balanced_indices.extend(pattern_indices)
        
#         random.shuffle(balanced_indices)
#         return balanced_indices
    
#     def __len__(self):
#         return len(self.balanced_indices)
    
#     def __getitem__(self, idx):
#         real_idx = self.balanced_indices[idx]
#         data = self.analyzed_data[real_idx]
        
#         # Load and process image
#         image = cv2.imread(str(data['image_path']))
#         if image is None:
#             raise ValueError(f"Could not read image: {data['image_path']}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed['image']
        
#         return {
#             'image': image,
#             'pattern': torch.tensor(self.pattern_mapping[data['pattern']], dtype=torch.long),
#             'joint_involvement': torch.tensor([data['joint_involvement']], dtype=torch.float32),
#             'displacement': torch.tensor(data['displacement'], dtype=torch.long),
#             'image_path': str(data['image_path'])
#         }
    
#     def save_sample_crops(self, num_samples=5):
#         """Save sample cropped regions with analysis visualization"""
#         print("\nSaving sample cropped regions for verification...")
        
#         os.makedirs('sample_crops', exist_ok=True)
#         samples_saved = 0
        
#         # Try to get samples of each type
#         pattern_types = set()
        
#         for data in self.analyzed_data:
#             if samples_saved >= num_samples and len(pattern_types) >= 3:
#                 break
                
#             try:
#                 # Load original image
#                 image = cv2.imread(str(data['image_path']))
#                 if image is None:
#                     continue
                
#                 # Get bounding box and crop
#                 x1, y1, x2, y2 = data['bbox']
#                 cropped = image[y1:y2, x1:x2]
                
#                 # Create visualization
#                 pattern = data['pattern']
#                 if pattern in pattern_types and len(pattern_types) < 3:
#                     continue
                    
#                 pattern_types.add(pattern)
                
#                 # Convert to grayscale for analysis
#                 gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
#                 # Get edge analysis
#                 clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#                 enhanced = clahe.apply(gray)
#                 edges_fine = cv2.Canny(enhanced, 30, 100)
#                 edges_coarse = cv2.Canny(enhanced, 100, 200)
#                 edges = cv2.bitwise_or(edges_fine, edges_coarse)
                
#                 # Create color visualization
#                 vis_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
#                 # Find and draw contours
#                 contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                 cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
                
#                 # Add text for pattern type
#                 text = f"Pattern: {pattern}"
#                 cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
#                 # Create side-by-side visualization
#                 combined = np.hstack([cropped, vis_img])
                
#                 # Save the visualization
#                 output_path = f'sample_crops/sample_{pattern}_{samples_saved+1}.png'
#                 cv2.imwrite(output_path, combined)
#                 print(f"Saved {pattern} sample: {output_path}")
                
#                 samples_saved += 1
                
#             except Exception as e:
#                 print(f"Error saving sample crop: {str(e)}")
#                 continue
            
#         print(f"Saved {samples_saved} sample crops with {len(pattern_types)} different patterns")

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np

class FracturePatternDataset(Dataset):
    def __init__(self, img_dir, label_dir, mode='train', image_size=640):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        self.image_size = image_size
        self.transform = self.get_default_transforms(mode)
        
        # Pattern classification mapping
        self.pattern_mapping = {
            'simple': 0,     # Single clean break
            'wedge': 1,      # Angled fracture
            'comminuted': 2  # Multiple fragments
        }
        
        # Check for image files
        self.image_files = list(self.img_dir.glob('*.png'))
        if not self.image_files:
            raise ValueError(f"No PNG files found in {self.img_dir}")
            
        print(f"\nFound {len(self.image_files)} images in {self.img_dir}")
        self.processed_data = self._process_dataset()

    def _process_yolo_coordinates(self, x_center, y_center, width, height, original_h, original_w):
        """Convert YOLO format to pixel coordinates with proper scaling"""
        # Convert normalized to pixel coordinates
        x_center_px = x_center * original_w
        y_center_px = y_center * original_h
        width_px = width * original_w
        height_px = height * original_h
        
        # Calculate corners
        x1 = int(x_center_px - width_px/2)
        y1 = int(y_center_px - height_px/2)
        x2 = int(x_center_px + width_px/2)
        y2 = int(y_center_px + height_px/2)
        
        return x1, y1, x2, y2
    
    def get_default_transforms(self, mode):
        if mode == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.8
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.8),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(p=0.6),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.6),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.6),
                ], p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-10, 10),
                    shear=(-5, 5),
                    p=0.5
                ),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
    
    def _enhance_image(self, image):
        """Apply advanced image enhancement"""
        # Convert to float32
        image_float = image.astype(np.float32)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Enhance edges using unsharp masking
        gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)
        
        return unsharp_image

    def _analyze_pattern(self, region):
        """Analyze fracture pattern"""
        # Apply edge detection
        edges = cv2.Canny(region, 30, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        min_area = 50  # Minimum area threshold
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return 'simple', 0.8
        
        # Calculate characteristics
        num_fragments = len(valid_contours)
        
        # Get the main contour
        main_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calculate solidity (area ratio)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(main_contour) / hull_area if hull_area > 0 else 0
        
        # Get orientation if possible
        if len(main_contour) >= 5:
            (_, _), (major, minor), angle = cv2.fitEllipse(main_contour)
            aspect_ratio = major / minor if minor > 0 else 1
            angle = min(abs(angle), abs(180 - angle))
        else:
            aspect_ratio = 1
            angle = 0
        
        # Classification logic
        if num_fragments >= 3:
            pattern = 'comminuted'
            conf = min(0.9, num_fragments / 10)
        elif solidity < 0.85 or (angle > 30 and aspect_ratio > 1.5):
            pattern = 'wedge'
            conf = 0.85
        else:
            pattern = 'simple'
            conf = 0.9
        
        return pattern, conf

    def _detect_joint(self, region):
        """Detect joint involvement"""
        # Edge analysis near borders
        edges = cv2.Canny(region, 30, 150)
        
        height, width = edges.shape
        border = int(min(height, width) * 0.1)
        
        # Calculate edge density near borders
        border_regions = [
            edges[:border, :],             # top
            edges[-border:, :],            # bottom
            edges[:, :border],             # left
            edges[:, -border:]             # right
        ]
        
        densities = [np.mean(region) / 255 for region in border_regions]
        max_density = max(densities)
        
        return max_density > 0.2, max_density

    def _process_dataset(self):
        """Process dataset with pattern analysis"""
        processed_data = []
        
        for img_file in tqdm(self.image_files, desc=f"Processing {self.mode} dataset"):
            label_file = self.label_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                continue
                
            try:
                # Read original image
                original_img = cv2.imread(str(img_file))
                if original_img is None:
                    continue
                
                original_h, original_w = original_img.shape[:2]
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                
                # Enhance image
                enhanced_img = self._enhance_image(original_img)
                resized_img = cv2.resize(enhanced_img, (self.image_size, self.image_size))
                
                with open(label_file, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 5 and int(values[0]) == 3:  # Fracture class
                            # Parse YOLO format
                            x_center, y_center = float(values[1]), float(values[2])
                            width, height = float(values[3]), float(values[4])
                            confidence = float(values[5]) if len(values) >= 6 else 1.0
                            
                            # Get coordinates in original image space
                            x1, y1, x2, y2 = self._process_yolo_coordinates(
                                x_center, y_center, width, height, 
                                original_h, original_w
                            )
                            
                            # Add padding
                            pad = int(min(original_h, original_w) * 0.05)
                            x1 = max(0, x1 - pad)
                            y1 = max(0, y1 - pad)
                            x2 = min(original_w, x2 + pad)
                            y2 = min(original_h, y2 + pad)
                            
                            # Scale coordinates
                            scale_x = self.image_size / original_w
                            scale_y = self.image_size / original_h
                            
                            x1_scaled = int(x1 * scale_x)
                            y1_scaled = int(y1 * scale_y)
                            x2_scaled = int(x2 * scale_x)
                            y2_scaled = int(y2 * scale_y)
                            
                            # Extract region
                            region = resized_img[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                            
                            if region.size == 0:
                                continue
                            
                            # Analyze pattern and joint involvement
                            pattern, pattern_conf = self._analyze_pattern(region)
                            joint_involved, joint_conf = self._detect_joint(region)
                            
                            processed_data.append({
                                'image_path': img_file,
                                'original_bbox': (x1, y1, x2, y2),
                                'scaled_bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                                'region': region,
                                'confidence': confidence,
                                'pattern': self.pattern_mapping[pattern],
                                'pattern_confidence': pattern_conf,
                                'joint_involved': joint_involved,
                                'joint_confidence': joint_conf
                            })
                            
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
                continue
        
        print(f"\nProcessed {len(processed_data)} fracture regions")
        return processed_data
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        region = data['region']
        
        if len(region.shape) == 2:
            region = np.expand_dims(region, -1)
        
        if self.transform:
            transformed = self.transform(image=region)
            region = transformed['image']
        
        return {
            'image': region,
            'image_path': str(data['image_path']),
            'original_bbox': data['original_bbox'],
            'scaled_bbox': data['scaled_bbox'],
            'confidence': data['confidence'],
            'pattern': torch.tensor(data['pattern'], dtype=torch.long),
            'joint_involvement': torch.tensor([data['joint_involved']], dtype=torch.float32)
        }