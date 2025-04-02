import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import glob
from tqdm import tqdm

class DatasetAnalyzer:
    def __init__(self):
        # Base paths (adjusted based on user input)
        self.detection_base = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\GRAZPEDWRI-DX_dataset")
        self.bodyparts_base = Path(r"C:\Users\HP\Downloads\SkeletalScan-main\SkeletalScan-main\Dataset")
        
        # Detection dataset class names
        self.detection_classes = [
            "boneanomaly", "bonelesion", "foreignbody", 
            "fracture", "metal", "periostealreaction", 
            "pronatorsign", "softtissue", "text"
        ]
        
        # Body parts classification
        self.body_parts = ["Elbow", "Hand", "Shoulder"]
        
        # Initialize stats dictionaries
        self.detection_stats = {}
        self.bodyparts_stats = {}
        
    def analyze_detection_dataset(self):
        """Analyze the fracture detection dataset (GRAZPEDWRI-DX)"""
        print("\n===== Analyzing GRAZPEDWRI-DX Dataset =====")
        
        yolo_dir = self.detection_base / "yolov5"
        
        if not yolo_dir.exists():
            print(f"Directory not found: {yolo_dir}")
            return None
            
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Image or label directories not found under {yolo_dir}")
            return None
            
        splits = ["train", "valid", "test"]
        image_counts = {}
        label_counts = {}
        
        for split in splits:
            img_dir = images_dir / split
            if img_dir.exists():
                image_counts[split] = len(list(img_dir.glob("*.png")))
            else:
                image_counts[split] = 0
                print(f"Warning: Image directory not found: {img_dir}")
            
            label_dir = labels_dir / split
            if label_dir.exists():
                label_counts[split] = len(list(label_dir.glob("*.txt")))
            else:
                label_counts[split] = 0
                print(f"Warning: Label directory not found: {label_dir}")
        
        total_images = sum(image_counts.values())
        total_labels = sum(label_counts.values())
        
        print(f"\nGRAZPEDWRI-DX Dataset Summary:")
        print(f"Total images: {total_images}")
        print(f"Total label files: {total_labels}")
        
        print("\nSplit distribution:")
        for split in splits:
            img_count = image_counts.get(split, 0)
            label_count = label_counts.get(split, 0)
            img_percent = (img_count / total_images) * 100 if total_images > 0 else 0
            
            print(f"  {split.capitalize()}: {img_count} images ({img_percent:.1f}%)")
            if img_count != label_count:
                print(f"  Warning: Image count ({img_count}) != Label count ({label_count}) for {split}")
        
        label_stats = self._analyze_detection_labels(labels_dir, splits)
        image_stats = self._analyze_detection_images(images_dir, splits)
        
        self.detection_stats = {
            'total_images': total_images,
            'total_labels': total_labels,
            'image_counts': image_counts,
            'label_counts': label_counts,
            'label_stats': label_stats,
            'image_stats': image_stats
        }
        
        return self.detection_stats
    
    def _analyze_detection_labels(self, labels_dir, splits):
        """Analyze label distribution in detection dataset"""
        print("\nAnalyzing class distribution in detection dataset...")
        
        all_label_files = []
        for split in splits:
            split_dir = labels_dir / split
            if split_dir.exists():
                all_label_files.extend(list(split_dir.glob("*.txt")))
        
        if not all_label_files:
            print("No label files found.")
            return {}
        
        class_counts = {i: 0 for i in range(len(self.detection_classes))}
        images_with_class = {i: 0 for i in range(len(self.detection_classes))}
        box_dimensions = {i: [] for i in range(len(self.detection_classes))}
        total_annotations = 0
        
        for label_file in tqdm(all_label_files, desc="Reading label files"):
            file_classes = set()
            
            with open(label_file, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            file_classes.add(class_id)
                            total_annotations += 1
                            
                            if len(values) >= 5:
                                width = float(values[3])
                                height = float(values[4])
                                box_dimensions[class_id].append((width, height))
            
            for class_id in file_classes:
                images_with_class[class_id] += 1
        
        box_stats = {}
        for class_id, dims in box_dimensions.items():
            if dims:
                widths, heights = zip(*dims)
                box_stats[class_id] = {
                    'avg_width': np.mean(widths),
                    'avg_height': np.mean(heights),
                    'avg_area': np.mean([w*h for w,h in dims]),
                    'count': len(dims)
                }
        
        print(f"\nTotal annotations: {total_annotations}")
        print(f"Total images: {len(all_label_files)}")
        print("\nClass distribution:")
        
        for class_id in range(len(self.detection_classes)):
            if class_id in class_counts:
                class_name = self.detection_classes[class_id]
                count = class_counts[class_id]
                img_count = images_with_class[class_id]
                percent = (count / total_annotations) * 100 if total_annotations > 0 else 0
                img_percent = (img_count / len(all_label_files)) * 100 if all_label_files else 0
                
                print(f"  Class {class_id} ({class_name}):")
                print(f"    - {count} instances ({percent:.2f}% of all annotations)")
                print(f"    - Found in {img_count} images ({img_percent:.2f}% of all images)")
                
                if class_id in box_stats:
                    bs = box_stats[class_id]
                    print(f"    - Avg box size: {bs['avg_width']*100:.1f}% × {bs['avg_height']*100:.1f}% of image")
                    print(f"    - Avg box area: {bs['avg_area']*100:.2f}% of image area")
        
        label_stats = {
            'class_counts': class_counts,
            'images_with_class': images_with_class,
            'total_annotations': total_annotations,
            'total_label_files': len(all_label_files),
            'box_stats': box_stats
        }
        
        return label_stats
    
    def _analyze_detection_images(self, images_dir, splits):
        """Analyze image properties in detection dataset"""
        print("\nAnalyzing image properties in detection dataset...")
        
        all_images = []
        for split in splits:
            split_dir = images_dir / split
            if split_dir.exists():
                split_images = list(split_dir.glob("*.png"))
                sample_size = min(50, len(split_images))
                if sample_size > 0:
                    sampled = np.random.choice(split_images, size=sample_size, replace=False)
                    all_images.extend(sampled)
        
        if not all_images:
            print("No image files found for analysis.")
            return {}
        
        widths, heights = [], []
        aspect_ratios = []
        channels = []
        file_sizes = []
        
        for img_path in tqdm(all_images, desc="Analyzing images"):
            try:
                file_sizes.append(os.path.getsize(img_path) / 1024)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                c = 1 if len(img.shape) == 2 else img.shape[2]
                channels.append(c)
                
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if not widths:
            print("Could not analyze any images.")
            return {}
            
        stats = {
            'width': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths)
            },
            'height': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights)
            },
            'aspect_ratio': {
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios)
            },
            'channels': Counter(channels),
            'file_size_kb': {
                'mean': np.mean(file_sizes),
                'std': np.std(file_sizes),
                'min': np.min(file_sizes),
                'max': np.max(file_sizes)
            },
            'sample_size': len(widths)
        }
        
        print(f"\nImage dimensions (from {len(widths)} samples):")
        print(f"  Width: mean = {stats['width']['mean']:.1f}px, std = {stats['width']['std']:.1f}px, range = [{stats['width']['min']:.0f}, {stats['width']['max']:.0f}]px")
        print(f"  Height: mean = {stats['height']['mean']:.1f}px, std = {stats['height']['std']:.1f}px, range = [{stats['height']['min']:.0f}, {stats['height']['max']:.0f}]px")
        print(f"  Aspect ratio: mean = {stats['aspect_ratio']['mean']:.3f}, std = {stats['aspect_ratio']['std']:.3f}")
        
        channel_info = ", ".join([f"{k} channels: {v} images" for k, v in stats['channels'].items()])
        print(f"  Channels: {channel_info}")
        
        print(f"  File size: mean = {stats['file_size_kb']['mean']:.1f}KB, std = {stats['file_size_kb']['std']:.1f}KB")
        print(f"  File size range: {stats['file_size_kb']['min']:.1f}KB - {stats['file_size_kb']['max']:.1f}KB")
        
        return stats
    
    def analyze_bodyparts_dataset(self):
        """Analyze the body parts classification dataset (MURA)"""
        print("\n===== Analyzing Body Parts Classification Dataset =====")
        
        if not self.bodyparts_base.exists():
            print(f"Body parts dataset directory not found: {self.bodyparts_base}")
            return None
        
        train_valid_dir = self.bodyparts_base / "train_valid"
        test_dir = self.bodyparts_base / "test"
        
        if not train_valid_dir.exists() and not test_dir.exists():
            train_valid_dir = self.bodyparts_base / "train"
            
        splits = ["train_valid", "test"]
        counts = {
            "train_valid": {part: {'positive': 0, 'negative': 0} for part in self.body_parts},
            "test": {part: {'positive': 0, 'negative': 0} for part in self.body_parts}
        }
        
        def process_split(split_dir, split_name):
            if not split_dir.exists():
                print(f"Split directory not found: {split_dir}")
                return 0
                
            total_images = 0
            
            for body_part in self.body_parts:
                part_dir = split_dir / body_part
                if not part_dir.exists():
                    print(f"Body part directory not found: {part_dir}")
                    continue
                    
                for patient_dir in part_dir.glob("patient*"):
                    for study_dir in patient_dir.glob("study1_*"):
                        status = 'positive' if 'positive' in study_dir.name else 'negative'
                        image_count = len(list(study_dir.glob("*.png")))
                        counts[split_name][body_part][status] += image_count
                        total_images += image_count
            
            return total_images
        
        train_valid_total = process_split(train_valid_dir, "train_valid")
        test_total = process_split(test_dir, "test")
        total_images = train_valid_total + test_total
        
        print(f"\nBody Parts Classification Dataset Summary:")
        print(f"Total images: {total_images}")
        print(f"Train/valid set: {train_valid_total} images ({train_valid_total/total_images*100:.1f}%)")
        print(f"Test set: {test_total} images ({test_total/total_images*100:.1f}%)")
        
        print("\nClass distribution:")
        for part in self.body_parts:
            train_pos = counts["train_valid"][part]["positive"]
            train_neg = counts["train_valid"][part]["negative"]
            test_pos = counts["test"][part]["positive"]
            test_neg = counts["test"][part]["negative"]
            
            total_part = train_pos + train_neg + test_pos + test_neg
            total_pos = train_pos + test_pos
            total_neg = train_neg + test_neg
            
            if total_part == 0:
                print(f"  {part}: No images found")
                continue
                
            percent = (total_part / total_images) * 100
            pos_percent = (total_pos / total_part) * 100
            neg_percent = (total_neg / total_part) * 100
            
            print(f"  {part}: {total_part} images ({percent:.1f}% of all images)")
            print(f"    - Positive (fractured): {total_pos} images ({pos_percent:.1f}%)")
            print(f"    - Negative (normal): {total_neg} images ({neg_percent:.1f}%)")
            print(f"    - Train/valid: {train_pos + train_neg} images")
            print(f"    - Test: {test_pos + test_neg} images")
        
        positive_total = sum(counts["train_valid"][part]["positive"] + counts["test"][part]["positive"] for part in self.body_parts)
        negative_total = sum(counts["train_valid"][part]["negative"] + counts["test"][part]["negative"] for part in self.body_parts)
        
        if total_images > 0:
            pos_percent = (positive_total / total_images) * 100
            neg_percent = (negative_total / total_images) * 100
            
            print(f"\nOverall dataset balance:")
            print(f"  Positive (fractured): {positive_total} images ({pos_percent:.1f}%)")
            print(f"  Negative (normal): {negative_total} images ({neg_percent:.1f}%)")
        
        self._analyze_bodyparts_images(train_valid_dir, test_dir)
        
        self.bodyparts_stats = {
            'total_images': total_images,
            'train_valid_images': train_valid_total,
            'test_images': test_total,
            'counts': counts,
            'positive_total': positive_total,
            'negative_total': negative_total
        }
        
        return self.bodyparts_stats
    
    def _analyze_bodyparts_images(self, train_valid_dir, test_dir):
        """Analyze image properties in body parts dataset"""
        print("\nAnalyzing image properties in body parts dataset...")
        
        def collect_images(base_dir, max_per_part=30):
            if not base_dir.exists():
                return []
                
            samples = []
            for part in self.body_parts:
                part_dir = base_dir / part
                if not part_dir.exists():
                    continue
                    
                part_images = []
                for patient_dir in part_dir.glob("patient*"):
                    for study_dir in patient_dir.glob("study1_*"):
                        part_images.extend(list(study_dir.glob("*.png")))
                
                if len(part_images) > max_per_part:
                    samples.extend(np.random.choice(part_images, size=max_per_part, replace=False))
                else:
                    samples.extend(part_images)
            
            return samples
        
        train_valid_samples = collect_images(train_valid_dir)
        test_samples = collect_images(test_dir)
        all_images = train_valid_samples + test_samples
        
        if not all_images:
            print("No images found for analysis.")
            return {}
        
        widths, heights = [], []
        aspect_ratios = []
        channels = []
        file_sizes = []
        
        for img_path in tqdm(all_images, desc="Analyzing images"):
            try:
                file_sizes.append(os.path.getsize(img_path) / 1024)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                c = 1 if len(img.shape) == 2 else img.shape[2]
                channels.append(c)
                
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if not widths:
            print("Could not analyze any images.")
            return {}
        
        stats = {
            'width': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths)
            },
            'height': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights)
            },
            'aspect_ratio': {
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios)
            },
            'channels': Counter(channels),
            'file_size_kb': {
                'mean': np.mean(file_sizes),
                'std': np.std(file_sizes),
                'min': np.min(file_sizes),
                'max': np.max(file_sizes)
            },
            'sample_size': len(widths)
        }
        
        print(f"\nImage dimensions (from {len(widths)} samples):")
        print(f"  Width: mean = {stats['width']['mean']:.1f}px, std = {stats['width']['std']:.1f}px, range = [{stats['width']['min']:.0f}, {stats['width']['max']:.0f}]px")
        print(f"  Height: mean = {stats['height']['mean']:.1f}px, std = {stats['height']['std']:.1f}px, range = [{stats['height']['min']:.0f}, {stats['height']['max']:.0f}]px")
        print(f"  Aspect ratio: mean = {stats['aspect_ratio']['mean']:.3f}, std = {stats['aspect_ratio']['std']:.3f}")
        
        channel_info = ", ".join([f"{k} channels: {v} images" for k, v in stats['channels'].items()])
        print(f"  Channels: {channel_info}")
        
        print(f"  File size: mean = {stats['file_size_kb']['mean']:.1f}KB, std = {stats['file_size_kb']['std']:.1f}KB")
        print(f"  File size range: {stats['file_size_kb']['min']:.1f}KB - {stats['file_size_kb']['max']:.1f}KB")
        
        self.bodyparts_stats['image_stats'] = stats
        
        return stats
    
    def plot_detection_stats(self):
        """Plot statistics for the detection dataset"""
        if not self.detection_stats or 'label_stats' not in self.detection_stats:
            print("No detection stats available to plot.")
            return
            
        try:
            plots_dir = Path("dataset_analysis_plots")
            plots_dir.mkdir(exist_ok=True)
            
            label_stats = self.detection_stats['label_stats']
            class_counts = label_stats.get('class_counts', {})
            img_counts = label_stats.get('images_with_class', {})
            
            if class_counts:
                plt.figure(figsize=(14, 8))
                class_ids = sorted(class_counts.keys())
                counts = [class_counts[i] for i in class_ids]
                names = [self.detection_classes[i] for i in class_ids]
                
                sorted_indices = np.argsort(counts)[::-1]
                sorted_counts = [counts[i] for i in sorted_indices]
                sorted_names = [names[i] for i in sorted_indices]
                
                plt.bar(sorted_names, sorted_counts)
                plt.title('GRAZPEDWRI-DX Dataset - Class Distribution', fontsize=15)
                plt.xlabel('Class', fontsize=12)
                plt.ylabel('Number of Annotations', fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                
                for i, v in enumerate(sorted_counts):
                    plt.text(i, v + 0.5, str(v), ha='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'grazpedwri_class_distribution.png', dpi=300)
                plt.close()
                print(f"Saved class distribution plot to 'dataset_analysis_plots/grazpedwri_class_distribution.png'")
                
                if img_counts:
                    plt.figure(figsize=(14, 8))
                    img_counts_list = [img_counts[i] for i in class_ids]
                    sorted_img_counts = [img_counts_list[i] for i in sorted_indices]
                    
                    plt.bar(sorted_names, sorted_img_counts)
                    plt.title('GRAZPEDWRI-DX Dataset - Images per Class', fontsize=15)
                    plt.xlabel('Class', fontsize=12)
                    plt.ylabel('Number of Images', fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    
                    for i, v in enumerate(sorted_img_counts):
                        plt.text(i, v + 0.5, str(v), ha='center', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'grazpedwri_images_per_class.png', dpi=300)
                    plt.close()
                    print(f"Saved images per class plot to 'dataset_analysis_plots/grazpedwri_images_per_class.png'")
                
                splits = ['train', 'valid', 'test']
                if 'image_counts' in self.detection_stats:
                    plt.figure(figsize=(10, 6))
                    split_counts = [self.detection_stats['image_counts'].get(split, 0) for split in splits]
                    
                    plt.bar(splits, split_counts)
                    plt.title('GRAZPEDWRI-DX Dataset - Split Distribution', fontsize=15)
                    plt.xlabel('Split', fontsize=12)
                    plt.ylabel('Number of Images', fontsize=12)
                    
                    total = sum(split_counts)
                    for i, v in enumerate(split_counts):
                        percentage = (v / total) * 100 if total > 0 else 0
                        plt.text(i, v + 5, f"{v}\n({percentage:.1f}%)", ha='center', fontsize=10)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'grazpedwri_split_distribution.png', dpi=300)
                    plt.close()
                    print(f"Saved split distribution plot to 'dataset_analysis_plots/grazpedwri_split_distribution.png'")
            
            if 'label_stats' in self.detection_stats and 'box_stats' in self.detection_stats['label_stats']:
                box_stats = self.detection_stats['label_stats']['box_stats']
                class_ids = sorted(box_stats.keys())
                
                if class_ids:
                    plt.figure(figsize=(12, 8))
                    areas = [box_stats[i]['avg_area'] * 100 for i in class_ids]
                    names = [self.detection_classes[i] for i in class_ids]
                    
                    sorted_indices = np.argsort(areas)[::-1]
                    sorted_areas = [areas[i] for i in sorted_indices]
                    sorted_names = [names[i] for i in sorted_indices]
                    
                    plt.bar(sorted_names, sorted_areas)
                    plt.title('GRAZPEDWRI-DX Dataset - Average Box Size by Class', fontsize=15)
                    plt.xlabel('Class', fontsize=12)
                    plt.ylabel('Average Box Area (% of image)', fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    
                    for i, v in enumerate(sorted_areas):
                        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'grazpedwri_box_sizes.png', dpi=300)
                    plt.close()
                    print(f"Saved box size plot to 'dataset_analysis_plots/grazpedwri_box_sizes.png'")
        except Exception as e:
            print(f"Error creating detection plots: {e}")

    def plot_bodyparts_stats(self):
        """Plot statistics for the body parts classification dataset"""
        if not self.bodyparts_stats or 'counts' not in self.bodyparts_stats:
            print("No body parts stats available to plot.")
            return
            
        try:
            plots_dir = Path("dataset_analysis_plots")
            plots_dir.mkdir(exist_ok=True)
            
            counts = self.bodyparts_stats['counts']
            
            # Plot 1: Distribution of images per body part
            plt.figure(figsize=(12, 8))
            total_per_part = [
                sum(counts['train_valid'][part].values()) + sum(counts['test'][part].values())
                for part in self.body_parts
            ]
            plt.bar(self.body_parts, total_per_part)
            plt.title('Body Parts Dataset - Images per Body Part', fontsize=15)
            plt.xlabel('Body Part', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            
            for i, v in enumerate(total_per_part):
                plt.text(i, v + 5, str(v), ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'bodyparts_distribution.png', dpi=300)
            plt.close()
            print(f"Saved body parts distribution plot to 'dataset_analysis_plots/bodyparts_distribution.png'")
            
            # Plot 2: Positive vs Negative per body part
            plt.figure(figsize=(12, 8))
            x = np.arange(len(self.body_parts))
            width = 0.35
            
            pos_counts = [
                counts['train_valid'][part]['positive'] + counts['test'][part]['positive']
                for part in self.body_parts
            ]
            neg_counts = [
                counts['train_valid'][part]['negative'] + counts['test'][part]['negative']
                for part in self.body_parts
            ]
            
            plt.bar(x - width/2, pos_counts, width, label='Positive (Fractured)')
            plt.bar(x + width/2, neg_counts, width, label='Negative (Normal)')
            
            plt.title('Body Parts Dataset - Positive vs Negative per Body Part', fontsize=15)
            plt.xlabel('Body Part', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            plt.xticks(x, self.body_parts)
            plt.legend()
            
            for i in range(len(self.body_parts)):
                plt.text(i - width/2, pos_counts[i] + 5, str(pos_counts[i]), ha='center', fontsize=10)
                plt.text(i + width/2, neg_counts[i] + 5, str(neg_counts[i]), ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'bodyparts_pos_neg_distribution.png', dpi=300)
            plt.close()
            print(f"Saved positive/negative distribution plot to 'dataset_analysis_plots/bodyparts_pos_neg_distribution.png'")
            
            # Plot 3: Split distribution
            plt.figure(figsize=(10, 6))
            splits = ['train_valid', 'test']
            split_counts = [
                self.bodyparts_stats['train_valid_images'],
                self.bodyparts_stats['test_images']
            ]
            
            plt.bar(splits, split_counts)
            plt.title('Body Parts Dataset - Split Distribution', fontsize=15)
            plt.xlabel('Split', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            
            total = sum(split_counts)
            for i, v in enumerate(split_counts):
                percentage = (v / total) * 100 if total > 0 else 0
                plt.text(i, v + 5, f"{v}\n({percentage:.1f}%)", ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'bodyparts_split_distribution.png', dpi=300)
            plt.close()
            print(f"Saved split distribution plot to 'dataset_analysis_plots/bodyparts_split_distribution.png'")
            
        except Exception as e:
            print(f"Error creating body parts plots: {e}")

    def run_analysis(self):
        """Run full analysis on both datasets and generate plots"""
        print("Starting full dataset analysis...")
        
        # Analyze detection dataset
        self.analyze_detection_dataset()
        self.plot_detection_stats()
        
        # Analyze body parts dataset
        self.analyze_bodyparts_dataset()
        self.plot_bodyparts_stats()
        
        print("\nAnalysis completed!")

# Example usage
if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.run_analysis()