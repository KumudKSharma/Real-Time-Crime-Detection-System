#!/usr/bin/env python3
"""
Generate Synthetic Training Data for Crime Detection
Creates artificial video data for demonstration and testing purposes
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

def create_synthetic_video(video_path, crime_type, duration_seconds=5, fps=20):
    """
    Create a synthetic video with patterns representing different crime types
    """
    frames = []
    total_frames = duration_seconds * fps
    
    # Base scene (static background)
    height, width = 224, 224
    background = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    for frame_idx in range(total_frames):
        frame = background.copy()
        
        # Add crime-specific patterns
        if crime_type == 'Normal':
            # Minimal movement - just slight variations
            noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif crime_type == 'Fighting':
            # Rapid movements, multiple moving objects
            for _ in range(3):
                x = np.random.randint(0, width-30)
                y = np.random.randint(0, height-30)
                color = (np.random.randint(100, 255), 0, 0)  # Red-ish
                cv2.rectangle(frame, (x, y), (x+30, y+30), color, -1)
            
            # Add motion blur effect
            kernel = np.ones((5, 5), np.float32) / 25
            frame = cv2.filter2D(frame, -1, kernel)
            
        elif crime_type == 'Robbery':
            # Quick movements, object transfer
            # Person 1 (victim)
            cv2.circle(frame, (50, height//2), 20, (0, 255, 0), -1)
            
            # Person 2 (robber) - moving quickly
            robber_x = int(150 + 30 * np.sin(frame_idx * 0.5))
            cv2.circle(frame, (robber_x, height//2), 20, (0, 0, 255), -1)
            
            # "Object" being taken
            if frame_idx > total_frames // 2:
                obj_x = robber_x
            else:
                obj_x = 50
            cv2.rectangle(frame, (obj_x-5, height//2-30), (obj_x+5, height//2-20), (255, 255, 0), -1)
            
        elif crime_type == 'Vandalism':
            # Destructive patterns, spray-like effects
            for _ in range(10):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                cv2.circle(frame, (x, y), np.random.randint(2, 8), 
                         (np.random.randint(200, 255), 0, np.random.randint(0, 100)), -1)
                         
        elif crime_type == 'Shooting':
            # Flash effects, rapid changes
            if frame_idx % 10 < 3:  # Flash every 10 frames for 3 frames
                frame = np.full_like(frame, 255)  # White flash
            
            # Add "bullet trails"
            cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 0), 2)
            
        elif crime_type == 'Shoplifting':
            # Subtle movements, concealment actions
            # Person near shelf
            cv2.rectangle(frame, (10, 10), (40, height-10), (139, 69, 19), -1)  # Shelf
            
            # Person moving items
            person_x = int(100 + 20 * np.sin(frame_idx * 0.2))
            cv2.circle(frame, (person_x, height//2), 15, (100, 100, 200), -1)
            
            # Item movement (subtle)
            if frame_idx % 30 < 5:  # Occasional item movement
                cv2.rectangle(frame, (person_x-10, height//2+20), (person_x-5, height//2+25), (255, 0, 255), -1)
                
        elif crime_type == 'Arson':
            # Fire-like effects, growing patterns
            fire_intensity = max(100, min(255, 100 + frame_idx * 3))  # Ensure minimum value
            for _ in range(20):
                x = np.random.randint(width//4, 3*width//4)
                y = np.random.randint(height//2, height)
                size = np.random.randint(5, 15)
                color = (0, np.random.randint(50, fire_intensity), fire_intensity)
                cv2.circle(frame, (x, y), size, color, -1)
                
        elif crime_type == 'Assault':
            # Aggressive movements between people
            # Person 1
            p1_x, p1_y = 80, height//2
            cv2.circle(frame, (p1_x, p1_y), 15, (0, 255, 0), -1)
            
            # Person 2 (aggressive)
            p2_x = int(120 + 15 * np.sin(frame_idx * 0.8))
            p2_y = int(height//2 + 10 * np.cos(frame_idx * 0.8))
            cv2.circle(frame, (p2_x, p2_y), 15, (0, 0, 255), -1)
            
            # Impact effects
            if frame_idx % 15 < 3:
                cv2.circle(frame, ((p1_x + p2_x)//2, (p1_y + p2_y)//2), 25, (255, 255, 255), 2)
        
        else:  # Default for other crimes
            # Generic suspicious activity
            for _ in range(2):
                x = np.random.randint(0, width-20)
                y = np.random.randint(0, height-20)
                cv2.rectangle(frame, (x, y), (x+20, y+20), (128, 128, 128), 2)
        
        frames.append(frame)
    
    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    return len(frames)

def generate_synthetic_dataset(output_dir='./synthetic_data', num_videos_per_class=10):
    """
    Generate a complete synthetic dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Crime categories
    crime_types = [
        'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
        'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 
        'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
    ]
    
    print(f"🎬 Generating Synthetic Crime Dataset")
    print(f"Output directory: {output_dir}")
    print(f"Videos per class: {num_videos_per_class}")
    print(f"Total videos: {len(crime_types) * num_videos_per_class}")
    print("=" * 50)
    
    # Create directories and videos
    train_annotations = []
    test_annotations = []
    
    for class_idx, crime_type in enumerate(tqdm(crime_types, desc="Crime Types")):
        class_dir = output_dir / crime_type
        class_dir.mkdir(exist_ok=True)
        
        for video_idx in tqdm(range(num_videos_per_class), desc=f"  {crime_type}", leave=False):
            video_name = f"{crime_type}_{video_idx:03d}.mp4"
            video_path = class_dir / video_name
            
            # Create video
            num_frames = create_synthetic_video(video_path, crime_type)
            
            # Add to annotations (80% train, 20% test)
            relative_path = f"{crime_type}/{video_name}"
            annotation = f"{relative_path} {class_idx}"
            
            if video_idx < int(0.8 * num_videos_per_class):
                train_annotations.append(annotation)
            else:
                test_annotations.append(annotation)
    
    # Save annotation files
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir(exist_ok=True)
    
    with open(annotations_dir / 'train.txt', 'w') as f:
        f.write("# Synthetic training annotations\n")
        f.write("# Format: video_path label_index\n")
        for annotation in train_annotations:
            f.write(annotation + "\n")
    
    with open(annotations_dir / 'test.txt', 'w') as f:
        f.write("# Synthetic test annotations\n")
        for annotation in test_annotations:
            f.write(annotation + "\n")
    
    # Save labels mapping
    with open(annotations_dir / 'labels.txt', 'w') as f:
        for i, label in enumerate(crime_types):
            f.write(f"{i} {label}\n")
    
    # Save dataset info
    dataset_info = {
        'name': 'Synthetic Crime Dataset',
        'classes': len(crime_types),
        'videos_per_class': num_videos_per_class,
        'total_videos': len(crime_types) * num_videos_per_class,
        'train_videos': len(train_annotations),
        'test_videos': len(test_annotations),
        'crime_types': crime_types,
        'video_specs': {
            'resolution': '224x224',
            'fps': 20,
            'duration': '5 seconds',
            'format': 'mp4'
        }
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Synthetic dataset generation completed!")
    print(f"📂 Dataset saved to: {output_dir}")
    print(f"📊 Training videos: {len(train_annotations)}")
    print(f"📊 Test videos: {len(test_annotations)}")
    print(f"📋 Annotation files: {annotations_dir}")
    
    return output_dir

def preview_synthetic_data(data_dir='./synthetic_data'):
    """
    Preview the generated synthetic data
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} not found!")
        return
    
    print(f"🔍 Previewing Synthetic Dataset: {data_dir}")
    print("=" * 50)
    
    # Load dataset info
    info_file = data_dir / 'dataset_info.json'
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print(f"Dataset: {info['name']}")
        print(f"Classes: {info['classes']}")
        print(f"Total Videos: {info['total_videos']}")
        print(f"Video Format: {info['video_specs']['resolution']} @ {info['video_specs']['fps']}fps")
        print()
    
    # Show directory structure
    print("📁 Directory Structure:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and item.name != 'annotations':
            video_count = len(list(item.glob('*.mp4')))
            print(f"  📂 {item.name}: {video_count} videos")
    
    # Show sample annotation
    annotations_dir = data_dir / 'annotations'
    if annotations_dir.exists():
        print(f"\n📋 Sample Annotations:")
        train_file = annotations_dir / 'train.txt'
        if train_file.exists():
            with open(train_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[2:7]):  # Skip header, show 5 samples
                    if line.strip():
                        print(f"  {line.strip()}")
    
    print("\n✅ Dataset preview completed!")

def main():
    """Main function for synthetic data generation"""
    parser = argparse.ArgumentParser(description='Generate Synthetic Crime Detection Dataset')
    parser.add_argument('--output-dir', default='./synthetic_data', 
                       help='Output directory for synthetic dataset')
    parser.add_argument('--num-videos', type=int, default=10,
                       help='Number of videos per crime class')
    parser.add_argument('--preview', action='store_true',
                       help='Preview existing synthetic dataset')
    
    args = parser.parse_args()
    
    if args.preview:
        preview_synthetic_data(args.output_dir)
    else:
        output_dir = generate_synthetic_dataset(args.output_dir, args.num_videos)
        print(f"\n🎯 Next Steps:")
        print(f"1. Preview dataset: python3 generate_synthetic_data.py --preview")
        print(f"2. Train model: python3 train_model.py --data-dir {output_dir}")
        print(f"3. Evaluate model: python3 train_model.py --evaluate --data-dir {output_dir}")

if __name__ == "__main__":
    main()