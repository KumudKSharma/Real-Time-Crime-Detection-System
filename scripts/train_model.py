#!/usr/bin/env python3
"""
Crime Detection Model Training Pipeline
Downloads UCF-Crime dataset, preprocesses videos, and trains the model
"""

import os
import sys

# Add parent directory to path to import crime_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import urllib.request
import zipfile
import subprocess
from pathlib import Path
import argparse

# Import our crime detector
from src.crime_detector import SimpleCrimeDetector

class UCFCrimeDataset(Dataset):
    """
    PyTorch Dataset for UCF-Crime dataset
    Loads video clips and their corresponding labels
    """
    
    def __init__(self, data_dir, annotation_file, sequence_length=16, transform=None, mode='train'):
        """
        Args:
            data_dir: Directory containing video files
            annotation_file: File containing video labels
            sequence_length: Number of frames per sequence
            transform: Optional transform to apply to frames
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.mode = mode
        
        # Crime categories from UCF-Crime dataset
        self.crime_labels = [
            'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 
            'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        ]
        
        # Load annotations
        self.samples = self._load_annotations(annotation_file)
        print(f"Loaded {len(self.samples)} {mode} samples")
    
    def _load_annotations(self, annotation_file):
        """Load video annotations from file"""
        samples = []
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file {annotation_file} not found")
            return samples
            
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_path = parts[0]
                    label = int(parts[1])
                    
                    # Check if video file exists
                    full_path = self.data_dir / video_path
                    if full_path.exists():
                        samples.append((str(full_path), label))
                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        if len(frames) == 0:
            # Return dummy data if video can't be loaded
            frames = torch.zeros(3, self.sequence_length, 224, 224)
        else:
            # Convert to tensor format
            frames = self._preprocess_frames(frames)
        
        return frames, label
    
    def _load_video_frames(self, video_path):
        """Load frames from video file"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            if frame_count > self.sequence_length:
                indices = np.linspace(0, frame_count-1, self.sequence_length, dtype=int)
            else:
                indices = list(range(frame_count))
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Resize and convert to RGB
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            
        return frames
    
    def _preprocess_frames(self, frames):
        """Convert frames to tensor format"""
        # Pad or sample to exact sequence length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        if len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]
        
        # Convert to tensor: (C, T, H, W)
        frames_array = np.stack(frames)  # (T, H, W, C)
        frames_array = frames_array.transpose(3, 0, 1, 2)  # (C, T, H, W)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        
        return frames_tensor


class CrimeDetectionTrainer:
    """
    Trainer class for crime detection model
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, save_path='crime_model.pth'):
        """Full training loop"""
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_history': {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'train_accuracies': self.train_accuracies,
                        'val_accuracies': self.val_accuracies
                    }
                }, save_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history saved to {save_path}")


def download_ucf_crime_dataset(data_dir='./data'):
    """
    Download and extract UCF-Crime dataset
    Note: This is a large dataset (~6.8GB), ensure you have sufficient space
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    print("🔄 UCF-Crime Dataset Download")
    print("=" * 50)
    print("Note: UCF-Crime is a large dataset (~6.8GB)")
    print("This will take significant time depending on your internet connection")
    
    # UCF-Crime dataset URLs (these are example URLs - replace with actual links)
    dataset_info = {
        'videos_url': 'http://webpages.uncc.edu/cchen62/dataset.html',
        'annotations_url': 'https://github.com/WaqasSultani/AnomalyDetectionCVPR2018',
        'description': 'UCF-Crime Dataset contains 1900+ real-world surveillance videos'
    }
    
    print(f"Dataset Info:")
    print(f"- Description: {dataset_info['description']}")
    print(f"- Videos: {dataset_info['videos_url']}")
    print(f"- Annotations: {dataset_info['annotations_url']}")
    print()
    
    # Since direct download links change frequently, provide instructions
    instructions = """
    📥 MANUAL DOWNLOAD INSTRUCTIONS:
    
    1. Visit: http://webpages.uncc.edu/cchen62/dataset.html
    2. Download the UCF-Crime dataset
    3. Extract to: {data_dir}/UCF-Crime/
    4. Download annotations from: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
    5. Place annotation files in: {data_dir}/annotations/
    
    Expected structure:
    {data_dir}/
    ├── UCF-Crime/
    │   ├── Abuse/
    │   ├── Arrest/
    │   ├── Arson/
    │   └── ... (other crime categories)
    └── annotations/
        ├── train.txt
        ├── test.txt
        └── labels.txt
    """.format(data_dir=data_dir)
    
    print(instructions)
    
    # Create directory structure
    ucf_dir = data_dir / 'UCF-Crime'
    ucf_dir.mkdir(exist_ok=True)
    
    annotations_dir = data_dir / 'annotations'
    annotations_dir.mkdir(exist_ok=True)
    
    # Create sample annotation files if they don't exist
    create_sample_annotations(annotations_dir)
    
    return ucf_dir, annotations_dir


def create_sample_annotations(annotations_dir):
    """Create sample annotation files for testing"""
    
    # Sample train.txt
    train_file = annotations_dir / 'train.txt'
    if not train_file.exists():
        with open(train_file, 'w') as f:
            f.write("# Sample training annotations\n")
            f.write("# Format: video_path label_index\n")
            f.write("# Replace with actual video paths and labels\n")
            f.write("Normal/Normal_Videos001.mp4 0\n")
            f.write("Fighting/Fighting001.mp4 7\n")
            f.write("Robbery/Robbery001.mp4 9\n")
    
    # Sample test.txt  
    test_file = annotations_dir / 'test.txt'
    if not test_file.exists():
        with open(test_file, 'w') as f:
            f.write("# Sample test annotations\n")
            f.write("Normal/Normal_Videos002.mp4 0\n")
            f.write("Fighting/Fighting002.mp4 7\n")
    
    # Labels mapping
    labels_file = annotations_dir / 'labels.txt'
    if not labels_file.exists():
        crime_labels = [
            'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 
            'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        ]
        
        with open(labels_file, 'w') as f:
            for i, label in enumerate(crime_labels):
                f.write(f"{i} {label}\n")


def evaluate_model(model, test_loader, device, crime_labels):
    """Evaluate trained model and generate detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("🔍 Evaluating Model Performance")
    print("=" * 50)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=crime_labels, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=crime_labels, yticklabels=crime_labels)
    plt.title('Confusion Matrix - Crime Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, all_predictions, all_targets


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Crime Detection Model')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--model-path', default='crime_model.pth', help='Model save/load path')
    
    args = parser.parse_args()
    
    print("🎯 CRIME DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download dataset if requested
    if args.download:
        ucf_dir, annotations_dir = download_ucf_crime_dataset(args.data_dir)
        print("Dataset download instructions provided above ⬆️")
        return
    
    # Check if data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} not found!")
        print("Run with --download flag first to get dataset instructions")
        return
    
    # Crime labels
    crime_labels = [
        'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
        'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 
        'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
    ]
    
    # Initialize model
    model = SimpleCrimeDetector(num_classes=len(crime_labels))
    
    # Evaluate existing model if requested
    if args.evaluate:
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Load test dataset
            test_dataset = UCFCrimeDataset(
                data_dir / 'UCF-Crime',
                data_dir / 'annotations' / 'test.txt',
                mode='test'
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Evaluate
            accuracy, predictions, targets = evaluate_model(model, test_loader, device, crime_labels)
            print(f"✅ Model evaluation completed! Accuracy: {accuracy:.2%}")
        else:
            print(f"❌ Model file {args.model_path} not found!")
        return
    
    # Create datasets
    print("📊 Loading Datasets...")
    
    train_dataset = UCFCrimeDataset(
        data_dir / 'UCF-Crime',
        data_dir / 'annotations' / 'train.txt',
        mode='train'
    )
    
    val_dataset = UCFCrimeDataset(
        data_dir / 'UCF-Crime', 
        data_dir / 'annotations' / 'test.txt',
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        print("Please ensure you have downloaded the UCF-Crime dataset")
        print("Run with --download flag for instructions")
        return
    
    # Initialize trainer
    trainer = CrimeDetectionTrainer(model, device)
    
    # Train model
    print("\n🚀 Starting Training...")
    best_accuracy = trainer.train(train_loader, val_loader, 
                                args.epochs, args.model_path)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {args.model_path}")
    
    # Final evaluation
    if len(val_dataset) > 0:
        print("\n📊 Final Model Evaluation:")
        accuracy, predictions, targets = evaluate_model(model, val_loader, device, crime_labels)


if __name__ == "__main__":
    main() 