# Models Directory

This directory will contain your trained model files after training.

## 🚀 Quick Start - Train Your First Model

### Option 1: Quick Demo (5 minutes)
```bash
# Train a demo model with synthetic data
python scripts/minimal_training_demo.py
```
- **Accuracy**: ~15-25% (synthetic data only)
- **Purpose**: Testing and demonstration
- **Output**: `models/crime_model.pth`

### Option 2: Professional Training (2-4 hours)
```bash
# Train with real UCF-Crime dataset
python scripts/train_model.py --dataset ucf_crime --epochs 50
```
- **Accuracy**: 70-85% (real surveillance data)
- **Purpose**: Production deployment
- **Dataset**: UCF-Crime (~6.8GB download)
- **Output**: `models/crime_model.pth`

## 📊 Expected Model Performance

| Training Method | Accuracy | Use Case |
|----------------|----------|----------|
| Synthetic Data | 15-25% | Demo/Testing |
| UCF-Crime Basic | 65-75% | Development |
| UCF-Crime Optimized | 80-90% | Production |

## 🔧 Model Requirements

- **File Format**: PyTorch `.pth` format
- **Architecture**: SimpleCrimeDetector (3D CNN)
- **Input**: 16 frames × 112×112 × 3 channels
- **Output**: 14 crime categories
- **Size**: ~18MB typical

## 📝 Training Notes

1. **No Pre-trained Model**: This repository doesn't include pre-trained weights
2. **Train Before Use**: You must train a model before using the GUI
3. **Custom Data**: You can train on your own datasets
4. **Model Persistence**: Trained models are automatically saved here

## 🆘 Troubleshooting

**Error: "Model file not found"**
- Solution: Run training script first to generate model

**Low Accuracy (<30%)**
- Use UCF-Crime dataset instead of synthetic data
- Increase training epochs
- Apply data augmentation

For detailed training instructions, see the main README.md file.