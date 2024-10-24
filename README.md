# Diabetic_Retinopathy

# APTOS 2019 Blindness Detection - Deep Learning Model

## Overview
This repository contains a deep learning solution for the APTOS 2019 Blindness Detection challenge. The model uses an ensemble approach combining EfficientNetV2S and DenseNet201 architectures to detect diabetic retinopathy from retinal images.

## Problem Description
Diabetic retinopathy is a leading cause of blindness in working-age adults. Early detection is crucial for preventing vision loss. This model aims to automate the detection and grading of diabetic retinopathy using retinal images.

## Dataset
The model uses the APTOS 2019 Blindness Detection dataset, which contains:
- Retinal images in PNG format
- Labels indicating the severity of diabetic retinopathy (0-4)
- Training data with image IDs and corresponding diagnosis labels

Dataset Structure:
```
/kaggle/input/aptos2019-blindness-detection/
├── train.csv
└── train_images/
    └── *.png
```

## Model Architecture
The solution implements a dual-backbone neural network:
1. Primary Features:
   - Ensemble of EfficientNetV2S and DenseNet201
   - Pre-trained on ImageNet
   - Custom top layers with dropout and batch normalization
   - Concatenated features from both backbones
   
2. Technical Specifications:
   - Input image size: 380x380 pixels
   - Batch size: 16
   - Learning rate: 0.0001
   - Optimizer: Adam
   - Loss function: Sparse Categorical Crossentropy

## Key Features
- K-fold cross-validation (n_splits=2)
- Data augmentation pipeline
- GPU support with memory growth
- Early stopping and learning rate reduction
- Model checkpointing
- Batch normalization for better training stability

## Requirements
```
tensorflow
numpy
pandas
scikit-learn
```

## Data Preprocessing
The pipeline includes:
- Image resizing to 380x380
- Pixel normalization
- Data augmentation:
  - Random flips (horizontal and vertical)
  - Random brightness adjustments
  - Random contrast variations
  - Random saturation changes
  - Random hue adjustments

## Training
The model implements a 2-fold cross-validation strategy with:
- 20 epochs per fold
- Early stopping with 15 epochs patience
- Learning rate reduction on plateau
- Best model checkpointing
- Automatic GPU memory growth configuration

## Model Performance
The model's performance can be monitored through:
- Training accuracy
- Validation accuracy
- Best validation accuracy per fold

## Usage
1. Ensure the APTOS dataset is in the correct directory structure
2. Run the training script:
```python
python train.py
```

## Model Outputs
The training process saves:
- Best model weights for each fold (`best_model_fold_*.keras`)
- Training history
- Validation metrics


For more information about the competition and dataset, visit [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
