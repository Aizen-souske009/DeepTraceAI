#!/usr/bin/env python3
"""
Deepfake Detection Model Validation Script
Validates trained models on individual images or datasets
Compatible with the web application API calls
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import sys
import os
import glob
from pathlib import Path

class ResNetDeepfake(nn.Module):
    """ResNet-based model for deepfake detection"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetDeepfake, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImprovedCNN(nn.Module):
    """Custom CNN architecture for deepfake detection"""
    
    def __init__(self, num_classes=1):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_model(model_path, model_type='resnet'):
    """Load the trained model"""
    try:
        if model_type == 'resnet':
            model = ResNetDeepfake(num_classes=1, pretrained=False)
        elif model_type == 'custom':
            model = ImprovedCNN(num_classes=1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load the model state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

def predict(model, image_tensor):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply sigmoid for binary classification
            probability = torch.sigmoid(outputs).item()
            
            # Determine prediction based on threshold
            # Based on training: label 0 = AI, label 1 = Real
            # Low probability = AI (label 0), High probability = Real (label 1)
            if probability > 0.5:
                prediction = 'Real'  # High probability = Real (label 1)
                confidence = probability
            else:
                prediction = 'AI'    # Low probability = AI (label 0)
                confidence = 1 - probability
            
            return prediction, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

def validate_folder(model, folder_path, model_type):
    """Validate all images in a folder"""
    print(f"Validating images in folder: {folder_path}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    
    # Determine expected label from folder name
    folder_name = os.path.basename(folder_path).lower()
    if 'ai' in folder_name or 'fake' in folder_name or 'synthetic' in folder_name:
        expected_label = 'AI'
    elif 'real' in folder_name or 'authentic' in folder_name or 'genuine' in folder_name:
        expected_label = 'Real'
    else:
        expected_label = None
        print("Warning: Cannot determine expected label from folder name")
    
    for img_path in image_files:
        try:
            image_tensor = preprocess_image(img_path)
            prediction, confidence = predict(model, image_tensor)
            
            predictions.append({
                'file': os.path.basename(img_path),
                'prediction': prediction,
                'confidence': confidence,
                'expected': expected_label
            })
            
            if expected_label and prediction == expected_label:
                correct_predictions += 1
            total_predictions += 1
            
            print(f"{os.path.basename(img_path)}: {prediction} ({confidence:.4f})")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Print summary
    print(f"\n--- Validation Summary ---")
    print(f"Total images processed: {total_predictions}")
    if expected_label:
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Expected label: {expected_label}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Count predictions by type
    ai_count = sum(1 for p in predictions if p['prediction'] == 'AI')
    real_count = sum(1 for p in predictions if p['prediction'] == 'Real')
    print(f"AI predictions: {ai_count}")
    print(f"Real predictions: {real_count}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Validate deepfake detection model')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--model_type', default='resnet', choices=['resnet', 'custom'], 
                       help='Type of model (resnet or custom)')
    parser.add_argument('--image', help='Path to a single image to analyze')
    parser.add_argument('--folder', help='Path to folder containing images to analyze')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
    
    if args.folder and not os.path.exists(args.folder):
        print(f"Error: Folder not found at {args.folder}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type)
    
    if args.image:
        # Single image prediction
        print(f"Processing image: {args.image}")
        image_tensor = preprocess_image(args.image)
        prediction, confidence = predict(model, image_tensor)
        
        # Output results in expected format for API
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Analysis complete")
        
    elif args.folder:
        # Folder validation
        validate_folder(model, args.folder, args.model_type)
        
    else:
        print("Error: Please specify either --image or --folder")
        sys.exit(1)

if __name__ == "__main__":
    main()