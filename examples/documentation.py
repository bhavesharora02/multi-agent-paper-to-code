"""
Example usage and demonstration of the ML/DL Paper to Code system.
"""

# Example 1: Basic usage with a sample paper
example_paper_text = """
Abstract: We propose a novel deep learning architecture for image classification tasks.
Our approach combines convolutional neural networks (CNN) with attention mechanisms
to achieve state-of-the-art performance on benchmark datasets.

1. Introduction
Deep learning has revolutionized computer vision tasks, particularly image classification.
Convolutional Neural Networks (CNNs) have become the standard approach for extracting
features from images.

2. Methodology
Our proposed architecture consists of three main components:
- A CNN backbone for feature extraction
- An attention mechanism for focusing on relevant regions
- A classification head for final predictions

Algorithm 1: CNN-Attention Classification
Input: Image X, CNN parameters θ_cnn, Attention parameters θ_att
Output: Class predictions y_pred

1. Extract features: f = CNN(X, θ_cnn)
2. Compute attention weights: α = Attention(f, θ_att)
3. Apply attention: f_att = f ⊙ α
4. Classify: y_pred = Classifier(f_att)
5. Return y_pred

The time complexity of our algorithm is O(n²) where n is the number of pixels.

3. Experiments
We evaluate our method on CIFAR-10 and ImageNet datasets using Adam optimizer
with learning rate 0.001 for 100 epochs.

4. Results
Our CNN-Attention model achieves 95.2% accuracy on CIFAR-10, outperforming
baseline CNN models by 3.1%.

References:
[1] LeCun, Y., et al. "Gradient-based learning applied to document recognition."
[2] Vaswani, A., et al. "Attention is all you need."
"""

# Example 2: Sample configuration for different frameworks
example_configs = {
    "pytorch_config": {
        "pdf_parser": {"method": "pdfplumber"},
        "extractor": {"confidence_threshold": 0.5},
        "generator": {
            "default_framework": "pytorch",
            "frameworks": {
                "pytorch": {
                    "default_hidden_size": 256,
                    "default_learning_rate": 0.001,
                    "default_epochs": 200
                }
            }
        }
    },
    
    "tensorflow_config": {
        "pdf_parser": {"method": "pdfplumber"},
        "extractor": {"confidence_threshold": 0.4},
        "generator": {
            "default_framework": "tensorflow",
            "frameworks": {
                "tensorflow": {
                    "default_hidden_units": 512,
                    "default_learning_rate": 0.0001,
                    "default_epochs": 150
                }
            }
        }
    },
    
    "sklearn_config": {
        "pdf_parser": {"method": "pdfplumber"},
        "extractor": {"confidence_threshold": 0.6},
        "generator": {
            "default_framework": "sklearn",
            "frameworks": {
                "sklearn": {
                    "default_test_size": 0.3,
                    "include_preprocessing": True,
                    "include_evaluation": True
                }
            }
        }
    }
}

# Example 3: Expected output structure
expected_output_structure = """
# Generated Code Structure

## Imports Section
- Framework-specific imports (PyTorch/TensorFlow/scikit-learn)
- Standard ML libraries (numpy, pandas, matplotlib)
- Utility imports

## Algorithm Classes
- One class per detected algorithm
- Proper docstrings with descriptions
- Parameter definitions
- Training and prediction methods

## Main Function
- Data loading and preprocessing
- Model instantiation and training
- Evaluation and visualization
- Results reporting

## Example Usage
- Command-line interface
- Configuration file support
- Batch processing capabilities
"""

# Example 4: Command-line usage examples
command_examples = """
# Basic usage
python src/main.py --input paper.pdf --output generated_code.py

# With specific framework
python src/main.py --input paper.pdf --output pytorch_code.py --framework pytorch

# With custom configuration
python src/main.py --input paper.pdf --output code.py --config config/custom.yaml

# Verbose output
python src/main.py --input paper.pdf --output code.py --verbose

# Batch processing multiple papers
for paper in papers/*.pdf; do
    python src/main.py --input "$paper" --output "generated/$(basename "$paper" .pdf).py"
done
"""

# Example 5: Sample generated code output
sample_generated_code = '''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

class CNN_Attention_0(nn.Module):
    """
    CNN-Attention Classification
    
    A novel deep learning architecture for image classification tasks.
    Our approach combines convolutional neural networks (CNN) with attention mechanisms
    to achieve state-of-the-art performance on benchmark datasets.
    
    Parameters: CNN parameters θ_cnn, Attention parameters θ_att
    Complexity: O(n²) where n is the number of pixels
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 10):
        super(CNN_Attention_0, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # CNN backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, 8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract features using CNN backbone
        features = self.cnn_backbone(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
        attn_features = attn_output.squeeze(1)
        
        # Classify
        output = self.classifier(attn_features)
        return output
    
    def train_model(self, train_loader, val_loader, epochs: int = 100, lr: float = 0.001):
        """Train the model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    output = self(data)
                    val_loss += criterion(output, target).item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        return train_losses, val_losses

def main():
    """Main execution function."""
    print("ML/DL Paper to Code - Generated Implementation")
    print("=" * 50)
    
    # Example usage
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Loading data from: {data_file}")
        # TODO: Load actual data
    else:
        print("No data file provided. Using synthetic data for demonstration.")
        # Generate synthetic data
        X = np.random.randn(1000, 3, 32, 32)  # CIFAR-10 like data
        y = np.random.randint(0, 10, 1000)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train CNN-Attention model
    print(f"\\nTraining CNN-Attention model...")
    model_0 = CNN_Attention_0(input_size=3*32*32, hidden_size=128, output_size=10)
    train_losses, val_losses = model_0.train_model(train_loader, val_loader, epochs=100, lr=0.001)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("\\nCNN-Attention model trained successfully!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    import sys
    main()
'''

print("ML/DL Paper to Code System - Examples and Documentation")
print("=" * 60)
print("\nThis file contains examples and documentation for the ML/DL Paper to Code system.")
print("The system automatically converts research papers into executable Python code.")
print("\nKey Features:")
print("- Automatic PDF text extraction")
print("- Algorithm detection and classification")
print("- Multi-framework code generation (PyTorch, TensorFlow, scikit-learn)")
print("- Configurable output formats")
print("- Comprehensive testing suite")
print("\nFor more information, see the README.md file and run the tests.")
