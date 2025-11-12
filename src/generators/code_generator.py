"""
Code generation module for creating Python implementations from extracted algorithms.
"""

import re
from typing import List, Dict, Optional
import logging
from extractors.algorithm_extractor import Algorithm


class CodeGenerator:
    """Generates Python code from extracted algorithms."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize code generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Framework-specific templates
        self.templates = {
            'pytorch': self._get_pytorch_templates(),
            'tensorflow': self._get_tensorflow_templates(),
            'sklearn': self._get_sklearn_templates()
        }
    
    def generate_code(self, algorithms: List[Algorithm], framework: str = 'pytorch') -> str:
        """
        Generate Python code for the given algorithms.
        
        Args:
            algorithms: List of extracted algorithms
            framework: Target ML framework
            
        Returns:
            Generated Python code
        """
        if not algorithms:
            return self._generate_empty_template(framework)
        
        # Generate imports
        imports = self._generate_imports(framework)
        
        # Generate code for each algorithm
        algorithm_codes = []
        for i, algorithm in enumerate(algorithms):
            code = self._generate_algorithm_code(algorithm, framework, i)
            algorithm_codes.append(code)
        
        # Combine all code
        full_code = imports + "\n\n" + "\n\n".join(algorithm_codes)
        
        # Add main execution block
        main_block = self._generate_main_block(algorithms, framework)
        full_code += "\n\n" + main_block
        
        return full_code
    
    def _generate_imports(self, framework: str) -> str:
        """Generate import statements based on framework."""
        base_imports = [
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "from typing import List, Dict, Optional, Tuple",
            "import logging"
        ]
        
        framework_imports = {
            'pytorch': [
                "import torch",
                "import torch.nn as nn",
                "import torch.optim as optim",
                "import torch.nn.functional as F",
                "from torch.utils.data import DataLoader, Dataset"
            ],
            'tensorflow': [
                "import tensorflow as tf",
                "from tensorflow import keras",
                "from tensorflow.keras import layers, models, optimizers",
                "from tensorflow.keras.datasets import mnist, cifar10"
            ],
            'sklearn': [
                "from sklearn.model_selection import train_test_split",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.metrics import accuracy_score, classification_report",
                "from sklearn.ensemble import RandomForestClassifier",
                "from sklearn.linear_model import LogisticRegression"
            ]
        }
        
        all_imports = base_imports + framework_imports.get(framework, [])
        return "\n".join(all_imports)
    
    def _generate_algorithm_code(self, algorithm: Algorithm, framework: str, index: int) -> str:
        """Generate code for a specific algorithm."""
        class_name = self._sanitize_name(algorithm.name) + f"_{index}"
        
        if framework == 'pytorch':
            return self._generate_pytorch_class(algorithm, class_name)
        elif framework == 'tensorflow':
            return self._generate_tensorflow_class(algorithm, class_name)
        elif framework == 'sklearn':
            return self._generate_sklearn_class(algorithm, class_name)
        else:
            return self._generate_generic_class(algorithm, class_name)
    
    def _generate_pytorch_class(self, algorithm: Algorithm, class_name: str) -> str:
        """Generate PyTorch implementation."""
        code = f"""
class {class_name}(nn.Module):
    \"\"\"
    {algorithm.description}
    
    Parameters: {', '.join(algorithm.parameters) if algorithm.parameters else 'None'}
    Complexity: {algorithm.complexity or 'Not specified'}
    \"\"\"
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 10):
        super({class_name}, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define layers based on algorithm type
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        \"\"\"Forward pass through the network.\"\"\"
        return self.layers(x)
    
    def train_model(self, train_loader, val_loader, epochs: int = 100, lr: float = 0.001):
        \"\"\"Train the model.\"\"\"
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
                print(f'Epoch {{epoch}}, Train Loss: {{train_losses[-1]:.4f}}, Val Loss: {{val_losses[-1]:.4f}}')
        
        return train_losses, val_losses
"""
        return code.strip()
    
    def _generate_tensorflow_class(self, algorithm: Algorithm, class_name: str) -> str:
        """Generate TensorFlow implementation."""
        code = f"""
class {class_name}:
    \"\"\"
    {algorithm.description}
    
    Parameters: {', '.join(algorithm.parameters) if algorithm.parameters else 'None'}
    Complexity: {algorithm.complexity or 'Not specified'}
    \"\"\"
    
    def __init__(self, input_shape: tuple, hidden_units: int = 128, num_classes: int = 10):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        \"\"\"Build the model architecture.\"\"\"
        model = keras.Sequential([
            layers.Dense(self.hidden_units, activation='relu', input_shape=self.input_shape),
            layers.Dropout(0.2),
            layers.Dense(self.hidden_units, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, x_train, y_train, x_val, y_val, epochs: int = 100, batch_size: int = 32):
        \"\"\"Train the model.\"\"\"
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict(self, x):
        \"\"\"Make predictions.\"\"\"
        return self.model.predict(x)
"""
        return code.strip()
    
    def _generate_sklearn_class(self, algorithm: Algorithm, class_name: str) -> str:
        """Generate scikit-learn implementation."""
        code = f"""
class {class_name}:
    \"\"\"
    {algorithm.description}
    
    Parameters: {', '.join(algorithm.parameters) if algorithm.parameters else 'None'}
    Complexity: {algorithm.complexity or 'Not specified'}
    \"\"\"
    
    def __init__(self, **kwargs):
        # Choose appropriate sklearn model based on algorithm name
        if 'random forest' in '{algorithm.name}'.lower():
            self.model = RandomForestClassifier(**kwargs)
        elif 'logistic' in '{algorithm.name}'.lower():
            self.model = LogisticRegression(**kwargs)
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(**kwargs)
        
        self.scaler = StandardScaler()
    
    def preprocess_data(self, X):
        \"\"\"Preprocess the input data.\"\"\"
        return self.scaler.fit_transform(X)
    
    def train(self, X_train, y_train):
        \"\"\"Train the model.\"\"\"
        X_train_scaled = self.preprocess_data(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        \"\"\"Evaluate the model.\"\"\"
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {{accuracy:.4f}}")
        print("Classification Report:")
        print(report)
        
        return accuracy, report
"""
        return code.strip()
    
    def _generate_generic_class(self, algorithm: Algorithm, class_name: str) -> str:
        """Generate generic Python implementation."""
        code = f"""
class {class_name}:
    \"\"\"
    {algorithm.description}
    
    Parameters: {', '.join(algorithm.parameters) if algorithm.parameters else 'None'}
    Complexity: {algorithm.complexity or 'Not specified'}
    \"\"\"
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.is_trained = False
    
    def fit(self, X, y):
        \"\"\"Train the algorithm.\"\"\"
        # TODO: Implement training logic based on algorithm
        print("Training {algorithm.name}...")
        self.is_trained = True
    
    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # TODO: Implement prediction logic
        print("Making predictions with {algorithm.name}...")
        return np.zeros(len(X))  # Placeholder
    
    def score(self, X, y):
        \"\"\"Calculate model score.\"\"\"
        predictions = self.predict(X)
        # TODO: Implement appropriate scoring metric
        return 0.0  # Placeholder
"""
        return code.strip()
    
    def _generate_main_block(self, algorithms: List[Algorithm], framework: str) -> str:
        """Generate main execution block."""
        code = """
def main():
    \"\"\"Main execution function.\"\"\"
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
        X = np.random.randn(1000, 20)
        y = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate each algorithm
"""
        
        for i, algorithm in enumerate(algorithms):
            class_name = self._sanitize_name(algorithm.name) + f"_{i}"
            code += f"""
    # {algorithm.name}
    print(f"\\nTraining {algorithm.name}...")
    model_{i} = {class_name}()
    model_{i}.fit(X_train, y_train)
    
    # Evaluate
    if hasattr(model_{i}, 'evaluate'):
        model_{i}.evaluate(X_test, y_test)
    else:
        predictions = model_{i}.predict(X_test)
        print(f"Predictions shape: {{predictions.shape}}")
"""
        
        code += """
    print("\\nAll algorithms trained and evaluated successfully!")


if __name__ == "__main__":
    import sys
    main()
"""
        
        return code
    
    def _generate_empty_template(self, framework: str) -> str:
        """Generate empty template when no algorithms are found."""
        imports = self._generate_imports(framework)
        
        template = f"""{imports}

class PlaceholderModel:
    \"\"\"
    Placeholder model for when no algorithms are detected.
    Please provide a paper with clear algorithm descriptions.
    \"\"\"
    
    def __init__(self):
        self.is_trained = False
    
    def fit(self, X, y):
        print("No algorithms detected in the paper.")
        print("Please ensure the paper contains clear algorithm descriptions.")
        self.is_trained = True
    
    def predict(self, X):
        return np.zeros(len(X))


def main():
    print("No algorithms detected in the provided paper.")
    print("Please check that the paper contains clear algorithm descriptions.")


if __name__ == "__main__":
    main()
"""
        return template
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize algorithm name for use as class name."""
        # Remove special characters and spaces
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = 'Algorithm' + sanitized
        return sanitized or 'Algorithm'
    
    def _get_pytorch_templates(self) -> Dict:
        """Get PyTorch-specific templates."""
        return {
            'neural_network': 'pytorch_nn_template',
            'cnn': 'pytorch_cnn_template',
            'rnn': 'pytorch_rnn_template'
        }
    
    def _get_tensorflow_templates(self) -> Dict:
        """Get TensorFlow-specific templates."""
        return {
            'neural_network': 'tf_nn_template',
            'cnn': 'tf_cnn_template',
            'rnn': 'tf_rnn_template'
        }
    
    def _get_sklearn_templates(self) -> Dict:
        """Get scikit-learn-specific templates."""
        return {
            'classifier': 'sklearn_classifier_template',
            'regressor': 'sklearn_regressor_template',
            'clustering': 'sklearn_clustering_template'
        }
