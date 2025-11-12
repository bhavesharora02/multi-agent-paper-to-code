"""
Example test cases for the ML/DL Paper to Code system.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys
sys.path.append('src')
from parsers.pdf_parser import PDFParser
from extractors.algorithm_extractor import AlgorithmExtractor
from generators.code_generator import CodeGenerator
from utils.helpers import clean_text, extract_equations, validate_pdf_file


class TestPDFParser:
    """Test cases for PDF parser."""
    
    def test_init(self):
        """Test PDF parser initialization."""
        parser = PDFParser()
        assert parser.config == {}
        assert parser.logger is not None
    
    def test_init_with_config(self):
        """Test PDF parser initialization with config."""
        config = {"method": "pdfplumber"}
        parser = PDFParser(config)
        assert parser.config == config
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        parser = PDFParser()
        text = "This   is   a   test   with   multiple   spaces."
        processed = parser.preprocess_text(text)
        assert processed == "This is a test with multiple spaces."
    
    def test_preprocess_text_with_artifacts(self):
        """Test text preprocessing with PDF artifacts."""
        parser = PDFParser()
        text = "This\x00is\x01a\x02test\x03with\x04artifacts\x05."
        processed = parser.preprocess_text(text)
        assert "\x00" not in processed
        assert "\x01" not in processed
        assert "\x02" not in processed


class TestAlgorithmExtractor:
    """Test cases for algorithm extractor."""
    
    def test_init(self):
        """Test algorithm extractor initialization."""
        extractor = AlgorithmExtractor()
        assert extractor.config == {}
        assert extractor.logger is not None
        assert extractor.algorithm_patterns is not None
    
    def test_detect_algorithms(self):
        """Test algorithm detection."""
        extractor = AlgorithmExtractor()
        text = "We propose a novel neural network architecture using CNN and RNN layers."
        algorithms = extractor._detect_algorithms(text)
        
        assert len(algorithms) > 0
        assert any("neural network" in alg.name.lower() for alg in algorithms)
        assert any("CNN" in alg.name for alg in algorithms)
    
    def test_extract_description(self):
        """Test description extraction."""
        extractor = AlgorithmExtractor()
        context = "This is a neural network that processes sequential data efficiently."
        description = extractor._extract_description(context)
        assert len(description) > 0
        assert "neural network" in description.lower()
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        extractor = AlgorithmExtractor()
        context = "We propose a novel algorithm for machine learning tasks."
        confidence = extractor._calculate_confidence(context, "algorithm")
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to "propose" and "novel"


class TestCodeGenerator:
    """Test cases for code generator."""
    
    def test_init(self):
        """Test code generator initialization."""
        generator = CodeGenerator()
        assert generator.config == {}
        assert generator.logger is not None
        assert generator.templates is not None
    
    def test_generate_imports_pytorch(self):
        """Test PyTorch import generation."""
        generator = CodeGenerator()
        imports = generator._generate_imports("pytorch")
        assert "import torch" in imports
        assert "import torch.nn as nn" in imports
        assert "import torch.optim as optim" in imports
    
    def test_generate_imports_tensorflow(self):
        """Test TensorFlow import generation."""
        generator = CodeGenerator()
        imports = generator._generate_imports("tensorflow")
        assert "import tensorflow as tf" in imports
        assert "from tensorflow import keras" in imports
    
    def test_generate_imports_sklearn(self):
        """Test scikit-learn import generation."""
        generator = CodeGenerator()
        imports = generator._generate_imports("sklearn")
        assert "from sklearn.model_selection import train_test_split" in imports
        assert "from sklearn.ensemble import RandomForestClassifier" in imports
    
    def test_sanitize_name(self):
        """Test name sanitization."""
        generator = CodeGenerator()
        
        # Test normal name
        assert generator._sanitize_name("Neural Network") == "NeuralNetwork"
        
        # Test name with special characters
        assert generator._sanitize_name("CNN-LSTM") == "CNNLSTM"
        
        # Test name starting with number
        assert generator._sanitize_name("3D CNN") == "Algorithm3DCNN"
        
        # Test empty name
        assert generator._sanitize_name("") == "Algorithm"


class TestHelpers:
    """Test cases for helper functions."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "This   is   a   test   with   multiple   spaces."
        cleaned = clean_text(text)
        assert cleaned == "This is a test with multiple spaces."
    
    def test_extract_equations(self):
        """Test equation extraction."""
        text = "The loss function is L = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2"
        equations = extract_equations(text)
        assert len(equations) > 0
        assert any("\\sum" in eq for eq in equations)
    
    def test_validate_pdf_file(self):
        """Test PDF file validation."""
        # Test with non-existent file
        assert not validate_pdf_file("nonexistent.pdf")
        
        # Test with non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not a PDF")
            temp_path = f.name
        
        try:
            assert not validate_pdf_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration test cases."""
    
    def test_empty_algorithm_list(self):
        """Test code generation with empty algorithm list."""
        generator = CodeGenerator()
        code = generator.generate_code([], "pytorch")
        
        assert "PlaceholderModel" in code
        assert "No algorithms detected" in code
    
    def test_single_algorithm(self):
        """Test code generation with single algorithm."""
        from extractors.algorithm_extractor import Algorithm
        
        algorithm = Algorithm(
            name="Neural Network",
            description="A simple neural network for classification",
            parameters=["input_size", "hidden_size", "output_size"],
            confidence=0.8
        )
        
        generator = CodeGenerator()
        code = generator.generate_code([algorithm], "pytorch")
        
        assert "class NeuralNetwork_0" in code
        assert "nn.Module" in code
        assert "forward" in code
        assert "train_model" in code


if __name__ == "__main__":
    pytest.main([__file__])
