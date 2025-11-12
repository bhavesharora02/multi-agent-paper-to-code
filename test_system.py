"""
Simple test script to verify the ML/DL Paper to Code system works.
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import PDFParser
from extractors.algorithm_extractor import AlgorithmExtractor
from generators.code_generator import CodeGenerator

def test_system():
    """Test the core system functionality."""
    print("Testing ML/DL Paper to Code System...")
    
    try:
        # Test PDF parser
        print("1. Testing PDF Parser...")
        pdf_parser = PDFParser()
        print("   [OK] PDF Parser initialized successfully")
        
        # Test algorithm extractor
        print("2. Testing Algorithm Extractor...")
        algorithm_extractor = AlgorithmExtractor()
        print("   [OK] Algorithm Extractor initialized successfully")
        
        # Test code generator
        print("3. Testing Code Generator...")
        code_generator = CodeGenerator()
        print("   [OK] Code Generator initialized successfully")
        
        # Test with sample text
        print("4. Testing with sample text...")
        sample_text = """
        We propose a novel deep learning architecture for image classification.
        Our approach uses convolutional neural networks (CNN) with attention mechanisms.
        The algorithm achieves state-of-the-art performance on benchmark datasets.
        """
        
        algorithms = algorithm_extractor.extract_algorithms(sample_text)
        print(f"   [OK] Found {len(algorithms)} algorithms")
        
        # Test code generation
        print("5. Testing code generation...")
        generated_code = code_generator.generate_code(algorithms, 'pytorch')
        print(f"   [OK] Generated {len(generated_code)} characters of code")
        
        print("\n[SUCCESS] All tests passed! The system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n[READY] System is ready for web application!")
    else:
        print("\n[FAILED] Please fix the issues before running the web app.")