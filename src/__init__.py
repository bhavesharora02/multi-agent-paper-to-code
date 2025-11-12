"""
ML/DL Paper to Code automation system.

This package provides tools for automatically converting machine learning and
deep learning research papers into executable Python code.
"""

__version__ = "1.0.0"
__author__ = "Bhavesh Arora"
__email__ = "bhavesh.arora@example.com"

# Import modules with proper error handling
try:
    from .parsers.pdf_parser import PDFParser
    from .extractors.algorithm_extractor import AlgorithmExtractor, Algorithm
    from .generators.code_generator import CodeGenerator
    from .utils.helpers import (
        setup_logging,
        clean_text,
        extract_equations,
        extract_citations,
        extract_references,
        validate_pdf_file,
        create_output_directory,
        format_code_output,
        get_algorithm_category,
        generate_docstring
    )
    
    __all__ = [
        "PDFParser",
        "AlgorithmExtractor", 
        "Algorithm",
        "CodeGenerator",
        "setup_logging",
        "clean_text",
        "extract_equations",
        "extract_citations",
        "extract_references",
        "validate_pdf_file",
        "create_output_directory",
        "format_code_output",
        "get_algorithm_category",
        "generate_docstring"
    ]
except ImportError as e:
    # Handle import errors gracefully
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
