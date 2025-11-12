"""
Utility functions for the ML/DL Paper to Code system.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    artifacts = [
        r'\x00', r'\x01', r'\x02', r'\x03', r'\x04', r'\x05',
        r'\x0b', r'\x0c', r'\x0e', r'\x0f', r'\x10', r'\x11',
        r'\x12', r'\x13', r'\x14', r'\x15', r'\x16', r'\x17',
        r'\x18', r'\x19', r'\x1a', r'\x1b', r'\x1c', r'\x1d',
        r'\x1e', r'\x1f'
    ]
    
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    # Remove page numbers and headers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[A-Z\s]+\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def extract_equations(text: str) -> List[str]:
    """
    Extract mathematical equations from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted equations
    """
    # Common equation patterns
    equation_patterns = [
        r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
        r'[a-zA-Z]\s*=\s*[a-zA-Z0-9\+\-\*\/\(\)\^]+',  # Simple equations
        r'\\sum[^\\]*',  # Summations
        r'\\prod[^\\]*',  # Products
        r'\\int[^\\]*',  # Integrals
        r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
        r'\\sqrt\{[^}]+\}',  # Square roots
        r'\\log[^\\]*',  # Logarithms
        r'\\exp[^\\]*',  # Exponentials
    ]
    
    equations = []
    for pattern in equation_patterns:
        matches = re.findall(pattern, text)
        equations.extend(matches)
    
    return list(set(equations))  # Remove duplicates


def extract_citations(text: str) -> List[str]:
    """
    Extract citations from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted citations
    """
    # Citation patterns
    citation_patterns = [
        r'\[[\d,\s\-]+\]',  # [1, 2, 3] format
        r'\([\d,\s\-]+\)',  # (1, 2, 3) format
        r'\[[\d]+\]',  # [1] format
        r'\([\d]+\)',  # (1) format
    ]
    
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))


def extract_references(text: str) -> List[str]:
    """
    Extract reference list from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted references
    """
    # Look for reference section
    ref_pattern = r'(?:references?|bibliography)\s*:?\s*\n(.*?)(?:\n\s*\n|\Z)'
    ref_match = re.search(ref_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if not ref_match:
        return []
    
    ref_text = ref_match.group(1)
    
    # Split by common reference separators
    references = re.split(r'\n\s*\d+\.\s*', ref_text)
    references = [ref.strip() for ref in references if ref.strip()]
    
    return references


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if the file is a valid PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() != '.pdf':
            return False
        
        # Check file size (not empty)
        if path.stat().st_size == 0:
            return False
        
        # Try to read PDF header
        with open(path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
        
        return True
        
    except Exception:
        return False


def create_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to the output file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def format_code_output(code: str, max_line_length: int = 88) -> str:
    """
    Format generated code for better readability.
    
    Args:
        code: Generated code string
        max_line_length: Maximum line length
        
    Returns:
        Formatted code
    """
    lines = code.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            formatted_lines.append('')
            continue
        
        # Handle long lines
        if len(line) > max_line_length:
            # Try to break at logical points
            if 'import' in line and ',' in line:
                # Break import statements
                parts = line.split(',')
                formatted_lines.append(parts[0] + ',')
                for part in parts[1:-1]:
                    formatted_lines.append('    ' + part.strip() + ',')
                formatted_lines.append('    ' + parts[-1].strip())
            else:
                # Simple line break
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def get_algorithm_category(algorithm_name: str) -> str:
    """
    Categorize algorithm based on its name.
    
    Args:
        algorithm_name: Name of the algorithm
        
    Returns:
        Algorithm category
    """
    name_lower = algorithm_name.lower()
    
    if any(keyword in name_lower for keyword in ['neural', 'deep', 'cnn', 'rnn', 'lstm', 'gru', 'transformer']):
        return 'deep_learning'
    elif any(keyword in name_lower for keyword in ['regression', 'classification', 'svm', 'random forest', 'decision tree']):
        return 'supervised_learning'
    elif any(keyword in name_lower for keyword in ['clustering', 'k-means', 'pca', 'dimensionality']):
        return 'unsupervised_learning'
    elif any(keyword in name_lower for keyword in ['optimization', 'gradient', 'adam', 'sgd']):
        return 'optimization'
    else:
        return 'other'


def generate_docstring(algorithm_name: str, description: str, parameters: List[str] = None) -> str:
    """
    Generate a standardized docstring for an algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        description: Algorithm description
        parameters: List of parameters
        
    Returns:
        Formatted docstring
    """
    docstring = f'"""\n{algorithm_name}\n\n{description}\n\n'
    
    if parameters:
        docstring += 'Parameters:\n'
        for param in parameters:
            docstring += f'    {param}: Description\n'
    
    docstring += '\nReturns:\n    None\n"""'
    
    return docstring
