"""
PDF parsing module for extracting text content from research papers.
"""

import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import logging


class PDFParser:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize PDF parser with configuration.
        
        Args:
            config: Configuration dictionary with parsing options
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Try pdfplumber first (better for complex layouts)
            text = self._extract_with_pdfplumber(pdf_path)
            if text.strip():
                return text
            
            # Fallback to PyPDF2
            text = self._extract_with_pypdf2(pdf_path)
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 as fallback."""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
        
        return "\n\n".join(text_parts)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text for better algorithm detection.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove page markers
        text = text.replace('--- Page', '')
        
        # Clean up common PDF artifacts
        artifacts = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        return text.strip()
