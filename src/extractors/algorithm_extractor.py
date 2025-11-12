"""
Algorithm extraction module for identifying ML/DL algorithms from paper text.
"""

import re
import spacy
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass


@dataclass
class Algorithm:
    """Represents an extracted algorithm."""
    name: str
    description: str
    pseudocode: Optional[str] = None
    parameters: List[str] = None
    complexity: Optional[str] = None
    framework: Optional[str] = None
    confidence: float = 0.0


class AlgorithmExtractor:
    """Extracts ML/DL algorithms and methodologies from paper text."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize algorithm extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            # Use a simple fallback for text processing
            import re
            self.simple_tokenizer = re.compile(r'\b\w+\b')
        
        # Common ML/DL algorithm patterns
        self.algorithm_patterns = {
            'neural_networks': [
                r'neural network', r'deep learning', r'deep neural network',
                r'CNN', r'convolutional neural network', r'RNN', r'recurrent neural network',
                r'LSTM', r'GRU', r'Transformer', r'attention mechanism'
            ],
            'supervised_learning': [
                r'linear regression', r'logistic regression', r'decision tree',
                r'random forest', r'support vector machine', r'SVM', r'naive bayes',
                r'k-nearest neighbors', r'k-NN', r'gradient boosting'
            ],
            'unsupervised_learning': [
                r'k-means', r'clustering', r'principal component analysis', r'PCA',
                r'independent component analysis', r'ICA', r't-SNE', r'UMAP',
                r'hierarchical clustering', r'DBSCAN'
            ],
            'optimization': [
                r'gradient descent', r'stochastic gradient descent', r'SGD',
                r'Adam', r'RMSprop', r'AdaGrad', r'AdamW', r'learning rate',
                r'backpropagation', r'batch normalization'
            ],
            'regularization': [
                r'dropout', r'L1 regularization', r'L2 regularization',
                r'weight decay', r'early stopping', r'data augmentation'
            ]
        }
        
        # Mathematical notation patterns
        self.math_patterns = [
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'[a-zA-Z]\s*=\s*[a-zA-Z0-9\+\-\*\/\(\)]+',  # Equations
            r'\\sum', r'\\prod', r'\\int',  # Mathematical operators
            r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
        ]
    
    def extract_algorithms(self, text: str) -> List[Algorithm]:
        """
        Extract algorithms from paper text.
        
        Args:
            text: Preprocessed paper text
            
        Returns:
            List of extracted algorithms
        """
        algorithms = []
        
        # Split text into sections
        sections = self._split_into_sections(text)
        
        for section in sections:
            # Detect algorithm mentions
            detected_algorithms = self._detect_algorithms(section)
            
            # Extract pseudocode and parameters
            for alg in detected_algorithms:
                alg.pseudocode = self._extract_pseudocode(section, alg.name)
                alg.parameters = self._extract_parameters(section, alg.name)
                alg.complexity = self._extract_complexity(section, alg.name)
                algorithms.append(alg)
        
        # Remove duplicates and rank by confidence
        algorithms = self._deduplicate_algorithms(algorithms)
        algorithms.sort(key=lambda x: x.confidence, reverse=True)
        
        return algorithms
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections."""
        # Common section headers
        section_patterns = [
            r'\n\d+\.\s+[A-Z][^\n]*',  # Numbered sections
            r'\n[A-Z][A-Z\s]+\n',  # All caps headers
            r'\n\d+\.\d+\s+[A-Z][^\n]*',  # Subsections
        ]
        
        sections = [text]
        
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        return sections
    
    def _detect_algorithms(self, text: str) -> List[Algorithm]:
        """Detect algorithm mentions in text."""
        algorithms = []
        
        for category, patterns in self.algorithm_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 200)
                    end = min(len(text), match.end() + 200)
                    context = text[start:end]
                    
                    # Create algorithm object
                    algorithm = Algorithm(
                        name=match.group().strip(),
                        description=self._extract_description(context),
                        confidence=self._calculate_confidence(context, pattern)
                    )
                    
                    algorithms.append(algorithm)
        
        return algorithms
    
    def _extract_description(self, context: str) -> str:
        """Extract algorithm description from context."""
        # Look for sentences containing the algorithm
        sentences = re.split(r'[.!?]+', context)
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Filter out very short sentences
                return sentence.strip()
        
        return context[:200] + "..." if len(context) > 200 else context
    
    def _extract_pseudocode(self, text: str, algorithm_name: str) -> Optional[str]:
        """Extract pseudocode for the algorithm."""
        # Look for algorithm blocks
        pseudocode_patterns = [
            r'Algorithm\s+\d+[:\s]*' + re.escape(algorithm_name),
            r'Procedure\s+' + re.escape(algorithm_name),
            r'Function\s+' + re.escape(algorithm_name),
        ]
        
        for pattern in pseudocode_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the algorithm block
                start = match.start()
                # Look for the end of the algorithm (next section or end of text)
                end_patterns = [r'\n\d+\.', r'\n[A-Z][A-Z\s]+\n', r'\n\n']
                
                end = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start:])
                    if end_match:
                        end = start + end_match.start()
                        break
                
                return text[start:end].strip()
        
        return None
    
    def _extract_parameters(self, text: str, algorithm_name: str) -> List[str]:
        """Extract algorithm parameters."""
        parameters = []
        
        # Look for parameter lists
        param_patterns = [
            r'parameters?\s*:?\s*([^.\n]+)',
            r'input\s*:?\s*([^.\n]+)',
            r'hyperparameters?\s*:?\s*([^.\n]+)',
        ]
        
        for pattern in param_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                param_text = match.group(1)
                # Split by common separators
                params = re.split(r'[,;]', param_text)
                parameters.extend([p.strip() for p in params if p.strip()])
        
        return list(set(parameters))  # Remove duplicates
    
    def _extract_complexity(self, text: str, algorithm_name: str) -> Optional[str]:
        """Extract computational complexity information."""
        complexity_patterns = [
            r'time complexity\s*:?\s*([^.\n]+)',
            r'space complexity\s*:?\s*([^.\n]+)',
            r'complexity\s*:?\s*([^.\n]+)',
            r'O\([^)]+\)',  # Big O notation
        ]
        
        for pattern in complexity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        return None
    
    def _calculate_confidence(self, context: str, pattern: str) -> float:
        """Calculate confidence score for algorithm detection."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on context indicators
        confidence_indicators = [
            r'algorithm', r'method', r'approach', r'technique',
            r'implementation', r'proposed', r'novel', r'new'
        ]
        
        for indicator in confidence_indicators:
            if re.search(indicator, context, re.IGNORECASE):
                confidence += 0.1
        
        # Increase confidence if mathematical notation is present
        for math_pattern in self.math_patterns:
            if re.search(math_pattern, context):
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _deduplicate_algorithms(self, algorithms: List[Algorithm]) -> List[Algorithm]:
        """Remove duplicate algorithms."""
        seen = set()
        unique_algorithms = []
        
        for alg in algorithms:
            # Create a key based on algorithm name and description
            key = (alg.name.lower(), alg.description[:100])
            
            if key not in seen:
                seen.add(key)
                unique_algorithms.append(alg)
        
        return unique_algorithms
