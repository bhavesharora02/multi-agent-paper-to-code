"""
LLM-enhanced algorithm extraction module.
Uses OpenAI/Anthropic LLMs for intelligent algorithm detection and extraction.
"""

import json
import logging
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from extractors.algorithm_extractor import Algorithm
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import PAPER_ANALYSIS_SYSTEM_PROMPT, PAPER_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class LLMAlgorithmExtractor:
    """
    Enhanced algorithm extractor using LLM for intelligent extraction.
    Falls back to rule-based extraction if LLM fails.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize LLM-based algorithm extractor.
        
        Args:
            config: Configuration dictionary with:
                - llm_provider: "openai" or "anthropic" (default: "openai")
                - use_llm: Whether to use LLM (default: True)
                - fallback_to_rules: Use rule-based if LLM fails (default: True)
                - max_text_length: Max text length to send to LLM (default: 8000)
        """
        self.config = config or {}
        self.use_llm = self.config.get("use_llm", True)
        self.fallback_to_rules = self.config.get("fallback_to_rules", True)
        self.max_text_length = self.config.get("max_text_length", 8000)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client if enabled
        if self.use_llm:
            try:
                provider_str = self.config.get("llm_provider", "openai").lower()
                if provider_str == "openai":
                    provider = LLMProvider.OPENAI
                elif provider_str == "anthropic":
                    provider = LLMProvider.ANTHROPIC
                elif provider_str == "openrouter":
                    provider = LLMProvider.OPENROUTER
                elif provider_str == "groq":
                    provider = LLMProvider.GROQ
                else:
                    provider = LLMProvider.OPENAI  # Default
                self.llm_client = LLMClient(provider=provider)
                self.logger.info(f"Initialized LLM client with provider: {provider.value}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM client: {e}. Falling back to rule-based extraction.")
                self.use_llm = False
                self.llm_client = None
        else:
            self.llm_client = None
        
        # Fallback to rule-based extractor if needed
        if self.fallback_to_rules:
            try:
                from extractors.algorithm_extractor import AlgorithmExtractor
                self.rule_extractor = AlgorithmExtractor(config)
                self.logger.info("Rule-based extractor initialized as fallback")
            except Exception as e:
                self.logger.warning(f"Failed to initialize rule-based extractor: {e}")
                self.rule_extractor = None
        else:
            self.rule_extractor = None
    
    def extract_algorithms(self, text: str) -> List[Algorithm]:
        """
        Extract algorithms from paper text using LLM.
        
        Args:
            text: Paper text content
            
        Returns:
            List of extracted Algorithm objects
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []
        
        # Try LLM extraction first
        if self.use_llm and self.llm_client:
            try:
                algorithms = self._extract_with_llm(text)
                if algorithms:
                    self.logger.info(f"LLM extraction successful: found {len(algorithms)} algorithms")
                    return algorithms
                else:
                    self.logger.warning("LLM extraction returned no algorithms")
            except Exception as e:
                self.logger.error(f"LLM extraction failed: {e}")
        
        # Fallback to rule-based extraction
        if self.fallback_to_rules and self.rule_extractor:
            self.logger.info("Falling back to rule-based extraction")
            try:
                return self.rule_extractor.extract_algorithms(text)
            except Exception as e:
                self.logger.error(f"Rule-based extraction also failed: {e}")
        
        return []
    
    def _extract_with_llm(self, text: str) -> List[Algorithm]:
        """Extract algorithms using LLM."""
        # Truncate text if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            self.logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        # Prepare prompt
        prompt = PAPER_ANALYSIS_PROMPT.format(text=text)
        
        # Generate response
        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=PAPER_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.3  # Lower temperature for more consistent extraction
            )
            
            # Convert response to Algorithm objects
            algorithms = []
            
            # Handle different response formats
            if isinstance(response, list):
                algorithm_data = response
            elif isinstance(response, dict) and "algorithms" in response:
                algorithm_data = response["algorithms"]
            elif isinstance(response, dict) and "raw_response" in response:
                # Failed to parse JSON, try to extract from raw response
                self.logger.warning("Failed to parse JSON, attempting to extract from raw response")
                return []
            else:
                algorithm_data = []
            
            for alg_data in algorithm_data:
                try:
                    algorithm = Algorithm(
                        name=alg_data.get("name", "Unknown Algorithm"),
                        description=alg_data.get("description", ""),
                        pseudocode=alg_data.get("pseudocode"),
                        parameters=alg_data.get("key_parameters", []) or [],
                        complexity=alg_data.get("complexity"),
                        framework=alg_data.get("framework_suggestions", [None])[0] if isinstance(alg_data.get("framework_suggestions"), list) else alg_data.get("framework_suggestions"),
                        confidence=float(alg_data.get("confidence", 0.5))
                    )
                    algorithms.append(algorithm)
                except Exception as e:
                    self.logger.warning(f"Failed to create Algorithm object from data: {alg_data}. Error: {e}")
                    continue
            
            return algorithms
            
        except Exception as e:
            self.logger.error(f"Error in LLM extraction: {e}")
            raise

