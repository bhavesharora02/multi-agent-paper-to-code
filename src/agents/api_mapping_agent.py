"""
API/Library Mapping Agent.
Maps algorithm components to appropriate framework libraries and APIs.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import (
    PaperToCodeIR, AlgorithmInfo, MappedComponent
)
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import API_MAPPING_SYSTEM_PROMPT, API_MAPPING_PROMPT


class APIMappingAgent(BaseAgent):
    """
    Maps algorithm components to appropriate libraries and frameworks.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize API Mapping Agent."""
        super().__init__(config, llm_client)
        
        # Initialize LLM client if not provided
        if not self.llm_client:
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
                self.log_progress(f"Initialized LLM client: {provider.value}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
        
        # Framework preferences
        self.default_framework = self.config.get("default_framework", "pytorch")
        self.supported_frameworks = ["pytorch", "tensorflow", "sklearn"]
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Map algorithms to framework APIs.
        
        Args:
            ir: Intermediate representation with interpreted algorithms
            
        Returns:
            Updated IR with mapped components
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        if not ir.algorithms:
            self.log_progress("No algorithms to map", "warning")
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress(f"Mapping {len(ir.algorithms)} algorithms to APIs...")
        
        try:
            # Determine target framework
            framework = self._determine_framework(ir)
            self.log_progress(f"Target framework: {framework}")
            
            # Map each algorithm
            mapped_components = []
            for i, algorithm in enumerate(ir.algorithms):
                self.log_progress(f"Mapping algorithm {i+1}/{len(ir.algorithms)}: {algorithm.name}")
                
                if self.llm_client:
                    mapping = self._map_with_llm(algorithm, framework)
                    if mapping:
                        component = self._create_mapped_component(algorithm, mapping, framework)
                        mapped_components.append(component)
                else:
                    # Fallback: basic mapping
                    component = self._basic_mapping(algorithm, framework)
                    mapped_components.append(component)
            
            ir.mapped_components = mapped_components
            self.update_ir_status(ir, "completed")
            self.log_progress(f"Mapped {len(mapped_components)} components")
            
        except Exception as e:
            self.logger.error(f"Error in API mapping: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def _determine_framework(self, ir: PaperToCodeIR) -> str:
        """Determine target framework from IR or config."""
        # Check if framework is specified in config
        if 'target_framework' in self.config:
            return self.config['target_framework']
        
        # Check algorithm suggestions
        if ir.algorithms:
            framework_suggestions = []
            for alg in ir.algorithms:
                if alg.framework_suggestions:
                    framework_suggestions.extend(alg.framework_suggestions)
            
            if framework_suggestions:
                # Count occurrences
                from collections import Counter
                counts = Counter(framework_suggestions)
                most_common = counts.most_common(1)[0][0]
                if most_common.lower() in self.supported_frameworks:
                    return most_common.lower()
        
        return self.default_framework
    
    def _map_with_llm(self, algorithm: AlgorithmInfo, framework: str) -> Optional[Dict]:
        """Map algorithm to framework using LLM."""
        if not self.llm_client:
            return None
        
        try:
            # Prepare component information
            components = {
                "algorithm_name": algorithm.name,
                "algorithm_type": algorithm.type,
                "description": algorithm.description,
                "parameters": algorithm.parameters,
                "workflow_steps": algorithm.workflow_steps
            }
            
            prompt = API_MAPPING_PROMPT.format(
                framework=framework,
                components=self._format_components(components)
            )
            
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=API_MAPPING_SYSTEM_PROMPT.replace("{framework}", framework),
                temperature=0.3
            )
            
            # Handle response format
            if isinstance(response, list):
                return response[0] if response else None
            elif isinstance(response, dict):
                return response
            return None
            
        except Exception as e:
            self.logger.error(f"Error in LLM mapping: {e}")
            return None
    
    def _format_components(self, components: Dict) -> str:
        """Format components for prompt."""
        lines = [
            f"Algorithm: {components['algorithm_name']}",
            f"Type: {components['algorithm_type']}",
            f"Description: {components['description']}",
        ]
        
        if components.get('parameters'):
            lines.append(f"Parameters: {', '.join(components['parameters'])}")
        
        if components.get('workflow_steps'):
            lines.append(f"\nWorkflow Steps: {len(components['workflow_steps'])} steps")
        
        return "\n".join(lines)
    
    def _create_mapped_component(
        self, 
        algorithm: AlgorithmInfo, 
        mapping: Dict, 
        framework: str
    ) -> MappedComponent:
        """Create MappedComponent from mapping result."""
        return MappedComponent(
            algorithm_id=algorithm.name,
            framework=framework,
            library=mapping.get("library", self._get_default_library(framework)),
            function_class=mapping.get("function/class", "Unknown"),
            parameters=mapping.get("parameters", {}),
            code_snippet=mapping.get("code_snippet"),
            documentation_reference=mapping.get("documentation_reference")
        )
    
    def _basic_mapping(self, algorithm: AlgorithmInfo, framework: str) -> MappedComponent:
        """Basic mapping without LLM (fallback)."""
        # Simple heuristic-based mapping
        library = self._get_default_library(framework)
        function_class = self._suggest_class_name(algorithm, framework)
        
        return MappedComponent(
            algorithm_id=algorithm.name,
            framework=framework,
            library=library,
            function_class=function_class,
            parameters={},
            code_snippet=None,
            documentation_reference=None
        )
    
    def _get_default_library(self, framework: str) -> str:
        """Get default library name for framework."""
        libraries = {
            "pytorch": "torch.nn",
            "tensorflow": "tensorflow.keras",
            "sklearn": "sklearn"
        }
        return libraries.get(framework.lower(), "unknown")
    
    def _suggest_class_name(self, algorithm: AlgorithmInfo, framework: str) -> str:
        """Suggest class/function name based on algorithm."""
        # Simple heuristics
        name_lower = algorithm.name.lower()
        
        if "transformer" in name_lower or "attention" in name_lower:
            if framework == "pytorch":
                return "nn.Transformer"
            elif framework == "tensorflow":
                return "keras.layers.MultiHeadAttention"
        
        if "cnn" in name_lower or "convolutional" in name_lower:
            if framework == "pytorch":
                return "nn.Conv2d"
            elif framework == "tensorflow":
                return "keras.layers.Conv2D"
        
        if "lstm" in name_lower or "rnn" in name_lower:
            if framework == "pytorch":
                return "nn.LSTM"
            elif framework == "tensorflow":
                return "keras.layers.LSTM"
        
        # Default
        return "Model" if framework in ["pytorch", "tensorflow"] else "Classifier"

