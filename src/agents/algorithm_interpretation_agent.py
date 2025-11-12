"""
Algorithm Interpretation Agent.
Translates mathematical notation and pseudocode into explicit computational workflows.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import PaperToCodeIR, AlgorithmInfo
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import (
    ALGORITHM_INTERPRETATION_SYSTEM_PROMPT,
    ALGORITHM_INTERPRETATION_PROMPT
)


class AlgorithmInterpretationAgent(BaseAgent):
    """
    Interprets algorithms by translating mathematical notation
    and pseudocode into explicit computational workflows.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Algorithm Interpretation Agent."""
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
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Interpret algorithms and create computational workflows.
        
        Args:
            ir: Intermediate representation with extracted algorithms
            
        Returns:
            Updated IR with interpreted workflows
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        if not ir.algorithms:
            self.log_progress("No algorithms to interpret", "warning")
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress(f"Interpreting {len(ir.algorithms)} algorithms...")
        
        try:
            # Interpret each algorithm
            for i, algorithm in enumerate(ir.algorithms):
                self.log_progress(f"Interpreting algorithm {i+1}/{len(ir.algorithms)}: {algorithm.name}")
                
                if self.llm_client:
                    interpretation = self._interpret_with_llm(algorithm)
                    if interpretation:
                        # Update algorithm with workflow information
                        algorithm.workflow_steps = interpretation.get("workflow_steps", [])
                        algorithm.data_dependencies = interpretation.get("data_dependencies", [])
                        # Store implementation notes in a custom field
                        if hasattr(algorithm, '__dict__'):
                            algorithm.__dict__['implementation_notes'] = interpretation.get("implementation_notes", "")
                            algorithm.__dict__['edge_cases'] = interpretation.get("edge_cases", [])
                else:
                    # Fallback: basic interpretation without LLM
                    algorithm.workflow_steps = self._basic_interpretation(algorithm)
                    self.log_progress("Using basic interpretation (no LLM)", "warning")
            
            self.update_ir_status(ir, "completed")
            self.log_progress("Algorithm interpretation completed")
            
        except Exception as e:
            self.logger.error(f"Error in algorithm interpretation: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def _interpret_with_llm(self, algorithm: AlgorithmInfo) -> Optional[Dict]:
        """Interpret algorithm using LLM."""
        if not self.llm_client:
            return None
        
        try:
            # Prepare algorithm information
            algorithm_info = {
                "name": algorithm.name,
                "description": algorithm.description,
                "type": algorithm.type,
                "pseudocode": algorithm.pseudocode or "",
                "mathematical_notation": algorithm.mathematical_notation or "",
                "parameters": algorithm.parameters
            }
            
            prompt = ALGORITHM_INTERPRETATION_PROMPT.format(
                algorithm_info=self._format_algorithm_info(algorithm_info)
            )
            
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=ALGORITHM_INTERPRETATION_SYSTEM_PROMPT,
                temperature=0.3
            )
            
            return response if isinstance(response, dict) else None
            
        except Exception as e:
            self.logger.error(f"Error in LLM interpretation: {e}")
            return None
    
    def _format_algorithm_info(self, info: Dict) -> str:
        """Format algorithm information for prompt."""
        lines = [
            f"Algorithm Name: {info['name']}",
            f"Type: {info['type']}",
            f"Description: {info['description']}",
        ]
        
        if info.get('parameters'):
            lines.append(f"Parameters: {', '.join(info['parameters'])}")
        
        if info.get('pseudocode'):
            lines.append(f"\nPseudocode:\n{info['pseudocode']}")
        
        if info.get('mathematical_notation'):
            lines.append(f"\nMathematical Notation:\n{info['mathematical_notation']}")
        
        return "\n".join(lines)
    
    def _basic_interpretation(self, algorithm: AlgorithmInfo) -> List[Dict]:
        """Basic interpretation without LLM (fallback)."""
        steps = []
        
        # Create basic workflow from algorithm description
        if algorithm.description:
            steps.append({
                "step": 1,
                "description": f"Initialize {algorithm.name}",
                "action": "initialize"
            })
            
            if algorithm.parameters:
                steps.append({
                    "step": 2,
                    "description": f"Set parameters: {', '.join(algorithm.parameters[:3])}",
                    "action": "configure"
                })
            
            steps.append({
                "step": len(steps) + 1,
                "description": f"Execute {algorithm.name} algorithm",
                "action": "execute"
            })
        
        return steps

