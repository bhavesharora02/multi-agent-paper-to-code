"""
Coder Agent: Generates code from specifications.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directories to path for imports
parent_src = Path(__file__).parent.parent.parent.parent / "src"
if parent_src.exists():
    sys.path.insert(0, str(parent_src))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent

# Try to import LLM client from parent project, fallback to local
try:
    from llm.llm_client import LLMClient, LLMProvider
except ImportError:
    # If parent project not available, create a minimal client
    import os
    from enum import Enum
    
    class LLMProvider(Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        OPENROUTER = "openrouter"
        GROQ = "groq"
    
    class LLMClient:
        def __init__(self, provider=None, model=None):
            raise ImportError("LLM client not available. Please ensure parent project is accessible or install LLM dependencies.")


class CoderAgent(BaseAgent):
    """
    Coder Agent generates function-level code from specifications.
    Uses LLM to generate Python code based on natural language descriptions.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Coder Agent."""
        super().__init__(config, llm_client)
        
        # Initialize LLM client if not provided
        if not self.llm_client:
            try:
                provider_str = self.config.get("llm_provider", "groq").lower()
                if provider_str == "openai":
                    provider = LLMProvider.OPENAI
                elif provider_str == "anthropic":
                    provider = LLMProvider.ANTHROPIC
                elif provider_str == "openrouter":
                    provider = LLMProvider.OPENROUTER
                elif provider_str == "groq":
                    provider = LLMProvider.GROQ
                else:
                    provider = LLMProvider.GROQ  # Default
                
                model = self.config.get("model", None)
                self.llm_client = LLMClient(provider=provider, model=model)
                self.log_progress(f"Initialized LLM client: {provider.value}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM client: {e}")
                raise
        
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code from specification.
        
        Args:
            state: Workflow state containing specification
            
        Returns:
            Updated state with generated code
        """
        if not self.validate_input(state):
            return state
        
        specification = state.get("specification", "")
        needs_improvement = state.get("needs_improvement", False)
        previous_rating = state.get("code_rating", 0)
        previous_feedback = state.get("rating_feedback", "")
        
        if not specification:
            self.logger.error("No specification provided")
            return state
        
        # Check if this is an improvement attempt
        if needs_improvement and previous_rating > 0:
            self.log_progress(f"Improving code (previous rating: {previous_rating:.1f}/10)...")
            improvement_context = f"\n\nIMPORTANT: The previous code received a rating of {previous_rating:.1f}/10. "
            improvement_context += f"Feedback: {previous_feedback[:500] if previous_feedback else 'Code needs improvement'}. "
            improvement_context += "Generate significantly improved code that addresses all issues mentioned in the feedback. "
            improvement_context += "Make sure the code is correct, handles all edge cases, follows best practices, and is well-documented."
        else:
            self.log_progress(f"Generating code for: {specification[:100]}...")
            improvement_context = ""
        
        # Generate code using LLM
        system_prompt = """You are an expert Python programmer. Generate clean, well-documented Python code based on specifications.

Requirements:
1. Write complete, runnable Python functions
2. Include proper type hints where appropriate
3. Add docstrings following Google style
4. Handle edge cases appropriately
5. Write idiomatic Python code
6. Follow best practices and ensure correctness

Return only the Python code, no explanations or markdown formatting."""
        
        user_prompt = f"""Generate a Python function based on this specification:

{specification}
{improvement_context}

If the specification includes function signature or examples, follow them exactly.
Generate complete, production-ready code that is correct, well-tested, and follows best practices."""
        
        try:
            generated_code = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Clean up the code (remove markdown code blocks if present)
            code = self._extract_code(generated_code)
            
            self.log_progress(f"Generated {len(code)} characters of code")
            
            # Update state
            updates = {
                "code": code,
                "code_history": state.get("code_history", []) + [code],
                "iteration": state.get("iteration", 0) + 1,
                "last_agent": "coder"
            }
            self.update_state(state, updates)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            state["error"] = str(e)
            state["last_agent"] = "coder"
            return state
    
    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.
        Removes markdown code blocks if present.
        
        Args:
            text: Raw LLM response
            
        Returns:
            Cleaned Python code
        """
        import re
        
        # Remove markdown code blocks
        code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return as-is (might be plain code)
        return text.strip()

