"""
Optimizer/Explainer Agent: Optimizes code and adds documentation.
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

# Try to import LLM client from parent project
try:
    from llm.llm_client import LLMClient, LLMProvider
except ImportError:
    from enum import Enum
    class LLMProvider(Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        OPENROUTER = "openrouter"
        GROQ = "groq"
    class LLMClient:
        def __init__(self, provider=None, model=None):
            raise ImportError("LLM client not available")


class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent reviews code for efficiency and adds documentation.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Optimizer Agent."""
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
                    provider = LLMProvider.GROQ
                
                model = self.config.get("model", None)
                self.llm_client = LLMClient(provider=provider, model=model)
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM client: {e}")
                raise
        
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 1024)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize code and add documentation.
        
        Args:
            state: Workflow state containing code
            
        Returns:
            Updated state with optimized code
        """
        if not self.validate_input(state):
            return state
        
        code = state.get("code", "")
        if not code:
            self.logger.error("No code to optimize")
            return state
        
        specification = state.get("specification", "")
        tests_passed = state.get("tests_passed", False)
        
        if not tests_passed:
            self.log_progress("Skipping optimization - tests not passed yet")
            return state
        
        self.log_progress("Optimizing code and adding documentation...")
        
        # Optimize and document code
        optimized_code = self._optimize_and_document(code, specification)
        
        if optimized_code:
            self.log_progress("Code optimized and documented")
            
            # Update state
            updates = {
                "code": optimized_code,
                "code_history": state.get("code_history", []) + [optimized_code],
                "optimized": True,
                "last_agent": "optimizer"
            }
            self.update_state(state, updates)
        else:
            self.logger.warning("Could not optimize code")
            state["last_agent"] = "optimizer"
        
        return state
    
    def _optimize_and_document(self, code: str, specification: str) -> Optional[str]:
        """
        Optimize code and add documentation.
        
        Args:
            code: Current code
            specification: Original specification
            
        Returns:
            Optimized and documented code
        """
        system_prompt = """You are an expert code reviewer and optimizer. Review code for efficiency and add comprehensive documentation.

Requirements:
1. Optimize code for performance if possible
2. Add comprehensive docstrings following Google style
3. Add inline comments for complex logic
4. Ensure code follows Python best practices
5. Maintain all functionality and test compatibility
6. Return only the optimized Python code"""
        
        user_prompt = f"""Review and optimize this code. Add comprehensive documentation and improve efficiency if possible.

Original specification:
{specification}

Code to optimize:
```python
{code}
```

Please:
1. Add detailed docstrings with parameter descriptions, return types, and examples
2. Add inline comments for complex logic
3. Optimize for performance if opportunities exist
4. Ensure code style follows PEP 8

Return only the optimized Python code."""
        
        try:
            optimized_code = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract code from markdown if present
            optimized_code = self._extract_code(optimized_code)
            
            return optimized_code
            
        except Exception as e:
            self.logger.error(f"Error optimizing code: {e}")
            return None
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

