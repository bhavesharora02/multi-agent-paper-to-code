"""
Rater Agent: Analyzes code quality and provides a rating (0-10).
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


class RaterAgent(BaseAgent):
    """
    Rater Agent analyzes code quality and provides a rating from 0-10.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Rater Agent."""
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
        Analyze code and provide rating.
        
        Args:
            state: Workflow state containing code
            
        Returns:
            Updated state with code rating
        """
        if not self.validate_input(state):
            return state
        
        code = state.get("code", "")
        if not code:
            self.logger.error("No code to rate")
            return state
        
        specification = state.get("specification", "")
        
        self.log_progress("Analyzing code quality...")
        
        # Analyze code and get rating
        rating_result = self._rate_code(code, specification)
        
        if rating_result:
            self.log_progress(f"Code rated: {rating_result.get('rating', 0)}/10")
            
            # Update state
            updates = {
                "code_rating": rating_result.get("rating", 0),
                "rating_details": rating_result.get("details", ""),
                "rating_feedback": rating_result.get("feedback", ""),
                "last_agent": "rater"
            }
            self.update_state(state, updates)
        else:
            self.logger.warning("Could not rate code")
            state["code_rating"] = 0
            state["last_agent"] = "rater"
        
        return state
    
    def _rate_code(self, code: str, specification: str) -> Optional[Dict[str, Any]]:
        """
        Analyze code and provide rating.
        
        Args:
            code: The code to rate
            specification: Original specification
            
        Returns:
            Dictionary with rating, details, and feedback
        """
        system_prompt = """You are an expert code reviewer. Analyze Python code and provide a quality rating from 0-10.

Rating Criteria (0-10 scale):
- 0-2: Code has critical errors, doesn't work, or completely wrong
- 3-4: Code has major issues, syntax errors, or doesn't meet requirements
- 5-6: Code works but has issues: missing edge cases, poor style, or incomplete
- 7-8: Good code: works correctly, handles most cases, clean style
- 9-10: Excellent code: perfect implementation, handles all edge cases, excellent style, well-documented

Consider:
1. Correctness: Does it work as specified?
2. Completeness: Does it handle edge cases?
3. Code quality: Style, readability, documentation
4. Best practices: Follows Python conventions
5. Efficiency: Is it reasonably efficient?

Return ONLY a JSON object with this exact format:
{
    "rating": <number 0-10>,
    "details": "<brief explanation of rating>",
    "feedback": "<detailed feedback on what's good and what could be improved>"
}"""
        
        user_prompt = f"""Analyze this Python code and provide a quality rating from 0-10.

Original specification:
{specification}

Code to analyze:
```python
{code}
```

Provide a rating based on:
1. Does it correctly implement the specification?
2. Does it handle edge cases?
3. Code quality and style
4. Documentation and comments
5. Best practices

Return ONLY valid JSON with rating, details, and feedback."""
        
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"rating"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    # Ensure rating is between 0-10
                    rating = max(0, min(10, float(result.get("rating", 0))))
                    return {
                        "rating": round(rating, 1),
                        "details": result.get("details", ""),
                        "feedback": result.get("feedback", "")
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract rating number
            rating_match = re.search(r'rating["\']?\s*:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if rating_match:
                rating = max(0, min(10, float(rating_match.group(1))))
                return {
                    "rating": round(rating, 1),
                    "details": "Code analyzed",
                    "feedback": response[:500]  # Use first 500 chars as feedback
                }
            
            # Last resort: default rating
            self.logger.warning("Could not parse rating from response")
            return {
                "rating": 5.0,
                "details": "Unable to parse rating",
                "feedback": response[:500]
            }
            
        except Exception as e:
            self.logger.error(f"Error rating code: {e}")
            return None

