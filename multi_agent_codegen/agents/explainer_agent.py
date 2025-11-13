"""
Explainer Agent: Answers questions about the generated code.
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


class ExplainerAgent(BaseAgent):
    """
    Explainer Agent answers questions about the generated code.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Explainer Agent."""
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
        
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method required by BaseAgent (not used in workflow).
        ExplainerAgent is only used via explain_code() method for Q&A.
        
        Args:
            state: Workflow state
            
        Returns:
            Unchanged state (this agent doesn't modify workflow state)
        """
        # This agent is not part of the workflow, only used for Q&A
        # So we just return the state unchanged
        return state
    
    def explain_code(self, code: str, question: str, specification: str = "") -> str:
        """
        Answer a question about the code.
        
        Args:
            code: The code to explain
            question: User's question
            specification: Original specification (optional)
            
        Returns:
            Answer to the question
        """
        system_prompt = """You are an expert code explainer. Answer questions about Python code clearly and helpfully.

Requirements:
1. Be clear and concise
2. Use examples when helpful
3. Explain code logic, not just syntax
4. If asked about specific parts, reference line numbers or function names
5. Be helpful and educational"""
        
        user_prompt = f"""Answer this question about the following Python code:

Code:
```python
{code}
```

Original specification:
{specification}

Question: {question}

Provide a clear, helpful answer. If the question is about a specific part of the code, reference it specifically."""
        
        try:
            answer = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"Error explaining code: {e}")
            return f"Sorry, I encountered an error while answering your question: {str(e)}"

