"""
Debugger Agent: Analyzes errors and suggests fixes.
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


class DebuggerAgent(BaseAgent):
    """
    Debugger Agent analyzes test failures and suggests code fixes.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Debugger Agent."""
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
        
        self.temperature = self.config.get("temperature", 0.5)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze errors and suggest fixes.
        
        Args:
            state: Workflow state containing code, tests, and errors
            
        Returns:
            Updated state with fixed code
        """
        if not self.validate_input(state):
            return state
        
        code = state.get("code", "")
        test_results = state.get("test_results", {})
        test_output = state.get("test_output", "")
        test_errors = state.get("test_errors", "")
        specification = state.get("specification", "")
        
        if not code:
            self.logger.error("No code to debug")
            return state
        
        if test_results.get("all_passed", False):
            self.log_progress("No errors to debug - tests passed")
            return state
        
        self.log_progress("Analyzing errors and generating fixes...")
        
        # Analyze errors and generate fix
        fixed_code = self._analyze_and_fix(
            code, test_output, test_errors, specification
        )
        
        if fixed_code:
            self.log_progress("Generated fixed code")
            
            # Update state
            updates = {
                "code": fixed_code,
                "code_history": state.get("code_history", []) + [fixed_code],
                "fix_history": state.get("fix_history", []) + [{
                    "iteration": state.get("iteration", 0),
                    "error": test_errors,
                    "fix": fixed_code
                }],
                "last_agent": "debugger"
            }
            self.update_state(state, updates)
        else:
            self.logger.warning("Could not generate fix")
            state["error"] = "Failed to generate fix"
            state["last_agent"] = "debugger"
        
        return state
    
    def _analyze_and_fix(
        self, 
        code: str, 
        test_output: str, 
        test_errors: str,
        specification: str
    ) -> Optional[str]:
        """
        Analyze errors and generate fixed code.
        
        Args:
            code: Current code with errors
            test_output: Test execution output
            test_errors: Error messages
            specification: Original specification
            
        Returns:
            Fixed code or None if fix generation failed
        """
        # Extract key error information
        error_summary = self._extract_error_summary(test_output, test_errors)
        
        system_prompt = """You are an expert debugger. Analyze test failures and fix code errors.

CRITICAL REQUIREMENTS:
1. Carefully read error messages and tracebacks
2. Identify the root cause of failures (syntax errors, logic errors, import issues, etc.)
3. Generate fixed code that addresses ALL issues
4. Maintain the original function signature and behavior
5. Preserve code style and documentation
6. Make sure the fixed code will pass the tests
7. Return ONLY the complete fixed Python code, no explanations or markdown"""
        
        user_prompt = f"""The following code has test failures. Analyze the errors and provide a fixed version.

Original specification:
{specification}

Current code (with errors):
```python
{code}
```

Test execution output:
{test_output}

Error messages:
{test_errors}

Error summary:
{error_summary}

IMPORTANT:
- Fix ALL errors mentioned in the test output
- Ensure the function signature matches what the tests expect
- Make sure imports are correct
- Fix any syntax errors
- Fix any logic errors that cause test failures
- The fixed code must pass all the tests

Provide the complete fixed Python code that will pass all tests. Return ONLY the code, no explanations."""
        
        try:
            fixed_code = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract code from markdown if present
            fixed_code = self._extract_code(fixed_code)
            
            # Clean the fixed code
            fixed_code = self._clean_fixed_code(fixed_code)
            
            self.logger.info(f"Generated fixed code ({len(fixed_code)} chars)")
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"Error generating fix: {e}")
            return None
    
    def _extract_error_summary(self, test_output: str, test_errors: str) -> str:
        """Extract key error information from test output."""
        import re
        
        errors = []
        
        # Look for common error patterns
        if "SyntaxError" in test_output or "SyntaxError" in test_errors:
            errors.append("Syntax error detected")
        
        if "NameError" in test_output or "NameError" in test_errors:
            errors.append("Name error - possibly undefined variable or import issue")
        
        if "AttributeError" in test_output or "AttributeError" in test_errors:
            errors.append("Attribute error - object doesn't have the expected attribute")
        
        if "TypeError" in test_output or "TypeError" in test_errors:
            errors.append("Type error - wrong type used")
        
        if "AssertionError" in test_output or "AssertionError" in test_errors:
            errors.append("Assertion failed - test expectation not met")
        
        if "ImportError" in test_output or "ImportError" in test_errors:
            errors.append("Import error - cannot import function or module")
        
        # Extract specific error lines
        error_lines = []
        for line in (test_output + test_errors).split('\n'):
            if any(keyword in line for keyword in ['Error:', 'FAILED', 'AssertionError', 'SyntaxError']):
                error_lines.append(line.strip())
        
        summary = "\n".join(errors) if errors else "Unknown error"
        if error_lines:
            summary += "\n\nKey error lines:\n" + "\n".join(error_lines[:5])
        
        return summary
    
    def _clean_fixed_code(self, code: str) -> str:
        """Clean fixed code."""
        # Remove markdown
        code = self._extract_code(code)
        
        # Remove REPL prompts
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('>>>'):
                cleaned_lines.append(line.replace('>>>', '').strip())
            elif line.strip().startswith('...'):
                cleaned_lines.append(line.replace('...', '').strip())
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

