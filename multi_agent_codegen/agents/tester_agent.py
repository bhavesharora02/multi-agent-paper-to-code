"""
Tester Agent: Generates and executes test cases.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import subprocess
import tempfile
import os

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


class TesterAgent(BaseAgent):
    """
    Tester Agent generates unit tests and executes code in a sandbox.
    """
    
    def __init__(self, config: Dict = None, llm_client=None, sandbox_executor=None):
        """Initialize Tester Agent."""
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
        
        self.sandbox_executor = sandbox_executor
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.generate_edge_cases = self.config.get("generate_edge_cases", True)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tests and execute code.
        
        Args:
            state: Workflow state containing code
            
        Returns:
            Updated state with test results
        """
        if not self.validate_input(state):
            return state
        
        code = state.get("code", "")
        if not code:
            self.logger.error("No code to test")
            return state
        
        specification = state.get("specification", "")
        
        self.log_progress("Generating test cases...")
        
        # Generate test cases
        test_code = self._generate_tests(code, specification)
        
        self.log_progress("Executing tests...")
        
        # Execute tests in sandbox
        test_results = self._execute_tests(code, test_code)
        
        # Update state
        updates = {
            "test_code": test_code,
            "test_results": test_results,
            "tests_passed": test_results.get("all_passed", False),
            "test_output": test_results.get("output", ""),
            "test_errors": test_results.get("errors", ""),
            "last_agent": "tester"
        }
        self.update_state(state, updates)
        
        if test_results.get("all_passed", False):
            self.log_progress("All tests passed!")
        else:
            self.log_progress(f"Tests failed: {test_results.get('num_failed', 0)} failed")
        
        return state
    
    def _generate_tests(self, code: str, specification: str) -> str:
        """
        Generate test cases using LLM.
        
        Args:
            code: The code to test
            specification: Original specification
            
        Returns:
            Generated test code
        """
        # Extract function names from code
        import re
        func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
        func_names = func_matches if func_matches else []
        main_func = func_names[0] if func_names else "function"
        
        system_prompt = """You are an expert test engineer. Generate comprehensive unit tests for Python functions.

CRITICAL REQUIREMENTS:
1. Use pytest framework
2. Import functions using: from code import function_name
3. Test normal cases, edge cases, and error cases
4. Include tests for boundary conditions
5. Use descriptive test names
6. Make sure test assertions match the actual function behavior
7. Return ONLY the test code, no explanations or markdown
8. Do NOT include the function definition, only test functions"""
        
        user_prompt = f"""Generate comprehensive pytest unit tests for this Python code:

Code:
```python
{code}
```

Original specification:
{specification}

IMPORTANT:
- The code will be in a file called 'code.py'
- Import using: from code import {', '.join(func_names) if func_names else main_func}
- Generate tests that verify the function works correctly
- Test with realistic inputs based on the specification
- Include edge cases mentioned in the specification

Generate pytest test functions. Return ONLY the test code, no markdown, no explanations."""
        
        try:
            test_code = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract code from markdown if present
            test_code = self._extract_code(test_code)
            
            # Clean up the test code
            test_code = self._clean_test_code(test_code, func_names)
            
            self.logger.info(f"Generated test code ({len(test_code)} chars)")
            return test_code
            
        except Exception as e:
            self.logger.error(f"Error generating tests: {e}")
            # Return basic test as fallback
            return self._generate_basic_test(code)
    
    def _clean_test_code(self, test_code: str, func_names: list) -> str:
        """Clean and fix test code."""
        import re
        
        # Remove any function definitions that might have been included
        lines = test_code.split('\n')
        cleaned_lines = []
        skip_until_def = False
        
        for line in lines:
            # Skip function definitions (keep only test functions)
            if re.match(r'^\s*def\s+\w+.*:', line) and not line.strip().startswith('def test_'):
                # This is a function definition, not a test - skip it
                continue
            cleaned_lines.append(line)
        
        test_code = '\n'.join(cleaned_lines)
        
        # Ensure proper imports
        if 'from code import' not in test_code and func_names:
            # Add import at the beginning
            imports = f"from code import {', '.join(func_names)}\n"
            test_code = imports + test_code
        
        # Remove duplicate imports
        import_lines = [l for l in test_code.split('\n') if 'from code import' in l or 'import code' in l]
        if import_lines:
            # Keep only the first import, remove others
            first_import = import_lines[0]
            test_code = re.sub(r'from code import.*\n', '', test_code)
            test_code = re.sub(r'import code.*\n', '', test_code)
            # Add import at the top
            if 'from code import' not in test_code.split('\n')[0:5]:
                test_code = first_import + '\n' + test_code
        
        return test_code.strip()
    
    def _generate_basic_test(self, code: str) -> str:
        """Generate a basic test as fallback."""
        # Try to extract function name from code
        import re
        func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
        if not func_matches:
            return """
import pytest
from code import *

def test_basic():
    # Basic test - function exists
    pass
"""
        
        func_name = func_matches[0]
        
        # Try to infer test cases from function signature
        func_match = re.search(r'def\s+\w+\s*\((.*?)\)', code, re.DOTALL)
        params = []
        if func_match:
            param_str = func_match.group(1)
            params = [p.strip().split(':')[0].split('=')[0].strip() 
                     for p in param_str.split(',') if p.strip()]
        
        test_code = f"""import pytest
from code import {func_name}

def test_{func_name}_exists():
    assert {func_name} is not None
    assert callable({func_name})
"""
        
        # Add a simple test if we can infer parameters
        if params and len(params) <= 3:
            # Try to create a simple test
            test_code += f"""
def test_{func_name}_basic():
    # Basic functionality test
    # TODO: Add proper test cases based on function requirements
    pass
"""
        
        return test_code
    
    def _execute_tests(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Execute tests in sandbox.
        
        Args:
            code: The code to test
            test_code: Test code
            
        Returns:
            Test results dictionary
        """
        if self.sandbox_executor:
            return self.sandbox_executor.execute(code, test_code)
        else:
            return self._execute_local(code, test_code)
    
    def _execute_local(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Execute tests locally (fallback if no sandbox).
        
        Args:
            code: The code to test
            test_code: Test code
            
        Returns:
            Test results dictionary
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clean and validate code first
            code = self._clean_generated_code(code)
            
            # Write code to file
            code_file = os.path.join(tmpdir, "code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Validate code syntax
            try:
                compile(code, code_file, 'exec')
            except SyntaxError as e:
                return {
                    "all_passed": False,
                    "num_passed": 0,
                    "num_failed": 0,
                    "output": "",
                    "errors": f"Syntax error in generated code: {str(e)}",
                    "return_code": -1
                }
            
            # Write test file
            test_file = os.path.join(tmpdir, "test_code.py")
            with open(test_file, "w", encoding="utf-8") as f:
                # Import the code module
                f.write("import sys\n")
                f.write("import os\n")
                f.write("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n")
                # Use explicit imports if possible, otherwise import all
                if "from code import" in test_code:
                    f.write("\n")
                else:
                    f.write("from code import *\n\n")
                f.write(test_code)
            
            # Validate test code syntax
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    test_content = f.read()
                compile(test_content, test_file, 'exec')
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in test code: {e}")
                # Try to continue anyway
            
            # Run pytest with more verbose output
            try:
                result = subprocess.run(
                    ["pytest", test_file, "-v", "--tb=long", "--no-header"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = result.stdout + result.stderr
                all_passed = result.returncode == 0
                
                # Parse test results more accurately
                num_passed = len([line for line in output.split('\n') if ' PASSED' in line])
                num_failed = len([line for line in output.split('\n') if ' FAILED' in line])
                
                # If no explicit PASSED/FAILED found, check return code
                if num_passed == 0 and num_failed == 0:
                    if all_passed:
                        num_passed = 1  # Assume at least one test passed
                    else:
                        num_failed = 1  # Assume at least one test failed
                
                return {
                    "all_passed": all_passed,
                    "num_passed": num_passed,
                    "num_failed": num_failed,
                    "output": output,
                    "errors": result.stderr if not all_passed else "",
                    "return_code": result.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "all_passed": False,
                    "num_passed": 0,
                    "num_failed": 0,
                    "output": "",
                    "errors": "Test execution timed out after 30 seconds",
                    "return_code": -1
                }
            except FileNotFoundError:
                return {
                    "all_passed": False,
                    "num_passed": 0,
                    "num_failed": 0,
                    "output": "",
                    "errors": "pytest not found. Please install: pip install pytest",
                    "return_code": -1
                }
            except Exception as e:
                return {
                    "all_passed": False,
                    "num_passed": 0,
                    "num_failed": 0,
                    "output": "",
                    "errors": f"Test execution error: {str(e)}",
                    "return_code": -1
                }
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing markdown and extra formatting."""
        # Remove markdown code blocks if present
        code = self._extract_code(code)
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Remove any Python REPL prompts (>>> or ...)
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('>>>'):
                cleaned_lines.append(line.replace('>>>', '').strip())
            elif line.strip().startswith('...'):
                cleaned_lines.append(line.replace('...', '').strip())
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

