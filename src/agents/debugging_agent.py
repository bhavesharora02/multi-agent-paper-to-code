"""
Debugging Agent.
Analyzes failures and iteratively refines code.
"""

import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import (
    PaperToCodeIR, VerificationResult, RefinementHistory, GeneratedFile
)
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import DEBUGGING_SYSTEM_PROMPT, DEBUGGING_PROMPT
from datetime import datetime


class DebuggingAgent(BaseAgent):
    """
    Debugging Agent that analyzes failures and refines code iteratively.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Debugging Agent."""
        super().__init__(config, llm_client)
        
        # Debugging settings
        self.max_iterations = self.config.get('max_iterations', 3)
        self.auto_fix = self.config.get('auto_fix', True)
        
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
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Analyze code and refine it iteratively.
        Works independently - can analyze code statically even without verification results.
        
        Args:
            ir: Intermediate representation with generated code
            
        Returns:
            Updated IR with refinements
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        # Check if we have generated code to debug
        if not ir.generated_code or len(ir.generated_code) == 0:
            self.log_progress("No generated code to debug", "info")
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress("Starting code analysis and debugging...")
        
        try:
            # First, perform static analysis
            static_issues = self._static_code_analysis(ir)
            
            # If we have verification results, use them too
            has_verification_issues = (
                ir.verification_results and 
                ir.verification_results.status != "pass"
            )
            
            if not static_issues and not has_verification_issues:
                self.log_progress("No issues found in code", "info")
                return ir
            
            iteration = 0
            issues_found = True
            
            while iteration < self.max_iterations and issues_found:
                iteration += 1
                self.log_progress(f"Debugging iteration {iteration}/{self.max_iterations}")
                
                # Analyze issues (static + verification if available)
                analysis = self._analyze_issues(ir, static_issues)
                
                if not analysis or not analysis.get("suggested_fixes"):
                    self.log_progress("No fixes suggested, stopping", "warning")
                    break
                
                # Apply fixes if auto_fix is enabled
                if self.auto_fix:
                    self.log_progress("Applying suggested fixes...")
                    ir, fixes_applied = self._apply_fixes(ir, analysis)
                    
                    # Record refinement history
                    refinement = RefinementHistory(
                        iteration=iteration,
                        timestamp=datetime.now().isoformat(),
                        changes=analysis.get("suggested_fixes", []),
                        verification_result=ir.verification_results if hasattr(ir, 'verification_results') else None,
                        success=fixes_applied
                    )
                    if not hasattr(ir, 'refinement_history') or ir.refinement_history is None:
                        ir.refinement_history = []
                    ir.refinement_history.append(refinement)
                    
                    self.log_progress(f"Applied {len(analysis.get('suggested_fixes', []))} fixes")
                    
                    # Re-analyze to check if issues are resolved
                    static_issues = self._static_code_analysis(ir)
                    issues_found = len(static_issues) > 0
                else:
                    # Just record suggestions
                    self.log_progress("Auto-fix disabled, suggestions recorded only")
                    refinement = RefinementHistory(
                        iteration=iteration,
                        timestamp=datetime.now().isoformat(),
                        changes=analysis.get("suggested_fixes", []),
                        verification_result=ir.verification_results if hasattr(ir, 'verification_results') else None,
                        success=False
                    )
                    if not hasattr(ir, 'refinement_history') or ir.refinement_history is None:
                        ir.refinement_history = []
                    ir.refinement_history.append(refinement)
                    issues_found = False  # Stop after first iteration if auto-fix is disabled
                
                # Check if we should continue
                if analysis.get("severity") == "low" or not analysis.get("suggested_fixes"):
                    break
            
            self.update_ir_status(ir, "refined")
            self.log_progress(f"Debugging completed after {iteration} iteration(s)")
            
        except Exception as e:
            self.logger.error(f"Error in debugging: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            if not hasattr(ir, 'extracted_content'):
                ir.extracted_content = {}
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def _static_code_analysis(self, ir: PaperToCodeIR) -> List[Dict]:
        """Perform static code analysis to find syntax and logical errors."""
        issues = []
        
        if not ir.generated_code:
            return issues
        
        for file_path, file_info in ir.generated_code.items():
            if not file_path.endswith(".py"):
                continue
            
            code = file_info.content if isinstance(file_info, GeneratedFile) else str(file_info)
            
            # Check for syntax errors
            syntax_errors = self._check_syntax(code, file_path)
            issues.extend(syntax_errors)
            
            # Check for common logical errors
            logical_errors = self._check_logical_errors(code, file_path)
            issues.extend(logical_errors)
            
            # Check for best practices
            best_practice_issues = self._check_best_practices(code, file_path)
            issues.extend(best_practice_issues)
        
        return issues
    
    def _check_syntax(self, code: str, file_path: str) -> List[Dict]:
        """Check for Python syntax errors."""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "severity": "critical",
                "file": file_path,
                "line": e.lineno,
                "message": f"Syntax error: {e.msg}",
                "error": str(e)
            })
        except Exception as e:
            issues.append({
                "type": "parse_error",
                "severity": "critical",
                "file": file_path,
                "message": f"Parse error: {str(e)}",
                "error": str(e)
            })
        return issues
    
    def _check_logical_errors(self, code: str, file_path: str) -> List[Dict]:
        """Check for common logical errors in ML/DL code."""
        issues = []
        
        # Check for common indexing errors
        if re.search(r'\[(\d+)\].*\[(\d+)\].*\[(\d+)\]', code):
            # Check for suspicious indexing patterns like q[0], k[1], v[2]
            matches = re.finditer(r'(\w+)\[0\].*(\w+)\[1\].*(\w+)\[2\]', code)
            for match in matches:
                if match.group(1) == match.group(2) or match.group(2) == match.group(3):
                    # This might be an error - same variable indexed differently
                    issues.append({
                        "type": "logical_error",
                        "severity": "high",
                        "file": file_path,
                        "message": f"Potential indexing error: {match.group(0)} - indices should likely be the same",
                        "code_snippet": match.group(0)
                    })
        
        # Check for undefined variables in forward pass
        if 'def forward' in code:
            # Look for potential undefined variables
            forward_match = re.search(r'def forward\([^)]*\):.*?(?=\n\s*(?:def |class |$))', code, re.DOTALL)
            if forward_match:
                forward_code = forward_match.group(0)
                # Check for variables used but not defined
                var_pattern = r'\b([a-z_][a-z0-9_]*)\b'
                used_vars = set(re.findall(var_pattern, forward_code))
                defined_vars = set(re.findall(r'\b(self\.)?([a-z_][a-z0-9_]*)\s*=', forward_code))
                # This is a simplified check - full implementation would be more sophisticated
        
        return issues
    
    def _check_best_practices(self, code: str, file_path: str) -> List[Dict]:
        """Check for best practice violations."""
        issues = []
        
        # Check for missing docstrings in classes
        class_pattern = r'class\s+(\w+).*?:\s*\n(?!\s+""")'
        if re.search(class_pattern, code):
            issues.append({
                "type": "best_practice",
                "severity": "low",
                "file": file_path,
                "message": "Some classes may be missing docstrings"
            })
        
        # Check for hardcoded values that should be parameters
        if re.search(r'batch_size\s*=\s*\d+', code) and 'batch_size' not in code[:200]:
            issues.append({
                "type": "best_practice",
                "severity": "medium",
                "file": file_path,
                "message": "Hardcoded batch_size found - consider making it a parameter"
            })
        
        return issues
    
    def _analyze_issues(self, ir: PaperToCodeIR, static_issues: List[Dict]) -> Optional[Dict]:
        """Analyze code issues using LLM."""
        if not self.llm_client:
            return None
        
        try:
            # Get code context
            code_context = ""
            main_file_path = None
            if ir.generated_code:
                # Get main code file
                for path, file_info in ir.generated_code.items():
                    if path.endswith(".py") and ("main" in path.lower() or "model" in path.lower()):
                        code_context = file_info.content if isinstance(file_info, GeneratedFile) else str(file_info)
                        main_file_path = path
                        # Limit to reasonable size (first 3000 chars for context, full code for analysis)
                        break
            
            if not code_context:
                # Fallback to any Python file
                for path, file_info in ir.generated_code.items():
                    if path.endswith(".py"):
                        code_context = file_info.content if isinstance(file_info, GeneratedFile) else str(file_info)
                        main_file_path = path
                        break
            
            if not code_context:
                return None
            
            # Prepare error information
            error_info = {
                "static_issues": static_issues,
                "verification_errors": [],
                "discrepancies": []
            }
            
            # Add verification results if available
            if ir.verification_results:
                error_info["verification_errors"] = ir.verification_results.errors or []
                error_info["discrepancies"] = ir.verification_results.discrepancies or []
            
            # Format prompt
            prompt = f"""Analyze the following Python code and identify issues that need to be fixed.

Code File: {main_file_path}

Code:
```python
{code_context[:4000]}  # Limit to 4000 chars for prompt
```

Issues Found:
{self._format_issues(static_issues, error_info)}

Please provide:
1. **root_cause**: Likely cause of the main issues
2. **severity**: "critical", "high", "medium", or "low"
3. **suggested_fixes**: List of specific fixes to try
4. **code_changes**: Complete corrected code sections (if code needs to be replaced) or specific modifications
5. **testing_approach**: How to verify the fix works

Return as JSON with the following structure:
{{
    "root_cause": "...",
    "severity": "...",
    "suggested_fixes": ["fix1", "fix2", ...],
    "code_changes": "corrected code or specific modifications",
    "testing_approach": "..."
}}"""
            
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=DEBUGGING_SYSTEM_PROMPT,
                temperature=0.2  # Lower temperature for more deterministic fixes
            )
            
            return response if isinstance(response, dict) and not response.get("_is_error") else None
            
        except Exception as e:
            self.logger.error(f"Error in issue analysis: {e}")
            return None
    
    def _format_issues(self, static_issues: List[Dict], error_info: Dict) -> str:
        """Format issues for prompt."""
        lines = []
        
        if static_issues:
            lines.append(f"Static Analysis Issues ({len(static_issues)}):")
            for issue in static_issues[:10]:  # Limit to 10
                lines.append(f"  - [{issue.get('severity', 'unknown')}] {issue.get('message', 'Unknown issue')}")
                if issue.get('line'):
                    lines.append(f"    Line {issue['line']}: {issue.get('code_snippet', '')}")
        
        if error_info.get("verification_errors"):
            lines.append(f"\nVerification Errors ({len(error_info['verification_errors'])}):")
            for error in error_info["verification_errors"][:5]:
                lines.append(f"  - {error[:200]}")
        
        if error_info.get("discrepancies"):
            lines.append(f"\nDiscrepancies ({len(error_info['discrepancies'])}):")
            for disc in error_info["discrepancies"][:5]:
                lines.append(f"  - {disc}")
        
        return "\n".join(lines) if lines else "No specific issues found, but code should be reviewed for best practices."
    
    
    def _apply_fixes(self, ir: PaperToCodeIR, analysis: Dict) -> Tuple[PaperToCodeIR, bool]:
        """Apply suggested fixes to generated code.
        
        Returns:
            Tuple of (updated IR, success boolean)
        """
        fixes = analysis.get("suggested_fixes", [])
        code_changes = analysis.get("code_changes", "")
        
        if not fixes and not code_changes:
            return ir, False
        
        fixes_applied = False
        
        # Try to apply code changes
        if code_changes:
            # Find the main code file to update
            main_file_path = None
            for path in ir.generated_code.keys():
                if path.endswith(".py") and ("main" in path.lower() or "model" in path.lower()):
                    main_file_path = path
                    break
            
            if not main_file_path:
                # Fallback to first Python file
                for path in ir.generated_code.keys():
                    if path.endswith(".py"):
                        main_file_path = path
                        break
            
            if main_file_path and main_file_path in ir.generated_code:
                file_info = ir.generated_code[main_file_path]
                
                # If code_changes contains complete code, replace it
                if code_changes.strip().startswith("```python") or code_changes.strip().startswith("```"):
                    # Extract code from markdown block
                    code_match = re.search(r'```python\s*(.*?)\s*```', code_changes, re.DOTALL)
                    if code_match:
                        new_code = code_match.group(1)
                    else:
                        code_match = re.search(r'```\s*(.*?)\s*```', code_changes, re.DOTALL)
                        new_code = code_match.group(1) if code_match else code_changes
                else:
                    # Try to apply specific fixes
                    old_code = file_info.content if isinstance(file_info, GeneratedFile) else str(file_info)
                    new_code = self._apply_specific_fixes(old_code, fixes, code_changes)
                
                # Update the file
                if isinstance(file_info, GeneratedFile):
                    file_info.content = new_code
                else:
                    ir.generated_code[main_file_path] = GeneratedFile(
                        path=main_file_path,
                        content=new_code,
                        file_type="model"
                    )
                
                fixes_applied = True
                self.log_progress(f"Updated code in {main_file_path}")
        
        # Store fixes in IR for manual review
        if not hasattr(ir, 'extracted_content') or ir.extracted_content is None:
            ir.extracted_content = {}
        
        if "fixes_applied" not in ir.extracted_content:
            ir.extracted_content["fixes_applied"] = []
        
        ir.extracted_content["fixes_applied"].append({
            "iteration": len(ir.refinement_history) + 1 if hasattr(ir, 'refinement_history') and ir.refinement_history else 1,
            "fixes": fixes,
            "code_changes": code_changes[:500] if code_changes else ""  # Store summary
        })
        
        return ir, fixes_applied
    
    def _apply_specific_fixes(self, old_code: str, fixes: List[str], code_changes: str) -> str:
        """Apply specific fixes to code."""
        new_code = old_code
        
        # Try to apply fixes based on patterns
        for fix in fixes:
            # Fix common indexing errors like q[0], k[1], v[2]
            if "indexing" in fix.lower() or "index" in fix.lower():
                # Pattern: q[0], k[1], v[2] should be q[0], k[0], v[0]
                pattern = r'(\w+)\[0\].*?(\w+)\[1\].*?(\w+)\[2\]'
                replacement = r'\1[0], \2[0], \3[0]'
                new_code = re.sub(pattern, replacement, new_code)
            
            # Fix undefined variable errors
            if "undefined" in fix.lower() or "not defined" in fix.lower():
                # This would require more sophisticated analysis
                pass
        
        # If code_changes contains specific replacements, try to apply them
        if code_changes and "replace" in code_changes.lower():
            # Try to extract old and new code blocks
            # This is a simplified implementation
            pass
        
        # If we have complete new code, use it
        if code_changes and len(code_changes) > len(new_code) * 0.5:
            # If code_changes is substantial, it might be a complete replacement
            if "def " in code_changes and "class " in code_changes:
                # Looks like complete code
                code_match = re.search(r'```python\s*(.*?)\s*```', code_changes, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                elif code_changes.strip().startswith("import") or code_changes.strip().startswith("class") or code_changes.strip().startswith("def"):
                    return code_changes
        
        return new_code

