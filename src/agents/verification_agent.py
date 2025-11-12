"""
Verification Agent.
Executes generated code and compares results with paper metrics.
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import (
    PaperToCodeIR, VerificationResult
)
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_PROMPT


class VerificationAgent(BaseAgent):
    """
    Verifies generated code by executing it and comparing results.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Verification Agent."""
        super().__init__(config, llm_client)
        
        # Verification settings
        self.execute_code = self.config.get('execute_code', True)
        self.tolerance = self.config.get('tolerance', 0.02)  # 2% tolerance
        self.timeout = self.config.get('timeout', 300)  # 5 minutes
        
        # Initialize LLM for metric comparison
        if not self.llm_client and self.config.get('use_llm_comparison', True):
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
        Verify generated code.
        
        Args:
            ir: Intermediate representation with generated code
            
        Returns:
            Updated IR with verification results
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        if not ir.generated_code:
            self.log_progress("No generated code to verify", "warning")
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress("Starting code verification...")
        
        try:
            # Extract paper metrics (if available)
            paper_metrics = self._extract_paper_metrics(ir)
            
            # Execute code (if enabled)
            actual_results = {}
            if self.execute_code:
                self.log_progress("Executing generated code...")
                actual_results = self._execute_code(ir)
            else:
                self.log_progress("Code execution disabled, skipping...")
                actual_results = {"status": "skipped"}
            
            # Compare results
            self.log_progress("Comparing results with paper metrics...")
            comparison = self._compare_results(paper_metrics, actual_results)
            
            # Create verification result
            verification_result = VerificationResult(
                status=comparison.get("status", "unknown"),
                metrics=actual_results.get("metrics", {}),
                paper_metrics=paper_metrics,
                comparison=comparison,
                discrepancies=comparison.get("discrepancies", []),
                errors=actual_results.get("errors", []),
                tolerance_check=comparison.get("tolerance_check", False)
            )
            
            ir.verification_results = verification_result
            
            # Update IR status based on verification
            if verification_result.status == "pass":
                self.update_ir_status(ir, "completed")
                self.log_progress("Verification passed!")
            else:
                self.update_ir_status(ir, "needs_refinement")
                self.log_progress(f"Verification {verification_result.status}")
            
        except Exception as e:
            self.logger.error(f"Error in verification: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
            
            # Create failed verification result
            ir.verification_results = VerificationResult(
                status="error",
                errors=[str(e)]
            )
        
        return ir
    
    def _extract_paper_metrics(self, ir: PaperToCodeIR) -> Dict[str, float]:
        """Extract metrics reported in the paper."""
        metrics = {}
        
        # Try to extract from paper text
        if ir.extracted_content.get('full_text'):
            text = ir.extracted_content['full_text']
            
            # Look for common metric patterns
            import re
            patterns = {
                'accuracy': r'accuracy[:\s]+([\d.]+)',
                'loss': r'loss[:\s]+([\d.]+)',
                'f1': r'f1[-\s]?score[:\s]+([\d.]+)',
                'precision': r'precision[:\s]+([\d.]+)',
                'recall': r'recall[:\s]+([\d.]+)',
            }
            
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        metrics[metric_name] = float(matches[-1])  # Take last occurrence
                    except ValueError:
                        pass
        
        return metrics
    
    def _execute_code(self, ir: PaperToCodeIR) -> Dict:
        """Execute generated code in isolated environment."""
        results = {
            "status": "unknown",
            "metrics": {},
            "errors": [],
            "output": ""
        }
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write code files
                for file_path, file_info in ir.generated_code.items():
                    # Skip non-Python files for execution (but keep them for reference)
                    if not file_path.endswith(('.py', '.txt')):
                        continue
                    
                    full_path = os.path.join(tmpdir, file_path)
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else tmpdir, exist_ok=True)
                    
                    # Get content (handle both GeneratedFile objects and strings)
                    content = file_info.content if hasattr(file_info, 'content') else str(file_info)
                    
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # Write requirements if available (but don't install - too risky)
                # Note: We skip installing requirements for safety in verification
                # In production, you might want to use a virtual environment
                if "requirements.txt" in ir.generated_code:
                    req_path = os.path.join(tmpdir, "requirements.txt")
                    req_content = ir.generated_code["requirements.txt"]
                    req_text = req_content.content if hasattr(req_content, 'content') else str(req_content)
                    with open(req_path, 'w') as f:
                        f.write(req_text)
                    self.log_progress("Requirements.txt written (not installed for safety)")
                
                # Try to execute main file
                main_file = None
                # Priority: main.py, then files with "main" in name, then first .py file
                for path in ir.generated_code.keys():
                    if path.endswith(".py"):
                        if path == "main.py" or os.path.basename(path) == "main.py":
                            main_file = os.path.join(tmpdir, path)
                            break
                
                if not main_file:
                    for path in ir.generated_code.keys():
                        if path.endswith(".py") and "main" in path.lower():
                            main_file = os.path.join(tmpdir, path)
                            break
                
                if not main_file:
                    # Fallback: use first Python file
                    for path in ir.generated_code.keys():
                        if path.endswith(".py"):
                            main_file = os.path.join(tmpdir, path)
                            break
                
                if not main_file:
                    results["errors"].append("No Python file found to execute")
                    results["status"] = "error"
                    return results
                
                # Execute (with timeout and safety checks)
                try:
                    # Safety: Only execute if code looks safe (basic check)
                    with open(main_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Basic safety check: warn about potentially dangerous operations
                    dangerous_patterns = ['__import__', 'eval(', 'exec(', 'open(', 'subprocess', 'os.system']
                    has_dangerous = any(pattern in code_content for pattern in dangerous_patterns)
                    if has_dangerous:
                        self.log_progress("Warning: Code contains potentially unsafe operations", "warning")
                    
                    result = subprocess.run(
                        ["python", main_file],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Ensure output is captured
                    )
                    
                    results["output"] = result.stdout + result.stderr
                    results["return_code"] = result.returncode
                    
                    if result.returncode == 0:
                        results["status"] = "success"
                        # Try to extract metrics from output
                        results["metrics"] = self._extract_metrics_from_output(results["output"])
                    else:
                        results["status"] = "error"
                        results["errors"].append(f"Execution failed with code {result.returncode}")
                        results["errors"].append(result.stderr)
                        
                except subprocess.TimeoutExpired:
                    results["status"] = "timeout"
                    results["errors"].append(f"Execution timed out after {self.timeout}s")
                except Exception as e:
                    results["status"] = "error"
                    results["errors"].append(str(e))
        
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"Failed to execute code: {e}")
        
        return results
    
    def _extract_metrics_from_output(self, output: str) -> Dict[str, float]:
        """Extract metrics from execution output."""
        metrics = {}
        import re
        
        patterns = {
            'accuracy': r'accuracy[:\s]+([\d.]+)',
            'loss': r'loss[:\s]+([\d.]+)',
            'f1': r'f1[-\s]?score[:\s]+([\d.]+)',
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    pass
        
        return metrics
    
    def _compare_results(
        self, 
        paper_metrics: Dict[str, float],
        actual_results: Dict
    ) -> Dict:
        """Compare actual results with paper metrics."""
        comparison = {
            "status": "unknown",
            "tolerance_check": False,
            "discrepancies": []
        }
        
        if not paper_metrics:
            comparison["status"] = "no_baseline"
            comparison["message"] = "No paper metrics available for comparison"
            return comparison
        
        if actual_results.get("status") != "success":
            comparison["status"] = "execution_failed"
            return comparison
        
        actual_metrics = actual_results.get("metrics", {})
        
        if not actual_metrics:
            comparison["status"] = "no_metrics"
            comparison["message"] = "No metrics extracted from execution"
            return comparison
        
        # Compare metrics
        all_match = True
        for metric_name, paper_value in paper_metrics.items():
            if metric_name in actual_metrics:
                actual_value = actual_metrics[metric_name]
                difference = abs(paper_value - actual_value)
                relative_diff = difference / paper_value if paper_value != 0 else difference
                
                if relative_diff > self.tolerance:
                    all_match = False
                    comparison["discrepancies"].append(
                        f"{metric_name}: paper={paper_value:.4f}, "
                        f"actual={actual_value:.4f}, diff={relative_diff:.2%}"
                    )
        
        if all_match and comparison["discrepancies"] == []:
            comparison["status"] = "pass"
            comparison["tolerance_check"] = True
        else:
            comparison["status"] = "fail"
            comparison["tolerance_check"] = False
        
        return comparison

