"""
Sandbox executor for safe code execution.
"""

import subprocess
import tempfile
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class SandboxExecutor:
    """
    Executes code in a safe sandbox environment.
    Supports both local execution and Docker-based isolation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize sandbox executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sandbox_type = self.config.get("type", "local")
        self.timeout = self.config.get("timeout", 30)
        self.memory_limit = self.config.get("memory_limit", "512MB")
        self.cpu_limit = self.config.get("cpu_limit", 1)
    
    def execute(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Execute code and tests in sandbox.
        
        Args:
            code: The code to execute
            test_code: Test code to run
            
        Returns:
            Test results dictionary
        """
        if self.sandbox_type == "docker":
            return self._execute_docker(code, test_code)
        else:
            return self._execute_local(code, test_code)
    
    def _execute_local(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Execute code locally with basic isolation.
        
        Args:
            code: The code to execute
            test_code: Test code to run
            
        Returns:
            Test results dictionary
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = os.path.join(tmpdir, "code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Write test file
            test_file = os.path.join(tmpdir, "test_code.py")
            with open(test_file, "w", encoding="utf-8") as f:
                # Import the code module
                f.write("import sys\n")
                f.write("import os\n")
                f.write("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n")
                f.write("from code import *\n\n")
                f.write(test_code)
            
            # Run pytest
            try:
                result = subprocess.run(
                    ["pytest", test_file, "-v", "--tb=short"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                output = result.stdout + result.stderr
                all_passed = result.returncode == 0
                
                # Parse test results
                num_passed = output.count("PASSED")
                num_failed = output.count("FAILED")
                
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
                    "errors": f"Test execution timed out after {self.timeout} seconds",
                    "return_code": -1
                }
            except Exception as e:
                return {
                    "all_passed": False,
                    "num_passed": 0,
                    "num_failed": 0,
                    "output": "",
                    "errors": str(e),
                    "return_code": -1
                }
    
    def _execute_docker(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Execute code in Docker container for better isolation.
        
        Args:
            code: The code to execute
            test_code: Test code to run
            
        Returns:
            Test results dictionary
        """
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Docker not available, falling back to local execution")
            return self._execute_local(code, test_code)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = os.path.join(tmpdir, "code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Write test file
            test_file = os.path.join(tmpdir, "test_code.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("import sys\n")
                f.write("import os\n")
                f.write("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n")
                f.write("from code import *\n\n")
                f.write(test_code)
            
            # Write requirements.txt (minimal)
            req_file = os.path.join(tmpdir, "requirements.txt")
            with open(req_file, "w") as f:
                f.write("pytest>=7.4.0\n")
            
            # Create Dockerfile
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile, "w") as f:
                f.write("""FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["pytest", "test_code.py", "-v"]
""")
            
            # Build and run Docker container
            try:
                container_name = f"codegen_test_{os.getpid()}"
                
                # Build image
                build_result = subprocess.run(
                    ["docker", "build", "-t", container_name, tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if build_result.returncode != 0:
                    self.logger.error(f"Docker build failed: {build_result.stderr}")
                    return self._execute_local(code, test_code)
                
                # Run container with resource limits
                run_cmd = [
                    "docker", "run", "--rm",
                    "--memory", self.memory_limit,
                    "--cpus", str(self.cpu_limit),
                    "--name", container_name,
                    container_name
                ]
                
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10  # Extra time for Docker overhead
                )
                
                output = result.stdout + result.stderr
                all_passed = result.returncode == 0
                
                # Parse test results
                num_passed = output.count("PASSED")
                num_failed = output.count("FAILED")
                
                # Cleanup
                try:
                    subprocess.run(["docker", "rmi", container_name], capture_output=True)
                except:
                    pass
                
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
                    "errors": f"Test execution timed out after {self.timeout} seconds",
                    "return_code": -1
                }
            except Exception as e:
                self.logger.error(f"Docker execution error: {e}")
                return self._execute_local(code, test_code)

