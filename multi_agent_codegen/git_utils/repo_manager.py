"""
Git repository manager for tracking code changes.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class GitRepoManager:
    """
    Manages Git repository for tracking code iterations.
    """
    
    def __init__(self, repo_path: str, auto_commit: bool = True):
        """
        Initialize Git repository manager.
        
        Args:
            repo_path: Path to git repository
            auto_commit: Whether to automatically commit after each iteration
        """
        self.repo_path = Path(repo_path)
        self.auto_commit = auto_commit
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize repository if it doesn't exist
        if not self.repo_path.exists():
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self._init_repo()
    
    def _init_repo(self):
        """Initialize git repository."""
        try:
            subprocess.run(
                ["git", "init"],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Initialized git repository at {self.repo_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize git repo: {e}")
        except FileNotFoundError:
            self.logger.warning("Git not found - version tracking disabled")
    
    def save_code(self, code: str, iteration: int, agent: str) -> bool:
        """
        Save code to repository.
        
        Args:
            code: Code to save
            iteration: Current iteration number
            agent: Agent that generated the code
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Write code to file
            code_file = self.repo_path / "code.py"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Write test file if exists
            test_file = self.repo_path / "test_code.py"
            if test_file.exists():
                # Keep existing test file
                pass
            
            # Commit if auto_commit is enabled
            if self.auto_commit:
                return self.commit(f"Iteration {iteration} - {agent} agent")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save code: {e}")
            return False
    
    def commit(self, message: str) -> bool:
        """
        Commit changes to git repository.
        
        Args:
            message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if git is available
            subprocess.run(
                ["git", "--version"],
                check=True,
                capture_output=True
            )
            
            # Add all files
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Committed: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git commit failed: {e}")
            return False
        except FileNotFoundError:
            self.logger.warning("Git not found - skipping commit")
            return False
    
    def get_history(self) -> list:
        """
        Get commit history.
        
        Returns:
            List of commit information
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--all"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1]
                        })
            
            return commits
            
        except Exception as e:
            self.logger.error(f"Failed to get history: {e}")
            return []

