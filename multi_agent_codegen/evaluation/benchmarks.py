"""
Benchmark datasets for evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BenchmarkLoader:
    """
    Loads benchmark datasets (HumanEval, APPS, LeetCode Easy).
    """
    
    def __init__(self, benchmark_name: str, data_path: Optional[str] = None):
        """
        Initialize benchmark loader.
        
        Args:
            benchmark_name: Name of benchmark (humaneval, leetcode_easy, apps)
            data_path: Path to benchmark data (optional)
        """
        self.benchmark_name = benchmark_name.lower()
        self.data_path = data_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_problems(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load problems from benchmark.
        
        Args:
            num_problems: Number of problems to load (None for all)
            
        Returns:
            List of problem dictionaries
        """
        if self.benchmark_name == "humaneval":
            return self._load_humaneval(num_problems)
        elif self.benchmark_name == "leetcode_easy":
            return self._load_leetcode_easy(num_problems)
        elif self.benchmark_name == "apps":
            return self._load_apps(num_problems)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark_name}")
    
    def _load_humaneval(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load HumanEval problems.
        
        Args:
            num_problems: Number of problems to load
            
        Returns:
            List of HumanEval problems
        """
        # Try to load from data path if provided
        if self.data_path:
            data_file = Path(self.data_path) / "HumanEval.jsonl"
            if data_file.exists():
                return self._load_jsonl(data_file, num_problems)
        
        # Generate sample problems if data file not found
        self.logger.warning("HumanEval data file not found, using sample problems")
        return self._get_sample_humaneval_problems(num_problems)
    
    def _load_leetcode_easy(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LeetCode Easy problems."""
        # Try to load from data path if provided
        if self.data_path:
            data_file = Path(self.data_path) / "leetcode_easy.json"
            if data_file.exists():
                with open(data_file, "r") as f:
                    problems = json.load(f)
                    return problems[:num_problems] if num_problems else problems
        
        # Generate sample problems
        self.logger.warning("LeetCode Easy data file not found, using sample problems")
        return self._get_sample_leetcode_problems(num_problems)
    
    def _load_apps(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load APPS problems."""
        # APPS is more complex, would need actual dataset
        self.logger.warning("APPS dataset not implemented, using sample problems")
        return self._get_sample_apps_problems(num_problems)
    
    def _load_jsonl(self, file_path: Path, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        problems = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if num_problems and i >= num_problems:
                    break
                problems.append(json.loads(line))
        return problems
    
    def _get_sample_humaneval_problems(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get sample HumanEval problems."""
        sample_problems = [
            {
                "task_id": "test/0",
                "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\"\"\"\n",
                "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True\nassert is_palindrome('a') == True",
                "entry_point": "is_palindrome"
            },
            {
                "task_id": "test/1",
                "prompt": "def two_sum(nums: list, target: int) -> list:\n    \"\"\"Find two numbers that add up to target.\"\"\"\n",
                "test": "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nassert two_sum([3, 2, 4], 6) == [1, 2]",
                "entry_point": "two_sum"
            },
            {
                "task_id": "test/2",
                "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Reverse a string.\"\"\"\n",
                "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
                "entry_point": "reverse_string"
            }
        ]
        
        if num_problems:
            return sample_problems[:num_problems]
        return sample_problems
    
    def _get_sample_leetcode_problems(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get sample LeetCode Easy problems."""
        sample_problems = [
            {
                "id": 1,
                "title": "Two Sum",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "specification": "Implement a function two_sum(nums: list, target: int) -> list that returns indices of two numbers that add up to target."
            },
            {
                "id": 2,
                "title": "Valid Parentheses",
                "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                "specification": "Implement a function is_valid(s: str) -> bool that checks if parentheses are valid."
            }
        ]
        
        if num_problems:
            return sample_problems[:num_problems]
        return sample_problems
    
    def _get_sample_apps_problems(self, num_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get sample APPS problems."""
        # APPS problems are more complex, would need actual dataset
        return []

