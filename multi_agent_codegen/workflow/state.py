"""
Workflow state management for LangGraph.
"""

from typing import Dict, Any, List, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """
    State structure for the multi-agent workflow.
    """
    # Input
    specification: str  # Original specification
    
    # Code artifacts
    code: str  # Current code
    code_history: List[str]  # History of code versions
    test_code: str  # Generated test code (internal, not shown in UI)
    test_results: Dict[str, Any]  # Test execution results (internal)
    tests_passed: bool  # Whether all tests passed (internal)
    test_output: str  # Test execution output (internal)
    test_errors: str  # Test error messages (internal)
    code_rating: float  # Code quality rating (0-10) - shown in UI
    rating_details: str  # Brief rating explanation
    rating_feedback: str  # Detailed feedback
    
    # Execution metadata
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum iterations allowed
    last_agent: str  # Last agent that ran
    error: Optional[str]  # Error message if any
    
    # Debugging (internal, not shown in UI)
    fix_history: List[Dict[str, Any]]  # History of fixes
    
    # Optimization
    optimized: bool  # Whether code has been optimized
    
    # Improvement loop
    needs_improvement: bool  # Whether code needs improvement (rating < 7)
    improvement_attempts: int  # Number of improvement attempts
    
    # Git
    git_repo_path: Optional[str]  # Path to git repository
    
    # Metrics
    iteration_count: int  # Total iterations taken
    success: bool  # Whether workflow succeeded

