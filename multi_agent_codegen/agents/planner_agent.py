"""
Planner Agent: Orchestrates the workflow and manages agent execution.
"""

from typing import Dict, Any, Optional
import logging


class PlannerAgent:
    """
    Planner Agent oversees the workflow, invokes agents, logs intermediate states,
    and halts when tests pass or max iterations reached.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Planner Agent."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_iterations = self.config.get("max_iterations", 10)
    
    def should_continue(self, state: Dict[str, Any]) -> bool:
        """
        Determine if workflow should continue.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if should continue, False otherwise
        """
        iteration = state.get("iteration", 0)
        tests_passed = state.get("tests_passed", False)
        error = state.get("error")
        
        # Stop if tests passed
        if tests_passed:
            self.logger.info("Tests passed - stopping workflow")
            return False
        
        # Stop if max iterations reached
        if iteration >= self.max_iterations:
            self.logger.info(f"Max iterations ({self.max_iterations}) reached - stopping workflow")
            return False
        
        # Stop if critical error
        if error and "critical" in error.lower():
            self.logger.error("Critical error - stopping workflow")
            return False
        
        return True
    
    def get_next_agent(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Determine which agent should run next.
        
        Args:
            state: Current workflow state
            
        Returns:
            Name of next agent or None
        """
        last_agent = state.get("last_agent")
        tests_passed = state.get("tests_passed", False)
        code = state.get("code")
        
        # If no code yet, start with coder
        if not code:
            return "coder"
        
        # If tests passed, run optimizer if enabled
        if tests_passed:
            if self.config.get("enable_optimizer", True):
                if not state.get("optimized", False):
                    return "optimizer"
            return None  # Done
        
        # If tests failed, run debugger
        if last_agent == "tester" and not tests_passed:
            return "debugger"
        
        # If debugger just ran, go back to tester
        if last_agent == "debugger":
            return "tester"
        
        # If coder just ran, go to tester
        if last_agent == "coder":
            return "tester"
        
        # Default: run coder
        return "coder"
    
    def log_state(self, state: Dict[str, Any]):
        """Log current workflow state."""
        iteration = state.get("iteration", 0)
        last_agent = state.get("last_agent", "unknown")
        tests_passed = state.get("tests_passed", False)
        
        self.logger.info(
            f"Iteration {iteration} | Agent: {last_agent} | "
            f"Tests passed: {tests_passed}"
        )

