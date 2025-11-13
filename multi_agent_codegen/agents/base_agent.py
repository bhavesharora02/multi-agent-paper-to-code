"""
Base agent class for multi-agent code generation pipeline.
All specialized agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent pipeline.
    Provides common functionality and interface.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """
        Initialize base agent.
        
        Args:
            config: Agent-specific configuration
            llm_client: LLM client instance (optional)
        """
        self.config = config or {}
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent_name = self.__class__.__name__
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Validate input state before processing.
        
        Args:
            state: Workflow state to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not state or not isinstance(state, dict):
            self.logger.error("Invalid state: not a dictionary")
            return False
        return True
    
    def log_progress(self, message: str, level: str = "info"):
        """Log progress message."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{self.agent_name}] {message}")
    
    def update_state(self, state: Dict[str, Any], updates: Dict[str, Any]):
        """Update state with new values."""
        state.update(updates)
        self.log_progress(f"State updated: {list(updates.keys())}")

