"""
Base agent class for multi-agent pipeline.
All specialized agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from utils.intermediate_representation import PaperToCodeIR


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
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Process the intermediate representation.
        
        Args:
            ir: Current intermediate representation
            
        Returns:
            Updated intermediate representation
        """
        pass
    
    def validate_input(self, ir: PaperToCodeIR) -> bool:
        """
        Validate input IR before processing.
        
        Args:
            ir: Intermediate representation to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not ir or not isinstance(ir, PaperToCodeIR):
            self.logger.error("Invalid IR: not a PaperToCodeIR instance")
            return False
        return True
    
    def log_progress(self, message: str, level: str = "info"):
        """Log progress message."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{self.agent_name}] {message}")
    
    def update_ir_status(self, ir: PaperToCodeIR, status: str):
        """Update IR status."""
        ir.update_status(status, self.agent_name)
        self.log_progress(f"Status updated to: {status}")

