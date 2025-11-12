"""
Planner Agent - Orchestrates the multi-agent pipeline.
Coordinates agent execution and manages workflow.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from agents.paper_analysis_agent import PaperAnalysisAgent
from agents.algorithm_interpretation_agent import AlgorithmInterpretationAgent
from agents.api_mapping_agent import APIMappingAgent
from agents.code_integration_agent import CodeIntegrationAgent
from agents.verification_agent import VerificationAgent
from agents.debugging_agent import DebuggingAgent
from utils.intermediate_representation import PaperToCodeIR
from llm.llm_client import LLMClient, LLMProvider


class PlannerAgent(BaseAgent):
    """
    Planner Agent that orchestrates the multi-agent pipeline.
    Manages agent execution sequence and context.
    """
    
    def __init__(self, config: Dict = None, llm_client=None, progress_callback=None):
        """Initialize Planner Agent.
        
        Args:
            config: Agent configuration
            llm_client: LLM client instance
            progress_callback: Optional callback function(progress, message) to report progress
        """
        super().__init__(config, llm_client)
        
        # Progress callback for real-time updates
        self.progress_callback = progress_callback
        
        # Initialize sub-agents
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all sub-agents."""
        agent_config = self.config.get('agents', {})
        
        # Paper Analysis Agent
        if self.config.get('use_paper_analysis', True):
            self.agents['paper_analysis'] = PaperAnalysisAgent(
                config=agent_config.get('paper_analysis', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized Paper Analysis Agent")
        
        # Algorithm Interpretation Agent
        if self.config.get('use_algorithm_interpretation', True):
            self.agents['algorithm_interpretation'] = AlgorithmInterpretationAgent(
                config=agent_config.get('algorithm_interpretation', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized Algorithm Interpretation Agent")
        
        # API Mapping Agent
        if self.config.get('use_api_mapping', True):
            self.agents['api_mapping'] = APIMappingAgent(
                config=agent_config.get('api_mapping', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized API Mapping Agent")
        
        # Code Integration Agent
        if self.config.get('use_code_integration', True):
            self.agents['code_integration'] = CodeIntegrationAgent(
                config=agent_config.get('code_integration', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized Code Integration Agent")
        
        # Verification Agent
        if self.config.get('use_verification', True):
            self.agents['verification'] = VerificationAgent(
                config=agent_config.get('verification', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized Verification Agent")
        
        # Debugging Agent
        if self.config.get('use_debugging', True):
            self.agents['debugging'] = DebuggingAgent(
                config=agent_config.get('debugging', {}),
                llm_client=self.llm_client
            )
            self.log_progress("Initialized Debugging Agent")
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Orchestrate the multi-agent pipeline.
        
        Args:
            ir: Initial intermediate representation
            
        Returns:
            Final processed IR
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress("Starting multi-agent pipeline orchestration")
        
        try:
            import time
            
            # Phase 1: Paper Analysis
            if 'paper_analysis' in self.agents:
                self.log_progress("Phase 1: Paper Analysis")
                if self.progress_callback:
                    self.progress_callback(20, "Phase 1: Analyzing paper content with AI...")
                ir = self.agents['paper_analysis'].process(ir)
                time.sleep(1)  # Small delay to avoid rate limits
                if ir.status == "failed":
                    self.log_progress("Paper analysis failed", "error")
                    if self.progress_callback:
                        self.progress_callback(20, "Paper analysis failed")
                    return ir
                self.log_progress(f"Found {len(ir.algorithms)} algorithms")
                if self.progress_callback:
                    self.progress_callback(30, f"Found {len(ir.algorithms)} algorithm(s) in paper")
            
            # Phase 2: Algorithm Interpretation
            if 'algorithm_interpretation' in self.agents:
                self.log_progress("Phase 2: Algorithm Interpretation")
                if self.progress_callback:
                    self.progress_callback(40, "Phase 2: Interpreting algorithms and mathematical notation...")
                time.sleep(1)  # Small delay to avoid rate limits
                ir = self.agents['algorithm_interpretation'].process(ir)
                if ir.status == "failed":
                    self.log_progress("Algorithm interpretation failed", "error")
                    if self.progress_callback:
                        self.progress_callback(40, "Algorithm interpretation failed")
                    return ir
                self.log_progress("Algorithms interpreted")
                if self.progress_callback:
                    self.progress_callback(50, "Algorithms successfully interpreted")
            
            # Phase 3: API/Library Mapping
            if 'api_mapping' in self.agents:
                self.log_progress("Phase 3: API/Library Mapping")
                if self.progress_callback:
                    self.progress_callback(60, "Phase 3: Mapping algorithms to ML framework APIs...")
                time.sleep(1)  # Small delay to avoid rate limits
                ir = self.agents['api_mapping'].process(ir)
                if ir.status == "failed":
                    self.log_progress("API mapping failed", "error")
                    if self.progress_callback:
                        self.progress_callback(60, "API mapping failed")
                    return ir
                self.log_progress(f"Mapped {len(ir.mapped_components)} components")
                if self.progress_callback:
                    self.progress_callback(70, f"Mapped {len(ir.mapped_components)} component(s) to framework APIs")
            
            # Phase 4: Code Integration
            if 'code_integration' in self.agents:
                self.log_progress("Phase 4: Code Integration")
                if self.progress_callback:
                    self.progress_callback(80, "Phase 4: Generating complete code implementation...")
                time.sleep(1)  # Small delay to avoid rate limits
                ir = self.agents['code_integration'].process(ir)
                if ir.status == "failed":
                    self.log_progress("Code integration failed", "error")
                    if self.progress_callback:
                        self.progress_callback(80, "Code integration failed")
                    return ir
                self.log_progress(f"Generated {len(ir.generated_code)} files")
                if self.progress_callback:
                    self.progress_callback(85, f"Generated {len(ir.generated_code)} code file(s)")
            
            # Phase 5: Verification
            if 'verification' in self.agents:
                self.log_progress("Phase 5: Verification")
                if self.progress_callback:
                    self.progress_callback(90, "Phase 5: Verifying generated code...")
                ir = self.agents['verification'].process(ir)
                if ir.verification_results:
                    self.log_progress(f"Verification status: {ir.verification_results.status}")
                    if self.progress_callback:
                        self.progress_callback(92, f"Verification: {ir.verification_results.status}")
            
            # Phase 6: Debugging (always run if enabled, not just when verification fails)
            if 'debugging' in self.agents:
                self.log_progress("Phase 6: Code Analysis and Debugging")
                if self.progress_callback:
                    self.progress_callback(93, "Phase 6: Analyzing and refining code...")
                time.sleep(1)  # Small delay to avoid rate limits
                ir = self.agents['debugging'].process(ir)
                refinement_count = len(ir.refinement_history) if hasattr(ir, 'refinement_history') and ir.refinement_history else 0
                self.log_progress(f"Refinement iterations: {refinement_count}")
                if self.progress_callback:
                    self.progress_callback(95, f"Completed {refinement_count} refinement iteration(s)")
            
            self.update_ir_status(ir, "completed")
            self.log_progress("Pipeline orchestration completed")
            if self.progress_callback:
                self.progress_callback(98, "Pipeline orchestration completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in pipeline orchestration: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def get_pipeline_status(self, ir: PaperToCodeIR) -> Dict:
        """Get current pipeline status."""
        return {
            "status": ir.status,
            "current_agent": ir.current_agent,
            "algorithms_found": len(ir.algorithms),
            "diagrams_found": len(ir.diagrams),
            "equations_found": len(ir.equations),
            "refinement_iterations": len(ir.refinement_history)
        }

