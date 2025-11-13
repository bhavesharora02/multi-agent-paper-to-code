"""
LangGraph workflow definition for multi-agent code generation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Literal
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback if LangGraph not installed
    logging.warning("LangGraph not installed. Using simple workflow.")
    StateGraph = None

from workflow.state import WorkflowState
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.debugger_agent import DebuggerAgent
from agents.rater_agent import RaterAgent
from agents.optimizer_agent import OptimizerAgent
from agents.planner_agent import PlannerAgent
from llm.llm_client import LLMClient, LLMProvider

logger = logging.getLogger(__name__)


def create_workflow(config: Dict[str, Any] = None) -> Any:
    """
    Create LangGraph workflow for multi-agent code generation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled workflow graph
    """
    if config is None:
        import yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # Initialize LLM client
    llm_config = config.get("llm", {})
    provider_str = llm_config.get("provider", "groq").lower()
    if provider_str == "openai":
        provider = LLMProvider.OPENAI
    elif provider_str == "anthropic":
        provider = LLMProvider.ANTHROPIC
    elif provider_str == "openrouter":
        provider = LLMProvider.OPENROUTER
    elif provider_str == "groq":
        provider = LLMProvider.GROQ
    else:
        provider = LLMProvider.GROQ
    
    model = llm_config.get("model", None)
    llm_client = LLMClient(provider=provider, model=model)
    
    # Initialize agents
    agent_configs = config.get("agents", {})
    
    coder_agent = CoderAgent(
        config={**llm_config, **agent_configs.get("coder", {})},
        llm_client=llm_client
    )
    
    # Initialize sandbox executor if available
    sandbox_executor = None
    try:
        from sandbox.executor import SandboxExecutor
        sandbox_config = config.get("sandbox", {})
        sandbox_executor = SandboxExecutor(sandbox_config)
    except Exception as e:
        logger.warning(f"Could not initialize sandbox executor: {e}")
    
    tester_agent = TesterAgent(
        config={**llm_config, **agent_configs.get("tester", {})},
        llm_client=llm_client,
        sandbox_executor=sandbox_executor
    )
    
    debugger_agent = DebuggerAgent(
        config={**llm_config, **agent_configs.get("debugger", {})},
        llm_client=llm_client
    )
    
    rater_agent = RaterAgent(
        config={**llm_config, **agent_configs.get("rater", {})},
        llm_client=llm_client
    )
    
    optimizer_agent = OptimizerAgent(
        config={**llm_config, **agent_configs.get("optimizer", {})},
        llm_client=llm_client
    )
    
    planner_agent = PlannerAgent(config.get("workflow", {}))
    
    # Create workflow graph
    if StateGraph is None:
        # Fallback: simple sequential workflow
        return SimpleWorkflow(
            coder_agent, tester_agent, debugger_agent, rater_agent, 
            optimizer_agent, planner_agent, config
        )
    
    # Define agent nodes
    def coder_node(state: WorkflowState) -> WorkflowState:
        """Coder agent node."""
        logger.info("Running Coder Agent...")
        return coder_agent.process(state)
    
    def tester_node(state: WorkflowState) -> WorkflowState:
        """Tester agent node (runs in background)."""
        logger.info("Running Tester Agent...")
        return tester_agent.process(state)
    
    def debugger_node(state: WorkflowState) -> WorkflowState:
        """Debugger agent node."""
        logger.info("Running Debugger Agent...")
        return debugger_agent.process(state)
    
    def rater_node(state: WorkflowState) -> WorkflowState:
        """Rater agent node."""
        logger.info("Running Rater Agent...")
        return rater_agent.process(state)
    
    def optimizer_node(state: WorkflowState) -> WorkflowState:
        """Optimizer agent node."""
        logger.info("Running Optimizer Agent...")
        return optimizer_agent.process(state)
    
    # Define routing functions
    def route_after_tester(state: WorkflowState) -> Literal["debugger", "rater", "end"]:
        """Route after tester based on test results."""
        tests_passed = state.get("tests_passed", False)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        needs_improvement = state.get("needs_improvement", False)
        
        # If we're in improvement mode (rating was low), go to rater after tests
        if needs_improvement:
            return "rater"  # Rate the improved code
        
        # If tests passed or max iterations, go to rater
        if tests_passed or iteration >= max_iterations:
            return "rater"
        # Otherwise, debug
        return "debugger"
    
    def route_after_debugger(state: WorkflowState) -> Literal["tester", "rater", "end"]:
        """Route after debugger - go back to tester or to rater if max iterations."""
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        if iteration >= max_iterations:
            return "rater"  # Go to rater even if tests didn't pass
        return "tester"
    
    def route_after_rater(state: WorkflowState) -> Literal["coder", "optimizer", "end"]:
        """Route after rater based on rating."""
        rating = state.get("code_rating", 0)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        
        # If rating is below 7, try to improve by going back to coder
        if rating < 7.0 and iteration < max_iterations:
            # Check if we've already tried improving
            improvement_attempts = state.get("improvement_attempts", 0)
            max_improvements = config.get("workflow", {}).get("max_improvement_attempts", 3)
            
            if improvement_attempts < max_improvements:
                # Try to improve the code
                state["improvement_attempts"] = improvement_attempts + 1
                state["needs_improvement"] = True
                logger.info(f"Rating {rating:.1f} below 7.0, attempting improvement (attempt {improvement_attempts + 1}/{max_improvements})")
                return "coder"  # Go back to coder to regenerate better code
        
        # If rating is good (>= 7), can optimize
        if rating >= 7.0:
            if config.get("workflow", {}).get("enable_optimizer", True):
                if not state.get("optimized", False):
                    return "optimizer"
        
        return "end"
    
    def route_after_optimizer(state: WorkflowState) -> Literal["end"]:
        """Route after optimizer - always end."""
        return "end"
    
    # Build graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("rater", rater_node)
    workflow.add_node("optimizer", optimizer_node)
    
    # Set entry point
    workflow.set_entry_point("coder")
    
    # Add edges
    workflow.add_edge("coder", "tester")
    workflow.add_conditional_edges(
        "tester",
        route_after_tester,
        {
            "debugger": "debugger",
            "rater": "rater",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "debugger",
        route_after_debugger,
        {
            "tester": "tester",
            "rater": "rater",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "rater",
        route_after_rater,
        {
            "coder": "coder",  # Loop back to improve if rating < 7
            "optimizer": "optimizer",
            "end": END
        }
    )
    workflow.add_edge("optimizer", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


class SimpleWorkflow:
    """
    Simple sequential workflow fallback when LangGraph is not available.
    """
    
    def __init__(self, coder, tester, debugger, rater, optimizer, planner, config):
        self.coder = coder
        self.tester = tester
        self.debugger = debugger
        self.rater = rater
        self.optimizer = optimizer
        self.planner = planner
        self.config = config
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow sequentially."""
        state.setdefault("iteration", 0)
        state.setdefault("code_history", [])
        state.setdefault("max_iterations", self.config.get("workflow", {}).get("max_iterations", 10))
        
        # Run coder
        state = self.coder.process(state)
        state["iteration"] = state.get("iteration", 0) + 1
        
        # Improvement loop: keep trying until rating >= 7 or max attempts
        max_improvements = self.config.get("workflow", {}).get("max_improvement_attempts", 3)
        improvement_attempts = 0
        state.setdefault("improvement_attempts", 0)
        
        while improvement_attempts <= max_improvements:
            # Run tester-debugger loop until tests pass or max iterations
            max_iterations = state.get("max_iterations", 10)
            test_iteration = 0
            while test_iteration < max_iterations:
                # Run tester
                state = self.tester.process(state)
                state["iteration"] = state.get("iteration", 0) + 1
                test_iteration += 1
                
                # Check if tests passed
                if state.get("tests_passed", False):
                    break
                
                # Run debugger if tests failed
                state = self.debugger.process(state)
                state["iteration"] = state.get("iteration", 0) + 1
                test_iteration += 1
            
            # Run rater (always runs, regardless of test results)
            state = self.rater.process(state)
            state["iteration"] = state.get("iteration", 0) + 1
            
            rating = state.get("code_rating", 0)
            
            # If rating is good (>= 7), break out of improvement loop
            if rating >= 7.0:
                break
            
            # If rating is still low and we have attempts left, try to improve
            if improvement_attempts < max_improvements:
                logger.info(f"Rating {rating:.1f} below 7.0, attempting improvement ({improvement_attempts + 1}/{max_improvements})")
                state["needs_improvement"] = True
                state["improvement_attempts"] = improvement_attempts + 1
                
                # Go back to coder to regenerate improved code
                state = self.coder.process(state)
                state["iteration"] = state.get("iteration", 0) + 1
                
                improvement_attempts += 1
            else:
                break
        
        # Run optimizer if rating is good (>= 7)
        rating = state.get("code_rating", 0)
        if rating >= 7.0 and self.config.get("workflow", {}).get("enable_optimizer", True):
            if not state.get("optimized", False):
                state = self.optimizer.process(state)
                state["iteration"] = state.get("iteration", 0) + 1
        
        state["iteration_count"] = state.get("iteration", 0)
        state["success"] = True  # Always successful - rating is shown, not test results
        
        return state

