"""
Main entry point for multi-agent code generation system.
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from workflow.graph import create_workflow
from workflow.state import WorkflowState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Code Generation and Debugging System"
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Code specification (natural language description)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for generated code"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override max_iterations if provided
    if args.max_iterations:
        config.setdefault("workflow", {})["max_iterations"] = args.max_iterations
    
    # Create workflow
    logger.info("Creating workflow...")
    workflow = create_workflow(config)
    
    # Initialize state
    state: WorkflowState = {
        "specification": args.spec,
        "code": "",
        "code_history": [],
        "iteration": 0,
        "max_iterations": config.get("workflow", {}).get("max_iterations", 10),
        "tests_passed": False,
        "optimized": False,
        "fix_history": []
    }
    
    # Run workflow
    logger.info(f"Running workflow for: {args.spec[:100]}...")
    try:
        result = workflow.invoke(state)
        
        # Print results
        print("\n" + "="*80)
        print("WORKFLOW RESULTS")
        print("="*80)
        print(f"Specification: {result.get('specification', '')}")
        print(f"Iterations: {result.get('iteration_count', 0)}")
        print(f"Tests Passed: {result.get('tests_passed', False)}")
        print(f"Success: {result.get('success', False)}")
        print("\nGenerated Code:")
        print("-"*80)
        print(result.get('code', ''))
        print("-"*80)
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.get('code', ''))
            logger.info(f"Code saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

