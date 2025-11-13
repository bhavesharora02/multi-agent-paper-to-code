"""
Benchmark runner for evaluating the multi-agent system.
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.benchmarks import BenchmarkLoader
from evaluation.metrics import calculate_metrics
from workflow.graph import create_workflow
from workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def run_benchmark(
    benchmark_name: str,
    num_problems: int = 20,
    max_iterations: int = 10,
    config: Dict[str, Any] = None,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Run evaluation on a benchmark.
    
    Args:
        benchmark_name: Name of benchmark (humaneval, leetcode_easy, apps)
        num_problems: Number of problems to evaluate
        max_iterations: Maximum iterations per problem
        config: Configuration dictionary
        output_file: Path to save results
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Running benchmark: {benchmark_name} with {num_problems} problems")
    
    # Load benchmark
    loader = BenchmarkLoader(benchmark_name)
    problems = loader.load_problems(num_problems)
    
    if not problems:
        logger.error("No problems loaded")
        return {}
    
    # Create workflow
    workflow = create_workflow(config)
    
    # Run on each problem
    results = []
    for i, problem in enumerate(problems):
        logger.info(f"Processing problem {i+1}/{len(problems)}: {problem.get('task_id', problem.get('id', 'unknown'))}")
        
        # Extract specification
        if "prompt" in problem:
            # HumanEval format
            specification = problem["prompt"] + "\n" + problem.get("test", "")
        elif "specification" in problem:
            specification = problem["specification"]
        else:
            specification = problem.get("description", "")
        
        # Initialize state
        state: WorkflowState = {
            "specification": specification,
            "code": "",
            "code_history": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "tests_passed": False,
            "optimized": False,
            "fix_history": []
        }
        
        # Run workflow
        start_time = time.time()
        try:
            result = workflow.invoke(state)
            time_taken = time.time() - start_time
            
            results.append({
                "problem_id": problem.get("task_id", problem.get("id", f"problem_{i}")),
                "specification": specification,
                "success": result.get("success", False),
                "tests_passed": result.get("tests_passed", False),
                "iteration_count": result.get("iteration_count", 0),
                "time_taken": time_taken,
                "code": result.get("code", ""),
                "error": result.get("error")
            })
            
            logger.info(f"Problem {i+1} completed: {'SUCCESS' if result.get('success') else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Error processing problem {i+1}: {e}")
            results.append({
                "problem_id": problem.get("task_id", problem.get("id", f"problem_{i}")),
                "specification": specification,
                "success": False,
                "tests_passed": False,
                "iteration_count": 0,
                "time_taken": time.time() - start_time,
                "error": str(e)
            })
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Compile final results
    final_results = {
        "benchmark": benchmark_name,
        "num_problems": num_problems,
        "max_iterations": max_iterations,
        "metrics": metrics,
        "results": results
    }
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Benchmark: {benchmark_name}")
    print(f"Problems: {num_problems}")
    print(f"\nMetrics:")
    print(f"  pass@1: {metrics['pass_at_1']:.3f}")
    print(f"  pass@5: {metrics['pass_at_5']:.3f}")
    print(f"  pass@10: {metrics['pass_at_10']:.3f}")
    print(f"  Success Rate: {metrics['success_rate']:.3f}")
    print(f"  Average Iterations: {metrics['avg_iterations']:.2f}")
    print(f"  Average Time: {metrics['avg_time']:.2f}s")
    print(f"  Solved: {metrics['solved']}/{metrics['total_problems']}")
    print("="*80)
    
    return final_results


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks for multi-agent code generation"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["humaneval", "leetcode_easy", "apps"],
        help="Benchmark to run"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=20,
        help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per problem"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config if provided
    config = None
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # Run benchmark
    run_benchmark(
        benchmark_name=args.benchmark,
        num_problems=args.num_problems,
        max_iterations=args.max_iterations,
        config=config,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

