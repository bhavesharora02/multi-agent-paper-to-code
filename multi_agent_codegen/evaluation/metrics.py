"""
Evaluation metrics for code generation benchmarks.
"""

from typing import List, Dict, Any
import statistics


def calculate_pass_at_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate pass@k metric.
    
    pass@k is the probability that at least one of k generated solutions
    passes all tests.
    
    Args:
        results: List of result dictionaries with 'success' key
        k: Number of samples to consider
        
    Returns:
        pass@k value (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    # For each problem, check if at least one of k samples passed
    passed = 0
    total = 0
    
    # Group results by problem
    problems = {}
    for result in results:
        problem_id = result.get("problem_id", "unknown")
        if problem_id not in problems:
            problems[problem_id] = []
        problems[problem_id].append(result.get("success", False))
    
    # Calculate pass@k for each problem
    for problem_id, successes in problems.items():
        total += 1
        # Take first k samples
        samples = successes[:k]
        if any(samples):
            passed += 1
    
    return passed / total if total > 0 else 0.0


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of metrics
    """
    if not results:
        return {
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,
            "pass_at_10": 0.0,
            "success_rate": 0.0,
            "avg_iterations": 0.0,
            "avg_time": 0.0
        }
    
    # Calculate pass@k for different k values
    pass_at_1 = calculate_pass_at_k(results, 1)
    pass_at_5 = calculate_pass_at_k(results, 5)
    pass_at_10 = calculate_pass_at_k(results, 10)
    
    # Calculate success rate
    successes = [r.get("success", False) for r in results]
    success_rate = sum(successes) / len(successes) if successes else 0.0
    
    # Calculate average iterations
    iterations = [r.get("iteration_count", 0) for r in results if r.get("iteration_count") is not None]
    avg_iterations = statistics.mean(iterations) if iterations else 0.0
    
    # Calculate average time
    times = [r.get("time_taken", 0.0) for r in results if r.get("time_taken") is not None]
    avg_time = statistics.mean(times) if times else 0.0
    
    return {
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "pass_at_10": pass_at_10,
        "success_rate": success_rate,
        "avg_iterations": avg_iterations,
        "avg_time": avg_time,
        "total_problems": len(results),
        "solved": sum(successes)
    }

