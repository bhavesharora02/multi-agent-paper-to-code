"""
Prompt templates for LLM agents.
"""

# Paper Analysis Agent Prompts
PAPER_ANALYSIS_SYSTEM_PROMPT = """You are an expert in machine learning and deep learning research. 
Your task is to analyze research papers and extract structured information about algorithms, 
models, and methodologies. Be precise, thorough, and focus on technical details."""

PAPER_ANALYSIS_PROMPT = """Analyze the following research paper text and extract all machine learning 
and deep learning algorithms, models, and methodologies mentioned.

For each algorithm/model found, provide:
1. **name**: The official name of the algorithm/model
2. **description**: A clear description of what it does
3. **type**: Category (e.g., "neural_network", "supervised_learning", "optimization", "regularization")
4. **key_parameters**: List of important hyperparameters or parameters mentioned
5. **mathematical_notation**: Any equations or mathematical formulas (if present)
6. **pseudocode**: Any pseudocode or algorithmic steps (if present)
7. **framework_suggestions**: Recommended frameworks (PyTorch, TensorFlow, scikit-learn, etc.)
8. **complexity**: Time/space complexity if mentioned
9. **confidence**: Your confidence in identifying this algorithm (0.0-1.0)

Paper text:
{text}

Return the results as a JSON array of algorithm objects. If no algorithms are found, return an empty array."""

# Algorithm Interpretation Agent Prompts
ALGORITHM_INTERPRETATION_SYSTEM_PROMPT = """You are an expert at translating mathematical notation 
and pseudocode into explicit computational workflows. You understand complex ML/DL algorithms 
and can break them down into implementable steps."""

ALGORITHM_INTERPRETATION_PROMPT = """Translate the following algorithm description into a structured 
computational workflow.

Algorithm Information:
{algorithm_info}

Provide:
1. **workflow_steps**: Ordered list of computational steps
2. **data_dependencies**: What data each step requires
3. **control_flow**: Any loops, conditionals, or iterative procedures
4. **implementation_notes**: Important implementation details
5. **edge_cases**: Potential edge cases to handle

Return as JSON."""

# Code Generation Prompts
CODE_GENERATION_SYSTEM_PROMPT = """You are an expert Python developer specializing in machine learning 
and deep learning frameworks. Generate clean, well-documented, production-ready code."""

CODE_GENERATION_PROMPT = """Generate a complete, runnable Python implementation of the following 
algorithm using {framework}.

Algorithm Specification:
{algorithm_spec}

Requirements:
1. Include all necessary imports
2. Create a class for the model/algorithm with proper structure
3. Include training/evaluation methods if applicable
4. Add comprehensive docstrings
5. Include example usage in a main block
6. Handle edge cases and errors
7. Make it production-ready and well-commented

Generate complete, executable code:"""

# API/Library Mapping Prompts
API_MAPPING_SYSTEM_PROMPT = """You are an expert at mapping ML/DL algorithm components to 
appropriate libraries and frameworks. You know the APIs of PyTorch, TensorFlow, scikit-learn, 
and other ML libraries."""

API_MAPPING_PROMPT = """Map the following algorithm components to appropriate {framework} libraries 
and functions.

Algorithm Components:
{components}

For each component, provide:
1. **library**: The library/module to use
2. **function/class**: Specific function or class name
3. **parameters**: Required parameters
4. **documentation_reference**: Link or reference to documentation
5. **code_snippet**: Example usage snippet

Return as JSON array."""

# Verification Prompts
VERIFICATION_SYSTEM_PROMPT = """You are an expert at analyzing code execution results and comparing 
them against expected outcomes. You can identify discrepancies and potential issues."""

VERIFICATION_PROMPT = """Compare the following execution results against the paper's reported metrics.

Paper Metrics:
{paper_metrics}

Actual Results:
{actual_results}

Provide:
1. **status**: "pass" or "fail"
2. **metric_comparison**: Detailed comparison of each metric
3. **discrepancies**: Any significant differences
4. **tolerance_check**: Whether differences are within acceptable tolerance
5. **potential_issues**: Possible reasons for discrepancies

Return as JSON."""

# Debugging Prompts
DEBUGGING_SYSTEM_PROMPT = """You are an expert debugger specializing in ML/DL code. You can analyze 
errors, identify root causes, and suggest targeted fixes."""

DEBUGGING_PROMPT = """Analyze the following error and suggest fixes.

Error Information:
{error_info}

Code Context:
{code_context}

Verification Results:
{verification_results}

Provide:
1. **root_cause**: Likely cause of the issue
2. **severity**: "critical", "high", "medium", or "low"
3. **suggested_fixes**: List of specific fixes to try
4. **code_changes**: Specific code modifications needed
5. **testing_approach**: How to verify the fix works

Return as JSON."""

