# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
cd multi_agent_codegen
pip install -r requirements.txt
```

2. **Set up LLM API key:**
```powershell
# For Groq (recommended - fast and free tier)
$env:GROQ_API_KEY="your_groq_api_key_here"

# Or for OpenAI
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Or for OpenRouter
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Or for Anthropic
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## Basic Usage

### Run a single problem:

```bash
python -m multi_agent_codegen.main --spec "Implement a function to check if a string is a palindrome"
```

### Run with custom configuration:

```bash
python -m multi_agent_codegen.main \
    --spec "Implement two_sum function" \
    --max-iterations 5 \
    --output generated_code.py \
    --verbose
```

### Run evaluation on benchmarks:

```bash
# Run HumanEval benchmark
python -m multi_agent_codegen.evaluation.run_benchmark \
    --benchmark humaneval \
    --num-problems 20 \
    --max-iterations 10 \
    --output results.json
```

## Example Use Cases

### 1. Simple Function Generation

```bash
python -m multi_agent_codegen.main \
    --spec "def is_palindrome(s: str) -> bool:
    Check if a string is a palindrome. Should handle empty strings and single characters."
```

### 2. LeetCode-style Problem

```bash
python -m multi_agent_codegen.main \
    --spec "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution."
```

### 3. Complex Algorithm

```bash
python -m multi_agent_codegen.main \
    --spec "Implement a binary search tree with insert, search, and delete operations" \
    --max-iterations 15
```

## Configuration

Edit `config.yaml` to customize:

- **LLM Provider**: Change `llm.provider` to use different LLM
- **Max Iterations**: Adjust `workflow.max_iterations`
- **Sandbox Type**: Switch between `local` and `docker` execution
- **Agent Settings**: Customize temperature and max_tokens per agent

## Understanding the Output

The system will:
1. Generate code from specification (Coder Agent)
2. Create and run tests (Tester Agent)
3. Debug and fix errors (Debugger Agent)
4. Optimize and document code (Optimizer Agent)

Final output includes:
- Generated code
- Number of iterations
- Test results
- Success status

## Troubleshooting

### Import Errors
If you see import errors for `llm.llm_client`, ensure the parent project's `src` directory is accessible, or install the LLM dependencies directly.

### LangGraph Not Found
Install LangGraph:
```bash
pip install langgraph langchain
```

### Docker Sandbox Issues
If Docker sandbox fails, it will automatically fall back to local execution. Ensure Docker is installed and running if you want to use it.

### Test Execution Failures
Check that `pytest` is installed:
```bash
pip install pytest
```

## Next Steps

- Read the full README.md for detailed documentation
- Explore the `config.yaml` for customization options
- Check `evaluation/` for benchmark evaluation tools
- Review agent implementations in `agents/` directory

