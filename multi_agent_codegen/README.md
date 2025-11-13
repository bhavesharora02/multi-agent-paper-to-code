# Collaborative Code Generation and Debugging Agents via Multi-Agent LLM Pipelines

## 🎯 Project Overview

This project implements a multi-agent LLM framework that mimics human software development workflows through specialist agents—Coder, Tester, Debugger, and Optimizer. The system uses LangGraph for orchestration and executes iterative cycles of code generation, testing, debugging, and optimization until correctness is achieved.

**Author**: Kanishka Dhindhwal (M24DE3043)  
**Project**: Major Technical Project (MTP)  
**Institution**: M. Tech in Data Engineering

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Coder Agent**: Generates function-level code from specifications using GPT-4o or Code LLaMA
- **Tester Agent**: Auto-generates unit tests (including edge cases) and executes code in a secure Python sandbox
- **Debugger Agent**: Interprets tracebacks and tester feedback, suggests targeted code edits
- **Optimizer/Explainer Agent**: Reviews final code for efficiency and documents logic via docstrings/comments
- **Planner Agent**: Oversees the workflow, invokes agents, logs intermediate states, and halts when tests pass or max iterations reached

### 🔧 Technical Capabilities
- **LangGraph Orchestration**: Graph-based workflow with conditional routing based on test results
- **Sandbox Execution**: Secure Python execution environment for testing
- **Git Integration**: Version tracking of code changes per iteration
- **Iterative Feedback Loop**: Continues until tests pass or resource/time limits reached
- **Evaluation Harness**: Benchmarks on HumanEval, APPS, LeetCode Easy with pass@k metrics

---

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)
- LLM API key (OpenAI, Anthropic, OpenRouter, or Groq)
- Docker (optional, for sandbox execution)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```powershell
# For OpenAI
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Or for Groq (recommended - fast and free tier available)
$env:GROQ_API_KEY="your_groq_api_key_here"

# Or for OpenRouter
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Or for Anthropic
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### 3. Run the System
```bash
# Run a single problem
python -m multi_agent_codegen.main --spec "Implement is_palindrome function"

# Run evaluation on benchmarks
python -m multi_agent_codegen.evaluation.run_benchmark --benchmark humaneval --num_problems 20
```

---

## 📁 Project Structure

```
multi_agent_codegen/
├── agents/                      # Multi-agent system
│   ├── __init__.py
│   ├── base_agent.py            # Base agent class
│   ├── coder_agent.py           # Code generation agent
│   ├── tester_agent.py          # Test generation and execution agent
│   ├── debugger_agent.py        # Error analysis and fixing agent
│   ├── optimizer_agent.py       # Code optimization and documentation agent
│   └── planner_agent.py         # Workflow orchestrator
├── workflow/                    # LangGraph workflow
│   ├── __init__.py
│   ├── graph.py                 # Main workflow graph definition
│   └── state.py                 # State management
├── sandbox/                     # Sandbox execution
│   ├── __init__.py
│   ├── executor.py              # Safe code execution
│   └── docker_executor.py       # Docker-based execution (optional)
├── git_utils/                   # Git integration
│   ├── __init__.py
│   └── repo_manager.py          # Git repository management
├── evaluation/                  # Evaluation harness
│   ├── __init__.py
│   ├── benchmarks.py            # Benchmark datasets
│   ├── metrics.py               # pass@k and other metrics
│   └── run_benchmark.py         # Benchmark runner
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── llm_client.py            # LLM client (reused from parent)
│   └── helpers.py               # Helper functions
├── main.py                      # Main entry point
└── config.yaml                  # Configuration file
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:
- **LLM Provider**: Choose between OpenAI, Anthropic, OpenRouter, or Groq
- **Model Selection**: Specify which model to use
- **Max Iterations**: Maximum number of debug cycles
- **Sandbox Type**: Local or Docker-based execution
- **Git Integration**: Enable/disable version tracking

---

## 🎬 Usage Examples

### Basic Usage
```python
from multi_agent_codegen.workflow.graph import create_workflow
from multi_agent_codegen.workflow.state import WorkflowState

# Create workflow
workflow = create_workflow()

# Initialize state
state = WorkflowState(
    specification="Implement a function to check if a string is a palindrome",
    max_iterations=5
)

# Run workflow
result = workflow.invoke(state)
print(f"Final code: {result.code}")
print(f"Iterations: {result.iteration_count}")
print(f"Tests passed: {result.tests_passed}")
```

### Evaluation on Benchmarks
```bash
# Run HumanEval benchmark
python -m multi_agent_codegen.evaluation.run_benchmark \
    --benchmark humaneval \
    --num_problems 20 \
    --max_iterations 5 \
    --output results.json
```

---

## 📊 Evaluation Metrics

- **pass@k**: Probability that at least one of k generated solutions passes all tests
- **Average Iteration Count**: Mean number of iterations needed to achieve correctness
- **Test Coverage**: Percentage of code covered by generated tests
- **Success Rate**: Percentage of problems solved within max iterations

---

## 🔬 System Architecture

1. **User/Planner Input Layer**: Receives specification (natural language or stub)
2. **Agentic Workflow (LangGraph)**: Nodes for Coder, Tester, Debugger, Optimizer with conditional routing
3. **Tool Integration**: Python Sandbox, Git, LLM APIs
4. **Evaluation Harness**: Benchmarks and metrics
5. **Feedback Loop**: Failed tests trigger error analysis and patch suggestions
6. **Repository Packaging**: Final code committed with CI-compatible test suite

---

## 🛡️ Security Notes

- **Sandbox Execution**: Code is executed in isolated environments
- **Never commit API keys**: Use environment variables
- **Resource Limits**: Time and memory limits enforced in sandbox

---

## 📝 Development Phases

### MTP-1: Basic Loop Construction
- Implement Planner + Coder + Tester agents
- Test on fixed prompt set (e.g., 20 starter problems)
- Metrics: code generation success (pass@1), iteration count

### MTP-2: Add Debugger Loop
- Introduce Debugger: parse errors, suggest fixes, re-commit
- Measure improvement in pass rates after multiple cycles
- Add Optimizer/Explainer
- Apply to more complex problems
- Compare with single-LLM baseline and academic benchmarks

---

## 🤝 Contributing

This is a thesis project. For questions or suggestions, please contact:
- **Email**: [Your Email]
- **Student ID**: M24DE3043

---

## 📄 License

This project is part of academic research. Please cite appropriately if used.

---

## 🙏 Acknowledgments

- OpenAI, Anthropic, OpenRouter, and Groq for LLM APIs
- LangGraph team for workflow orchestration framework
- Research community for inspiration and feedback

