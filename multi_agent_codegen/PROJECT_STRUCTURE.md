# Project Structure

```
multi_agent_codegen/
├── README.md                      # Main project documentation
├── QUICK_START.md                 # Quick start guide
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
│
├── agents/                        # Multi-agent system
│   ├── __init__.py
│   ├── base_agent.py              # Base agent class
│   ├── coder_agent.py            # Code generation agent
│   ├── tester_agent.py           # Test generation and execution
│   ├── debugger_agent.py         # Error analysis and fixing
│   ├── optimizer_agent.py        # Code optimization and documentation
│   └── planner_agent.py          # Workflow orchestrator
│
├── workflow/                      # LangGraph workflow
│   ├── __init__.py
│   ├── graph.py                   # Main workflow graph definition
│   └── state.py                   # State management (TypedDict)
│
├── sandbox/                       # Sandbox execution
│   ├── __init__.py
│   └── executor.py                # Safe code execution (local/Docker)
│
├── git_utils/                     # Git integration
│   ├── __init__.py
│   └── repo_manager.py           # Git repository management
│
├── evaluation/                    # Evaluation harness
│   ├── __init__.py
│   ├── benchmarks.py             # Benchmark datasets (HumanEval, LeetCode, APPS)
│   ├── metrics.py                 # Evaluation metrics (pass@k, etc.)
│   └── run_benchmark.py          # Benchmark runner
│
├── utils/                         # Utilities
│   └── __init__.py               # Helper functions and LLM client imports
│
└── main.py                        # Main entry point
```

## Key Components

### Agents (`agents/`)

- **BaseAgent**: Abstract base class for all agents
- **CoderAgent**: Generates Python code from specifications using LLM
- **TesterAgent**: Generates unit tests and executes them in sandbox
- **DebuggerAgent**: Analyzes test failures and suggests fixes
- **OptimizerAgent**: Optimizes code and adds documentation
- **PlannerAgent**: Orchestrates workflow and decides next agent

### Workflow (`workflow/`)

- **graph.py**: Defines LangGraph workflow with nodes and edges
- **state.py**: TypedDict for workflow state management
- Supports conditional routing based on test results

### Sandbox (`sandbox/`)

- **executor.py**: Executes code safely in isolated environment
- Supports both local execution and Docker-based isolation
- Handles timeouts and resource limits

### Git Utils (`git_utils/`)

- **repo_manager.py**: Manages Git repository for version tracking
- Automatically commits code after each iteration
- Tracks code history and changes

### Evaluation (`evaluation/`)

- **benchmarks.py**: Loads benchmark datasets
- **metrics.py**: Calculates pass@k and other metrics
- **run_benchmark.py**: Runs evaluation on benchmarks

## Data Flow

1. **Input**: Specification (natural language or function stub)
2. **Coder Agent**: Generates initial code
3. **Tester Agent**: Generates tests and executes them
4. **Conditional Routing**:
   - If tests pass → Optimizer Agent (if enabled) → End
   - If tests fail → Debugger Agent → Tester Agent (loop)
5. **Output**: Final code with tests, iteration count, success status

## State Management

The workflow state (`WorkflowState`) contains:
- `specification`: Original problem description
- `code`: Current code version
- `code_history`: All code versions
- `test_results`: Test execution results
- `tests_passed`: Boolean flag
- `iteration`: Current iteration number
- `max_iterations`: Maximum allowed iterations
- `fix_history`: History of fixes applied
- `optimized`: Whether code has been optimized

## Configuration

See `config.yaml` for:
- LLM provider and model settings
- Workflow parameters (max iterations, enable optimizer)
- Sandbox configuration (type, timeout, resource limits)
- Agent-specific settings (temperature, max_tokens)
- Git integration settings

