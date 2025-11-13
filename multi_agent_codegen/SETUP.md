# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version tracking)
- Docker (optional, for Docker-based sandbox)

## Step-by-Step Setup

### 1. Navigate to Project Directory

```bash
cd multi_agent_codegen
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with LangGraph, you can install it separately:
```bash
pip install langgraph langchain
```

### 4. Set Up LLM API Key

Choose one of the following providers:

#### Option A: Groq (Recommended - Fast and Free Tier)
```powershell
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

```bash
# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"
```

#### Option B: OpenAI
```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

```bash
# Linux/Mac
export OPENAI_API_KEY="your_openai_api_key_here"
```

#### Option C: OpenRouter
```powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

#### Option D: Anthropic
```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

**To make environment variables persistent**, add them to your shell profile:
- Windows: Add to System Environment Variables
- Linux/Mac: Add to `~/.bashrc` or `~/.zshrc`

### 5. Verify Installation

Test that everything is set up correctly:

```bash
python -m multi_agent_codegen.main --spec "def hello(): return 'world'"
```

If you see generated code output, the setup is successful!

### 6. (Optional) Install Docker for Sandbox

If you want to use Docker-based sandbox execution:

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Start Docker Desktop
3. Update `config.yaml`:
   ```yaml
   sandbox:
     type: "docker"  # Change from "local" to "docker"
   ```

### 7. (Optional) Configure Git Integration

Git integration is enabled by default. To disable:

Edit `config.yaml`:
```yaml
workflow:
  enable_git: false
```

## Troubleshooting

### Import Errors

If you see errors like `ModuleNotFoundError: No module named 'llm'`:

1. Ensure the parent project's `src` directory is accessible
2. Or install LLM dependencies directly:
   ```bash
   pip install openai anthropic requests
   ```

### LangGraph Not Found

If LangGraph is not installed:
```bash
pip install langgraph langchain
```

The system will automatically fall back to a simple workflow if LangGraph is unavailable.

### Pytest Not Found

Install pytest:
```bash
pip install pytest
```

### Docker Issues

If Docker sandbox fails, the system automatically falls back to local execution. Ensure Docker is installed and running if you want to use it.

## Configuration

Edit `config.yaml` to customize:

- **LLM Provider**: Change `llm.provider`
- **Model**: Change `llm.model`
- **Max Iterations**: Adjust `workflow.max_iterations`
- **Sandbox Type**: Switch between `local` and `docker`
- **Agent Settings**: Customize per-agent temperature and max_tokens

## Next Steps

1. Read `QUICK_START.md` for usage examples
2. Read `README.md` for detailed documentation
3. Try running a simple problem:
   ```bash
   python -m multi_agent_codegen.main --spec "Implement is_palindrome function"
   ```

## Getting API Keys

### Groq (Recommended)
1. Visit https://console.groq.com/
2. Sign up for free account
3. Generate API key from dashboard

### OpenAI
1. Visit https://platform.openai.com/
2. Sign up and add payment method
3. Generate API key from settings

### OpenRouter
1. Visit https://openrouter.ai/
2. Sign up for account
3. Generate API key from dashboard

### Anthropic
1. Visit https://console.anthropic.com/
2. Sign up for account
3. Generate API key from settings

## Support

For issues or questions:
- Check `README.md` for detailed documentation
- Review `QUICK_START.md` for examples
- Check `PROJECT_STRUCTURE.md` for architecture details

