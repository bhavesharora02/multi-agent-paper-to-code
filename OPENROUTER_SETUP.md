# OpenRouter AI Integration Setup

## ✅ Integration Complete!

OpenRouter AI has been successfully integrated into the multi-agent pipeline!

---

## 🔑 Setup Instructions

### 1. Set Your OpenRouter API Key

**Windows PowerShell:**
```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

**Windows Command Prompt:**
```cmd
set OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 2. Update Configuration

Edit `config/default.yaml`:

```yaml
extractor:
  use_llm: true
  llm_provider: "openrouter"  # Use OpenRouter

generator:
  use_llm: true
  llm_provider: "openrouter"  # Use OpenRouter
```

### 3. Test the Integration

```bash
python test_openrouter.py
```

---

## 🎯 Using OpenRouter

### Available Models

OpenRouter supports many models. You can specify them in the config:

- `openai/gpt-4o` - GPT-4 Omni (default)
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo (cheaper)
- `anthropic/claude-3-opus` - Claude 3 Opus
- `anthropic/claude-3-sonnet` - Claude 3 Sonnet
- `google/gemini-pro` - Google Gemini
- And many more!

### Model Selection

You can specify the model when initializing the client:

```python
from llm.llm_client import LLMClient, LLMProvider

# Use GPT-4o (default)
client = LLMClient(provider=LLMProvider.OPENROUTER)

# Use a specific model
client = LLMClient(
    provider=LLMProvider.OPENROUTER,
    model="openai/gpt-3.5-turbo"  # Cheaper option
)
```

---

## 💰 Cost Considerations

OpenRouter provides:
- **Pay-as-you-go** pricing
- **Multiple model options** (some cheaper than direct APIs)
- **Unified API** for all models
- **Cost transparency** - see pricing at https://openrouter.ai/models

### Cost Optimization Tips

1. **Use GPT-3.5-turbo** for simpler tasks (much cheaper)
2. **Use GPT-4o** for complex algorithm extraction
3. **Monitor usage** on OpenRouter dashboard
4. **Set budget limits** in OpenRouter settings

---

## 🔧 Configuration Options

### In Code

```python
from agents.planner_agent import PlannerAgent

planner = PlannerAgent(config={
    "agents": {
        "paper_analysis": {
            "use_llm": True,
            "llm_provider": "openrouter",
            "model": "openai/gpt-4o"  # Optional model override
        }
    }
})
```

### In YAML Config

```yaml
extractor:
  use_llm: true
  llm_provider: "openrouter"
  model: "openai/gpt-4o"  # Optional

generator:
  use_llm: true
  llm_provider: "openrouter"
  model: "openai/gpt-4o"  # Optional
```

---

## 🐛 Troubleshooting

### "402 Payment Required"
- Your OpenRouter account needs credits
- Add credits at https://openrouter.ai/credits
- Check your account balance

### "Invalid API Key"
- Verify the API key is correct
- Check it starts with `sk-or-v1-`
- Ensure no extra spaces

### "Model not found"
- Check model name is correct
- Use format: `provider/model-name`
- See available models at https://openrouter.ai/models

---

## ✅ Benefits of OpenRouter

1. **Multiple Models** - Access to OpenAI, Anthropic, Google, and more
2. **Unified API** - Same interface for all models
3. **Cost Flexibility** - Choose cheaper models when appropriate
4. **Easy Switching** - Change models without code changes
5. **Rate Limiting** - Built-in rate limiting and retries

---

## 🚀 Next Steps

1. **Add Credits** to your OpenRouter account
2. **Test Integration** with `python test_openrouter.py`
3. **Enable LLM Features** in config
4. **Test with Real Papers** to see LLM-powered extraction

---

## 📝 Example Usage

```python
from llm.llm_client import LLMClient, LLMProvider

# Initialize with OpenRouter
client = LLMClient(provider=LLMProvider.OPENROUTER)

# Generate text
response = client.generate(
    prompt="Extract algorithms from this paper text...",
    system_prompt="You are an expert in ML/DL algorithms."
)

# Generate JSON
json_response = client.generate_json(
    prompt="Extract algorithms as JSON...",
    temperature=0.3
)
```

---

**Status: OpenRouter Integration Complete!** ✅

Once you add credits to your OpenRouter account, the system will work with full LLM capabilities!

