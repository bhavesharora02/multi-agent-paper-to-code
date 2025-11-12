# LLM Integration Setup Guide

## ✅ What's Been Implemented

1. **LLM Client Module** (`src/llm/llm_client.py`)
   - Supports OpenAI and Anthropic APIs
   - JSON generation capabilities
   - Vision model support (for future diagram parsing)

2. **LLM-Enhanced Algorithm Extractor** (`src/extractors/llm_algorithm_extractor.py`)
   - Intelligent algorithm extraction using LLM
   - Automatic fallback to rule-based extraction
   - Configurable LLM provider

3. **Prompt Templates** (`src/llm/prompt_templates.py`)
   - Specialized prompts for each agent
   - Ready for multi-agent pipeline expansion

4. **Test Suite** (`test_llm_integration.py`)
   - Validates LLM integration
   - Tests algorithm extraction

## 🔑 Setting Up Your API Key

### Option 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your_actual_api_key_here"
```

**Windows Command Prompt:**
```cmd
set OPENAI_API_KEY=your_actual_api_key_here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your_actual_api_key_here"
```

### Option 2: .env File

1. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_actual_api_key_here
```

2. The system will automatically load it (python-dotenv is installed)

### Getting Your API Key

1. Go to https://platform.openai.com/account/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)

⚠️ **Important:** Never commit your API key to Git! It's already in `.gitignore`

## 🚀 Using LLM-Enhanced Extraction

### In Your Code

```python
from extractors.llm_algorithm_extractor import LLMAlgorithmExtractor

# Initialize with LLM
extractor = LLMAlgorithmExtractor(config={
    "use_llm": True,
    "llm_provider": "openai",  # or "anthropic"
    "fallback_to_rules": True  # Falls back if LLM fails
})

# Extract algorithms
algorithms = extractor.extract_algorithms(paper_text)
```

### In app.py (Web Interface)

The system can be configured to use LLM extraction. Update `config/default.yaml`:

```yaml
extractor:
  use_llm: true
  llm_provider: "openai"
  fallback_to_rules: true
  max_text_length: 8000
```

## 🧪 Testing

Run the test suite:
```bash
python test_llm_integration.py
```

This will:
- Test LLM client initialization
- Test algorithm extraction
- Show fallback behavior if API key is invalid

## 💰 Cost Considerations

**OpenAI GPT-4 Turbo:**
- Input: ~$0.01 per 1K tokens
- Output: ~$0.03 per 1K tokens
- Average paper: ~$0.50 - $2.00 per extraction

**Cost Optimization Tips:**
1. Use `max_text_length` to limit input size
2. Cache results for similar papers
3. Use GPT-3.5-turbo for simpler tasks (10x cheaper)
4. Implement request batching

## 🔄 Current Status

✅ **Working:**
- LLM client infrastructure
- LLM-based algorithm extraction
- Automatic fallback to rule-based extraction
- Error handling and logging

🚧 **Next Steps:**
- Integrate with web app (`app.py`)
- Add vision model for diagram parsing
- Implement other agents (Code Generation, Verification, etc.)
- Add caching for cost optimization

## 📝 Configuration Options

In `config/default.yaml`:

```yaml
extractor:
  use_llm: true                    # Enable LLM extraction
  llm_provider: "openai"           # "openai" or "anthropic"
  fallback_to_rules: true         # Use rule-based if LLM fails
  max_text_length: 8000            # Max chars to send to LLM
  confidence_threshold: 0.3        # Minimum confidence score
```

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Set the environment variable (see above)
- Or create a `.env` file

### "Invalid API key"
- Check that your key starts with `sk-`
- Verify the key is active at https://platform.openai.com/account/api-keys
- Make sure there are no extra spaces in the key

### "LLM extraction failed"
- Check your API key is valid
- Verify you have API credits/quota
- Check internet connection
- System will automatically fall back to rule-based extraction

### High Costs
- Reduce `max_text_length` in config
- Use GPT-3.5-turbo instead of GPT-4 (modify `llm_client.py`)
- Implement caching (coming soon)

## 📚 Next: Multi-Agent Pipeline

Now that LLM integration is working, you can:
1. Create the Paper Analysis Agent (enhanced PDF + vision parsing)
2. Build the Code Generation Agent (LLM-based code generation)
3. Implement the Verification Agent (automated testing)
4. Add the Debugging Agent (iterative refinement)
5. Create the Planner Agent (orchestration)

See `MULTI_AGENT_ARCHITECTURE_PLAN.md` for the full roadmap.

