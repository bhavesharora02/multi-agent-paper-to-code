# Quick Start: Using OpenRouter AI

## ✅ OpenRouter Integration Complete!

Your multi-agent pipeline now supports OpenRouter AI with your API key!

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Set API Key
```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### Step 2: Enable LLM in Config
Edit `config/default.yaml`:
```yaml
extractor:
  use_llm: true
  llm_provider: "openrouter"

generator:
  use_llm: true
  llm_provider: "openrouter"
```

### Step 3: Add Credits
1. Go to https://openrouter.ai/credits
2. Add credits to your account
3. Start using!

---

## 🎯 Test It

```bash
# Test OpenRouter integration
python test_openrouter.py

# Test complete pipeline
python COMPLETE_PIPELINE_DEMO.py

# Test with web app
python app.py
# Then visit http://localhost:5000
```

---

## 💡 What You Get

✅ **Access to 100+ Models** - OpenAI, Anthropic, Google, etc.  
✅ **Cost Flexibility** - Choose cheaper models when needed  
✅ **Unified API** - Same code works with all models  
✅ **Better Pricing** - Often cheaper than direct APIs  

---

## 📝 Your API Key

```
your_openrouter_api_key_here
```

**Status:** ✅ Configured and ready  
**Next:** Add credits to start using!

---

**Ready to use OpenRouter AI!** 🚀

