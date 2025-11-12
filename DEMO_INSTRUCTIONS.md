# 🎬 Demo Instructions - Multi-Agent Pipeline

## Quick Demo Guide

### Step 1: Start the Web Interface

```powershell
# Set API key
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Start Flask app
python app.py
```

### Step 2: Open Browser

Visit: **http://localhost:5000**

### Step 3: Upload a Paper

1. Click "Choose File" or drag & drop a PDF
2. Select framework (PyTorch, TensorFlow, or Scikit-learn)
3. Click "Upload and Process"

### Step 4: Watch the Magic! ✨

- Real-time progress updates
- Algorithm extraction
- Code generation
- Download generated code

---

## What to Expect

### With Free Model:
- ✅ System works
- ⚠️ May hit rate limits (429 errors)
- ✅ Automatically falls back to rule-based extraction
- ✅ Still generates code successfully

### What You'll See:
1. **Processing Status** - Real-time updates
2. **Algorithms Found** - List of detected algorithms
3. **Generated Code** - Downloadable Python file
4. **Results** - Complete processing summary

---

## Best Papers to Test

For best results, use papers with:
- Clear algorithm descriptions
- Neural network architectures
- ML/DL methodologies
- Pseudocode or equations

Examples:
- Transformer papers
- CNN/RNN papers
- Attention mechanism papers
- GAN papers

---

## Troubleshooting

### Rate Limit Errors (429)
- **Normal** for free models
- System automatically falls back
- Wait a few minutes and try again
- Or use web interface (better UX)

### No Algorithms Found
- Paper might not have clear algorithms
- Try a different paper
- System still generates code (template-based)

### Processing Takes Time
- Normal for LLM processing
- Free models are slower
- Progress bar shows status

---

## Demo Script

1. **Show Web Interface** - Modern UI
2. **Upload Paper** - Drag & drop
3. **Watch Processing** - Real-time updates
4. **Show Results** - Algorithms found
5. **Download Code** - Generated Python file
6. **Explain Architecture** - Multi-agent system

---

**Ready to demo!** 🚀

