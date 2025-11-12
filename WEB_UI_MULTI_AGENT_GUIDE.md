# Web UI - Multi-Agent Pipeline Guide

## ✅ Web App Updated!

Your web interface now uses the **complete multi-agent pipeline** with all 7 agents!

---

## 🚀 How to Use

### Step 1: Set API Key

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### Step 2: Enable Multi-Agent Pipeline

The config is already set! Check `config/default.yaml`:
```yaml
use_multi_agent_pipeline: true  # ✅ Enabled!
```

### Step 3: Start Web App

```bash
python app.py
```

### Step 4: Open Browser

Visit: **http://localhost:5000**

---

## 🎯 What Happens Now

When you upload a paper, the system uses the **complete multi-agent pipeline**:

1. **Paper Analysis Agent** - Extracts text, metadata, algorithms
2. **Algorithm Interpretation Agent** - Interprets algorithms
3. **API Mapping Agent** - Maps to framework APIs
4. **Code Integration Agent** - Generates complete codebase
5. **Verification Agent** - (Optional) Validates code
6. **Debugging Agent** - (Optional) Refines if needed

All orchestrated by the **Planner Agent**!

---

## 📊 What You'll See

### Progress Updates:
- "Initializing multi-agent pipeline..."
- "Phase 1: Paper Analysis..."
- "Found X algorithms. Phase 2: Interpretation..."
- "Phase 3: API Mapping..."
- "Phase 4: Code Integration..."
- "Multi-agent pipeline completed!"

### Results Include:
- Algorithms found (with details)
- Mapped components count
- Generated files count
- Pipeline status
- Current agent executing

---

## ⚙️ Configuration Options

### Enable/Disable Multi-Agent Pipeline

Edit `config/default.yaml`:
```yaml
use_multi_agent_pipeline: true  # Use complete pipeline
# or
use_multi_agent_pipeline: false  # Use legacy method
```

### Enable Verification & Debugging

```yaml
use_verification: true  # Enable verification agent
use_debugging: true     # Enable debugging agent
```

### LLM Settings

```yaml
extractor:
  use_llm: true
  llm_provider: "openrouter"
  model: "openai/gpt-oss-20b:free"  # Free model

generator:
  use_llm: true
  llm_provider: "openrouter"
  model: "openai/gpt-oss-20b:free"
```

---

## 🎬 Demo Flow

1. **Upload Paper** - Drag & drop PDF
2. **Select Framework** - PyTorch, TensorFlow, or Scikit-learn
3. **Watch Progress** - See each agent phase
4. **View Results** - Algorithms, mapped components, files
5. **Download Code** - Complete generated codebase

---

## 💡 What's Different

### Old Method:
- Direct extractor → generator
- Simple workflow
- Basic results

### New Multi-Agent Method:
- ✅ Complete 7-agent pipeline
- ✅ Algorithm interpretation
- ✅ API mapping
- ✅ Code integration
- ✅ Repository structure
- ✅ Detailed results

---

## 🔍 Check Results

After processing, you'll see:
- **Algorithms Found** - List with details
- **Mapped Components** - Framework mappings
- **Generated Files** - Complete codebase
- **Pipeline Status** - Current phase
- **Current Agent** - Which agent is working

---

## 🎓 For Your Thesis

You can now demonstrate:
- ✅ Complete multi-agent architecture
- ✅ All 7 agents working together
- ✅ Planner agent orchestration
- ✅ Real-time progress tracking
- ✅ Detailed results

---

**Your web UI now uses the complete multi-agent pipeline!** 🚀

Visit http://localhost:5000 and upload a paper to see it in action!

