# Quick Start: Using the Multi-Agent System

## 🚀 Getting Started

### 1. Set Your API Key
```powershell
$env:OPENAI_API_KEY="your_real_api_key_here"
```

### 2. Enable LLM Features

Edit `config/default.yaml`:

```yaml
extractor:
  use_llm: true  # Enable LLM algorithm extraction

generator:
  use_llm: true   # Enable LLM code generation
```

### 3. Test the System

```bash
python test_multi_agent.py
```

---

## 📝 Using the New Agents

### Option 1: Use Planner Agent (Recommended)

```python
from agents.planner_agent import PlannerAgent
from utils.intermediate_representation import PaperToCodeIR, PaperMetadata

# Create initial IR
ir = PaperToCodeIR(
    paper_id="paper_001",
    paper_metadata=PaperMetadata(title="My Paper"),
    paper_path="path/to/paper.pdf"
)

# Process with planner
planner = PlannerAgent(config={
    "use_paper_analysis": True,
    "agents": {
        "paper_analysis": {
            "use_llm": True,
            "llm_provider": "openai"
        }
    }
})

ir = planner.process(ir)

# Check results
print(f"Status: {ir.status}")
print(f"Algorithms found: {len(ir.algorithms)}")
for alg in ir.algorithms:
    print(f"  - {alg.name} (confidence: {alg.confidence:.2f})")
```

### Option 2: Use Individual Agents

```python
from agents.paper_analysis_agent import PaperAnalysisAgent
from generators.llm_code_generator import LLMCodeGenerator
from utils.intermediate_representation import PaperToCodeIR, PaperMetadata

# Step 1: Analyze paper
ir = PaperToCodeIR(
    paper_id="paper_001",
    paper_metadata=PaperMetadata(title="My Paper"),
    paper_path="paper.pdf"
)

analysis_agent = PaperAnalysisAgent(config={
    "use_llm": True,
    "llm_provider": "openai"
})

ir = analysis_agent.process(ir)

# Step 2: Generate code
code_generator = LLMCodeGenerator(config={
    "use_llm": True,
    "llm_provider": "openai"
})

# Convert AlgorithmInfo to Algorithm objects
from extractors.algorithm_extractor import Algorithm
algorithms = [
    Algorithm(
        name=alg.name,
        description=alg.description,
        parameters=alg.parameters,
        confidence=alg.confidence
    )
    for alg in ir.algorithms
]

code = code_generator.generate_code(algorithms, framework="pytorch")
print(code)
```

---

## 🌐 Using in Web App

The web app (`app.py`) now supports:
- ✅ LLM algorithm extraction (via config)
- ✅ LLM code generation (via config)

Just update `config/default.yaml`:
```yaml
extractor:
  use_llm: true

generator:
  use_llm: true
```

Then restart the Flask app and upload a paper!

---

## 🔍 What Each Agent Does

### Paper Analysis Agent
- Extracts text from PDF
- Identifies paper metadata (title, authors, year)
- Uses LLM to extract algorithms intelligently
- Extracts equations and mathematical notation
- Ready for diagram parsing (vision model integration)

### LLM Code Generator
- Generates code using LLM instead of templates
- Creates framework-specific implementations
- Produces clean, executable code
- Falls back to templates if LLM fails

### Planner Agent
- Orchestrates the entire pipeline
- Manages agent execution sequence
- Tracks progress and status
- Handles errors and recovery

---

## 📊 Intermediate Representation

All agents communicate via `PaperToCodeIR`:

```python
from utils.intermediate_representation import PaperToCodeIR

# Access extracted data
ir.algorithms          # List of AlgorithmInfo
ir.diagrams           # List of DiagramInfo
ir.equations          # List of equations
ir.paper_metadata     # PaperMetadata object
ir.status             # Current status
ir.current_agent      # Currently executing agent

# Serialize/deserialize
json_str = ir.to_json()
ir = PaperToCodeIR.from_json(json_str)
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Set environment variable: `$env:OPENAI_API_KEY="your_key"`
- Or create `.env` file with `OPENAI_API_KEY=your_key`

### "LLM extraction failed"
- Check API key is valid
- Verify you have API credits
- System will fall back to rule-based extraction automatically

### "No algorithms found"
- Paper may not contain clear algorithm descriptions
- Try enabling LLM extraction for better results
- Check paper text was extracted correctly

---

## 🎯 Next Steps

1. **Test with real papers** - Upload PDFs and see the agents in action
2. **Enable LLM features** - Get better algorithm detection and code generation
3. **Monitor costs** - Track API usage and optimize
4. **Extend agents** - Add more specialized agents as needed

---

## 💡 Tips

- Start with `use_llm: false` to test basic functionality
- Enable LLM features one at a time to see improvements
- Use the test suite to validate setup
- Check logs for detailed processing information

---

**Ready to process papers with the multi-agent system!** 🚀

