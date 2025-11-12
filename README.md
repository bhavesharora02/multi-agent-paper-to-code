# Multi-Agent LLM Pipeline for ML/DL Paper-to-Code Translation

## 🎯 Project Overview

This project implements an automated system that translates machine learning and deep learning research papers into fully runnable code repositories using a sophisticated multi-agent LLM pipeline. The system employs specialized AI agents for paper analysis, algorithm interpretation, API/library mapping, code integration, verification, and iterative debugging.

**Author**: Bhavesh Arora (M24DE3022)  
**Project**: Major Technical Project 1 (MTP1)  
**Institution**: M. Tech in Data Engineering

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Paper Analysis Agent**: Extracts algorithms, mathematical notation, and implementation details from PDFs
- **Algorithm Interpretation Agent**: Translates mathematical notation into computational workflows
- **API/Library Mapping Agent**: Maps components to ML frameworks (PyTorch, TensorFlow, Scikit-learn)
- **Code Integration Agent**: Generates complete, production-ready code repositories
- **Verification Agent**: Executes and validates generated code
- **Debugging Agent**: Performs static analysis and automatically fixes code issues
- **Planner Agent**: Orchestrates the entire pipeline

### 🔧 Technical Capabilities
- **LLM Integration**: Supports OpenAI, Anthropic, OpenRouter, and Groq APIs
- **Vision Parsing**: Extracts and analyzes diagrams from research papers (when using vision-capable models)
- **Multi-Framework Support**: Generates code for PyTorch, TensorFlow, and Scikit-learn
- **Web Interface**: User-friendly Flask-based UI with real-time progress tracking
- **Error Handling**: Robust fallback mechanisms and graceful error recovery
- **Code Quality**: Automatic syntax checking, logical error detection, and best practice enforcement

---

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)
- LLM API key (OpenAI, Anthropic, OpenRouter, or Groq)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repository-name>
```

### 2. Set Up Virtual Environment
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

### 4. Set API Key
```powershell
# For Groq (recommended - fast and free tier available)
$env:GROQ_API_KEY="your_groq_api_key_here"

# Or for OpenRouter
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Or for OpenAI
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Or for Anthropic
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### 5. Run the Web Interface
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

---

## 📁 Project Structure

```
.
├── app.py                          # Flask web application
├── src/
│   ├── agents/                     # Multi-agent system
│   │   ├── base_agent.py           # Base agent class
│   │   ├── planner_agent.py        # Pipeline orchestrator
│   │   ├── paper_analysis_agent.py # Paper extraction
│   │   ├── algorithm_interpretation_agent.py
│   │   ├── api_mapping_agent.py
│   │   ├── code_integration_agent.py
│   │   ├── verification_agent.py  # Code execution & validation
│   │   └── debugging_agent.py     # Static analysis & fixes
│   ├── extractors/                 # Algorithm extraction
│   ├── generators/                 # Code generation
│   ├── llm/                        # LLM client integration
│   ├── parsers/                    # PDF parsing
│   └── utils/                      # Utilities & IR
├── config/
│   └── default.yaml                # Configuration file
├── templates/                      # HTML templates
├── static/                         # CSS, JS, assets
├── uploads/                        # Uploaded PDFs (gitignored)
├── outputs/                        # Generated code (gitignored)
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Configuration

Edit `config/default.yaml` to customize:

- **LLM Provider**: Choose between OpenAI, Anthropic, OpenRouter, or Groq
- **Model Selection**: Specify which model to use
- **Pipeline Settings**: Enable/disable verification, debugging, vision parsing
- **Framework Preferences**: Default framework for code generation

Example:
```yaml
use_multi_agent_pipeline: true
use_verification: true
use_debugging: true

extractor:
  use_llm: true
  llm_provider: "groq"
  model: "llama-3.3-70b-versatile"
```

---

## 🎬 Usage

### Web Interface
1. Upload a PDF research paper
2. Select ML framework (PyTorch, TensorFlow, or Scikit-learn)
3. Click "Generate Code"
4. Monitor real-time progress
5. Download generated code

### Command Line
```bash
python src/main.py --input paper.pdf --output generated_code.py --framework pytorch
```

---

## 🔬 Features in Detail

### Verification Agent
- Executes generated code in isolated environment
- Extracts metrics (accuracy, loss, F1-score)
- Compares with paper-reported results
- Reports pass/fail status with tolerance checks

### Debugging Agent
- Static code analysis (syntax, logical errors)
- Automatic code fixes
- Best practice enforcement
- Iterative refinement

### Vision Parsing
- Extracts images from PDF pages
- Analyzes architecture diagrams
- Extracts component relationships
- Enhances code generation accuracy

---

## 📊 Thesis Completion Status

**Overall: ~95% Complete**

✅ **Fully Implemented:**
- Multi-agent pipeline architecture
- LLM integration (4 providers)
- Code generation for 3 frameworks
- Web interface with progress tracking
- Verification agent with safety checks
- Debugging agent with auto-fixes
- Vision parsing infrastructure

⚠️ **Partially Implemented:**
- Vision parsing (requires vision-capable LLM)
- Git repository generation (planned)
- CI/CD integration (planned)

---

## 🛡️ Security Notes

- **Never commit API keys** - Use environment variables
- API keys are excluded via `.gitignore`
- Code execution is sandboxed in verification agent
- Safety checks warn about potentially dangerous operations

---

## 📝 Documentation

- `PROJECT_CAPABILITIES_SUMMARY.md` - Overview of capabilities
- `VERIFICATION_AND_VISION_COMPLETE.md` - Verification & vision features
- `DEBUGGING_AGENT_FIXED.md` - Debugging agent details
- `THESIS_COMPLETION_STATUS.md` - Implementation status

---

## 🤝 Contributing

This is a thesis project. For questions or suggestions, please contact:
- **Email**: bhavesharora127@gmail.com
- **Student ID**: M24DE3022

---

## 📄 License

This project is part of academic research. Please cite appropriately if used.

---

## 🙏 Acknowledgments

- OpenAI, Anthropic, OpenRouter, and Groq for LLM APIs
- Flask, PyPDF2, pdfplumber, and other open-source libraries
- Research community for inspiration and feedback

---

## 📚 References

See thesis proposal document for detailed architecture and methodology.
