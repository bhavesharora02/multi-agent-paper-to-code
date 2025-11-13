# Multi-Agent Code Generation and Debugging System

**Author**: Kanishka Dhindhwal (M24DE3043)  
**Project**: Major Technical Project (MTP)  
**Institution**: M. Tech in Data Engineering

## 🎯 Project Overview

This project implements a multi-agent LLM framework for collaborative code generation and debugging. The system uses specialist AI agents (Coder, Tester, Debugger, Rater, Optimizer, Explainer) working together to generate, test, debug, and optimize code automatically.

## 🚀 Quick Start

### 1. Navigate to Project
```bash
cd multi_agent_codegen
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Web Server
```bash
python run.py
```

### 4. Open Browser
Go to: **http://localhost:5000**

## 📁 Project Structure

```
.
├── multi_agent_codegen/          # Main project directory
│   ├── app.py                    # Flask web application
│   ├── agents/                   # Multi-agent system
│   ├── workflow/                 # LangGraph workflow
│   ├── evaluation/               # Benchmark evaluation
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS & JavaScript
│   └── README.md                 # Detailed documentation
├── src/                          # LLM client (shared dependency)
└── venv/                         # Virtual environment
```

## ✨ Features

- **Code Generation**: AI-powered code generation from specifications
- **Quality Rating**: 0-10 rating system for code quality
- **Interactive Q&A**: "Understand the Code" chat interface
- **Multi-Agent Workflow**: Coder → Tester → Debugger → Rater → Optimizer
- **Web Interface**: Modern, user-friendly UI

## 📚 Documentation

All detailed documentation is in the `multi_agent_codegen/` directory:
- `README.md` - Complete project documentation
- `QUICK_START.md` - Quick start guide
- `SETUP.md` - Setup instructions
- `WORKFLOW_EXPLANATION.md` - Workflow details

## 🔧 Configuration

Edit `multi_agent_codegen/config.yaml` to customize:
- LLM provider and model
- Workflow parameters
- Agent settings

## 📝 License

This project is part of academic research. Please cite appropriately if used.

---

**For detailed documentation, see `multi_agent_codegen/README.md`**

