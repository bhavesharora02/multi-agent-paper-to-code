# Multi-Agent LLM Pipeline Architecture Plan
## Automating ML/DL Paper-to-Code Translation

**Thesis Project:** Automating ML/DL Paper-to-Code Translation via Multi-Agent LLM Pipelines  
**Student:** Bhavesh Arora (M24DE3022)  
**Current Status:** Basic prototype → Full multi-agent system

---

## 🎯 Current System vs. Target System

### Current System (Prototype)
- ✅ Basic PDF text extraction
- ✅ Rule-based pattern matching for algorithms
- ✅ Template-based code generation
- ❌ No LLM integration
- ❌ No vision/OCR capabilities
- ❌ No multi-agent architecture
- ❌ No verification/debugging loops
- ❌ No Git repository generation

### Target System (Thesis)
- ✅ Multi-agent LLM pipeline
- ✅ Vision-enabled parsing (OCR/vision models)
- ✅ Specialized agents for each task
- ✅ Automated verification and debugging
- ✅ Git repository with CI/CD generation
- ✅ End-to-end paper → validated code pipeline

---

## 🏗️ Proposed Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLANNER AGENT                            │
│         (Orchestrates workflow, manages context)            │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   PAPER      │   │  ALGORITHM   │   │  API/LIBRARY │
│  ANALYSIS    │──▶│INTERPRETATION│──▶│   MAPPING    │
│   AGENT      │   │    AGENT     │   │    AGENT     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │    CODE      │
                   │ INTEGRATION  │
                   │    AGENT     │
                   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ VERIFICATION │
                   │    AGENT     │
                   └──────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
            ┌──────────────┐  ┌──────────────┐
            │   PASS       │  │   FAIL       │
            │  (Deploy)    │  │ (Debugging)  │
            └──────────────┘  └──────────────┘
                    │               │
                    │               └───▶ DEBUGGING AGENT
                    │                       (Iterative refinement)
                    │                               │
                    │                               ▼
                    │                       ┌──────────────┐
                    │                       │   CODE       │
                    │                       │ INTEGRATION  │
                    │                       │  (Refined)   │
                    │                       └──────────────┘
                    │                               │
                    └───────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │   GIT REPO   │
                            │  GENERATION  │
                            │  + CI/CD     │
                            └──────────────┘
```

---

## 🤖 Agent Specifications

### 1. **Paper Analysis Agent**
**Purpose:** Extract structured information from paper PDFs

**Capabilities:**
- PDF text extraction (enhanced with OCR for scanned papers)
- Vision model integration for diagram/table parsing
- LLM-powered text understanding
- Extract: model descriptions, pseudocode, experiment settings, hyperparameters

**Output:** Structured JSON schema
```json
{
  "paper_metadata": {...},
  "algorithms": [...],
  "diagrams": [...],
  "tables": [...],
  "experiments": [...],
  "hyperparameters": {...}
}
```

**Technologies:**
- LLM: OpenAI GPT-4 / Anthropic Claude (for text understanding)
- Vision: GPT-4 Vision / Claude 3 Vision (for diagrams)
- OCR: Tesseract / EasyOCR (for scanned text)
- PDF: pdfplumber, PyMuPDF (enhanced extraction)

---

### 2. **Algorithm Interpretation Agent**
**Purpose:** Translate mathematical notation and pseudocode into computational workflows

**Capabilities:**
- Parse mathematical equations and notation
- Convert pseudocode to structured steps
- Handle custom losses, iterative procedures
- Extract control-flow logic
- Identify data dependencies

**Input:** Structured paper representation from Paper Analysis Agent
**Output:** Computational workflow specification

**Technologies:**
- LLM: GPT-4 / Claude (for interpretation)
- Math parsing: SymPy (for equation handling)
- Custom prompt engineering for algorithm translation

---

### 3. **API/Library Mapping Agent**
**Purpose:** Map components to appropriate frameworks and libraries

**Capabilities:**
- Identify framework requirements (PyTorch, TensorFlow, scikit-learn, etc.)
- Map algorithm components to library functions
- Retrieve code snippets from documentation
- Suggest optimal library choices
- Handle framework-specific implementations

**Input:** Algorithm workflow from Interpretation Agent
**Output:** Mapped components with library references

**Technologies:**
- LLM: GPT-4 / Claude (for intelligent mapping)
- Documentation retrieval: LangChain / LlamaIndex
- Library knowledge base: Custom embeddings of framework docs

---

### 4. **Code Integration Agent**
**Purpose:** Assemble modules into coherent codebase

**Capabilities:**
- Generate model definitions
- Create data loaders
- Write training/evaluation scripts
- Generate configuration files
- Create dependency manifests (requirements.txt)
- Structure repository (train.py, config/, models/, etc.)

**Input:** Mapped components from API/Library Mapping Agent
**Output:** Complete codebase structure

**Technologies:**
- LLM: GPT-4 / Claude (for code generation)
- Code formatting: Black, autopep8
- Template system: Jinja2 (for structured generation)

---

### 5. **Verification Agent**
**Purpose:** Execute and validate generated code

**Capabilities:**
- Run experiments on benchmark datasets
- Execute on synthetic inputs if needed
- Compare metrics against paper results
- Flag discrepancies beyond tolerance thresholds
- Generate execution logs and reports

**Input:** Generated codebase
**Output:** Verification report with pass/fail status

**Technologies:**
- Code execution: Subprocess / Docker containers
- Metric comparison: Custom validation logic
- Dataset handling: HuggingFace Datasets, torchvision
- Logging: Structured logging system

---

### 6. **Debugging Agent**
**Purpose:** Diagnose failures and refine code iteratively

**Capabilities:**
- Analyze verification failures
- Identify root causes (missing hyperparameters, bugs, etc.)
- Generate targeted fixes
- Iterate through integration → verification loop
- Track refinement history

**Input:** Verification failure reports
**Output:** Refined code with fixes

**Technologies:**
- LLM: GPT-4 / Claude (for debugging analysis)
- Error analysis: Custom diagnostic tools
- Iterative refinement: Loop management system

---

### 7. **Planner Agent** (Orchestrator)
**Purpose:** Coordinate all agents and manage workflow

**Capabilities:**
- Orchestrate agent execution sequence
- Manage shared context memory
- Resolve inter-agent conflicts
- Handle error recovery
- Optimize agent communication
- Track overall pipeline progress

**Technologies:**
- Agent framework: LangGraph / AutoGen / CrewAI
- State management: Redis / SQLite (for intermediate representations)
- Workflow orchestration: Custom planner logic

---

## 📊 Data Flow & Intermediate Representations

### Shared Intermediate Representation (JSON Schema)

```json
{
  "paper_id": "unique_id",
  "paper_metadata": {
    "title": "...",
    "authors": [...],
    "year": 2024
  },
  "extracted_content": {
    "text_sections": [...],
    "diagrams": [
      {
        "type": "architecture",
        "image_path": "...",
        "parsed_structure": {...}
      }
    ],
    "tables": [...],
    "equations": [...]
  },
  "algorithms": [
    {
      "name": "...",
      "description": "...",
      "pseudocode": "...",
      "mathematical_notation": "...",
      "workflow_steps": [...],
      "parameters": {...}
    }
  ],
  "mapped_components": [
    {
      "algorithm_id": "...",
      "framework": "pytorch",
      "library_mappings": [...],
      "code_snippets": [...]
    }
  ],
  "generated_code": {
    "repository_structure": {...},
    "files": [
      {
        "path": "models/transformer.py",
        "content": "...",
        "dependencies": [...]
      }
    ]
  },
  "verification_results": {
    "status": "pass|fail",
    "metrics": {...},
    "comparison": {...},
    "errors": [...]
  },
  "refinement_history": [...]
}
```

---

## 🛠️ Technology Stack

### LLM Integration
- **Primary:** OpenAI GPT-4 / Anthropic Claude 3
- **Vision:** GPT-4 Vision / Claude 3 Vision
- **Embeddings:** OpenAI text-embedding-3 / Anthropic embeddings
- **Framework:** LangChain / LlamaIndex (for orchestration)

### Vision & OCR
- **OCR:** Tesseract, EasyOCR
- **Diagram Parsing:** GPT-4 Vision, Claude 3 Vision
- **Image Processing:** PIL/Pillow, OpenCV

### Agent Framework
- **Options:** LangGraph, AutoGen, CrewAI, or custom implementation
- **State Management:** Redis, SQLite, or in-memory with persistence

### Code Execution & Testing
- **Execution:** Subprocess, Docker containers (for isolation)
- **Testing:** pytest, unittest
- **Metrics:** Custom validation framework

### Repository Generation
- **Git:** GitPython
- **CI/CD:** GitHub Actions / GitLab CI templates
- **Structure:** Custom repository generator

---

## 📁 Proposed Project Structure

```
ml-paper-to-code-llm/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner_agent.py          # Orchestrator
│   │   ├── paper_analysis_agent.py   # Paper parsing + vision
│   │   ├── algorithm_interpretation_agent.py
│   │   ├── api_mapping_agent.py
│   │   ├── code_integration_agent.py
│   │   ├── verification_agent.py
│   │   └── debugging_agent.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   ├── vision_processor.py
│   │   └── prompt_templates.py
│   ├── parsers/
│   │   ├── pdf_parser.py             # Enhanced with OCR
│   │   ├── diagram_parser.py          # Vision model integration
│   │   └── table_parser.py
│   ├── execution/
│   │   ├── code_runner.py
│   │   ├── test_harness.py
│   │   └── metric_comparator.py
│   ├── repository/
│   │   ├── git_generator.py
│   │   ├── ci_cd_generator.py
│   │   └── structure_builder.py
│   ├── utils/
│   │   ├── intermediate_representation.py
│   │   ├── state_manager.py
│   │   └── helpers.py
│   └── main.py
├── config/
│   ├── agent_config.yaml
│   ├── llm_config.yaml
│   └── default.yaml
├── prompts/
│   ├── paper_analysis_prompts.txt
│   ├── algorithm_interpretation_prompts.txt
│   ├── code_generation_prompts.txt
│   └── debugging_prompts.txt
├── tests/
│   ├── test_agents/
│   ├── test_llm_integration/
│   └── test_end_to_end/
├── requirements.txt
└── README.md
```

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up LLM integration (OpenAI/Anthropic)
- [ ] Enhance PDF parser with OCR capabilities
- [ ] Implement vision model integration for diagrams
- [ ] Create intermediate representation schema
- [ ] Set up agent framework infrastructure

### Phase 2: Core Agents (Weeks 3-5)
- [ ] Implement Paper Analysis Agent
- [ ] Implement Algorithm Interpretation Agent
- [ ] Implement API/Library Mapping Agent
- [ ] Implement Code Integration Agent
- [ ] Create prompt templates for each agent

### Phase 3: Verification & Debugging (Weeks 6-7)
- [ ] Implement Verification Agent
- [ ] Build test harness and execution system
- [ ] Implement Debugging Agent
- [ ] Create iterative refinement loop

### Phase 4: Orchestration (Week 8)
- [ ] Implement Planner Agent
- [ ] Set up state management system
- [ ] Create agent communication protocol
- [ ] Build workflow orchestration

### Phase 5: Repository Generation (Week 9)
- [ ] Implement Git repository generator
- [ ] Create CI/CD template generator
- [ ] Build repository structure builder

### Phase 6: Integration & Testing (Weeks 10-11)
- [ ] End-to-end pipeline integration
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Error handling and recovery

### Phase 7: Documentation & Deployment (Week 12)
- [ ] Complete documentation
- [ ] Create demo examples
- [ ] Prepare thesis materials
- [ ] Final testing and validation

---

## 🔑 Key Implementation Details

### LLM Integration Strategy
1. **API Selection:** Start with OpenAI GPT-4, add Anthropic Claude as alternative
2. **Prompt Engineering:** Create specialized prompts for each agent
3. **Cost Management:** Implement caching, token optimization
4. **Error Handling:** Robust retry logic, fallback mechanisms

### Vision Model Integration
1. **Diagram Parsing:** Use GPT-4 Vision / Claude 3 Vision for architecture diagrams
2. **Table Extraction:** Combine OCR with LLM for structured data extraction
3. **Image Preprocessing:** Clean and enhance images before sending to vision models

### Agent Communication
1. **Shared State:** Use Redis or SQLite for intermediate representations
2. **Message Passing:** Structured JSON messages between agents
3. **Context Management:** Maintain conversation history and context

### Verification Strategy
1. **Execution Isolation:** Use Docker containers for safe code execution
2. **Metric Comparison:** Implement tolerance-based comparison
3. **Dataset Handling:** Support benchmark datasets and synthetic data

### Iterative Refinement
1. **Failure Analysis:** LLM-powered root cause analysis
2. **Targeted Fixes:** Generate specific code patches
3. **Loop Limits:** Prevent infinite loops with max iterations

---

## 📝 Next Steps

1. **Choose LLM Provider:** Decide on OpenAI vs Anthropic (or both)
2. **Select Agent Framework:** LangGraph, AutoGen, CrewAI, or custom
3. **Set up Development Environment:** Install dependencies, configure API keys
4. **Start with Paper Analysis Agent:** Build foundation with LLM integration
5. **Incremental Development:** Build and test each agent independently
6. **Integration Testing:** Test agent interactions and data flow

---

## 🔐 Configuration Requirements

### Environment Variables
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379  # Optional for state management
```

### API Keys Setup
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

---

## 📚 References & Resources

- **LangChain:** https://python.langchain.com/
- **AutoGen:** https://microsoft.github.io/autogen/
- **CrewAI:** https://www.crewai.com/
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **OpenAI API:** https://platform.openai.com/docs
- **Anthropic API:** https://docs.anthropic.com/

---

**Status:** Planning Phase  
**Last Updated:** November 2025

