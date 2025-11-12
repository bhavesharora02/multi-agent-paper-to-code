# Quick Start: Implementing Multi-Agent LLM Pipeline

## Immediate Next Steps

### Step 1: Set Up LLM Integration

First, let's add LLM capabilities to your existing system. Choose one:

**Option A: OpenAI GPT-4**
```bash
pip install openai langchain
```

**Option B: Anthropic Claude**
```bash
pip install anthropic langchain
```

**Option C: Both (Recommended for flexibility)**
```bash
pip install openai anthropic langchain langchain-openai langchain-anthropic
```

### Step 2: Create LLM Client Module

Create `src/llm/llm_client.py`:

```python
"""
LLM client for multi-agent pipeline.
Supports both OpenAI and Anthropic.
"""
import os
from typing import Optional, Dict, List
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMClient:
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        self.provider = provider
        self._initialize_client()
    
    def _initialize_client(self):
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
        elif self.provider == LLMProvider.ANTHROPIC:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-opus-20240229"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using LLM."""
        if self.provider == LLMProvider.OPENAI:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        else:  # Anthropic
            messages = [{"role": "user", "content": prompt}]
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4096),
                system=system_prompt or "",
                messages=messages
            )
            return response.content[0].text
    
    def analyze_with_vision(self, image_path: str, prompt: str) -> str:
        """Analyze image using vision model."""
        # Implementation for vision analysis
        # This requires base64 encoding of images
        pass
```

### Step 3: Transform Paper Analysis Agent

Update `src/extractors/algorithm_extractor.py` to use LLM:

```python
"""
Enhanced Algorithm Extractor using LLM.
"""
from llm.llm_client import LLMClient, LLMProvider
import json

class LLMAlgorithmExtractor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.llm_client = LLMClient(LLMProvider.OPENAI)  # or ANTHROPIC
        
        # LLM prompt template
        self.extraction_prompt = """
        Analyze the following research paper text and extract all machine learning 
        and deep learning algorithms mentioned. For each algorithm, provide:
        
        1. Algorithm name
        2. Description
        3. Key parameters
        4. Mathematical notation (if any)
        5. Pseudocode (if available)
        6. Framework suggestions (PyTorch, TensorFlow, scikit-learn)
        
        Paper text:
        {text}
        
        Return the results as a JSON array.
        """
    
    def extract_algorithms(self, text: str) -> List[Dict]:
        """Extract algorithms using LLM."""
        prompt = self.extraction_prompt.format(text=text[:8000])  # Limit token usage
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt="You are an expert in machine learning and deep learning. Extract algorithms accurately from research papers.",
            temperature=0.3  # Lower temperature for more consistent extraction
        )
        
        # Parse JSON response
        try:
            algorithms = json.loads(response)
            return algorithms
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                algorithms = json.loads(json_match.group(1))
                return algorithms
            return []
```

### Step 4: Enhanced Code Generator with LLM

Update `src/generators/code_generator.py` to use LLM:

```python
"""
Enhanced Code Generator using LLM.
"""
from llm.llm_client import LLMClient, LLMProvider

class LLMCodeGenerator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.llm_client = LLMClient(LLMProvider.OPENAI)
    
    def generate_code(self, algorithm_spec: Dict, framework: str = 'pytorch') -> str:
        """Generate code using LLM based on algorithm specification."""
        
        prompt = f"""
        Generate a complete, runnable Python implementation of the following 
        algorithm using {framework}.
        
        Algorithm Specification:
        {json.dumps(algorithm_spec, indent=2)}
        
        Requirements:
        1. Include all necessary imports
        2. Create a class for the model/algorithm
        3. Include training/evaluation methods
        4. Add proper docstrings
        5. Include example usage
        6. Make it production-ready
        
        Generate complete, executable code:
        """
        
        system_prompt = f"""You are an expert Python developer specializing in {framework}. 
        Generate clean, well-documented, and executable code."""
        
        code = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for more deterministic code
        )
        
        # Clean up code (remove markdown code blocks if present)
        import re
        code = re.sub(r'```python\n?(.*?)\n?```', r'\1', code, flags=re.DOTALL)
        code = re.sub(r'```\n?(.*?)\n?```', r'\1', code, flags=re.DOTALL)
        
        return code.strip()
```

### Step 5: Create Environment Configuration

Create `.env.example`:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Create `.gitignore` entry:
```
.env
*.env
```

### Step 6: Update Requirements

Add to `requirements.txt`:
```
openai>=1.0.0
anthropic>=0.18.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-anthropic>=0.1.0
python-dotenv>=1.0.0
```

### Step 7: Test LLM Integration

Create `test_llm_integration.py`:

```python
"""Test LLM integration."""
import os
from dotenv import load_dotenv
from src.llm.llm_client import LLMClient, LLMProvider

load_dotenv()

def test_llm_extraction():
    """Test algorithm extraction with LLM."""
    client = LLMClient(LLMProvider.OPENAI)
    
    sample_text = """
    We propose a novel transformer architecture for natural language processing.
    The model uses multi-head attention mechanisms and feed-forward networks.
    """
    
    prompt = f"Extract ML/DL algorithms from: {sample_text}"
    response = client.generate(prompt)
    print("LLM Response:", response)

if __name__ == "__main__":
    test_llm_extraction()
```

## Migration Strategy

### Phase 1: Hybrid Approach (Recommended Start)
1. Keep existing rule-based extractor as fallback
2. Add LLM-based extractor as primary
3. Compare results and gradually phase out rule-based

### Phase 2: Full LLM Integration
1. Replace all rule-based components with LLM agents
2. Add vision model integration
3. Implement multi-agent orchestration

### Phase 3: Advanced Features
1. Add verification agent
2. Implement debugging loop
3. Add Git repository generation

## Quick Test Command

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

# Test LLM integration
python test_llm_integration.py
```

## Cost Considerations

- **OpenAI GPT-4:** ~$0.03 per 1K input tokens, $0.06 per 1K output tokens
- **Anthropic Claude 3 Opus:** ~$0.015 per 1K input tokens, $0.075 per 1K output tokens
- **Estimate per paper:** $0.50 - $2.00 depending on paper length

**Optimization Tips:**
- Cache LLM responses for similar papers
- Use GPT-3.5-turbo for simpler tasks (10x cheaper)
- Implement token counting and limits
- Batch similar requests

## Next: Implement First Agent

Start with the **Paper Analysis Agent** as it's the foundation:

1. Enhance PDF parser with OCR
2. Add vision model for diagrams
3. Use LLM for text understanding
4. Output structured JSON

Then move to other agents incrementally.

