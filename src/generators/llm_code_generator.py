"""
LLM-based code generator for creating Python implementations.
Replaces template-based generation with intelligent LLM code generation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import re

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from generators.code_generator import CodeGenerator
from extractors.algorithm_extractor import Algorithm
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import CODE_GENERATION_SYSTEM_PROMPT, CODE_GENERATION_PROMPT
from utils.intermediate_representation import AlgorithmInfo, GeneratedFile


class LLMCodeGenerator:
    """
    LLM-based code generator that creates Python implementations
    using large language models instead of templates.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """
        Initialize LLM code generator.
        
        Args:
            config: Configuration dictionary
            llm_client: LLM client instance (optional)
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            try:
                provider_str = self.config.get("llm_provider", "openai").lower()
                if provider_str == "openai":
                    provider = LLMProvider.OPENAI
                elif provider_str == "anthropic":
                    provider = LLMProvider.ANTHROPIC
                elif provider_str == "openrouter":
                    provider = LLMProvider.OPENROUTER
                elif provider_str == "groq":
                    provider = LLMProvider.GROQ
                else:
                    provider = LLMProvider.OPENAI  # Default
                self.llm_client = LLMClient(provider=provider)
                self.logger.info(f"Initialized LLM client: {provider.value}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM client: {e}")
                raise
        
        # Fallback to template-based generator
        self.fallback_generator = CodeGenerator(config)
        self.use_fallback = self.config.get("use_fallback", True)
    
    def generate_code(
        self, 
        algorithms: List[Algorithm], 
        framework: str = 'pytorch'
    ) -> str:
        """
        Generate Python code for algorithms using LLM.
        
        Args:
            algorithms: List of Algorithm objects
            framework: Target ML framework
            
        Returns:
            Generated Python code
        """
        if not algorithms:
            return self._generate_empty_template(framework)
        
        try:
            # Convert Algorithm objects to dict for LLM
            algorithm_specs = []
            for alg in algorithms:
                spec = {
                    "name": alg.name,
                    "description": alg.description,
                    "parameters": alg.parameters or [],
                    "complexity": alg.complexity,
                    "pseudocode": alg.pseudocode
                }
                algorithm_specs.append(spec)
            
            # Generate code for each algorithm
            generated_files = []
            
            for i, alg_spec in enumerate(algorithm_specs):
                self.logger.info(f"Generating code for algorithm {i+1}/{len(algorithm_specs)}: {alg_spec['name']}")
                
                code = self._generate_algorithm_code(alg_spec, framework, i)
                
                # Create file structure
                file_name = self._sanitize_name(alg_spec['name']) + f"_{i}.py"
                generated_files.append({
                    "name": file_name,
                    "content": code,
                    "algorithm": alg_spec['name']
                })
            
            # Combine all files into main code
            full_code = self._combine_code_files(generated_files, framework)
            
            return full_code
            
        except Exception as e:
            self.logger.error(f"Error in LLM code generation: {e}")
            if self.use_fallback:
                self.logger.info("Falling back to template-based generation")
                return self.fallback_generator.generate_code(algorithms, framework)
            raise
    
    def _generate_algorithm_code(
        self, 
        algorithm_spec: Dict, 
        framework: str, 
        index: int
    ) -> str:
        """Generate code for a single algorithm using LLM."""
        
        prompt = CODE_GENERATION_PROMPT.format(
            framework=framework,
            algorithm_spec=self._format_algorithm_spec(algorithm_spec)
        )
        
        system_prompt = CODE_GENERATION_SYSTEM_PROMPT.replace("{framework}", framework)
        
        try:
            code = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Lower temperature for more deterministic code
                max_tokens=4000
            )
            
            # Clean up code (remove markdown code blocks if present)
            code = self._clean_generated_code(code)
            
            return code
            
        except Exception as e:
            self.logger.error(f"Error generating code for {algorithm_spec['name']}: {e}")
            raise
    
    def _format_algorithm_spec(self, spec: Dict) -> str:
        """Format algorithm specification for prompt."""
        lines = [
            f"Algorithm Name: {spec['name']}",
            f"Description: {spec['description']}",
        ]
        
        if spec.get('parameters'):
            lines.append(f"Parameters: {', '.join(spec['parameters'])}")
        
        if spec.get('complexity'):
            lines.append(f"Complexity: {spec['complexity']}")
        
        if spec.get('pseudocode'):
            lines.append(f"\nPseudocode:\n{spec['pseudocode']}")
        
        return "\n".join(lines)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing markdown formatting."""
        # Remove markdown code blocks
        code = re.sub(r'```python\n?(.*?)\n?```', r'\1', code, flags=re.DOTALL)
        code = re.sub(r'```\n?(.*?)\n?```', r'\1', code, flags=re.DOTALL)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _combine_code_files(self, files: List[Dict], framework: str) -> str:
        """Combine multiple code files into a single file."""
        # Generate imports
        imports = self._generate_imports(framework)
        
        # Combine all algorithm implementations
        algorithm_codes = []
        for file_info in files:
            algorithm_codes.append(f"# {file_info['algorithm']}\n{file_info['content']}")
        
        # Combine
        full_code = imports + "\n\n" + "\n\n".join(algorithm_codes)
        
        # Add main execution block
        main_block = self._generate_main_block(files, framework)
        full_code += "\n\n" + main_block
        
        return full_code
    
    def _generate_imports(self, framework: str) -> str:
        """Generate import statements."""
        base_imports = [
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from typing import List, Dict, Optional, Tuple",
            "import logging"
        ]
        
        framework_imports = {
            'pytorch': [
                "import torch",
                "import torch.nn as nn",
                "import torch.optim as optim",
                "import torch.nn.functional as F",
                "from torch.utils.data import DataLoader, Dataset"
            ],
            'tensorflow': [
                "import tensorflow as tf",
                "from tensorflow import keras",
                "from tensorflow.keras import layers, models, optimizers"
            ],
            'sklearn': [
                "from sklearn.model_selection import train_test_split",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.metrics import accuracy_score, classification_report"
            ]
        }
        
        all_imports = base_imports + framework_imports.get(framework, [])
        return "\n".join(all_imports)
    
    def _generate_main_block(self, files: List[Dict], framework: str) -> str:
        """Generate main execution block."""
        code = """
def main():
    \"\"\"Main execution function.\"\"\"
    print("ML/DL Paper to Code - Generated Implementation")
    print("=" * 50)
    
    # Example usage
    print("Generated algorithms:")
"""
        for i, file_info in enumerate(files):
            code += f'    print(f"  {i+1}. {file_info["algorithm"]}")\n'
        
        code += """
    print("\\nPlease implement the main training/evaluation logic.")
    print("The generated code provides the model definitions.")


if __name__ == "__main__":
    import sys
    main()
"""
        return code
    
    def _generate_empty_template(self, framework: str) -> str:
        """Generate empty template when no algorithms found."""
        imports = self._generate_imports(framework)
        return f"""{imports}

# No algorithms detected in the paper.
# Please ensure the paper contains clear algorithm descriptions.

def main():
    print("No algorithms detected in the provided paper.")
    print("Please check that the paper contains clear algorithm descriptions.")

if __name__ == "__main__":
    main()
"""
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize algorithm name for use as filename."""
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        if sanitized and sanitized[0].isdigit():
            sanitized = 'Algorithm' + sanitized
        return sanitized or 'Algorithm'

