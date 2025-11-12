"""
Code Integration Agent.
Assembles modules into a coherent codebase with proper structure.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import (
    PaperToCodeIR, MappedComponent, GeneratedFile
)
from generators.llm_code_generator import LLMCodeGenerator
from extractors.algorithm_extractor import Algorithm


class CodeIntegrationAgent(BaseAgent):
    """
    Integrates mapped components into a complete, structured codebase.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Code Integration Agent."""
        super().__init__(config, llm_client)
        
        # Initialize code generator
        self.code_generator = LLMCodeGenerator(
            config=self.config.get('generator', {}),
            llm_client=llm_client
        )
        
        # Repository structure configuration
        self.include_tests = self.config.get('include_tests', False)
        self.include_examples = self.config.get('include_examples', True)
        self.include_docs = self.config.get('include_docs', True)
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Generate and integrate code into repository structure.
        
        Args:
            ir: Intermediate representation with mapped components
            
        Returns:
            Updated IR with generated code files
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        if not ir.mapped_components:
            self.log_progress("No mapped components to integrate", "warning")
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress("Integrating code into repository structure...")
        
        try:
            # Determine framework
            framework = ir.mapped_components[0].framework if ir.mapped_components else "pytorch"
            
            # Convert mapped components to algorithms for code generation
            algorithms = self._components_to_algorithms(ir.mapped_components, ir.algorithms)
            
            # Generate code
            self.log_progress("Generating code files...")
            generated_code = self.code_generator.generate_code(algorithms, framework)
            
            # Create repository structure
            self.log_progress("Creating repository structure...")
            repository_structure = self._create_repository_structure(
                generated_code, 
                framework,
                ir
            )
            
            # Store generated files in IR
            ir.generated_code = repository_structure['files']
            ir.repository_structure = repository_structure['structure']
            
            self.update_ir_status(ir, "completed")
            self.log_progress(f"Generated {len(ir.generated_code)} files")
            
        except Exception as e:
            self.logger.error(f"Error in code integration: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def _components_to_algorithms(
        self, 
        components: List[MappedComponent],
        original_algorithms: List
    ) -> List[Algorithm]:
        """Convert mapped components back to Algorithm objects for code generation."""
        from extractors.algorithm_extractor import Algorithm
        
        algorithms = []
        algorithm_map = {alg.name: alg for alg in original_algorithms}
        
        for component in components:
            # Find original algorithm
            original_alg = algorithm_map.get(component.algorithm_id)
            
            if original_alg:
                # Create Algorithm object
                # Handle parameters - could be dict or string
                if isinstance(component.parameters, dict):
                    param_list = list(component.parameters.keys())
                elif isinstance(component.parameters, str):
                    # If parameters is a string, try to parse it or use empty list
                    param_list = []
                else:
                    param_list = original_alg.parameters if hasattr(original_alg, 'parameters') else []
                
                alg = Algorithm(
                    name=component.algorithm_id,
                    description=original_alg.description,
                    parameters=param_list,
                    confidence=original_alg.confidence if hasattr(original_alg, 'confidence') else 0.8
                )
                algorithms.append(alg)
        
        return algorithms
    
    def _create_repository_structure(
        self, 
        generated_code: str,
        framework: str,
        ir: PaperToCodeIR
    ) -> Dict:
        """Create complete repository structure."""
        files = {}
        structure = {
            "framework": framework,
            "root_files": [],
            "directories": []
        }
        
        # Main model file
        files["models/main.py"] = GeneratedFile(
            path="models/main.py",
            content=generated_code,
            file_type="model",
            dependencies=self._extract_dependencies(generated_code, framework)
        )
        structure["root_files"].append("models/main.py")
        structure["directories"].append("models")
        
        # Requirements file
        requirements_content = self._generate_requirements(framework, files["models/main.py"].dependencies)
        files["requirements.txt"] = GeneratedFile(
            path="requirements.txt",
            content=requirements_content,
            file_type="config",
            dependencies=[]
        )
        structure["root_files"].append("requirements.txt")
        
        # README
        if self.include_docs:
            readme_content = self._generate_readme(ir, framework)
            files["README.md"] = GeneratedFile(
                path="README.md",
                content=readme_content,
                file_type="documentation",
                dependencies=[]
            )
            structure["root_files"].append("README.md")
        
        # Example usage
        if self.include_examples:
            example_content = self._generate_example(framework)
            files["example.py"] = GeneratedFile(
                path="example.py",
                content=example_content,
                file_type="example",
                dependencies=["models/main.py"]
            )
            structure["root_files"].append("example.py")
        
        # Config file
        config_content = self._generate_config(framework)
        files["config.yaml"] = GeneratedFile(
            path="config.yaml",
            content=config_content,
            file_type="config",
            dependencies=[]
        )
        structure["root_files"].append("config.yaml")
        
        return {
            "files": files,
            "structure": structure
        }
    
    def _extract_dependencies(self, code: str, framework: str) -> List[str]:
        """Extract dependencies from generated code."""
        dependencies = []
        
        # Framework-specific dependencies
        if framework == "pytorch":
            dependencies.append("torch>=2.0.0")
        elif framework == "tensorflow":
            dependencies.append("tensorflow>=2.13.0")
        elif framework == "sklearn":
            dependencies.append("scikit-learn>=1.3.0")
        
        # Common dependencies
        common_deps = ["numpy>=1.24.0", "matplotlib>=3.7.0"]
        dependencies.extend(common_deps)
        
        # Check for specific imports in code
        if "pandas" in code:
            dependencies.append("pandas>=2.0.0")
        if "seaborn" in code:
            dependencies.append("seaborn>=0.12.0")
        
        return dependencies
    
    def _generate_requirements(self, framework: str, dependencies: List[str]) -> str:
        """Generate requirements.txt content."""
        lines = dependencies.copy()
        
        # Add framework-specific extras
        if framework == "pytorch":
            if "torch" not in str(dependencies):
                lines.insert(0, "torch>=2.0.0")
        elif framework == "tensorflow":
            if "tensorflow" not in str(dependencies):
                lines.insert(0, "tensorflow>=2.13.0")
        
        return "\n".join(sorted(set(lines))) + "\n"
    
    def _generate_readme(self, ir: PaperToCodeIR, framework: str) -> str:
        """Generate README.md content."""
        title = ir.paper_metadata.title if ir.paper_metadata else "Generated Code"
        
        readme = f"""# {title}

## Generated Code Repository

This repository contains automatically generated code from the research paper:
**{title}**

### Framework
- **Framework:** {framework.capitalize()}

### Algorithms Implemented
"""
        for i, alg in enumerate(ir.algorithms[:10], 1):  # Limit to 10
            readme += f"{i}. {alg.name}\n"
        
        readme += """
### Installation

```bash
pip install -r requirements.txt
```

### Usage

See `example.py` for usage examples.

### Generated Files

- `models/main.py` - Main model implementation
- `config.yaml` - Configuration file
- `example.py` - Example usage
- `requirements.txt` - Python dependencies

### Notes

This code was automatically generated. Please review and test before use.
"""
        return readme
    
    def _generate_example(self, framework: str) -> str:
        """Generate example usage file."""
        if framework == "pytorch":
            return """import torch
from models.main import *

# Example usage
if __name__ == "__main__":
    print("Example usage of generated models")
    # Add your example code here
"""
        elif framework == "tensorflow":
            return """import tensorflow as tf
from models.main import *

# Example usage
if __name__ == "__main__":
    print("Example usage of generated models")
    # Add your example code here
"""
        else:
            return """from models.main import *

# Example usage
if __name__ == "__main__":
    print("Example usage of generated models")
    # Add your example code here
"""
    
    def _generate_config(self, framework: str) -> str:
        """Generate configuration file."""
        config = f"""# Configuration for {framework} implementation

framework: {framework}

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001

model:
  # Add model-specific configuration here
"""
        return config

