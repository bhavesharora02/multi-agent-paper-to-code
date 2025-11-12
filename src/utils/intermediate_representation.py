"""
Intermediate representation schema for multi-agent pipeline.
Defines the structured data format passed between agents.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class PaperMetadata:
    """Metadata about the research paper."""
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    paper_id: Optional[str] = None


@dataclass
class DiagramInfo:
    """Information about extracted diagrams."""
    type: str  # "architecture", "flowchart", "graph", "table", etc.
    image_path: str
    parsed_structure: Optional[Dict] = None
    description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class AlgorithmInfo:
    """Structured algorithm information."""
    name: str
    description: str
    type: str  # "neural_network", "supervised_learning", etc.
    pseudocode: Optional[str] = None
    mathematical_notation: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    complexity: Optional[str] = None
    framework_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    workflow_steps: List[Dict] = field(default_factory=list)
    data_dependencies: List[str] = field(default_factory=list)


@dataclass
class MappedComponent:
    """Component mapped to a specific library/framework."""
    algorithm_id: str
    framework: str
    library: str
    function_class: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    code_snippet: Optional[str] = None
    documentation_reference: Optional[str] = None


@dataclass
class GeneratedFile:
    """Information about a generated code file."""
    path: str
    content: str
    file_type: str  # "model", "trainer", "data_loader", "config", etc.
    dependencies: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Results from code verification."""
    status: str  # "pass", "fail", "error"
    metrics: Dict[str, float] = field(default_factory=dict)
    paper_metrics: Dict[str, float] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    discrepancies: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    tolerance_check: bool = False


@dataclass
class RefinementHistory:
    """History of code refinements."""
    iteration: int
    timestamp: str
    changes: List[str] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    success: bool = False


@dataclass
class PaperToCodeIR:
    """
    Intermediate Representation for the entire paper-to-code pipeline.
    This is the shared data structure passed between agents.
    """
    # Paper information
    paper_id: str
    paper_metadata: PaperMetadata
    paper_path: str
    
    # Extracted content
    extracted_content: Dict[str, Any] = field(default_factory=dict)
    text_sections: List[str] = field(default_factory=list)
    diagrams: List[DiagramInfo] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    
    # Algorithms
    algorithms: List[AlgorithmInfo] = field(default_factory=list)
    
    # Mapped components
    mapped_components: List[MappedComponent] = field(default_factory=list)
    
    # Generated code
    generated_code: Dict[str, GeneratedFile] = field(default_factory=dict)
    repository_structure: Dict[str, Any] = field(default_factory=dict)
    
    # Verification
    verification_results: Optional[VerificationResult] = None
    
    # Refinement history
    refinement_history: List[RefinementHistory] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_agent: Optional[str] = None
    status: str = "initialized"  # "initialized", "processing", "completed", "failed"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperToCodeIR':
        """Create from dictionary."""
        # Handle nested dataclasses
        if 'paper_metadata' in data and isinstance(data['paper_metadata'], dict):
            data['paper_metadata'] = PaperMetadata(**data['paper_metadata'])
        
        if 'diagrams' in data:
            data['diagrams'] = [DiagramInfo(**d) if isinstance(d, dict) else d for d in data['diagrams']]
        
        if 'algorithms' in data:
            data['algorithms'] = [AlgorithmInfo(**a) if isinstance(a, dict) else a for a in data['algorithms']]
        
        if 'mapped_components' in data:
            data['mapped_components'] = [MappedComponent(**m) if isinstance(m, dict) else m for m in data['mapped_components']]
        
        if 'verification_results' in data and data['verification_results']:
            if isinstance(data['verification_results'], dict):
                data['verification_results'] = VerificationResult(**data['verification_results'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PaperToCodeIR':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def update_status(self, status: str, agent: Optional[str] = None):
        """Update status and timestamp."""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if agent:
            self.current_agent = agent

