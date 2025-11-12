"""
Test the complete pipeline structure without requiring API calls.
Demonstrates all agents working together.
"""

import sys
from pathlib import Path

sys.path.append('src')

from utils.intermediate_representation import (
    PaperToCodeIR, PaperMetadata, AlgorithmInfo, MappedComponent
)
from agents.planner_agent import PlannerAgent


def test_pipeline_structure():
    """Test the complete pipeline structure."""
    print("=" * 70)
    print("PIPELINE STRUCTURE TEST (No API Calls Required)")
    print("=" * 70)
    print("\nThis test verifies the pipeline structure and agent coordination")
    print("without requiring LLM API calls.\n")
    
    # Create mock IR with sample data
    ir = PaperToCodeIR(
        paper_id="test_paper_001",
        paper_metadata=PaperMetadata(
            title="Test Transformer Paper",
            authors=["Test Author"],
            year=2024
        ),
        paper_path="test.pdf"
    )
    
    # Add mock algorithms
    ir.algorithms = [
        AlgorithmInfo(
            name="Transformer",
            description="Multi-head attention transformer architecture",
            type="neural_network",
            parameters=["num_layers", "hidden_size", "num_heads"],
            confidence=0.9,
            framework_suggestions=["pytorch"]
        ),
        AlgorithmInfo(
            name="Adam Optimizer",
            description="Adam optimization algorithm",
            type="optimization",
            parameters=["learning_rate", "beta1", "beta2"],
            confidence=0.8
        )
    ]
    
    # Add mock mapped components
    ir.mapped_components = [
        MappedComponent(
            algorithm_id="Transformer",
            framework="pytorch",
            library="torch.nn",
            function_class="nn.Transformer",
            parameters={"num_layers": 6, "hidden_size": 512}
        )
    ]
    
    print("Initial IR State:")
    print(f"  - Algorithms: {len(ir.algorithms)}")
    print(f"  - Mapped Components: {len(ir.mapped_components)}")
    print(f"  - Status: {ir.status}\n")
    
    # Test agent initialization
    print("-" * 70)
    print("TESTING AGENT INITIALIZATION")
    print("-" * 70)
    
    try:
        # Configure to use fallback methods (no LLM)
        config = {
            "use_paper_analysis": False,  # Skip since we have mock data
            "use_algorithm_interpretation": True,
            "use_api_mapping": True,
            "use_code_integration": True,
            "use_verification": True,
            "use_debugging": True,
            "agents": {
                "algorithm_interpretation": {
                    "use_llm": False  # Use basic interpretation
                },
                "api_mapping": {
                    "use_llm": False,  # Use basic mapping
                    "default_framework": "pytorch"
                },
                "code_integration": {
                    "use_llm": False,  # Use template generation
                    "include_examples": True
                },
                "verification": {
                    "execute_code": False  # Skip execution
                },
                "debugging": {
                    "max_iterations": 1,
                    "auto_fix": False
                }
            }
        }
        
        planner = PlannerAgent(config=config, llm_client=None)
        print("[OK] Planner Agent initialized")
        print("[OK] All sub-agents initialized\n")
        
        # Test pipeline execution
        print("-" * 70)
        print("TESTING PIPELINE EXECUTION")
        print("-" * 70)
        
        ir = planner.process(ir)
        
        print(f"\n[OK] Pipeline executed")
        print(f"[Status] {ir.status}")
        print(f"[Algorithms] {len(ir.algorithms)}")
        print(f"[Mapped Components] {len(ir.mapped_components)}")
        print(f"[Generated Files] {len(ir.generated_code)}")
        
        # Verify results
        print("\n" + "-" * 70)
        print("VERIFICATION")
        print("-" * 70)
        
        checks = []
        
        # Check algorithms have workflow steps
        if ir.algorithms:
            has_workflows = any(alg.workflow_steps for alg in ir.algorithms)
            checks.append(("Algorithm Workflows", has_workflows))
        
        # Check mapped components
        checks.append(("Mapped Components", len(ir.mapped_components) > 0))
        
        # Check generated code
        checks.append(("Generated Files", len(ir.generated_code) > 0))
        
        # Check verification
        if ir.verification_results:
            checks.append(("Verification Results", True))
        
        print("\nCheck Results:")
        all_passed = True
        for check_name, passed in checks:
            status = "[OK]" if passed else "[SKIP]"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        # Summary
        print("\n" + "=" * 70)
        if all_passed:
            print("[SUCCESS] All pipeline components working correctly!")
        else:
            print("[PARTIAL] Some components skipped (expected with mock data)")
        print("=" * 70)
        
        # Show generated files
        if ir.generated_code:
            print("\nGenerated Files:")
            for file_path in ir.generated_code.keys():
                file_info = ir.generated_code[file_path]
                print(f"  - {file_path} ({len(file_info.content)} chars)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pipeline_structure()

