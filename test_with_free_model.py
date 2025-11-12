"""
Test the complete pipeline with OpenRouter free model.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append('src')

from utils.intermediate_representation import PaperToCodeIR, PaperMetadata
from agents.planner_agent import PlannerAgent
from llm.llm_client import LLMClient, LLMProvider


def test_pipeline_with_free_model():
    """Test complete pipeline with free OpenRouter model."""
    print("=" * 70)
    print("TESTING COMPLETE PIPELINE WITH OPENROUTER FREE MODEL")
    print("=" * 70)
    
    # Set API key
    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("Please set it using: $env:OPENROUTER_API_KEY='your_key_here'")
        return
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Find a test PDF
    uploads_dir = Path("uploads")
    test_pdf = None
    
    # Try to find a paper with ML/DL content
    preferred_names = ["attention", "gan", "transformer", "cnn", "lstm"]
    for name in preferred_names:
        pdf_files = list(uploads_dir.glob(f"*{name}*.pdf"))
        if pdf_files:
            test_pdf = str(pdf_files[0])
            print(f"[OK] Found test PDF: {test_pdf}")
            break
    
    if not test_pdf:
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = str(pdf_files[0])
            print(f"[OK] Using PDF: {test_pdf}")
        else:
            print("[ERROR] No PDF files found in uploads/")
            return False
    
    try:
        # Test LLM client first
        print("\n[1] Testing OpenRouter LLM Client...")
        client = LLMClient(provider=LLMProvider.OPENROUTER, model="openai/gpt-oss-20b:free")
        test_response = client.generate("What is a neural network? One sentence.", temperature=0.7)
        print(f"[OK] LLM Client working: {test_response[:50]}...")
        
        # Create IR
        print("\n[2] Creating Intermediate Representation...")
        ir = PaperToCodeIR(
            paper_id=f"test_{Path(test_pdf).stem}",
            paper_metadata=PaperMetadata(title="Test Paper"),
            paper_path=test_pdf
        )
        print("[OK] IR created")
        
        # Configure agents
        print("\n[3] Configuring agents with free model...")
        agent_config = {
            "use_paper_analysis": True,
            "use_algorithm_interpretation": True,
            "use_api_mapping": True,
            "use_code_integration": True,
            "use_verification": False,  # Skip verification for now
            "use_debugging": False,  # Skip debugging for now
            "agents": {
                "paper_analysis": {
                    "use_llm": True,
                    "llm_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free",
                    "extract_diagrams": False
                },
                "algorithm_interpretation": {
                    "use_llm": True,
                    "llm_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free"
                },
                "api_mapping": {
                    "use_llm": True,
                    "llm_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free",
                    "default_framework": "pytorch"
                },
                "code_integration": {
                    "use_llm": True,
                    "llm_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free",
                    "include_examples": True
                }
            }
        }
        print("[OK] Configuration ready")
        
        # Run pipeline
        print("\n[4] Running complete pipeline...")
        print("-" * 70)
        planner = PlannerAgent(config=agent_config, llm_client=client)
        ir = planner.process(ir)
        
        # Display results
        print("\n" + "=" * 70)
        print("PIPELINE RESULTS")
        print("=" * 70)
        print(f"Status: {ir.status}")
        print(f"Algorithms Found: {len(ir.algorithms)}")
        print(f"Components Mapped: {len(ir.mapped_components)}")
        print(f"Files Generated: {len(ir.generated_code)}")
        
        if ir.algorithms:
            print("\nExtracted Algorithms:")
            for i, alg in enumerate(ir.algorithms, 1):
                print(f"  {i}. {alg.name}")
                print(f"     Type: {alg.type}")
                print(f"     Confidence: {alg.confidence:.2f}")
                if alg.description:
                    print(f"     Description: {alg.description[:60]}...")
        
        if ir.mapped_components:
            print("\nMapped Components:")
            for i, comp in enumerate(ir.mapped_components, 1):
                print(f"  {i}. {comp.algorithm_id} → {comp.framework}/{comp.library}")
        
        if ir.generated_code:
            print("\nGenerated Files:")
            for file_path in ir.generated_code.keys():
                file_info = ir.generated_code[file_path]
                print(f"  - {file_path} ({len(file_info.content)} chars)")
        
        # Save results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{ir.paper_id}_free_model_result.json"
        with open(output_file, 'w') as f:
            f.write(ir.to_json())
        print(f"\n[OK] Results saved to {output_file}")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Pipeline completed!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pipeline_with_free_model()

