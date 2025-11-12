"""
Complete Multi-Agent Pipeline Demo.
Demonstrates the full paper-to-code translation workflow.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from utils.intermediate_representation import PaperToCodeIR, PaperMetadata
from agents.planner_agent import PlannerAgent
from agents.verification_agent import VerificationAgent
from agents.debugging_agent import DebuggingAgent
from llm.llm_client import LLMClient, LLMProvider


def run_complete_pipeline(pdf_path: str, framework: str = "pytorch"):
    """
    Run the complete multi-agent pipeline.
    
    Args:
        pdf_path: Path to PDF paper
        framework: Target framework (pytorch, tensorflow, sklearn)
    """
    print("=" * 70)
    print("COMPLETE MULTI-AGENT PIPELINE DEMO")
    print("=" * 70)
    print(f"Paper: {pdf_path}")
    print(f"Framework: {framework}\n")
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not found")
        print("   Please set: $env:OPENROUTER_API_KEY='your_key_here'")
        return None
    
    # Initialize LLM client with OpenRouter
    try:
        llm_client = LLMClient(provider=LLMProvider.OPENROUTER, model="openai/gpt-oss-20b:free")
        print("[OK] OpenRouter LLM Client initialized (using free model)")
        print(f"[OK] Model: openai/gpt-oss-20b:free\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        return None
    
    # Create initial IR
    ir = PaperToCodeIR(
        paper_id=f"paper_{Path(pdf_path).stem}",
        paper_metadata=PaperMetadata(title="Research Paper"),
        paper_path=pdf_path
    )
    
    # Configure agents to use OpenRouter free model
    agent_config = {
        "use_paper_analysis": True,
        "use_algorithm_interpretation": True,
        "use_api_mapping": True,
        "use_code_integration": True,
        "agents": {
            "paper_analysis": {
                "use_llm": True,
                "llm_provider": "openrouter",
                "model": "openai/gpt-oss-20b:free",  # Free model!
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
                "default_framework": framework
            },
            "code_integration": {
                "use_llm": True,
                "llm_provider": "openrouter",
                "model": "openai/gpt-oss-20b:free",
                "include_examples": True,
                "include_docs": True
            }
        }
    }
    
    # Phase 1: Run Planner Agent (orchestrates all agents)
    print("-" * 70)
    print("PHASE 1: PLANNER AGENT (Orchestration)")
    print("-" * 70)
    
    planner = PlannerAgent(config=agent_config, llm_client=llm_client)
    ir = planner.process(ir)
    
    print(f"\n[Status] {ir.status}")
    print(f"[Algorithms Found] {len(ir.algorithms)}")
    print(f"[Components Mapped] {len(ir.mapped_components)}")
    print(f"[Files Generated] {len(ir.generated_code)}")
    
    if ir.status == "failed":
        print("\n[ERROR] Pipeline failed during planning phase")
        return ir
    
    # Phase 2: Verification (optional)
    print("\n" + "-" * 70)
    print("PHASE 2: VERIFICATION AGENT")
    print("-" * 70)
    
    verification_config = {
        "execute_code": False,  # Set to True to actually execute code
        "use_llm_comparison": True,
        "tolerance": 0.02
    }
    
    verification_agent = VerificationAgent(
        config=verification_config,
        llm_client=llm_client
    )
    ir = verification_agent.process(ir)
    
    if ir.verification_results:
        print(f"\n[Verification Status] {ir.verification_results.status}")
        if ir.verification_results.discrepancies:
            print(f"[Discrepancies] {len(ir.verification_results.discrepancies)}")
    
    # Phase 3: Debugging (if needed)
    if ir.verification_results and ir.verification_results.status != "pass":
        print("\n" + "-" * 70)
        print("PHASE 3: DEBUGGING AGENT")
        print("-" * 70)
        
        debugging_config = {
            "max_iterations": 2,
            "auto_fix": False,  # Set to True for automatic fixes
            "use_llm": True
        }
        
        debugging_agent = DebuggingAgent(
            config=debugging_config,
            llm_client=llm_client
        )
        ir = debugging_agent.process(ir)
        
        print(f"\n[Refinement Iterations] {len(ir.refinement_history)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Status: {ir.status}")
    print(f"Algorithms: {len(ir.algorithms)}")
    print(f"Mapped Components: {len(ir.mapped_components)}")
    print(f"Generated Files: {len(ir.generated_code)}")
    print(f"Refinement Iterations: {len(ir.refinement_history)}")
    
    if ir.generated_code:
        print("\nGenerated Files:")
        for file_path in ir.generated_code.keys():
            print(f"  - {file_path}")
    
    return ir


def main():
    """Main function."""
    # Check for test PDF
    uploads_dir = Path("uploads")
    test_pdf = None
    
    if uploads_dir.exists():
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = str(pdf_files[0])
        else:
            print("[INFO] No PDF files found in uploads/")
            print("[INFO] Please add a PDF file to the uploads/ directory")
            return
    
    if test_pdf:
        ir = run_complete_pipeline(test_pdf, framework="pytorch")
        
        # Save results
        if ir:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Save IR as JSON
            ir_json = ir.to_json()
            output_file = output_dir / f"{ir.paper_id}_pipeline_result.json"
            with open(output_file, 'w') as f:
                f.write(ir_json)
            print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    main()

