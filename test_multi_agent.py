"""
Test script for multi-agent pipeline.
Demonstrates the enhanced Paper Analysis Agent and LLM Code Generator.
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
from agents.paper_analysis_agent import PaperAnalysisAgent
from generators.llm_code_generator import LLMCodeGenerator
from extractors.algorithm_extractor import Algorithm

def test_paper_analysis_agent():
    """Test the Paper Analysis Agent."""
    print("=" * 60)
    print("Testing Paper Analysis Agent")
    print("=" * 60)
    
    # Check for a test PDF
    test_pdf = None
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = str(pdf_files[0])
            print(f"[OK] Found test PDF: {test_pdf}")
        else:
            print("[INFO] No PDF files found in uploads/ directory")
            print("[INFO] Creating a mock IR for testing...")
            return test_with_mock_data()
    else:
        print("[INFO] No uploads directory found")
        return test_with_mock_data()
    
    try:
        # Create initial IR
        ir = PaperToCodeIR(
            paper_id="test_paper_001",
            paper_metadata=PaperMetadata(title="Test Paper"),
            paper_path=test_pdf
        )
        
        # Initialize agent
        agent = PaperAnalysisAgent(config={
            "use_llm": True,
            "llm_provider": "openai",
            "extract_diagrams": False,  # Skip diagrams for now
            "use_vision": False
        })
        
        print("\n[OK] Paper Analysis Agent initialized")
        print("[INFO] Processing paper...")
        
        # Process paper
        ir = agent.process(ir)
        
        # Display results
        print(f"\n[OK] Processing completed. Status: {ir.status}")
        print(f"[INFO] Extracted {len(ir.text_sections)} text sections")
        print(f"[INFO] Found {len(ir.algorithms)} algorithms")
        print(f"[INFO] Found {len(ir.equations)} equations")
        
        if ir.algorithms:
            print("\n[INFO] Extracted Algorithms:")
            for i, alg in enumerate(ir.algorithms[:3], 1):  # Show first 3
                print(f"  {i}. {alg.name} (confidence: {alg.confidence:.2f})")
                print(f"     Type: {alg.type}")
                if alg.description:
                    print(f"     Description: {alg.description[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_mock_data():
    """Test with mock data when no PDF is available."""
    print("\n[INFO] Testing with mock text data...")
    
    sample_text = """
    Title: Transformer Architecture for Natural Language Processing
    
    We propose a novel transformer architecture that uses multi-head attention
    mechanisms. The model consists of encoder and decoder stacks with residual
    connections and layer normalization.
    
    The training procedure uses Adam optimizer with learning rate scheduling.
    We apply dropout regularization to prevent overfitting.
    """
    
    try:
        # Create mock IR
        ir = PaperToCodeIR(
            paper_id="mock_paper_001",
            paper_metadata=PaperMetadata(
                title="Transformer Architecture for NLP",
                authors=["Test Author"],
                year=2024
            ),
            paper_path="mock_path.pdf"
        )
        
        # Manually set text content
        ir.text_sections = [sample_text]
        ir.extracted_content['full_text'] = sample_text
        
        # Initialize agent
        agent = PaperAnalysisAgent(config={
            "use_llm": True,
            "llm_provider": "openai",
            "extract_diagrams": False,
            "use_vision": False
        })
        
        print("[OK] Agent initialized")
        print("[INFO] Extracting algorithms from mock text...")
        
        # Extract algorithms
        algorithms = agent._extract_algorithms_llm(sample_text)
        ir.algorithms = algorithms
        
        print(f"[OK] Found {len(algorithms)} algorithms")
        if algorithms:
            for i, alg in enumerate(algorithms, 1):
                print(f"  {i}. {alg.name} (confidence: {alg.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_code_generator():
    """Test the LLM Code Generator."""
    print("\n" + "=" * 60)
    print("Testing LLM Code Generator")
    print("=" * 60)
    
    try:
        # Create sample algorithms
        algorithms = [
            Algorithm(
                name="Transformer",
                description="Multi-head attention transformer architecture",
                parameters=["num_layers", "hidden_size", "num_heads"],
                confidence=0.9
            ),
            Algorithm(
                name="Adam Optimizer",
                description="Adam optimization algorithm with learning rate scheduling",
                parameters=["learning_rate", "beta1", "beta2"],
                confidence=0.8
            )
        ]
        
        # Initialize generator
        generator = LLMCodeGenerator(config={
            "llm_provider": "openai",
            "use_fallback": True
        })
        
        print("[OK] LLM Code Generator initialized")
        print("[INFO] Generating code for PyTorch...")
        
        # Generate code
        code = generator.generate_code(algorithms, framework="pytorch")
        
        print(f"[OK] Code generated ({len(code)} characters)")
        print("\n[INFO] First 500 characters of generated code:")
        print("-" * 60)
        print(code[:500])
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Multi-Agent Pipeline Test Suite")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found")
        print("   Please set it in your environment or .env file")
        return
    
    if api_key.startswith("sk-") and len(api_key) > 20:
        print(f"[OK] API Key found (length: {len(api_key)})")
    else:
        print("[WARNING] API key format may be incorrect")
    
    # Run tests
    results = []
    results.append(("Paper Analysis Agent", test_paper_analysis_agent()))
    results.append(("LLM Code Generator", test_llm_code_generator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILURE] Some tests failed. Check errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

