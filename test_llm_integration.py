"""
Test script for LLM integration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from llm.llm_client import LLMClient, LLMProvider
from extractors.llm_algorithm_extractor import LLMAlgorithmExtractor

def test_llm_client():
    """Test basic LLM client functionality."""
    print("=" * 60)
    print("Testing LLM Client")
    print("=" * 60)
    
    try:
        client = LLMClient(provider=LLMProvider.OPENAI)
        print(f"[OK] LLM Client initialized with provider: {client.provider.value}")
        print(f"[OK] Model: {client.model}")
        
        # Test simple generation
        response = client.generate(
            prompt="What is a neural network? Answer in one sentence.",
            temperature=0.7
        )
        print(f"\n[OK] LLM Response received:")
        print(f"  {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def test_algorithm_extraction():
    """Test algorithm extraction with LLM."""
    print("\n" + "=" * 60)
    print("Testing LLM Algorithm Extraction")
    print("=" * 60)
    
    sample_text = """
    We propose a novel transformer architecture for natural language processing.
    The model uses multi-head attention mechanisms and feed-forward networks.
    Our approach combines self-attention with positional encoding to process
    sequential data. We also implement a residual connection and layer normalization
    following the standard transformer architecture.
    
    The training uses Adam optimizer with learning rate scheduling.
    We apply dropout regularization to prevent overfitting.
    """
    
    try:
        extractor = LLMAlgorithmExtractor(config={
            "use_llm": True,
            "fallback_to_rules": True,
            "llm_provider": "openai"
        })
        
        print("[OK] LLM Algorithm Extractor initialized")
        
        algorithms = extractor.extract_algorithms(sample_text)
        
        print(f"\n[OK] Extraction completed: Found {len(algorithms)} algorithms")
        
        for i, alg in enumerate(algorithms, 1):
            print(f"\n  Algorithm {i}:")
            print(f"    Name: {alg.name}")
            print(f"    Description: {alg.description[:80]}...")
            print(f"    Confidence: {alg.confidence:.2f}")
            if alg.parameters:
                print(f"    Parameters: {', '.join(alg.parameters[:3])}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM Integration Test Suite")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found in environment variables")
        print("   Please set it in your .env file or environment")
        return
    
    if api_key.startswith("sk-") and len(api_key) > 20:
        print(f"[OK] API Key found (length: {len(api_key)})")
    else:
        print("[WARNING] API key format may be incorrect")
    
    # Run tests
    results = []
    results.append(("LLM Client", test_llm_client()))
    results.append(("Algorithm Extraction", test_algorithm_extraction()))
    
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
        print("[FAILURE] Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

