"""
Test OpenRouter AI integration.
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


def test_openrouter():
    """Test OpenRouter API integration."""
    print("=" * 70)
    print("OPENROUTER AI INTEGRATION TEST")
    print("=" * 70)
    
    # Set API key
    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        return
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    try:
        # Initialize OpenRouter client
        print("\n[1] Initializing OpenRouter client...")
        client = LLMClient(provider=LLMProvider.OPENROUTER, model="openai/gpt-4o")
        print(f"[OK] Client initialized with model: {client.model}")
        
        # Test simple generation
        print("\n[2] Testing text generation...")
        response = client.generate(
            prompt="What is a neural network? Answer in one sentence.",
            temperature=0.7
        )
        print(f"[OK] Response received:")
        print(f"     {response[:100]}...")
        
        # Test JSON generation
        print("\n[3] Testing JSON generation...")
        json_response = client.generate_json(
            prompt="Extract algorithms from this text: 'We use a transformer architecture with multi-head attention and a CNN for image processing.' Return as JSON with algorithm names.",
            temperature=0.3
        )
        print(f"[OK] JSON response received:")
        print(f"     {json_response}")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] OpenRouter integration working!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_openrouter()

