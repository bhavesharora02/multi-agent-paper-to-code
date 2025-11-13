"""
Simple runner script for Multi-Agent Code Generation Web App
"""

import os
import sys
from pathlib import Path

# Set Groq API key from environment variable
# Set it using: $env:GROQ_API_KEY="your_key_here" (Windows) or export GROQ_API_KEY="your_key_here" (Linux/Mac)
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
else:
    print("[WARNING] GROQ_API_KEY not set. Please set it as an environment variable.")

# Add parent src to path for LLM client
parent_src = Path(__file__).parent.parent / "src"
if parent_src.exists():
    sys.path.insert(0, str(parent_src))
    print(f"[OK] Added parent src to path: {parent_src}")

# Import and run app
try:
    from app import app
    
    print("\n" + "="*60)
    print("Multi-Agent Code Generation Web Server")
    print("="*60)
    print("[OK] Groq API key configured")
    print("[OK] Server starting on http://localhost:5000")
    print("="*60)
    print("\nOpen http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the multi_agent_codegen directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Check that the parent project's src directory exists")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

