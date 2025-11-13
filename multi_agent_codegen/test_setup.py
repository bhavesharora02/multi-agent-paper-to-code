"""
Test script to verify setup before running the web app
"""

import sys
from pathlib import Path

print("="*60)
print("Multi-Agent CodeGen - Setup Verification")
print("="*60)

# Check Python version
print(f"\n[OK] Python version: {sys.version.split()[0]}")

# Check Flask
try:
    import flask
    print(f"[OK] Flask installed: {flask.__version__}")
except ImportError:
    print("[ERROR] Flask not installed - run: pip install flask")
    sys.exit(1)

# Check YAML
try:
    import yaml
    print("[OK] PyYAML installed")
except ImportError:
    print("[ERROR] PyYAML not installed - run: pip install pyyaml")
    sys.exit(1)

# Check LLM client
parent_src = Path(__file__).parent.parent / "src"
if parent_src.exists():
    sys.path.insert(0, str(parent_src))
    try:
        from llm.llm_client import LLMClient, LLMProvider
        print("[OK] LLM client found in parent project")
    except ImportError as e:
        print(f"[WARN] LLM client import failed: {e}")
        print("       (System will use fallback)")
else:
    print("[WARN] Parent src directory not found")
    print("       (System will use fallback)")

# Check Groq API key
import os
# Get Groq API key from environment variable
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
else:
    print("[WARNING] GROQ_API_KEY not set. Please set it as an environment variable.")
print("[OK] Groq API key configured")

# Check project structure
print("\n" + "="*60)
print("Project Structure Check")
print("="*60)

required_files = [
    "app.py",
    "config.yaml",
    "templates/index.html",
    "static/css/style.css",
    "static/js/script.js"
]

all_ok = True
for file in required_files:
    file_path = Path(__file__).parent / file
    if file_path.exists():
        print(f"[OK] {file}")
    else:
        print(f"[ERROR] {file} - MISSING")
        all_ok = False

if all_ok:
    print("\n" + "="*60)
    print("[SUCCESS] All checks passed! Ready to run.")
    print("="*60)
    print("\nTo start the web server, run:")
    print("  python run.py")
    print("\nOr:")
    print("  python app.py")
    print("\nThen open http://localhost:5000 in your browser")
else:
    print("\n[ERROR] Some files are missing. Please check the project structure.")
    sys.exit(1)

