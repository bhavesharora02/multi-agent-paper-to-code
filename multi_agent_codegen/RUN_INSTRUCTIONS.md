# How to Run Multi-Agent Code Generation

## Quick Start (3 Commands)

### Option 1: Using run.py (Recommended)
```powershell
cd multi_agent_codegen
python run.py
```

### Option 2: Using app.py directly
```powershell
cd multi_agent_codegen
python app.py
```

### Option 3: Using Flask command
```powershell
cd multi_agent_codegen
set FLASK_APP=app.py
flask run
```

## Step-by-Step Instructions

### 1. Open Terminal/PowerShell
Navigate to the project directory:
```powershell
cd C:\Users\acer\Music\Kanishka\multi_agent_codegen
```

### 2. Install Dependencies (First Time Only)
```powershell
pip install -r requirements.txt
```

If you get errors, install key packages individually:
```powershell
pip install flask pyyaml openai requests pytest
```

### 3. Start the Server
```powershell
python run.py
```

Or:
```powershell
python app.py
```

### 4. Open Browser
Once you see:
```
✓ Groq API key configured
✓ Server starting on http://localhost:5000
```

Open your browser and go to: **http://localhost:5000**

## What You Should See

### In Terminal:
```
============================================================
Multi-Agent Code Generation Web Server
============================================================
✓ Groq API key configured
✓ Server starting on http://localhost:5000
============================================================

Open http://localhost:5000 in your browser
Press Ctrl+C to stop the server

 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### In Browser:
- Beautiful landing page
- Input area for code specifications
- "Generate Code" button
- Real-time progress tracking

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```powershell
pip install flask
```

### Error: "ModuleNotFoundError: No module named 'llm'"
**Solution:** This means the parent project's src directory isn't accessible. The system will use fallback, but you can:
1. Make sure parent project exists at: `../src/`
2. Or install LLM dependencies: `pip install openai anthropic requests`

### Error: "Port 5000 already in use"
**Solution:** Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Error: "GROQ_API_KEY not found"
**Solution:** The key is already set in `app.py` and `run.py`. If you still get this error, set it manually:
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

## Testing the System

1. **Enter a simple specification:**
   ```
   Implement a function to check if a string is a palindrome
   ```

2. **Click "Generate Code"**

3. **Watch the progress:**
   - Coder Agent generates code
   - Tester Agent runs tests
   - Debugger Agent fixes errors (if needed)
   - Optimizer Agent optimizes code

4. **View results:**
   - Generated code
   - Test results
   - Download code

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

## Next Steps

- Try different code specifications
- Experiment with max iterations
- Review generated code quality
- Check the `outputs/` folder for saved code

---

**Ready to run?** Just execute: `python run.py`

