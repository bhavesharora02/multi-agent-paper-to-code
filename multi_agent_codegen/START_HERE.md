# 🚀 START HERE - Run Multi-Agent CodeGen

## Quick Start (Copy & Paste)

### Step 1: Navigate to Project
```powershell
cd C:\Users\acer\Music\Kanishka\multi_agent_codegen
```

### Step 2: Test Setup (Optional)
```powershell
python test_setup.py
```

### Step 3: Start Web Server
```powershell
python run.py
```

### Step 4: Open Browser
Go to: **http://localhost:5000**

---

## That's It! 🎉

The web interface will open where you can:
- Enter code specifications
- Watch AI agents work in real-time
- View generated code
- Download results

---

## If You Get Errors

### "ModuleNotFoundError: No module named 'flask'"
```powershell
pip install flask pyyaml openai requests pytest
```

### "Port 5000 already in use"
Edit `app.py` line 270, change port to 5001:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Other Issues?
Check `RUN_INSTRUCTIONS.md` for detailed troubleshooting.

---

## What to Try First

1. Enter this specification:
   ```
   Implement a function to check if a string is a palindrome
   ```

2. Click "Generate Code"

3. Watch the agents work!

---

**Ready?** Run: `python run.py`

