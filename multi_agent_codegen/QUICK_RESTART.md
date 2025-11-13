# Quick Restart Guide

## Restart Flask Server

### Method 1: Using run.py (Recommended)
```powershell
cd C:\Users\acer\Music\Kanishka\multi_agent_codegen
python run.py
```

### Method 2: Using restart script
```powershell
cd C:\Users\acer\Music\Kanishka\multi_agent_codegen
.\restart_server.ps1
```

### Method 3: Manual restart
1. **Stop the server:** Press `Ctrl+C` in the terminal where Flask is running
2. **Start again:**
   ```powershell
   python run.py
   ```

## Verify Server is Running

Open your browser and go to: **http://localhost:5000**

You should see the Multi-Agent CodeGen interface.

## If Port is Already in Use

If you get "Port 5000 already in use":

1. **Find and stop the process:**
   ```powershell
   Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess
   Stop-Process -Id <PID> -Force
   ```

2. **Or change the port** in `app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

## What's New After Restart

After restarting with the updated code, you'll have:

✅ **Improved Test Generation**
- Better function name detection
- Proper import statements
- Cleaner test code

✅ **Enhanced Error Handling**
- Syntax validation before testing
- More detailed error messages
- Better pytest output

✅ **Smarter Debugging**
- Better error analysis
- More effective code fixes
- Improved error summaries

## Test the Improvements

Try this specification:
```
Implement a function to check if a string is a palindrome. 
The function should handle empty strings and be case-insensitive.
```

The system should now:
1. Generate better tests
2. Show detailed error messages if tests fail
3. Fix issues more effectively

