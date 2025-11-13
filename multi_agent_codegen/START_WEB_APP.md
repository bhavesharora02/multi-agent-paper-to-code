# Starting the Web Application

## Quick Start

1. **Navigate to the project directory:**
```bash
cd multi_agent_codegen
```

2. **Install dependencies (if not already installed):**
```bash
pip install -r requirements.txt
```

3. **Start the web server:**
```bash
python app.py
```

4. **Open your browser:**
Navigate to `http://localhost:5000`

## What You'll See

- **Home Page**: Beautiful landing page with project information
- **Demo Section**: Input area to enter code specifications
- **Real-time Progress**: Watch agents work in real-time
- **Results**: View generated code, test results, and download

## Features

- ✅ Enter code specifications in natural language
- ✅ Watch multi-agent workflow in real-time
- ✅ See progress updates as agents work
- ✅ View generated code with syntax highlighting
- ✅ Download generated code as Python file
- ✅ See test results and iteration count

## Example Usage

1. Enter a specification like:
   ```
   Implement a function to check if a string is a palindrome. 
   The function should handle empty strings and be case-insensitive.
   ```

2. Click "Generate Code"

3. Watch the agents:
   - **Coder Agent**: Generates initial code
   - **Tester Agent**: Creates and runs tests
   - **Debugger Agent**: Fixes any errors (if needed)
   - **Optimizer Agent**: Optimizes and documents code

4. View and download the final code

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, you can change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
```

### API Key Issues
The Groq API key is automatically set in `app.py`. If you need to use a different key:
- Set environment variable: `$env:GROQ_API_KEY="your_key"`
- Or update the key directly in `app.py`

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

- Try different code specifications
- Experiment with max iterations
- Check the generated code quality
- Review test results

