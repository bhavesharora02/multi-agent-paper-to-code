# Web Application - Multi-Agent Code Generation

## Overview

A beautiful, modern web interface for the Multi-Agent Code Generation system. The web app allows users to interact with the multi-agent system through an intuitive UI.

## Features

### 🎨 Modern UI
- Beautiful gradient design
- Responsive layout
- Real-time progress tracking
- Smooth animations

### 🤖 Multi-Agent Workflow
- **Coder Agent**: Generates code from specifications
- **Tester Agent**: Creates and runs unit tests
- **Debugger Agent**: Fixes errors automatically
- **Optimizer Agent**: Optimizes and documents code

### 📊 Real-time Updates
- Live progress bar
- Current agent status
- Iteration counter
- Test results

### 💾 Code Management
- Preview generated code
- Download as Python file
- View test results
- See iteration history

## Quick Start

```bash
cd multi_agent_codegen
python app.py
```

Then open `http://localhost:5000` in your browser.

## API Endpoints

### POST `/api/process`
Start processing a code specification.

**Request:**
```json
{
  "specification": "Implement is_palindrome function",
  "max_iterations": 10
}
```

**Response:**
```json
{
  "task_id": "uuid-here"
}
```

### GET `/api/status/<task_id>`
Get processing status.

**Response:**
```json
{
  "status": "processing",
  "progress": 50,
  "message": "Running Tester Agent...",
  "iteration": 2,
  "current_agent": "tester"
}
```

### GET `/api/download/<task_id>`
Download generated code as Python file.

### GET `/api/test-results/<task_id>`
Get test execution results.

## File Structure

```
multi_agent_codegen/
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Styles
│   └── js/
│       └── script.js     # Frontend JavaScript
└── outputs/              # Generated code files
```

## Configuration

The Groq API key is automatically configured in `app.py`. To use a different key:

1. Set environment variable:
   ```powershell
   $env:GROQ_API_KEY="your_key_here"
   ```

2. Or update `app.py` directly

## Usage Example

1. **Enter Specification:**
   ```
   Implement a function to check if a string is a palindrome. 
   The function should handle empty strings and be case-insensitive.
   ```

2. **Set Max Iterations:** (default: 10)

3. **Click "Generate Code"**

4. **Watch Progress:**
   - See which agent is running
   - Track iteration count
   - Monitor progress percentage

5. **View Results:**
   - Generated code
   - Test results
   - Iteration count
   - Download code

## Features in Detail

### Progress Tracking
- Real-time progress bar (0-100%)
- Current agent name
- Iteration counter
- Status messages

### Code Preview
- Syntax-highlighted code display
- Modal popup for full-screen view
- Copy-friendly format

### Test Results
- Pass/fail status
- Test output
- Error messages (if any)
- Execution details

## Troubleshooting

### Port Conflict
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### API Key Issues
- Check that Groq API key is set
- Verify key is valid
- Check console for error messages

### Import Errors
Install dependencies:
```bash
pip install -r requirements.txt
```

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Next Steps

- Try different code specifications
- Experiment with max iterations
- Review generated code quality
- Check test coverage

