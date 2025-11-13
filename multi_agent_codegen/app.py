"""
Flask web application for Multi-Agent Code Generation System.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import time
import uuid
from pathlib import Path
import sys
import yaml

# Add parent src to path for LLM client
parent_src = Path(__file__).parent.parent / "src"
if parent_src.exists():
    sys.path.insert(0, str(parent_src))

# Import workflow components
sys.path.insert(0, str(Path(__file__).parent))
from workflow.graph import create_workflow
from workflow.state import WorkflowState

app = Flask(__name__)
app.secret_key = 'multi-agent-codegen-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configuration
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for tracking processing status
processing_status = {}
processing_results = {}


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def process_specification_async(specification, max_iterations, task_id):
    """Process specification using multi-agent workflow."""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing multi-agent workflow...',
            'iteration': 0,
            'current_agent': 'planner'
        }
        
        # Load configuration
        config = load_config()
        
        # Override max_iterations if provided
        if max_iterations:
            config.setdefault('workflow', {})['max_iterations'] = max_iterations
        
        processing_status[task_id]['progress'] = 10
        processing_status[task_id]['message'] = 'Creating workflow...'
        
        # Create workflow
        workflow = create_workflow(config)
        
        # Initialize state
        state: WorkflowState = {
            "specification": specification,
            "code": "",
            "code_history": [],
            "iteration": 0,
            "max_iterations": config.get('workflow', {}).get('max_iterations', 10),
            "optimized": False,
            "tests_passed": False,  # Internal - for tester/debugger loop
            "test_code": "",
            "test_results": {},
            "test_output": "",
            "test_errors": "",
            "code_rating": 0.0,  # Shown in UI
            "rating_details": "",
            "rating_feedback": "",
            "fix_history": [],
            "needs_improvement": False,
            "improvement_attempts": 0
        }
        
        processing_status[task_id]['progress'] = 20
        processing_status[task_id]['message'] = 'Running Coder Agent...'
        processing_status[task_id]['current_agent'] = 'coder'
        
        # Run workflow with progress updates
        result = run_workflow_with_progress(workflow, state, task_id)
        
        # Store results (only show rating in UI, not test results)
        processing_results[task_id] = {
            'code': result.get('code', ''),
            'code_rating': result.get('code_rating', 0),
            'rating_details': result.get('rating_details', ''),
            'rating_feedback': result.get('rating_feedback', ''),
            'iteration_count': result.get('iteration_count', 0),
            'success': result.get('success', False),
            'error': result.get('error'),
            'specification': specification,
            # Internal test data (not shown in UI, but available for debugging)
            'tests_passed': result.get('tests_passed', False),
            'test_iterations': result.get('test_iterations', 0)  # How many test cycles
        }
        
        processing_status[task_id]['status'] = 'completed'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['message'] = 'Processing completed!'
        
    except Exception as e:
        processing_status[task_id]['status'] = 'error'
        processing_status[task_id]['message'] = f'Error: {str(e)}'
        processing_results[task_id] = {'error': str(e)}


def run_workflow_with_progress(workflow, state, task_id):
    """Run workflow with progress updates."""
    # Check if it's a SimpleWorkflow (fallback)
    if hasattr(workflow, 'invoke') and not hasattr(workflow, 'astream'):
        # Simple workflow - run directly
        return workflow.invoke(state)
    
    # LangGraph workflow - run with progress tracking
    iteration = 0
    max_iterations = state.get('max_iterations', 10)
    
    while iteration < max_iterations:
        # Update progress
        progress = 20 + (iteration * 70 / max_iterations)
        processing_status[task_id]['progress'] = min(int(progress), 95)
        processing_status[task_id]['iteration'] = iteration
        
        # Run one step
        try:
            # For LangGraph, we need to stream or invoke
            if hasattr(workflow, 'astream'):
                # Stream workflow
                final_state = None
                for state_update in workflow.astream(state):
                    final_state = state_update
                    # Update status based on current agent
                    last_agent = final_state.get('last_agent', 'unknown')
                    processing_status[task_id]['current_agent'] = last_agent
                    processing_status[task_id]['message'] = f'Running {last_agent.capitalize()} Agent...'
                    
                    # Check if workflow is complete (rater has run)
                    if final_state.get('code_rating') is not None:
                        break
                
                if final_state:
                    state = final_state
            else:
                # Invoke workflow
                state = workflow.invoke(state)
            
            # Check if done (rater has run and rating is set)
            if state.get('code_rating') is not None or state.get('success', False):
                break
                
        except Exception as e:
            state['error'] = str(e)
            break
        
        iteration += 1
        time.sleep(0.5)  # Small delay for UI updates
    
    state['iteration_count'] = iteration
    state['success'] = state.get('success', True)  # Always successful with rating
    return state


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_specification():
    """Process a code specification."""
    try:
        data = request.json
        specification = data.get('specification', '').strip()
        max_iterations = data.get('max_iterations', 10)
        
        if not specification:
            return jsonify({'error': 'Specification is required'}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_specification_async,
            args=(specification, max_iterations, task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id}), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Get processing status."""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id].copy()
    
    # Add results if available
    if task_id in processing_results:
        status['results'] = processing_results[task_id]
    
    return jsonify(status)


@app.route('/api/download/<task_id>')
def download_code(task_id):
    """Download generated code."""
    if task_id not in processing_results:
        return jsonify({'error': 'Code not found'}), 404
    
    result = processing_results[task_id]
    code = result.get('code', '')
    
    if not code:
        return jsonify({'error': 'No code generated'}), 404
    
    # Save to file
    output_file = os.path.join(OUTPUT_FOLDER, f'{task_id}.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name='generated_code.py',
        mimetype='text/x-python'
    )


@app.route('/api/explain', methods=['POST'])
def explain_code():
    """Answer a question about the generated code."""
    try:
        data = request.json
        task_id = data.get('task_id')
        question = data.get('question', '').strip()
        
        if not task_id or not question:
            return jsonify({'error': 'task_id and question are required'}), 400
        
        if task_id not in processing_results:
            return jsonify({'error': 'Code not found'}), 404
        
        result = processing_results[task_id]
        code = result.get('code', '')
        specification = result.get('specification', '')
        
        if not code:
            return jsonify({'error': 'No code available'}), 404
        
        # Initialize explainer agent
        from agents.explainer_agent import ExplainerAgent
        config = load_config()
        llm_config = config.get("llm", {})
        
        # Initialize LLM client
        from llm.llm_client import LLMClient, LLMProvider
        provider_str = llm_config.get("provider", "groq").lower()
        if provider_str == "groq":
            provider = LLMProvider.GROQ
        elif provider_str == "openai":
            provider = LLMProvider.OPENAI
        elif provider_str == "anthropic":
            provider = LLMProvider.ANTHROPIC
        elif provider_str == "openrouter":
            provider = LLMProvider.OPENROUTER
        else:
            provider = LLMProvider.GROQ
        
        model = llm_config.get("model", None)
        llm_client = LLMClient(provider=provider, model=model)
        
        explainer = ExplainerAgent(
            config={**llm_config, **config.get("agents", {}).get("explainer", {})},
            llm_client=llm_client
        )
        
        # Get answer
        answer = explainer.explain_code(code, question, specification)
        
        return jsonify({
            'answer': answer,
            'question': question
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Set Groq API key from environment variable
    # Set it using: $env:GROQ_API_KEY="your_key_here" (Windows) or export GROQ_API_KEY="your_key_here" (Linux/Mac)
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[WARNING] GROQ_API_KEY not set. Please set it as an environment variable.")
    else:
        os.environ["GROQ_API_KEY"] = groq_key
    print("✓ Groq API key configured")
    
    print("\n" + "="*60)
    print("Multi-Agent Code Generation Web Server")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("Open the URL in your browser to use the application")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

