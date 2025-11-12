"""
Flask web application for ML/DL Paper to Code automation system.
"""

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import os
import tempfile
import json
from pathlib import Path
import sys
import threading
import time
from werkzeug.utils import secure_filename

# Add src to path for imports
sys.path.append('src')
from parsers.pdf_parser import PDFParser
from extractors.algorithm_extractor import AlgorithmExtractor
from extractors.llm_algorithm_extractor import LLMAlgorithmExtractor
from generators.code_generator import CodeGenerator
from generators.llm_code_generator import LLMCodeGenerator
# Multi-agent pipeline imports
from agents.planner_agent import PlannerAgent
from utils.intermediate_representation import PaperToCodeIR, PaperMetadata
from llm.llm_client import LLMClient, LLMProvider

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Disable Flask's auto-reloader to prevent issues with background threads
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for tracking processing status
processing_status = {}
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_paper_async(file_path, framework, task_id):
    """Process paper using complete multi-agent pipeline."""
    try:
        processing_status[task_id] = {'status': 'processing', 'progress': 0, 'message': 'Initializing multi-agent pipeline...'}
        
        # Load configuration
        config_path = Path('config/default.yaml')
        config_data = {}
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        
        # Check if we should use multi-agent pipeline
        use_multi_agent = config_data.get('use_multi_agent_pipeline', True)
        
        print(f"[DEBUG] use_multi_agent_pipeline = {use_multi_agent}")
        print(f"[DEBUG] Config loaded: {config_data.get('extractor', {}).get('use_llm', False)}")
        
        if use_multi_agent:
            print("[DEBUG] Using MULTI-AGENT PIPELINE with PlannerAgent")
            # Use complete multi-agent pipeline
            processing_status[task_id]['progress'] = 5
            processing_status[task_id]['message'] = 'Initializing Planner Agent...'
            
            # Initialize LLM client if needed
            llm_client = None
            extractor_config = config_data.get('extractor', {})
            if extractor_config.get('use_llm', False):
                try:
                    provider_str = extractor_config.get('llm_provider', 'groq').lower()
                    model = extractor_config.get('model', 'llama-3.3-70b-versatile')
                    if provider_str == 'groq':
                        llm_client = LLMClient(provider=LLMProvider.GROQ, model=model)
                    elif provider_str == 'openrouter':
                        llm_client = LLMClient(provider=LLMProvider.OPENROUTER, model=model)
                    elif provider_str == 'openai':
                        llm_client = LLMClient(provider=LLMProvider.OPENAI)
                    elif provider_str == 'anthropic':
                        llm_client = LLMClient(provider=LLMProvider.ANTHROPIC)
                except Exception as e:
                    print(f"Failed to initialize LLM client: {e}")
                    llm_client = None
            
            # Create initial IR
            processing_status[task_id]['progress'] = 10
            processing_status[task_id]['message'] = 'Creating intermediate representation...'
            
            ir = PaperToCodeIR(
                paper_id=task_id,
                paper_metadata=PaperMetadata(title="Research Paper"),
                paper_path=file_path
            )
            
            # Configure agents
            agent_config = {
                "use_paper_analysis": True,
                "use_algorithm_interpretation": True,
                "use_api_mapping": True,
                "use_code_integration": True,
                "use_verification": config_data.get('use_verification', False),  # Optional
                "use_debugging": config_data.get('use_debugging', False),  # Optional
                "agents": {
                    "paper_analysis": {
                        "use_llm": extractor_config.get('use_llm', False),
                        "llm_provider": extractor_config.get('llm_provider', 'groq'),
                        "model": extractor_config.get('model', 'llama-3.3-70b-versatile'),
                        "extract_diagrams": True,  # Enable diagram extraction
                        "use_vision": True  # Enable vision model for diagram analysis
                    },
                    "algorithm_interpretation": {
                        "use_llm": extractor_config.get('use_llm', False),
                        "llm_provider": extractor_config.get('llm_provider', 'groq'),
                        "model": extractor_config.get('model', 'llama-3.3-70b-versatile')
                    },
                    "api_mapping": {
                        "use_llm": extractor_config.get('use_llm', False),
                        "llm_provider": extractor_config.get('llm_provider', 'groq'),
                        "model": extractor_config.get('model', 'llama-3.3-70b-versatile'),
                        "default_framework": framework
                    },
                    "code_integration": {
                        "use_llm": config_data.get('generator', {}).get('use_llm', False),
                        "llm_provider": config_data.get('generator', {}).get('llm_provider', 'groq'),
                        "model": config_data.get('generator', {}).get('model', 'llama-3.3-70b-versatile'),
                        "include_examples": True,
                        "include_docs": True
                    }
                }
            }
            
            # Initialize and run Planner Agent with progress callback
            processing_status[task_id]['progress'] = 15
            processing_status[task_id]['message'] = 'Starting multi-agent pipeline...'
            
            # Define progress callback function
            def update_progress(progress, message):
                """Update processing status with progress callback."""
                if task_id in processing_status:
                    processing_status[task_id]['progress'] = progress
                    processing_status[task_id]['message'] = message
                    print(f"[PROGRESS] {progress}%: {message}")  # Also print to console for debugging
            
            planner = PlannerAgent(config=agent_config, llm_client=llm_client, progress_callback=update_progress)
            
            # Run the pipeline (progress updates will come from callback)
            ir = planner.process(ir)
            
            # Extract generated code from IR
            generated_code = ""
            if ir.generated_code:
                # Get main code file
                for file_path_key, file_info in ir.generated_code.items():
                    if isinstance(file_info, dict):
                        # Handle dict format
                        if file_path_key.endswith(".py") and "main" in file_path_key.lower():
                            generated_code = file_info.get('content', '')
                            break
                    else:
                        # Handle GeneratedFile object
                        if file_path_key.endswith(".py") and "main" in file_path_key.lower():
                            generated_code = file_info.content
                            break
                
                # If no main file, combine all Python files
                if not generated_code:
                    python_files = []
                    for file_path_key, file_info in ir.generated_code.items():
                        if isinstance(file_info, dict):
                            if file_path_key.endswith(".py"):
                                python_files.append(file_info.get('content', ''))
                        else:
                            if file_path_key.endswith(".py"):
                                python_files.append(file_info.content)
                    generated_code = "\n\n".join(python_files)
            
            # Fallback: generate code from algorithms if no code in IR
            if not generated_code and ir.algorithms:
                processing_status[task_id]['message'] = 'Generating code from algorithms...'
                from extractors.algorithm_extractor import Algorithm
                algorithms = [
                    Algorithm(
                        name=alg.name,
                        description=alg.description,
                        parameters=alg.parameters or [],
                        confidence=alg.confidence if hasattr(alg, 'confidence') else 0.8
                    )
                    for alg in ir.algorithms
                ]
                
                generator_config = config_data.get('generator', {})
                if generator_config.get('use_llm', False):
                    code_generator = LLMCodeGenerator(generator_config, llm_client=llm_client)
                else:
                    code_generator = CodeGenerator(generator_config)
                
                generated_code = code_generator.generate_code(algorithms, framework)
            
            processing_status[task_id]['progress'] = 90
            processing_status[task_id]['message'] = 'Finalizing output...'
            
            # Save results
            output_filename = f"{task_id}_{framework}_generated_code.py"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            if generated_code:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
            
            # Determine if code was generated by LLM or fallback
            code_source = "llm" if ir.generated_code and len(ir.generated_code) > 0 else "fallback"
            if not generated_code and ir.algorithms:
                code_source = "fallback"  # Used fallback code generator
            
            # Store results
            processing_results[task_id] = {
                'status': 'completed',
                'algorithms_found': len(ir.algorithms),
                'framework': framework,
                'output_file': output_filename,
                'code': generated_code,
                'code_source': code_source,  # Track if code came from LLM or fallback
                'algorithms': [
                    {
                        'name': alg.name,
                        'description': alg.description,
                        'confidence': alg.confidence if hasattr(alg, 'confidence') else 0.8,
                        'parameters': alg.parameters or [],
                        'complexity': alg.complexity if hasattr(alg, 'complexity') else None,
                        'type': alg.type if hasattr(alg, 'type') else 'unknown'
                    }
                    for alg in ir.algorithms
                ],
                'mapped_components': len(ir.mapped_components),
                'generated_files': len(ir.generated_code),
                'pipeline_status': ir.status,
                'current_agent': ir.current_agent
            }
            
            print(f"[INFO] Code generation source: {code_source.upper()}")
            if code_source == "fallback":
                print("[WARNING] Code was generated using fallback (template-based) method due to LLM rate limits or errors")
            
            processing_status[task_id] = {'status': 'completed', 'progress': 100, 'message': 'Multi-agent pipeline completed!'}
        
        else:
            # Fallback to old method
            print("[DEBUG] Using LEGACY/TEMPLATE-BASED method (NOT multi-agent)")
            processing_status[task_id]['progress'] = 10
            processing_status[task_id]['message'] = 'Using legacy processing method...'
            
            pdf_parser = PDFParser(config_data.get('pdf_parser', {}))
            
            extractor_config = config_data.get('extractor', {})
            if extractor_config.get('use_llm', False):
                try:
                    algorithm_extractor = LLMAlgorithmExtractor(extractor_config)
                except Exception as e:
                    print(f"Failed to initialize LLM extractor: {e}. Falling back to rule-based.")
                    algorithm_extractor = AlgorithmExtractor(extractor_config)
            else:
                algorithm_extractor = AlgorithmExtractor(extractor_config)
            
            generator_config = config_data.get('generator', {})
            if generator_config.get('use_llm', False):
                try:
                    code_generator = LLMCodeGenerator(generator_config)
                except Exception as e:
                    print(f"Failed to initialize LLM code generator: {e}. Falling back to template-based.")
                    code_generator = CodeGenerator(generator_config)
            else:
                code_generator = CodeGenerator(generator_config)
            
            processing_status[task_id]['progress'] = 30
            processing_status[task_id]['message'] = 'Parsing PDF content...'
            
            text_content = pdf_parser.extract_text(file_path)
            
            processing_status[task_id]['progress'] = 50
            processing_status[task_id]['message'] = 'Detecting algorithms...'
            
            algorithms = algorithm_extractor.extract_algorithms(text_content)
            
            processing_status[task_id]['progress'] = 70
            processing_status[task_id]['message'] = f'Found {len(algorithms)} algorithms. Generating code...'
            
            generated_code = code_generator.generate_code(algorithms, framework)
            
            processing_status[task_id]['progress'] = 90
            processing_status[task_id]['message'] = 'Finalizing output...'
            
            output_filename = f"{task_id}_{framework}_generated_code.py"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            processing_results[task_id] = {
                'status': 'completed',
                'algorithms_found': len(algorithms),
                'framework': framework,
                'output_file': output_filename,
                'code': generated_code,
                'algorithms': [
                    {
                        'name': alg.name,
                        'description': alg.description,
                        'confidence': alg.confidence,
                        'parameters': alg.parameters or [],
                        'complexity': alg.complexity
                    }
                    for alg in algorithms
                ]
            }
            
            processing_status[task_id] = {'status': 'completed', 'progress': 100, 'message': 'Processing completed!'}
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        processing_status[task_id] = {'status': 'error', 'progress': 0, 'message': f'Error: {str(e)}'}
        processing_results[task_id] = {'status': 'error', 'error': str(e)}

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    framework = request.form.get('framework', 'pytorch')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        task_id = f"{int(time.time())}_{filename}"
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
        file.save(file_path)
        
        # Start background processing
        thread = threading.Thread(target=process_paper_async, args=(file_path, framework, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': task_id,
            'task_id': task_id,
            'message': 'File uploaded successfully. Processing started.'
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

@app.route('/check-config')
def check_config():
    """Check current configuration."""
    config_path = Path('config/default.yaml')
    config_data = {}
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    
    return jsonify({
        'use_multi_agent': config_data.get('use_multi_agent_pipeline', False),
        'use_llm_extractor': config_data.get('extractor', {}).get('use_llm', False),
        'llm_provider': config_data.get('extractor', {}).get('llm_provider', 'N/A'),
        'model': config_data.get('extractor', {}).get('model', 'N/A')
    })

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status."""
    if task_id in processing_status:
        status = processing_status[task_id]
        # Ensure status has required fields
        if 'status' not in status:
            status['status'] = 'processing'
        if 'progress' not in status:
            status['progress'] = 0
        if 'message' not in status:
            status['message'] = 'Processing...'
        return jsonify(status)
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/results/<task_id>')
def get_results(task_id):
    """Get processing results."""
    if task_id in processing_results:
        return jsonify(processing_results[task_id])
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/download/<task_id>')
def download_file(task_id):
    """Download generated code file."""
    if task_id in processing_results and processing_results[task_id]['status'] == 'completed':
        output_file = processing_results[task_id]['output_file']
        file_path = os.path.join(OUTPUT_FOLDER, output_file)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=output_file)
    
    return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<task_id>')
def preview_code(task_id):
    """Preview generated code."""
    if task_id in processing_results and processing_results[task_id]['status'] == 'completed':
        output_file = processing_results[task_id]['output_file']
        file_path = os.path.join(OUTPUT_FOLDER, output_file)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            return jsonify({'code': code_content})
    
    return jsonify({'error': 'Code not found'}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
