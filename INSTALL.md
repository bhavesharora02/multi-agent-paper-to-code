# ML/DL Paper to Code System
# Installation and Usage Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Run the System**
   ```bash
   python src/main.py --input your_paper.pdf --output generated_code.py
   ```

## Installation Options

### Option 1: Direct Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Option 2: Development Installation
```bash
pip install -e .
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Option 3: Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage Examples

### Basic Usage
```bash
python src/main.py --input paper.pdf --output code.py
```

### With Specific Framework
```bash
python src/main.py --input paper.pdf --output pytorch_code.py --framework pytorch
python src/main.py --input paper.pdf --output tf_code.py --framework tensorflow
python src/main.py --input paper.pdf --output sklearn_code.py --framework sklearn
```

### With Custom Configuration
```bash
python src/main.py --input paper.pdf --output code.py --config config/custom.yaml
```

### Verbose Output
```bash
python src/main.py --input paper.pdf --output code.py --verbose
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_system.py::TestPDFParser
python -m pytest tests/test_system.py::TestAlgorithmExtractor
python -m pytest tests/test_system.py::TestCodeGenerator
```

## Configuration

The system uses YAML configuration files. See `config/default.yaml` for available options.

### Key Configuration Options

- **PDF Parser**: Choose between `pdfplumber` and `pypdf2`
- **Algorithm Extractor**: Set confidence thresholds and detection patterns
- **Code Generator**: Configure framework-specific settings
- **Logging**: Set log levels and output destinations

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **PDF Parsing Errors**
   - Ensure PDF is not password-protected
   - Try different PDF parsing methods in config
   - Check if PDF contains text (not just images)

3. **No Algorithms Detected**
   - Ensure paper contains clear algorithm descriptions
   - Adjust confidence threshold in config
   - Check algorithm detection patterns

4. **Import Errors**
   - Verify all dependencies are installed
   - Check Python version (requires 3.8+)
   - Ensure virtual environment is activated

### Getting Help

- Check the logs for detailed error messages
- Use `--verbose` flag for more output
- Review the example configurations in `config/`
- See `examples/documentation.py` for usage examples

## Development

### Code Style
```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Running Tests
```bash
pytest tests/ -v
```
