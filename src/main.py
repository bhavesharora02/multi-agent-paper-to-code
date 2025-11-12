"""
Main application entry point for ML/DL Paper to Code automation system.
"""

import click
import yaml
from pathlib import Path
from parsers.pdf_parser import PDFParser
from extractors.algorithm_extractor import AlgorithmExtractor
from generators.code_generator import CodeGenerator


@click.command()
@click.option('--input', '-i', required=True, help='Path to input PDF file')
@click.option('--output', '-o', required=True, help='Path to output Python file')
@click.option('--config', '-c', default='config/default.yaml', help='Configuration file path')
@click.option('--framework', '-f', default='pytorch', type=click.Choice(['pytorch', 'tensorflow', 'sklearn']), help='Target ML framework')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(input, output, config, framework, verbose):
    """
    Convert ML/DL research papers to executable Python code.
    
    This tool automatically extracts algorithms and methodologies from academic papers
    and generates corresponding Python implementations using the specified framework.
    """
    try:
        # Load configuration
        config_path = Path(config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
        
        if verbose:
            click.echo(f"Processing paper: {input}")
            click.echo(f"Output file: {output}")
            click.echo(f"Framework: {framework}")
        
        # Initialize components
        pdf_parser = PDFParser(config_data.get('pdf_parser', {}))
        algorithm_extractor = AlgorithmExtractor(config_data.get('extractor', {}))
        code_generator = CodeGenerator(config_data.get('generator', {}))
        
        # Process the paper
        if verbose:
            click.echo("Extracting text from PDF...")
        
        text_content = pdf_parser.extract_text(input)
        
        if verbose:
            click.echo("Extracting algorithms and methodologies...")
        
        algorithms = algorithm_extractor.extract_algorithms(text_content)
        
        if verbose:
            click.echo(f"Found {len(algorithms)} algorithms")
            click.echo("Generating code...")
        
        generated_code = code_generator.generate_code(algorithms, framework)
        
        # Write output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        click.echo(f"Successfully generated code: {output}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
