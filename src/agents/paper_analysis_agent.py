"""
Paper Analysis Agent - Enhanced with vision capabilities.
Extracts structured information from research papers using LLM and vision models.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent
from utils.intermediate_representation import (
    PaperToCodeIR, PaperMetadata, DiagramInfo, AlgorithmInfo
)
from llm.llm_client import LLMClient, LLMProvider
from llm.prompt_templates import PAPER_ANALYSIS_SYSTEM_PROMPT, PAPER_ANALYSIS_PROMPT
from parsers.pdf_parser import PDFParser


class PaperAnalysisAgent(BaseAgent):
    """
    Paper Analysis Agent with LLM and vision capabilities.
    Extracts text, diagrams, tables, and algorithms from research papers.
    """
    
    def __init__(self, config: Dict = None, llm_client=None):
        """Initialize Paper Analysis Agent."""
        super().__init__(config, llm_client)
        
        # Initialize LLM client if not provided
        if not self.llm_client:
            try:
                provider_str = self.config.get("llm_provider", "openai").lower()
                if provider_str == "openai":
                    provider = LLMProvider.OPENAI
                elif provider_str == "anthropic":
                    provider = LLMProvider.ANTHROPIC
                elif provider_str == "openrouter":
                    provider = LLMProvider.OPENROUTER
                elif provider_str == "groq":
                    provider = LLMProvider.GROQ
                else:
                    provider = LLMProvider.OPENAI  # Default
                self.llm_client = LLMClient(provider=provider)
                self.log_progress(f"Initialized LLM client: {provider.value}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
        
        # Initialize PDF parser
        self.pdf_parser = PDFParser(self.config.get('pdf_parser', {}))
        
        # Configuration
        self.extract_diagrams = self.config.get('extract_diagrams', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.use_vision = self.config.get('use_vision', True)
        self.max_text_length = self.config.get('max_text_length', 8000)
    
    def process(self, ir: PaperToCodeIR) -> PaperToCodeIR:
        """
        Process paper and extract structured information.
        
        Args:
            ir: Intermediate representation with paper path
            
        Returns:
            Updated IR with extracted content
        """
        if not self.validate_input(ir):
            ir.update_status("failed", self.agent_name)
            return ir
        
        self.update_ir_status(ir, "processing")
        self.log_progress("Starting paper analysis...")
        
        try:
            # Step 1: Extract text from PDF
            self.log_progress("Extracting text from PDF...")
            text_content = self.pdf_parser.extract_text(ir.paper_path)
            ir.text_sections = self._split_into_sections(text_content)
            ir.extracted_content['full_text'] = text_content
            self.log_progress(f"Extracted {len(text_content)} characters of text")
            
            # Step 2: Extract metadata
            self.log_progress("Extracting paper metadata...")
            metadata = self._extract_metadata(text_content)
            ir.paper_metadata = metadata
            self.log_progress(f"Extracted metadata: {metadata.title}")
            
            # Step 3: Extract diagrams (if enabled and LLM available)
            if self.extract_diagrams and self.use_vision and self.llm_client:
                self.log_progress("Extracting diagrams...")
                diagrams = self._extract_diagrams(ir.paper_path, text_content)
                ir.diagrams = diagrams
                self.log_progress(f"Found {len(diagrams)} diagrams")
            
            # Step 4: Extract algorithms using LLM
            if self.llm_client:
                self.log_progress("Extracting algorithms using LLM...")
                algorithms = self._extract_algorithms_llm(text_content)
                if algorithms:
                    ir.algorithms = algorithms
                    self.log_progress(f"✅ Found {len(algorithms)} algorithm(s) using LLM")
                    for alg in algorithms:
                        self.log_progress(f"  - {alg.name} (confidence: {alg.confidence:.2f})")
                else:
                    self.log_progress("⚠️ LLM extraction found no algorithms, trying fallback...")
                    # Fallback to rule-based extraction
                    from extractors.algorithm_extractor import AlgorithmExtractor
                    fallback_extractor = AlgorithmExtractor({})
                    rule_based_algorithms = fallback_extractor.extract_algorithms(text_content)
                    if rule_based_algorithms:
                        # Convert to AlgorithmInfo
                        algorithms = []
                        for alg in rule_based_algorithms:
                            algorithm = AlgorithmInfo(
                                name=alg.name,
                                description=alg.description,
                                type="unknown",
                                parameters=alg.parameters or [],
                                confidence=alg.confidence if hasattr(alg, 'confidence') else 0.5
                            )
                            algorithms.append(algorithm)
                        ir.algorithms = algorithms
                        self.log_progress(f"✅ Found {len(algorithms)} algorithm(s) using rule-based fallback")
                    else:
                        self.log_progress("⚠️ No algorithms found with either method")
            else:
                self.log_progress("LLM not available, skipping algorithm extraction")
            
            # Step 5: Extract equations and mathematical notation
            self.log_progress("Extracting equations...")
            equations = self._extract_equations(text_content)
            ir.equations = equations
            self.log_progress(f"Found {len(equations)} equations")
            
            self.update_ir_status(ir, "completed")
            self.log_progress("Paper analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in paper analysis: {e}", exc_info=True)
            ir.update_status("failed", self.agent_name)
            ir.extracted_content['error'] = str(e)
        
        return ir
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections."""
        # Simple section splitting by common headers
        import re
        sections = re.split(r'\n\s*\d+\.\s+[A-Z]', text)
        sections = [s.strip() for s in sections if s.strip()]
        return sections if sections else [text]
    
    def _extract_metadata(self, text: str) -> PaperMetadata:
        """Extract paper metadata from text."""
        import re
        
        # Try to extract title (usually in first few lines)
        title_match = re.search(r'^([A-Z][^\n]{10,200})\n', text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Unknown Title"
        
        # Try to extract authors
        authors = []
        author_patterns = [
            r'Authors?:\s*([^\n]+)',
            r'By:\s*([^\n]+)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.',
        ]
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                authors_str = match.group(1) if match.groups() else match.group(0)
                authors = [a.strip() for a in re.split(r'[,;]|and', authors_str)]
                break
        
        # Try to extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        year = int(year_match.group(0)) if year_match else None
        
        return PaperMetadata(
            title=title[:200],  # Limit title length
            authors=authors[:10],  # Limit authors
            year=year
        )
    
    def _extract_diagrams(self, paper_path: str, text: str) -> List[DiagramInfo]:
        """Extract and analyze diagrams from paper."""
        diagrams = []
        
        if not self.llm_client or not self.use_vision:
            self.log_progress("Vision parsing disabled or LLM not available", "info")
            return diagrams
        
        try:
            import pdfplumber
            import tempfile
            
            paper_path_obj = Path(paper_path)
            if not paper_path_obj.exists():
                self.log_progress(f"Paper path not found: {paper_path}", "warning")
                return diagrams
            
            self.log_progress("Extracting images from PDF...")
            
            # Extract images from PDF using pdfplumber
            extracted_images = []
            with pdfplumber.open(paper_path_obj) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract images from this page
                        images = page.images
                        for img_idx, img in enumerate(images):
                            # Get image bounding box
                            bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                            
                            # Try to extract image data
                            # pdfplumber doesn't directly extract image bytes, so we use PyMuPDF as fallback
                            extracted_images.append({
                                'page': page_num + 1,
                                'index': img_idx,
                                'bbox': bbox,
                                'width': img.get('width', 0),
                                'height': img.get('height', 0)
                            })
                    except Exception as e:
                        self.logger.warning(f"Error extracting images from page {page_num + 1}: {e}")
                        continue
            
            # If pdfplumber didn't find images, try PyMuPDF
            if not extracted_images:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(paper_path_obj)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        image_list = page.get_images()
                        for img_idx, img in enumerate(image_list):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Save to temporary file for vision analysis
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_ext}') as tmp_file:
                                tmp_file.write(image_bytes)
                                tmp_path = tmp_file.name
                            
                            extracted_images.append({
                                'page': page_num + 1,
                                'index': img_idx,
                                'path': tmp_path,
                                'format': image_ext
                            })
                    doc.close()
                except ImportError:
                    self.log_progress("PyMuPDF not available, skipping image extraction", "warning")
                except Exception as e:
                    self.logger.warning(f"Error extracting images with PyMuPDF: {e}")
            
            if not extracted_images:
                self.log_progress("No images found in PDF", "info")
                return diagrams
            
            self.log_progress(f"Found {len(extracted_images)} image(s), analyzing with vision model...")
            
            # Analyze each image with vision model
            for img_info in extracted_images[:5]:  # Limit to first 5 images for performance
                try:
                    if 'path' in img_info and Path(img_info['path']).exists():
                        image_path = img_info['path']
                    else:
                        # Skip if we don't have image path (pdfplumber-only extraction)
                        continue
                    
                    # Use vision model to analyze image
                    vision_prompt = f"""Analyze this diagram/image from a research paper. 
                    Identify:
                    1. Diagram type (architecture diagram, algorithm flowchart, mathematical graph, table, etc.)
                    2. Key components and their relationships
                    3. Any text labels or annotations
                    4. Algorithm or model structure if visible
                    
                    Return a structured description that can help understand the paper's methodology."""
                    
                    analysis = self.llm_client.analyze_with_vision(
                        image_path=image_path,
                        prompt=vision_prompt,
                        system_prompt="You are an expert at analyzing scientific diagrams and extracting structured information from research paper figures."
                    )
                    
                    # Create DiagramInfo
                    diagram = DiagramInfo(
                        page_number=img_info['page'],
                        diagram_type="unknown",  # Could be extracted from analysis
                        description=analysis[:500] if analysis else "No description available",
                        extracted_structure={},
                        confidence=0.7
                    )
                    diagrams.append(diagram)
                    
                    # Clean up temporary file
                    try:
                        if 'path' in img_info:
                            os.unlink(img_info['path'])
                    except:
                        pass
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing image from page {img_info.get('page', 'unknown')}: {e}")
                    continue
            
            self.log_progress(f"Successfully analyzed {len(diagrams)} diagram(s)")
            
        except Exception as e:
            self.logger.error(f"Error in diagram extraction: {e}", exc_info=True)
            self.log_progress(f"Diagram extraction failed: {str(e)}", "warning")
        
        return diagrams
    
    def _extract_algorithms_llm(self, text: str) -> List[AlgorithmInfo]:
        """Extract algorithms using LLM."""
        if not self.llm_client:
            return []
        
        # Truncate text if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            self.log_progress(f"Text truncated to {self.max_text_length} characters")
        
        try:
            prompt = PAPER_ANALYSIS_PROMPT.format(text=text)
            
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=PAPER_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.3
            )
            
            # Convert response to AlgorithmInfo objects
            algorithms = []
            
            # Check if response is an error dict
            if isinstance(response, dict) and response.get("_is_error"):
                self.logger.warning("JSON parsing failed, attempting manual extraction...")
                raw_response = response.get("raw_response", "")
                # Try to manually extract algorithms from raw response
                import re
                # Look for algorithm entries in the raw text
                algorithm_blocks = re.findall(r'\{"name":\s*"[^"]+".*?\}', raw_response, re.DOTALL)
                if algorithm_blocks:
                    try:
                        algorithm_data = [json.loads(block) for block in algorithm_blocks]
                    except:
                        algorithm_data = []
                else:
                    algorithm_data = []
            elif isinstance(response, list):
                algorithm_data = response
            elif isinstance(response, dict) and "algorithms" in response:
                algorithm_data = response["algorithms"]
            elif isinstance(response, dict) and not response.get("_is_error"):
                # Single algorithm object
                algorithm_data = [response]
            else:
                algorithm_data = []
            
            for alg_data in algorithm_data:
                try:
                    algorithm = AlgorithmInfo(
                        name=alg_data.get("name", "Unknown"),
                        description=alg_data.get("description", ""),
                        type=alg_data.get("type", "unknown"),
                        pseudocode=alg_data.get("pseudocode"),
                        mathematical_notation=alg_data.get("mathematical_notation"),
                        parameters=alg_data.get("key_parameters", []) or [],
                        hyperparameters=alg_data.get("hyperparameters", {}),
                        complexity=alg_data.get("complexity"),
                        framework_suggestions=alg_data.get("framework_suggestions", []) or [],
                        confidence=float(alg_data.get("confidence", 0.5))
                    )
                    algorithms.append(algorithm)
                except Exception as e:
                    self.logger.warning(f"Failed to create AlgorithmInfo: {e}")
                    continue
            
            return algorithms
            
        except Exception as e:
            self.logger.error(f"Error in LLM algorithm extraction: {e}")
            return []
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        import re
        
        # Look for LaTeX-style equations
        equation_patterns = [
            r'\$[^$]+\$',  # Inline math: $...$
            r'\$\$[^$]+\$\$',  # Display math: $$...$$
            r'\\\[.*?\\\]',  # LaTeX display: \[...\]
            r'\\\(.*?\\\)',  # LaTeX inline: \(...\)
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_equations = []
        for eq in equations:
            if eq not in seen:
                seen.add(eq)
                unique_equations.append(eq)
        
        return unique_equations

