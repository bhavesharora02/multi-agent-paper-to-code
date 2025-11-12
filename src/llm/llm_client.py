"""
LLM client for multi-agent pipeline.
Supports OpenAI and Anthropic APIs.
"""

import os
import base64
from typing import Optional, Dict, List, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GROQ = "groq"


class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (OpenAI or Anthropic)
            model: Specific model to use (optional, uses default if not provided)
        """
        self.provider = provider
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        try:
            if self.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self.client = OpenAI(api_key=api_key)
                self.model = self.model or "gpt-4o"  # Updated to valid model name
                logger.info(f"Initialized OpenAI client with model: {self.model}")
                
            elif self.provider == LLMProvider.ANTHROPIC:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                self.client = Anthropic(api_key=api_key)
                self.model = self.model or "claude-3-opus-20240229"
                logger.info(f"Initialized Anthropic client with model: {self.model}")
                
            elif self.provider == LLMProvider.OPENROUTER:
                # OpenRouter uses requests library
                import requests
                self.requests = requests
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
                self.api_key = api_key
                self.model = self.model or "openai/gpt-4o"
                self.base_url = "https://openrouter.ai/api/v1"
                logger.info(f"Initialized OpenRouter client with model: {self.model}")
                
            elif self.provider == LLMProvider.GROQ:
                # Groq is compatible with OpenAI client library
                from openai import OpenAI
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                # Log first 20 chars of API key for verification (for debugging)
                logger.info(f"Initializing Groq client with API key: {api_key[:20]}...")
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                self.model = self.model or "llama-3.3-70b-versatile"  # Default Groq model
                logger.info(f"Initialized Groq client with model: {self.model}")
        except ImportError as e:
            logger.error(f"Failed to import LLM library: {e}")
            if self.provider == LLMProvider.OPENROUTER:
                raise ImportError("Please install the required package: pip install requests")
            elif self.provider == LLMProvider.GROQ or self.provider == LLMProvider.OPENAI:
                raise ImportError("Please install the required package: pip install openai")
            raise ImportError(f"Please install the required package: pip install {'openai' if self.provider == LLMProvider.OPENAI else 'anthropic'}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        try:
            if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.GROQ:
                # Groq uses OpenAI-compatible API
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or kwargs.get("max_tokens", 4096),
                    **{k: v for k, v in kwargs.items() if k != "max_tokens"}
                )
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                system = system_prompt or ""
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or 4096,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    **kwargs
                )
                return response.content[0].text
                
            elif self.provider == LLMProvider.OPENROUTER:
                # OpenRouter API call with retry logic for rate limiting
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                # Retry logic with exponential backoff for rate limiting
                import time
                max_retries = 3
                base_delay = 2  # Start with 2 seconds
                
                for attempt in range(max_retries):
                    try:
                        response = self.requests.post(
                            url=f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://github.com/bhavesh-arora/multi-agent-pipeline",
                                "X-Title": "ML/DL Paper to Code Pipeline"
                            },
                            json={
                                "model": self.model,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens or 4096,
                                **{k: v for k, v in kwargs.items() if k not in ["max_tokens"]}
                            },
                            timeout=120
                        )
                        
                        # Handle rate limiting (429)
                        if response.status_code == 429:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                                logger.warning(f"Rate limited (429). Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(delay)
                                continue
                            else:
                                logger.error("Rate limit exceeded. Max retries reached.")
                                raise Exception("Rate limit exceeded. Please wait before trying again or use a paid model.")
                        
                        response.raise_for_status()
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                        
                    except self.requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1 and "429" in str(e):
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Request failed with 429. Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                        raise
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> Dict:
        """
        Generate JSON response from LLM.
        
        Args:
            prompt: User prompt requesting JSON
            system_prompt: System prompt
            temperature: Lower temperature for more consistent JSON
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON format instruction to prompt
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
        
        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt or "You are a helpful assistant that responds with valid JSON.",
            temperature=temperature,
            **kwargs
        )
        
        # Try to parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            import re
            
            # Step 1: Try to extract JSON from markdown code blocks (with or without closing ```)
            # Handle cases where closing ``` might be missing
            json_match = re.search(r'```json\s*(.*?)(?:\s*```|$)', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to find complete JSON by matching brackets/braces
                    # For arrays: find matching [ and ]
                    if json_str.strip().startswith('['):
                        bracket_count = 0
                        for i, char in enumerate(json_str):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    try:
                                        return json.loads(json_str[:i+1])
                                    except:
                                        pass
                    # For objects: find matching { and }
                    elif json_str.strip().startswith('{'):
                        brace_count = 0
                        for i, char in enumerate(json_str):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        return json.loads(json_str[:i+1])
                                    except:
                                        pass
            
            # Step 2: Try code blocks without language tag
            json_match = re.search(r'```\s*(\[.*?\]|\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
            
            # Step 3: Try to find JSON array (common for algorithm lists)
            # Look for array that starts after ```json or at start of response
            # First, try to find the array start position
            array_start = response.find('[')
            if array_start >= 0:
                # Extract from array start to end, then find matching bracket
                remaining = response[array_start:]
                bracket_count = 0
                for i, char in enumerate(remaining):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = remaining[:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                # Try cleaning up the string
                                json_str = json_str.strip()
                                # Remove any trailing markdown
                                json_str = re.sub(r'\s*```\s*$', '', json_str, flags=re.MULTILINE)
                                try:
                                    return json.loads(json_str)
                                except:
                                    pass
                            break
            
            # Step 4: Try to find JSON object
            object_pattern = r'(\{[\s\S]*?\})'
            object_match = re.search(object_pattern, response, re.DOTALL)
            if object_match:
                json_str = object_match.group(1)
                try:
                    return json.loads(json_str)
                except:
                    pass
            
            # Step 5: Try to fix common JSON issues and parse
            cleaned = response.strip()
            # Remove markdown code block markers
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
            # Remove trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            # Try to find first valid JSON structure
            for pattern in [r'\[[\s\S]*?\]', r'\{[\s\S]*?\}']:
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except:
                        pass
            
            logger.warning(f"Failed to parse JSON from response: {e}")
            logger.warning(f"Response preview: {response[:500]}...")
            # Return the raw response wrapped in a dict so caller can handle it
            return {"raw_response": response, "error": "JSON parsing failed", "_is_error": True}
    
    def analyze_with_vision(
        self, 
        image_path: str, 
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze image using vision model.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            system_prompt: System prompt
            
        Returns:
            Analysis result
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            if self.provider == LLMProvider.OPENAI:
                # Determine image format
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = f"image/{ext[1:]}" if ext else "image/png"
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                })
                
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",  # Use vision model
                    messages=messages,
                    max_tokens=4096
                )
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.OPENROUTER:
                # OpenRouter vision support
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = f"image/{ext[1:]}" if ext else "image/png"
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                })
                
                response = self.requests.post(
                    url=f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/bhavesh-arora/multi-agent-pipeline",
                        "X-Title": "ML/DL Paper to Code Pipeline"
                    },
                    json={
                        "model": "openai/gpt-4o",  # Use vision-capable model
                        "messages": messages,
                        "max_tokens": 4096
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            else:  # Anthropic Claude 3 Vision
                with open(image_path, "rb") as img_file:
                    image_data_bytes = img_file.read()
                
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    system=system_prompt or "",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64.b64encode(image_data_bytes).decode('utf-8')
                                    }
                                },
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

