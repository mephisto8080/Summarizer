from typing import List, Dict
from .base_client import BaseLLMClient

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local LLM interactions.
    Ollama allows running models like Llama, Mistral, etc. locally.
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3", temperature: float = 0):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Model name (e.g., 'llama3', 'mistral', etc.)
            temperature: Sampling temperature
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for OllamaClient. Install it with: pip install requests")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        temperature = kwargs.get("temperature", self.temperature)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json().get("response", "")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from a chat conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/chat"
        temperature = kwargs.get("temperature", self.temperature)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json().get("message", {}).get("content", "")

    def validate_config(self, config: Dict) -> bool:
        """
        Validate Ollama configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
