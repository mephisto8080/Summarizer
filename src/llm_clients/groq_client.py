from groq import Groq
from typing import List, Dict
from .base_client import BaseLLMClient


class GroqClient(BaseLLMClient):
    """Groq API client for LLM interactions."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0, max_tokens: int = 2000):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using chat interface.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from a chat conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def validate_config(self, config: Dict) -> bool:
        """
        Validate Groq configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        required_keys = ["api_key"]
        return all(key in config for key in required_keys)
