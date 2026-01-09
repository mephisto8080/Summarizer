from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters specific to the implementation

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from a chat conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters specific to the implementation

        Returns:
            Generated text
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for the LLM client.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        return True
