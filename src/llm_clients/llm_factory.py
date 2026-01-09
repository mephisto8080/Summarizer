from typing import Dict, Any
from .base_client import BaseLLMClient
from .groq_client import GroqClient
from .ollama_client import OllamaClient


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(provider: str, config: Dict[str, Any]) -> BaseLLMClient:
        """
        Create an LLM client based on provider type.

        Args:
            provider: Provider name ('groq', 'ollama', etc.)
            config: Configuration dictionary

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()

        if provider == "groq":
            return GroqClient(
                api_key=config.get("api_key"),
                model=config.get("model_name", "qwen/qwen3-32b"),
                temperature=config.get("temperature", 0),
                max_tokens=config.get("max_tokens", 2000)
            )

        elif provider == "ollama":
            return OllamaClient(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model_name", "llama3"),
                temperature=config.get("temperature", 0)
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported providers: groq, ollama")

    @staticmethod
    def get_supported_providers() -> list:
        """
        Get list of supported providers.

        Returns:
            List of provider names
        """
        return ["groq", "ollama"]
