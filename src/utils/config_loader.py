import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "processing": {
                "chunk_size": 1800,
                "chunk_overlap": 250,
                "meta_section_size": 5,
                "compression_max_chars": 700
            },
            "models": {
                "groq": {
                    "model_name": "llama-3.3-70b-versatile",
                    "temperature": 0,
                    "max_tokens_meta": 3500,
                    "max_tokens_global": 1800
                }
            },
            "output": {
                "format": "markdown",
                "include_metadata": True
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'processing.chunk_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_env(self, key: str, default: str = None) -> str:
        """
        Get value from environment variables.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(key, default)

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.get("processing", {})

    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific provider.

        Args:
            provider: Provider name (e.g., 'groq', 'ollama')

        Returns:
            Model configuration dictionary
        """
        return self.get(f"models.{provider}", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})
