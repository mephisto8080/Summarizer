import re
from typing import List


class TextCompressor:
    """Compress text by removing noise and redundant information."""

    def __init__(self, max_chars: int = 700):
        """
        Initialize the text compressor.

        Args:
            max_chars: Maximum number of characters to keep after compression
        """
        self.max_chars = max_chars

    def compress(self, text: str) -> str:
        """
        Compress a single text string.

        Args:
            text: Text to compress

        Returns:
            Compressed text
        """
        # Remove multiple spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()

        # Remove page numbers and headings
        text = re.sub(r"PAGE\s*\d+", " ", text, flags=re.IGNORECASE)

        # Keep only text sentences (drop noisy OCR numbers and special characters)
        text = re.sub(r"[^a-zA-Z0-9.,;:()?\- ]", " ", text)

        # Remove multiple spaces again
        text = re.sub(r"\s+", " ", text).strip()

        # Trim to max length
        return text[:self.max_chars]

    def compress_batch(self, texts: List[str]) -> List[str]:
        """
        Compress a batch of texts.

        Args:
            texts: List of texts to compress

        Returns:
            List of compressed texts
        """
        return [self.compress(text) for text in texts]

    def compress_with_custom_rules(self, text: str, remove_patterns: List[str] = None) -> str:
        """
        Compress text with custom removal patterns.

        Args:
            text: Text to compress
            remove_patterns: List of regex patterns to remove

        Returns:
            Compressed text
        """
        # Apply standard compression
        compressed = self.compress(text)

        # Apply custom patterns if provided
        if remove_patterns:
            for pattern in remove_patterns:
                compressed = re.sub(pattern, " ", compressed)
            compressed = re.sub(r"\s+", " ", compressed).strip()

        return compressed[:self.max_chars]
