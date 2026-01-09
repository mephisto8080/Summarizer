from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import pandas as pd


class TextSplitter:
    """Split text into manageable chunks for processing."""

    def __init__(self, chunk_size: int = 1800, chunk_overlap: int = 250,
                 separators: List[str] = None):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", ".", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def split_pages(self, pages: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Split pages into chunks.

        Args:
            pages: List of page dictionaries with 'page' and 'text' keys

        Returns:
            DataFrame with columns: page, chunk_id, text
        """
        chunks = []

        for pg in pages:
            page_text = pg["text"] or ""
            parts = self.splitter.split_text(page_text)

            if not parts:
                parts = [""]

            for j, text in enumerate(parts):
                chunks.append({
                    "page": pg["page"],
                    "chunk_id": f"{pg['page']}_{j + 1}",
                    "text": text
                })

        return pd.DataFrame(chunks)

    def create_meta_sections(self, df_chunks: pd.DataFrame, meta_size: int = 5) -> List[str]:
        """
        Create meta-sections by combining chunks.

        Args:
            df_chunks: DataFrame containing chunks
            meta_size: Number of chunks to combine into one meta-section

        Returns:
            List of meta-section texts
        """
        meta_sections = []

        for i in range(0, len(df_chunks), meta_size):
            meta_text = "\n".join(df_chunks["text"].iloc[i:i + meta_size].tolist())
            meta_sections.append(meta_text)

        return meta_sections
