from typing import List, Dict
from tqdm import tqdm
import pandas as pd

from ..extractors.pdf_extractor import PDFExtractor
from ..splitters.text_splitter import TextSplitter
from ..compressors.text_compressor import TextCompressor
from ..llm_clients.base_client import BaseLLMClient


class HierarchicalSummarizer:
    """
    Hierarchical summarization using meta-sections approach.
    Process: PDF -> Pages -> Chunks -> Meta-sections -> Meta-summaries -> Global summary
    """

    def __init__(self, llm_client: BaseLLMClient, config: Dict):
        """
        Initialize hierarchical summarizer.

        Args:
            llm_client: LLM client instance
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config

        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_splitter = TextSplitter(
            chunk_size=config.get("chunk_size", 1800),
            chunk_overlap=config.get("chunk_overlap", 250),
            separators=config.get("separators", ["\n\n", "\n", ".", " ", ""])
        )
        self.text_compressor = TextCompressor(
            max_chars=config.get("compression_max_chars", 700)
        )

        self.meta_section_size = config.get("meta_section_size", 5)

    def process_pdf(self, pdf_path: str, save_intermediate: bool = True) -> Dict:
        """
        Process a PDF file and generate hierarchical summary.

        Args:
            pdf_path: Path to PDF file
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary containing all results
        """
        print(f"Processing PDF: {pdf_path}")

        # Step 1: Extract pages
        print("Step 1: Extracting pages...")
        pages = self.pdf_extractor.extract_pages(pdf_path)
        print(f"Extracted {len(pages)} pages")

        # Step 2: Split into chunks
        print("Step 2: Splitting text into chunks...")
        df_chunks = self.text_splitter.split_pages(pages)
        print(f"Created {len(df_chunks)} chunks")

        # Step 3: Create meta-sections
        print("Step 3: Creating meta-sections...")
        meta_sections = self.text_splitter.create_meta_sections(
            df_chunks, self.meta_section_size
        )
        print(f"Created {len(meta_sections)} meta-sections")

        # Step 4: Compress meta-sections
        print("Step 4: Compressing meta-sections...")
        compressed_meta = self.text_compressor.compress_batch(meta_sections)
        print(f"Compressed to average {sum(len(s) for s in compressed_meta) // len(compressed_meta)} chars per section")

        # Step 5: Generate meta-summaries
        print("Step 5: Generating meta-summaries...")
        meta_summaries = self._generate_meta_summaries(compressed_meta)
        print(f"Generated {len(meta_summaries)} meta-summaries")

        # Step 6: Generate global summary
        print("Step 6: Generating global summary...")
        global_summary = self._generate_global_summary(meta_summaries)
        print("Global summary generated")

        return {
            "pages": pages,
            "chunks": df_chunks,
            "meta_sections": meta_sections,
            "compressed_meta": compressed_meta,
            "meta_summaries": meta_summaries,
            "global_summary": global_summary
        }

    def _generate_meta_summaries(self, compressed_meta: List[str]) -> List[Dict]:
        """
        Generate summaries for each meta-section.

        Args:
            compressed_meta: List of compressed meta-sections

        Returns:
            List of meta-summary dictionaries
        """
        meta_prompt = """
You are an expert at summarizing regulatory and legal documents.

Below are multiple META-SECTIONS.
Each META-SECTION is a pre-compressed excerpt from the original document.

Your task:
- Expand each META-SECTION into a 120–200 word refined summary.
- Maintain accuracy.
- Add back missing clarity and connections.
- NO hallucination.
- Format MUST be:

###SECTION <N>
<summary>

META-SECTIONS BELOW:
"""

        for i, sec in enumerate(compressed_meta):
            meta_prompt += f"\n<META id='{i + 1}'> {sec} </META>"

        # Generate summaries
        max_tokens = self.config.get("max_tokens_meta", 3500)
        response = self.llm_client.generate(meta_prompt, max_tokens=max_tokens)

        # Parse response
        meta_summaries = []
        for block in response.split("###SECTION"):
            block = block.strip()
            if not block or not block[0].isdigit():
                continue

            try:
                number = int(block.split("\n")[0].strip())
                summary = "\n".join(block.split("\n")[1:]).strip()
                meta_summaries.append({"section": number, "summary": summary})
            except (ValueError, IndexError):
                continue

        return sorted(meta_summaries, key=lambda x: x["section"])

    def _generate_global_summary(self, meta_summaries: List[Dict]) -> str:
        """
        Generate global summary from meta-summaries.

        Args:
            meta_summaries: List of meta-summary dictionaries

        Returns:
            Global summary text
        """
        all_meta = "\n\n".join(
            [f"<S{m['section']}>\n{m['summary']}\n</S{m['section']}>"
             for m in meta_summaries]
        )

        global_prompt = f"""
You are a senior expert summarizer for complex long documents.

Create a unified GLOBAL SUMMARY from the refined meta-section summaries.

Include:
- Main Purpose
- Problems addressed
- Key findings
- Observations
- Critical outcomes
- Conclusions
- Important insights

Avoid repetition. Provide a coherent 4–6 paragraph summary. Provide the summarization under each sections of Include the sections as headers.

META-SECTION SUMMARIES:
{all_meta}

Write the final global summary:
"""

        max_tokens = self.config.get("max_tokens_global", 1800)
        return self.llm_client.generate(global_prompt, max_tokens=max_tokens)
