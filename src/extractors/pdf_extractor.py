import fitz
from typing import List, Dict


class PDFExtractor:
    """Extract text content from PDF files page by page."""

    def __init__(self):
        pass

    def extract_pages(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from all pages of a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page number and text content
        """
        doc = fitz.open(pdf_path)
        pages = []

        for i in range(len(doc)):
            page_text = doc[i].get_text("text")
            pages.append({
                "page": i + 1,
                "text": page_text
            })

        doc.close()
        return pages

    def extract_page_range(self, pdf_path: str, start_page: int, end_page: int) -> List[Dict[str, any]]:
        """
        Extract text from a specific range of pages.

        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)

        Returns:
            List of dictionaries containing page number and text content
        """
        doc = fitz.open(pdf_path)
        pages = []

        for i in range(start_page - 1, min(end_page, len(doc))):
            page_text = doc[i].get_text("text")
            pages.append({
                "page": i + 1,
                "text": page_text
            })

        doc.close()
        return pages

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the total number of pages in a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Total number of pages
        """
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
