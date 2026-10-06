"""
PDF Loader module using pdfplumber to extract text and track page numbers.
Handles error cases such as corrupt files, empty PDFs, and scanned PDFs with no OCR text.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber


class PDFLoadError(Exception):
    """Base exception for PDF loading errors."""
    pass


class EmptyPDFError(PDFLoadError):
    """Raised when the PDF has 0 pages or contains no pages with content."""
    pass


class ScannedPDFError(PDFLoadError):
    """Raised when the PDF appears to be scanned or contains no extractable text."""
    pass


class EncryptedPDFError(PDFLoadError):
    """Raised when the PDF is password-protected or encrypted."""
    pass


def normalize_page_text(raw_text: str) -> str:
    """
    Cleans up extracted page text:
    - Normalizes non-breaking spaces and exotic whitespace.
    - Collapses excessive blank lines (>2) into standard double newlines.
    - Preserves logical sentence and paragraph layout.
    """
    if not raw_text:
        return ""
    # Normalize non-breaking and special whitespaces
    text = raw_text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def load_pdf(pdf_path: str | Path, password: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF file using pdfplumber,
    retaining 1-indexed page numbers.

    Args:
        pdf_path: Path to the target PDF file.
        password: Optional password for protected PDFs.

    Returns:
        List of dicts: [{"text": str, "page_number": int}, ...]

    Raises:
        FileNotFoundError: If the file path does not exist.
        EncryptedPDFError: If the PDF is password-protected and cannot be opened.
        EmptyPDFError: If the PDF contains no pages.
        ScannedPDFError: If no extractable text is found across any pages.
        PDFLoadError: If any unhandled extraction failure occurs.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    pages_data: List[Dict[str, Any]] = []

    try:
        open_kwargs = {"password": password} if password else {}
        with pdfplumber.open(str(path), **open_kwargs) as pdf:
            if not pdf.pages:
                raise EmptyPDFError(f"PDF '{path.name}' contains no pages.")

            total_extracted_chars = 0

            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    raw_text = page.extract_text()
                except Exception as page_exc:
                    # Ignore minor page-level font errors, continue with other pages
                    raw_text = None

                if raw_text:
                    cleaned_text = normalize_page_text(raw_text)
                    if cleaned_text:
                        pages_data.append({
                            "text": cleaned_text,
                            "page_number": page_idx
                        })
                        total_extracted_chars += len(cleaned_text)

            if not pages_data or total_extracted_chars == 0:
                raise ScannedPDFError(
                    f"PDF '{path.name}' contains no extractable text. "
                    "The document may be scanned, image-only, or rendered as outlines."
                )

    except (EmptyPDFError, ScannedPDFError, EncryptedPDFError):
        raise
    except Exception as exc:
        exc_lower = str(exc).lower()
        if "password" in exc_lower or "encrypt" in exc_lower:
            raise EncryptedPDFError(f"PDF '{path.name}' is password-protected or encrypted.") from exc
        raise PDFLoadError(f"Failed to read PDF '{path.name}': {str(exc)}") from exc

    return pages_data


def get_pdf_metadata(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Quickly inspects a PDF file and returns document metadata including
    page count, file size in KB, and document information.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return {}

    try:
        with pdfplumber.open(str(path)) as pdf:
            meta = pdf.metadata or {}
            return {
                "filename": path.name,
                "file_size_kb": round(path.stat().st_size / 1024, 1),
                "total_pages": len(pdf.pages),
                "title": meta.get("Title") or meta.get("title") or path.stem,
                "author": meta.get("Author") or meta.get("author") or "Unknown",
                "creator": meta.get("Creator") or meta.get("creator") or "Unknown"
            }
    except Exception:
        return {
            "filename": path.name,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "total_pages": 0,
            "title": path.stem,
            "author": "Unknown",
            "creator": "Unknown"
        }

