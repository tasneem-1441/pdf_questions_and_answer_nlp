"""
Text chunking module for PDF Question Answering.
Splits extracted document text into ~500-character chunks with 50-character overlap,
preserving page numbers and sentence boundaries where possible.
"""

from typing import List, Dict, Any
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_into_chunks(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page text into overlapping chunks, tracking the source page number.

    Args:
        pages_data: List of dicts, each with keys 'text' and 'page_number'.
        chunk_size: Target size in characters for each chunk (default ~500).
        chunk_overlap: Number of characters to overlap between sequential chunks (default ~50).

    Returns:
        List of dicts:
        [
            {
                "chunk_id": int,
                "text": str,
                "page_number": int,
                "char_count": int
            },
            ...
        ]
    """
    if chunk_size <= chunk_overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap}).")

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    step = chunk_size - chunk_overlap

    for page_item in pages_data:
        text = page_item["text"]
        page_number = page_item["page_number"]

        # If page text is very short, keep it as a single chunk
        if len(text) <= chunk_size:
            clean_chunk = text.strip()
            if clean_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": clean_chunk,
                    "page_number": page_number,
                    "char_count": len(clean_chunk)
                })
                chunk_id += 1
            continue

        # Sliding window chunking with natural boundary preference
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + chunk_size, total_len)

            # Try to break at a natural sentence boundary (., !, ?, \n) or whitespace near the end
            if end < total_len:
                # Look for sentence boundary in the last 60 chars of the window
                lookback_window = text[max(start, end - 60):end]
                match = re.search(r'([.!?\n]\s+)', lookback_window)
                if match:
                    # Adjust end to the end of sentence
                    offset = max(start, end - 60) + match.end()
                    if offset > start + chunk_overlap:
                        end = offset
                else:
                    # Otherwise look for a whitespace to avoid cutting a word in half
                    space_idx = text.rfind(" ", max(start, end - 30), end)
                    if space_idx != -1 and space_idx > start + chunk_overlap:
                        end = space_idx + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= 15:  # Filter out trivial fragments
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page_number": page_number,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1

            if end >= total_len:
                break

            # Advance by step, but ensure we advance at least by 1 character
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = start + step
            start = next_start

    return chunks
