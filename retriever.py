"""
Vector search, BM25 hybrid document retrieval, and page rendering module.
Features:
- Hybrid retrieval combining dense cosine similarity with BM25 keyword scoring
- Preserves semantic similarity-threshold refusal logic
- Source PDF page image preview rendering using PyMuPDF (fitz)
- Salient key terms extraction with persistent SHA-256 caching
- Persistent disk caching per PDF for index, metadata, and key terms
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from config import (
    CACHE_DIR,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_THRESHOLD,
    PAGES_PER_SUMMARY_GROUP,
    USE_HYBRID_SEARCH,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    TOP_KEY_TERMS_COUNT
)
from loader import load_pdf
from chunker import split_text_into_chunks
from embedder import Embedder

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Calculates SHA-256 checksum of a file to provide a deterministic cache key."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


class FAISSRetriever:
    """
    Manages vector index creation, BM25 hybrid search, page previews, and persistent disk caching.
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder.get_instance()
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[Any] = None
        self.pdf_hash: Optional[str] = None
        self.pdf_file_path: Optional[str] = None
        self.is_cached: bool = False
        self.source_filename: str = ""

    def get_cache_paths(self, pdf_hash: str) -> Tuple[Path, Path, Path]:
        """Returns the file paths for the FAISS index, numpy matrix, and chunk metadata in CACHE_DIR."""
        model_name = getattr(self.embedder, "model_name", "default")
        model_tag = re.sub(r"[^\w\-]", "_", model_name)
        index_path = CACHE_DIR / f"{pdf_hash}_{model_tag}.faiss"
        npy_path = CACHE_DIR / f"{pdf_hash}_{model_tag}.npy"
        meta_path = CACHE_DIR / f"{pdf_hash}_{model_tag}_meta.json"
        return index_path, npy_path, meta_path

    def _init_bm25(self) -> None:
        """Builds BM25 index over the current document chunks."""
        if not self.chunks or BM25Okapi is None:
            self.bm25 = None
            return
        tokenized_corpus = [
            re.findall(r"\b\w+\b", c["text"].lower())
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def index_pdf(self, pdf_path: str | Path, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Indexes a PDF document. If the document has already been processed and cached,
        the index and chunk metadata are instantly reloaded from disk.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {path}")

        self.source_filename = path.name
        self.pdf_file_path = str(path.resolve())
        self.pdf_hash = compute_file_sha256(path)
        index_path, npy_path, meta_path = self.get_cache_paths(self.pdf_hash)

        # Check disk cache
        has_cached_index = (faiss is not None and index_path.exists()) or npy_path.exists()
        if not force_reindex and has_cached_index and meta_path.exists():
            logger.info("Loading cached index and chunks for %s...", path.name)
            self._load_from_cache(index_path, npy_path, meta_path)
            self.is_cached = True
            self._init_bm25()

            unique_pages = len(set(c["page_number"] for c in self.chunks))
            return {
                "pdf_name": self.source_filename,
                "pdf_hash": self.pdf_hash,
                "total_pages": unique_pages,
                "total_chunks": len(self.chunks),
                "is_cached": True
            }

        # Cache miss: load, chunk, embed, and index
        logger.info("Cache miss. Extracting and indexing %s...", path.name)
        pages_data = load_pdf(path)
        chunks = split_text_into_chunks(
            pages_data,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        if not chunks:
            raise ValueError(f"No textual chunks could be extracted from '{path.name}'.")

        # Generate normalized embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # Always save npy backup for zero-dependency portability
        np.save(str(npy_path), embeddings)

        # Build FAISS IndexFlatIP or fallback
        dimension = embeddings.shape[1]
        if faiss is not None:
            try:
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                self.index = index
                faiss.write_index(self.index, str(index_path))
            except Exception as e:
                logger.warning("FAISS native indexing failed (%s), using numpy fallback", e)
                self.index = embeddings
        else:
            self.index = embeddings

        self.chunks = chunks
        self.is_cached = False
        self._init_bm25()

        # Save metadata to cache
        meta_payload = {
            "source_filename": self.source_filename,
            "pdf_file_path": self.pdf_file_path,
            "pdf_hash": self.pdf_hash,
            "chunks": self.chunks
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        return {
            "pdf_name": self.source_filename,
            "pdf_hash": self.pdf_hash,
            "total_pages": len(pages_data),
            "total_chunks": len(self.chunks),
            "is_cached": False
        }

    def _load_from_cache(self, index_path: Path, npy_path: Path, meta_path: Path) -> None:
        """Loads FAISS index or numpy embeddings and chunk metadata from cached disk files."""
        self.index = None
        if faiss is not None and index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
            except Exception as e:
                logger.warning("Failed to load FAISS index (%s), trying npy", e)
                self.index = None

        if self.index is None and npy_path.exists():
            self.index = np.load(str(npy_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)
        self.chunks = meta_payload["chunks"]
        self.source_filename = meta_payload.get("source_filename", "")
        if "pdf_file_path" in meta_payload and Path(meta_payload["pdf_file_path"]).exists():
            self.pdf_file_path = meta_payload["pdf_file_path"]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
        use_hybrid: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using Hybrid Search (Dense Semantic + BM25).
        Preserves strict semantic similarity thresholding to reject out-of-scope queries.

        Args:
            query: User's question string.
            top_k: Number of chunks to retrieve (default: TOP_K).
            similarity_threshold: Minimum semantic cosine similarity score required.
            use_hybrid: Override flag for hybrid retrieval (defaults to USE_HYBRID_SEARCH).

        Returns:
            List of chunk dicts sorted by score.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Retriever has no indexed document. Please index a PDF first.")

        query_clean = query.strip()
        if not query_clean:
            return []

        hybrid_enabled = USE_HYBRID_SEARCH if use_hybrid is None else use_hybrid

        # 1. Dense Semantic Cosine Similarity
        query_vec = self.embedder.embed_query(query_clean)

        if faiss is not None and isinstance(self.index, faiss.Index):
            # FAISS inner product on unit vectors == cosine similarity
            actual_k = min(len(self.chunks), max(top_k * 3, 20))
            raw_scores, raw_indices = self.index.search(query_vec, actual_k)
            # Map back to full-corpus array for consistent hybrid blending
            sims = np.zeros(len(self.chunks), dtype=np.float32)
            for idx, sc in zip(raw_indices[0], raw_scores[0]):
                if 0 <= idx < len(self.chunks):
                    sims[idx] = float(sc)
        else:
            # Direct NumPy matrix multiplication
            sims = np.dot(self.index, query_vec.T).squeeze()
            if np.ndim(sims) == 0:
                sims = np.array([float(sims)], dtype=np.float32)

        # 2. BM25 Keyword Scoring (if hybrid enabled)
        if hybrid_enabled and self.bm25 is not None:
            query_tokens = re.findall(r"\b\w+\b", query_clean.lower())
            bm25_raw = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
            b_min, b_max = bm25_raw.min(), bm25_raw.max()
            if b_max > b_min:
                bm25_norm = (bm25_raw - b_min) / (b_max - b_min)
            else:
                bm25_norm = np.zeros_like(bm25_raw)

            # Combined hybrid score
            final_scores = (SEMANTIC_WEIGHT * sims) + (BM25_WEIGHT * bm25_norm)
        else:
            final_scores = sims

        # 3. Rank candidates by final score
        ranked_indices = np.argsort(final_scores)[::-1]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            if idx >= len(self.chunks):
                continue

            semantic_score = float(sims[idx])

            # CRITICAL REFUSAL LOGIC: Enforce similarity_threshold on SEMANTIC score
            # to guarantee that purely keyword-matching out-of-scope questions are rejected.
            if similarity_threshold is not None and semantic_score < similarity_threshold:
                continue

            chunk_info = self.chunks[idx].copy()
            chunk_info["similarity_score"] = semantic_score
            chunk_info["hybrid_score"] = float(final_scores[idx])
            results.append(chunk_info)

            if len(results) >= top_k:
                break

        return results

    def extract_key_terms(self, count: int = TOP_KEY_TERMS_COUNT) -> List[str]:
        """
        Extracts salient key terms and named entities across document chunks.
        Caches results per PDF hash for instantaneous loading.
        """
        if not self.chunks:
            return []

        cache_path = (CACHE_DIR / f"{self.pdf_hash}_key_terms.json") if self.pdf_hash else None
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "are", "which", "have", "has",
            "will", "each", "been", "also", "their", "they", "such", "into", "more", "than",
            "about", "were", "when", "what", "where", "these", "those", "can", "may", "page",
            "unit", "chapter", "section", "notes", "document", "using", "used", "some", "other"
        }
        word_freq: Dict[str, int] = {}
        bigram_freq: Dict[str, int] = {}

        for c in self.chunks:
            words = [
                w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", c["text"])
                if w.lower() not in stopwords
            ]
            for w in words:
                cap_w = w.capitalize()
                word_freq[cap_w] = word_freq.get(cap_w, 0) + 1
            for i in range(len(words) - 1):
                bg = f"{words[i].capitalize()} {words[i+1].capitalize()}"
                bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

        # Multi-word salient phrases prioritized
        ranked_bigrams = sorted(bigram_freq.items(), key=lambda x: (x[1] >= 2, x[1]), reverse=True)
        terms = [t for t, freq in ranked_bigrams if freq >= 2]

        if len(terms) < count:
            single_ranked = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for w, freq in single_ranked:
                if w not in terms and not any(w in t for t in terms) and len(w) > 3 and freq >= 2:
                    terms.append(w)
                if len(terms) >= count:
                    break

        terms = terms[:count]
        if cache_path and terms:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return terms

    def render_page_preview(self, page_number: int, highlight_text: Optional[str] = None) -> Optional[str]:
        """
        Renders the specified 1-indexed PDF page as a high-resolution PNG image using PyMuPDF.
        Optionally finds the retrieved chunk's text on that page and draws a yellow highlight over matches.
        """
        if not self.pdf_file_path or not Path(self.pdf_file_path).exists():
            return None

        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(self.pdf_file_path)
            if 1 <= page_number <= len(doc):
                page = doc[page_number - 1]

                # Feature 1: Evidence highlighting
                if highlight_text and highlight_text.strip():
                    clean_chunk = highlight_text.strip()
                    sentences = re.split(r"(?<=[.!?\n])\s+", clean_chunk)
                    for s in sentences:
                        s_clean = s.strip().strip('"\'')
                        if len(s_clean) >= 12:
                            rects = page.search_for(s_clean[:70])
                            if not rects:
                                words = s_clean.split()
                                if len(words) >= 4:
                                    rects = page.search_for(" ".join(words[:5]))
                            for r in rects:
                                try:
                                    annot = page.add_highlight_annot(r)
                                    annot.set_colors(stroke=[1.0, 0.9, 0.2])  # Yellow
                                    annot.update()
                                except Exception:
                                    pass

                pix = page.get_pixmap(dpi=140)
                cache_suffix = f"_p{page_number}_hl.png" if highlight_text else f"_page_{page_number}.png"
                out_path = CACHE_DIR / f"{self.pdf_hash}{cache_suffix}" if self.pdf_hash else CACHE_DIR / f"page_{page_number}.png"
                pix.save(str(out_path))
                return str(out_path)
        except Exception as e:
            logger.warning("Error rendering page preview for page %s: %s", page_number, e)

        return None

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Returns all chunks of the currently indexed document."""
        return self.chunks

    def get_representative_chunks(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Picks count representative chunks spread evenly across the entire document
        (including first, last, and evenly spaced chunks in between).
        """
        if not self.chunks:
            return []
        if len(self.chunks) <= count:
            return list(self.chunks)

        indices = [int(round(i * (len(self.chunks) - 1) / (count - 1))) for i in range(count)]
        seen = set()
        selected = []
        for idx in indices:
            if idx not in seen and 0 <= idx < len(self.chunks):
                seen.add(idx)
                selected.append(self.chunks[idx])
        return selected

    def get_chunks_grouped_by_page_range(self, pages_per_group: int = PAGES_PER_SUMMARY_GROUP) -> List[Dict[str, Any]]:
        """
        Groups chunks by contiguous page ranges (e.g. pp. 1-6, pp. 7-12) for MAP summarization.
        """
        if not self.chunks:
            return []

        all_pages = sorted(list(set(c["page_number"] for c in self.chunks if "page_number" in c)))
        if not all_pages:
            all_pages = [1]

        min_p = all_pages[0]
        max_p = all_pages[-1]

        groups = []
        cur_start = min_p
        while cur_start <= max_p:
            cur_end = min(cur_start + pages_per_group - 1, max_p)
            group_chunks = [
                c for c in self.chunks
                if cur_start <= c.get("page_number", 1) <= cur_end
            ]
            if group_chunks:
                text_content = "\n\n".join(c["text"] for c in group_chunks)
                label = f"pp. {cur_start}-{cur_end}" if cur_start != cur_end else f"p. {cur_start}"
                groups.append({
                    "page_range": label,
                    "start_page": cur_start,
                    "end_page": cur_end,
                    "text": text_content,
                    "chunk_count": len(group_chunks)
                })
            cur_start = cur_end + 1

        return groups

    def get_document_headings(self) -> List[str]:
        """Extracts prominent headings or titles from the chunks."""
        headings = []
        seen = set()
        for chunk in self.chunks:
            lines = chunk.get("text", "").split("\n")
            for line in lines[:3]:
                clean = line.strip().strip("#*-_ ")
                if 4 <= len(clean) <= 60 and clean.lower() not in seen:
                    if clean.isupper() or clean.istitle() or any(clean.startswith(p) for p in ["Chapter", "Unit", "Section", "Part"]):
                        seen.add(clean.lower())
                        headings.append(clean)
                        if len(headings) >= 12:
                            return headings
        return headings
