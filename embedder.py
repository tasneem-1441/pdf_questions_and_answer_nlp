"""
Embedder module using sentence-transformers (all-MiniLM-L6-v2) with HuggingFace AutoModel fallback.
Generates normalized dense vector embeddings for document chunks and user queries.
"""

from typing import List, Optional
import numpy as np
import torch
from config import EMBEDDING_MODEL_NAME

logger = None


class Embedder:
    """
    Singleton-style wrapper for SentenceTransformer embedding model.
    Normalizes embeddings to unit length for direct cosine similarity computation via inner product.
    Supports seamless fallback to HuggingFace AutoModel and mean pooling.
    """
    _instance: Optional["Embedder"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._use_st = False
        self.device = "cpu"

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._use_st = True
        except Exception:
            # Fallback to direct HuggingFace AutoTokenizer and AutoModel
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModel.from_pretrained(model_name)
            self.hf_model.to(self.device)
            self.hf_model.eval()

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "Embedder":
        """Returns or initializes the singleton Embedder instance."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of text strings into L2-normalized dense float32 vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Processing batch size.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        if self._use_st:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.astype(np.float32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                model_output = self.hf_model(**encoded)
            
            token_embeddings = model_output[0]
            attention_mask = encoded["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embs.append(normalized.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into a normalized 2D numpy array.

        Args:
            query: The user's question string.

        Returns:
            np.ndarray of shape (1, embedding_dim) with dtype float32.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        if self._use_st:
            query_embedding = self.model.encode(
                [clean_query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return query_embedding.astype(np.float32)

        return self.embed_texts([clean_query], batch_size=1)

    def get_dimension(self) -> int:
        """Returns the embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        if self._use_st:
            return self.model.get_sentence_embedding_dimension()
        return self.hf_model.config.hidden_size

