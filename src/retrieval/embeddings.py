from typing import List

import torch
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """
    Local SentenceTransformer embedder.

    Uses CUDA when available, otherwise CPU.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading embedding model: {model_name}"
        )
        print(
            f"Embedding device: {self.device}"
        )

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
    ):
        """
        Encode document chunks.

        Returns a normalized numpy matrix.
        """

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def encode_query(
        self,
        query: str,
    ):
        """
        Encode a single query.
        """

        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )