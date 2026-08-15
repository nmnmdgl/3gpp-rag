from sentence_transformers import SentenceTransformer
import numpy as np


class QueryEmbedder:
    """
    Runtime query embedder.

    Uses the SAME embedding model that was used to create
    the document embeddings.

    Document embeddings were generated offline on Colab.
    Query embeddings are generated locally at runtime.
    """

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5",
    ):
        print(
            f"Loading query embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            f"Query embedding device: "
            f"{self.model.device}"
        )

    def encode(self, query: str) -> np.ndarray:
        """
        Convert a user query into a normalized
        768-dimensional embedding.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                f"Expected 1D query embedding, "
                f"got shape {vector.shape}"
            )

        if np.isnan(vector).any():
            raise ValueError(
                "Query embedding contains NaN."
            )

        if np.isinf(vector).any():
            raise ValueError(
                "Query embedding contains Inf."
            )

        return vector