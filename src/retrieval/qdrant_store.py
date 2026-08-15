from pathlib import Path
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from .embeddings import LocalEmbedder


load_dotenv()


class QdrantStore:
    """
    Local persistent Qdrant vector store.

    Qdrant runs entirely locally using on-disk storage.
    No Docker or external Qdrant server is required.
    """

    def __init__(self):
        self.collection = os.getenv(
            "QDRANT_COLLECTION",
            "3gpp_specs",
        )

        self.storage_path = Path(
            os.getenv(
                "QDRANT_PATH",
                "data/qdrant",
            )
        )

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(self.storage_path)
        )

        self.embedder = LocalEmbedder(
            os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-base-en-v1.5",
            )
        )

    # ------------------------------------------------------------------
    # COLLECTION
    # ------------------------------------------------------------------

    def recreate_collection(self, vector_size: int):
        """
        Delete and recreate the collection.
        Used during a complete index rebuild.
        """

        if self.client.collection_exists(
            collection_name=self.collection
        ):
            self.client.delete_collection(
                collection_name=self.collection
            )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def index(
        self,
        chunks,
        embeddings=None,
        batch_size: int = 256,
    ):
        """
        Index chunks into Qdrant.

        If embeddings are supplied, they are reused instead
        of generating them again.

        This is important because embeddings are generated
        separately on Colab and copied to the local machine.
        """

        if not chunks:
            raise ValueError(
                "No chunks were provided for indexing."
            )

        # --------------------------------------------------------------
        # USE PRECOMPUTED EMBEDDINGS
        # --------------------------------------------------------------

        if embeddings is None:

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            print(
                f"Generating embeddings for "
                f"{len(texts)} chunks..."
            )

            embeddings = self.embedder.encode_documents(
                texts,
                batch_size=batch_size,
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding/chunk mismatch: "
                f"{len(embeddings)} embeddings for "
                f"{len(chunks)} chunks."
            )

        vector_size = embeddings.shape[1]

        print(
            f"Chunks: {len(chunks)}"
        )

        print(
            f"Embedding shape: {embeddings.shape}"
        )

        print(
            f"Vector dimension: {vector_size}"
        )

        # --------------------------------------------------------------
        # CREATE COLLECTION
        # --------------------------------------------------------------

        self.recreate_collection(
            vector_size=vector_size
        )

        print(
            f"\nStoring {len(chunks)} vectors "
            f"in local Qdrant..."
        )

        # --------------------------------------------------------------
        # UPSERT
        # --------------------------------------------------------------

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(chunks),
            )

            points = []

            for absolute_index in range(
                start,
                end,
            ):
                points.append(
                    PointStruct(
                        id=absolute_index,
                        vector=embeddings[
                            absolute_index
                        ].tolist(),
                        payload=chunks[
                            absolute_index
                        ],
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )

            print(
                f"Indexed {end}/{len(chunks)}"
            )

        print(
            "\nQdrant indexing completed."
        )

        print(
            f"Qdrant path: "
            f"{self.storage_path.resolve()}"
        )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 12,
    ):
        """
        Perform semantic vector search.

        IMPORTANT:
        The parameter is intentionally named `query`
        because HybridRetriever calls this method using
        query=query.
        """

        if not query or not query.strip():
            return []

        if not self.client.collection_exists(
            collection_name=self.collection
        ):
            raise RuntimeError(
                f"Qdrant collection "
                f"'{self.collection}' does not exist."
            )

        vector = self.embedder.encode_query(
            query
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points

        output = []

        for result in results:

            payload = result.payload or {}

            output.append(
                {
                    "id": str(result.id),
                    "score": float(result.score),
                    **payload,
                }
            )

        return output

    # ------------------------------------------------------------------
    # COLLECTION INFO
    # ------------------------------------------------------------------

    def collection_info(self):
        """
        Return information about the current collection.
        """

        if not self.client.collection_exists(
            collection_name=self.collection
        ):
            return None

        return self.client.get_collection(
            collection_name=self.collection
        )

    # ------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------

    def check(self):
        """
        Basic Qdrant index health check.
        """

        info = self.collection_info()

        if info is None:
            return {
                "collection": self.collection,
                "exists": False,
            }

        return {
            "collection": self.collection,
            "exists": True,
            "vectors": info.points_count,
            "status": str(info.status),
            "vector_config": info.config.params.vectors,
            "path": str(
                self.storage_path.resolve()
            ),
        }