from typing import List, Dict

from .qdrant_store import QdrantStore
from .bm25_store import BM25Store


class HybridRetriever:
    """
    Hybrid retrieval for the 3GPP RAG system.

    Retrieval sources:
        1. Dense semantic retrieval -> Qdrant
        2. Sparse lexical retrieval -> BM25

    Results are combined using Reciprocal Rank Fusion (RRF).

    The interface intentionally uses `bm25_k` because this is
    the parameter expected by Retriever.
    """

    def __init__(
        self,
        qdrant=None,
        bm25=None,
        dense_k: int = 20,
        bm25_k: int = 20,
        final_k: int = 20,
        rrf_k: int = 60,
    ):
        self.qdrant = (
            qdrant
            if qdrant is not None
            else QdrantStore()
        )

        self.bm25 = (
            bm25
            if bm25 is not None
            else BM25Store()
        )

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_score(self, rank: int) -> float:
        """
        Reciprocal Rank Fusion score.

        RRF(rank) = 1 / (rrf_k + rank)
        """

        return 1.0 / (
            self.rrf_k + rank
        )

    # ------------------------------------------------------------------
    # HYBRID RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_k: int = None,
    ) -> List[Dict]:

        if not query or not query.strip():
            return []

        if final_k is None:
            final_k = self.final_k

        # --------------------------------------------------------------
        # 1. DENSE SEARCH
        # --------------------------------------------------------------

        dense_results = self.qdrant.search(
            query=query,
            limit=self.dense_k,
        )

        # --------------------------------------------------------------
        # 2. BM25 SEARCH
        # --------------------------------------------------------------

        sparse_results = self.bm25.search(
            query=query,
            limit=self.bm25_k,
        )

        # --------------------------------------------------------------
        # 3. RECIPROCAL RANK FUSION
        # --------------------------------------------------------------

        fused = {}

        # Dense results
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            doc_id = str(result["id"])

            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                }

            fused[doc_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

        # BM25 results
        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            doc_id = str(result["id"])

            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0,
                }
            else:
                fused[doc_id]["bm25_rank"] = rank

            fused[doc_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

        # --------------------------------------------------------------
        # 4. SORT BY FUSED SCORE
        # --------------------------------------------------------------

        results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # --------------------------------------------------------------
        # 5. TOP K
        # --------------------------------------------------------------

        return results[:final_k]