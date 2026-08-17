import resource

from typing import List, Dict

from .qdrant_store import QdrantStore
from .bm25_store import BM25Store


# =========================================================
# MEMORY LOGGING
# =========================================================

def _log_mem(label: str):
    """
    Linux/Railway memory logging.

    ru_maxrss is reported in KB on Linux.
    """

    rss_mb = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024

    print(
        f"[MEM] {label}: {rss_mb:.0f} MB",
        flush=True,
    )


# =========================================================
# HYBRID RETRIEVER
# =========================================================

class HybridRetriever:
    """
    Hybrid retrieval for the 3GPP RAG system.

    Retrieval sources:
        1. Dense semantic retrieval -> Qdrant
        2. Sparse lexical retrieval -> BM25

    Results are combined using Reciprocal Rank Fusion.
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

        print(
            "HYBRID RETRIEVER INITIALIZATION",
            flush=True,
        )

        _log_mem(
            "before QdrantStore"
        )

        # -------------------------------------------------
        # QDRANT
        # -------------------------------------------------

        self.qdrant = (
            qdrant
            if qdrant is not None
            else QdrantStore()
        )

        _log_mem(
            "after QdrantStore"
        )

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------

        print(
            "Initializing BM25 store...",
            flush=True,
        )

        self.bm25 = (
            bm25
            if bm25 is not None
            else BM25Store()
        )

        _log_mem(
            "after BM25Store"
        )

        # -------------------------------------------------
        # CONFIG
        # -------------------------------------------------

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

        print(
            "HybridRetriever initialized.",
            flush=True,
        )

        _log_mem(
            "HybridRetriever complete"
        )

    # =====================================================
    # RRF
    # =====================================================

    def _rrf_score(
        self,
        rank: int,
    ) -> float:

        return 1.0 / (
            self.rrf_k + rank
        )

    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        query: str,
        final_k: int = None,
    ) -> List[Dict]:

        if not query or not query.strip():
            return []

        if final_k is None:
            final_k = self.final_k

        print(
            "HYBRID RETRIEVAL START",
            flush=True,
        )

        _log_mem(
            "before dense search"
        )

        # -------------------------------------------------
        # 1. DENSE SEARCH
        # -------------------------------------------------

        print(
            f"Dense search: top {self.dense_k}",
            flush=True,
        )

        dense_results = self.qdrant.search(
            query=query,
            limit=self.dense_k,
        )

        print(
            f"Dense results: "
            f"{len(dense_results)}",
            flush=True,
        )

        _log_mem(
            "after dense search"
        )

        # -------------------------------------------------
        # 2. BM25 SEARCH
        # -------------------------------------------------

        print(
            f"BM25 search: top {self.bm25_k}",
            flush=True,
        )

        sparse_results = self.bm25.search(
            query=query,
            limit=self.bm25_k,
        )

        print(
            f"BM25 results: "
            f"{len(sparse_results)}",
            flush=True,
        )

        _log_mem(
            "after BM25 search"
        )

        # -------------------------------------------------
        # 3. RECIPROCAL RANK FUSION
        # -------------------------------------------------

        print(
            "Performing RRF...",
            flush=True,
        )

        fused = {}

        # -------------------------------------------------
        # DENSE RESULTS
        # -------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            doc_id = str(
                result["id"]
            )

            if doc_id not in fused:

                fused[doc_id] = {
                    **result,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                }

            fused[doc_id][
                "rrf_score"
            ] += self._rrf_score(rank)

        # -------------------------------------------------
        # BM25 RESULTS
        # -------------------------------------------------

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):

            doc_id = str(
                result["id"]
            )

            if doc_id not in fused:

                fused[doc_id] = {
                    **result,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0,
                }

            else:

                fused[doc_id][
                    "bm25_rank"
                ] = rank

            fused[doc_id][
                "rrf_score"
            ] += self._rrf_score(rank)

        _log_mem(
            "after RRF"
        )

        # -------------------------------------------------
        # 4. SORT
        # -------------------------------------------------

        results = sorted(
            fused.values(),
            key=lambda x: x[
                "rrf_score"
            ],
            reverse=True,
        )

        # -------------------------------------------------
        # 5. TOP K
        # -------------------------------------------------

        results = results[:final_k]

        print(
            f"Hybrid final results: "
            f"{len(results)}",
            flush=True,
        )

        _log_mem(
            "hybrid retrieval complete"
        )

        print(
            "HYBRID RETRIEVAL COMPLETE",
            flush=True,
        )

        return results