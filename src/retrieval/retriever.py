import os
import resource
from typing import Dict, List

from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


# =========================================================
# MEMORY LOGGING
# =========================================================

def _log_mem(label: str):
    """
    Log maximum resident set size.

    Linux reports ru_maxrss in KB.
    Railway runs Linux, so divide by 1024 -> MB.
    """

    rss_mb = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024

    print(
        f"[MEM] {label}: {rss_mb:.0f} MB",
        flush=True,
    )


# =========================================================
# RETRIEVER
# =========================================================

class Retriever:

    def __init__(
        self,
        dense_k: int = 50,
        bm25_k: int = 50,
        final_k: int = 50,
        rerank_k: int = 20,
    ):

        print("=" * 70, flush=True)
        print("RETRIEVER INITIALIZATION START", flush=True)
        print("=" * 70, flush=True)

        _log_mem("before HybridRetriever")

        # -----------------------------------------------------
        # HYBRID RETRIEVER
        # -----------------------------------------------------

        self.hybrid = HybridRetriever(
            dense_k=dense_k,
            bm25_k=bm25_k,
            final_k=final_k,
        )

        _log_mem("after HybridRetriever")

        # -----------------------------------------------------
        # OPTIONAL RERANKER
        # -----------------------------------------------------

        self.reranker = None

        self.enable_reranker = (
            os.getenv(
                "ENABLE_RERANKER",
                "false",
            ).lower()
            == "true"
        )

        print(
            f"Reranker enabled: "
            f"{self.enable_reranker}",
            flush=True,
        )

        if self.enable_reranker:

            print(
                "Loading reranker...",
                flush=True,
            )

            _log_mem(
                "before CrossEncoder"
            )

            self.reranker = Reranker(
                top_k=rerank_k,
            )

            _log_mem(
                "after CrossEncoder"
            )

        else:

            print(
                "Reranker disabled.",
                flush=True,
            )

        print("=" * 70, flush=True)
        print(
            "RETRIEVER INITIALIZATION COMPLETE.",
            flush=True,
        )
        print("=" * 70, flush=True)

        _log_mem(
            "retriever initialization complete"
        )

    # =====================================================
    # RETRIEVE
    # =====================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Dict]:

        print("=" * 70, flush=True)
        print("RETRIEVER REQUEST START", flush=True)
        print(
            f"Query: {query}",
            flush=True,
        )
        print("=" * 70, flush=True)

        _log_mem(
            "before hybrid retrieval"
        )

        # -------------------------------------------------
        # HYBRID RETRIEVAL
        # -------------------------------------------------

        candidates = self.hybrid.retrieve(
            query,
        )

        print(
            f"Hybrid candidates: "
            f"{len(candidates)}",
            flush=True,
        )

        _log_mem(
            "after hybrid retrieval"
        )

        # -------------------------------------------------
        # RERANKING
        # -------------------------------------------------

        if self.reranker is None:

            print(
                "Skipping reranking.",
                flush=True,
            )

            results = candidates

        else:

            print(
                "Starting reranking...",
                flush=True,
            )

            _log_mem(
                "before reranking"
            )

            results = self.reranker.rerank(
                query,
                candidates,
            )

            _log_mem(
                "after reranking"
            )

        # -------------------------------------------------
        # FINAL
        # -------------------------------------------------

        print(
            f"Final retrieved documents: "
            f"{len(results)}",
            flush=True,
        )

        _log_mem(
            "retriever request complete"
        )

        print("=" * 70, flush=True)
        print(
            "RETRIEVER REQUEST COMPLETE",
            flush=True,
        )
        print("=" * 70, flush=True)

        return results