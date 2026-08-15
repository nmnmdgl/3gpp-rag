from typing import Dict, List

from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


class Retriever:
    """
    Hybrid retrieval pipeline.

    Combines:
        - Dense retrieval
        - BM25 retrieval
        - Reranking

    LLM generation does NOT belong here.
    """

    def __init__(
        self,
        dense_k: int = 50,
        bm25_k: int = 50,
        final_k: int = 50,
        rerank_k: int = 20,
    ):
        self.hybrid = HybridRetriever(
            dense_k=dense_k,
            bm25_k=bm25_k,
            final_k=final_k,
        )

        self.reranker = Reranker(
            top_k=rerank_k,
        )

    def retrieve(
        self,
        query: str,
    ) -> List[Dict]:

        candidates = self.hybrid.retrieve(
            query,
        )

        reranked = self.reranker.rerank(
            query,
            candidates,
        )

        return reranked