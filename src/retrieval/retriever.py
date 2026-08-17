from typing import List, Dict


class Retriever:

    def __init__(
        self,
        dense_k: int = 12,
        bm25_k: int = 12,
        final_k: int = 8,
        rerank_k: int = 5,
    ):

        print("=" * 70)
        print("RETRIEVER INITIALIZATION START")
        print("=" * 70)

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rerank_k = rerank_k

        # ---------------------------------------------------------
        # HYBRID RETRIEVER
        # ---------------------------------------------------------

        print("=" * 70)
        print("STEP 1: Initializing HybridRetriever")
        print("=" * 70)

        from .hybrid_retriever import HybridRetriever

        self.hybrid = HybridRetriever(
            dense_k=dense_k,
            bm25_k=bm25_k,
            final_k=final_k,
        )

        print("=" * 70)
        print("STEP 1 COMPLETE: HybridRetriever initialized")
        print("=" * 70)

        # ---------------------------------------------------------
        # RERANKER
        # ---------------------------------------------------------

        print("=" * 70)
        print("STEP 2: Initializing reranker")
        print("=" * 70)

        try:

            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            print("=" * 70)
            print("STEP 2 COMPLETE: Reranker initialized")
            print("=" * 70)

        except Exception as exc:

            print("=" * 70)
            print("RERANKER INITIALIZATION FAILED")
            print(
                f"{type(exc).__name__}: {exc}"
            )
            print("=" * 70)

            raise

        print("=" * 70)
        print("RETRIEVER INITIALIZATION COMPLETE")
        print("=" * 70)

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Dict]:

        print("=" * 70)
        print("RETRIEVER QUERY")
        print(f"Query: {query}")
        print("=" * 70)

        print("STEP 3: Running hybrid retrieval")

        documents = self.hybrid.retrieve(
            query
        )

        print(
            f"Hybrid retrieval returned "
            f"{len(documents)} documents"
        )

        if not documents:
            return []

        # ---------------------------------------------------------
        # RERANK
        # ---------------------------------------------------------

        rerank_documents = documents[
            : self.rerank_k
        ]

        print(
            f"STEP 4: Reranking "
            f"{len(rerank_documents)} documents"
        )

        pairs = [
            [
                query,
                document.get(
                    "text",
                    "",
                ),
            ]
            for document in rerank_documents
        ]

        scores = self.reranker.predict(
            pairs
        )

        ranked = sorted(
            zip(
                rerank_documents,
                scores,
            ),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = [
            document
            for document, _score in ranked
        ]

        results = results[
            : self.final_k
        ]

        print(
            f"Final retrieved documents: "
            f"{len(results)}"
        )

        return results