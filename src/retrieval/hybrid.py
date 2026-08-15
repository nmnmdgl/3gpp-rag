from collections import defaultdict


class HybridRetriever:
    def __init__(
        self,
        dense_store,
        bm25_store,
        rrf_k=60,
    ):
        self.dense = dense_store
        self.bm25 = bm25_store
        self.rrf_k = rrf_k

    def search(
        self,
        query,
        dense_k=12,
        bm25_k=12,
        final_k=8,
    ):
        dense_results = self.dense.search(
            query,
            dense_k,
        )

        lexical_results = self.bm25.search(
            query,
            bm25_k,
        )

        fused_scores = defaultdict(float)
        documents = {}

        for rank, item in enumerate(
            dense_results,
            start=1,
        ):
            fused_scores[item["id"]] += (
                1 / (self.rrf_k + rank)
            )
            documents[item["id"]] = item

        for rank, item in enumerate(
            lexical_results,
            start=1,
        ):
            fused_scores[item["id"]] += (
                1 / (self.rrf_k + rank)
            )
            documents[item["id"]] = item

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        output = []

        for doc_id in ranked_ids[:final_k]:
            result = dict(documents[doc_id])
            result["hybrid_score"] = (
                fused_scores[doc_id]
            )
            output.append(result)

        return output
