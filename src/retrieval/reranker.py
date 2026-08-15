from typing import Dict, List

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int = 20,
    ):
        self.model_name = (
            model_name
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.top_k = top_k

        print(
            f"Loading reranker: "
            f"{self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
    ) -> List[Dict]:

        if not documents:
            return []

        pairs = [
            (
                query,
                document.get("text", ""),
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = []

        for document, score in zip(
            documents,
            scores,
        ):
            item = dict(document)

            item["rerank_score"] = float(
                score
            )

            ranked.append(item)

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return ranked[:self.top_k]