from typing import List, Dict


def evidence_is_sufficient(
    documents: List[Dict],
    minimum_documents: int = 1,
    minimum_score: float = 0.15,
) -> bool:
    """
    Conservative retrieval evidence gate.

    This gate is intentionally simple.

    Final grounding is still performed by citation validation
    after LLM generation.
    """

    if not documents:
        return False

    if len(documents) < minimum_documents:
        return False

    scores = []

    for document in documents:

        score = document.get(
            "rerank_score"
        )

        if score is None:
            continue

        try:
            scores.append(
                float(score)
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    if not scores:
        return False

    return max(scores) >= minimum_score