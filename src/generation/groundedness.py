from __future__ import annotations


ABSTENTION_PHRASES = (
    "insufficient_evidence",
    "insufficient evidence",
    "not enough evidence",
    "cannot be determined from the supplied documents",
    "not specified in the supplied documents",
)


def is_abstention(
    answer: str,
) -> bool:

    text = (
        answer or ""
    ).strip().lower()

    return any(
        phrase in text
        for phrase in ABSTENTION_PHRASES
    )