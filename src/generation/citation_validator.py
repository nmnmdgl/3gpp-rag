# src/generation/citation_validator.py

from __future__ import annotations

import re
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Citation format
# ---------------------------------------------------------------------------

CITATION_RE = re.compile(
    r"\[\s*"
    r"(TR|TS)"
    r"\s*"
    r"([0-9]{2}\.[0-9]{3})"
    r"\s*\|\s*"
    r"Clause\s*"
    r"([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)"
    r"\s*\]",
    re.IGNORECASE,
)


# Detect malformed citation-like constructs.
CITATION_LIKE_RE = re.compile(
    r"\[\s*(?:TR|TS|TR\s*|TS\s*)"
    r"[^\]]{0,100}"
    r"\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _norm_spec(value: Any) -> str:
    if not value:
        return ""

    s = str(value).upper().strip()

    if s.startswith(("TS ", "TR ")):
        return s

    if s.startswith(("TS", "TR")) and len(s) > 2:
        return s[:2] + " " + s[2:].strip()

    return s


def _norm_clause(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def extract_citations(
    answer: str,
) -> list[tuple[str, str]]:

    return [
        (
            f"{match.group(1).upper()} {match.group(2)}",
            match.group(3),
        )
        for match in CITATION_RE.finditer(
            answer or ""
        )
    ]


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _evidence_keys(
    documents: Iterable[dict],
) -> set[tuple[str, str]]:

    keys: set[tuple[str, str]] = set()

    for doc in documents or []:

        spec = _norm_spec(
            doc.get("spec_number")
            or doc.get("spec")
            or doc.get("document")
            or doc.get("document_id")
        )

        clause = _norm_clause(
            doc.get("clause")
        )

        if spec and clause:
            keys.add(
                (
                    spec,
                    clause,
                )
            )

    return keys


# ---------------------------------------------------------------------------
# Citation proximity
# ---------------------------------------------------------------------------

def _citation_positions(
    answer: str,
) -> list[tuple[int, int]]:

    return [
        match.span()
        for match in CITATION_RE.finditer(
            answer or ""
        )
    ]


def _has_nearby_citation(
    sentence: str,
) -> bool:

    return bool(
        CITATION_RE.search(
            sentence
        )
    )


# ---------------------------------------------------------------------------
# Sentence analysis
# ---------------------------------------------------------------------------

def _split_sentences(
    answer: str,
) -> list[str]:

    if not answer:
        return []

    # Preserve bullets as independent units.
    normalized = re.sub(
        r"\n+",
        "\n",
        answer.strip(),
    )

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        normalized,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def _looks_factual(
    sentence: str,
) -> bool:

    text = sentence.strip()

    if not text:
        return False

    # Pure headings are not factual claims.
    if (
        text.startswith("#")
        or text.startswith("**")
        or text.endswith(":")
    ):
        return False

    # Citation-only fragments are not claims.
    if CITATION_RE.fullmatch(text):
        return False

    # Common non-factual transition phrases.
    lowered = text.lower()

    if lowered in {
        "insufficient_evidence",
        "answer:",
        "summary:",
        "therefore:",
    }:
        return False

    # Technical/factual language indicators.
    factual_patterns = [
        r"\b(is|are|was|were|has|have|contains|provides|supports)\b",
        r"\b(uses|used|performs|provides|selects|stores|handles)\b",
        r"\b(consists|includes|requires|allows|enables)\b",
        r"\b(specifies|defines|indicates|identifies)\b",
        r"\b\d+(?:\.\d+)?\b",
        r"\bAMF\b",
        r"\bSMF\b",
        r"\bUPF\b",
        r"\b5G\b",
        r"\bNAS\b",
        r"\bPDU\b",
        r"\bNGAP\b",
        r"\bUE\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in factual_patterns
    )


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_citations(
    answer: str,
    retrieved_documents: list[dict],
) -> dict:

    answer = (
        answer or ""
    ).strip()

    evidence = _evidence_keys(
        retrieved_documents
    )

    citations = extract_citations(
        answer
    )

    # ---------------------------------------------------------------
    # Empty answer
    # ---------------------------------------------------------------

    if not answer:

        return {
            "valid": False,
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "available_evidence": sorted(
                evidence
            ),
            "reason": "empty_answer",
        }

    # ---------------------------------------------------------------
    # Explicit abstention
    # ---------------------------------------------------------------

    if answer == "INSUFFICIENT_EVIDENCE":

        return {
            "valid": False,
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "available_evidence": sorted(
                evidence
            ),
            "reason": "explicit_abstention",
        }

    # ---------------------------------------------------------------
    # Malformed citation detection
    # ---------------------------------------------------------------

    citation_like = CITATION_LIKE_RE.findall(
        answer
    )

    valid_citation_texts = {
        match.group(0)
        for match in CITATION_RE.finditer(
            answer
        )
    }

    malformed_citations = []

    for candidate in citation_like:

        if candidate not in valid_citation_texts:
            malformed_citations.append(
                candidate
            )

    if malformed_citations:

        return {
            "valid": False,
            "citations": citations,
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": len(citations),
            "malformed_citations": malformed_citations,
            "available_evidence": sorted(
                evidence
            ),
            "reason": "malformed_citation",
        }

    # ---------------------------------------------------------------
    # No citation
    # ---------------------------------------------------------------

    if not citations:

        return {
            "valid": False,
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "available_evidence": sorted(
                evidence
            ),
            "reason": "missing_citations",
        }

    # ---------------------------------------------------------------
    # Validate every citation
    # ---------------------------------------------------------------

    valid_citations = []
    invalid_citations = []

    for spec, clause in citations:

        key = (
            _norm_spec(spec),
            _norm_clause(clause),
        )

        if key in evidence:
            valid_citations.append(
                (
                    spec,
                    clause,
                )
            )
        else:
            invalid_citations.append(
                (
                    spec,
                    clause,
                )
            )

    if invalid_citations:

        return {
            "valid": False,
            "citations": citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_count": len(citations),
            "available_evidence": sorted(
                evidence
            ),
            "reason": "invalid_citations",
        }

    # ---------------------------------------------------------------
    # Detect uncited factual statements
    # ---------------------------------------------------------------

    sentences = _split_sentences(
        answer
    )

    uncited_factual_sentences = []

    for sentence in sentences:

        if not _looks_factual(sentence):
            continue

        if not _has_nearby_citation(sentence):

            # Ignore standalone markdown bullets containing only
            # formatting fragments.
            stripped = sentence.strip(
                "-*• \t"
            )

            if stripped:
                uncited_factual_sentences.append(
                    sentence
                )

    if uncited_factual_sentences:

        return {
            "valid": False,
            "citations": citations,
            "valid_citations": valid_citations,
            "invalid_citations": [],
            "citation_count": len(citations),
            "uncited_factual_sentences":
                uncited_factual_sentences,
            "available_evidence": sorted(
                evidence
            ),
            "reason": "uncited_factual_claim",
        }

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    return {
        "valid": True,
        "citations": citations,
        "valid_citations": valid_citations,
        "invalid_citations": [],
        "citation_count": len(citations),
        "available_evidence": sorted(
            evidence
        ),
        "reason": "valid_citations",
    }