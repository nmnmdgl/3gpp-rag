import json
import re
from pathlib import Path

from src.retrieval.retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = Path(
    "evaluation/questions.json"
)

K_VALUES = [5, 10, 20]


# ============================================================
# SPECIFICATION NORMALIZATION
# ============================================================

def normalize_spec(spec: str) -> str:
    """
    Convert internal 3GPP document IDs and canonical
    specification numbers into one common representation.

    Examples
    --------
    23501-k20  -> TS 23.501
    23502-k20  -> TS 23.502
    21905-j20  -> TR 21.905
    38300-fn0  -> TS 38.300

    TS 23.501 -> TS 23.501
    TR 21.905 -> TR 21.905
    """

    if not spec:
        return ""

    spec = str(spec).strip().upper()

    # --------------------------------------------------------
    # Already canonical:
    #
    # TS 23.501
    # TR 21.905
    # --------------------------------------------------------

    match = re.fullmatch(
        r"(TS|TR)\s*(\d{2})\.(\d{3})",
        spec,
    )

    if match:
        return (
            f"{match.group(1)} "
            f"{match.group(2)}."
            f"{match.group(3)}"
        )

    # --------------------------------------------------------
    # Internal document IDs:
    #
    # 23501-k20
    # 23502-k20
    # 21905-j20
    # 38300-fn0
    # --------------------------------------------------------

    match = re.match(
        r"^(\d{5})[-_]",
        spec,
    )

    if not match:
        return spec

    number = match.group(1)

    series = number[:2]
    suffix = number[2:]

    # 21xxx -> TR
    # 23xxx -> TS
    # 38xxx -> TS
    #
    # This matches the four-document corpus being evaluated.
    if series == "21":
        document_type = "TR"
    else:
        document_type = "TS"

    return f"{document_type} {series}.{suffix}"


# ============================================================
# SOURCE NORMALIZATION
# ============================================================

def normalize_expected_source(source: dict) -> dict:
    """
    Normalize an expected source from evaluation/questions.json.
    """

    return {
        "spec": normalize_spec(
            source.get("spec", "")
        ),
        "clause": str(
            source.get("clause", "")
        ).strip(),
    }


def normalize_retrieved_source(document: dict) -> dict:
    """
    Normalize a source returned by the Retriever.
    """

    raw_spec = (
        document.get("spec_number")
        or document.get("document_id")
        or ""
    )

    return {
        "spec": normalize_spec(raw_spec),
        "clause": str(
            document.get("clause", "")
        ).strip(),
    }


# ============================================================
# SOURCE MATCHING
# ============================================================

def source_matches(
    expected: dict,
    retrieved: dict,
) -> bool:
    """
    Determine whether a retrieved document matches
    an expected source.

    Matching is performed on:

        specification + clause
    """

    expected_source = normalize_expected_source(
        expected
    )

    retrieved_source = normalize_retrieved_source(
        retrieved
    )

    return (
        expected_source["spec"]
        == retrieved_source["spec"]
        and
        expected_source["clause"]
        ==
        retrieved_source["clause"]
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved: list,
    expected_sources: list,
    k: int,
) -> float:
    """
    Calculate source-level Recall@K.

    If a question has three expected sources and
    two are retrieved in the top-K:

        Recall@K = 2 / 3 = 0.667

    If all three are retrieved:

        Recall@K = 1.0
    """

    if not expected_sources:
        return 0.0

    top_k = retrieved[:k]

    matched = 0

    for expected in expected_sources:

        found = any(
            source_matches(
                expected,
                result,
            )
            for result in top_k
        )

        if found:
            matched += 1

    return matched / len(expected_sources)


# ============================================================
# MATCH DETAILS
# ============================================================

def get_matched_sources(
    retrieved: list,
    expected_sources: list,
    k: int,
):
    """
    Return which expected sources were successfully
    retrieved within top-K.
    """

    top_k = retrieved[:k]

    matches = []

    for expected in expected_sources:

        found_rank = None

        for rank, result in enumerate(
            top_k,
            start=1,
        ):
            if source_matches(
                expected,
                result,
            ):
                found_rank = rank
                break

        matches.append(
            {
                "expected": normalize_expected_source(
                    expected
                ),
                "retrieved": found_rank is not None,
                "rank": found_rank,
            }
        )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: "
            f"{QUESTIONS_PATH.resolve()}"
        )

    questions = json.loads(
        QUESTIONS_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"K values: {K_VALUES}"
    )

    print()

    # --------------------------------------------------------
    # Load retriever once.
    # --------------------------------------------------------

    retriever = Retriever()

    # --------------------------------------------------------
    # Metrics storage
    # --------------------------------------------------------

    totals = {
        k: 0.0
        for k in K_VALUES
    }

    hits = {
        k: 0
        for k in K_VALUES
    }

    # --------------------------------------------------------
    # Evaluate every question.
    # --------------------------------------------------------

    for index, question in enumerate(
        questions,
        start=1,
    ):

        question_id = question.get(
            "id",
            str(index),
        )

        question_text = question.get(
            "question",
            "",
        )

        expected_sources = question.get(
            "expected_sources",
            [],
        )

        print("-" * 80)

        print(
            f"[{index}/{len(questions)}] "
            f"{question_text}"
        )

        print()

        print("EXPECTED SOURCES:")

        for expected in expected_sources:

            normalized = (
                normalize_expected_source(
                    expected
                )
            )

            print(
                f"  - "
                f"{normalized['spec']} | "
                f"Clause {normalized['clause']}"
            )

        # ----------------------------------------------------
        # Retrieve.
        # ----------------------------------------------------

        retrieved = retriever.retrieve(
            question_text
        )

        print()

        print(
            f"Retrieved results: "
            f"{len(retrieved)}"
        )

        # ----------------------------------------------------
        # Show retrieved results.
        # ----------------------------------------------------

        for rank, document in enumerate(
            retrieved,
            start=1,
        ):

            normalized = (
                normalize_retrieved_source(
                    document
                )
            )

            print(
                f"  {rank:2d}. "
                f"{normalized['spec']} | "
                f"Clause "
                f"{normalized['clause']} | "
                f"{document.get('clause_title', '')}"
            )

        print()

        # ----------------------------------------------------
        # Calculate Recall@K.
        # ----------------------------------------------------

        for k in K_VALUES:

            recall = recall_at_k(
                retrieved,
                expected_sources,
                k,
            )

            totals[k] += recall

            if recall == 1.0:
                hits[k] += 1

            print(
                f"Recall@{k}: "
                f"{recall:.3f}"
            )

        # ----------------------------------------------------
        # Detailed source matching for Recall@20.
        # ----------------------------------------------------

        matches = get_matched_sources(
            retrieved,
            expected_sources,
            max(K_VALUES),
        )

        print()

        print("SOURCE MATCHES:")

        for match in matches:

            expected = match["expected"]

            if match["retrieved"]:
                print(
                    f"  ✓ "
                    f"{expected['spec']} | "
                    f"Clause "
                    f"{expected['clause']} "
                    f"(rank {match['rank']})"
                )
            else:
                print(
                    f"  ✗ "
                    f"{expected['spec']} | "
                    f"Clause "
                    f"{expected['clause']} "
                    f"(not retrieved)"
                )

        print()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 80)

    total_questions = len(
        questions
    )

    for k in K_VALUES:

        mean_recall = (
            totals[k]
            / total_questions
            if total_questions
            else 0.0
        )

        perfect_recall = (
            hits[k]
            / total_questions
            if total_questions
            else 0.0
        )

        print(
            f"Recall@{k:<2}: "
            f"{mean_recall:.3f} "
            f"({mean_recall * 100:.2f}%)"
        )

        print(
            f"Perfect Recall@{k:<2}: "
            f"{perfect_recall:.3f} "
            f"({perfect_recall * 100:.2f}%)"
        )

        print()

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()