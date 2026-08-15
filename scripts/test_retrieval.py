from src.retrieval.hybrid_retriever import HybridRetriever


def print_result(
    result,
):
    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Rank       : {result.get('rank')}"
    )

    print(
        f"RRF score  : "
        f"{result.get('rrf_score', 0):.6f}"
    )

    print(
        f"Dense score: "
        f"{result.get('dense_score', 0):.6f}"
    )

    if "bm25_score" in result:
        print(
            f"BM25 score : "
            f"{result['bm25_score']:.6f}"
        )

    print(
        f"Sources    : "
        f"{', '.join(result.get('retrieval_sources', []))}"
    )

    print(
        f"Spec       : "
        f"{result.get('spec_number')}"
    )

    print(
        f"Clause     : "
        f"{result.get('clause')}"
    )

    print(
        f"Clause     : "
        f"{result.get('clause_title')}"
    )

    print(
        f"Type       : "
        f"{result.get('content_type')}"
    )

    print(
        "\nTEXT:"
    )

    print(
        result.get("text", "")[:1200]
    )


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "3GPP HYBRID RETRIEVAL TEST"
    )

    print(
        "=" * 70
    )

    retriever = HybridRetriever(
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=10,
    )

    queries = [
        "What is the architecture of the 5G System?",
        "What is the role of the AMF?",
        "What is the role of the SMF?",
        "What is the purpose of the UPF?",
        "What is RRC?",
    ]

    for query in queries:

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "=" * 70
        )

        results = retriever.retrieve(
            query
        )

        if not results:
            print(
                "NO RESULTS"
            )
            continue

        for result in results:
            print_result(
                result
            )


if __name__ == "__main__":
    main()