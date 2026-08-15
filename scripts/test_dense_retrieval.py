from src.retrieval.qdrant_store import QdrantStore


QUERIES = [
    # TS 23.501 — 5GS architecture
    "What is the architecture of the 5G System?",
    "What is the role of the AMF in the 5G System?",
    "What is the role of the SMF?",
    "What are the network functions in the 5G Core Network?",

    # TS 23.502 — procedures
    "How is a PDU session established in the 5G System?",
    "What is the registration procedure in 5GS?",

    # TS 38.300 — NR / NG-RAN
    "What is NG-RAN?",
    "What is the overall description of the 5G NR system?",

    # TR 21.905 — terminology
    "What does the term UE mean in 3GPP specifications?",
    "What is the definition of a network function?",
]


def print_results(query, results):

    print("\n" + "=" * 90)
    print("QUERY")
    print("=" * 90)
    print(query)

    print("\n" + "-" * 90)
    print("TOP RESULTS")
    print("-" * 90)

    for rank, result in enumerate(results, start=1):

        print(f"\n[{rank}] Score: {result['score']:.4f}")

        print(
            f"Spec       : "
            f"{result.get('spec_number')}"
        )

        print(
            f"Clause     : "
            f"{result.get('clause')}"
        )

        print(
            f"Clause title: "
            f"{result.get('clause_title')}"
        )

        print(
            f"Content    : "
            f"{result.get('content_type')}"
        )

        print(
            f"Page       : "
            f"{result.get('page')}"
        )

        print("\nText:")
        print(
            result.get("text", "")[:1200]
        )


def main():

    store = QdrantStore()

    print("Testing dense retrieval...")
    print(f"Queries: {len(QUERIES)}")

    for query in QUERIES:

        results = store.search(
            query=query,
            limit=5,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()