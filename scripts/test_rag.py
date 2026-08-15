from src.generation.graph import build_graph


def main():

    graph = build_graph()

    question = (
        "What is the role of the AMF in the 5G System?"
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print(question)
    print("=" * 80)

    result = graph.invoke(
        {
            "question": question
        }
    )

    print("\nANSWER:")
    print(
        result.get(
            "answer",
            "NO ANSWER",
        )
    )

    print("\nABSTAINED:")
    print(
        result.get(
            "abstained",
            False,
        )
    )

    print("\nGROUNDED:")
    print(
        result.get(
            "grounded",
            False,
        )
    )

    print("\nREASON:")
    print(
        result.get(
            "reason",
            "UNKNOWN",
        )
    )

    print("\nCITATION CHECK:")
    print(
        result.get(
            "citation_check",
            {},
        )
    )

    print("\nRETRIEVED SOURCES:")

    for i, doc in enumerate(
        result.get("documents", []),
        start=1,
    ):

        print(
            f"{i}. "
            f"{doc.get('spec_number', doc.get('spec'))} | "
            f"Clause {doc.get('clause')} | "
            f"{doc.get('clause_title', '')}"
        )


if __name__ == "__main__":
    main()