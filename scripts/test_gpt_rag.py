from src.generation.graph import build_graph


graph = build_graph()


questions = [
    "What is the role of the UPF?",
    "What is the architecture of the 5G System?",
    "What is the capital of France?",
    "What is the exact hardware model of the UPF used by a specific operator?",
]


for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    result = graph.invoke(
        {
            "question": question
        }
    )

    print("\nANSWER:")
    print(
        result.get(
            "answer",
            "NO ANSWER"
        )
    )

    print("\nGROUNDED:")
    print(
        result.get(
            "grounded",
            False
        )
    )

    print("\nREASON:")
    print(
        result.get(
            "reason",
            "N/A"
        )
    )