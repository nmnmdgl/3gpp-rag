import json
import time

from src.retrieval.embeddings import LocalEmbedder


def main():

    chunks = json.loads(
        open(
            "data/processed/chunks/chunks.json",
            encoding="utf-8",
        ).read()
    )

    texts = [
        chunk["text"]
        for chunk in chunks[:128]
    ]

    embedder = LocalEmbedder()

    print(
        f"Benchmarking {len(texts)} chunks..."
    )

    start = time.perf_counter()

    vectors = embedder.encode_documents(
        texts,
        batch_size=32,
    )

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"\nTime: {elapsed:.2f} seconds"
    )

    print(
        f"Chunks/sec: "
        f"{len(texts) / elapsed:.2f}"
    )

    print(
        f"Vector shape: {vectors.shape}"
    )


if __name__ == "__main__":
    main()