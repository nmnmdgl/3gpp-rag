import argparse
import json
from pathlib import Path

import numpy as np

from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.bm25_store import BM25Store


CHUNKS_PATH = Path(
    "data/processed/chunks/chunks.json"
)

EMBEDDINGS_PATH = Path(
    "data/processed/embeddings/embeddings.npy"
)


def load_chunks():
    """
    Load processed chunks.
    """

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found:\n"
            f"{CHUNKS_PATH.resolve()}"
        )

    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not chunks:
        raise ValueError(
            "chunks.json is empty."
        )

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    return chunks


def load_embeddings():
    """
    Load precomputed embeddings generated on Colab.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"embeddings.npy not found:\n"
            f"{EMBEDDINGS_PATH.resolve()}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Loaded embeddings: {embeddings.shape}"
    )

    return embeddings


def validate_chunks_and_embeddings(
    chunks,
    embeddings,
):
    """
    Ensure the chunks and embeddings correspond exactly.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "\nChunk/embedding mismatch!\n"
            f"Chunks     : {len(chunks)}\n"
            f"Embeddings : {len(embeddings)}\n\n"
            "The embeddings.npy must have been generated "
            "from this exact chunks.json."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings.npy must be a 2D array.\n"
            f"Received shape: {embeddings.shape}"
        )

    if np.isnan(embeddings).any():
        raise ValueError(
            "embeddings.npy contains NaN values."
        )

    if np.isinf(embeddings).any():
        raise ValueError(
            "embeddings.npy contains Inf values."
        )

    print(
        "\nChunk/embedding validation: PASSED"
    )


def build_qdrant(
    chunks,
    embeddings,
):
    """
    Build the local Qdrant dense vector index.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "[1/2] BUILDING QDRANT INDEX"
    )

    print(
        "=" * 60
    )

    store = QdrantStore()

    store.index(
        chunks=chunks,
        embeddings=embeddings,
        batch_size=256,
    )

    print(
        "\nQdrant index:"
    )

    store.check_index()


def build_bm25(chunks):
    """
    Build the local BM25 sparse retrieval index.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "[2/2] BUILDING BM25 INDEX"
    )

    print(
        "=" * 60
    )

    store = BM25Store()

    store.build(
        chunks
    )

    print(
        "\nBM25 index built successfully."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build local 3GPP RAG retrieval indexes "
            "from precomputed Colab embeddings."
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "index",
            "qdrant",
            "bm25",
            "check",
        ],
        default="index",
        help=(
            "index = Qdrant + BM25, "
            "qdrant = Qdrant only, "
            "bm25 = BM25 only, "
            "check = Qdrant health check"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # CHECK
    # ---------------------------------------------------------

    if args.stage == "check":
        QdrantStore().check_index()
        return

    # ---------------------------------------------------------
    # LOAD CHUNKS
    # ---------------------------------------------------------

    chunks = load_chunks()

    # ---------------------------------------------------------
    # BM25 ONLY
    # ---------------------------------------------------------

    if args.stage == "bm25":
        build_bm25(chunks)
        return

    # ---------------------------------------------------------
    # LOAD PRECOMPUTED EMBEDDINGS
    # ---------------------------------------------------------

    embeddings = load_embeddings()

    validate_chunks_and_embeddings(
        chunks,
        embeddings,
    )

    # ---------------------------------------------------------
    # QDRANT ONLY
    # ---------------------------------------------------------

    if args.stage == "qdrant":
        build_qdrant(
            chunks,
            embeddings,
        )
        return

    # ---------------------------------------------------------
    # FULL INDEX
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STAGE 2 — INDEXING"
    )

    print(
        "=" * 60
    )

    print(
        f"\nChunks     : {len(chunks)}"
    )

    print(
        f"Embeddings : {embeddings.shape}"
    )

    build_qdrant(
        chunks,
        embeddings,
    )

    build_bm25(
        chunks
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "INDEXING COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()