import json
from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT BM25 STORE
# ============================================================

from src.retrieval.bm25_store import BM25Store


# ============================================================
# PATHS
# ============================================================

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)

BM25_PATH = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "bm25.pkl"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    # --------------------------------------------------------
    # Check chunks
    # --------------------------------------------------------

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"Chunks file not found:\n"
            f"{CHUNKS_PATH}"
        )

    print()
    print(f"Chunks file:")
    print(f"  {CHUNKS_PATH}")

    # --------------------------------------------------------
    # Load repaired chunks
    # --------------------------------------------------------

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    if not chunks:

        raise ValueError(
            "chunks.json contains zero chunks."
        )

    print()
    print(
        f"Loaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Basic metadata verification
    # --------------------------------------------------------

    document_counts = {}

    for chunk in chunks:

        document_id = chunk.get(
            "document_id",
            "UNKNOWN",
        )

        document_counts[document_id] = (
            document_counts.get(
                document_id,
                0,
            )
            + 1
        )

    print()
    print("Documents:")

    for document_id, count in sorted(
        document_counts.items()
    ):

        print(
            f"  {document_id}: {count}"
        )

    # --------------------------------------------------------
    # Check repaired metadata
    # --------------------------------------------------------

    print()
    print("Checking metadata...")

    bad_metadata = []

    for chunk in chunks:

        document_id = chunk.get(
            "document_id"
        )

        spec_number = chunk.get(
            "spec_number"
        )

        title = chunk.get(
            "title"
        )

        document_type = chunk.get(
            "document_type"
        )

        if document_id in {
            "23501-k20",
            "23502-k20",
        }:

            if (
                not spec_number
                or not title
                or not document_type
            ):

                bad_metadata.append(
                    {
                        "id": chunk.get("id"),
                        "document_id": document_id,
                        "spec_number": spec_number,
                        "title": title,
                        "document_type": document_type,
                    }
                )

    print(
        f"Bad metadata entries: "
        f"{len(bad_metadata)}"
    )

    if bad_metadata:

        print()
        print(
            "ERROR: Metadata repair is incomplete."
        )

        print(
            "Example:"
        )

        print(
            bad_metadata[0]
        )

        raise RuntimeError(
            "Refusing to rebuild BM25 "
            "with invalid metadata."
        )

    # --------------------------------------------------------
    # Existing BM25 backup
    # --------------------------------------------------------

    if BM25_PATH.exists():

        backup_path = BM25_PATH.with_suffix(
            ".before_metadata_repair.pkl"
        )

        print()
        print(
            "Existing BM25 index detected."
        )

        print(
            f"Creating backup:"
        )

        print(
            f"  {backup_path}"
        )

        backup_path.write_bytes(
            BM25_PATH.read_bytes()
        )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print()
    print(
        "Building BM25 index..."
    )

    store = BM25Store(
        path=str(BM25_PATH)
    )

    store.build(
        chunks
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print(
        "Verifying rebuilt BM25..."
    )

    verification = BM25Store(
        path=str(BM25_PATH)
    )

    verification.load()

    if len(verification.chunks) != len(
        chunks
    ):

        raise RuntimeError(
            "BM25 chunk count does not match "
            "chunks.json."
        )

    print()
    print(
        f"BM25 chunks: "
        f"{len(verification.chunks)}"
    )

    # --------------------------------------------------------
    # Check Foreword corruption
    # --------------------------------------------------------

    bad = [
        chunk
        for chunk in verification.chunks
        if chunk.get("title", "").startswith(
            "Foreword"
        )
    ]

    print(
        f"Bad Foreword metadata: "
        f"{len(bad)}"
    )

    if bad:

        raise RuntimeError(
            "BM25 still contains corrupted "
            "Foreword metadata."
        )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print(
        "Example metadata:"
    )

    for document_id in [
        "23501-k20",
        "23502-k20",
    ]:

        examples = [
            chunk
            for chunk in verification.chunks
            if chunk.get(
                "document_id"
            ) == document_id
        ]

        if examples:

            chunk = examples[0]

            print()
            print(
                f"Document: {document_id}"
            )

            print(
                f"  spec_number   : "
                f"{chunk.get('spec_number')}"
            )

            print(
                f"  document_type : "
                f"{chunk.get('document_type')}"
            )

            print(
                f"  title         : "
                f"{chunk.get('title')}"
            )

            print(
                f"  release       : "
                f"{chunk.get('release')}"
            )

            print(
                f"  clause        : "
                f"{chunk.get('clause')}"
            )

            print(
                f"  clause_title  : "
                f"{chunk.get('clause_title')}"
            )

    print()
    print("=" * 70)
    print("BM25 BUILD SUCCESSFUL")
    print("=" * 70)
    print()
    print(
        f"Output: {BM25_PATH}"
    )
    print(
        f"Chunks: {len(verification.chunks)}"
    )
    print(
        "Vectors/embeddings were NOT modified."
    )
    print(
        "Qdrant was NOT modified."
    )
    print()


if __name__ == "__main__":
    main()