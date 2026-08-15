import json
from pathlib import Path

from qdrant_client import QdrantClient


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "qdrant"
)

COLLECTION_NAME = "3gpp_specs"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REPAIRING QDRANT METADATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load repaired chunks
    # --------------------------------------------------------

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    print()
    print(f"Chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # Create metadata lookup by immutable chunk ID
    # --------------------------------------------------------

    metadata_by_id = {}

    for chunk in chunks:

        chunk_id = chunk.get("id")

        if not chunk_id:
            continue

        metadata_by_id[str(chunk_id)] = {
            "id": chunk.get("id"),
            "document_id": chunk.get("document_id"),
            "spec_number": chunk.get("spec_number"),
            "document_type": chunk.get("document_type"),
            "title": chunk.get("title"),
            "version": chunk.get("version"),
            "release": chunk.get("release"),
            "source_file": chunk.get("source_file"),
            "clause": chunk.get("clause"),
            "clause_title": chunk.get("clause_title"),
            "clause_path": chunk.get("clause_path"),
            "content_type": chunk.get("content_type"),
            "block_index": chunk.get("block_index"),
            "chunk_index": chunk.get("chunk_index"),
            "page": chunk.get("page"),
        }

    print(
        f"Metadata lookup entries: "
        f"{len(metadata_by_id)}"
    )

    # --------------------------------------------------------
    # Connect to local Qdrant
    # --------------------------------------------------------

    print()
    print(
        f"Opening Qdrant: {QDRANT_PATH}"
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Qdrant points: "
        f"{collection.points_count}"
    )

    # --------------------------------------------------------
    # Iterate through ALL Qdrant points
    # --------------------------------------------------------

    offset = None

    total = 0
    updated = 0
    missing = 0

    print()
    print("Updating payloads...")

    while True:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:

            total += 1

            payload = point.payload or {}

            chunk_id = payload.get("id")

            if chunk_id is None:
                missing += 1
                continue

            chunk_id = str(chunk_id)

            metadata = metadata_by_id.get(
                chunk_id
            )

            if metadata is None:
                missing += 1
                continue

            # Only replace metadata payload.
            # The vector is NOT touched.
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload=metadata,
                points=[point.id],
            )

            updated += 1

        offset = next_offset

        if offset is None:
            break

        print(
            f"Processed: {total} | "
            f"Updated: {updated}"
        )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    client.close()

    print()
    print("=" * 70)
    print("QDRANT METADATA REPAIR COMPLETE")
    print("=" * 70)

    print(
        f"Qdrant points processed : {total}"
    )

    print(
        f"Payloads updated        : {updated}"
    )

    print(
        f"Missing metadata        : {missing}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Vectors/embeddings were NOT modified."
    )

    print(
        "Only Qdrant payload metadata was updated."
    )

    print()


if __name__ == "__main__":
    main()