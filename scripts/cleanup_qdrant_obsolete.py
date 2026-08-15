import json
from pathlib import Path

from qdrant_client import QdrantClient


CHUNKS_PATH = Path("data/processed/chunks/chunks.json")
QDRANT_PATH = "data/qdrant"
COLLECTION = "3gpp_specs"


def main():
    print("=" * 70)
    print("QDRANT OBSOLETE POINT CLEANUP")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load current chunk IDs
    # ---------------------------------------------------------

    print("\nLoading current chunks...")

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    current_ids = {
        str(chunk["id"])
        for chunk in chunks
    }

    print(f"Current chunks : {len(current_ids)}")

    # ---------------------------------------------------------
    # Open Qdrant
    # ---------------------------------------------------------

    client = QdrantClient(path=QDRANT_PATH)

    collection = client.get_collection(COLLECTION)

    print(f"Qdrant points  : {collection.points_count}")

    # ---------------------------------------------------------
    # Read all points
    # ---------------------------------------------------------

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=20000,
        with_payload=True,
        with_vectors=False,
    )

    print(f"Points loaded  : {len(points)}")

    # ---------------------------------------------------------
    # Identify obsolete points
    # ---------------------------------------------------------

    obsolete_point_ids = []

    for point in points:
        payload = point.payload or {}
        chunk_id = str(payload.get("id", ""))

        if chunk_id not in current_ids:
            obsolete_point_ids.append(point.id)

    print(f"Obsolete points: {len(obsolete_point_ids)}")

    if not obsolete_point_ids:
        print("\nNo obsolete points found.")
        client.close()
        return

    # ---------------------------------------------------------
    # Show examples
    # ---------------------------------------------------------

    print("\nExamples of points to delete:")

    shown = 0

    for point in points:
        if point.id in obsolete_point_ids:
            payload = point.payload or {}

            print({
                "point_id": point.id,
                "id": payload.get("id"),
                "document_id": payload.get("document_id"),
                "spec_number": payload.get("spec_number"),
                "title": payload.get("title"),
                "clause": payload.get("clause"),
            })

            shown += 1

            if shown >= 5:
                break

    # ---------------------------------------------------------
    # Delete obsolete points
    # ---------------------------------------------------------

    print("\nDeleting obsolete points...")

    client.delete(
        collection_name=COLLECTION,
        points_selector=obsolete_point_ids,
        wait=True,
    )

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    collection = client.get_collection(COLLECTION)

    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)

    print(f"Original points : {len(points)}")
    print(f"Deleted points  : {len(obsolete_point_ids)}")
    print(f"Remaining points: {collection.points_count}")
    print(f"Current chunks  : {len(current_ids)}")

    if collection.points_count == len(current_ids):
        print("\nSTATUS: SUCCESS")
        print("Qdrant is now synchronized with chunks.json.")
    else:
        print("\nSTATUS: WARNING")
        print("Qdrant count does not match chunks.json.")

    client.close()


if __name__ == "__main__":
    main()