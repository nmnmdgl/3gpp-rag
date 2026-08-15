import json
import pickle
from pathlib import Path


CHUNKS_PATH = Path("data/processed/chunks/chunks.json")
BM25_PATH = Path("data/indexes/bm25.pkl")


DOCUMENT_METADATA = {
    "21905-j20": {
        "spec_number": "TR 21.905",
        "document_type": "TR",
        "title": "GSM, UMTS, LTE, 5G, Vocabulary",
        "release": "19",
    },

    "23501-k20": {
        "spec_number": "TS 23.501",
        "document_type": "TS",
        "title": "System architecture for the 5G System",
        "release": "19",
    },

    "23502-k20": {
        "spec_number": "TS 23.502",
        "document_type": "TS",
        "title": "Procedures for the 5G System",
        "release": "19",
    },

    "38300-fn0": {
        "spec_number": "TS 38.300",
        "document_type": "TS",
        "title": "NR; NR and NG-RAN Overall Description",
        "release": "15",
    },
}


def repair_chunk(chunk):
    document_id = chunk.get("document_id")

    metadata = DOCUMENT_METADATA.get(document_id)

    if metadata is None:
        print(
            f"WARNING: No metadata mapping for "
            f"{document_id}"
        )
        return chunk

    # --------------------------------------------------------
    # Repair document-level metadata
    # --------------------------------------------------------

    chunk["spec_number"] = metadata["spec_number"]
    chunk["document_type"] = metadata["document_type"]
    chunk["title"] = metadata["title"]
    chunk["release"] = metadata["release"]

    # --------------------------------------------------------
    # Keep existing version if meaningful.
    # Otherwise leave it empty.
    # --------------------------------------------------------

    if not chunk.get("version"):
        chunk["version"] = ""

    return chunk


def main():

    print("=" * 70)
    print("3GPP METADATA REPAIR")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print(f"\nLoading chunks:")
    print(CHUNKS_PATH)

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)

    print(f"Chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # Repair
    # --------------------------------------------------------

    repaired = 0

    for chunk in chunks:

        before = (
            chunk.get("spec_number"),
            chunk.get("document_type"),
            chunk.get("title"),
            chunk.get("release"),
        )

        repair_chunk(chunk)

        after = (
            chunk.get("spec_number"),
            chunk.get("document_type"),
            chunk.get("title"),
            chunk.get("release"),
        )

        if before != after:
            repaired += 1

    print(
        f"Chunks whose metadata changed: {repaired}"
    )

    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nUpdated:"
        f"\n{CHUNKS_PATH}"
    )

    # --------------------------------------------------------
    # Repair BM25 metadata
    # --------------------------------------------------------

    if BM25_PATH.exists():

        print("\nLoading BM25 index...")

        with BM25_PATH.open(
            "rb"
        ) as f:
            data = pickle.load(f)

        bm25_chunks = data["chunks"]

        print(
            f"BM25 chunks: {len(bm25_chunks)}"
        )

        if len(bm25_chunks) != len(chunks):
            raise RuntimeError(
                "BM25 chunk count does not match "
                "chunks.json"
            )

        for chunk in bm25_chunks:

            document_id = chunk.get(
                "document_id"
            )

            metadata = DOCUMENT_METADATA.get(
                document_id
            )

            if metadata is None:
                continue

            chunk["spec_number"] = metadata[
                "spec_number"
            ]

            chunk["document_type"] = metadata[
                "document_type"
            ]

            chunk["title"] = metadata[
                "title"
            ]

            chunk["release"] = metadata[
                "release"
            ]

        with BM25_PATH.open(
            "wb"
        ) as f:
            pickle.dump(
                data,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Updated BM25 metadata:"
            f"\n{BM25_PATH}"
        )

    else:
        print(
            "\nWARNING: BM25 index not found."
        )

    print("\n" + "=" * 70)
    print("REPAIR COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()