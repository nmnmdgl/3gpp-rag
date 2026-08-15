from src.retrieval.qdrant_store import QdrantStore


def main():
    store = QdrantStore()

    info = store.collection_info()

    if info is None:
        print("ERROR: Qdrant collection does not exist.")
        return

    print("=" * 60)
    print("QDRANT INDEX CHECK")
    print("=" * 60)

    print(f"Collection : {store.collection}")
    print(f"Vectors    : {info.points_count}")
    print(f"Status     : {info.status}")

    print("\nVector configuration:")
    print(info.config.params.vectors)

    print("\nQdrant path:")
    print(store.storage_path.resolve())

    print("=" * 60)


if __name__ == "__main__":
    main()