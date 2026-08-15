import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------

TOKEN_RE = re.compile(
    r"[A-Za-z0-9_.:/-]+"
)


def tokenize(text: str):
    """
    Tokenizer designed for technical 3GPP terminology.

    Preserves:
        AMF
        SMF
        RRC_INACTIVE
        TS 23.501
        5.15.5
        38.300
        NG-RAN
        N2/N3
    """

    if not text:
        return []

    return TOKEN_RE.findall(
        text.lower()
    )


# ------------------------------------------------------------
# BM25 STORE
# ------------------------------------------------------------

class BM25Store:
    """
    Persistent BM25 index for the 3GPP chunk corpus.

    The complete chunk metadata is stored alongside the
    BM25 index so retrieval results can immediately be passed
    to the hybrid retriever and, later, the RAG generation
    pipeline.
    """

    def __init__(
        self,
        path="data/indexes/bm25.pkl",
    ):
        self.path = Path(path)

        self.index = None
        self.chunks = None

    # ========================================================
    # BUILD
    # ========================================================

    def build(self, chunks):
        """
        Build and persist the BM25 index.
        """

        if not chunks:
            raise ValueError(
                "Cannot build BM25 index from empty chunks."
            )

        print(
            f"Building BM25 index for "
            f"{len(chunks)} chunks..."
        )

        corpus = [
            tokenize(
                chunk.get("text", "")
            )
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            corpus
        )

        self.chunks = chunks

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "wb"
        ) as f:

            pickle.dump(
                {
                    "index": self.index,
                    "chunks": self.chunks,
                },
                f,
            )

        print(
            "BM25 index built successfully."
        )

        print(
            f"BM25 path: "
            f"{self.path.resolve()}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):
        """
        Load the persisted BM25 index.
        """

        if not self.path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: "
                f"{self.path.resolve()}"
            )

        with self.path.open(
            "rb"
        ) as f:

            data = pickle.load(f)

        self.index = data["index"]
        self.chunks = data["chunks"]

        if self.index is None:
            raise ValueError(
                "Loaded BM25 index is empty."
            )

        if self.chunks is None:
            raise ValueError(
                "Loaded BM25 chunks are empty."
            )

        if len(self.chunks) == 0:
            raise ValueError(
                "BM25 index contains zero chunks."
            )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        limit: int = 12,
    ):
        """
        Perform BM25 lexical retrieval.

        Returns results in descending BM25 score order.

        Each result contains:

            id
            bm25_score
            bm25_rank
            complete chunk metadata
        """

        if not query or not query.strip():
            raise ValueError(
                "BM25 query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "BM25 limit must be greater than zero."
            )

        if self.index is None:
            self.load()

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = (
            scores
            .argsort()[::-1][:limit]
        )

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            index = int(index)

            chunk = self.chunks[index]

            results.append(
                {
                    "id": str(
                        chunk["id"]
                    ),

                    "bm25_score": float(
                        scores[index]
                    ),

                    "bm25_rank": rank,

                    **chunk,
                }
            )

        return results

    # ========================================================
    # CHECK
    # ========================================================

    def check_index(self):
        """
        Validate that the persisted BM25 index exists
        and contains the expected number of chunks.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "BM25 INDEX CHECK"
        )

        print(
            "=" * 60
        )

        print(
            f"Path: {self.path.resolve()}"
        )

        if not self.path.exists():
            print(
                "Status: NOT FOUND"
            )

            print(
                "=" * 60
            )

            return False

        try:
            self.load()

            print(
                "Status : OK"
            )

            print(
                f"Chunks : {len(self.chunks)}"
            )

            print(
                "=" * 60
            )

            return True

        except Exception as exc:

            print(
                "Status : INVALID"
            )

            print(
                f"Error  : {exc}"
            )

            print(
                "=" * 60
            )

            return False