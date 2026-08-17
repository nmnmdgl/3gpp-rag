from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from .schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    Source,
)


router = APIRouter(
    prefix="/api",
    tags=["RAG"],
)


# =========================================================
# GLOBAL RAG GRAPH
# =========================================================

_graph = None


def get_graph():
    """
    Return the initialized RAG graph.

    The graph is initialized by the background
    initialization process in main.py.
    """

    global _graph

    if _graph is None:
        print(
            "get_graph(): graph is not initialized. "
            "Building graph now...",
            flush=True,
        )

        from src.generation.graph import build_graph

        _graph = build_graph()

        print(
            "get_graph(): graph built successfully.",
            flush=True,
        )

    return _graph


def set_graph(graph):
    """
    Store the initialized RAG graph.

    Called by the background initialization
    process in main.py.
    """

    global _graph

    _graph = graph

    print(
        "set_graph(): RAG graph stored successfully.",
        flush=True,
    )


# =========================================================
# DOCUMENT NAME NORMALIZATION
# =========================================================

def normalize_document_name(
    document: str,
) -> str:

    mapping = {
        "21905": "TR 21.905",
        "23501": "TS 23.501",
        "23502": "TS 23.502",
        "38300": "TS 38.300",
    }

    value = str(document).strip()

    for key, name in mapping.items():

        if value.startswith(key):
            return name

    return value


# =========================================================
# CITATIONS
# =========================================================

def extract_citations(
    citation_check: Dict[str, Any],
) -> List[Citation]:

    citations = []

    for item in citation_check.get(
        "valid_citations",
        [],
    ):

        if not item or len(item) < 2:
            continue

        citations.append(
            Citation(
                document=str(item[0]),
                clause=str(item[1]),
            )
        )

    return citations


# =========================================================
# SOURCES
# =========================================================

def extract_sources(
    documents: List[Dict[str, Any]],
) -> List[Source]:

    sources = []

    seen = set()

    for document in documents:

        if not isinstance(
            document,
            dict,
        ):
            continue

        raw_document = (
            document.get("document")
            or document.get("doc")
            or document.get("spec")
            or document.get("filename")
            or ""
        )

        clause = (
            document.get("clause")
            or document.get("clause_number")
            or document.get("section")
            or ""
        )

        title = (
            document.get("title")
            or document.get("heading")
            or document.get("clause_title")
        )

        score = (
            document.get("score")
            if document.get("score") is not None
            else document.get("rerank_score")
            if document.get("rerank_score") is not None
            else document.get("similarity")
        )

        key = (
            str(raw_document),
            str(clause),
        )

        if key in seen:
            continue

        seen.add(key)

        try:

            numeric_score = (
                float(score)
                if score is not None
                else None
            )

        except (
            TypeError,
            ValueError,
        ):

            numeric_score = None

        sources.append(
            Source(
                document=normalize_document_name(
                    str(raw_document)
                ),
                clause=str(clause),
                title=(
                    str(title)
                    if title
                    else None
                ),
                score=numeric_score,
            )
        )

    return sources


# =========================================================
# HEALTH
# =========================================================

@router.get("/health")
def health():

    return {
        "status": "ok",
        "service": "3GPP RAG API",
        "graph_initialized": _graph is not None,
    }


# =========================================================
# INFO
# =========================================================

@router.get("/info")
def info():

    return {
        "name": "3GPP RAG",
        "description": (
            "Retrieval-Augmented Generation chatbot "
            "for the indexed 3GPP specifications."
        ),
        "model": "openai/gpt-oss-120b",
        "provider": "Groq",
        "documents": [
            {
                "file": "21905-j20",
                "specification": "TR 21.905",
                "description": (
                    "Vocabulary for 3GPP specifications"
                ),
            },
            {
                "file": "23501-k20",
                "specification": "TS 23.501",
                "description": (
                    "System architecture for the 5G System"
                ),
            },
            {
                "file": "23502-k20",
                "specification": "TS 23.502",
                "description": (
                    "Procedures for the 5G System"
                ),
            },
            {
                "file": "38300-fn0",
                "specification": "TS 38.300",
                "description": (
                    "NR and NG-RAN overall description"
                ),
            },
        ],
        "capabilities": [
            "Hybrid retrieval",
            "Dense retrieval",
            "BM25 retrieval",
            "Reranking",
            "Evidence-based generation",
            "Citation validation",
            "Abstention",
            "Groundedness validation",
        ],
        "graph_initialized": _graph is not None,
    }


# =========================================================
# CHAT
# =========================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    print("=" * 70, flush=True)

    print(
        "CHAT REQUEST RECEIVED",
        flush=True,
    )

    print(
        f"Question: {request.question}",
        flush=True,
    )

    print(
        f"Conversation ID: "
        f"{request.conversation_id}",
        flush=True,
    )

    print(
        f"Graph initialized before request: "
        f"{_graph is not None}",
        flush=True,
    )

    print("=" * 70, flush=True)

    # -----------------------------------------------------
    # QUESTION VALIDATION
    # -----------------------------------------------------

    question = request.question.strip()

    if not question:

        print(
            "CHAT ERROR: Empty question.",
            flush=True,
        )

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # GET GRAPH
    # -----------------------------------------------------

    try:

        print(
            "STEP 1: Calling get_graph()...",
            flush=True,
        )

        graph = get_graph()

        print(
            "STEP 2: get_graph() returned.",
            flush=True,
        )

        print(
            f"Graph object type: "
            f"{type(graph).__name__}",
            flush=True,
        )

    except Exception as exc:

        print("=" * 70, flush=True)

        print(
            "GRAPH INITIALIZATION ERROR",
            flush=True,
        )

        print(
            f"Exception type: "
            f"{type(exc).__name__}",
            flush=True,
        )

        print(
            f"Exception: {repr(exc)}",
            flush=True,
        )

        import traceback

        traceback.print_exc()

        print("=" * 70, flush=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG graph could not be initialized."
            ),
        ) from exc

    # -----------------------------------------------------
    # GRAPH INVOCATION
    # -----------------------------------------------------

    try:

        print(
            "STEP 3: Starting graph.invoke()...",
            flush=True,
        )

        result = graph.invoke(
            {
                "question": question,
            }
        )

        print(
            "STEP 4: graph.invoke() COMPLETED.",
            flush=True,
        )

        print(
            f"Result type: "
            f"{type(result).__name__}",
            flush=True,
        )

        if isinstance(
            result,
            dict,
        ):

            print(
                "Result keys: "
                f"{list(result.keys())}",
                flush=True,
            )

        else:

            print(
                "WARNING: Graph result is not a dict.",
                flush=True,
            )

    except Exception as exc:

        print("=" * 70, flush=True)

        print(
            "RAG ERROR DURING GRAPH INVOCATION",
            flush=True,
        )

        print(
            f"Exception type: "
            f"{type(exc).__name__}",
            flush=True,
        )

        print(
            f"Exception: {repr(exc)}",
            flush=True,
        )

        import traceback

        traceback.print_exc()

        print("=" * 70, flush=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG pipeline failed while "
                "processing the question."
            ),
        ) from exc

    # -----------------------------------------------------
    # RESPONSE EXTRACTION
    # -----------------------------------------------------

    print(
        "STEP 5: Extracting answer...",
        flush=True,
    )

    answer = str(
        result.get(
            "answer",
            "INSUFFICIENT_EVIDENCE",
        )
    )

    grounded = bool(
        result.get(
            "grounded",
            False,
        )
    )

    abstained = bool(
        result.get(
            "abstained",
            False,
        )
    )

    reason = str(
        result.get(
            "reason",
            "unknown",
        )
    )

    print(
        f"Answer length: {len(answer)}",
        flush=True,
    )

    print(
        f"Grounded: {grounded}",
        flush=True,
    )

    print(
        f"Abstained: {abstained}",
        flush=True,
    )

    print(
        f"Reason: {reason}",
        flush=True,
    )

    # -----------------------------------------------------
    # CITATION DATA
    # -----------------------------------------------------

    print(
        "STEP 6: Extracting citation data...",
        flush=True,
    )

    citation_check = (
        result.get(
            "citation_check",
            {},
        )
        or {}
    )

    documents = (
        result.get(
            "documents",
            [],
        )
        or []
    )

    print(
        f"Retrieved documents: "
        f"{len(documents)}",
        flush=True,
    )

    citations = extract_citations(
        citation_check
    )

    print(
        f"Extracted citations: "
        f"{len(citations)}",
        flush=True,
    )

    # -----------------------------------------------------
    # SOURCE DATA
    # -----------------------------------------------------

    print(
        "STEP 7: Extracting sources...",
        flush=True,
    )

    sources = extract_sources(
        documents
    )

    print(
        f"Extracted sources: "
        f"{len(sources)}",
        flush=True,
    )

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    print(
        "STEP 8: Creating ChatResponse...",
        flush=True,
    )

    response = ChatResponse(
        answer=answer,
        grounded=grounded,
        abstained=abstained,
        reason=reason,
        citations=citations,
        sources=sources,
        conversation_id=(
            request.conversation_id
        ),
    )

    print(
        "STEP 9: ChatResponse created successfully.",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "CHAT REQUEST COMPLETED SUCCESSFULLY",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    return response