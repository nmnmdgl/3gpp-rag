from typing import Any, Dict, List
import threading
import traceback

from fastapi import APIRouter, HTTPException

from .schemas import ChatRequest, ChatResponse, Citation, Source


router = APIRouter(prefix="/api", tags=["RAG"])


# =========================================================
# GLOBAL RAG GRAPH
# =========================================================

_graph = None
_graph_lock = threading.Lock()
_graph_initializing = False
_graph_error = None


# =========================================================
# RAG GRAPH INITIALIZATION
# =========================================================

def initialize_graph_background():
    """
    Initialize the RAG graph in a background thread.

    This prevents heavy model loading from blocking the
    FastAPI startup/healthcheck.
    """

    global _graph
    global _graph_initializing
    global _graph_error

    with _graph_lock:

        if _graph is not None:
            print("RAG graph already initialized.")
            return

        if _graph_initializing:
            print("RAG graph initialization already running.")
            return

        _graph_initializing = True

    print("=" * 70)
    print("Background RAG initialization started")
    print("=" * 70)

    try:

        print("Initializing RAG graph...")

        from src.generation.graph import build_graph

        graph = build_graph()

        with _graph_lock:
            _graph = graph
            _graph_error = None

        print("=" * 70)
        print("RAG graph initialized successfully.")
        print("=" * 70)

    except Exception as exc:

        with _graph_lock:
            _graph_error = repr(exc)

        print("=" * 70)
        print("RAG graph initialization failed:")
        print(repr(exc))
        print("=" * 70)

        traceback.print_exc()

    finally:

        with _graph_lock:
            _graph_initializing = False


def start_graph_initialization():

    thread = threading.Thread(
        target=initialize_graph_background,
        daemon=True,
        name="rag-graph-init",
    )

    thread.start()

    return thread


def get_graph():

    global _graph

    if _graph is not None:
        return _graph

    raise RuntimeError(
        "RAG graph is still initializing. "
        "Please retry in a few seconds."
    )


# =========================================================
# DOCUMENT NORMALIZATION
# =========================================================

def normalize_document_name(document: str) -> str:

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
    citation_check: Dict[str, Any]
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
    documents: List[Dict[str, Any]]
) -> List[Source]:

    sources = []

    seen = set()

    for document in documents:

        if not isinstance(document, dict):
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

    global _graph
    global _graph_initializing
    global _graph_error

    if _graph is not None:

        return {
            "status": "ok",
            "service": "3GPP RAG API",
            "rag_status": "ready",
        }

    if _graph_initializing:

        return {
            "status": "ok",
            "service": "3GPP RAG API",
            "rag_status": "initializing",
        }

    if _graph_error is not None:

        return {
            "status": "ok",
            "service": "3GPP RAG API",
            "rag_status": "error",
            "error": _graph_error,
        }

    return {
        "status": "ok",
        "service": "3GPP RAG API",
        "rag_status": "not_started",
    }


# =========================================================
# RAG STATUS
# =========================================================

@router.get("/rag-status")
def rag_status():

    if _graph is not None:

        return {
            "status": "ready",
            "message": "RAG graph is initialized.",
        }

    if _graph_initializing:

        return {
            "status": "initializing",
            "message": (
                "RAG graph is still loading "
                "the embedding/reranker models."
            ),
        }

    if _graph_error:

        return {
            "status": "error",
            "message": "RAG graph initialization failed.",
            "error": _graph_error,
        }

    return {
        "status": "not_started",
        "message": "RAG graph initialization has not started.",
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
    }


# =========================================================
# CHAT
# =========================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # IMPORTANT:
    # Do not try to initialize the graph here.
    #
    # The graph is initialized in the background.
    # -----------------------------------------------------

    try:

        graph = get_graph()

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    # -----------------------------------------------------
    # RUN RAG
    # -----------------------------------------------------

    try:

        print("=" * 70)
        print("RAG REQUEST")
        print(f"Question: {question}")
        print("=" * 70)

        result = graph.invoke(
            {
                "question": question,
            }
        )

        print("RAG REQUEST COMPLETED")

    except Exception as exc:

        print("=" * 70)
        print("RAG ERROR")
        print(repr(exc))
        print("=" * 70)

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG pipeline failed while "
                "processing the question."
            ),
        ) from exc

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

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

    return ChatResponse(
        answer=answer,
        grounded=grounded,
        abstained=abstained,
        reason=reason,
        citations=extract_citations(
            citation_check
        ),
        sources=extract_sources(
            documents
        ),
        conversation_id=request.conversation_id,
    )