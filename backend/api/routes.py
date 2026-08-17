from typing import Any, Dict, List
import threading

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


_graph = None
_graph_error = None

_graph_lock = threading.Lock()
_initialization_started = False


def initialize_graph_background():
    global _graph
    global _graph_error
    global _initialization_started

    with _graph_lock:
        if _initialization_started:
            return

        _initialization_started = True

    print("=" * 70, flush=True)
    print("Background RAG initialization started", flush=True)
    print("=" * 70, flush=True)

    try:
        print(
            "Initializing RAG graph...",
            flush=True,
        )

        from src.generation.graph import build_graph

        graph = build_graph()

        _graph = graph
        _graph_error = None

        print("=" * 70, flush=True)
        print(
            "RAG graph initialized successfully.",
            flush=True,
        )
        print("=" * 70, flush=True)

    except Exception as exc:
        _graph_error = repr(exc)

        print("=" * 70, flush=True)
        print(
            "WARNING: RAG graph initialization failed:",
            flush=True,
        )
        print(
            repr(exc),
            flush=True,
        )
        print("=" * 70, flush=True)


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


def extract_sources(
    documents: List[Dict[str, Any]],
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


@router.get("/health")
def health():

    if _graph is not None:
        return {
            "status": "ok",
            "service": "3GPP RAG API",
            "rag": "ready",
        }

    if _graph_error is not None:
        return {
            "status": "ok",
            "service": "3GPP RAG API",
            "rag": "error",
        }

    return {
        "status": "ok",
        "service": "3GPP RAG API",
        "rag": "initializing",
    }


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


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # IMPORTANT:
    # Never initialize the RAG system from the frontend request.
    # If models/indexes are still loading, return immediately.
    if _graph is None:

        if _graph_error is not None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG initialization failed. "
                    "Check backend logs."
                ),
            )

        raise HTTPException(
            status_code=503,
            detail=(
                "RAG system is still initializing. "
                "Please try again shortly."
            ),
        )

    try:

        result = _graph.invoke(
            {
                "question": question,
            }
        )

    except Exception as exc:

        print(
            "RAG ERROR:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG pipeline failed while "
                "processing the question."
            ),
        ) from exc

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