from typing import TypedDict, List, Dict
from functools import lru_cache

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from .llm import get_llm

from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)

from .context import (
    format_context,
)

from .citation_validator import (
    validate_citations,
)

from .groundedness import (
    is_abstention,
)

from ..retrieval.retriever import (
    Retriever,
)


# =========================================================
# STATE
# =========================================================

class RAGState(TypedDict, total=False):

    question: str

    documents: List[Dict]

    context: str

    answer: str

    citation_check: Dict

    grounded: bool

    abstained: bool

    reason: str


# =========================================================
# LAZY RETRIEVER
# =========================================================

@lru_cache(maxsize=1)
def get_retriever():

    print("=" * 70)
    print("Initializing RAG retriever...")
    print("=" * 70)

    retriever = Retriever(
        dense_k=12,
        bm25_k=12,
        final_k=8,
        rerank_k=5,
    )

    print("=" * 70)
    print("RAG retriever initialized successfully.")
    print("=" * 70)

    return retriever


# =========================================================
# LAZY LLM
# =========================================================

@lru_cache(maxsize=1)
def get_rag_llm():

    print("=" * 70)
    print("Initializing RAG LLM...")
    print("=" * 70)

    llm = get_llm()

    print("=" * 70)
    print("RAG LLM initialized successfully.")
    print("=" * 70)

    return llm


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_node(
    state: RAGState,
):

    question = state["question"]

    print("=" * 70)
    print("RAG RETRIEVAL")
    print(f"Question: {question}")
    print("=" * 70)

    retriever = get_retriever()

    documents = retriever.retrieve(
        question
    )

    print(
        f"Retrieved documents: {len(documents)}"
    )

    context = format_context(
        documents,
        max_documents=8,
    )

    print(
        f"Context length: {len(context)} characters"
    )

    return {
        "documents": documents,
        "context": context,
    }


# =========================================================
# EVIDENCE GATE
# =========================================================

def evidence_node(
    state: RAGState,
):

    documents = state.get(
        "documents",
        [],
    )

    print("=" * 70)
    print("EVIDENCE CHECK")
    print(
        f"Documents available: {len(documents)}"
    )
    print("=" * 70)

    if not documents:

        print(
            "No documents retrieved. "
            "Abstaining."
        )

        return {
            "abstained": True,
            "answer": "INSUFFICIENT_EVIDENCE",
            "reason": "no_retrieved_documents",
        }

    return {
        "abstained": False,
        "reason": "evidence_available",
    }


# =========================================================
# GENERATION
# =========================================================

def generate_node(
    state: RAGState,
):

    if state.get("abstained"):

        print(
            "Generation skipped because "
            "pipeline is abstaining."
        )

        return {}

    question = state["question"]

    context = state.get(
        "context",
        "",
    )

    print("=" * 70)
    print("RAG GENERATION")
    print(f"Question: {question}")
    print(
        f"Context length: {len(context)} characters"
    )
    print("=" * 70)

    llm = get_rag_llm()

    prompt = USER_PROMPT.format(
        question=question,
        context=context,
    )

    try:

        print("Calling LLM...")

        response = llm.invoke(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    prompt,
                ),
            ]
        )

    except Exception as exc:

        print("=" * 70)
        print("LLM ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 70)

        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "abstained": True,
            "grounded": False,
            "reason": (
                "llm_error:"
                f"{type(exc).__name__}"
            ),
        }

    answer = (
        getattr(
            response,
            "content",
            "",
        )
        or ""
    ).strip()

    print(
        f"LLM response length: {len(answer)}"
    )

    if not answer:

        print(
            "LLM returned an empty response."
        )

        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "abstained": True,
            "grounded": False,
            "reason": "empty_llm_response",
        }

    print("=" * 70)
    print("LLM GENERATION COMPLETE")
    print("=" * 70)

    return {
        "answer": answer,
        "abstained": False,
        "reason": "llm_generated",
    }


# =========================================================
# CITATION VALIDATION
# =========================================================

def citation_node(
    state: RAGState,
):

    answer = (
        state.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    documents = state.get(
        "documents",
        [],
    )

    print("=" * 70)
    print("CITATION VALIDATION")
    print(
        f"Answer length: {len(answer)}"
    )
    print(
        f"Documents: {len(documents)}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Explicit LLM abstention
    # -----------------------------------------------------

    if is_abstention(answer):

        print(
            "LLM explicitly abstained."
        )

        return {
            "grounded": False,
            "abstained": True,
            "reason": "llm_abstention",
            "citation_check": {},
        }

    # -----------------------------------------------------
    # Validate citations
    # -----------------------------------------------------

    try:

        result = validate_citations(
            answer,
            documents,
        )

    except Exception as exc:

        print("=" * 70)
        print("CITATION VALIDATION ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 70)

        return {
            "grounded": False,
            "abstained": True,
            "answer": "INSUFFICIENT_EVIDENCE",
            "reason": (
                "citation_validation_error:"
                f"{type(exc).__name__}"
            ),
            "citation_check": {},
        }

    # -----------------------------------------------------
    # No citations
    # -----------------------------------------------------

    if result["citation_count"] == 0:

        print(
            "No citations found."
        )

        return {
            "grounded": False,
            "abstained": True,
            "answer": "INSUFFICIENT_EVIDENCE",
            "reason": "missing_citations",
            "citation_check": result,
        }

    # -----------------------------------------------------
    # Invalid citations
    # -----------------------------------------------------

    if result["invalid_citations"]:

        print(
            "Invalid citations detected."
        )

        return {
            "grounded": False,
            "abstained": True,
            "answer": "INSUFFICIENT_EVIDENCE",
            "reason": "invalid_citations",
            "citation_check": result,
        }

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print(
        "Citation validation successful."
    )

    return {
        "grounded": True,
        "abstained": False,
        "reason": "valid_citations",
        "citation_check": result,
    }


# =========================================================
# FINAL SAFETY GATE
# =========================================================

def final_node(
    state: RAGState,
):

    if state.get("grounded"):

        print("=" * 70)
        print("RAG REQUEST SUCCESSFUL")
        print("=" * 70)

        return {}

    print("=" * 70)
    print("FINAL SAFETY GATE: ABSTAINING")
    print("=" * 70)

    return {
        "answer": "INSUFFICIENT_EVIDENCE",
        "abstained": True,
    }


# =========================================================
# ROUTING
# =========================================================

def evidence_router(
    state: RAGState,
):

    if state.get("abstained"):

        return "final"

    return "generate"


# =========================================================
# GRAPH BUILDER
# =========================================================

def build_graph():

    print("=" * 70)
    print("Building RAG graph...")
    print("=" * 70)

    graph = StateGraph(
        RAGState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "evidence",
        evidence_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "citation",
        citation_node,
    )

    graph.add_node(
        "final",
        final_node,
    )

    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    graph.add_edge(
        START,
        "retrieve",
    )

    # -----------------------------------------------------
    # RETRIEVE -> EVIDENCE
    # -----------------------------------------------------

    graph.add_edge(
        "retrieve",
        "evidence",
    )

    # -----------------------------------------------------
    # EVIDENCE ROUTING
    # -----------------------------------------------------

    graph.add_conditional_edges(
        "evidence",
        evidence_router,
        {
            "generate": "generate",
            "final": "final",
        },
    )

    # -----------------------------------------------------
    # GENERATE -> CITATION
    # -----------------------------------------------------

    graph.add_edge(
        "generate",
        "citation",
    )

    # -----------------------------------------------------
    # CITATION -> FINAL
    # -----------------------------------------------------

    graph.add_edge(
        "citation",
        "final",
    )

    # -----------------------------------------------------
    # FINAL -> END
    # -----------------------------------------------------

    graph.add_edge(
        "final",
        END,
    )

    compiled_graph = graph.compile()

    print("=" * 70)
    print("RAG graph built successfully.")
    print("=" * 70)

    return compiled_graph