from typing import TypedDict, List, Dict

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
# COMPONENTS
# =========================================================

retriever = Retriever(
    dense_k=50,
    bm25_k=50,
    final_k=50,
    rerank_k=20,
)

llm = get_llm()


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_node(
    state: RAGState,
):

    question = state["question"]

    documents = retriever.retrieve(
        question
    )

    context = format_context(
        documents,
        max_documents=12,
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

    # Only reject if retrieval returned nothing.
    #
    # Do NOT reject based on reranker score here.
    #
    # The LLM + citation validator are responsible for deciding
    # whether the retrieved evidence actually supports an answer.

    if not documents:

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

        return {}

    prompt = USER_PROMPT.format(
        question=state["question"],
        context=state["context"],
    )

    try:

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

    if not answer:

        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "abstained": True,
            "grounded": False,
            "reason": "empty_llm_response",
        }

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

    # -----------------------------------------------------
    # Explicit LLM abstention
    # -----------------------------------------------------

    if is_abstention(answer):

        return {
            "grounded": False,
            "abstained": True,
            "reason": "llm_abstention",
            "citation_check": {},
        }

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    result = validate_citations(
        answer,
        documents,
    )

    # -----------------------------------------------------
    # No citations
    # -----------------------------------------------------

    if result["citation_count"] == 0:

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

        return {}

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
# GRAPH
# =========================================================

def build_graph():

    graph = StateGraph(
        RAGState
    )

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

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "evidence",
    )

    graph.add_conditional_edges(
        "evidence",
        evidence_router,
        {
            "generate": "generate",
            "final": "final",
        },
    )

    graph.add_edge(
        "generate",
        "citation",
    )

    graph.add_edge(
        "citation",
        "final",
    )

    graph.add_edge(
        "final",
        END,
    )

    return graph.compile()