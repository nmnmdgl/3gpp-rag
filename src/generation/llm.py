import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """
    Create the Groq-hosted GPT-OSS-120B model.

    The model is deliberately configured for deterministic,
    citation-grounded generation.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in the .env file."
        )

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    return ChatGroq(
        model=model,
        temperature=0,
        api_key=api_key,

        # Keep this comfortably below the Groq TPM/request budget.
        max_tokens=1200,

        # We do not need long hidden reasoning for this RAG task.
        reasoning_effort="low",
    )