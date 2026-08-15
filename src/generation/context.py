from typing import Dict, List


MAX_CONTEXT_DOCUMENTS = 8
MAX_TEXT_CHARS = 4000


def _canonical_spec(
    value: str,
) -> str:

    if not value:
        return "UNKNOWN"

    value = str(value).strip().upper()

    # Indexed document IDs.
    aliases = {
        "23501-K20": "TS 23.501",
        "23502-K20": "TS 23.502",
        "38300-FN0": "TS 38.300",
        "21905-J20": "TR 21.905",
    }

    if value in aliases:
        return aliases[value]

    return value


def _clean_text(
    text: str,
) -> str:

    if not text:
        return ""

    lines = []

    for line in str(text).splitlines():

        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    return "\n".join(lines)


def format_context(
    documents: List[Dict],
    max_documents: int = MAX_CONTEXT_DOCUMENTS,
) -> str:

    if not documents:

        return "NO_RETRIEVED_EVIDENCE"

    sections = []

    for index, doc in enumerate(
        documents[:max_documents],
        start=1,
    ):

        raw_spec = (
            doc.get("spec_number")
            or doc.get("spec")
            or doc.get("document_id")
            or "UNKNOWN"
        )

        spec = _canonical_spec(
            raw_spec
        )

        clause = (
            str(
                doc.get(
                    "clause",
                    "UNKNOWN",
                )
            )
            .strip()
        )

        title = (
            doc.get(
                "clause_title",
                "",
            )
            or ""
        )

        text = _clean_text(
            doc.get(
                "text",
                "",
            )
        )

        if len(text) > MAX_TEXT_CHARS:

            text = (
                text[
                    :MAX_TEXT_CHARS
                ].rstrip()
                + "\n[TRUNCATED]"
            )

        sections.append(
            f"""
================ EVIDENCE {index} ================

CANONICAL SPECIFICATION:
{spec}

CLAUSE:
{clause}

CLAUSE TITLE:
{title}

TEXT:
{text}

CITATION FOR THIS EVIDENCE:
[{spec} | Clause {clause}]

===================================================
"""
        )

    return "\n".join(
        sections
    )