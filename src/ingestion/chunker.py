import re
from typing import List

import tiktoken

from .models import (
    Block,
    Chunk,
    DocumentMetadata,
)


ENCODER = tiktoken.get_encoding(
    "cl100k_base"
)


# ---------------------------------------------------------------------------
# Token splitting
# ---------------------------------------------------------------------------

def split_tokens(
    text: str,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    """
    Split oversized text into token-based windows.

    This is a fallback mechanism.

    Semantic boundaries are handled BEFORE this function.
    """

    encoded = ENCODER.encode(
        text
    )

    if len(encoded) <= max_tokens:
        return [text]

    if overlap >= max_tokens:
        raise ValueError(
            "overlap must be smaller than max_tokens"
        )

    output = []

    start = 0

    while start < len(encoded):

        end = min(
            start + max_tokens,
            len(encoded),
        )

        piece = ENCODER.decode(
            encoded[start:end]
        )

        output.append(
            piece
        )

        if end == len(encoded):
            break

        start = end - overlap

    return output


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def build_header(
    meta: DocumentMetadata,
    block: Block,
) -> str:
    """
    Build a metadata header that is included in the
    embedding text.

    This significantly improves retrieval because
    specification number, clause and semantic type
    become part of the searchable representation.
    """

    lines = [
        f"Specification: {meta.spec_number}",
        f"Title: {meta.title}",
        f"Version: {meta.version}",
        f"Release: {meta.release}",
        f"Content type: {block.semantic_type}",
    ]

    if block.clause:
        lines.append(
            f"Clause: {block.clause}"
        )

    if block.clause_title:
        lines.append(
            f"Clause title: {block.clause_title}"
        )

    if block.clause_path:
        lines.append(
            "Clause path: "
            + " > ".join(
                block.clause_path
            )
        )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------------------------
# Definition chunking
# ---------------------------------------------------------------------------

def build_definition_chunk(
    meta: DocumentMetadata,
    block: Block,
) -> str:
    """
    Give vocabulary definitions a stronger retrieval representation.

    Example:

        Term: User Equipment (UE)
        Definition: ...
    """

    text = block.text.strip()

    # Try to split "TERM: definition".
    match = re.match(
        r"^\s*(.+?)\s*:\s*(.+)$",
        text,
        re.DOTALL,
    )

    if match:

        term = match.group(1).strip()
        definition = match.group(2).strip()

        return (
            f"{build_header(meta, block)}\n\n"
            f"Term: {term}\n"
            f"Definition: {definition}"
        )

    return (
        f"{build_header(meta, block)}\n\n"
        f"{text}"
    )


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def make_chunks(
    metadata: DocumentMetadata,
    blocks: List[Block],
    max_tokens: int = 700,
    overlap: int = 100,
) -> List[Chunk]:

    chunks: List[Chunk] = []

    for block in blocks:

        # ---------------------------------------------------------------
        # Build semantic text representation.
        # ---------------------------------------------------------------

        if block.semantic_type == "definition":

            full_text = (
                build_definition_chunk(
                    metadata,
                    block,
                )
            )

        else:

            prefix = build_header(
                metadata,
                block,
            )

            full_text = (
                f"{prefix}\n\n"
                f"{block.text}"
            )

        # ---------------------------------------------------------------
        # Administrative tables
        # ---------------------------------------------------------------

        # Administrative/document-history tables are still retained,
        # but we make their semantic identity explicit.
        #
        # This allows the retriever later to down-weight them if needed.
        if (
            block.semantic_type
            == "administrative_table"
        ):

            full_text = (
                f"{build_header(metadata, block)}\n\n"
                "Administrative/document-history table.\n\n"
                f"{block.text}"
            )

        # ---------------------------------------------------------------
        # Split only if necessary.
        # ---------------------------------------------------------------

        pieces = split_tokens(
            full_text,
            max_tokens=max_tokens,
            overlap=overlap,
        )

        for piece_index, piece in enumerate(
            pieces
        ):

            safe_clause = re.sub(
                r"[^A-Za-z0-9_.-]",
                "_",
                block.clause or "root",
            )

            safe_type = re.sub(
                r"[^A-Za-z0-9_.-]",
                "_",
                block.semantic_type,
            )

            chunk_id = (
                f"{metadata.document_id}"
                f"__{safe_clause}"
                f"__{safe_type}"
                f"__b{block.index}"
                f"__c{piece_index}"
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    document_id=metadata.document_id,
                    spec_number=metadata.spec_number,
                    document_type=metadata.document_type,
                    title=metadata.title,
                    version=metadata.version,
                    release=metadata.release,
                    source_file=metadata.source_file,
                    clause=block.clause,
                    clause_title=block.clause_title,
                    clause_path=block.clause_path,
                    content_type=block.block_type,
                    semantic_type=block.semantic_type,
                    block_index=block.index,
                    chunk_index=piece_index,
                    page=block.page,
                )
            )

    return chunks