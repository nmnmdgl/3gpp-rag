from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentMetadata:
    """
    Metadata describing a single 3GPP document.
    """

    document_id: str
    spec_number: str
    document_type: str

    title: str

    version: str
    release: str

    source_file: str


@dataclass
class Block:
    """
    A structurally meaningful unit extracted from a DOCX.

    A block can represent:
    - paragraph
    - definition
    - heading
    - technical table
    - administrative table
    - reference
    """

    index: int

    text: str

    block_type: str

    # Semantic classification used by the chunker/retriever.
    semantic_type: str = "paragraph"

    # Clause information.
    clause: Optional[str] = None
    clause_title: Optional[str] = None

    clause_path: List[str] = field(
        default_factory=list
    )

    # Source document position.
    page: Optional[int] = None


@dataclass
class Chunk:
    """
    Final retrieval unit.

    Every chunk retains enough metadata to produce
    precise citations later.
    """

    id: str

    text: str

    document_id: str

    spec_number: str

    document_type: str

    title: str

    version: str

    release: str

    source_file: str

    clause: Optional[str]

    clause_title: Optional[str]

    clause_path: List[str]

    # Original block type.
    content_type: str

    # Semantic classification.
    semantic_type: str

    block_index: int

    chunk_index: int

    page: Optional[int]