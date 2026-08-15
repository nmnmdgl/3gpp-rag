from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    document: str
    clause: str


class Source(BaseModel):
    document: str
    clause: str
    title: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    abstained: bool
    reason: str
    citations: List[Citation] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    conversation_id: Optional[str] = None
