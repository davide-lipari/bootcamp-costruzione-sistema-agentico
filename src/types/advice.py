# src/types/advice.py — il contratto dell'advisor e dell'ingestione
from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    document_id: str = Field(..., max_length=100)
    content: str = Field(..., min_length=10)
    metadata: dict[str, str] | None = None
    visibility: Literal["public", "internal", "risk_only", "compliance_only"] = "public"


class IngestResponse(BaseModel):
    chunk_count: int
    embedding_dim: int


class AdviceRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    excerpt: str
    similarity: float


class AdviceResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tokens_used: int
    cost_eur: float  # la sola generazione: la riscrittura la conta il log del Giorno 9
    rewritten_query: str | None = None  # Giorno 6: cosa è stato cercato al posto della domanda
