# src/types/glossary.py
from pydantic import BaseModel, Field


class GlossaryRequest(BaseModel):
    term: str = Field(..., min_length=2, max_length=60, description="Il termine da spiegare")


class GlossaryResponse(BaseModel):
    term: str
    definition: str
    source: str
