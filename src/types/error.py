# src/types/error.py — la busta unica di ogni errore
from datetime import datetime

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    timestamp: datetime
    status: int
    error: str  # il codice stabile, per il client: NOT_FOUND, RATE_LIMIT…
    message: str  # la frase, per una persona
    path: str
    details: list[str] | None = None
