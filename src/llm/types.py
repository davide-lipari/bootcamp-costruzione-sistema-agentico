# src/llm/types.py — i contratti di dato: i messaggi, gli strumenti, la risposta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Uno strumento, nella forma che conosce il tuo codice: ogni provider la traduce nella sua."""

    name: str
    description: str  # l'unica cosa che il modello legge per decidere se usarlo
    parameters: dict[str, Any]  # lo schema JSON degli argomenti


class ToolCall(BaseModel):
    """Il modello chiede di eseguire uno strumento. Non lo esegue: lo chiede."""

    id: str
    name: str
    arguments: str  # JSON prodotto dal modello: testo non fidato, da validare prima di usarlo


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)  # l'assistente che chiede strumenti
    tool_call_id: str | None = None  # il risultato di uno strumento: a quale richiesta risponde


class LLMResponse(BaseModel):
    content: str
    tokens_used: int
    cost_eur: Decimal  # denaro: decimale, come la colonna del Giorno 3
    model: str
    tool_calls: list[ToolCall] = Field(default_factory=list)  # vuota: il modello ha risposto
