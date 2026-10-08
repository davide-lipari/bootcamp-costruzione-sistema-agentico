# src/llm/client.py — il contratto di comportamento
from collections.abc import AsyncIterator
from typing import Protocol

from src.llm.types import LLMResponse, Message, ToolSpec


class LLMProvider(Protocol):
    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        """La risposta intera, con il suo consumo e il suo costo; o le richieste di strumento."""
        ...

    def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        """I pezzi di testo man mano che arrivano; per ultima la LLMResponse, con il costo."""
        ...
