# src/llm/anthropic_provider.py — lo stesso contratto, tradotto nella lingua di Anthropic
import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal

import anthropic
from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    ContentBlock,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    Usage,
)

from src.exceptions import LLMProviderError
from src.llm.types import LLMResponse, Message, ToolCall, ToolSpec

# EUR per 1000 token, (input, output): la stessa fotografia del listino, da datare come l'altra
PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "claude-haiku-4-5-20251001": (Decimal("0.00092"), Decimal("0.0046")),
}


def _dividi(messages: list[Message]) -> tuple[str, list[MessageParam]]:
    """Il prompt di sistema a parte; il resto nei due soli ruoli che Anthropic ammette."""
    sistema = "\n\n".join(m.content for m in messages if m.role == "system")
    resto: list[MessageParam] = []
    risultati: list[ToolResultBlockParam] = []  # i risultati degli strumenti, raccolti
    for m in messages:
        if m.role == "tool":
            risultati.append(
                {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.content}
            )
            continue
        if risultati:  # tutti i risultati di un giro vanno in un solo messaggio dell'utente
            resto.append({"role": "user", "content": risultati})
            risultati = []
        if m.role == "user":
            resto.append({"role": "user", "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            blocchi: list[TextBlockParam | ToolUseBlockParam] = []
            if m.content:
                blocchi.append({"type": "text", "text": m.content})
            blocchi += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": json.loads(c.arguments)}
                for c in m.tool_calls
            ]
            resto.append({"role": "assistant", "content": blocchi})
        elif m.role == "assistant":
            resto.append({"role": "assistant", "content": m.content})
    if risultati:
        resto.append({"role": "user", "content": risultati})
    return sistema, resto


def _strumenti(tools: list[ToolSpec]) -> list[ToolParam]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
    ]


class AnthropicProvider:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        if model not in PRICING:
            raise ValueError(f"Nessun prezzo per {model}: aggiungilo a PRICING")
        self._client = client
        self._model = model

    def _risposta(self, blocchi: Sequence[ContentBlock], usage: Usage) -> LLMResponse:
        prezzo_in, prezzo_out = PRICING[self._model]
        return LLMResponse(
            # il testo sta nei blocchi di tipo "text", non per forza nel primo
            content="".join(b.text for b in blocchi if b.type == "text"),
            tokens_used=usage.input_tokens + usage.output_tokens,
            cost_eur=(usage.input_tokens * prezzo_in + usage.output_tokens * prezzo_out) / 1000,
            model=self._model,
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=json.dumps(b.input))
                for b in blocchi
                if b.type == "tool_use"
            ],
        )

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        sistema, resto = _dividi(messages)
        try:
            r = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,  # obbligatorio, e senza valore di riserva
                system=sistema,
                messages=resto,
                tools=_strumenti(tools) if tools else omit,
            )
        except anthropic.APIError as exc:
            raise LLMProviderError("anthropic", str(exc)) from exc
        return self._risposta(r.content, r.usage)

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        sistema, resto = _dividi(messages)
        try:
            async with self._client.messages.stream(
                model=self._model, max_tokens=max_tokens, system=sistema, messages=resto
            ) as s:
                async for testo in s.text_stream:
                    yield testo
                finale = await s.get_final_message()
        except anthropic.APIError as exc:
            raise LLMProviderError("anthropic", str(exc)) from exc
        yield self._risposta(finale.content, finale.usage)
