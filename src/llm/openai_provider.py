# src/llm/openai_provider.py — il contratto, tradotto nella lingua di OpenAI
from collections.abc import AsyncIterator
from decimal import Decimal

import openai
from openai import AsyncOpenAI, omit
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from src.exceptions import LLMProviderError
from src.llm.types import LLMResponse, Message, ToolCall, ToolSpec

# EUR per 1000 token, (input, output). È una fotografia del listino: va aggiornata con una data.
PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o-mini": (Decimal("0.00014"), Decimal("0.00056")),
    "gpt-4o": (Decimal("0.0023"), Decimal("0.0091")),
    # Groq, listino di openai/gpt-oss-120b (0,10 e 0,50 USD per milione) in euro
    "openai/gpt-oss-120b": (Decimal("0.00009"), Decimal("0.00046")),
}


def _messaggi(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    tradotti: list[ChatCompletionMessageParam] = []
    for m in messages:
        if m.role == "system":
            tradotti.append({"role": "system", "content": m.content})
        elif m.role == "user":
            tradotti.append({"role": "user", "content": m.content})
        elif m.role == "tool":  # il risultato di uno strumento, agganciato alla sua richiesta
            tradotti.append(
                {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content}
            )
        elif m.tool_calls:  # l'assistente che ha chiesto strumenti: rientra così com'era
            chiamate: list[ChatCompletionMessageFunctionToolCallParam] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in m.tool_calls
            ]
            tradotti.append({"role": "assistant", "content": m.content, "tool_calls": chiamate})
        else:
            tradotti.append({"role": "assistant", "content": m.content})
    return tradotti


def _strumenti(tools: list[ToolSpec]) -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OpenAIProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        if model not in PRICING:  # un modello senza prezzo è un costo che nessuno conta
            raise ValueError(f"Nessun prezzo per {model}: aggiungilo a PRICING")
        self._client = client  # creato una volta, dalla factory, e iniettato
        self._model = model

    def _risposta(
        self, testo: str, usage: CompletionUsage | None, chiamate: list[ToolCall] | None = None
    ) -> LLMResponse:
        entrata, uscita = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        prezzo_in, prezzo_out = PRICING[self._model]
        return LLMResponse(
            content=testo,
            tokens_used=entrata + uscita,
            cost_eur=(entrata * prezzo_in + uscita * prezzo_out) / 1000,
            model=self._model,
            tool_calls=chiamate or [],
        )

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        try:
            r = await self._client.chat.completions.create(
                model=self._model,
                messages=_messaggi(messages),
                max_tokens=max_tokens,
                temperature=0.3,
                tools=_strumenti(tools) if tools else omit,
            )
        except openai.APIError as exc:  # dopo i tentativi dell'SDK: il guasto è del fornitore
            raise LLMProviderError("openai", str(exc)) from exc
        messaggio = r.choices[0].message
        chiamate = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
            for c in messaggio.tool_calls or []
            if c.type == "function"
        ]
        return self._risposta(messaggio.content or "", r.usage, chiamate)

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        testo: list[str] = []
        usage: CompletionUsage | None = None
        try:
            pezzi = await self._client.chat.completions.create(
                model=self._model,
                messages=_messaggi(messages),
                max_tokens=max_tokens,
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},  # senza, il consumo non arriva
            )
            async for pezzo in pezzi:
                if pezzo.usage is not None:  # l'ultimo pezzo: il consumo, e nessuna scelta
                    usage = pezzo.usage
                if pezzo.choices and pezzo.choices[0].delta.content:
                    testo.append(pezzo.choices[0].delta.content)
                    yield pezzo.choices[0].delta.content
        except openai.APIError as exc:
            raise LLMProviderError("openai", str(exc)) from exc
        yield self._risposta("".join(testo), usage)
