# tests/finti.py — i modelli finti: a livello HTTP, dove gli SDK lavorano davvero, e a eco
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any

import httpx2
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.auth.deps import UserContext
from src.llm.embedding_client import EmbeddingClient
from src.llm.rewriter import QueryRewriter
from src.llm.types import LLMResponse, Message, ToolSpec
from src.services.categorize_rules import categorizza
from src.types.categorize import CategorizeRequest, CategorizeResponse

Gestore = Callable[[httpx2.Request], httpx2.Response]


class Registro:
    """Le richieste arrivate al finto, con il corpo JSON già letto."""

    def __init__(self) -> None:
        self.richieste: list[dict[str, Any]] = []
        self.percorsi: list[str] = []

    def annota(self, request: httpx2.Request) -> dict[str, Any]:
        corpo: dict[str, Any] = json.loads(request.content or b"{}")
        self.richieste.append(corpo)
        self.percorsi.append(request.url.path)
        return corpo


def openai_finto(gestore: Gestore, max_retries: int = 2) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="sk-test",
        base_url="http://finto/v1",
        max_retries=max_retries,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(gestore)),
    )


def anthropic_finto(gestore: Gestore) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="sk-ant-test",
        base_url="http://finto",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(gestore)),
    )


def completion(
    testo: str | None,
    *,
    prompt: int = 100,
    uscita: int = 20,
    tool_calls: list[dict[str, Any]] | None = None,
) -> httpx2.Response:
    messaggio: dict[str, Any] = {"role": "assistant", "content": testo}
    if tool_calls:
        messaggio["tool_calls"] = tool_calls
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": messaggio,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": uscita,
                "total_tokens": prompt + uscita,
            },
        },
    )


def chiamata(id_: str, nome: str, argomenti: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": id_,
        "type": "function",
        "function": {"name": nome, "arguments": json.dumps(argomenti)},
    }


def stream_openai(pezzi: list[str], *, prompt: int = 80, uscita: int = 12) -> httpx2.Response:
    righe = []
    for p in pezzi:
        righe.append(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}],
            }
        )
    righe.append(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [],  # il pezzo del consumo: nessuna scelta
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": uscita,
                "total_tokens": prompt + uscita,
            },
        }
    )
    corpo = "".join(f"data: {json.dumps(r)}\n\n" for r in righe) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=corpo, headers={"content-type": "text/event-stream"})


def messaggio_anthropic(
    testo: str,
    *,
    entrata: int = 50,
    uscita: int = 10,
    strumenti: list[tuple[str, str, dict[str, Any]]] | None = None,
) -> httpx2.Response:
    """Una risposta di Anthropic: il testo, e i blocchi tool_use (id, nome, argomenti)."""
    blocchi: list[dict[str, Any]] = [{"type": "text", "text": testo}] if testo else []
    blocchi += [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in strumenti or []]
    return httpx2.Response(
        200,
        json={
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": blocchi,
            "stop_reason": "tool_use" if strumenti else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": entrata, "output_tokens": uscita},
        },
    )


def stream_anthropic(pezzi: list[str], *, entrata: int = 50, uscita: int = 10) -> httpx2.Response:
    eventi: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5-20251001",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": entrata, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for p in pezzi:
        eventi.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": p},
                },
            )
        )
    eventi += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": uscita},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    corpo = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in eventi)
    return httpx2.Response(200, text=corpo, headers={"content-type": "text/event-stream"})


class ModelloEco:
    """Il modello dei test che non gli chiedono niente di preciso: ripete la domanda, gratis."""

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content=f"Echo: {messages[-1].content}",
            tokens_used=0,
            cost_eur=Decimal("0"),
            model="eco",
        )

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        risposta = await self.complete(messages, max_tokens)
        yield risposta.content
        yield risposta


class CategorizzazioneARegole:
    """La categorizzazione dei test: la regola del Giorno 2, che non costa e non cambia."""

    async def categorize(self, req: CategorizeRequest) -> CategorizeResponse:
        return categorizza(req)


def vettore(testo: str) -> list[float]:
    """Un embedding finto ma coerente: le parole in comune avvicinano i testi."""
    v = [0.0] * 1536
    for parola in re.findall(r"\w{3,}", testo.lower()):
        v[int(hashlib.md5(parola.encode()).hexdigest(), 16) % 1536] += 1.0
    norma = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norma for x in v]


def gestore_embedding(r: httpx2.Request) -> httpx2.Response:
    """L'API degli embedding, finta: un vettore per ogni testo, nell'ordine in cui arrivano."""
    testi: list[str] = json.loads(r.content)["input"]
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": i, "embedding": vettore(t)}
                for i, t in enumerate(testi)
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
    )


def embedder_finto() -> EmbeddingClient:
    return EmbeddingClient(openai_finto(gestore_embedding), "text-embedding-3-small")


class ModelloFisso:
    """Risponde con un testo scelto dal test, e ricorda cosa ha ricevuto."""

    def __init__(self, risposta: str) -> None:
        self.risposta = risposta
        self.ricevuti: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.ricevuti.append(messages)
        return LLMResponse(
            content=self.risposta, tokens_used=100, cost_eur=Decimal("0.0001"), model="finto"
        )

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        yield await self.complete(messages, max_tokens)


MARCO = UserContext(username="mbianchi", role="operator")  # chi chiede, nei test senza token
GIULIA = UserContext(username="grossi", role="compliance_lead")


def riscrittore_spento() -> QueryRewriter:
    """Il riscrittore dei test che non parlano di riscrittura: la domanda passa com'è."""
    return QueryRewriter(ModelloEco(), "", enabled=False)
