# src/llm/embedding_client.py — i vettori, dallo stesso modello per documenti e domande
import asyncio
from functools import cache
from typing import TYPE_CHECKING, cast

import openai
from openai import AsyncOpenAI

from src.exceptions import LLMProviderError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingClient:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client  # lo stesso della factory: un pool di connessioni per processo
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """I vettori di più testi in una chiamata, nello stesso ordine dei testi."""
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts)
        except openai.APIError as e:  # come la chat: un guasto del fornitore è un 502, non un bug
            raise LLMProviderError("openai", str(e)) from e
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


@cache
def _modello(nome: str) -> "SentenceTransformer":
    """Il modello scaricato una volta sola: la prima codifica paga il caricamento."""
    from sentence_transformers import SentenceTransformer

    return cast("SentenceTransformer", SentenceTransformer(nome, trust_remote_code=True))


def _codifica(nome: str, testi: list[str]) -> list[list[float]]:
    vettori = _modello(nome).encode(testi, convert_to_numpy=True)
    return cast("list[list[float]]", vettori.tolist())


class EmbeddingClientLocale(EmbeddingClient):
    """La stessa interfaccia, con un sentence-transformers girato in locale.

    Nessuna chiave e nessuna chiamata di rete: il vettore esce dal modello già sul disco.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # encode è bloccante: in un thread, o il server si ferma durante la codifica
        return await asyncio.to_thread(_codifica, self.model, texts)

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
