# src/services/rag_service.py — la pipeline a fasi: riscrivi, cerca per ruolo, genera, misura
import asyncio
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal

from src.auth.deps import UserContext
from src.exceptions import LLMProviderError
from src.llm.client import LLMProvider
from src.llm.embedding_client import EmbeddingClient
from src.llm.rewriter import QueryRewriter
from src.llm.types import Message
from src.services.retrieval_service import RetrievalResult, RetrievalService
from src.types.advice import AdviceResponse, Citation

logger = logging.getLogger(__name__)

RIFIUTO = "Non ho trovato informazioni sufficienti nei documenti disponibili."
MARCATORE = re.compile(r"\[fonte-(\d+)\]")


def contesto(passaggi: list[RetrievalResult]) -> str:
    """I passaggi etichettati: senza etichette, il modello non sa quale citare."""
    return "\n\n".join(
        f"[fonte-{i}] (documento: {p.document_id})\n{p.content}"
        for i, p in enumerate(passaggi, start=1)
    )


def citazioni(risposta: str, passaggi: list[RetrievalResult]) -> list[Citation]:
    """Le fonti che la risposta cita davvero, una volta ciascuna, nell'ordine in cui compaiono."""
    usati: list[Citation] = []
    visti: set[tuple[str, str]] = set()
    for numero in MARCATORE.findall(risposta):
        indice = int(numero) - 1
        if not 0 <= indice < len(passaggi):
            continue  # un marcatore inventato non diventa una fonte
        p = passaggi[indice]
        if (p.document_id, p.content) in visti:
            continue
        visti.add((p.document_id, p.content))
        usati.append(
            Citation(
                document_id=p.document_id,
                chunk_id=p.chunk_id,
                excerpt=p.content[:200],
                similarity=p.similarity,
            )
        )
    return usati


@dataclass
class Fasi:
    """Millisecondi per fase, per il log strutturato della richiesta."""

    rewrite_ms: int = 0
    embedding_ms: int = 0
    retrieval_ms: int = 0
    prompt_ms: int = 0
    llm_ms: int = 0
    used_fallback: bool = False

    @property
    def total_ms(self) -> int:
        return (
            self.rewrite_ms + self.embedding_ms + self.retrieval_ms + self.prompt_ms + self.llm_ms
        )


@dataclass
class Generazione:
    """Cosa esce dalla generazione. Quando il modello non si chiama, zero è il costo vero."""

    testo: str
    tokens_used: int = 0
    cost_eur: Decimal = Decimal("0")
    degradata: bool = False


@contextmanager
def _cronometro(fasi: Fasi, campo: str) -> Iterator[None]:
    inizio = time.perf_counter()
    try:
        yield
    finally:
        setattr(fasi, campo, int((time.perf_counter() - inizio) * 1000))


class RAGService:
    """Riscrive, recupera per ruolo, genera, e misura ogni fase."""

    TIMEOUT_GENERAZIONE_S = 10

    def __init__(
        self,
        retrieval: RetrievalService,
        embedder: EmbeddingClient,
        llm: LLMProvider,
        rewriter: QueryRewriter,
        prompt: str,
    ) -> None:
        self.retrieval = retrieval
        self.embedder = embedder
        self.llm = llm
        self.rewriter = rewriter
        self.prompt = prompt

    async def advise(self, question: str, user: UserContext) -> AdviceResponse:
        fasi = Fasi()

        with _cronometro(fasi, "rewrite_ms"):
            search_query = await self.rewriter.rewrite_cached(question)

        with _cronometro(fasi, "embedding_ms"):
            query_vec = await self.embedder.embed_one(search_query)

        with _cronometro(fasi, "retrieval_ms"):
            chunks = await self.retrieval.search_for_user(query_vec, user.role)

        if chunks:
            with _cronometro(fasi, "prompt_ms"):
                # La riscritta serve al retrieval. Al modello va la domanda dell'utente.
                messaggi = [
                    Message(role="system", content=self.prompt),
                    Message(
                        role="user",
                        content=f"CONTESTO\n{contesto(chunks)}\n\nDOMANDA\n{question}",
                    ),
                ]
            with _cronometro(fasi, "llm_ms"):
                gen = await self._generate_or_degrade(messaggi, chunks)
        else:
            # il rifiuto del Giorno 5: niente sopra la soglia, e il modello non si chiama
            gen = Generazione(RIFIUTO)
        fasi.used_fallback = gen.degradata

        logger.info(
            "advice_completata",
            extra={
                "username": user.username,
                "role": user.role,
                "chunk_count": len(chunks),
                "chunk_ids": [r.chunk_id for r in chunks],  # gli id, mai il testo
                **asdict(fasi),
                "total_ms": fasi.total_ms,
            },
        )
        return AdviceResponse(
            answer=gen.testo,
            citations=[] if RIFIUTO in gen.testo else citazioni(gen.testo, chunks),
            tokens_used=gen.tokens_used,
            cost_eur=float(gen.cost_eur),
            rewritten_query=search_query,
        )

    async def _generate_or_degrade(
        self, messaggi: list[Message], chunks: list[RetrievalResult]
    ) -> Generazione:
        try:
            async with asyncio.timeout(self.TIMEOUT_GENERAZIONE_S):
                risposta = await self.llm.complete(messaggi, max_tokens=500)
            return Generazione(risposta.content, risposta.tokens_used, risposta.cost_eur)
        except (LLMProviderError, TimeoutError):
            logger.warning("generator_non_disponibile_fallback_su_chunk")
            estratti = "\n\n".join(
                f"[fonte-{i}] {r.document_id}\n{r.content}"
                for i, r in enumerate(chunks[:3], start=1)
            )
            return Generazione(
                "⚠️ Risposta parziale: il servizio di sintesi non è momentaneamente "
                "disponibile. Di seguito i passaggi dei documenti più pertinenti "
                f"alla tua domanda.\n\n{estratti}",
                degradata=True,
            )
