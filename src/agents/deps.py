# src/agents/deps.py — i servizi che i tool usano, costruiti per ogni richiesta
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.repos import AccountRepository, MovementRepository
from src.db.runs import RunRepository
from src.db.session import get_db
from src.llm.embedding_client import EmbeddingClient
from src.llm.factory import get_embedder, get_openai
from src.services.alerts import AlertService
from src.services.retrieval_service import RetrievalService


@dataclass(frozen=True)
class Deps:
    accounts: AccountRepository
    movements: MovementRepository
    alerts: AlertService
    runs: RunRepository  # Giorno 8: dove un run sospeso aspetta
    retrieval: RetrievalService
    embedder: EmbeddingClient
    openai: AsyncOpenAI  # l'SDK, direttamente: il ciclo di oggi è scritto a mano, campo per campo
    model: str


def crea_deps(
    db: AsyncSession, *, embedder: EmbeddingClient | None = None, openai: AsyncOpenAI | None = None
) -> Deps:
    """I servizi su una sessione. La usano l'endpoint e, al Giorno 8, il server MCP.

    I due client si possono passare da fuori — è quello che fanno i test, con i finti — e
    altrimenti arrivano dalla factory, solo allora: senza chiave, un 503 che dice quale manca.
    """
    return Deps(
        accounts=AccountRepository(db),
        movements=MovementRepository(db),
        alerts=AlertService(db),
        runs=RunRepository(db),
        retrieval=RetrievalService(db),
        embedder=embedder or get_embedder(),  # dalla factory del Giorno 4: uno per processo
        openai=openai or get_openai(),
        model=settings.default_model,
    )


async def get_deps(db: Annotated[AsyncSession, Depends(get_db)]) -> Deps:
    return crea_deps(db)
