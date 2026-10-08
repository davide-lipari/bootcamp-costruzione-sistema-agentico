# src/observability/cost_tracker.py — Giorno 9: il tetto del giorno, letto dal registro dei costi
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import func, select

from src.db.models import LlmCall
from src.db.repos import ChatRepository
from src.exceptions import RateLimitError


class CostTracker:
    def __init__(self, repo: ChatRepository, tetto_eur: Decimal) -> None:
        self.repo = repo
        self.tetto_eur = tetto_eur

    async def speso(self, giorno: date | None = None) -> Decimal:
        # None, e il giorno calcolato qui dentro: un default date.today() nella firma si
        # valuterebbe una volta sola, all'import, e il processo conterebbe per sempre quel giorno
        giorno = giorno or datetime.now(UTC).date()
        inizio = datetime.combine(giorno, time.min, tzinfo=UTC)
        # Giorno 9: dal registro, dove scrivono chat, advice, agente e categorizzazione. Fino a
        # ieri si sommava la sola chat, l'unico posto dove un costo si salvava
        totale = await self.repo.session.scalar(
            select(func.coalesce(func.sum(LlmCall.cost_eur), 0)).where(
                LlmCall.created_at >= inizio, LlmCall.created_at < inizio + timedelta(days=1)
            )
        )
        return totale if totale is not None else Decimal("0")  # il coalesce lo esclude già

    async def verifica(self) -> None:
        """Da chiamare PRIMA del modello: dopo, la spesa è già fatta."""
        if await self.speso() >= self.tetto_eur:
            adesso = datetime.now(UTC)
            mezzanotte = datetime.combine(adesso.date() + timedelta(days=1), time.min, tzinfo=UTC)
            raise RateLimitError(int((mezzanotte - adesso).total_seconds()) + 1)
