# src/api/admin.py — il rapporto sui costi: chi spende, dove, e quali esecuzioni costano di più
import datetime as dt
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from src.auth.deps import UserContext, require_role
from src.db.models import LlmCall
from src.db.session import get_db

router = APIRouter(prefix="/api/admin", tags=["Admin"])


class Voce(BaseModel):
    chiave: str  # il modello, l'utente, l'endpoint o il run
    chiamate: int
    token: int
    costo_eur: Decimal


class CostReport(BaseModel):
    dal: dt.date
    totale_eur: Decimal
    per_modello: list[Voce]
    per_utente: list[Voce]
    per_endpoint: list[Voce]
    run_piu_cari: list[Voce]


async def _per(
    session: AsyncSession,
    colonna: InstrumentedAttribute[str] | InstrumentedAttribute[str | None],
    dal: dt.datetime,
    limite: int = 50,
) -> list[Voce]:
    """La GROUP BY del Giorno 3, su una colonna: chiamate, token e costo, dal più caro."""
    stmt = (
        select(colonna, func.count(), func.sum(LlmCall.tokens), func.sum(LlmCall.cost_eur))
        .where(LlmCall.created_at >= dal, colonna.is_not(None))
        .group_by(colonna)
        .order_by(func.sum(LlmCall.cost_eur).desc())
        .limit(limite)
    )
    return [
        Voce(chiave=str(k), chiamate=n, token=int(t or 0), costo_eur=c or Decimal("0"))
        for k, n, t, c in (await session.execute(stmt)).all()
    ]


@router.get("/cost-report", response_model=CostReport)
async def cost_report(
    dal: dt.date,
    user: Annotated[UserContext, Depends(require_role("risk_lead", "admin"))],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CostReport:
    """Quanto costa il sistema: un'informazione riservata, quindi un ruolo che la può vedere."""
    # `dal` è un giorno UTC, come `created_at`: in Italia comincia all'una, o alle due d'estate
    inizio = dt.datetime.combine(dal, dt.time.min, tzinfo=dt.UTC)
    totale = await session.scalar(
        select(func.coalesce(func.sum(LlmCall.cost_eur), 0)).where(LlmCall.created_at >= inizio)
    )
    return CostReport(
        dal=dal,
        totale_eur=Decimal(totale or 0),
        per_modello=await _per(session, LlmCall.model, inizio),
        per_utente=await _per(session, LlmCall.username, inizio),
        per_endpoint=await _per(session, LlmCall.endpoint, inizio),
        run_piu_cari=await _per(session, LlmCall.run_id, inizio, limite=5),
    )
