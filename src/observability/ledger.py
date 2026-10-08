# src/observability/ledger.py — il registro dei costi: una riga per ogni risposta che ha speso
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import LlmCall


class CostLedger:
    """Aggiunge la riga nella transazione di chi chiama: la conferma chi conferma il resto."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def aggiungi(
        self,
        *,
        endpoint: str,
        username: str,
        model: str,
        cost_eur: Decimal,
        tokens: int = 0,
        run_id: str | None = None,
    ) -> None:
        if cost_eur <= 0 and tokens == 0:
            return  # nessuna chiamata al modello (un rifiuto, un finto): niente da registrare
        self.session.add(
            LlmCall(
                endpoint=endpoint,
                username=username,
                model=model,
                tokens=tokens,
                cost_eur=cost_eur,
                run_id=run_id,
            )
        )
