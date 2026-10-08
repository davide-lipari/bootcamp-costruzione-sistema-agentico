# src/services/alerts.py — l'unico punto del progetto che apre una segnalazione
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, ComplianceAlert, Customer


class AlertService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def apri(
        self, *, autore: str, account_id: str, motivo: str, importo: Decimal | None
    ) -> tuple[ComplianceAlert, bool]:
        """Apre la segnalazione, o restituisce quella già aperta. Il bool dice se è nuova."""
        # secondo controllo, indipendente da chi chiama: il conto è nel portafoglio di chi scrive?
        operatore = await self.session.scalar(
            select(Customer.operator)
            .join(Account, Account.customer_id == Customer.id)
            .where(Account.id == account_id)
        )
        if operatore != autore:
            raise PermissionError(f"{autore} non ha in portafoglio il conto {account_id}")

        # una pratica per conto, operatore, giorno e importo: la stessa richiesta ripetuta oggi
        # è la stessa pratica, un importo diverso è un'altra operazione
        cifra = "-" if importo is None else format(importo.normalize(), "f")  # 25000 = 25000.00
        chiave = f"{account_id}:{autore}:{date.today().isoformat()}:{cifra}"
        esistente = await self._per_chiave(chiave)
        if esistente is not None:
            return esistente, False

        alert = ComplianceAlert(
            account_id=account_id,
            opened_by=autore,
            reason=motivo,
            amount=importo,
            idempotency_key=chiave,
        )
        try:
            async with self.session.begin_nested():  # due chiamate insieme: decide il vincolo
                self.session.add(alert)
        except IntegrityError:
            vincitrice = await self._per_chiave(chiave)
            if vincitrice is None:
                raise
            return vincitrice, False
        await self.session.commit()
        return alert, True

    async def _per_chiave(self, chiave: str) -> ComplianceAlert | None:
        return await self.session.scalar(
            select(ComplianceAlert).where(ComplianceAlert.idempotency_key == chiave)
        )
