# src/db/runs.py — lo stato dei run sospesi: si scrive, si rilegge, si decide una volta sola
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AgentRunState


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def sospendi(
        self,
        *,
        run_id: str,
        username: str,
        role: str,
        messages: list[dict[str, Any]],
        pending_calls: list[dict[str, Any]],
        description: str,
        steps: int,
        cost_eur: Decimal,
        tool_calls: list[str],
    ) -> None:
        """Salva il run in attesa. `merge`: lo stesso run può fermarsi più di una volta.

        `decisions` non è fra i campi: il merge non lo tocca, e le firme di prima restano.
        """
        await self.session.merge(
            AgentRunState(
                id=run_id,
                username=username,
                role=role,
                status="awaiting_approval",
                messages=messages,
                pending_calls=pending_calls,
                description=description,
                steps=steps,
                cost_eur=cost_eur,
                tool_calls=tool_calls,
                decided_by=None,
                decision_reason=None,
            )
        )
        await self.session.commit()

    async def get(self, run_id: str) -> AgentRunState | None:
        return await self.session.get(AgentRunState, run_id)

    async def decidi(self, run_id: str, *, da: str, stato: str, motivo: str | None) -> bool:
        """Esce dall'attesa, solo se era in attesa. Il bool dice se questa decisione ha vinto.

        È un UPDATE condizionato, non un «leggi e poi scrivi»: con due clic insieme uno solo
        trova la riga in attesa, e l'altro riceve False.
        """
        esito = await self.session.execute(
            update(AgentRunState)
            .where(AgentRunState.id == run_id, AgentRunState.status == "awaiting_approval")
            .values(
                status=stato, decided_by=da, decision_reason=motivo, updated_at=datetime.now(UTC)
            )
        )
        vinta = bool(getattr(esito, "rowcount", 0))
        if vinta:
            # nello storico, nella stessa transazione: chi ha deciso cosa, prima che l'azione parta
            riga = await self.session.get(AgentRunState, run_id, populate_existing=True)
            if riga is not None:
                decisione = {
                    "da": da,
                    "esito": "approvata" if stato == "running" else "respinta",
                    "motivo": motivo,
                    "azione": riga.description,
                    "quando": datetime.now(UTC).isoformat(),
                }
                riga.decisions = [*riga.decisions, decisione]  # una lista nuova: il JSON cambia
        await self.session.commit()
        return vinta

    async def chiudi(self, run_id: str, stato: str) -> None:
        await self.session.execute(
            update(AgentRunState)
            .where(AgentRunState.id == run_id)
            .values(status=stato, updated_at=datetime.now(UTC))
        )
        await self.session.commit()
