# src/llm/tools.py — lo strumento: lo schema che il modello vede, e il codice che solo tu esegui
import logging
from collections.abc import Awaitable, Callable
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError

from src.llm.types import ToolSpec

logger = logging.getLogger(__name__)

TOOLS_SCHEMA: list[ToolSpec] = [
    ToolSpec(
        name="get_account_balance",
        description=(
            "Restituisce il saldo disponibile di un conto di un cliente dell'operatore "
            "che sta parlando. Usalo quando la domanda riguarda quanto c'è su un conto o "
            "la capienza per un'operazione. Non usarlo per l'elenco dei movimenti."
        ),
        parameters={
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "L'IBAN del conto. Se non l'hai, chiedilo all'utente.",
                }
            },
            "required": ["account_id"],
        },
    )
]

Esecutore = Callable[[str], Awaitable[str]]  # riceve gli argomenti grezzi, restituisce il testo


class SaldoArgs(BaseModel):
    account_id: str = Field(min_length=5, max_length=34)


def costruisci_strumenti(
    operatore: str,
    conti_del_portafoglio: set[str],
    leggi_saldo: Callable[[str], Awaitable[Decimal]],
) -> dict[str, Esecutore]:
    """Gli strumenti di QUESTA richiesta: l'operatore non è un argomento, è nel contesto."""

    async def saldo(argomenti: str) -> str:
        try:  # 1. la forma: gli argomenti sono testo prodotto da un modello
            args = SaldoArgs.model_validate_json(argomenti)
        except ValidationError:
            return "ERRORE: serve account_id, l'IBAN del conto."
        if args.account_id not in conti_del_portafoglio:  # 2. il diritto, nel codice
            logger.warning(
                "strumento_negato",
                extra={
                    "tool": "get_account_balance",
                    "operatore": operatore,
                    "conto": args.account_id,
                },
            )
            return "ERRORE: conto non disponibile per questo operatore."
        importo = await leggi_saldo(args.account_id)  # 3. solo allora, il dato
        return f"Saldo del conto {args.account_id}: {importo:.2f} EUR"

    return {"get_account_balance": saldo}
