# src/agents/tools.py — i cinque tool, costruiti per l'operatore della richiesta
import logging
from decimal import Decimal

from pydantic import BaseModel, Field

from src.agents.deps import Deps
from src.agents.registry import Tool, impronta
from src.auth.deps import UserContext
from src.config import settings

logger = logging.getLogger(__name__)

CONTO = "L'IBAN di un conto del cliente, come lo restituisce find_customer_accounts."
NON_DISPONIBILE = "Non risulta nel portafoglio di questo operatore."
NESSUN_MOVIMENTO = "Nessun movimento per questo conto."
NIENTE_NEI_DOCUMENTI = "Niente nei documenti visibili a questo ruolo."
SENZA_DATI = frozenset({NON_DISPONIBILE, NESSUN_MOVIMENTO, NIENTE_NEI_DOCUMENTI})


class ClienteArgs(BaseModel):
    customer_id: str = Field(
        min_length=1, description="Il codice cliente, come compare nella sua scheda."
    )


class SaldoArgs(BaseModel):
    account_id: str = Field(min_length=1, description=CONTO)


class MovimentiArgs(BaseModel):
    account_id: str = Field(min_length=1, description=CONTO)
    n: int = Field(default=5, ge=1, le=20, description="Quanti movimenti, dal più recente.")


class RicercaArgs(BaseModel):
    query: str = Field(
        min_length=3, description="La domanda in forma completa, non una parola sola."
    )


class SegnalazioneArgs(BaseModel):
    account_id: str = Field(min_length=1, description=CONTO)
    motivo: str = Field(
        min_length=10,
        max_length=500,
        description="Perché il caso va segnalato, in una o due frasi.",
    )
    importo: Decimal | None = Field(
        default=None, gt=0, description="L'importo dell'operazione, se c'è."
    )


def build_tools_for(user: UserContext, deps: Deps) -> list[Tool]:
    """Costruisce i tool già legati a questo operatore. Fuori da qui l'identità non passa."""

    def rifiuta(tool: str, richiesto: str) -> str:
        # il tentativo si registra: un cliente altrui chiesto cinque volte è qualcuno che prova.
        # Con l'impronta, non con il valore: cinque tentativi uguali hanno la stessa
        logger.warning(
            "tool_accesso_negato",
            extra={"tool": tool, "username": user.username, "requested": impronta(richiesto)},
        )
        return NON_DISPONIBILE

    async def conti_del_cliente(a: ClienteArgs) -> str:
        # `user` non è un parametro di questa funzione: è nella closure, e il modello
        # non ha modo di passarne un altro. È tutta qui la lezione del giorno.
        conti = await deps.accounts.of_customer(user.username, a.customer_id)
        if not conti:
            return rifiuta("find_customer_accounts", a.customer_id)
        righe = "\n".join(f"- {c.id} ({c.label})" for c in conti)
        return f"Conti del cliente {a.customer_id}:\n{righe}"

    async def saldo(a: SaldoArgs) -> str:
        conto = await deps.accounts.of_user(user.username, a.account_id)
        if conto is None:
            return rifiuta("get_account_balance", a.account_id)
        return f"Saldo di {conto.id}: {conto.balance:.2f} EUR"

    async def movimenti(a: MovimentiArgs) -> str:
        if await deps.accounts.of_user(user.username, a.account_id) is None:
            return rifiuta("list_recent_movements", a.account_id)
        righe = await deps.movements.recent(user.username, a.account_id, limite=a.n)
        if not righe:
            return NESSUN_MOVIMENTO
        return "\n".join(f"{r.booking_date:%d/%m} {r.description} {r.amount:+.2f}" for r in righe)

    async def documenti(a: RicercaArgs) -> str:
        # il retrieval del Giorno 6, con l'ACL del ruolo: non è un tool nuovo
        passaggi = await deps.retrieval.search_for_user(
            await deps.embedder.embed_one(a.query), user.role, top_k=3
        )
        if not passaggi:
            return NIENTE_NEI_DOCUMENTI
        return "\n\n".join(f"[{r.document_id}] {r.content[:300]}" for r in passaggi)

    async def segnalazione(a: SegnalazioneArgs) -> str:
        # il solo tool che SCRIVE: il muro vale anche qui, prima di scrivere
        if await deps.accounts.of_user(user.username, a.account_id) is None:
            return rifiuta("apri_segnalazione_compliance", a.account_id)
        alert, nuova = await deps.alerts.apri(
            autore=user.username, account_id=a.account_id, motivo=a.motivo, importo=a.importo
        )
        if not nuova:
            return (
                f"La segnalazione {alert.id} su questo conto è già aperta da oggi: nessuna "
                "pratica nuova. Riferisci all'utente questo numero e non riaprirla."
            )
        return (
            f"Segnalazione {alert.id} aperta. Riferisci all'utente questo numero di pratica "
            "e che la Compliance la prenderà in carico; non riaprirla."
        )

    return [
        Tool(
            "find_customer_accounts",
            "I conti di un cliente del portafoglio, con IBAN ed etichetta. Usalo per primo "
            "quando la domanda nomina un cliente e non un IBAN; poi usa gli altri tool sul "
            "conto giusto.",
            ClienteArgs,
            conti_del_cliente,
            scrive=False,
            senza_dati=SENZA_DATI,
        ),
        Tool(
            "get_account_balance",
            "Saldo disponibile di un conto. Usalo quando la domanda riguarda quanto c'è su un "
            "conto o se basta per un'operazione. Per le operazioni già fatte usa "
            "list_recent_movements.",
            SaldoArgs,
            saldo,
            scrive=False,
            senza_dati=SENZA_DATI,
        ),
        Tool(
            "list_recent_movements",
            "Ultimi movimenti di un conto. Usalo quando la domanda riguarda operazioni già "
            "fatte: bonifici del mese, addebiti, pagamenti ricorrenti. Per la disponibilità "
            "usa get_account_balance.",
            MovimentiArgs,
            movimenti,
            scrive=False,
            senza_dati=SENZA_DATI,
        ),
        Tool(
            "search_documents",
            "Usa questo tool quando la domanda riguarda regole, soglie, procedure o adempimenti "
            "della banca. Non usarlo per dati di clienti, conti o movimenti: per quelli "
            "esistono find_customer_accounts, get_account_balance e list_recent_movements.",
            RicercaArgs,
            documenti,
            scrive=False,
            senza_dati=SENZA_DATI,
        ),
        Tool(
            "apri_segnalazione_compliance",
            "Apre una segnalazione alla Compliance su un conto del portafoglio. Usalo solo se "
            "l'utente lo chiede o se la policy recuperata la rende obbligatoria.",
            SegnalazioneArgs,
            segnalazione,
            # Giorno 8: sopra soglia, o senza importo, decide una persona
            serve_approvazione=_sopra_soglia,
            senza_dati=SENZA_DATI,
        ),
    ]


def _sopra_soglia(a: SegnalazioneArgs) -> bool:
    """Un importo che non c'è non è un importo piccolo: nel dubbio si chiede."""
    return a.importo is None or a.importo > settings.soglia_approvazione_eur
