# evals/durata.py — una conversazione di cinquanta turni: cosa resta, quanto costa, dove si perde
import asyncio
import io
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

CLIENTE = "C-10234"
SONDA = "Di quale cliente stiamo parlando, e quali sono i suoi conti?"
# dieci domande di una mattina allo sportello, ripetute: il cliente si nomina solo nella prima
DOMANDE = [
    f"Parliamo del cliente {CLIENTE}: quali conti ha?",
    "Il saldo del conto principale?",
    "E l'altro conto?",
    "Gli ultimi cinque movimenti del conto principale.",
    "Cosa dice la policy sui bonifici verso l'estero?",
    "Grazie. Mi ridai il saldo del conto principale?",
    "Ci sono addebiti ricorrenti sul principale?",
    "E sull'altro conto, gli ultimi movimenti?",
    "Quali commissioni paga per un bonifico allo sportello?",
    SONDA,  # la risposta giusta nomina il cliente: senza memoria, non lo sa
]


@dataclass(frozen=True)
class Riga:
    turno: int
    messaggi: int  # quanti messaggi tiene lo stato dopo il turno
    cliente: str | None  # cosa dicono i fatti
    costo_eur: Decimal  # il costo cumulato della sessione
    sonda: bool | None  # sui turni sonda: la risposta nomina il cliente?


async def misura(
    conv: CompiledStateGraph[Any, Any, Any, Any], config: RunnableConfig, turni: int = 50
) -> list[Riga]:
    """Tutto dallo stato della conversazione: nessun contatore tenuto qui accanto."""
    righe = []
    for i in range(turni):
        domanda = DOMANDE[i % len(DOMANDE)]
        out = await conv.ainvoke({"messages": [HumanMessage(domanda)]}, config)
        risposta = out["messages"][-1].text
        righe.append(
            Riga(
                turno=out["turni"],
                messaggi=len(out["messages"]),
                cliente=out["fatti"]["cliente"],
                costo_eur=out["costo_eur"],
                sonda=CLIENTE in risposta if domanda == SONDA else None,
            )
        )
    return righe


def rapporto(righe: list[Riga], blocco: int = 10) -> str:
    """Una riga ogni `blocco` turni, e il costo del blocco: è lui che deve smettere di salire."""
    testo = ["turno | messaggi nello stato | cliente nei fatti | costo del blocco | costo totale"]
    precedente = Decimal("0")
    for r in righe[blocco - 1 :: blocco]:
        testo.append(
            f"{r.turno:5} | {r.messaggi:20} | {r.cliente or '-':17} | "
            f"{r.costo_eur - precedente:16.4f} | {r.costo_eur:.4f}"
        )
        precedente = r.costo_eur
    sonde = [r.sonda for r in righe if r.sonda is not None]
    testo.append(f"sonde con il cliente nella risposta: {sum(sonde)} su {len(sonde)}")
    return "\n".join(testo)


async def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    from uuid import uuid4

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from src.agents.deps import crea_deps
    from src.agents.supervisor import MODELLO_TRIAGE
    from src.auth.deps import UserContext
    from src.config import settings
    from src.db.session import AsyncSessionLocal
    from src.lc.modelli import modello_chat
    from src.lg.conversazione import costruisci_conversazione, thread_della_sessione
    from src.llm.factory import get_embedder, get_openai

    marco = UserContext(username="mbianchi", role="operator")
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_della_sessione(marco, str(uuid4()))}
    }
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=get_embedder(), openai=get_openai())
        conv = costruisci_conversazione(
            marco,
            deps,
            # cinquanta turni in fila superano i tetti del fornitore (token al minuto e al
            # giorno), che si aprono in minuti: le prove del client fanno aspettare la misura
            # al suo turno, invece di farla morire sul primo 429
            modello_chat(settings.default_model, max_retries=90),
            modello_chat(MODELLO_TRIAGE, max_retries=90),
            InMemoryStore(),
            InMemorySaver(),
        )
        print(rapporto(await misura(conv, config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
