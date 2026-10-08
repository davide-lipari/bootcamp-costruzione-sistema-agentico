# tests/test_g1_strada_a.py — il tetto sulle ricerche, e il controllo sul valore della risposta
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from evals.confronto import CONTROLLI_CON_VALORE, controlla
from evals.implementazioni import Esito
from evals.runner import CONTROLLI, DATASETS, leggi
from src.db.session import AsyncSessionLocal, engine
from src.lc.agente import RICERCHE_MAX, come_e_finita, costruisci_agente
from src.lib.chunking import chunk_text
from src.services.ingest_service import senza_ripetizioni
from tests.finti import completion, embedder_finto, openai_finto
from tests.test_g1_langchain import MARCO, Copione, chat, tool

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
NUOVI = ("traj-006", "traj-007", "traj-008", "traj-009")
MULTI_HOP = ("traj-006", "traj-007", "traj-008")
COPIONE_5_RICERCHE = [
    tool("search_documents", json.dumps({"query": f"commissioni {i}"}), i) for i in range(1, 6)
]


@pytest.fixture(autouse=True)
async def pool_svuotato() -> AsyncIterator[None]:
    # come in test_g3: il TestClient di test_g2 gira in un suo event loop, e le connessioni
    # aperte qui restano nel pool legate a quello che le ha aperte
    await engine.dispose()
    yield
    await engine.dispose()


def ricerche_eseguite(stato: dict[str, Any]) -> int:
    """Le search_documents arrivate al tool: la chiamata bloccata dal tetto non è un'esecuzione."""
    return sum(
        1
        for m in stato["messages"]
        if isinstance(m, ToolMessage) and m.name == "search_documents" and m.status == "success"
    )


async def _agente(copione: Copione) -> tuple[AsyncSession, CompiledStateGraph[Any, Any, Any, Any]]:
    from src.agents.deps import crea_deps
    from src.agents.tools import build_tools_for
    from src.auth.deps import UserContext

    s = AsyncSessionLocal()
    deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
    tools = build_tools_for(UserContext(**MARCO), deps)
    return s, costruisci_agente(chat(copione), tools, InMemorySaver())


async def _chiedi(copione: Copione, domanda: str, thread: str) -> dict[str, Any]:
    s, agente = await _agente(copione)
    async with s:
        return await agente.ainvoke(
            {"messages": [HumanMessage(domanda)]}, {"configurable": {"thread_id": thread}}
        )


async def test_tetto_ricerche() -> None:
    """Cinque ricerche chieste: il tetto le blocca a tre, e la risposta resta del modello."""
    copione = Copione(COPIONE_5_RICERCHE + [completion("Non trovo questa risposta nei documenti.")])
    stato = await _chiedi(
        copione, "Qual è la commissione del deposito cauzionale?", "mbianchi:tetto-ricerche"
    )
    assert ricerche_eseguite(stato) <= RICERCHE_MAX
    assert come_e_finita(stato)[0] == "model"
    assert not copione.risposte, "il copione non è finito: il run si è fermato prima della risposta"


async def test_il_caso_multi_hop_legittimo_sta_sotto_il_tetto() -> None:
    """Il caso legittimo peggiore: una ricerca per il documento di contesto, una per la voce che
    cita. Se queste due non stanno sotto il tetto, il tetto taglia il caso giusto."""
    copione = Copione(
        [
            tool("search_documents", json.dumps({"query": "bonifico verso Panama"}), 1),
            tool("search_documents", json.dumps({"query": "commissioni bonifico estero"}), 2),
            completion(
                "Un bonifico extra-SEPA costa 15,00 più 0,20% dell'importo: per 26.800 euro "
                "sono 68,60. [commissioni_bonifico]"
            ),
        ]
    )
    stato = await _chiedi(
        copione, "Quanto costa un bonifico di 26.800 euro verso Panama?", "mbianchi:multi-hop"
    )
    assert ricerche_eseguite(stato) == 2
    assert ricerche_eseguite(stato) <= RICERCHE_MAX
    assert come_e_finita(stato)[0] == "model"
    assert not copione.risposte, "il copione non è finito"


def _esito_perfetto(caso: dict[str, Any]) -> Esito:
    atteso = caso["expected"]
    return Esito(
        reply=atteso.get("risposta_contiene") or "Non risulta nei documenti.",
        tool_calls=list(atteso["tool_richiesti"]),
        steps=1,
        stopped_by=atteso["terminazione"],
    )


def test_sesto_controllo_valore() -> None:
    caso = {
        "tool_richiesti": ["search_documents"],
        "tool_vietati": [],
        "passi_max": 5,
        "terminazione": "model",
        "risposta_contiene": "68,60",
    }
    # traiettoria perfetta ma valore sbagliato: è il sesto controllo che boccia
    sbagliato = Esito(
        "Costa 15,00 euro più il 0,20% dell'importo.", ["search_documents"], 3, "model"
    )
    assert controlla(caso, sbagliato) == ["risposta_contiene"]
    assert controlla(caso, Esito("68,60", ["search_documents"], 3, "model")) == []
    # lo stesso numero scritto in altri modi passa lo stesso
    for risposta in ("68,6", "68,60 €", "68.60", "per 26.800 euro viene 68,60 EUR"):
        assert controlla(caso, Esito(risposta, ["search_documents"], 3, "model")) == [], risposta
    # senza il campo il controllo vale True: la maggior parte dei casi non misura il valore
    senza = {k: v for k, v in caso.items() if k != "risposta_contiene"}
    assert controlla(senza, Esito("qualunque cosa", ["search_documents"], 3, "model")) == []
    assert "risposta_contiene" in CONTROLLI_CON_VALORE and "risposta_contiene" not in CONTROLLI


def test_casi_nel_dataset() -> None:
    # il caricamento del runner, non una lettura del JSON fatta a mano
    casi = leggi(DATASETS / "agent_traiettorie.jsonl")
    nuovi = [c for c in casi if c["id"] in NUOVI]
    assert [c["id"] for c in nuovi] == list(NUOVI)
    assert all(c["expected"]["tool_richiesti"] == ["search_documents"] for c in nuovi)
    assert all("risposta_contiene" in c["expected"] for c in nuovi if c["id"] in MULTI_HOP)
    assert "risposta_contiene" not in next(c for c in nuovi if c["id"] == "traj-009")["expected"]
    # e l'harness le consuma: il caso perfetto passa, col valore sbagliato no
    for caso in nuovi:
        assert controlla(caso["expected"], _esito_perfetto(caso)) == [], caso["id"]
    sbagliato = Esito("0,00", ["search_documents"], 3, "model")
    assert "risposta_contiene" in controlla(nuovi[0]["expected"], sbagliato)


def test_nome_entro_300() -> None:
    for doc, chiave in (
        ("panama_operativita", "commissioni_bonifico"),
        ("operativita_nepal", "bonifico_urgente"),
    ):
        testo = (DOCS / f"{doc}.md").read_text(encoding="utf-8")
        primo = chunk_text(senza_ripetizioni(testo))[0]
        assert chiave in primo[:300], f"{chiave} non sta entro i primi 300 caratteri di {doc}"
    # la tariffa sta solo in commissioni_bonifico: se fosse qui, non servirebbe la seconda ricerca
    panama = (DOCS / "panama_operativita.md").read_text(encoding="utf-8")
    assert not any(c.isdigit() for c in panama)
