# src/lg/grafo.py — il supervisor del repository, con il flusso deciso dal codice e la sosta dentro
import json
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, Protocol, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from src.agents.approval import RIFIUTO
from src.agents.deps import Deps
from src.agents.loop import BUDGET_PER_RUN, PREZZI_PER_1K
from src.agents.supervisor import PROMPT_SINTESI, SPECIALISTI
from src.agents.tools import SegnalazioneArgs, _sopra_soglia, build_tools_for
from src.auth.deps import UserContext
from src.config import settings
from src.lc.agente import chiamate_vere, come_e_finita
from src.lc.strumenti import in_fila
from src.lg.stato import Contesto, Contributo, Decisione, Instradamento, StatoBanca

MAX_PASSI = 12  # chiamate al modello in tutto il grafo: il piano più lungo ne fa dieci
PASSI_SPECIALISTA = 4
ISTRADAMENTI: tuple[Instradamento, ...] = (
    "dati_conto",
    "policy",
    "entrambi",
    "segnalazione",
    "istruttoria",
)

# il flusso, dichiarato: per ogni tipo di domanda, chi lavora e in che ordine
PERCORSI: dict[Instradamento, list[str]] = {
    "dati_conto": ["dati_conto"],
    "policy": ["policy"],
    "entrambi": ["dati_conto", "policy"],
    # prima i dati del conto e la policy, poi la proposta, la sosta, e solo dopo la scrittura
    "segnalazione": ["dati_conto", "policy", "prepara", "chiedi", "scrivi"],
    # un'analisi che chiede di ragionare: prima i conti, poi lo specialista sul modello che ragiona
    "istruttoria": ["dati_conto", "istruttoria"],
}
SERVONO_LA_PROPOSTA = {"chiedi", "scrivi"}

PROMPT_TRIAGE = """Classifica la domanda di un operatore di filiale. Rispondi con una parola sola:
dati_conto    se riguarda clienti, conti, saldi o movimenti;
policy        se riguarda regole, soglie, procedure o adempimenti;
segnalazione  se l'operatore chiede di aprire una segnalazione alla Compliance;
istruttoria   se chiede di analizzare i movimenti di un conto per capire se vanno segnalati;
entrambi      se servono dati e regole insieme, o se non sei sicuro."""

PROMPT_PREPARA = """Prepara gli argomenti di una segnalazione alla Compliance a partire dalla
domanda dell'operatore e dai dati raccolti. L'IBAN va copiato esattamente come compare nei dati
del conto; l'importo solo se la domanda o i dati lo indicano. Non inventare nulla."""


class Nodo(Protocol):
    """La forma di un nodo asincrono. Il parametro si chiama `state` perché il protocollo dei
    nodi di LangGraph lo chiama così, e Callable perderebbe il nome: mypy strict rifiuterebbe
    add_node."""

    def __call__(self, state: StatoBanca) -> Awaitable[dict[str, Any]]: ...


# il listino del fornitore, con la data nel commento del modulo da cui viene: il modello che
# ragiona è già dentro, perché è il modello del giorno. Il ragionamento si paga come uscita
PREZZI = {**PREZZI_PER_1K}

PROMPT_ISTRUTTORIA = """Sei lo specialista delle istruttorie di LipariBank. Leggi i movimenti del
conto con il tool, cerca la policy che li riguarda, e di' se ci sono elementi da segnalare: quali
movimenti, con date e importi esatti, e quale documento, citato fra parentesi quadre. Non decidi tu
se aprire una segnalazione: lo decide l'operatore, e sopra soglia un responsabile."""

ETICHETTE = {  # quello che l'operatore legge mentre il grafo lavora: mai i dati
    "triage": "Sto capendo la richiesta",
    "dati_conto": "Sto leggendo i conti",
    "policy": "Sto cercando nelle policy",
    "istruttoria": "Sto analizzando i movimenti",
    "prepara": "Sto preparando la segnalazione",
    "chiedi": "La segnalazione aspetta un responsabile",
    "scrivi": "Sto aprendo la segnalazione",
    "sintesi": "Sto scrivendo la risposta",
    "chiusura": "Chiudo con quello che ho",
}


STILI = {
    "breve": " Rispondi in tre frasi al massimo, con i numeri esatti.",
    "dettagliato": " Rispondi con tutti i dati dei contributi, in un elenco.",
}


def avanza(nodo: str) -> None:
    """Una riga di avanzamento: arriva a chi ascolta con stream_mode="custom", gli altri no."""
    get_stream_writer()({"fase": ETICHETTE[nodo]})


def costo(messaggi: list[BaseMessage], modello: str) -> Decimal:
    ingresso, uscita = PREZZI[modello]  # un modello senza prezzo si ferma qui
    totale = Decimal("0")
    for m in messaggi:
        uso = m.usage_metadata if isinstance(m, AIMessage) else None
        if uso:
            totale += (uso["input_tokens"] * ingresso + uso["output_tokens"] * uscita) / 1000
    return totale


def successori(nodo: str) -> list[str]:
    """Dove si può andare da un nodo: il nodo che lo segue in una riga della tabella, più le
    due uscite. Il disegno del grafo mostra così il flusso vero, non ogni arco possibile."""
    dopo = {"sintesi", "chiusura"}
    for percorso in PERCORSI.values():
        riga = ["triage", *percorso]
        if nodo in riga and riga.index(nodo) + 1 < len(riga):
            dopo.add(riga[riga.index(nodo) + 1])
    return sorted(dopo)


def instrada(state: StatoBanca) -> str:
    """Chi lavora dopo. Legge solo lo stato, e i tetti vengono prima di tutto."""
    if state["costo_eur"] > BUDGET_PER_RUN:  # i tetti, prima di qualunque scelta
        return "chiusura"
    if state["passi"] >= MAX_PASSI:
        return "chiusura"
    for nodo in PERCORSI[state["instradamento"]]:
        if nodo in state["visitati"]:
            continue
        if nodo in SERVONO_LA_PROPOSTA and not state.get("proposta"):
            continue  # la proposta non c'è: niente sosta e niente scrittura
        return nodo
    return "sintesi"


def chiudi(state: StatoBanca) -> dict[str, Any]:
    """L'uscita per tetto: compone quello che c'è, non chiama nessuno, non può fallire. Il
    motivo lo decide qui: se sono sforati tutti e due i tetti, vince il costo."""
    sforato = state.get("costo_eur", Decimal("0")) > BUDGET_PER_RUN
    fatti = [c["contenuto"] for c in state.get("contributi", []) if c["completo"]]
    mancano = [
        n
        for n in PERCORSI[state.get("instradamento", "entrambi")]
        if n not in state.get("visitati", [])
    ]
    testo = "Non ho completato la richiesta: " + (
        "il budget previsto si è esaurito." if sforato else "i passi previsti si sono esauriti."
    )
    if fatti:
        testo += " Quello che ho verificato: " + " ".join(fatti)
    if mancano:
        testo += " Resta da fare: " + ", ".join(mancano) + "."
    return {
        "risposta": testo,
        "stopped_by": "budget" if sforato else "max_steps",
        "visitati": ["chiusura"],
    }


def costruisci_grafo(
    user: UserContext,
    deps: Deps,
    llm: BaseChatModel,
    llm_triage: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any] | None,
    modello: str = settings.default_model,
    modello_triage: str = settings.triage_model,
    llm_ragiona: BaseChatModel | None = None,
    modello_ragiona: str = settings.default_model,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """L'operatore entra qui, in closure: i tool nascono per lui e lo stato non lo contiene."""
    tools = build_tools_for(user, deps)
    by_name = {t.name: t for t in tools}

    async def triage(state: StatoBanca) -> dict[str, Any]:
        avanza("triage")
        # pochi token, come il triage del supervisor: la risposta è una parola sola
        breve = llm_triage.bind(max_tokens=5)
        r = await breve.ainvoke([SystemMessage(PROMPT_TRIAGE), HumanMessage(state["domanda"])])
        scelta = r.text.strip().lower()
        if scelta not in ISTRADAMENTI:
            scelta = "entrambi"  # una parola fuori elenco prende la strada che non scrive
        return {
            "instradamento": scelta,
            "passi": 1,
            "costo_eur": costo([r], modello_triage),
            "visitati": ["triage"],
        }

    def specialista(nome: str) -> Nodo:
        s = SPECIALISTI[nome]
        agente = create_agent(
            llm,
            in_fila([by_name[t] for t in s.tool_names]),  # solo i SUOI tool, uno alla volta
            system_prompt=s.system_prompt,
            middleware=[ModelCallLimitMiddleware(run_limit=PASSI_SPECIALISTA, exit_behavior="end")],
        )

        async def nodo(state: StatoBanca) -> dict[str, Any]:
            # lo specialista non vede gli altri contributi: riceve la domanda, e basta
            avanza(nome)
            out = await agente.ainvoke({"messages": [HumanMessage(state["domanda"])]})
            vere = chiamate_vere(out["messages"])
            uscita, testo = come_e_finita(out)
            contributo: Contributo = {
                "specialista": s.nome,
                "completo": uscita == "model",
                "contenuto": testo,
                "tool_calls": [c["name"] for m in vere for c in m.tool_calls],
            }
            return {
                "contributi": [contributo],  # una lista di un elemento: fonde il reducer
                "passi": len(vere),
                "costo_eur": costo(out["messages"], modello),
                "visitati": [nome],
            }

        return nodo

    async def prepara(state: StatoBanca) -> dict[str, Any]:
        # la chiamata al modello sta QUI, prima della sosta: alla ripresa non si ripaga
        avanza("prepara")
        estrattore = llm.with_structured_output(
            SegnalazioneArgs, method="function_calling", include_raw=True
        )
        materiale = json.dumps(
            {"domanda": state["domanda"], "contributi": state["contributi"]}, ensure_ascii=False
        )
        out = cast(
            dict[str, Any],
            await estrattore.ainvoke([SystemMessage(PROMPT_PREPARA), HumanMessage(materiale)]),
        )
        args = out["parsed"]
        aggiornamento: dict[str, Any] = {
            "passi": 1,
            "costo_eur": costo([out["raw"]], modello),
            "visitati": ["prepara"],
        }
        if isinstance(args, SegnalazioneArgs):  # validati da Pydantic, come nel ciclo
            aggiornamento["proposta"] = args.model_dump(mode="json")
        else:
            aggiornamento["contributi"] = [
                {
                    "specialista": "segnalazione",
                    "completo": False,
                    "contenuto": "La segnalazione non è stata preparata: mancano dati validi.",
                    "tool_calls": [],
                }
            ]
        return aggiornamento

    def chiedi(state: StatoBanca) -> dict[str, Any]:
        # alla ripresa questo nodo riparte dall'inizio: prima dell'interrupt solo letture
        args = SegnalazioneArgs.model_validate(state["proposta"])
        if not _sopra_soglia(args):
            sotto: Decisione = {"approvato": True, "da": None, "motivo": None}
            return {"decisione": sotto, "visitati": ["chiedi"]}
        avanza("chiedi")  # alla ripresa si riemette: è una riga per chi guarda, non un effetto
        risposta = interrupt(
            {  # tutto e solo quello che serve a chi decide
                "azione": "apri_segnalazione_compliance",
                "conto": args.account_id,
                "importo": str(args.importo) if args.importo is not None else "non indicato",
                "motivo": args.motivo,
                "richiesto_da": state["richiedente"],
            }
        )
        grezza = risposta if isinstance(risposta, dict) else {}  # non si fida della forma
        decisione: Decisione = {
            "approvato": grezza.get("approvato") is True,  # solo un True vero è un sì
            "da": grezza.get("da"),
            "motivo": grezza.get("motivo") if grezza else "decisione non leggibile",
        }
        return {"decisione": decisione, "visitati": ["chiedi"]}

    async def scrivi(state: StatoBanca) -> dict[str, Any]:
        # l'unico nodo con un effetto, dopo la sosta e in un nodo suo
        avanza("scrivi")
        decisione = state["decisione"]
        if decisione["approvato"]:
            try:
                args = SegnalazioneArgs.model_validate(state["proposta"])
            except ValidationError:
                esito = "La segnalazione non è stata aperta: argomenti non validi."
            else:  # lo stesso tool del ciclo: il muro e l'idempotenza vengono con lui
                esito = await by_name["apri_segnalazione_compliance"].run(args)
        else:
            esito = RIFIUTO.format(
                da=decisione["da"] or "di turno", motivo=decisione["motivo"] or "non indicato"
            )
        contributo: Contributo = {
            "specialista": "segnalazione",
            "completo": True,
            "contenuto": esito,
            "tool_calls": ["apri_segnalazione_compliance"] if decisione["approvato"] else [],
        }
        return {"contributi": [contributo], "visitati": ["scrivi"]}

    async def istruttoria(state: StatoBanca, runtime: Runtime[Contesto]) -> dict[str, Any]:
        # il modello che ragiona, su questo specialista soltanto, con il budget dell'invocazione
        avanza("istruttoria")
        contesto = runtime.context or Contesto()
        # lo sforzo lo capisce solo un modello che ragiona: a un altro, il fornitore lo rifiuta
        modello_del_giorno = (
            llm_ragiona.model_copy(
                update={
                    "reasoning_effort": contesto.sforzo,
                    "max_tokens": contesto.max_token_istruttoria,
                }
            )
            if llm_ragiona
            else llm
        )
        agente = create_agent(
            modello_del_giorno,
            in_fila([by_name[t] for t in ("list_recent_movements", "search_documents")]),
            system_prompt=PROMPT_ISTRUTTORIA,
            middleware=[ModelCallLimitMiddleware(run_limit=PASSI_SPECIALISTA, exit_behavior="end")],
        )
        materiale = json.dumps(
            {"domanda": state["domanda"], "contributi": state["contributi"]}, ensure_ascii=False
        )
        out = await agente.ainvoke({"messages": [HumanMessage(materiale)]})
        vere = chiamate_vere(out["messages"])
        uscita, testo = come_e_finita(out)
        if uscita == "model" and not testo.strip():
            # il ragionamento ha consumato il tetto d'uscita: pagato, e la risposta non c'è
            testo = "L'istruttoria non ha prodotto una risposta entro il budget di ragionamento."
            uscita = "max_steps"
        contributo: Contributo = {
            "specialista": "istruttoria",
            "completo": uscita == "model",
            "contenuto": testo,
            "tool_calls": [c["name"] for m in vere for c in m.tool_calls],
        }
        return {
            "contributi": [contributo],
            "passi": len(vere),
            "costo_eur": costo(out["messages"], modello_ragiona if llm_ragiona else modello),
            "visitati": ["istruttoria"],
        }

    def chiusura(state: StatoBanca) -> dict[str, Any]:
        avanza("chiusura")
        return chiudi(state)

    async def sintesi(state: StatoBanca, runtime: Runtime[Contesto]) -> dict[str, Any]:
        avanza("sintesi")
        contesto = runtime.context or Contesto()
        stile = STILI[contesto.stile]  # una preferenza dell'operatore, non un'istruzione sua
        materiale = json.dumps(
            {"domanda": state["domanda"], "contributi": state["contributi"]}, ensure_ascii=False
        )
        # un tetto alla risposta, come la sintesi del supervisor
        r = await llm.bind(max_tokens=500).ainvoke(
            [SystemMessage(PROMPT_SINTESI + stile), HumanMessage(materiale)]
        )
        return {
            "risposta": r.text,
            "stopped_by": "model",
            "passi": 1,
            "costo_eur": costo([r], modello),
            "visitati": ["sintesi"],
        }

    g = StateGraph(StatoBanca, context_schema=Contesto)
    g.add_node("triage", triage)
    for nome in ("dati_conto", "policy"):
        g.add_node(nome, specialista(nome))  # il nome passa come argomento, non in una lambda
    g.add_node("prepara", prepara)
    g.add_node("chiedi", chiedi)
    g.add_node("scrivi", scrivi)
    g.add_node("istruttoria", istruttoria)
    g.add_node("sintesi", sintesi)
    g.add_node("chiusura", chiusura)

    g.add_edge(START, "triage")
    for da in ("triage", "dati_conto", "policy", "istruttoria", "prepara", "chiedi", "scrivi"):
        g.add_conditional_edges(da, instrada, successori(da))  # un insieme chiuso, per nodo
    g.add_edge("sintesi", END)
    g.add_edge("chiusura", END)
    return g.compile(checkpointer=checkpointer)


def stato_iniziale(domanda: str, user: UserContext) -> dict[str, Any]:
    """Il contratto d'ingresso, in un posto solo: lo usano l'harness e i test."""
    return {"domanda": domanda, "richiedente": user.username}


def thread_per(user: UserContext) -> str:
    """La chiave della ripresa: la compone il server dall'utente autenticato, mai il chiamante."""
    return f"{user.username}:{uuid4()}"


@asynccontextmanager
async def checkpointer_postgres() -> AsyncIterator[AsyncPostgresSaver]:
    """Si apre una volta, nel ciclo di vita del servizio: la sosta può durare ore."""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()  # le sue tabelle: idempotente, e fuori dalle migration
        yield saver


# le tabelle che LangGraph si crea da sé: non sono nei modelli, quindi Alembic non le conosce
TABELLE_DI_LANGGRAPH = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
    | {"store", "store_migrations"}  # la memoria fra conversazioni
)


def fuori_dalle_migration(
    oggetto: object, nome: str | None, tipo: str, riflesso: bool, confronto: object
) -> bool:
    """L'`include_object` di Alembic. Senza, `revision --autogenerate` trova le tabelle del
    checkpointer nel database e non nei modelli, e scrive una migration che le cancella: con
    le soste che contengono."""
    return not (tipo == "table" and nome in TABELLE_DI_LANGGRAPH)
