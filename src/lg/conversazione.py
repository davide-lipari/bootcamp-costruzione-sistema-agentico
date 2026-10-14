# src/lg/conversazione.py — il grafo del G2, dentro una conversazione che dura cinquanta turni
import re
from decimal import Decimal
from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from src.agents.deps import Deps
from src.auth.deps import UserContext
from src.config import settings
from src.lg import memoria
from src.lg.grafo import costo, costruisci_grafo, stato_iniziale
from src.lg.stato import Contesto

MAX_MESSAGGI = 12  # oltre, i turni vecchi escono dallo stato: i fatti restano
TOKEN_CONTESTO = 600  # quanto della conversazione recente accompagna ogni domanda
CLIENTE = re.compile(r"\bC-\d{5}\b")
IBAN = re.compile(r"\bIT\d{2}[A-Z]\d{10,23}\b")
# la richiesta arriva scritta come capita: la parola che si salva, invece, è solo una di queste
PROMPT_PREFERENZE = (
    "L'operatore dà istruzioni su come vuole le risposte. Rispondi con una sola parola, "
    "senza altro:\n"
    "«breve» se chiede risposte brevi o concise;\n"
    "«dettagliato» se chiede risposte dettagliate, complete o in elenco;\n"
    "«dimentica» se chiede di dimenticare o cancellare le sue preferenze;\n"
    "«nessuna» per ogni altra domanda di lavoro, anche se cita clienti o conti."
)


class Fatti(TypedDict):
    """Quello che non deve perdersi in una giornata di lavoro: in campi, non in un riassunto."""

    cliente: str | None
    conti: list[str]


class StatoConversazione(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # la conversazione, per chi la legge
    fatti: Fatti
    turni: Annotated[int, add]
    costo_eur: Annotated[Decimal, add]
    preferenze: dict[str, Any]
    conferma: str | None  # cosa dire all'operatore quando una preferenza cambia


def aggiorna_fatti(fatti: Fatti | None, domanda: str, risposta: str) -> Fatti:
    """Dal testo, con il codice: il cliente nominato per ultimo, i conti apparsi nelle risposte."""
    prima = fatti or {"cliente": None, "conti": []}
    nominati = CLIENTE.findall(domanda)
    conti = list(dict.fromkeys([*prima["conti"], *IBAN.findall(risposta)]))[-10:]
    return {"cliente": nominati[-1] if nominati else prima["cliente"], "conti": conti}


def domanda_con_contesto(messaggi: list[AnyMessage], fatti: Fatti | None, domanda: str) -> str:
    """La domanda di adesso, con i fatti tenuti e gli ultimi turni potati a un budget fisso."""
    recenti = trim_messages(
        messaggi[:-1],  # l'ultimo è la domanda di adesso
        max_tokens=TOKEN_CONTESTO,
        token_counter="approximate",
        strategy="last",
        start_on="human",  # mai una risposta senza la sua domanda
    )
    righe = []
    if fatti and fatti["cliente"]:
        righe.append(f"Il cliente di cui si parla: {fatti['cliente']}.")
    if fatti and fatti["conti"]:
        righe.append("Conti già nominati: " + ", ".join(fatti["conti"]) + ".")
    for m in recenti:
        chi = "operatore" if isinstance(m, HumanMessage) else "assistente"
        righe.append(f"{chi}: {m.text}")
    if not righe:
        return domanda
    return (
        "Contesto della conversazione:\n" + "\n".join(righe) + f"\n\nDomanda di adesso: {domanda}"
    )


def costruisci_conversazione(
    user: UserContext,
    deps: Deps,
    llm: BaseChatModel,
    llm_triage: BaseChatModel,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any] | None,
    llm_ragiona: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Un thread per sessione. Il grafo del G2 lavora dentro un turno: invocato in un nodo,
    usa il checkpointer della conversazione, con un namespace suo per ogni turno, e la sua sosta
    risale alla sessione. Gliene passassimo uno suo, non lo userebbe: per questo, None."""
    banca = costruisci_grafo(user, deps, llm, llm_triage, None, llm_ragiona=llm_ragiona)

    async def ricorda(state: StatoConversazione) -> dict[str, Any]:
        preferenze = await memoria.leggi(store, user)  # il namespace è dell'operatore, in closure
        return {"preferenze": preferenze.model_dump()}

    async def ascolta(state: StatoConversazione) -> dict[str, Any]:
        """Se l'operatore chiede una preferenza, si salva: ma solo il valore ammesso dallo
        schema, mai la frase, e mai un cliente che stia nella stessa domanda."""
        breve = llm_triage.bind(max_tokens=5)  # una parola, come il triage
        r = await breve.ainvoke(
            [SystemMessage(PROMPT_PREFERENZE), HumanMessage(state["messages"][-1].text)]
        )
        scelta = r.text.strip().lower()
        aggiornamento: dict[str, Any] = {
            "costo_eur": costo([r], settings.triage_model),
            "conferma": None,
        }
        if scelta == "dimentica":
            await memoria.dimentica(store, user)
            aggiornamento["preferenze"] = memoria.Preferenze().model_dump()
            aggiornamento["conferma"] = "Va bene: ho dimenticato le mie preferenze."
        elif scelta in ("breve", "dettagliato"):
            preferenze = memoria.Preferenze(formato=scelta)  # un valore dello schema, e basta
            await memoria.salva(store, user, preferenze)
            aggiornamento["preferenze"] = preferenze.model_dump()
            aggiornamento["conferma"] = f"Va bene: da qui in poi rispondo in modo {scelta}."
        return aggiornamento

    async def turno(state: StatoConversazione) -> dict[str, Any]:
        domanda = state["messages"][-1].text
        testo = domanda_con_contesto(state["messages"], state.get("fatti"), domanda)
        stile = memoria.Preferenze.model_validate(state["preferenze"]).formato
        out = await banca.ainvoke(stato_iniziale(testo, user), context=Contesto(stile=stile))
        risposta = out["risposta"]
        if state.get("conferma"):  # il cambio si vede subito, insieme alla risposta
            risposta = f"{risposta}\n\n{state['conferma']}"
        return {
            "messages": [AIMessage(risposta)],
            "fatti": aggiorna_fatti(state.get("fatti"), domanda, out["risposta"]),
            "turni": 1,
            "costo_eur": out["costo_eur"],
        }

    def pota(state: StatoConversazione) -> dict[str, Any]:
        vecchi = state["messages"][:-MAX_MESSAGGI]
        return {"messages": [RemoveMessage(id=m.id) for m in vecchi if m.id]} if vecchi else {}

    g = StateGraph(StatoConversazione)
    g.add_node("ricorda", ricorda)
    g.add_node("ascolta", ascolta)
    g.add_node("turno", turno)
    g.add_node("pota", pota)
    g.add_edge(START, "ricorda")
    g.add_edge("ricorda", "ascolta")
    g.add_edge("ascolta", "turno")
    g.add_edge("turno", "pota")
    g.add_edge("pota", END)
    return g.compile(checkpointer=checkpointer)


def thread_della_sessione(user: UserContext, sessione: str) -> str:
    """Un thread per sessione di chat: la chiave la compone il server, come nel G2."""
    return f"{user.username}:{sessione}"
