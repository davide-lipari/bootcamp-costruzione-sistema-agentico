# evals/implementazioni.py — quattro modi di rispondere alla stessa domanda, un contratto solo
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from openai import AsyncOpenAI

from src.agents.deps import Deps
from src.agents.loop import PREZZI_PER_1K, run_agent
from src.agents.prompts import AGENT_SYSTEM
from src.agents.tools import build_tools_for
from src.auth.deps import UserContext
from src.lc.agente import chiamate_vere, come_e_finita, costruisci_agente
from src.lc.catena import RispostaCitata, costruisci_catena, fonti_verificate
from src.lg.grafo import costruisci_grafo, stato_iniziale, thread_per
from src.services.retrieval_service import RetrievalResult


@dataclass
class Esito:
    """Quello che l'harness guarda, qualunque sia l'implementazione che ha risposto."""

    reply: str
    tool_calls: list[str] = field(default_factory=list)  # anche quelli in attesa di approvazione
    steps: int = 0  # chiamate pagate al modello
    stopped_by: str = ""  # "model" | "max_steps" | "budget" | "awaiting_approval" | "errore"
    cost_eur: Decimal = Decimal("0")
    ricerche: int = 0  # le search_documents eseguite: la spesa del retrieval, non del modello


class Implementazione(Protocol):
    nome: str

    async def rispondi(self, domanda: str, utente: UserContext, deps: Deps) -> Esito: ...


def costo_messaggi(messaggi: list[BaseMessage], model: str) -> Decimal:
    """Lo stesso prezzo del ciclo, letto dai metadati che LangChain mette su ogni risposta."""
    ingresso, uscita = PREZZI_PER_1K[model]  # un modello senza prezzo si ferma qui, come nel ciclo
    totale = Decimal("0")
    for m in messaggi:
        uso = m.usage_metadata if isinstance(m, AIMessage) else None
        if uso:
            totale += (uso["input_tokens"] * ingresso + uso["output_tokens"] * uscita) / 1000
    return totale


# ---------------------------------------------------------------- 1. il ciclo del repository
class CicloAMano:
    nome = "ciclo"

    def __init__(self, client: AsyncOpenAI, model: str, max_steps: int = 8) -> None:
        self.client, self.model, self.max_steps = client, model, max_steps

    async def rispondi(self, domanda: str, utente: UserContext, deps: Deps) -> Esito:
        run = await run_agent(
            messaggi=[
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": domanda},
            ],
            tools=build_tools_for(utente, deps),
            client=self.client,
            model=self.model,
            max_steps=self.max_steps,
            runs=deps.runs,
            user=utente,
        )
        chiamati = list(run.tool_calls)
        if run.stopped_by == "awaiting_approval":  # il tool sospeso non è partito, ma è chiesto
            stato = await deps.runs.get(run.run_id)
            chiamati += [c["function"]["name"] for c in (stato.pending_calls if stato else [])]
            # una prova non lascia azioni da approvare, come nel banco del repository: chi decide
            # vedrebbe un caso finto, e approvandolo aprirebbe una pratica vera
            await deps.runs.chiudi(run.run_id, "closed_by_eval")
        ricerche = sum(1 for c in chiamati if c == "search_documents")  # solo quelle eseguite
        return Esito(run.reply, chiamati, run.steps, run.stopped_by, run.cost_eur, ricerche)


# ---------------------------------------------------------------- 2. la catena
class Catena:
    nome = "catena"

    def __init__(self, llm: BaseChatModel, model: str) -> None:
        self.llm, self.model = llm, model

    async def rispondi(self, domanda: str, utente: UserContext, deps: Deps) -> Esito:
        async def recupera(q: str) -> list[RetrievalResult]:
            # il retrieval del repository, coi permessi del ruolo: la catena non ne ha un altro
            vettore = await deps.embedder.embed_one(q)
            return await deps.retrieval.search_for_user(vettore, utente.role, top_k=3)

        out: dict[str, Any] = await costruisci_catena(self.llm, recupera).ainvoke(
            {"domanda": domanda}
        )
        grezza, risposta = out["uscita"]["raw"], out["uscita"]["parsed"]
        costo = costo_messaggi([grezza], self.model)
        # per l'harness la ricerca fissa della catena vale come search_documents: è ciò che fa
        if not isinstance(risposta, RispostaCitata):  # il parsing è fallito: non è una risposta
            return Esito("", ["search_documents"], 1, "errore", costo, ricerche=1)
        fonti = fonti_verificate(risposta, out["passaggi"])
        testo = risposta.risposta + "".join(f" [{f}]" for f in fonti)
        return Esito(testo, ["search_documents"], 1, "model", costo, ricerche=1)


# ---------------------------------------------------------------- 3. l'agente prebuilt
class AgentePrebuilt:
    nome = "agente"

    def __init__(self, llm: BaseChatModel, model: str, max_steps: int = 8) -> None:
        self.llm, self.model, self.max_steps = llm, model, max_steps

    async def rispondi(self, domanda: str, utente: UserContext, deps: Deps) -> Esito:
        agente = costruisci_agente(
            self.llm, build_tools_for(utente, deps), InMemorySaver(), self.max_steps
        )
        stato: dict[str, Any] = await agente.ainvoke(
            {"messages": [HumanMessage(domanda)]},
            {"configurable": {"thread_id": str(uuid4())}},
        )
        vere = chiamate_vere(stato["messages"])
        chiamati = [c["name"] for m in vere for c in m.tool_calls]  # anche quello in sosta
        stopped_by, testo = come_e_finita(stato)
        costo = costo_messaggi(stato["messages"], self.model)
        # quelle eseguite: la chiamata bloccata dal tetto lascia un ToolMessage in errore
        ricerche = sum(
            1
            for m in stato["messages"]
            if isinstance(m, ToolMessage) and m.name == "search_documents" and m.status == "success"
        )
        return Esito(testo, chiamati, len(vere), stopped_by, costo, ricerche)


# ---------------------------------------------------------------- 4. il grafo
class Grafo:
    nome = "grafo"

    def __init__(self, llm: BaseChatModel, llm_triage: BaseChatModel, model: str) -> None:
        self.llm, self.llm_triage, self.model = llm, llm_triage, model

    async def rispondi(self, domanda: str, utente: UserContext, deps: Deps) -> Esito:
        # la misura non deve sopravvivere a niente: qui il salvataggio in memoria è giusto
        grafo = costruisci_grafo(
            utente, deps, self.llm, self.llm_triage, InMemorySaver(), modello=self.model
        )
        config: Any = {"configurable": {"thread_id": thread_per(utente)}}
        stato: dict[str, Any] = await grafo.ainvoke(stato_iniziale(domanda, utente), config)
        chiamati = [t for c in stato.get("contributi", []) for t in c["tool_calls"]]
        ricerche = sum(1 for t in chiamati if t == "search_documents")
        if stato.get("__interrupt__"):  # la segnalazione è chiesta, e aspetta una persona
            return Esito(
                "In attesa di approvazione.",
                [*chiamati, "apri_segnalazione_compliance"],
                stato["passi"],
                "awaiting_approval",
                stato["costo_eur"],
                ricerche,
            )
        return Esito(
            stato["risposta"],
            chiamati,
            stato["passi"],
            stato["stopped_by"],
            stato["costo_eur"],
            ricerche,
        )
