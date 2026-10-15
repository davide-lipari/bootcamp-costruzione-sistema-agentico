# src/lg/delega.py — un obiettivo a un altro agente: si manda, si sosta quando chiede, si riprende
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from a2a.client import Client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from a2a.utils.errors import A2AError
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

GIRI_MAX = 3  # un agente che continua a chiedere è un caso per una persona, non un ciclo
CHIEDE = TaskState.Name(TaskState.TASK_STATE_INPUT_REQUIRED)
FINITO = TaskState.Name(TaskState.TASK_STATE_COMPLETED)
TROPPI_GIRI = (
    "L'agente dell'istruttoria continua a chiedere: la richiesta va ripresa da una persona."
)
PERSO = "TASK_PERSO"
TASK_PERSO = (
    "L'agente dell'istruttoria non ha più la richiesta: va rifatta, o ripresa da una persona."
)


class StatoDelega(TypedDict, total=False):
    richiesta: str  # cosa mandare all'agente adesso
    task_id: str  # il task dall'altra parte: nello stato, così sopravvive a una sosta lunga
    context_id: str
    stato: str  # lo stato del task, come lo dice l'agente
    domanda: str  # quello che l'agente ha chiesto, se ha chiesto
    esito: str
    giri: Annotated[int, add]


def costruisci_delega(
    client: Client, checkpointer: BaseCheckpointSaver[Any] | None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Il client A2A arriva in closure, con il token dell'operatore per QUEL agente: alla ripresa
    se ne costruisce uno nuovo, e il task si ritrova dall'id nello stato."""

    async def invia(state: StatoDelega) -> dict[str, Any]:
        messaggio = new_text_message(state["richiesta"], role=Role.ROLE_USER)
        if state.get("task_id"):
            messaggio.task_id, messaggio.context_id = state["task_id"], state["context_id"]
        ultimo = None
        try:
            async for risposta in client.send_message(SendMessageRequest(message=messaggio)):
                ultimo = risposta
        except A2AError:
            # task non trovato, già concluso, o perso in un riavvio dell'agente: un errore
            # esplicito dall'altra parte, e chi delega lo dichiara invece di rompersi
            return {"stato": PERSO, "esito": TASK_PERSO, "domanda": "", "giri": 1}
        if ultimo is None or not ultimo.task.id:
            raise RuntimeError("l'agente non ha restituito un task")
        task = ultimo.task
        stato = TaskState.Name(task.status.state)
        aggiornamento: dict[str, Any] = {
            "task_id": task.id,
            "context_id": task.context_id,
            "stato": stato,
            "domanda": get_message_text(task.status.message) if stato == CHIEDE else "",
            "giri": 1,
        }
        if stato == FINITO:
            aggiornamento["esito"] = "\n".join(get_artifact_text(a) for a in task.artifacts)
        elif stato != CHIEDE:  # fallito, rifiutato, annullato: il motivo è il messaggio di stato
            aggiornamento["esito"] = get_message_text(task.status.message) or stato
        return aggiornamento

    def chiedi(state: StatoDelega) -> dict[str, Any]:
        # alla ripresa riparte da qui: prima della sosta soltanto letture dello stato
        risposta = interrupt({"agente": "istruttoria-lipari", "domanda": state["domanda"]})
        return {"richiesta": risposta if isinstance(risposta, str) else ""}

    def troppi_giri(state: StatoDelega) -> dict[str, Any]:
        return {"esito": TROPPI_GIRI}

    def dopo_invio(state: StatoDelega) -> Literal["chiedi", "troppi_giri", "__end__"]:
        if state["stato"] != CHIEDE:
            return "__end__"
        return "chiedi" if state.get("giri", 0) < GIRI_MAX else "troppi_giri"

    g = StateGraph(StatoDelega)
    g.add_node("invia", invia)
    g.add_node("chiedi", chiedi)
    g.add_node("troppi_giri", troppi_giri)
    g.add_edge(START, "invia")
    g.add_conditional_edges("invia", dopo_invio, ["chiedi", "troppi_giri", END])
    g.add_edge("chiedi", "invia")
    g.add_edge("troppi_giri", END)
    return g.compile(checkpointer=checkpointer)
