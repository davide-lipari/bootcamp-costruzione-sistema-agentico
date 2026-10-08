# src/lc/agente.py — lo stesso agente del ciclo, costruito con create_agent
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from src.agents.prompts import AGENT_SYSTEM
from src.agents.registry import Tool
from src.lc.strumenti import in_fila, quando

TETTO = (
    "Non ho completato la richiesta entro i passi previsti: la risposta non c'è. "
    "Riformula la domanda in modo più specifico, o dividila in domande più semplici."
)

RICERCHE_MAX = 3  # quante search_documents in un thread: al caso multi-hop peggiore ne bastano due


def costruisci_agente(
    llm: BaseChatModel,
    tools: list[Tool],
    checkpointer: BaseCheckpointSaver[Any],
    max_steps: int = 6,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """I tool arrivano già costruiti per l'operatore: da qui in poi l'identità non passa."""
    # due middleware di tipi generici diversi: senza l'annotazione mypy rifiuta la lista
    middleware: list[AgentMiddleware[Any, Any]] = [
        # thread_limit, non run_limit: dopo una sosta la ripresa è un'invocazione nuova, e
        # run_limit ripartirebbe da zero; il ciclo del repository conta anche i passi di prima
        ModelCallLimitMiddleware(thread_limit=max_steps, exit_behavior="end"),
        # il retrieval lo sceglie il modello, ma non quanto vuole. "continue" non chiude il run:
        # la chiamata bloccata torna al modello come errore, lui risponde, e la terminazione
        # resta "model". "end" ferma il run qui, e il caso si leggerebbe come max_steps.
        ToolCallLimitMiddleware(
            tool_name="search_documents", thread_limit=RICERCHE_MAX, exit_behavior="continue"
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                t.name: {"allowed_decisions": ["approve", "reject"], "when": quando(t)}
                for t in tools
                if t.scrive  # chi non dichiara di leggere è trattato come chi scrive
            }
        ),
    ]
    return create_agent(
        llm,
        in_fila(tools),  # una fila sola: le chiamate dello stesso passo condividono la sessione
        system_prompt=AGENT_SYSTEM,
        middleware=middleware,
        checkpointer=checkpointer,  # senza, la sosta non ha dove aspettare
    )


def chiamate_vere(messaggi: list[BaseMessage]) -> list[AIMessage]:
    """Le risposte del provider. Quando scatta il tetto il middleware scrive un messaggio
    dell'assistente anche lui, ma non viene dal provider: non ha usage_metadata."""
    return [m for m in messaggi if isinstance(m, AIMessage) and m.usage_metadata]


def come_e_finita(stato: dict[str, Any]) -> tuple[str, str]:
    """(stopped_by, testo), con gli stessi nomi del ciclo del repository."""
    if stato.get("__interrupt__"):
        return "awaiting_approval", "In attesa di approvazione."
    messaggi: list[BaseMessage] = stato["messages"]
    ultima = messaggi[-1] if messaggi else None
    if isinstance(ultima, AIMessage) and ultima.usage_metadata and not ultima.tool_calls:
        return "model", ultima.text
    return "max_steps", TETTO  # il tetto non è una risposta, anche se il middleware ne scrive una
