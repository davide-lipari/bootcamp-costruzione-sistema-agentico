# src/lc/strumenti.py — i cinque tool del repository, portati su LangChain senza toccarli
import asyncio
import logging
from collections.abc import Callable

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from src.agents.registry import Tool

logger = logging.getLogger(__name__)


def fallito(nome: str) -> str:
    """La stessa frase del ciclo del repository: un tool che si rompe è un'osservazione."""
    return (
        f"ERRORE: il tool '{nome}' non ha potuto completare l'operazione. "
        "Non riprovare la stessa chiamata: riferisci all'utente che il dato ora non è disponibile."
    )


def a_langchain(tool: Tool, fila: asyncio.Lock | None = None) -> StructuredTool:
    """Lo stesso Tool, con la stessa closure: l'operatore viaggia con lui, non negli argomenti.

    Due cose del ciclo che LangChain non fa da sé. Un'eccezione del tool torna al modello come
    osservazione, invece di fermare l'agente. E, con una `fila`, le chiamate dello stesso passo
    partono una alla volta: il nodo dei tool le esegue insieme, e i tool del repository
    condividono una sessione del database che non regge due operazioni contemporanee.
    """

    async def chiama(**kwargs: object) -> str:
        args = tool.args_model.model_validate(kwargs)
        try:
            if fila is None:
                return await tool.run(args)
            async with fila:
                return await tool.run(args)
        except Exception as e:
            # come il ciclo: nel log il tipo dell'errore, non il traceback con l'SQL e i parametri
            logger.error("tool_fallito", extra={"tool": tool.name, "errore": type(e).__name__})
            return fallito(tool.name)

    return StructuredTool.from_function(
        coroutine=chiama, name=tool.name, description=tool.description, args_schema=tool.args_model
    )


def in_fila(tools: list[Tool]) -> list[StructuredTool]:
    """I tool di un'esecuzione, con una fila sola: costruiti insieme, per la stessa richiesta."""
    fila = asyncio.Lock()
    return [a_langchain(t, fila) for t in tools]


def quando(tool: Tool) -> Callable[[ToolCallRequest], bool]:
    """La regola della sosta, per il middleware: decide sugli argomenti VALIDATI, e nel dubbio
    chiede. `when` riceve gli argomenti grezzi: un importo illeggibile non è un importo piccolo."""

    def when(req: ToolCallRequest) -> bool:
        try:
            args = tool.args_model.model_validate(req.tool_call["args"])
        except ValidationError:
            return True
        return tool.serve_approvazione is None or tool.serve_approvazione(args)

    return when
