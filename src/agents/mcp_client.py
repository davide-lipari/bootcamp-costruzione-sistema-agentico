# src/agents/mcp_client.py — i tool di un server MCP, tradotti nel Tool del Giorno 7
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.types import TextContent
from pydantic import BaseModel, Field, create_model

from src.agents.registry import Tool

logger = logging.getLogger(__name__)

TIPI: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


def client_per(token: str) -> Client[Any]:
    """Il server gira come processo figlio, con il token dell'utente nel suo ambiente."""
    return Client(
        StdioTransport(
            command=sys.executable,
            args=["-m", "liparibank_mcp.server"],
            env={**os.environ, "LIPARI_TOKEN": token},
        )
    )


async def tools_dal_server(client: Client[Any], nomi: set[str] | None = None) -> list[Tool]:
    """Chiede l'elenco al server e ne fa dei Tool: il ciclo del Giorno 7 non se ne accorge."""
    tradotti: list[Tool] = []
    for t in await client.list_tools():
        if nomi is not None and t.name not in nomi:
            continue
        modello = modello_da_schema(t.name, t.input_schema)
        if modello is None:
            # uno schema che non sappiamo tradurre è un tool in meno, non un agente rotto
            logger.warning("mcp_tool_ignorato", extra={"tool": t.name})
            continue
        sola_lettura = bool(t.annotations and t.annotations.read_only_hint)
        tradotti.append(
            Tool(
                name=t.name,
                description=t.description or "",  # la docstring del server: la scrive lui
                args_model=modello,
                run=_chiamata(client, t.name),
                scrive=not sola_lettura,  # senza dichiarazione, si assume che scriva
                # senza_dati resta vuoto: il protocollo non dice quali risposte sono «nessun
                # dato», e nel log una risposta vuota del server conta come "ok"
            )
        )
    return tradotti


def modello_da_schema(nome: str, schema: dict[str, Any]) -> type[BaseModel] | None:
    """Un modello Pydantic da uno schema JSON piatto. None se non è piatto."""
    obbligatori = set(schema.get("required", []))
    campi: dict[str, Any] = {}
    for campo, prop in schema.get("properties", {}).items():
        tipo = TIPI.get(prop.get("type", ""))
        if tipo is None:
            return None
        if campo in obbligatori:
            campi[campo] = (tipo, Field(description=prop.get("description")))
        else:
            campi[campo] = (
                tipo | None,
                Field(prop.get("default"), description=prop.get("description")),
            )
    modello: type[BaseModel] = create_model(f"Args_{nome}", **campi)
    return modello


def _chiamata(client: Client[Any], nome: str) -> Callable[[BaseModel], Awaitable[str]]:
    # una funzione che fabbrica la funzione: `nome` resta quello di QUESTO giro del ciclo.
    # Scritta dentro il for, ogni tool chiamerebbe l'ultimo
    async def run(args: BaseModel) -> str:
        esito = await client.call_tool(
            nome, args.model_dump(exclude_none=True), raise_on_error=False
        )
        testo = "\n".join(c.text for c in esito.content if isinstance(c, TextContent))
        return f"ERRORE dal server MCP: {testo}" if esito.is_error else testo

    return run
