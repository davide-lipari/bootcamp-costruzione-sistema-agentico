# src/agents/mcp_http.py — un server MCP pubblicato, consumato dal grafo: token e nomi suoi
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.exceptions import MCPError

from src.agents.mcp_client import tools_dal_server
from src.agents.registry import Tool
from src.exceptions import AppError

logger = logging.getLogger(__name__)


class ServerMcpNonDisponibileError(AppError):
    """Il server non risponde, o rifiuta il token: un guasto nostro, non un'osservazione per il
    modello. Se diventasse testo, il modello riproverebbe in altri modi e inventerebbe."""

    def __init__(self) -> None:
        super().__init__(503, "MCP_NON_DISPONIBILE", "Il servizio dei conti non è disponibile.")


def trasporto_per(url: str, token: str) -> StreamableHttpTransport:
    """HTTP con il token dell'operatore emesso per QUEL server, mai il token della API."""
    return StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"})


@asynccontextmanager
async def collegati(
    trasporto: StreamableHttpTransport,
) -> AsyncIterator[Client[StreamableHttpTransport]]:
    """Una sessione con il server: si apre qui e si chiude all'uscita. Un rifiuto all'apertura
    diventa un 503 per chi chiama, prima che il ciclo cominci. Si apre per richiesta, con un
    token appena scambiato: un token di dieci minuti tenuto per una conversazione scade a metà,
    e allora il rifiuto arriva al ciclo come l'errore di un tool."""
    pila = AsyncExitStack()
    try:
        client = await pila.enter_async_context(Client(trasporto))
    except (MCPError, OSError) as e:
        # il client non distingue un 401 da un server giù: tutti e due dicono «non si entra»
        logger.error(
            "mcp_non_disponibile", extra={"url": trasporto.url, "errore": type(e).__name__}
        )
        raise ServerMcpNonDisponibileError() from e
    async with pila:
        yield client


async def tools_remoti(client: Client[Any], prefisso: str) -> list[Tool]:
    """I tool del server, con il nome prefissato: due server con un `search` non si coprono."""
    return [replace(t, name=f"{prefisso}__{t.name}") for t in await tools_dal_server(client)]
