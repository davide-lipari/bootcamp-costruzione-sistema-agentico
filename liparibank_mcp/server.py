# liparibank_mcp/server.py — i tool del Giorno 7, esposti con il protocollo MCP
import os

from fastapi import HTTPException
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ValidationError

from src.agents.deps import crea_deps
from src.agents.tools import ClienteArgs, RicercaArgs, SaldoArgs, build_tools_for
from src.auth.deps import UserContext
from src.auth.tokens import decode_token
from src.db.session import AsyncSessionLocal

mcp = FastMCP("liparibank")
SOLA_LETTURA = {"readOnlyHint": True}  # lo leggono i client: questi tool non scrivono


def identita() -> UserContext:
    """Chi sta chiamando. Non lo dice un parametro: lo dice un token firmato, e qui si verifica."""
    token = os.environ.get("LIPARI_TOKEN", "")
    try:
        payload = decode_token(token)  # firma e scadenza, come al Giorno 6
    except HTTPException as e:
        raise ToolError(f"Identità non verificata: {e.detail}.") from e
    return UserContext(username=str(payload["sub"]), role=str(payload.get("role", "public")))


async def _esegui(nome: str, args: BaseModel) -> str:
    utente = identita()  # a ogni chiamata: un token scade
    async with AsyncSessionLocal() as db:  # la sessione è del server, l'identità no
        tool = next(t for t in build_tools_for(utente, crea_deps(db)) if t.name == nome)
        return await tool.run(args)


def _argomenti[M: BaseModel](modello: type[M], **valori: object) -> M:
    """I vincoli dei modelli del Giorno 7 valgono anche qui: un client esterno li rispetta."""
    try:
        return modello.model_validate(valori)
    except ValidationError as e:
        raise ToolError(f"Argomenti non validi: {e.errors(include_url=False)}") from e


@mcp.tool(annotations=SOLA_LETTURA)
async def search_policy(query: str) -> str:
    """Cerca nelle procedure e nelle policy di LipariBank, con i permessi di chi chiama.

    Usalo quando la domanda riguarda regole, soglie, procedure o adempimenti della banca.
    Non usarlo per dati di clienti o di conti.

    Args:
        query: la domanda in forma completa, non una parola sola.
    """
    return await _esegui("search_documents", _argomenti(RicercaArgs, query=query))


@mcp.tool(annotations=SOLA_LETTURA)
async def find_customer_accounts(customer_id: str) -> str:
    """I conti di un cliente, con IBAN ed etichetta, se il cliente è nel portafoglio di chi chiama.

    Usalo quando la domanda nomina un cliente e non un IBAN.

    Args:
        customer_id: il codice cliente, come compare nella sua scheda.
    """
    return await _esegui("find_customer_accounts", _argomenti(ClienteArgs, customer_id=customer_id))


@mcp.tool(annotations=SOLA_LETTURA)
async def get_account_balance(account_id: str) -> str:
    """Saldo disponibile di un conto, se il conto è di un cliente nel portafoglio di chi chiama.

    Args:
        account_id: l'IBAN del conto, come lo restituisce find_customer_accounts.
    """
    return await _esegui("get_account_balance", _argomenti(SaldoArgs, account_id=account_id))


if __name__ == "__main__":
    mcp.run()  # transport stdio, il default
