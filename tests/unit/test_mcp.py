# tests/unit/test_mcp.py — il server che si descrive, verifica chi chiama, e rispetta il muro
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import jwt
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.types import TextContent
from sqlalchemy import text

from liparibank_mcp import server
from src.agents.loop import run_agent
from src.agents.mcp_client import client_per, tools_dal_server
from src.auth.tokens import create_access_token
from src.db.session import engine
from tests.banco_app import prepara_db
from tests.conftest import CONTO_ALTRUI, CONTO_DI_MARCO, risposta
from tests.finti import embedder_finto

RADICE = Path(__file__).resolve().parents[2]


def _falso(ruolo: str) -> str:
    """Un token con il ruolo che il chiamante preferisce, firmato con un segreto qualunque."""
    return jwt.encode(
        {"sub": "mbianchi", "role": ruolo},
        "un-altro-segreto-lungo-abbastanza-32b",
        algorithm="HS256",
    )


async def test_il_server_si_descrive() -> None:
    async with Client(server.mcp) as c:  # in memoria: niente processo
        tools = {t.name: t for t in await c.list_tools()}
    assert set(tools) == {"search_policy", "find_customer_accounts", "get_account_balance"}
    assert "Usalo quando" in (tools["search_policy"].description or "")
    assert all(t.annotations and t.annotations.read_only_hint for t in tools.values())


@pytest.mark.parametrize("token", ["", "non-un-token", _falso("compliance_lead")])
async def test_un_identita_non_verificata_non_passa(
    token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIPARI_TOKEN", token)
    async with Client(server.mcp) as c:
        esito = await c.call_tool(
            "search_policy", {"query": "soglie paesi a rischio"}, raise_on_error=False
        )
    assert esito.is_error and "Identità non verificata" in str(esito.content)


async def test_via_stdio_il_server_rispetta_il_portafoglio(tmp_path: Path) -> None:
    db = tmp_path / "banca.db"
    await prepara_db(db)
    ambiente = {
        **os.environ,
        "DATABASE_URL": f"sqlite+aiosqlite:///{db}",
        "LIPARI_TOKEN": create_access_token("mbianchi", "operator"),
        # il server costruisce i suoi client anche per gli strumenti che non chiamano il modello:
        # una chiave finta basta, e nessuna chiamata la usa
        "OPENAI_API_KEY": "sk-test-mai-chiamata",
    }
    trasporto = StdioTransport(
        command=sys.executable, args=["-m", "liparibank_mcp.server"], env=ambiente, cwd=str(RADICE)
    )
    async with Client(trasporto) as c:  # un processo vero, figlio del test
        proprio = await c.call_tool("get_account_balance", {"account_id": CONTO_DI_MARCO})
        altrui = await c.call_tool("get_account_balance", {"account_id": CONTO_ALTRUI})
    testi = [c.content[0] for c in (proprio, altrui)]
    assert all(isinstance(x, TextContent) for x in testi)
    assert [x.text for x in testi if isinstance(x, TextContent)] == [
        f"Saldo di {CONTO_DI_MARCO}: 48200.00 EUR",
        "Non risulta nel portafoglio di questo operatore.",
    ]


async def test_il_loop_del_giorno_7_usa_il_server_senza_cambiare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIPARI_TOKEN", create_access_token("mbianchi", "operator"))
    # il server, qui nello stesso processo, costruisce l'embedder con la factory: si sostituisce
    # dove è usato, in src.agents.deps, non dove è definito
    monkeypatch.setattr("src.agents.deps.get_embedder", embedder_finto)
    monkeypatch.setattr("src.agents.deps.get_openai", AsyncMock)
    async with engine.begin() as conn:  # nessun documento: la risposta attesa è «niente»
        await conn.execute(text("TRUNCATE document_chunks"))
    modello = AsyncMock()
    modello.chat.completions.create.side_effect = [
        risposta(tool="search_policy", argomenti='{"query": "soglie paesi a rischio"}'),
        risposta(testo="Nei documenti non c'è una policy su questo."),
    ]
    async with Client(server.mcp) as c:
        tools = await tools_dal_server(c, nomi={"search_policy"})
        run = await run_agent(
            messaggi=[{"role": "user", "content": "soglie?"}],
            tools=tools,
            client=modello,
            model="gpt-4o-mini",
        )
    assert tools[0].scrive is False  # dichiarato dal server
    osservazione = modello.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert osservazione == "Niente nei documenti visibili a questo ruolo."
    assert (run.stopped_by, run.tool_calls) == ("model", ["search_policy"])


def test_il_client_passa_il_token_nell_ambiente_del_server() -> None:
    trasporto = client_per("tok").transport
    assert isinstance(trasporto, StdioTransport)
    assert trasporto.env is not None and trasporto.env["LIPARI_TOKEN"] == "tok"
