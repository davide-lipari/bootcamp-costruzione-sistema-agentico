# tests/test_g4_strada_c.py — find_customer_accounts v2 rotta, affiancata alla v1, con la disdetta
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx2
from fastmcp import FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent

from evals import superficie
from liparibank_mcp import pubblico
from liparibank_mcp.pubblico import crea_server
from src.agents.mcp_http import collegati
from src.auth.deps import UserContext
from src.auth.mcp_tokens import RISORSA_MCP, emetti
from tests.conftest import CLIENTE_ALTRUI, CLIENTE_DI_MARCO, CONTO_DI_MARCO
from tests.test_g1_langchain import banca  # noqa: F401 — il fixture autouse del banco

__all__ = ["banca"]

RADICE = Path(__file__).resolve().parents[1]
BASE_MCP = RISORSA_MCP.removesuffix("/mcp")
MARCO = UserContext(username="mbianchi", role="operator")


def token_mcp(client_id: str = "assistente-filiale", user: UserContext = MARCO) -> str:
    return emetti(user, client_id, RISORSA_MCP)[0]


@asynccontextmanager
async def server_mcp() -> AsyncIterator[
    tuple[FastMCP, Callable[[str | None], StreamableHttpTransport]]
]:
    """Il server pubblicato, su HTTP vero ma nello stesso processo, come in test_g4."""
    mcp = crea_server(BASE_MCP)
    app = mcp.http_app(path="/mcp")

    def trasporto(token: str | None) -> StreamableHttpTransport:
        def fabbrica(
            headers: dict[str, str] | None = None,
            timeout: httpx2.Timeout | None = None,
            auth: httpx2.Auth | None = None,
            **_: object,
        ) -> httpx2.AsyncClient:
            return httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url=BASE_MCP,
                headers=headers,
                timeout=timeout,
                auth=auth,
            )

        intestazioni = {"Authorization": f"Bearer {token}"} if token else None
        return StreamableHttpTransport(
            RISORSA_MCP, headers=intestazioni, httpx_client_factory=fabbrica
        )

    async with app.router.lifespan_context(app):
        yield mcp, trasporto


def leggi_testo(esito: CallToolResult) -> str:
    return " ".join(c.text for c in esito.content if isinstance(c, TextContent))


async def test_le_due_versioni_di_find_customer_accounts_convivono() -> None:
    """Sotto lo stesso nome servono due forme diverse: la 1 è il testo di prima, la 2 è JSON.

    Il numero della strada è la convivenza: entrambe le versioni sono registrate, e un client
    che chiede esplicitamente la 1 ottiene la 1, non la 2. Se togli la registrazione della v1,
    `version="1"` non trova più niente e questo test cade.
    """
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            [tool] = [t for t in await c.list_tools() if t.name == "find_customer_accounts"]
            vecchia = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_DI_MARCO}, version="1"
            )
            nuova = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_DI_MARCO}, version="2"
            )
    # l'elenco dichiara entrambe le versioni, come per search_policy
    assert (tool.meta or {})["fastmcp"]["versions"] == ["2", "1"]
    # la 1: il testo di sempre, il contratto che un client vecchio si aspetta
    assert leggi_testo(vecchia).startswith(f"Conti del cliente {CLIENTE_DI_MARCO}:")
    assert f"- {CONTO_DI_MARCO}" in leggi_testo(vecchia)
    # la 2: un documento JSON, campi che un programma legge senza fare il parse del testo
    documento = json.loads(leggi_testo(nuova))
    assert documento["cliente"] == CLIENTE_DI_MARCO
    assert CONTO_DI_MARCO in [conto["iban"] for conto in documento["conti"]]


async def test_senza_versione_arriva_la_piu_alta_e_questo_rompe_il_vecchio() -> None:
    """Il cambio che rompe, dimostrato: chi non sceglie la versione prende la 2, e la 2 non
    è il testo di prima. È qui che un client vecchio si rompe, e perché la v1 resta servita.

    Protegge: la scelta della versione di default (la più alta) e il fatto che la v1 sia
    ancora registrata. Se togli la v1, il `version="1"` qui sotto non trova più la versione.
    """
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            predefinita = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_DI_MARCO}
            )
            esplicita = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_DI_MARCO}, version="1"
            )
    # senza `version`, fastmcp dà la più alta: la 2, che è JSON — un client che parsava il
    # testo di prima si rompeva qui; per questo la v1 resta chiedibile
    assert leggi_testo(predefinita).startswith("{")
    assert "cliente" in json.loads(leggi_testo(predefinita))
    # mentre la 1, chiesta apposta, resta il testo di prima: il vecchio contratto regge
    assert leggi_testo(esplicita).startswith(f"Conti del cliente {CLIENTE_DI_MARCO}:")


async def test_il_muro_del_portafoglio_vale_anche_sulla_v2() -> None:
    """Il cliente altrui, sulla versione JSON, ha conti vuoti: il muro non salta perché la
    forma della risposta è cambiata. Senza la chiamata a `of_customer` con l'identità dal
    token, qui passerebbero i conti di un altro."""
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            proprio = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_DI_MARCO}, version="2"
            )
            altrui = await c.call_tool(
                "find_customer_accounts", {"customer_id": CLIENTE_ALTRUI}, version="2"
            )
    assert CONTO_DI_MARCO in [conto["iban"] for conto in json.loads(leggi_testo(proprio))["conti"]]
    assert json.loads(leggi_testo(altrui))["conti"] == []  # non nel portafoglio: niente


def test_la_voce_di_registry_dice_le_versioni_e_la_disdetta() -> None:
    """La voce dichiara quali versioni si servono (2 e 1) e fino a quando la 1 (data di
    disdetta). Il server non la mente: `come_e_fatto` legge la 2 e la 1 dai tool registrati,
    e il test la collega al server. Se togli la v1 dal server, `versioni_servite` non la
    include più e questo test cade."""
    voce = json.loads((RADICE / "registry" / "server.json").read_text(encoding="utf-8"))
    governo = voce["_meta"]["io.modelcontextprotocol.registry/publisher-provided"][
        "it.liparibank/governo"
    ]
    assert governo["versioni_servite"]["find_customer_accounts"] == ["2", "1"]
    # la convivenza ha un numero: la data oltre la quale la 1 si può togliere
    disdetta = governo["disdette"]["find_customer_accounts"]["1"]
    assert date.fromisoformat(disdetta) > date.today()


async def test_la_disdetta_registrata_corrisponde_al_server_vivo() -> None:
    """Collega la voce al server: le versioni che il registry promette sono proprio quelle che
    il server, costruito adesso, registra davvero. È il test che si accorge se menti."""
    mcp = crea_server(BASE_MCP)
    fatto = await pubblico.come_e_fatto(mcp)
    voce = json.loads((RADICE / "registry" / "server.json").read_text(encoding="utf-8"))
    governo = voce["_meta"]["io.modelcontextprotocol.registry/publisher-provided"][
        "it.liparibank/governo"
    ]
    assert governo["versioni_servite"] == fatto["versioni_servite"]
    # ogni versione disdetta è una che il server serve ancora: la disdetta promette, non inventa
    for nome, per_versione in governo["disdette"].items():
        for versione in per_versione:
            assert versione in fatto["versioni_servite"][nome]


async def test_superficie_c_vede_le_due_versioni_sotto_lo_stesso_tool() -> None:
    """La superficie dopo l'estensione: find_customer_accounts ora elenca v2/1, come
    search_policy. Un solo client (assistente-filiale) e un solo ruolo (operator) lo vedono,
    e resta in sola lettura."""
    righe = {(r.client_id, r.ruolo): r for r in await superficie.misura()}
    assert righe["assistente-filiale", "operator"].tool == (
        "find_customer_accounts v2/1",
        "get_account_balance",
        "search_policy v2/1",
    )
    # il tool che ora ha due versioni resta solo di assistente-filiale/operator: nessun altro
    # ruolo o client lo vede, e nessun tool scrive
    altri = {
        k: r.tool for k, r in righe.items() if k != ("assistente-filiale", "operator") and r.tool
    }
    assert all("find_customer_accounts" not in " ".join(t) for t in altri.values())
    assert sum(r.scrivono for r in righe.values()) == 0
