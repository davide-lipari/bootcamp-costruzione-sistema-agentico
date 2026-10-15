# tests/test_g4_pubblicare.py — il server pubblicato, i suoi token, le versioni, la carta, la delega
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import httpx2
import jsonschema
import jwt
import pytest
from a2a.client import Client as ClientA2A
from a2a.client import ClientCallContext, ClientConfig, create_client
from a2a.client.auth.credentials import InMemoryContextCredentialStore
from a2a.client.auth.interceptor import AuthInterceptor
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, Task, TaskState
from a2a.utils.errors import TaskNotFoundError, UnsupportedOperationError
from fastapi import FastAPI
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.types import TextContent
from sqlalchemy import select

from evals import superficie
from liparibank_a2a import agente as modulo_agente
from liparibank_a2a.agente import NON_RIUSCITA, Rispondi, crea_app
from liparibank_mcp import pubblico
from liparibank_mcp.pubblico import crea_server
from src.agents.mcp_http import ServerMcpNonDisponibileError, collegati, tools_remoti
from src.auth.deps import UserContext, get_current_user
from src.auth.mcp_tokens import (
    DURATA,
    EMITTENTE,
    RISORSA_A2A,
    RISORSA_MCP,
    SCOPE_DEL_RUOLO,
    TokenNegatoError,
    emetti,
    segreto_del_client,
)
from src.auth.passwords import verify_password
from src.auth.tokens import ALGORITHM, create_access_token
from src.config import settings
from src.db.models import AppUser
from src.db.session import AsyncSessionLocal
from src.lg.delega import PERSO, TASK_PERSO, TROPPI_GIRI, costruisci_delega
from src.main import app as api
from src.services.ingest_service import IngestService
from tests.conftest import CONTO_ALTRUI, CONTO_DI_MARCO
from tests.finti import completion, embedder_finto, openai_finto
from tests.test_g1_langchain import Copione, banca, chat

__all__ = ["banca"]

RADICE = Path(__file__).resolve().parents[1]
BASE_MCP = RISORSA_MCP.removesuffix("/mcp")
MARCO = UserContext(username="mbianchi", role="operator")
LEAD = UserContext(username="lverdi", role="risk_lead")  # un utente del seed di sviluppo
GIULIA = UserContext(username="grossi", role="compliance_lead")


def token_fatto_a_mano(**campi: object) -> str:
    """Un token firmato con la chiave giusta, ma con i campi che il test decide."""
    adesso = datetime.now(UTC)
    payload: dict[str, object] = {
        "iss": EMITTENTE,
        "sub": "mbianchi",
        "role": "operator",
        "aud": RISORSA_MCP,
        "scope": "conti:leggi policy:leggi istruttoria:chiedi",
        "iat": adesso,
        "exp": adesso + timedelta(minutes=5),
    }
    payload.update(campi)
    return jwt.encode(
        {k: v for k, v in payload.items() if v is not None}, settings.jwt_secret, ALGORITHM
    )


def token_mcp(client_id: str = "assistente-filiale", user: UserContext = MARCO) -> str:
    return emetti(user, client_id, RISORSA_MCP)[0]


def token_a2a(user: UserContext = MARCO) -> str:
    return emetti(user, "assistente-filiale", RISORSA_A2A)[0]


@asynccontextmanager
async def server_mcp() -> AsyncIterator[
    tuple[FastMCP, Callable[[str | None], StreamableHttpTransport]]
]:
    """Il server pubblicato, su HTTP vero ma nello stesso processo e nello stesso loop."""
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


def contenuto(esito: CallToolResult) -> list[str]:
    return [c.text for c in esito.content if isinstance(c, TextContent)]


# ---------------------------------------------------------------- i token: per chi, per cosa
def test_lo_scope_e_l_intersezione_fra_client_ruolo_e_servizio() -> None:
    assert emetti(MARCO, "supporto-procedure", RISORSA_MCP)[1] == {"policy:leggi"}
    # lo stesso client, lo stesso operatore: ogni servizio riceve solo gli scope suoi
    assert emetti(MARCO, "assistente-filiale", RISORSA_MCP)[1] == {"conti:leggi", "policy:leggi"}
    assert emetti(MARCO, "assistente-filiale", RISORSA_A2A)[1] == {"istruttoria:chiedi"}
    # un client largo non allarga lei
    assert emetti(LEAD, "assistente-filiale", RISORSA_MCP)[1] == {"policy:leggi"}
    assert emetti(GIULIA, "agente-compliance", RISORSA_A2A)[1] == {"istruttoria:chiedi"}


def test_il_token_e_breve_e_dice_per_chi_e_per_quale_client() -> None:
    token, _ = emetti(MARCO, "supporto-procedure", RISORSA_MCP)
    dati = jwt.decode(token, settings.jwt_secret, [ALGORITHM], audience=RISORSA_MCP)
    assert dati["exp"] - dati["iat"] == DURATA.total_seconds() == 600
    assert (dati["sub"], dati["azp"], dati["iss"]) == ("mbianchi", "supporto-procedure", EMITTENTE)


@pytest.mark.parametrize(
    ("client_id", "risorsa", "ruolo"),
    [
        ("sconosciuto", RISORSA_MCP, "operator"),
        ("supporto-procedure", RISORSA_A2A, "operator"),  # un servizio che non è suo
        ("agente-compliance", RISORSA_MCP, "compliance_lead"),  # nemmeno per lui
        ("assistente-filiale", RISORSA_MCP, "public"),  # niente da concedere
        ("assistente-filiale", RISORSA_A2A, "risk_lead"),  # niente da concedere su quel servizio
    ],
)
def test_lo_scambio_si_nega(client_id: str, risorsa: str, ruolo: str) -> None:
    with pytest.raises(TokenNegatoError):
        emetti(UserContext(username="mbianchi", role=ruolo), client_id, risorsa)


async def test_l_endpoint_scambia_e_ogni_token_vale_solo_per_il_suo_servizio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # i test del repository sostituiscono l'utente: qui serve l'autenticazione vera
    monkeypatch.delitem(api.dependency_overrides, get_current_user, raising=False)
    rest = create_access_token("mbianchi", "operator")
    supporto = {
        "client_id": "supporto-procedure",
        "client_secret": segreto_del_client("supporto-procedure"),
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://t") as h:
        r = await h.post(
            "/api/auth/token-mcp",
            json={**supporto, "resource": RISORSA_MCP},
            headers={"Authorization": f"Bearer {rest}"},
        )
        assert r.status_code == 200 and r.json()["scope"] == "policy:leggi"
        mcp = r.json()["access_token"]
        # il token del server MCP non apre la API: ha un'audience, e non è la sua
        r2 = await h.post(
            "/api/auth/token-mcp",
            json={**supporto, "resource": RISORSA_MCP},
            headers={"Authorization": f"Bearer {mcp}"},
        )
        assert r2.status_code == 401
        negato = await h.post(
            "/api/auth/token-mcp",
            json={**supporto, "resource": RISORSA_A2A},
            headers={"Authorization": f"Bearer {rest}"},
        )
        assert negato.status_code == 403


@pytest.mark.parametrize(
    "client",
    [
        # il supporto si spaccia per l'assistente: il nome è giusto, il segreto è il suo
        {
            "client_id": "assistente-filiale",
            "client_secret": segreto_del_client("supporto-procedure"),
        },
        {"client_id": "assistente-filiale", "client_secret": "indovinato"},
        {"client_id": "sconosciuto", "client_secret": segreto_del_client("sconosciuto")},
    ],
    ids=["spacciato", "segreto-sbagliato", "sconosciuto"],
)
async def test_un_client_che_non_prova_chi_e_non_riceve_niente(
    client: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(api.dependency_overrides, get_current_user, raising=False)
    rest = create_access_token("mbianchi", "operator")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://t") as h:
        r = await h.post(
            "/api/auth/token-mcp",
            json={**client, "resource": RISORSA_MCP},
            headers={"Authorization": f"Bearer {rest}"},
        )
    assert r.status_code == 401 and "access_token" not in r.json()


# ---------------------------------------------------------------- il server pubblicato
@pytest.mark.parametrize(
    "token",
    # nessuno; quello della API REST; uno della stessa banca, ma per l'agente A2A
    [None, create_access_token("mbianchi", "operator"), "A2A"],
    ids=["senza-token", "token-della-api", "token-di-un-altro-servizio"],
)
async def test_al_server_si_entra_solo_con_un_token_emesso_per_lui(token: str | None) -> None:
    token = token_a2a() if token == "A2A" else token
    async with server_mcp() as (_, trasporto):
        with pytest.raises(ServerMcpNonDisponibileError):
            async with collegati(trasporto(token)):
                pass


@pytest.mark.parametrize(
    "campi",
    [{"iss": "http://un-altro-emittente/"}, {"exp": None}],
    ids=["altro-emittente", "senza-scadenza"],
)
async def test_firma_giusta_non_basta(campi: dict[str, object]) -> None:
    async with server_mcp() as (_, trasporto):
        with pytest.raises(ServerMcpNonDisponibileError):
            async with collegati(trasporto(token_fatto_a_mano(**campi))):
                pass


async def test_ogni_client_vede_i_tool_del_suo_scope() -> None:
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp("supporto-procedure"))) as c:
            assert {t.name for t in await c.list_tools()} == {"search_policy"}
            # quello che non vede non lo può chiamare, e il rifiuto non dice che esiste
            esito = await c.call_tool(
                "get_account_balance", {"account_id": CONTO_DI_MARCO}, raise_on_error=False
            )
            assert esito.is_error and "Unknown tool" in contenuto(esito)[0]
        async with collegati(trasporto(token_mcp())) as c:
            assert {t.name for t in await c.list_tools()} == {
                "search_policy",
                "find_customer_accounts",
                "get_account_balance",
            }


async def test_le_due_versioni_convivono(monkeypatch: pytest.MonkeyPatch) -> None:
    # il server costruisce l'embedder con la factory: si sostituisce dove è usato
    monkeypatch.setattr("src.agents.deps.get_embedder", embedder_finto)
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            [policy] = [t for t in await c.list_tools() if t.name == "search_policy"]
            nuova = await c.call_tool(
                "search_policy", {"query": "soglie", "quanti": 9}, raise_on_error=False
            )
            vecchia = await c.call_tool(
                "search_policy", {"query": "soglie", "quanti": 2}, version="1", raise_on_error=False
            )
            per_la_vecchia = await c.call_tool("search_policy", {"query": "soglie"}, version="1")
    # l'elenco mostra la versione più alta, e dice quali altre il server serve ancora
    assert set(policy.input_schema["properties"]) == {"query", "quanti"}
    assert (policy.meta or {})["fastmcp"]["versions"] == ["2", "1"]
    assert nuova.is_error and "quanti" in contenuto(nuova)[0]  # il tetto vale anche da fuori
    assert vecchia.is_error and "unexpected_keyword_argument" in contenuto(vecchia)[0]
    assert not per_la_vecchia.is_error  # chi chiede la 1 ottiene la 1


async def test_il_muro_vale_anche_da_fuori() -> None:
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            proprio = await c.call_tool("get_account_balance", {"account_id": CONTO_DI_MARCO})
            altrui = await c.call_tool("get_account_balance", {"account_id": CONTO_ALTRUI})
    assert contenuto(proprio) == [f"Saldo di {CONTO_DI_MARCO}: 48200.00 EUR"]
    assert contenuto(altrui) == ["Non risulta nel portafoglio di questo operatore."]


async def test_anche_i_documenti_hanno_il_muro_da_fuori(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.agents.deps.get_embedder", embedder_finto)
    riservato = "# Controparti\nBonifico SEPA verso controparti segnalate: verifica rafforzata."
    async with AsyncSessionLocal() as s:
        await IngestService(s, embedder_finto()).ingest(
            "controparti_riservato", riservato, visibility="compliance_only"
        )
    domanda = {"query": "bonifico SEPA controparti segnalate", "quanti": 5}
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:  # l'operatore
            operatore = " ".join(contenuto(await c.call_tool("search_policy", domanda)))
        async with collegati(trasporto(token_mcp(user=GIULIA))) as c:  # la compliance
            compliance = " ".join(contenuto(await c.call_tool("search_policy", domanda)))
    # stesso client, stesso tool: il ruolo arriva dal token, e decide cosa si legge
    assert "controparti_riservato" not in operatore
    assert "controparti_riservato" in compliance


def test_senza_un_token_non_c_e_nessuno() -> None:
    with pytest.raises(ToolError, match="Identità non verificata"):
        pubblico.identita()  # fuori da una richiesta HTTP: nessun token, nessun operatore


async def test_in_memoria_nessuno_e_qualcuno() -> None:
    # il client in memoria non passa da HTTP: niente token, e il server lo tratta da estraneo
    async with Client(crea_server(BASE_MCP)) as c:
        assert {t.name for t in await c.list_tools()} == set()  # nessuno scope, nessun tool


async def test_un_eccezione_non_dice_niente_di_noi(monkeypatch: pytest.MonkeyPatch) -> None:
    def rotto(*_: object, **__: object) -> None:
        raise RuntimeError("postgresql://utente:password@db/banca")

    monkeypatch.setattr(pubblico, "crea_deps", rotto)
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp())) as c:
            esito = await c.call_tool(
                "find_customer_accounts", {"customer_id": "C-10234"}, raise_on_error=False
            )
    assert esito.is_error and "postgresql" not in " ".join(contenuto(esito))


def test_due_tool_con_lo_stesso_nome_fermano_l_avvio() -> None:
    mcp = crea_server(BASE_MCP)
    with pytest.raises(ValueError, match="already exists"):

        @mcp.tool(name="get_account_balance")
        def doppione(account_id: str) -> str:
            return account_id


async def test_i_tool_remoti_hanno_il_nome_del_server_davanti() -> None:
    async with server_mcp() as (_, trasporto):
        async with collegati(trasporto(token_mcp("supporto-procedure"))) as c:
            tools = await tools_remoti(c, "banca")
    assert [t.name for t in tools] == ["banca__search_policy"]
    assert tools[0].scrive is False


# ---------------------------------------------------------------- la voce di registry
async def test_la_voce_di_registry_e_valida_e_dice_il_vero() -> None:
    schema = json.loads((RADICE / "registry" / "server.schema.json").read_text(encoding="utf-8"))
    voce = json.loads((RADICE / "registry" / "server.json").read_text(encoding="utf-8"))
    jsonschema.Draft7Validator(schema).validate(voce)
    assert voce["$schema"] == schema["$id"]  # la voce si valida contro lo schema che dichiara
    governo = voce["_meta"]["io.modelcontextprotocol.registry/publisher-provided"][
        "it.liparibank/governo"
    ]
    async with server_mcp() as (mcp, trasporto):
        fatto = await pubblico.come_e_fatto(mcp)  # letto dai tool registrati, non da un elenco
        async with collegati(trasporto(token_mcp())) as c:
            presentato = c.server_info  # dall'apertura della sessione, come la vede il client
            policy = next(t for t in await c.list_tools() if t.name == "search_policy")
    # la voce non mente: versione, scope, sola lettura e versioni servite sono quelle del server
    assert presentato is not None and presentato.version == voce["version"]
    assert voce["version"] == fatto["version"]
    assert governo["scope"] == fatto["scope"] == sorted(pubblico.SCOPE)
    assert governo["sola_lettura"] is fatto["sola_lettura"]
    assert governo["versioni_servite"] == fatto["versioni_servite"]
    assert fatto["versioni_servite"]["search_policy"] == (policy.meta or {})["fastmcp"]["versions"]


# ---------------------------------------------------------------- la superficie
async def test_la_superficie_e_quella_registrata() -> None:
    righe = {(r.client_id, r.ruolo): r for r in await superficie.misura()}
    policy = "search_policy v2/1"
    conti = "find_customer_accounts v2/1"
    assert righe["assistente-filiale", "operator"].tool == (
        conti,
        "get_account_balance",
        policy,
    )
    assert righe["assistente-filiale", "compliance_lead"].tool == (policy,)  # niente conti
    assert all(righe["supporto-procedure", r].tool == (policy,) for r in SCOPE_DEL_RUOLO)
    # all'agente arriva solo chi ha l'istruttoria nel ruolo e un client registrato per lui
    assert {k for k, r in righe.items() if r.agente} == {
        ("assistente-filiale", "operator"),
        ("assistente-filiale", "compliance_lead"),
        ("agente-compliance", "operator"),
        ("agente-compliance", "compliance_lead"),
    }
    # l'agente della Compliance arriva all'istruttoria, e al server dei conti no
    assert all(righe["agente-compliance", r].tool == () for r in SCOPE_DEL_RUOLO)
    assert sum(r.scrivono for r in righe.values()) == 0
    assert superficie.rapporto(list(righe.values())).endswith("tool esposti in tutto: 8")


# ---------------------------------------------------------------- l'agente A2A
def agente(rispondi: Rispondi) -> FastAPI:
    return crea_app(rispondi, base_url=RISORSA_A2A)


@asynccontextmanager
async def client_a2a(app: FastAPI, token: str | None) -> AsyncIterator[ClientA2A]:
    intestazioni = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=RISORSA_A2A, headers=intestazioni
    ) as h:
        yield await create_client(RISORSA_A2A, ClientConfig(httpx_client=h, streaming=False))


async def manda(client: ClientA2A, testo: str, task_id: str = "", context_id: str = "") -> Task:
    m = new_text_message(testo, role=Role.ROLE_USER)
    if task_id:
        m.task_id, m.context_id = task_id, context_id
    ultimo = None
    async for r in client.send_message(SendMessageRequest(message=m)):
        ultimo = r
    assert ultimo is not None
    return ultimo.task


async def test_la_carta_e_quella_della_versione_1() -> None:
    async def mai(user: UserContext, testo: str) -> str:
        raise AssertionError

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agente(mai)), base_url=RISORSA_A2A
    ) as h:
        carta = (await h.get("/.well-known/agent-card.json")).json()  # pubblica: si scopre
        senza = await h.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"})
        con_mcp = await h.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"},
            headers={"Authorization": f"Bearer {token_mcp()}"},
        )
        # per l'agente e con la firma giusta, ma senza lo scope dell'istruttoria
        solo_policy = token_fatto_a_mano(aud=RISORSA_A2A, scope="policy:leggi")
        senza_scope = await h.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"},
            headers={"Authorization": f"Bearer {solo_policy}"},
        )
        senza_scadenza = await h.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"},
            headers={"Authorization": f"Bearer {token_fatto_a_mano(aud=RISORSA_A2A, exp=None)}"},
        )
        altro = token_fatto_a_mano(aud=RISORSA_A2A, iss="http://un-altro-emittente/")
        altro_emittente = await h.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"},
            headers={"Authorization": f"Bearer {altro}"},
        )
    assert "url" not in carta  # la 1.0 dice dove con le interfacce, non con un campo in cima
    assert carta["supportedInterfaces"] == [
        {"url": f"{RISORSA_A2A}/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    assert "bearer" in carta["securitySchemes"]
    assert carta["securityRequirements"] == [{"schemes": {"bearer": {}}}]  # e la chiede
    # il token MCP qui non vale, né uno senza lo scope dell'istruttoria, né uno che non scade,
    # né uno di un altro emittente con la firma giusta
    stati = (senza.status_code, con_mcp.status_code, senza_scope.status_code)
    assert stati + (senza_scadenza.status_code, altro_emittente.status_code) == (401,) * 5


async def test_un_client_a2a_standard_legge_la_carta_e_manda_il_token() -> None:
    async def rispondi(user: UserContext, testo: str) -> str:
        return f"per {user.username}"

    credenziali = InMemoryContextCredentialStore()
    await credenziali.set_credentials("sessione-1", "bearer", token_a2a())
    async with httpx.AsyncClient(  # nessuna intestazione scritta a mano: la mette l'SDK
        transport=httpx.ASGITransport(app=agente(rispondi)), base_url=RISORSA_A2A
    ) as h:
        c = await create_client(
            RISORSA_A2A,
            ClientConfig(httpx_client=h, streaming=False),
            interceptors=[AuthInterceptor(credenziali)],
        )
        m = new_text_message(f"analizza {CONTO_DI_MARCO}", role=Role.ROLE_USER)
        contesto = ClientCallContext(state={"sessionId": "sessione-1"})
        ultimo = None
        async for r in c.send_message(SendMessageRequest(message=m), context=contesto):
            ultimo = r
    assert ultimo is not None and get_artifact_text(ultimo.task.artifacts[0]) == "per mbianchi"


async def test_il_dato_che_manca_l_agente_lo_chiede_e_poi_lavora() -> None:
    ricevute: list[tuple[str, str]] = []

    async def rispondi(user: UserContext, testo: str) -> str:
        ricevute.append((user.username, testo))
        return "Due bonifici verso Panama da segnalare."

    async with client_a2a(agente(rispondi), token_a2a()) as c:
        t1 = await manda(c, "ci sono movimenti da segnalare?")
        assert TaskState.Name(t1.status.state) == "TASK_STATE_INPUT_REQUIRED"
        assert get_message_text(t1.status.message) == "Su quale conto? Serve l'IBAN."
        t2 = await manda(c, f"il conto è {CONTO_DI_MARCO}", t1.id, t1.context_id)
        assert TaskState.Name(t2.status.state) == "TASK_STATE_COMPLETED" and t2.id == t1.id
        assert get_artifact_text(t2.artifacts[0]) == "Due bonifici verso Panama da segnalare."
        with pytest.raises(UnsupportedOperationError):  # un task concluso non si riapre
            await manda(c, "e ancora?", t1.id, t1.context_id)
        assert TaskState.Name((await c.get_task(GetTaskRequest(id=t1.id))).status.state) == (
            "TASK_STATE_COMPLETED"
        )
    # l'operatore è quello del token, e la richiesta è tutto quello che ha scritto, una volta
    assert ricevute == [
        ("mbianchi", f"ci sono movimenti da segnalare? il conto è {CONTO_DI_MARCO}")
    ]


async def test_il_task_di_un_operatore_un_altro_non_lo_vede() -> None:
    async def rispondi(user: UserContext, testo: str) -> str:
        return "Niente da segnalare."

    app = agente(rispondi)
    async with client_a2a(app, token_a2a()) as c:
        aperto = await manda(c, "ci sono movimenti da segnalare?")  # aspetta l'IBAN
    paola = UserContext(username="pgalli", role="operator")
    async with client_a2a(app, token_a2a(paola)) as c:
        # il proprietario del task è chi ha il token, e un altro non lo legge né lo completa
        with pytest.raises(TaskNotFoundError):
            await c.get_task(GetTaskRequest(id=aperto.id))
        with pytest.raises(TaskNotFoundError):
            await manda(c, f"il conto è {CONTO_DI_MARCO}", aperto.id, aperto.context_id)


async def test_con_il_conto_nella_prima_richiesta_lavora_subito() -> None:
    ricevute: list[str] = []

    async def rispondi(user: UserContext, testo: str) -> str:
        ricevute.append(testo)
        return "Niente da segnalare."

    async with client_a2a(agente(rispondi), token_a2a()) as c:
        t = await manda(c, f"movimenti da segnalare su {CONTO_DI_MARCO}?")
    assert TaskState.Name(t.status.state) == "TASK_STATE_COMPLETED"
    assert ricevute == [f"movimenti da segnalare su {CONTO_DI_MARCO}?"]  # una volta sola


async def test_quando_l_istruttoria_si_rompe_il_task_fallisce_senza_dettagli() -> None:
    async def rotta(user: UserContext, testo: str) -> str:
        raise RuntimeError("postgresql://utente:password@db/banca")

    async with client_a2a(agente(rotta), token_a2a()) as c:
        t = await manda(c, f"analizza {CONTO_DI_MARCO}")
    assert TaskState.Name(t.status.state) == "TASK_STATE_FAILED"
    assert get_message_text(t.status.message) == NON_RIUSCITA


async def test_in_servizio_dietro_l_agente_c_e_il_grafo(monkeypatch: pytest.MonkeyPatch) -> None:
    copione = Copione(
        [
            completion("istruttoria"),  # il triage sceglie la riga
            completion("Conto principale."),  # lo specialista dei conti
            completion("Due bonifici verso Panama da segnalare."),  # l'istruttoria
            completion("Due bonifici verso Panama, da segnalare."),  # la sintesi
        ]
    )
    monkeypatch.setattr("src.lc.modelli.modello_chat", lambda *_a, **_k: chat(copione))
    monkeypatch.setattr("src.agents.deps.get_embedder", embedder_finto)
    monkeypatch.setattr("src.agents.deps.get_openai", lambda: openai_finto(copione))
    esito = await modulo_agente.istruttoria_del_grafo(MARCO, f"analizza {CONTO_DI_MARCO}")
    assert esito == "Due bonifici verso Panama, da segnalare."
    assert not copione.risposte


# ---------------------------------------------------------------- il grafo che delega
async def test_il_grafo_sosta_sulla_domanda_dell_agente_e_riprende() -> None:
    async def rispondi(user: UserContext, testo: str) -> str:
        return f"Istruttoria su: {testo}"

    app, saver = agente(rispondi), InMemorySaver()
    config: Any = {"configurable": {"thread_id": "mbianchi:delega-1"}}
    async with client_a2a(app, token_a2a()) as c:
        out = await costruisci_delega(c, saver).ainvoke({"richiesta": "da segnalare?"}, config)
    assert out["__interrupt__"][0].value == {
        "agente": "istruttoria-lipari",
        "domanda": "Su quale conto? Serve l'IBAN.",
    }
    async with client_a2a(app, token_a2a()) as c:  # un client nuovo: il task è nello stato
        out = await costruisci_delega(c, saver).ainvoke(Command(resume=CONTO_DI_MARCO), config)
    assert out["stato"] == "TASK_STATE_COMPLETED"
    assert out["esito"] == f"Istruttoria su: da segnalare? {CONTO_DI_MARCO}"


async def test_un_agente_che_chiede_sempre_si_ferma_al_tetto() -> None:
    async def mai(user: UserContext, testo: str) -> str:
        raise AssertionError

    app, saver = agente(mai), InMemorySaver()
    config: Any = {"configurable": {"thread_id": "mbianchi:delega-2"}}
    async with client_a2a(app, token_a2a()) as c:
        grafo = costruisci_delega(c, saver)
        out = await grafo.ainvoke({"richiesta": "da segnalare?"}, config)
        for _ in range(10):
            if "__interrupt__" not in out:
                break
            out = await grafo.ainvoke(Command(resume="non lo so"), config)
    assert out["esito"] == TROPPI_GIRI and out["giri"] == 3  # tre giri, scritti come numero


async def test_un_task_perso_dall_agente_chiude_la_delega_con_un_esito() -> None:
    async def rispondi(user: UserContext, testo: str) -> str:
        return "mai"

    saver = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "mbianchi:delega-3"}}
    async with client_a2a(agente(rispondi), token_a2a()) as c:
        out = await costruisci_delega(c, saver).ainvoke({"richiesta": "da segnalare?"}, config)
    assert "__interrupt__" in out
    # l'agente riparte da capo fra la sosta e la ripresa: il task non c'è più
    async with client_a2a(agente(rispondi), token_a2a()) as c:
        out = await costruisci_delega(c, saver).ainvoke(Command(resume=CONTO_DI_MARCO), config)
    assert (out["stato"], out["esito"]) == (PERSO, TASK_PERSO)


# ---------------------------------------------------------------- la password di sviluppo
@pytest.mark.parametrize(
    ("valore", "motivo"),
    [("", "Manca"), ("bootcamp", "pubblica"), ("dodici-meno", "corta")],
)
def test_il_seed_non_parte_con_una_password_assente_bruciata_o_corta(
    monkeypatch: pytest.MonkeyPatch, valore: str, motivo: str
) -> None:
    from scripts.seed_users import password_di_sviluppo

    monkeypatch.setenv("SEED_PASSWORD", valore)  # l'ambiente vince sul .env della tua macchina
    with pytest.raises(SystemExit, match=motivo):
        password_di_sviluppo()


async def _hash_di(username: str) -> str:
    async with AsyncSessionLocal() as s:
        utente = await s.scalar(select(AppUser).where(AppUser.username == username))
    assert utente is not None
    return utente.password_hash


async def test_cambiare_il_env_non_basta_bisogna_ruotare() -> None:
    from scripts.seed_users import UTENTI, semina

    await semina("la-password-di-prima", ruota=True)  # uno stato noto, qualunque test sia passato
    assert await semina("la-password-nuova") == (0, 0)  # senza --ruota chi c'era resta com'era
    assert verify_password("la-password-di-prima", await _hash_di("mbianchi"))
    assert await semina("la-password-nuova", ruota=True) == (0, len(UTENTI))
    nuovo = await _hash_di("mbianchi")
    assert verify_password("la-password-nuova", nuovo)
    assert not verify_password("la-password-di-prima", nuovo)


async def test_il_seed_non_stampa_la_password(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import seed_users

    monkeypatch.setenv("SEED_PASSWORD", "una-password-da-non-stampare")
    await seed_users.main(["--ruota"])
    assert "una-password-da-non-stampare" not in capsys.readouterr().out
