# tests/test_g4.py — la voce: gli SDK veri, con le API finte a livello HTTP
import json
import logging
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal

import httpx2
import instructor
import openai
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text

from scripts import seed_accounts
from src.config import settings
from src.db.models import ChatMessage
from src.db.repos import ChatRepository
from src.db.session import AsyncSessionLocal, engine
from src.exceptions import LLMProviderError, RateLimitError
from src.llm import factory
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.factory import get_llm_provider
from src.llm.openai_provider import OpenAIProvider
from src.llm.prompt import load_prompt
from src.llm.tools import TOOLS_SCHEMA
from src.llm.types import LLMResponse, Message, ToolCall
from src.main import app
from src.observability.cost_tracker import CostTracker
from src.services.categorize_service import CategorizeService
from src.types.categorize import CategorizeRequest, CategorizeResponse
from tests.finti import (
    Gestore,
    Registro,
    anthropic_finto,
    chiamata,
    completion,
    messaggio_anthropic,
    openai_finto,
    stream_anthropic,
    stream_openai,
)

MSG = [Message(role="system", content="Sei l'assistente."), Message(role="user", content="Ciao")]
CONTO_DI_MARCO, _, CONTO_DI_ANNA = (c[0] for c in seed_accounts.CONTI)


@pytest.fixture(autouse=True)
async def tabelle_vuote() -> AsyncIterator[None]:
    await engine.dispose()  # le connessioni dei TestClient dei giorni prima: via dal pool
    async with engine.begin() as c:
        await c.execute(
            text("TRUNCATE chat_messages, chat_sessions, movements, accounts, customers CASCADE")
        )
    yield


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)  # i finti della sessione, dal conftest della radice
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


def client_con(gestore: Gestore) -> AsyncClient:
    """L'applicazione con il provider di OpenAI vero, su un'API finta."""
    provider = OpenAIProvider(openai_finto(gestore), "gpt-4o-mini")
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------- i due provider
async def test_openai_traduce_e_conta():
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        return completion("Buongiorno", prompt=1000, uscita=500)

    esito = await OpenAIProvider(openai_finto(gestore), "gpt-4o-mini").complete(MSG, max_tokens=300)
    assert [m["role"] for m in reg.richieste[0]["messages"]] == ["system", "user"]
    assert "tools" not in reg.richieste[0]  # senza strumenti, il parametro non parte nemmeno
    assert esito.content == "Buongiorno" and esito.tokens_used == 1500
    assert esito.cost_eur == Decimal("0.00042")  # 1000·0.00014/1000 + 500·0.00056/1000, esatto


async def test_lo_stream_chiede_il_consumo_e_lo_mette_in_fondo():
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        return stream_openai(["Buon", "giorno", "!"], prompt=80, uscita=12)

    pezzi = [p async for p in OpenAIProvider(openai_finto(gestore), "gpt-4o-mini").stream(MSG)]
    assert pezzi[:3] == ["Buon", "giorno", "!"]
    assert isinstance(pezzi[-1], LLMResponse) and pezzi[-1].tokens_used == 92
    assert reg.richieste[0]["stream_options"] == {"include_usage": True}


async def test_l_sdk_riprova_i_transitori_e_non_gli_altri():
    for stato, attese in ((503, 3), (429, 3), (401, 1), (400, 1)):
        reg = Registro()

        def gestore(r: httpx2.Request, s: int = stato, reg: Registro = reg) -> httpx2.Response:
            reg.annota(r)
            return httpx2.Response(
                s, json={"error": {"message": "x"}}, headers={"retry-after-ms": "1"}
            )

        with pytest.raises(LLMProviderError) as errore:
            await OpenAIProvider(openai_finto(gestore), "gpt-4o-mini").complete(MSG)
        assert isinstance(errore.value.__cause__, openai.APIStatusError)
        assert len(reg.richieste) == attese, stato  # max_retries=2: tre tentativi sui transitori


async def test_il_gemello_traduce_la_stessa_conversazione():
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        corpo = reg.annota(r)
        if corpo.get("stream"):
            return stream_anthropic(["Buon", "giorno"])
        return messaggio_anthropic("Buongiorno", entrata=50, uscita=10)

    gemello = AnthropicProvider(anthropic_finto(gestore), "claude-haiku-4-5-20251001")
    esito = await gemello.complete(MSG)
    assert reg.richieste[0]["system"] == "Sei l'assistente."  # il sistema è un parametro a sé
    assert [m["role"] for m in reg.richieste[0]["messages"]] == ["user"]
    assert esito.content == "Buongiorno" and esito.tokens_used == 60
    pezzi = [p async for p in gemello.stream(MSG)]
    assert pezzi[:2] == ["Buon", "giorno"] and isinstance(pezzi[-1], LLMResponse)


async def test_gli_strumenti_nelle_due_lingue():
    """Lo stesso giro — il modello chiede, lo strumento risponde — tradotto dai due provider."""
    richiesta = ToolCall(id="t1", name="get_account_balance", arguments='{"account_id": "IT60X"}')
    giro = [
        *MSG,
        Message(role="assistant", content="", tool_calls=[richiesta]),
        Message(role="tool", content="Saldo: 48200.00 EUR", tool_call_id="t1"),
    ]
    reg_o, reg_a = Registro(), Registro()

    def gestore_o(r: httpx2.Request) -> httpx2.Response:
        reg_o.annota(r)
        return completion(None, tool_calls=[chiamata("t2", "get_account_balance", {"x": 1})])

    def gestore_a(r: httpx2.Request) -> httpx2.Response:
        reg_a.annota(r)
        return messaggio_anthropic("", strumenti=[("t2", "get_account_balance", {"x": 1})])

    esito_o = await OpenAIProvider(openai_finto(gestore_o), "gpt-4o-mini").complete(
        giro, tools=TOOLS_SCHEMA
    )
    esito_a = await AnthropicProvider(
        anthropic_finto(gestore_a), "claude-haiku-4-5-20251001"
    ).complete(giro, tools=TOOLS_SCHEMA)
    # OpenAI: lo strumento in "function", il risultato come ruolo "tool"
    corpo_o = reg_o.richieste[0]
    assert corpo_o["tools"][0]["function"]["name"] == "get_account_balance"
    assert corpo_o["messages"][2]["tool_calls"][0]["id"] == "t1"
    assert corpo_o["messages"][3] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "Saldo: 48200.00 EUR",
    }
    # Anthropic: lo schema in input_schema, la richiesta come blocco, il risultato dall'utente
    corpo_a = reg_a.richieste[0]
    assert corpo_a["tools"][0]["input_schema"] == TOOLS_SCHEMA[0].parameters
    assert corpo_a["messages"][1]["content"][0]["type"] == "tool_use"
    assert corpo_a["messages"][2]["role"] == "user"
    assert corpo_a["messages"][2]["content"][0]["tool_use_id"] == "t1"
    # e le due risposte, tornate nella stessa forma
    for esito in (esito_o, esito_a):
        assert [(c.id, c.name, json.loads(c.arguments)) for c in esito.tool_calls] == [
            ("t2", "get_account_balance", {"x": 1})
        ]


# ---------------------------------------------------------------- la factory
async def test_senza_chiave_la_chat_dice_quale_manca(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "default_model", "gpt-4o-mini")
    factory._openai.cache_clear()
    app.dependency_overrides.pop(get_llm_provider)  # qui serve la factory vera
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/ai/chat", json={"session_id": "new", "message": "x"})
    factory._openai.cache_clear()
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["message"]


def test_la_factory_sceglie_e_fallisce_forte(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    factory._openai.cache_clear()
    factory._anthropic.cache_clear()
    monkeypatch.setattr(settings, "default_model", "gpt-4o-mini")
    assert isinstance(get_llm_provider(), OpenAIProvider)
    monkeypatch.setattr(settings, "default_model", "claude-haiku-4-5-20251001")
    assert isinstance(get_llm_provider(), AnthropicProvider)
    for sbagliato in ("llama-3", "gpt-5-inesistente"):
        monkeypatch.setattr(settings, "default_model", sbagliato)
        with pytest.raises(ValueError):
            get_llm_provider()
    factory._openai.cache_clear()
    factory._anthropic.cache_clear()


# ---------------------------------------------------------------- la chat
async def test_la_conversazione_tiene_il_filo_e_salva_il_costo():
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        return completion(f"risposta {len(reg.richieste)}")

    async with client_con(gestore) as c:
        r1 = await c.post("/api/ai/chat", json={"session_id": "new", "message": "Sono Marco"})
        sid = r1.json()["session_id"]
        r2 = await c.post("/api/ai/chat", json={"session_id": sid, "message": "Come mi chiamo?"})
    assert r1.status_code == r2.status_code == 200
    seconda = reg.richieste[1]["messages"]
    assert [m["role"] for m in seconda] == ["system", "user", "assistant", "user"]
    assert seconda[0]["content"] == load_prompt("chat_system_v1")
    async with AsyncSessionLocal() as s:
        salvati = (await s.scalars(select(ChatMessage).where(ChatMessage.session_id == sid))).all()
    # 100 token in entrata e 20 in uscita costano 0.0000252: la colonna a sei decimali
    # salva 0.000025
    assert [m.cost_eur for m in salvati if m.role == "assistant"] == [Decimal("0.000025")] * 2


async def test_lo_stream_manda_json_salva_dopo_e_un_404_resta_un_404():
    async with client_con(lambda r: stream_openai(["riga uno\n", "riga due"])) as c:
        r = await c.post("/api/ai/chat/stream", json={"session_id": "new", "message": "x"})
        assente = await c.post("/api/ai/chat/stream", json={"session_id": "no", "message": "x"})
    eventi = [e for e in r.text.split("\n\n") if e]
    assert r.status_code == 200 and eventi[-1] == "data: [DONE]"
    assert json.loads(eventi[0].removeprefix("data: ")) == {"text": "riga uno\n"}
    assert assente.status_code == 404
    async with AsyncSessionLocal() as s:
        risposta = (
            await s.scalars(select(ChatMessage).where(ChatMessage.role == "assistant"))
        ).one()
    assert risposta.content == "riga uno\nriga due" and risposta.tokens == 92


async def test_il_giro_dello_strumento_legge_il_saldo_e_conta_due_chiamate():
    await seed_accounts.main()
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        if len(reg.richieste) == 1:  # il modello chiede lo strumento
            argomenti = {"account_id": CONTO_DI_MARCO}
            return completion(None, tool_calls=[chiamata("t1", "get_account_balance", argomenti)])
        return completion("Paolo Ferri ha 48.200,00 euro sul conto principale.")

    async with client_con(gestore) as c:
        r = await c.post("/api/ai/chat", json={"session_id": "new", "message": "Saldo di Paolo?"})
    assert r.status_code == 200 and len(reg.richieste) == 2  # chiede, poi risponde
    assert r.json()["tool_calls"] == [
        {
            "name": "get_account_balance",
            "arguments": {"account_id": CONTO_DI_MARCO},
            "result": f"Saldo del conto {CONTO_DI_MARCO}: 48200.00 EUR",
        }
    ]
    seconda = reg.richieste[1]["messages"]
    assert seconda[-2]["tool_calls"][0]["id"] == "t1"  # l'assistente che chiede, prima
    assert seconda[-1]["role"] == "tool" and "48200.00" in seconda[-1]["content"]
    assert r.json()["tokens_used"] == 240  # le due chiamate, non solo l'ultima


async def test_il_conto_di_un_altro_portafoglio_lo_rifiuta_lo_strumento(
    caplog: pytest.LogCaptureFixture,
):
    await seed_accounts.main()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        if "tool" not in {m["role"] for m in json.loads(r.content)["messages"]}:
            argomenti = {"account_id": CONTO_DI_ANNA}
            return completion(None, tool_calls=[chiamata("t1", "get_account_balance", argomenti)])
        return completion("Non posso accedere a quel conto.")

    with caplog.at_level(logging.WARNING):
        async with client_con(gestore) as c:
            domanda = f"Quanto c'è sul conto {CONTO_DI_ANNA}?"
            r = await c.post("/api/ai/chat", json={"session_id": "new", "message": domanda})
    assert (
        r.json()["tool_calls"][0]["result"] == "ERRORE: conto non disponibile per questo operatore."
    )
    assert "15300" not in r.text  # il saldo di Anna non esce da nessuna parte
    assert any(rec.message == "strumento_negato" for rec in caplog.records)


async def test_il_tetto_a_zero_rifiuta_prima_di_chiamare_il_modello(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "daily_budget_eur", Decimal("0"))
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        return completion("non dovrei esserci")

    async with client_con(gestore) as c:
        r = await c.post("/api/ai/chat", json={"session_id": "new", "message": "x"})
    assert r.status_code == 429 and r.json()["error"] == "RATE_LIMIT"
    assert int(r.headers["Retry-After"]) > 0
    # e il tetto vale anche per la categorizzazione: ogni chiamata al modello spende
    async with AsyncSessionLocal() as s:
        servizio = CategorizeService(
            instructor.from_openai(openai_finto(gestore)),
            "gpt-4o-mini",
            "prompt",
            CostTracker(ChatRepository(s), Decimal("0")),
        )
        with pytest.raises(RateLimitError):
            await servizio.categorize(CategorizeRequest(description="Enel", amount=Decimal("8")))
    assert reg.richieste == []  # il modello non è stato chiamato: niente speso


# ---------------------------------------------------------------- lo structured output
async def test_instructor_ritenta_e_ogni_tentativo_e_una_chiamata():
    reg = Registro()
    risposte = [
        {"category": "BOLLETTE", "subcategory": "ENERGY", "confidence": 0.9, "reasoning": "x"},
        {"category": "UTILITIES", "subcategory": "ENERGY", "confidence": 0.92, "reasoning": "Enel"},
    ]

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        argomenti = risposte[min(len(reg.richieste), 2) - 1]
        return completion(None, tool_calls=[chiamata("c1", "CategorizeResponse", argomenti)])

    servizio = CategorizeService(
        instructor.from_openai(openai_finto(gestore)),
        "gpt-4o-mini",
        load_prompt("categorize_system_v1"),
    )
    esito = await servizio.categorize(
        CategorizeRequest(description="Bonifico Enel Energia bolletta", amount=Decimal("85.50"))
    )
    assert isinstance(esito, CategorizeResponse) and esito.category == "UTILITIES"
    assert len(reg.richieste) == 2  # la categoria inventata non è passata, e si è pagata
    # e uno schema mai rispettato, dopo i tentativi, è un guasto del fornitore: il 502
    sempre_fuori = [risposte[0]] * 3
    risposte[:] = sempre_fuori
    with pytest.raises(LLMProviderError):
        await servizio.categorize(
            CategorizeRequest(description="Bonifico Enel Energia bolletta", amount=Decimal("85.50"))
        )
