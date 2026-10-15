# tests/test_g6.py — chi sei: login, token, ruoli, canarino, riscrittura, degradazione, limite
import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.api import auth as auth_api
from src.api.advice import get_rewriter, limiter
from src.auth.acl import visible_to
from src.auth.deps import UserContext, get_current_user
from src.auth.passwords import verify_password
from src.auth.tokens import ALGORITHM, create_access_token
from src.config import settings
from src.db.models import ChatSession
from src.db.session import AsyncSessionLocal, engine
from src.exceptions import LLMProviderError
from src.llm.factory import get_llm_provider
from src.llm.rewriter import QueryRewriter
from src.llm.types import LLMResponse, Message, ToolSpec
from src.main import app
from src.services.ingest_service import IngestService
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService
from tests.finti import ModelloFisso, embedder_finto, riscrittore_spento, vettore

CANARINO = "La parola d'ordine della compliance è ortensia-viola-sette."
RISERVATO = f"""# Controparti venezuelane — istruttoria Compliance (riservato)

## Operazioni verso il Venezuela
Le operazioni verso controparti con sede in Venezuela sono consentite solo dopo una verifica
rafforzata (EDD). {CANARINO}
"""
PUBBLICO = """# Bonifici verso l'estero
Le operazioni verso il Venezuela seguono le regole dei bonifici extra-SEPA: costo €15.00.
"""


@pytest.fixture(autouse=True)
async def indice_vuoto() -> AsyncIterator[None]:
    await engine.dispose()  # le connessioni dei TestClient dei giorni prima: via dal pool
    async with engine.begin() as c:
        await c.execute(text("TRUNCATE document_chunks, chat_messages, chat_sessions CASCADE"))
    yield


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)  # i finti della sessione, dal conftest della radice
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# la password dei test: la riceve il seed da qui, perché sulla tua macchina il .env ha quella vera
PASSWORD_DEI_TEST = "solo-per-i-test-del-login"


async def _seed_utenti() -> None:
    from scripts.seed_users import semina

    await semina(PASSWORD_DEI_TEST, ruota=True)  # un test di prima può averne lasciata un'altra


async def _token(username: str) -> str:
    app.dependency_overrides.pop(get_current_user, None)
    async with client() as c:
        r = await c.post(
            "/api/auth/login", data={"username": username, "password": PASSWORD_DEI_TEST}
        )
    assert r.status_code == 200, r.text
    return str(r.json()["access_token"])


async def _ingerisci_corpus() -> None:
    async with AsyncSessionLocal() as s:
        servizio = IngestService(s, embedder_finto())
        await servizio.ingest("bonifici_estero", PUBBLICO)
        await servizio.ingest("aml_controparti_venezuela", RISERVATO, None, "compliance_only")


# ---------------------------------------------------------------- login e token


async def test_il_login_da_un_token_e_non_dice_quale_credenziale_e_sbagliata(
    monkeypatch: pytest.MonkeyPatch,
):
    await _seed_utenti()
    app.dependency_overrides.pop(get_current_user, None)
    verifiche: list[str] = []
    originale = verify_password  # l'originale, da dove è definita

    def contata(plain: str, hashed: str) -> bool:
        verifiche.append(hashed)
        return originale(plain, hashed)

    monkeypatch.setattr(auth_api, "verify_password", contata)
    async with client() as c:
        ok = await c.post(
            "/api/auth/login", data={"username": "mbianchi", "password": PASSWORD_DEI_TEST}
        )
        errata = await c.post("/api/auth/login", data={"username": "mbianchi", "password": "x"})
        nessuno = await c.post("/api/auth/login", data={"username": "nessuno", "password": "x"})
    assert ok.status_code == 200 and ok.json()["token_type"] == "bearer"
    assert errata.status_code == nessuno.status_code == 401
    assert errata.json()["message"] == nessuno.json()["message"] == "Credenziali non valide"
    assert len(verifiche) == 3 and verifiche[2] == auth_api.DUMMY_HASH  # l'hash si verifica sempre


def test_il_payload_si_legge_senza_chiave():
    token = create_access_token("mbianchi", "operator")
    payload = token.split(".")[1]
    dati = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    assert set(dati) == {"sub", "role", "iat", "exp"} and dati["role"] == "operator"


async def test_token_alterato_scaduto_e_alg_none_sono_respinti():
    app.dependency_overrides.pop(get_current_user, None)
    buono = create_access_token("mbianchi", "operator")
    alterato = buono[:-3] + ("A" if buono[-3] != "A" else "B") + buono[-2:]
    scaduto = jwt.encode(
        {"sub": "mbianchi", "role": "operator", "exp": datetime.now(UTC) - timedelta(minutes=1)},
        settings.jwt_secret,
        algorithm=ALGORITHM,
    )
    nessuno = jwt.encode({"sub": "mbianchi", "role": "admin"}, key="", algorithm="none")
    async with client() as c:
        esiti = {}
        for nome, tok in (("alterato", alterato), ("scaduto", scaduto), ("none", nessuno)):
            r = await c.post(
                "/api/ai/chat",
                json={"session_id": "new", "message": "x"},
                headers={"Authorization": f"Bearer {tok}"},
            )
            esiti[nome] = (r.status_code, r.json().get("message"))
        senza = await c.post("/api/ai/chat", json={"session_id": "new", "message": "x"})
    assert esiti["alterato"] == (401, "Token non valido")
    assert esiti["scaduto"] == (401, "Token scaduto")
    assert esiti["none"] == (401, "Token non valido")
    assert senza.status_code == 401 and senza.headers["www-authenticate"] == "Bearer"


def test_un_ruolo_sconosciuto_vede_il_pubblico_e_finisce_nel_log(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        assert visible_to("stagista") == ["public"]
    assert any(r.message == "ruolo_sconosciuto" for r in caplog.records)
    assert "compliance_only" not in visible_to("risk_lead")  # i ruoli non sono una scala


async def test_ingerire_e_della_compliance_e_il_livello_sbagliato_e_422():
    await _seed_utenti()
    marco, giulia = await _token("mbianchi"), await _token("grossi")
    corpo = {"document_id": "d", "content": PUBBLICO}
    async with client() as c:
        vietato = await c.post(
            "/api/ai/documents/ingest", json=corpo, headers={"Authorization": f"Bearer {marco}"}
        )
        ammesso = await c.post(
            "/api/ai/documents/ingest", json=corpo, headers={"Authorization": f"Bearer {giulia}"}
        )
        sbagliato = await c.post(
            "/api/ai/documents/ingest",
            json=corpo | {"visibility": "compliance-only"},
            headers={"Authorization": f"Bearer {giulia}"},
        )
    assert (vietato.status_code, ammesso.status_code, sbagliato.status_code) == (403, 200, 422)


# ---------------------------------------------------------------- il filtro nel WHERE


async def test_il_canarino_non_arriva_a_marco_e_arriva_a_giulia():
    await _ingerisci_corpus()
    vettore_domanda = vettore("le operazioni verso il Venezuela sono consentite")
    async with AsyncSessionLocal() as s:
        marco = await RetrievalService(s).search_for_user(vettore_domanda, "operator")
        giulia = await RetrievalService(s).search_for_user(vettore_domanda, "compliance_lead")
    assert marco and all(CANARINO not in p.content for p in marco)
    assert any(CANARINO in p.content for p in giulia)


def test_il_filtro_sta_nella_query_prima_dell_ordinamento():
    testo = " ".join(RetrievalService.SQL_PER_RUOLO.text.split())
    assert "AND visibility = ANY(:livelli)" in testo
    assert testo.index("visibility") < testo.index("ORDER BY")


async def test_la_stessa_domanda_da_due_ruoli_da_fonti_diverse():
    await _seed_utenti()
    await _ingerisci_corpus()
    marco, giulia = await _token("mbianchi"), await _token("grossi")
    modello = ModelloFisso("Risposta [fonte-1] [fonte-2].")
    app.dependency_overrides[get_llm_provider] = lambda: modello
    domanda = {"question": "le operazioni verso il Venezuela sono consentite"}
    async with client() as c:
        rm = await c.post(
            "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {marco}"}
        )
        rg = await c.post(
            "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {giulia}"}
        )
    fonti_m = {x["document_id"] for x in rm.json()["citations"]}
    fonti_g = {x["document_id"] for x in rg.json()["citations"]}
    assert "aml_controparti_venezuela" in fonti_g and "aml_controparti_venezuela" not in fonti_m
    assert rm.json()["rewritten_query"] == domanda["question"]  # riscrittore spento nel test
    assert CANARINO not in json.dumps(rm.json())
    # e con un riscrittore che riscrive davvero, al generatore va comunque la domanda dell'utente
    riscritta = "Le operazioni verso il Venezuela sono consentite dalla policy antiriciclaggio?"
    app.dependency_overrides[get_rewriter] = lambda: QueryRewriter(
        ModelloCheRiscrive([riscritta]), "prompt"
    )
    try:
        async with client() as c:
            r = await c.post(
                "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {giulia}"}
            )
    finally:
        app.dependency_overrides[get_rewriter] = riscrittore_spento
    assert r.json()["rewritten_query"] == riscritta  # il recupero ha cercato la riscritta
    ultimo = modello.ricevuti[-1][-1].content
    assert ultimo.endswith(f"DOMANDA\n{domanda['question']}") and riscritta not in ultimo


# ---------------------------------------------------------------- riscrittura


class ModelloCheRiscrive:
    def __init__(self, uscite: list[str | Exception]) -> None:
        self.uscite = uscite
        self.chiamate = 0

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.chiamate += 1
        uscita = self.uscite[min(self.chiamate, len(self.uscite)) - 1]
        if isinstance(uscita, Exception):
            raise uscita
        return LLMResponse(content=uscita, tokens_used=1, cost_eur=Decimal("0"), model="finto")

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        yield await self.complete(messages)


async def test_il_riscrittore_ripiega_non_mette_in_cache_il_ripiego_e_scarta_le_risposte():
    guasto = ModelloCheRiscrive(
        [LLMProviderError("openai", "giù"), "Le operazioni verso il Venezuela?"]
    )
    r = QueryRewriter(guasto, "PROMPT")
    assert await r.rewrite_cached("ven ok?") == "ven ok?"  # il ripiego...
    assert (
        await r.rewrite_cached("ven ok?") == "Le operazioni verso il Venezuela?"
    )  # ...non era in cache
    assert await r.rewrite_cached("ven ok?") == "Le operazioni verso il Venezuela?"
    assert guasto.chiamate == 2  # la terza è dalla cache
    lungo = ModelloCheRiscrive(["x" * 500])
    assert await QueryRewriter(lungo, "PROMPT").rewrite("ven ok?") == "ven ok?"
    assert await QueryRewriter(lungo, "PROMPT", enabled=False).rewrite("ven ok?") == "ven ok?"


def test_gli_esempi_della_pagina_passano_la_guardia():
    for domanda, riscritta in (
        (
            "ven ok?",
            "Le operazioni verso il Venezuela sono consentite secondo la policy "
            "antiriciclaggio (AML) di LipariBank?",
        ),
        (
            "carta clonata",
            "Qual è la procedura da seguire quando un cliente segnala che la sua "
            "carta di pagamento è stata clonata?",
        ),
        (
            "bonifico 20k cliente nuovo",
            "Qual è la procedura per un bonifico superiore a 20.000 "
            "euro disposto da un cliente acquisito di recente?",
        ),
    ):
        assert len(riscritta) <= 4 * len(domanda) + 80, domanda


# ---------------------------------------------------------------- degradazione e limite


async def test_se_il_modello_cade_arrivano_i_passaggi_con_le_fonti(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    await _ingerisci_corpus()

    class ModelloGiu(ModelloFisso):
        async def complete(
            self,
            messages: list[Message],
            max_tokens: int = 500,
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            raise LLMProviderError("openai", "503")

    class ModelloLento(ModelloFisso):
        async def complete(
            self,
            messages: list[Message],
            max_tokens: int = 500,
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            await asyncio.sleep(1)
            return await super().complete(messages, max_tokens)

    utente = UserContext(username="mbianchi", role="operator")
    for modello, timeout in ((ModelloGiu("x"), 10), (ModelloLento("x"), 0.05)):
        async with AsyncSessionLocal() as s:
            servizio = RAGService(
                RetrievalService(s),
                embedder_finto(),
                modello,
                QueryRewriter(modello, "", enabled=False),
                "PROMPT",
            )
            monkeypatch.setattr(servizio, "TIMEOUT_GENERAZIONE_S", timeout)
            with caplog.at_level(logging.INFO):
                esito = await servizio.advise("bonifici estero Venezuela costo", utente)
        assert esito.answer.startswith("⚠️ Risposta parziale") and esito.cost_eur == 0
        assert esito.citations and esito.citations[0].document_id == "bonifici_estero"
    riga = next(r for r in caplog.records if r.message == "advice_completata")
    assert riga.used_fallback is True and CANARINO not in str(riga.__dict__)  # type: ignore[attr-defined]


async def test_il_limite_e_per_utente_e_risponde_con_la_busta():
    await _seed_utenti()
    marco, giulia = await _token("mbianchi"), await _token("grossi")
    app.dependency_overrides[get_llm_provider] = lambda: ModelloFisso("x")
    limiter.reset()  # il contatore vive per la sessione: le richieste dei test prima non contano
    domanda = {"question": "domanda qualunque"}
    async with client() as c:
        stati = [
            (
                await c.post(
                    "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {marco}"}
                )
            ).status_code
            for _ in range(16)
        ]
        ultima = await c.post(
            "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {marco}"}
        )
        di_giulia = await c.post(
            "/api/ai/advice", json=domanda, headers={"Authorization": f"Bearer {giulia}"}
        )
    assert stati[:15] == [200] * 15 and stati[15] == 429
    assert ultima.json()["error"] == "RATE_LIMIT" and ultima.headers["retry-after"] == "60"
    assert di_giulia.status_code == 200  # il limite di Marco non è quello di Giulia


async def test_la_chat_scrive_la_conversazione_a_nome_dell_utente_del_token():
    await _seed_utenti()
    giulia = await _token("grossi")
    async with client() as c:
        r = await c.post(
            "/api/ai/chat",
            json={"session_id": "new", "message": "ciao"},
            headers={"Authorization": f"Bearer {giulia}"},
        )
    async with AsyncSessionLocal() as s:
        sessione = await s.get(ChatSession, r.json()["session_id"])
    assert sessione is not None and sessione.user_id == "grossi"


async def test_i_conti_sono_quelli_dell_operatore_del_token():
    from scripts import seed_accounts

    await _seed_utenti()
    await seed_accounts.main()
    marco, giulia = await _token("mbianchi"), await _token("grossi")
    conto = seed_accounts.CONTI[0][0]  # un conto di Paolo Ferri, cliente di Marco
    periodo = {"date_from": "2020-01-01", "date_to": "2100-01-01"}
    async with client() as c:
        di_marco = await c.get(
            f"/api/ai/accounts/{conto}/spending",
            params=periodo,
            headers={"Authorization": f"Bearer {marco}"},
        )
        di_giulia = await c.get(
            f"/api/ai/accounts/{conto}/spending",
            params=periodo,
            headers={"Authorization": f"Bearer {giulia}"},
        )
    assert di_marco.status_code == 200 and di_giulia.status_code == 404
