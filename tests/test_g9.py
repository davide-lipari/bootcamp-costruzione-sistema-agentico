# tests/test_g9.py — i misuratori misurano: eval con finti, cancello, registro dei costi, log
import json
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import instructor
import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import func, select, text

from evals import runner
from evals.runner import Metriche, Spesa, cancello, eval_advice, eval_categorize, eval_traiettorie
from src.api.advice import get_rewriter
from src.auth.deps import UserContext, get_current_user
from src.config import settings
from src.db.models import AgentRunState, LlmCall
from src.db.session import AsyncSessionLocal, engine
from src.llm.factory import get_llm_provider
from src.llm.rewriter import QueryRewriter
from src.llm.types import LLMResponse, Message, ToolSpec
from src.main import app
from src.observability.json_log import JsonFormatter, request_id
from src.observability.ledger import CostLedger
from src.services.categorize_service import CategorizeService
from src.services.ingest_service import IngestService
from src.services.retrieval_service import RetrievalService
from src.types.categorize import CategorizeRequest, CategorizeResponse
from tests.conftest import CONTO_DI_MARCO, risposta
from tests.finti import (
    ModelloFisso,
    chiamata,
    completion,
    embedder_finto,
    openai_finto,
    riscrittore_spento,
)

LUIGI = UserContext(username="lverdi", role="risk_lead")
MODELLO = "gpt-4o-mini"
PUBBLICO = "# Bonifici estero\nI bonifici verso il Venezuela sono extra-SEPA: costo €15.00."
RISERVATO = "# Istruttoria (riservato)\nControparti in Venezuela: verifica rafforzata EDD."
# sopra la soglia del Giorno 8, e con un motivo che passa la validazione (almeno 10 caratteri)
SEGNALA_25K = json.dumps(
    {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama", "importo": "25000"}
)


@pytest.fixture(autouse=True)
async def database_pulito() -> AsyncIterator[None]:
    await engine.dispose()
    async with engine.begin() as c:
        await c.execute(
            text(
                "TRUNCATE llm_calls, document_chunks, agent_runs, compliance_alerts, "
                "movements, accounts, customers, chat_messages, chat_sessions CASCADE"
            )
        )
    yield


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


def dataset(tmp_path: Path, casi: list[dict[str, Any]]) -> Path:
    percorso = tmp_path / "casi.jsonl"
    percorso.write_text("\n".join(json.dumps(c) for c in casi) + "\n\n", encoding="utf-8")
    return percorso


# ---------------------------------------------------------------- la classificazione
class SempreSpesa:
    """Un classificatore che dice sempre GROCERIES, ed esplode su una descrizione."""

    async def categorize(self, req: CategorizeRequest) -> CategorizeResponse:
        if req.description == "esplode":
            raise RuntimeError("il modello non ha risposto")
        return CategorizeResponse(
            category="GROCERIES", subcategory="SUPERMARKET", confidence=0.9, reasoning="sempre"
        )


def _caso_cat(id_: str, descrizione: str, categoria: str) -> dict[str, Any]:
    return {
        "id": id_,
        "input": {"description": descrizione, "amount": 10},
        "expected": {"category": categoria},
    }


async def test_eval_categorize_conta_e_trova_la_colonna_vuota(tmp_path: Path) -> None:
    casi = [
        _caso_cat("a", "Esselunga", "GROCERIES"),
        _caso_cat("b", "Trenitalia", "TRANSPORT"),
        _caso_cat("c", "esplode", "GROCERIES"),
    ]
    servizio: Any = SempreSpesa()  # ha il metodo che il runner chiama: gli basta questo
    m = await eval_categorize(dataset(tmp_path, casi), servizio, Spesa("gpt-4o-mini"))
    assert m.n == 3 and m.punteggio == pytest.approx(1 / 3)
    assert {f["id"]: f["ottenuto"] for f in m.fallimenti} == {
        "b": "GROCERIES",
        "c": "ERRORE: RuntimeError",  # un caso che esplode è un fallimento, non un crash
    }
    assert m.dettagli["mai_prodotte"] == ["TRANSPORT"]  # la colonna vuota della matrice


# ---------------------------------------------------------------- il recupero
class RiscritturaDaTabella:
    """Il riscrittore che fa il suo lavoro: la domanda informale diventa una da cercare."""

    def __init__(self, tabella: dict[str, str]) -> None:
        self.tabella = tabella

    async def complete(
        self, messages: list[Message], max_tokens: int = 500, tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        domanda = messages[-1].content.split("Domanda: ")[-1]
        return LLMResponse(
            content=self.tabella.get(domanda, domanda),
            tokens_used=20,
            cost_eur=Decimal("0.00001"),
            model="finto",
        )

    async def stream(
        self, messages: list[Message], max_tokens: int = 500
    ) -> AsyncIterator[str | LLMResponse]:
        yield await self.complete(messages, max_tokens)


def _caso_adv(id_: str, domanda: str, ruolo: str, doc_id: str | None) -> dict[str, Any]:
    return {
        "id": id_,
        "input": {"question": domanda, "role": ruolo},
        "expected": {"doc_id": doc_id},
    }


async def test_eval_advice_passa_dalla_riscrittura_e_dai_permessi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with AsyncSessionLocal() as s:
        indice = IngestService(s, embedder_finto())
        await indice.ingest("bonifici_estero", PUBBLICO)
        await indice.ingest("aml_controparti_venezuela", RISERVATO, None, "compliance_only")
    casi = [
        _caso_adv("g", "ven ok?", "compliance_lead", "aml_controparti_venezuela"),
        _caso_adv("m", "ven ok?", "operator", "bonifici_estero"),  # il pubblico, mai il riservato
        _caso_adv("x", "tasso mutuo giovani coppie", "operator", None),  # nessuna risposta
    ]
    percorso = dataset(tmp_path, casi)
    riscrittore = QueryRewriter(
        RiscritturaDaTabella({"ven ok?": "controparti Venezuela verifica bonifici"}), "riscrivi"
    )
    async with AsyncSessionLocal() as s:
        con = await eval_advice(percorso, riscrittore, embedder_finto(), RetrievalService(s))
        senza = await eval_advice(
            percorso, riscrittore_spento(), embedder_finto(), RetrievalService(s)
        )
    assert con.punteggio == 1.0, con.fallimenti
    # senza riscrittura «ven ok?» non trova niente: l'eval che salta il riscrittore misura altro
    assert senza.punteggio == pytest.approx(1 / 3)
    assert {f["id"] for f in senza.fallimenti} == {"g", "m"}
    assert senza.fallimenti[0]["query_riscritta"] == "ven ok?"
    # un embedding che fallisce su una domanda è un caso fallito, e gli altri si misurano
    embedder = embedder_finto()
    vero = embedder.embed_one

    async def cade_su_una(testo: str) -> list[float]:
        if testo == "esplode":
            raise RuntimeError("400 dal fornitore")
        return await vero(testo)

    monkeypatch.setattr(embedder, "embed_one", cade_su_una)
    casi_rotti = [_caso_adv("e", "esplode", "operator", None), casi[1]]
    async with AsyncSessionLocal() as s:
        rotto = await eval_advice(
            dataset(tmp_path, casi_rotti), riscrittore_spento(), embedder, RetrievalService(s)
        )
    assert (rotto.n, rotto.fallimenti[0]["id"], rotto.fallimenti[0]["errore"]) == (
        2,
        "e",
        "RuntimeError",
    )


# ---------------------------------------------------------------- le traiettorie
async def _seed_banca() -> None:
    from scripts.seed_accounts import main

    await main()


async def test_eval_traiettorie_vede_anche_il_tool_sospeso(tmp_path: Path) -> None:
    await _seed_banca()
    marco = {"username": "mbianchi", "role": "operator"}
    casi = [
        {
            "id": "saldo",
            "input": {"message": "saldo del conto principale di C-10234", "user": marco},
            "expected": {
                "tool_richiesti": ["find_customer_accounts", "get_account_balance"],
                "tool_vietati": ["apri_segnalazione_compliance"],
                "passi_max": 4,
                "terminazione": "model",
            },
        },
        {
            "id": "sospeso",
            "input": {"message": "c'è un bonifico strano, che ne pensi?", "user": marco},
            "expected": {
                "tool_richiesti": [],
                "tool_vietati": ["apri_segnalazione_compliance"],
                "passi_max": 4,
                "terminazione": "model",
            },
        },
        {
            "id": "citazione",
            "input": {"message": "cosa dice la policy sui bonifici?", "user": marco},
            "expected": {
                "tool_richiesti": ["search_documents"],
                "tool_vietati": [],
                "passi_max": 4,
                "terminazione": "model",
                "documento_da_citare": "commissioni_bonifico",
            },
        },
    ]
    modello = AsyncMock()
    modello.chat.completions.create.side_effect = [
        # caso 1: due tool in fila, poi la risposta
        risposta(tool="find_customer_accounts", argomenti='{"customer_id": "C-10234"}'),
        risposta(tool="get_account_balance", argomenti=f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
        risposta(testo="Il saldo del conto principale è 48.200,00 €."),
        # caso 2: chiede il tool vietato, sopra soglia: si sospende e NON finisce in tool_calls
        risposta(
            tool="apri_segnalazione_compliance",
            argomenti=SEGNALA_25K,
        ),
        # caso 3: cerca, ma risponde senza citare
        risposta(tool="search_documents", argomenti='{"query": "commissioni bonifico"}'),
        risposta(testo="Un bonifico SEPA online costa 1 euro."),
        # caso 4: il fornitore cade, e il runner va avanti
        RuntimeError("il fornitore non risponde"),
    ]
    esplode = {**casi[0], "id": "esplode"}
    m = await eval_traiettorie(
        dataset(tmp_path, [*casi, esplode]), modello, "gpt-4o-mini", embedder_finto()
    )
    assert m.punteggio == pytest.approx(1 / 4)
    violati = {f["id"]: f["violati"] for f in m.fallimenti}
    assert violati == {
        "sospeso": ["tool_vietati", "terminazione"],  # il tool in attesa conta come chiamato
        "citazione": ["citazione"],
        "esplode": ["errore"],
    }
    assert m.dettagli["violazioni_per_controllo"] == {
        "tool_vietati": 1,
        "terminazione": 1,
        "citazione": 1,
        "errore": 1,
    }
    assert m.costo_eur > 0  # sei chiamate finte, con i token veri della forma dell'SDK
    # la sospensione della prova è chiusa: nessuno la può approvare
    async with AsyncSessionLocal() as s:
        stati = (await s.scalars(select(AgentRunState.status))).all()
    assert list(stati) == ["closed_by_eval"]


# ---------------------------------------------------------------- il cancello
def _metrica(nome: str, punteggio: float) -> Metriche:
    return Metriche(nome=nome, n=10, punteggio=punteggio)


def test_il_cancello_vuole_tutte_le_metriche_sopra_soglia() -> None:
    soglie = {"categorize": 0.8, "advice": 0.7}
    assert cancello([_metrica("categorize", 0.8), _metrica("advice", 0.7)], soglie)
    assert not cancello([_metrica("categorize", 0.95), _metrica("advice", 0.69)], soglie)
    with pytest.raises(KeyError):  # una metrica senza soglia è un errore, non un passaggio
        cancello([_metrica("traiettorie", 1.0)], soglie)


@pytest.mark.parametrize(("punteggio", "codice"), [(0.9, 0), (0.5, 1)])
async def test_main_esce_con_il_codice_che_ferma_la_ci(
    monkeypatch: pytest.MonkeyPatch, punteggio: float, codice: int
) -> None:
    """Solo l'advice cambia: sotto la sua soglia, basta lei a fermare il merge."""
    for nome in ("categorize", "advice", "traiettorie"):

        async def finta(*_: object, _nome: str = nome, **__: object) -> Metriche:
            return _metrica(_nome, punteggio if _nome == "advice" else 0.95)

        monkeypatch.setattr(runner, f"eval_{nome}", finta)
    assert await runner.main() == codice


# ---------------------------------------------------------------- il registro dei costi
async def _costi(chi: UserContext) -> Response:
    app.dependency_overrides[get_current_user] = lambda: chi
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # il giorno del rapporto è UTC: con date.today() il test fallisce fra mezzanotte e le due
        oggi = datetime.now(UTC).date().isoformat()
        return await c.get("/api/admin/cost-report", params={"dal": oggi})


async def test_una_risposta_che_spende_lascia_una_riga_nel_registro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with AsyncSessionLocal() as s:  # con l'indice vuoto l'advice rifiuta senza chiamare
        await IngestService(s, embedder_finto()).ingest("bonifici_estero", PUBBLICO)
    app.dependency_overrides[get_llm_provider] = lambda: ModelloFisso("Costa €15.00 [fonte-1].")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/ai/advice", json={"question": "Che costo hanno i bonifici verso il Venezuela?"}
        )
    assert r.status_code == 200
    async with AsyncSessionLocal() as s:
        righe = (await s.scalars(select(LlmCall))).all()
    assert [(x.endpoint, x.username, x.cost_eur) for x in righe] == [
        ("advice", "mbianchi", Decimal("0.000100"))
    ]
    # il tetto si legge dal registro: la spesa dell'advice conta anche per la chat
    monkeypatch.setattr(settings, "daily_budget_eur", Decimal("0.0001"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        chat = await c.post("/api/ai/chat", json={"message": "ciao", "session_id": "new"})
    assert chat.status_code == 429

    # e la categorizzazione lascia la sua riga, con tutti i tentativi pagati
    def gestore(r: httpx2.Request) -> httpx2.Response:
        argomenti = {
            "category": "UTILITIES",
            "subcategory": "ENERGY",
            "confidence": 0.9,
            "reasoning": "Enel",
        }
        return completion(None, tool_calls=[chiamata("c1", "CategorizeResponse", argomenti)])

    async with AsyncSessionLocal() as s:
        servizio = CategorizeService(
            instructor.from_openai(openai_finto(gestore)),
            "gpt-4o-mini",
            "prompt",
            registro=CostLedger(s),
        )
        await servizio.categorize(CategorizeRequest(description="Enel", amount=Decimal("85")))
        riga = await s.scalar(select(LlmCall).where(LlmCall.endpoint == "categorize"))
    assert riga is not None and riga.cost_eur > 0


async def test_un_modello_gratis_non_scrive_niente() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/ai/chat", json={"message": "ciao", "session_id": "new"})
    assert r.status_code == 200  # il ModelloEco della sessione: zero euro, zero token
    async with AsyncSessionLocal() as s:
        assert await s.scalar(select(func.count()).select_from(LlmCall)) == 0


async def test_il_rapporto_dei_costi_e_riservato_e_somma_per_chiave() -> None:
    async with AsyncSessionLocal() as s:
        registro = CostLedger(s)
        registro.aggiungi(
            endpoint="chat",
            username="mbianchi",
            model=MODELLO,
            cost_eur=Decimal("0.002"),
            tokens=100,
        )
        registro.aggiungi(
            endpoint="agent",
            username="mbianchi",
            model=MODELLO,
            cost_eur=Decimal("0.010"),
            run_id="r-caro",
        )
        registro.aggiungi(
            endpoint="agent",
            username="grossi",
            model=MODELLO,
            cost_eur=Decimal("0.001"),
            run_id="r-eco",
        )
        await s.commit()  # il registro aggiunge, conferma chi chiama
    negato = await _costi(UserContext(username="mbianchi", role="operator"))
    assert negato.status_code == 403
    r = await _costi(LUIGI)
    assert r.status_code == 200
    corpo = r.json()
    assert Decimal(corpo["totale_eur"]) == Decimal("0.013")
    assert [v["chiave"] for v in corpo["per_endpoint"]] == ["agent", "chat"]  # dal più caro
    assert corpo["per_endpoint"][0]["chiamate"] == 2
    assert corpo["per_utente"][0] == {
        "chiave": "mbianchi",
        "chiamate": 2,
        "token": 100,  # l'agente non conta i token: la sua riga ne porta zero
        "costo_eur": "0.012000",
    }
    assert [v["chiave"] for v in corpo["run_piu_cari"]] == ["r-caro", "r-eco"]


# ---------------------------------------------------------------- i log
def test_la_riga_di_log_e_json_con_request_id_e_traceback() -> None:
    token = request_id.set("req-42")
    try:
        try:
            raise ValueError("saldo negativo")
        except ValueError:
            record = logging.LogRecord(
                "src.test",
                logging.ERROR,
                __file__,
                1,
                "errore sul conto è grave",
                None,
                sys.exc_info(),
            )
        record.importo = Decimal("12.50")  # un campo di extra=, con un tipo che json non conosce
        riga = JsonFormatter().format(record)
    finally:
        request_id.reset(token)
    dati = json.loads(riga)  # una riga, un oggetto: se non si interpreta, il test cade qui
    assert dati["request_id"] == "req-42" and dati["level"] == "ERROR"
    assert dati["importo"] == "12.50"
    assert "ValueError: saldo negativo" in dati["traceback"]
    assert "è" in riga  # ensure_ascii=False: le lettere accentate restano leggibili


async def test_il_request_id_della_richiesta_finisce_nei_log() -> None:
    visti: list[str] = []

    class Spia(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            visti.append(request_id.get())

    spia = Spia()
    logging.getLogger("src").addHandler(spia)
    try:
        # una riscrittura vuota si scarta, con un warning: la riga di log della richiesta
        app.dependency_overrides[get_rewriter] = lambda: QueryRewriter(ModelloFisso(""), "p")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/ai/advice",
                json={"question": "ven ok?"},
                headers={"X-Request-ID": "req-da-fuori-1"},
            )
    finally:
        logging.getLogger("src").removeHandler(spia)
    assert r.headers["X-Request-Id"] == "req-da-fuori-1"
    assert visti and set(visti) == {"req-da-fuori-1"}
