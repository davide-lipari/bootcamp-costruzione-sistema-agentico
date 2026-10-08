# tests/test_g5.py — i documenti: il recupero su un Postgres con pgvector, e le quattro prove
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from pathlib import Path

import httpx2
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text

from src.auth.deps import get_current_user
from src.config import settings
from src.db.models import DocumentChunk
from src.db.session import AsyncSessionLocal, engine
from src.lib.chunking import chunk_text
from src.llm.embedding_client import EmbeddingClient
from src.llm.factory import get_embedder, get_llm_provider
from src.main import app
from src.services.ingest_service import IngestService
from src.services.rag_service import RIFIUTO, RAGService, citazioni
from src.services.retrieval_service import RetrievalResult, RetrievalService
from src.types.advice import AdviceResponse
from tests.finti import (
    GIULIA,
    MARCO,
    ModelloFisso,
    embedder_finto,
    openai_finto,
    riscrittore_spento,
    vettore,
)

COMMISSIONI = Path("data/docs/commissioni_bonifico.md").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
async def indice_vuoto() -> AsyncIterator[None]:
    await engine.dispose()  # le connessioni dei TestClient dei giorni prima: via dal pool
    async with engine.begin() as c:
        await c.execute(text("TRUNCATE document_chunks"))
    yield


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


async def ingerisci(document_id: str, testo: str) -> int:
    async with AsyncSessionLocal() as s:
        return await IngestService(s, embedder_finto()).ingest(
            document_id, testo, {"fonte": "test"}
        )


async def conta(document_id: str) -> int:
    async with AsyncSessionLocal() as s:
        quanti = await s.scalar(
            select(func.count()).where(DocumentChunk.document_id == document_id)
        )
        return int(quanti or 0)


async def chiedi(modello: ModelloFisso, domanda: str) -> AdviceResponse:
    async with AsyncSessionLocal() as s:
        servizio = RAGService(
            RetrievalService(s), embedder_finto(), modello, riscrittore_spento(), "PROMPT"
        )
        return await servizio.advise(domanda, MARCO)


# ---------------------------------------------------------------- il taglio
def test_chunk_text_taglia_con_sovrapposizione_e_senza_perdere_niente():
    testo = COMMISSIONI * 3
    pezzi = chunk_text(testo, chunk_size=500, overlap=50)
    assert len(pezzi) > 3 and all(len(p) <= 500 for p in pezzi)
    assert all(p in testo for p in pezzi)  # i pezzi vengono dal testo
    assert chunk_text("   ") == [] and chunk_text("breve") == ["breve"]


# ---------------------------------------------------------------- l'ingestione
async def test_ingerire_due_volte_sostituisce_e_non_raddoppia():
    primo = await ingerisci("commissioni_bonifico", COMMISSIONI)
    secondo = await ingerisci("commissioni_bonifico", COMMISSIONI)
    assert primo == secondo and await conta("commissioni_bonifico") == primo


async def test_la_pagina_ripetuta_non_produce_passaggi_identici():
    pagina = COMMISSIONI.split("## Bonifico estero")[0].strip()
    quanti = await ingerisci("pdf_unito_a_mano", pagina + "\n\n" + pagina)
    una_volta = await ingerisci("la_stessa_pagina_una_volta", pagina)
    assert quanti == una_volta  # la seconda copia non aggiunge niente
    async with AsyncSessionLocal() as s:
        testi = list(
            (
                await s.scalars(
                    select(DocumentChunk.content).where(
                        DocumentChunk.document_id == "pdf_unito_a_mano"
                    )
                )
            ).all()
        )
    assert quanti == len(testi) == len(set(testi))


async def test_il_vettore_torna_dal_database():
    await ingerisci("commissioni_bonifico", COMMISSIONI)
    async with AsyncSessionLocal() as s:
        uno = (await s.scalars(select(DocumentChunk).limit(1))).one()
    assert len(uno.embedding) == 1536 and uno.chunk_metadata == {"fonte": "test"}


# ---------------------------------------------------------------- recupero e risposta
async def test_la_domanda_fuori_dai_documenti_non_chiama_il_modello():
    await ingerisci("commissioni_bonifico", COMMISSIONI)
    modello = ModelloFisso("non dovrebbe servire")
    esito = await chiedi(modello, "Qual è il tasso del mutuo agevolato per giovani coppie?")
    assert esito.answer == RIFIUTO and esito.citations == [] and modello.ricevuti == []


async def test_si_cita_solo_quello_che_la_risposta_usa():
    await ingerisci("commissioni_bonifico", COMMISSIONI)
    modello = ModelloFisso("Il bonifico istantaneo SEPA costa €1.00 [fonte-1].")
    esito = await chiedi(modello, "Quanto costa il bonifico istantaneo SEPA?")
    utente = modello.ricevuti[0][1].content
    assert utente.startswith("CONTESTO\n[fonte-1] (documento: commissioni_bonifico)")
    assert len(esito.citations) == 1 and esito.citations[0].document_id == "commissioni_bonifico"


async def test_il_rifiuto_del_modello_non_porta_fonti():
    await ingerisci("commissioni_bonifico", COMMISSIONI)
    esito = await chiedi(ModelloFisso(RIFIUTO), "Bonifico SEPA verso la Svizzera: serve il BIC?")
    assert esito.answer == RIFIUTO and esito.citations == []


def test_un_marcatore_inventato_o_ripetuto_non_diventa_una_fonte():
    p = RetrievalResult(chunk_id="c1", document_id="d", content="testo", similarity=0.8)
    doppio = RetrievalResult(chunk_id="c2", document_id="d", content="testo", similarity=0.7)
    usate = citazioni("uno [fonte-1], due [fonte-2], tre [fonte-9], ancora [fonte-1]", [p, doppio])
    assert [c.chunk_id for c in usate] == ["c1"]


async def test_la_soglia_filtra_i_passaggi_lontani():
    await ingerisci("commissioni_bonifico", COMMISSIONI)
    async with AsyncSessionLocal() as s:
        vicini = await RetrievalService(s).search(vettore("costo bonifico istantaneo SEPA"))
        lontani = await RetrievalService(s).search(vettore("ricetta della torta di mele"))
    assert vicini and all(p.similarity >= 0.35 for p in vicini) and lontani == []
    assert vicini == sorted(vicini, key=lambda p: p.similarity, reverse=True)


# ---------------------------------------------------------------- gli endpoint
async def test_gli_endpoint_ingeriscono_e_rispondono(monkeypatch: pytest.MonkeyPatch):
    modello = ModelloFisso("Online il bonifico SEPA costa €1.00 [fonte-1].")
    app.dependency_overrides[get_llm_provider] = lambda: modello  # l'embedder è quello finto
    app.dependency_overrides[get_current_user] = lambda: GIULIA  # dal Giorno 6 ingerisce lei
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/ai/documents/ingest",
            json={
                "document_id": "commissioni_bonifico",
                "content": COMMISSIONI,
                "metadata": {"source": "data/docs/commissioni_bonifico.md"},
            },
        )
        a = await c.post(
            "/api/ai/advice", json={"question": "Quanto costa un bonifico SEPA online?"}
        )
    assert r.status_code == 200 and r.json()["chunk_count"] > 0
    assert a.status_code == 200 and len(a.json()["citations"]) == 1
    assert a.json()["cost_eur"] == 0.0001
    # un guasto degli embedding è un guasto del fornitore, come nella chat: il 502
    giu = EmbeddingClient(
        openai_finto(lambda r: httpx2.Response(503, json={"error": {"message": "giù"}}), 0),
        "text-embedding-3-small",
    )
    app.dependency_overrides[get_embedder] = lambda: giu
    domanda = {"question": "Quanto costa un bonifico SEPA online?"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        guasto = await c.post("/api/ai/advice", json=domanda)
        monkeypatch.setattr(settings, "daily_budget_eur", Decimal("0"))
        tetto = await c.post("/api/ai/advice", json=domanda)
    assert guasto.status_code == 502 and guasto.json()["error"] == "LLM_PROVIDER_ERROR"
    assert tetto.status_code == 429  # il tetto della chat vale anche per l'advisor
