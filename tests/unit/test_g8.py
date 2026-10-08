# tests/unit/test_g8.py — fermarsi, decidere, riprendere
import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai.types.chat import ChatCompletion
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.agents.deps import Deps
from src.agents.loop import SENZA_APPROVAZIONE, run_agent
from src.agents.tools import build_tools_for
from src.auth.deps import UserContext
from src.config import settings
from src.db.models import AgentRunState, ComplianceAlert
from tests.banco_app import crea_app, prepara_db
from tests.conftest import CONTO_DI_MARCO, risposta

RADICE = Path(__file__).resolve().parents[2]
SEGNALA_25K = (
    f'{{"account_id": "{CONTO_DI_MARCO}", '
    '"motivo": "Bonifico da 25.000 euro verso il Venezuela", "importo": "25000"}'
)
SEGNALA_30K = (
    f'{{"account_id": "{CONTO_DI_MARCO}", '
    '"motivo": "Secondo bonifico verso il Venezuela, 30.000 euro", "importo": "30000"}'
)


def chiede(*chiamate: tuple[str, str]) -> ChatCompletion:
    """Una risposta del modello che chiede uno o più tool nello stesso passo."""
    base = risposta(tool=chiamate[0][0], argomenti=chiamate[0][1])
    dati: dict[str, Any] = base.model_dump()
    dati["choices"][0]["message"]["tool_calls"] = [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
        for i, (n, a) in enumerate(chiamate, start=1)
    ]
    return ChatCompletion.model_validate(dati)


@pytest.fixture
async def fabbrica(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    return await prepara_db(tmp_path / "banca.db")


@pytest.fixture
def modello() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
async def client(
    fabbrica: async_sessionmaker[AsyncSession], modello: AsyncMock
) -> AsyncIterator[httpx.AsyncClient]:
    app = crea_app(fabbrica, modello)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def _conta(fabbrica: async_sessionmaker[AsyncSession], modello_db: type[Any]) -> int:
    async with fabbrica() as s:
        return await s.scalar(select(func.count()).select_from(modello_db)) or 0


async def _chiedi(client: httpx.AsyncClient, chi: str = "mbianchi") -> dict[str, Any]:
    r = await client.post(
        "/api/ai/agent",
        headers={"X-Utente": chi},
        json={"message": "apri una segnalazione sul conto principale"},
    )
    assert r.status_code == 200
    corpo: dict[str, Any] = r.json()
    return corpo


# ---------------------------------------------------------------- fermarsi
async def test_sopra_soglia_si_ferma_e_non_scrive(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.return_value = chiede(
        ("apri_segnalazione_compliance", SEGNALA_25K)
    )
    corpo = await _chiedi(client)
    assert corpo["stopped_by"] == "awaiting_approval"
    assert await _conta(fabbrica, ComplianceAlert) == 0  # nessun effetto: è la prova
    async with fabbrica() as s:
        stato = await s.get(AgentRunState, corpo["run_id"])
    assert stato is not None and stato.status == "awaiting_approval"
    assert "importo=25000" in stato.description


async def test_sotto_soglia_non_si_ferma(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.side_effect = [
        chiede(
            (
                "apri_segnalazione_compliance",
                f'{{"account_id": "{CONTO_DI_MARCO}", "motivo": "Addebito ripetuto da verificare",'
                ' "importo": "1200"}',
            )
        ),
        risposta(testo="Segnalazione aperta."),
    ]
    corpo = await _chiedi(client)
    assert corpo["stopped_by"] == "model"
    assert await _conta(fabbrica, ComplianceAlert) == 1


async def test_senza_importo_si_chiede(client: httpx.AsyncClient, modello: AsyncMock) -> None:
    modello.chat.completions.create.return_value = chiede(
        (
            "apri_segnalazione_compliance",
            f'{{"account_id": "{CONTO_DI_MARCO}", "motivo": "Movimenti anomali nel mese"}}',
        )
    )
    assert (await _chiedi(client))["stopped_by"] == "awaiting_approval"


async def test_nello_stesso_passo_non_parte_niente(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.side_effect = [
        chiede(
            ("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            ("apri_segnalazione_compliance", SEGNALA_25K),
        ),
        risposta(testo="Fatto."),
    ]
    corpo = await _chiedi(client)
    assert corpo["stopped_by"] == "awaiting_approval" and corpo["tool_calls"] == []

    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/approve", headers={"X-Utente": "grossi"}
    )
    assert r.json()["tool_calls"] == ["get_account_balance", "apri_segnalazione_compliance"]
    inviati = modello.chat.completions.create.call_args.kwargs["messages"]
    assert [m["tool_call_id"] for m in inviati if m["role"] == "tool"] == ["call_1", "call_2"]
    assert await _conta(fabbrica, ComplianceAlert) == 1


async def test_senza_dove_fermarsi_il_tool_non_parte(
    deps_reali: Deps, marco: UserContext, session: AsyncSession
) -> None:
    modello = AsyncMock()
    modello.chat.completions.create.side_effect = [
        chiede(("apri_segnalazione_compliance", SEGNALA_25K)),
        risposta(testo="Non posso."),
    ]
    await run_agent(
        messaggi=[{"role": "user", "content": "segnala"}],
        tools=build_tools_for(marco, deps_reali),
        client=modello,
        model="gpt-4o-mini",
    )  # niente runs, niente user
    ultimo = modello.chat.completions.create.call_args.kwargs["messages"][-1]
    assert ultimo["content"] == SENZA_APPROVAZIONE
    assert await session.scalar(select(func.count()).select_from(ComplianceAlert)) == 0


# ---------------------------------------------------------------- decidere
async def test_approvata_riprende_e_scrive(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.side_effect = [
        chiede(("apri_segnalazione_compliance", SEGNALA_25K)),
        chiede(("apri_segnalazione_compliance", SEGNALA_30K)),  # e si ferma di nuovo
        risposta(testo="Segnalazioni aperte, pratiche comunicate."),
    ]
    corpo = await _chiedi(client)
    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/approve", headers={"X-Utente": "grossi"}
    )
    assert r.status_code == 200 and r.json()["stopped_by"] == "awaiting_approval"
    assert r.json()["steps"] == 2  # il passo di prima conta
    assert await _conta(fabbrica, ComplianceAlert) == 1
    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/approve", headers={"X-Utente": "lverdi"}
    )
    assert r.status_code == 200 and r.json()["stopped_by"] == "model"
    # un importo diverso sullo stesso conto è un'altra pratica, non «già aperta»
    assert await _conta(fabbrica, ComplianceAlert) == 2
    async with fabbrica() as s:
        stato = await s.get(AgentRunState, corpo["run_id"])
    assert stato is not None and (stato.status, stato.decided_by) == ("done", "lverdi")
    # la seconda pausa non ha cancellato la prima firma: lo storico le tiene tutte e due
    firme = [(d["da"], d["esito"]) for d in stato.decisions]
    assert firme == [("grossi", "approvata"), ("lverdi", "approvata")]


async def test_chi_chiede_non_approva(client: httpx.AsyncClient, modello: AsyncMock) -> None:
    modello.chat.completions.create.return_value = chiede(
        ("apri_segnalazione_compliance", SEGNALA_25K)
    )
    corpo = await _chiedi(client, chi="lverdi")  # ha il ruolo per approvare
    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/approve", headers={"X-Utente": "lverdi"}
    )
    assert r.status_code == 403


async def test_un_operatore_non_approva(client: httpx.AsyncClient, modello: AsyncMock) -> None:
    modello.chat.completions.create.return_value = chiede(
        ("apri_segnalazione_compliance", SEGNALA_25K)
    )
    corpo = await _chiedi(client)
    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/approve", headers={"X-Utente": "mbianchi"}
    )
    assert r.status_code == 403


async def test_doppio_clic_una_sola_segnalazione(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.side_effect = [
        chiede(("apri_segnalazione_compliance", SEGNALA_25K)),
        risposta(testo="Segnalazione aperta."),
        risposta(testo="Segnalazione aperta."),
    ]
    corpo = await _chiedi(client)
    url, chi = f"/api/ai/agent/{corpo['run_id']}/approve", {"X-Utente": "grossi"}
    esiti = await asyncio.gather(client.post(url, headers=chi), client.post(url, headers=chi))
    assert sorted(r.status_code for r in esiti) == [200, 409]
    assert await _conta(fabbrica, ComplianceAlert) == 1


async def test_respinta_non_scrive_e_non_ritenta(
    client: httpx.AsyncClient, modello: AsyncMock, fabbrica: async_sessionmaker[AsyncSession]
) -> None:
    modello.chat.completions.create.side_effect = [
        chiede(("apri_segnalazione_compliance", SEGNALA_25K)),
        risposta(testo="La richiesta è stata respinta dalla Compliance."),
    ]
    corpo = await _chiedi(client)
    r = await client.post(
        f"/api/ai/agent/{corpo['run_id']}/reject",
        headers={"X-Utente": "grossi"},
        json={"motivo": "Operazione già segnalata dalla filiale ieri"},
    )
    assert r.status_code == 200
    osservazione = modello.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "non cercare un'altra strada" in osservazione and "già segnalata" in osservazione
    assert await _conta(fabbrica, ComplianceAlert) == 0
    async with fabbrica() as s:
        stato = await s.get(AgentRunState, corpo["run_id"])
    assert stato is not None and stato.status == "rejected"


# ---------------------------------------------------------------- il riavvio
async def test_l_approvazione_sopravvive_a_un_riavvio(
    client: httpx.AsyncClient,
    modello: AsyncMock,
    fabbrica: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    modello.chat.completions.create.return_value = chiede(
        ("apri_segnalazione_compliance", SEGNALA_25K)
    )
    corpo = await _chiedi(client)
    assert corpo["stopped_by"] == "awaiting_approval"
    assert await _conta(fabbrica, ComplianceAlert) == 0

    # il riavvio: un altro interprete Python, che di questo processo non ha niente in memoria
    processo = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.riavvio_altro_processo",
        str(tmp_path / "banca.db"),
        corpo["run_id"],
        cwd=RADICE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    uscita, errori = await asyncio.wait_for(processo.communicate(), timeout=60)
    assert processo.returncode == 0, errori.decode()
    assert json.loads(uscita.decode().strip().splitlines()[-1]) == {
        "status": 200,
        "stopped_by": "model",
    }
    assert await _conta(fabbrica, ComplianceAlert) == 1  # la segnalazione c'è adesso


# ---------------------------------------------------------------- il supervisor
async def test_il_triage_instrada_e_la_sintesi_riceve_dati(
    deps_reali: Deps, marco: UserContext
) -> None:
    from src.agents.supervisor import run_supervisor

    modello = cast(AsyncMock, deps_reali.openai)
    modello.chat.completions.create.side_effect = [
        risposta(testo="dati_conto"),  # triage
        risposta(tool="get_account_balance", argomenti=f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
        risposta(testo="Saldo: 48200.00 EUR."),  # specialista
        risposta(testo="Sul conto principale ci sono 48.200,00 euro."),  # sintesi
    ]
    esito = await run_supervisor(marco, "quanto c'è sul conto principale?", deps_reali)
    chiamate = modello.chat.completions.create.call_args_list
    assert chiamate[0].kwargs["model"] == settings.triage_model  # triage piccolo
    offerti = {s["function"]["name"] for s in chiamate[1].kwargs["tools"]}
    assert "search_documents" not in offerti and "apri_segnalazione_compliance" not in offerti
    materiale = json.loads(chiamate[-1].kwargs["messages"][-1]["content"])
    assert materiale["contributi"][0]["completo"] is True  # dati, non prosa
    assert esito.instradamento == "dati_conto" and esito.cost_eur > 0


async def test_un_triage_che_non_decide_manda_a_entrambi(
    deps_reali: Deps, marco: UserContext
) -> None:
    from src.agents.supervisor import run_supervisor

    modello = cast(AsyncMock, deps_reali.openai)
    modello.chat.completions.create.side_effect = [
        risposta(testo="Dipende da cosa intendi."),  # non è una parola
        risposta(testo="Nessun dato richiesto."),
        risposta(testo="Nessuna policy richiesta."),
        risposta(testo="Serve una domanda più precisa."),
    ]
    esito = await run_supervisor(marco, "com'è la situazione?", deps_reali)
    assert esito.instradamento == "entrambi" and len(esito.contributi) == 2
