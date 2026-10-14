# tests/test_g2_langgraph.py — il grafo: stesso banco del Giorno 1, più la sosta nel database
import asyncio
import json
import selectors
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from evals.confronto import confronta
from evals.implementazioni import AgentePrebuilt, Catena, CicloAMano, Grafo
from src.agents.deps import crea_deps
from src.agents.loop import BUDGET_PER_RUN
from src.auth.deps import UserContext
from src.db.session import AsyncSessionLocal
from src.lg.grafo import (
    MAX_PASSI,
    PERCORSI,
    TABELLE_DI_LANGGRAPH,
    checkpointer_postgres,
    chiudi,
    costruisci_grafo,
    fuori_dalle_migration,
    instrada,
    stato_iniziale,
    successori,
    thread_per,
)
from src.lg.stato import StatoBanca
from tests.conftest import CONTO_DI_MARCO
from tests.finti import completion, embedder_finto, openai_finto
from tests.test_g1_langchain import (  # noqa: F401 — banca è il fixture autouse del banco
    CASI,
    MARCO,
    MODELLO,
    SEGNALA_25K,
    Copione,
    banca,
    chat,
    dataset,
    tool,
)

SEGNALA_DOMANDA = "apri una segnalazione: due bonifici verso Panama"
SEGNALA_PICCOLO = json.dumps(  # sotto soglia: il grafo scrive senza chiedere a nessuno
    {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama", "importo": "100"}
)


def copione_segnalazione() -> list[httpx2.Response]:
    """Il cammino di una segnalazione sopra soglia fino alla proposta: il triage, i due
    specialisti, e la chiamata strutturata che prepara gli argomenti. La sosta viene dopo."""
    return [
        completion("segnalazione"),
        tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
        tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
        completion("Due bonifici verso Panama."),
        tool("search_documents", '{"query": "segnalazione compliance"}'),
        completion("Obbligatoria sopra soglia."),
        tool("SegnalazioneArgs", SEGNALA_25K),
    ]


async def alert() -> int:
    """Quante segnalazioni ci sono adesso: il conto che guarda chi aspetta la scrittura."""
    from sqlalchemy import text

    async with AsyncSessionLocal() as s:
        return int((await s.execute(text("select count(*) from compliance_alerts"))).scalar_one())


def _stato(**campi: object) -> StatoBanca:
    """Uno stato di partenza completo: i campi di lavoro servono solo a chi li legge."""
    return cast(
        StatoBanca,
        {
            "domanda": "x",
            "richiedente": "mbianchi",
            "contributi": [],
            "visitati": [],
            "passi": 0,
            "costo_eur": Decimal("0"),
            "instradamento": "entrambi",
            "proposta": None,
            "decisione": {"approvato": False, "da": None, "motivo": None},
            "risposta": "",
            "stopped_by": "model",
            **campi,
        },
    )


def _domanda_di(caso: dict[str, Any]) -> str:
    """La domanda di un caso del banco, senza far combattere mypy con i dizionari annidati."""
    ingresso = cast(dict[str, Any], caso["input"])
    return cast(str, ingresso["message"])


async def _run(
    copione: Copione, domanda: str, checkpointer: BaseCheckpointSaver[Any] | None
) -> dict[str, Any]:
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agents.deps import crea_deps
    from src.auth.deps import UserContext
    from src.lg.grafo import costruisci_grafo

    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            UserContext(**MARCO),
            deps,
            chat(copione),
            chat(copione),
            checkpointer or InMemorySaver(),
            modello=MODELLO,
            modello_triage=MODELLO,
        )
        return await g.ainvoke(
            stato_iniziale(domanda, UserContext(**MARCO)),
            {"configurable": {"thread_id": thread_per(UserContext(**MARCO))}},
        )


async def test_davanti_al_banco_il_grafo_paga_l_ordine_sulle_domande_semplici(
    tmp_path: Path,
) -> None:
    """Gli stessi tre casi su quattro implementazioni. Il grafa paga il suo ordine: triage e
    sintesi sono chiamate vere, e sui casi da quattro passi il tetto si vede."""
    ciclo = Copione(
        [
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Il saldo è di 48.200,00 EUR."),
            tool("apri_segnalazione_compliance", SEGNALA_25K),
            tool("search_documents", '{"query": "costo bonifico SEPA online"}'),
            completion("Costa 1 euro [commissioni_bonifico]."),
        ]
    )
    catena = Copione(
        [
            tool(
                "RispostaCitata",
                json.dumps({"risposta": "Nel contesto non ci sono saldi.", "fonti": []}),
            ),
            tool(
                "RispostaCitata",
                json.dumps({"risposta": "Non posso aprire segnalazioni.", "fonti": []}),
            ),
            tool(
                "RispostaCitata",
                json.dumps(
                    {"risposta": "Costa 1 euro.", "fonti": ["commissioni_bonifico", "inventata"]}
                ),
            ),
        ]
    )
    agente = Copione(
        [
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Il saldo è di 48.200,00 EUR."),
            tool("apri_segnalazione_compliance", SEGNALA_25K),
            tool("search_documents", '{"query": "costo bonifico SEPA online"}'),
            completion("Costa 1 euro [commissioni_bonifico]."),
        ]
    )
    grafo = Copione(
        [
            # saldo: triage, lo specialista dei dati, la sintesi
            completion("dati_conto"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Il saldo è di 48.200,00 EUR."),
            completion("Il saldo del conto principale è 48.200,00 EUR."),
            # segnalazione: i due specialisti, la proposta strutturata, e la sosta
            completion("segnalazione"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Ci sono due bonifici verso Panama sul conto principale."),
            tool("search_documents", '{"query": "segnalazione compliance bonifici esteri"}'),
            completion("La policy rende obbligatoria la segnalazione per bonifici verso Panama."),
            tool("SegnalazioneArgs", SEGNALA_25K),
            # policy: la ricerca dello specialista, e la sintesi che cita
            completion("policy"),
            tool("search_documents", '{"query": "costo bonifico SEPA online"}'),
            completion("Costa 1 euro [commissioni_bonifico]."),
            completion("Un bonifico SEPA online costa 1,00 EUR [commissioni_bonifico]."),
        ]
    )
    risultati = await confronta(
        dataset(tmp_path, CASI),
        [
            CicloAMano(openai_finto(ciclo), MODELLO),
            Catena(chat(catena), MODELLO),
            AgentePrebuilt(chat(agente), MODELLO),
            Grafo(chat(grafo), chat(grafo), MODELLO),
        ],
        embedder_finto(),
        openai_finto(ciclo),
    )
    esiti = {
        n: (m.punteggio, {f["id"]: f["violati"] for f in m.fallimenti})
        for n, m in risultati.items()
    }
    # il ciclo e l'agente fanno le stesse tre cose, sosta compresa: come al Giorno 1
    assert esiti["ciclo"] == (1.0, {})
    assert esiti["agente"] == (1.0, {}), esiti["agente"]
    # la catena regge la sola domanda di policy, e la fonte inventata non passa
    assert esiti["catena"][0] == pytest.approx(1 / 3)
    assert set(esiti["catena"][1]) == {"saldo", "segnalazione"}
    # il grafo risponde a tutte e tre, ma il triage e la sintesi si pagano: cinque chiamate
    # sul saldo e sette sulla segnalazione, contro i quattro passi del caso
    assert esiti["grafo"][0] == pytest.approx(1 / 3)
    assert esiti["grafo"][1] == {"saldo": ["passi"], "segnalazione": ["passi"]}
    assert all(not c for c in (ciclo.risposte, catena.risposte, agente.risposte, grafo.risposte))
    for m in risultati.values():
        assert m.costo_eur > 0


async def test_il_grafo_legge_prima_i_conti_poi_il_saldo() -> None:
    """L'ordine della tabella: lo specialista dei dati fa le letture in fila, una dopo l'altra,
    e la sintesi vede solo i contributi, non la conversazione."""
    copione = Copione(
        [
            completion("dati_conto"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Il saldo è di 48.200,00 EUR."),
            completion("Il saldo del conto principale è 48.200,00 EUR."),
        ]
    )
    stato = await _run(copione, _domanda_di(CASI[0]), None)
    assert stato["instradamento"] == "dati_conto"
    assert stato["visitati"] == ["triage", "dati_conto", "sintesi"]
    assert stato["contributi"][0]["tool_calls"] == [
        "find_customer_accounts",
        "get_account_balance",
    ]
    assert stato["stopped_by"] == "model"
    assert stato["risposta"] == "Il saldo del conto principale è 48.200,00 EUR."
    assert stato["costo_eur"] > 0


async def test_sotto_soglia_il_grafo_scrive_senza_chiedere() -> None:
    """Un importo piccolo non passa dal interrupt: la scrittura parte, e nessuno è coinvolto."""
    copione = Copione(
        [
            completion("segnalazione"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Due bonifici verso Panama."),
            tool("search_documents", '{"query": "segnalazione compliance"}'),
            completion("Obbligatoria sopra certi importi."),
            tool("SegnalazioneArgs", SEGNALA_PICCOLO),
            completion("Segnalazione aperta."),
        ]
    )
    async with AsyncSessionLocal() as s:
        from sqlalchemy import text

        from src.agents.deps import crea_deps
        from src.auth.deps import UserContext
        from src.lg.grafo import costruisci_grafo

        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            UserContext(**MARCO),
            deps,
            chat(copione),
            chat(copione),
            None,
            modello=MODELLO,
            modello_triage=MODELLO,
        )
        stato = await g.ainvoke(
            stato_iniziale(SEGNALA_DOMANDA, UserContext(**MARCO)),
            {"configurable": {"thread_id": thread_per(UserContext(**MARCO))}},
        )
        n: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
    assert "__interrupt__" not in stato
    assert stato["decisione"] == {"approvato": True, "da": None, "motivo": None}
    assert n == 1
    assert stato["stopped_by"] == "model"


async def test_sopra_soglia_il_grafo_si_ferma_e_una_persona_scrive() -> None:
    """La sosta: il grafo si ferma con gli argomenti validati, e alla ripresa la decisione
    arriva da fuori, sullo stesso thread."""
    copione = Copione(
        [
            completion("segnalazione"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Due bonifici verso Panama."),
            tool("search_documents", '{"query": "segnalazione compliance"}'),
            completion("Obbligatoria sopra soglia."),
            tool("SegnalazioneArgs", SEGNALA_25K),
            completion("Segnalazione aperta, la Compliance la prenderà in carico."),
        ]
    )
    async with AsyncSessionLocal() as s:
        from sqlalchemy import text

        from src.agents.deps import crea_deps
        from src.auth.deps import UserContext
        from src.lg.grafo import costruisci_grafo

        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            UserContext(**MARCO),
            deps,
            chat(copione),
            chat(copione),
            InMemorySaver(),
            modello=MODELLO,
            modello_triage=MODELLO,
        )
        config: Any = {"configurable": {"thread_id": thread_per(UserContext(**MARCO))}}
        stato = await g.ainvoke(stato_iniziale(SEGNALA_DOMANDA, UserContext(**MARCO)), config)
        assert stato["__interrupt__"], "il grafo non si è fermato sopra soglia"
        attesa = stato["__interrupt__"][0].value
        assert attesa["azione"] == "apri_segnalazione_compliance"
        assert attesa["importo"] == "25000"
        assert attesa["richiesto_da"] == MARCO["username"]
        prima: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
        stato = await g.ainvoke(
            Command(resume={"approvato": True, "da": "g.rossi", "motivo": "confermato"}), config
        )
        dopo: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
    assert (prima, dopo) == (0, 1)
    assert stato["stopped_by"] == "model"
    assert not copione.risposte, "il copione non è finito: la ripresa ha pagato un triage"


async def test_una_respinta_non_scrive_e_lo_dice_nella_risposta() -> None:
    copione = Copione(
        [
            completion("segnalazione"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Due bonifici verso Panama."),
            tool("search_documents", '{"query": "segnalazione compliance"}'),
            completion("Obbligatoria sopra soglia."),
            tool("SegnalazioneArgs", SEGNALA_25K),
            completion(
                "La Compliance, per bocca di g.rossi, ha respinto la segnalazione: "
                "motivo non priorità."
            ),
        ]
    )
    async with AsyncSessionLocal() as s:
        from sqlalchemy import text

        from src.agents.deps import crea_deps
        from src.auth.deps import UserContext
        from src.lg.grafo import costruisci_grafo

        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            UserContext(**MARCO),
            deps,
            chat(copione),
            chat(copione),
            InMemorySaver(),
            modello=MODELLO,
            modello_triage=MODELLO,
        )
        config: Any = {"configurable": {"thread_id": thread_per(UserContext(**MARCO))}}
        stato = await g.ainvoke(stato_iniziale(SEGNALA_DOMANDA, UserContext(**MARCO)), config)
        stato = await g.ainvoke(
            Command(resume={"approvato": False, "da": "g.rossi", "motivo": "non priorità"}),
            config,
        )
        n: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
    assert n == 0
    assert "g.rossi" in stato["risposta"] and "non priorità" in stato["risposta"]
    assert any("g.rossi" in c["contenuto"] for c in stato["contributi"])  # il rifiuto è lì


def test_la_sosta_sopravvive_a_un_checkpointer_nuovo() -> None:
    """La sosta non sta nel processo: un checkpointer nuovo, aperto sullo stesso database,
    riprende il thread dov'era — e solo la ripresa paga il modello. psycopg su Windows
    rifiuta il ProactorEventLoop: il test accende un selettore tutto suo."""
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    try:
        loop.run_until_complete(_la_sosta_sopravvive())
    finally:
        loop.close()


async def _la_sosta_sopravvive() -> None:
    copione = Copione(
        [
            completion("segnalazione"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            tool("get_account_balance", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            completion("Due bonifici verso Panama."),
            tool("search_documents", '{"query": "segnalazione compliance"}'),
            completion("Obbligatoria sopra soglia."),
            tool("SegnalazioneArgs", SEGNALA_25K),
            completion("Segnalazione aperta."),
        ]
    )
    from sqlalchemy import text

    from src.db.session import engine

    try:
        async with AsyncSessionLocal() as s:
            deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
            utente = UserContext(**MARCO)
            config: Any = {"configurable": {"thread_id": thread_per(utente)}}
            async with checkpointer_postgres() as saver:
                g1 = costruisci_grafo(
                    utente,
                    deps,
                    chat(copione),
                    chat(copione),
                    saver,
                    modello=MODELLO,
                    modello_triage=MODELLO,
                )
                stato = await g1.ainvoke(stato_iniziale(SEGNALA_DOMANDA, utente), config)
                assert "__interrupt__" in stato
            # il primo checkpointer si è chiuso: la sosta è nel database, non nel processo
            async with checkpointer_postgres() as nuovo:
                g2 = costruisci_grafo(
                    utente,
                    deps,
                    chat(copione),
                    chat(copione),
                    nuovo,
                    modello=MODELLO,
                    modello_triage=MODELLO,
                )
                stato = await g2.ainvoke(
                    Command(resume={"approvato": True, "da": "g.rossi", "motivo": "ok"}), config
                )
                n: int = (
                    await s.execute(text("select count(*) from compliance_alerts"))
                ).scalar_one()
        assert n == 1
        assert stato["stopped_by"] == "model"
        assert not copione.risposte
    finally:
        # le connessioni asyncpg aperte su questo loop non devono restare nel pool:
        # il dispose del fixture banca girerebbe su un altro loop e resterebbe appeso
        await engine.dispose()


def test_la_tabella_dichiara_un_insime_chiuso_di_percorsi() -> None:
    """Ogni percorso comincia da uno specialista, e segnalazione passa dalla sosta prima della
    scrittura: l'ordine sta nella tabella, non nel codice dei nodi."""
    assert set(PERCORSI) == {"dati_conto", "policy", "entrambi", "segnalazione", "istruttoria"}
    for percorso in PERCORSI.values():
        assert percorso[0] in {"dati_conto", "policy"}
    assert PERCORSI["segnalazione"] == [
        "dati_conto",
        "policy",
        "prepara",
        "chiedi",
        "scrivi",
    ]
    # da ogni nodo, solo le uscite che il flusso prevede: mai un arco a sorpresa
    assert successori("triage") == ["chiusura", "dati_conto", "policy", "sintesi"]
    assert successori("prepara") == ["chiedi", "chiusura", "sintesi"]
    assert successori("scrivi") == ["chiusura", "sintesi"]
    assert successori("sintesi") == ["chiusura", "sintesi"]


def test_i_tetti_vengono_prima_di_qualunque_scelta() -> None:
    """Il budget o i passi: la chiusura scatta prima di guardare il percorso, e se sono sforati
    tutti e due vince il costo."""
    assert instrada(_stato(costo_eur=BUDGET_PER_RUN + Decimal("0.01"))) == "chiusura"
    assert instrada(_stato(passi=MAX_PASSI)) == "chiusura"
    assert instrada(_stato(instradamento="dati_conto")) == "dati_conto"
    assert instrada(_stato(instradamento="dati_conto", visitati=["triage", "dati_conto"])) == (
        "sintesi"
    )
    # la proposta non c'è ancora: prepara deve girare, è lui a farla
    senza_proposta = _stato(instradamento="segnalazione", visitati=["dati_conto", "policy"])
    assert instrada(senza_proposta) == "prepara"


def test_la_chiusura_componi_quello_che_c_e_non_puo_fallire() -> None:
    chiuso_budget = chiudi(
        _stato(
            costo_eur=BUDGET_PER_RUN + Decimal("1"),
            passi=MAX_PASSI + 5,  # sforati tutti e due: vince il costo
            instradamento="segnalazione",
            visitati=["triage", "dati_conto"],
            contributi=[
                {
                    "specialista": "dati_conto",
                    "completo": True,
                    "contenuto": "Due bonifici verso Panama.",
                    "tool_calls": [],
                }
            ],
        )
    )
    assert chiuso_budget["stopped_by"] == "budget"
    assert "budget previsto" in chiuso_budget["risposta"]
    assert "Due bonifici verso Panama." in chiuso_budget["risposta"]
    assert "Resta da fare: policy" in chiuso_budget["risposta"]

    chiuso_passi = chiudi(_stato(passi=MAX_PASSI + 1, instradamento="policy"))
    assert chiuso_passi["stopped_by"] == "max_steps"
    assert "passi previsti" in chiuso_passi["risposta"]


async def test_sopra_il_budget_il_grafo_chiude_senza_modello() -> None:
    """Un triage costoso da solo supera il budget: il grafo non chiama nessuno e componi."""
    copione = Copione([completion("dati_conto", prompt=600_000)])
    stato = await _run(copione, _domanda_di(CASI[0]), None)
    assert stato["stopped_by"] == "budget"
    assert stato["costo_eur"] > BUDGET_PER_RUN
    assert not copione.risposte and len(copione.ricevute) == 1  # solo il triage


def test_il_thread_porta_l_username_e_un_uuid() -> None:
    utente = UserContext(**MARCO)
    assert thread_per(utente).startswith(f"{utente.username}:")
    assert thread_per(utente) != thread_per(utente)  # uno nuovo a ogni richiesta


def test_le_tabelle_del_checkpointer_non_nelle_migration() -> None:
    """Senza, `revision --autogenerate` vorrebbe cancellarle: e con le soste che contengono."""
    for tabella in TABELLE_DI_LANGGRAPH:
        assert not fuori_dalle_migration(None, tabella, "table", False, None)
    assert fuori_dalle_migration(None, "accounts", "table", False, None)
    assert fuori_dalle_migration(None, "checkpoints", "index", False, None)
