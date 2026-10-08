# tests/test_g1_langchain.py — tre implementazioni, stessi casi, stesso harness
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from evals.confronto import confronta
from evals.implementazioni import AgentePrebuilt, Catena, CicloAMano
from src.db.session import AsyncSessionLocal, engine
from src.services.ingest_service import IngestService
from tests.conftest import CONTO_DI_MARCO
from tests.finti import completion, embedder_finto, openai_finto

MODELLO = "gpt-4o-mini"
MARCO = {"username": "mbianchi", "role": "operator"}
POLICY = "# Commissioni bonifico\nBonifico SEPA: online 1,00 EUR, allo sportello 5,00 EUR."
SEGNALA_25K = json.dumps(
    {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama", "importo": "25000"}
)
SEGNALA_SENZA_IMPORTO = json.dumps(  # valida: l'importo è facoltativo, e il tool scriverebbe
    {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama"}
)
SEGNALA_ILLEGGIBILE = json.dumps(  # non valida: il tool la rifiuterebbe comunque
    {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama", "importo": "tanti"}
)
CASI = [
    {
        "id": "saldo",
        "input": {"message": "saldo del conto principale di C-10234", "user": MARCO},
        "expected": {
            "tool_richiesti": ["find_customer_accounts", "get_account_balance"],
            "tool_vietati": ["apri_segnalazione_compliance"],
            "passi_max": 4,
            "terminazione": "model",
        },
    },
    {
        "id": "segnalazione",
        "input": {"message": "apri una segnalazione: due bonifici verso Panama", "user": MARCO},
        "expected": {
            "tool_richiesti": ["apri_segnalazione_compliance"],
            "tool_vietati": [],
            "passi_max": 4,
            "terminazione": "awaiting_approval",
        },
    },
    {
        "id": "policy",
        "input": {"message": "quanto costa un bonifico SEPA online?", "user": MARCO},
        "expected": {
            "tool_richiesti": ["search_documents"],
            "tool_vietati": [],
            "passi_max": 4,
            "terminazione": "model",
            "documento_da_citare": "commissioni_bonifico",
        },
    },
]


class Copione:
    """Il modello finto: risponde nell'ordine scritto dal test, e annota cosa ha ricevuto."""

    def __init__(self, risposte: list[httpx2.Response]) -> None:
        self.risposte = list(risposte)
        self.ricevute: list[dict[str, Any]] = []

    def __call__(self, r: httpx2.Request) -> httpx2.Response:
        self.ricevute.append(json.loads(r.content))
        return self.risposte.pop(0)


def tool(nome: str, argomenti: str, i: int = 1) -> httpx2.Response:
    return completion(
        None,
        tool_calls=[
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": nome, "arguments": argomenti},
            }
        ],
    )


def chat(copione: Copione) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODELLO,
        api_key="sk-test",
        base_url="http://finto/v1",
        max_retries=0,
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(copione)),
    )


@pytest.fixture(autouse=True)
async def banca() -> AsyncIterator[None]:
    await engine.dispose()
    async with engine.begin() as c:
        await c.execute(
            text(
                "TRUNCATE llm_calls, document_chunks, agent_runs, compliance_alerts, "
                "movements, accounts, customers CASCADE"
            )
        )
    from scripts.seed_accounts import main

    await main()
    async with AsyncSessionLocal() as s:
        await IngestService(s, embedder_finto()).ingest("commissioni_bonifico", POLICY)
    # le connessioni aperte qui restano legate a questo event loop: né il test, che può girare
    # su un altro, né quello che viene dopo devono ritrovarle nel pool
    await engine.dispose()
    yield
    await engine.dispose()


def dataset(tmp_path: Path, casi: list[dict[str, Any]]) -> Path:
    p = tmp_path / "casi.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in casi), encoding="utf-8")
    return p


async def test_tre_implementazioni_sugli_stessi_casi(tmp_path: Path) -> None:
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
    risultati = await confronta(
        dataset(tmp_path, CASI),
        [
            CicloAMano(openai_finto(ciclo), MODELLO),
            Catena(chat(catena), MODELLO),
            AgentePrebuilt(chat(agente), MODELLO),
        ],
        embedder_finto(),
        openai_finto(ciclo),
    )
    esiti = {
        n: (m.punteggio, {f["id"]: f["violati"] for f in m.fallimenti})
        for n, m in risultati.items()
    }
    # il ciclo e l'agente fanno le stesse tre cose, sosta compresa
    assert esiti["ciclo"] == (1.0, {})
    assert esiti["agente"] == (1.0, {}), esiti["agente"]
    # la catena regge la sola domanda di policy, e la fonte inventata non passa
    assert esiti["catena"][0] == pytest.approx(1 / 3)
    assert set(esiti["catena"][1]) == {"saldo", "segnalazione"}
    assert all(not c for c in (ciclo.risposte, catena.risposte, agente.risposte))  # copioni finiti
    for m in risultati.values():
        assert m.costo_eur > 0
    # la sosta del ciclo si è vista, e la misura non l'ha lasciata da decidere a qualcuno
    async with AsyncSessionLocal() as s:
        stati: list[str] = list((await s.execute(text("select status from agent_runs"))).scalars())
    assert "closed_by_eval" in stati and "awaiting_approval" not in stati


async def test_la_sosta_dell_agente_decide_sugli_argomenti_validati(tmp_path: Path) -> None:
    """Un importo illeggibile non è un importo piccolo: `when` riceve gli argomenti grezzi, e nel
    dubbio chiede. Che poi il tool lo rifiuterebbe comunque non è un presidio su cui contare."""
    caso = dict(CASI[1], id="illeggibile")
    agente = Copione([tool("apri_segnalazione_compliance", SEGNALA_ILLEGGIBILE)])
    risultati = await confronta(
        dataset(tmp_path, [caso]),
        [AgentePrebuilt(chat(agente), MODELLO)],
        embedder_finto(),
        openai_finto(agente),
    )
    assert risultati["agente"].punteggio == 1.0, risultati["agente"].fallimenti


async def test_la_catena_non_cita_una_fonte_che_il_retrieval_non_ha_dato() -> None:
    from src.agents.deps import crea_deps
    from src.auth.deps import UserContext

    catena = Copione(
        [
            tool(
                "RispostaCitata",
                json.dumps(
                    {"risposta": "Costa 1 euro.", "fonti": ["commissioni_bonifico", "inventata"]}
                ),
            )
        ]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(catena))
        esito = await Catena(chat(catena), MODELLO).rispondi(
            "quanto costa un bonifico SEPA online?", UserContext(**MARCO), deps
        )
    assert "[commissioni_bonifico]" in esito.reply
    assert "[inventata]" not in esito.reply
    # e il contesto che il modello ha ricevuto porta l'identificativo che poi deve citare
    assert "[commissioni_bonifico]" in json.dumps(catena.ricevute[0], ensure_ascii=False)


async def test_il_tetto_dell_agente_non_diventa_una_risposta(tmp_path: Path) -> None:
    """Col tetto, il middleware scrive un messaggio dell'assistente: non è una risposta."""
    atteso = {"tool_richiesti": [], "tool_vietati": [], "passi_max": 2, "terminazione": "max_steps"}
    caso = {"id": "tetto", "input": CASI[0]["input"], "expected": atteso}
    giri = [
        tool("list_recent_movements", f'{{"account_id": "{CONTO_DI_MARCO}"}}', i) for i in range(5)
    ]
    agente = Copione(giri)
    impl = AgentePrebuilt(chat(agente), MODELLO, max_steps=2)
    risultati = await confronta(
        dataset(tmp_path, [caso]), [impl], embedder_finto(), openai_finto(agente)
    )
    assert risultati["agente"].punteggio == 1.0, risultati["agente"].fallimenti
    assert len(agente.risposte) == 3  # due chiamate vere; la terza non è partita


@pytest.mark.parametrize(
    ("decisione", "scritte", "segnalazione"),
    [
        ("approve", 1, SEGNALA_25K),
        ("reject", 0, SEGNALA_25K),
        # senza importo la segnalazione è valida, e scriverebbe: è il caso che il `when` non deve
        # trattare come piccolo, perché è l'unico in cui una scrittura partirebbe davvero
        ("approve", 1, SEGNALA_SENZA_IMPORTO),
        ("reject", 0, SEGNALA_SENZA_IMPORTO),
    ],
    ids=[
        "sopra-soglia-approvata",
        "sopra-soglia-respinta",
        "senza-importo-approvata",
        "senza-importo-respinta",
    ],
)
async def test_dopo_la_sosta_decide_una_persona(
    decisione: str, scritte: int, segnalazione: str
) -> None:
    """La sosta si riprende con la decisione, sullo stesso thread e con gli stessi tool."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from src.agents.deps import crea_deps
    from src.agents.tools import build_tools_for
    from src.auth.deps import UserContext
    from src.lc.agente import come_e_finita, costruisci_agente

    copione = Copione(
        [tool("apri_segnalazione_compliance", segnalazione), completion("Fatto, o respinta.")]
    )
    memoria = InMemorySaver()  # la sosta aspetta qui: un checkpointer per richiesta la perderebbe
    config: Any = {"configurable": {"thread_id": "mbianchi:prova"}}
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        agente = costruisci_agente(
            chat(copione), build_tools_for(UserContext(**MARCO), deps), memoria
        )
        stato = await agente.ainvoke({"messages": [HumanMessage("apri la segnalazione")]}, config)
        assert come_e_finita(stato)[0] == "awaiting_approval"
        prima: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
        stato = await agente.ainvoke(
            Command(
                resume={"decisions": [{"type": decisione, "message": "motivo del responsabile"}]}
            ),
            config,
        )
        assert come_e_finita(stato)[0] == "model"
        dopo: int = (await s.execute(text("select count(*) from compliance_alerts"))).scalar_one()
    assert (prima, dopo) == (0, scritte)


async def test_un_uscita_fuori_schema_e_un_errore_e_il_costo_resta() -> None:
    from src.agents.deps import crea_deps
    from src.auth.deps import UserContext

    catena = Copione([tool("RispostaCitata", json.dumps({"risposta": "Costa 1 euro."}))])
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(catena))
        esito = await Catena(chat(catena), MODELLO).rispondi(
            "quanto costa un bonifico SEPA online?", UserContext(**MARCO), deps
        )
    # senza `fonti` lo schema è violato: è un caso fallito, e la chiamata si è pagata lo stesso
    assert (esito.stopped_by, esito.reply) == ("errore", "")
    assert esito.cost_eur > 0


async def test_la_catena_non_porta_al_modello_un_documento_che_il_ruolo_non_vede() -> None:
    from src.agents.deps import crea_deps
    from src.auth.deps import UserContext

    # vicino alla domanda apposta: senza il filtro del ruolo, la ricerca lo restituirebbe
    riservato = "# Bonifico SEPA\nQuanto costa un bonifico SEPA verso controparti segnalate."
    async with AsyncSessionLocal() as s:
        await IngestService(s, embedder_finto()).ingest(
            "controparti_riservato", riservato, visibility="compliance_only"
        )
    catena = Copione(
        [tool("RispostaCitata", json.dumps({"risposta": "Costa 1 euro.", "fonti": []}))]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(catena))
        await Catena(chat(catena), MODELLO).rispondi(
            "quanto costa un bonifico SEPA?", UserContext(**MARCO), deps
        )
    contesto = json.dumps(catena.ricevute[0], ensure_ascii=False)
    # il controllo è su quello che il modello ha ricevuto, non sulla risposta: il finto risponde
    # sempre uguale, e un passaggio riservato nel contesto sarebbe già una fuoriuscita
    assert "[commissioni_bonifico]" in contesto
    assert "controparti_riservato" not in contesto


async def _agente_su(
    copione: Copione, max_steps: int = 6
) -> tuple[AsyncSession, CompiledStateGraph[Any, Any, Any, Any]]:
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agents.deps import crea_deps
    from src.agents.tools import build_tools_for
    from src.auth.deps import UserContext
    from src.lc.agente import costruisci_agente

    s = AsyncSessionLocal()
    deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
    tools = build_tools_for(UserContext(**MARCO), deps)
    return s, costruisci_agente(chat(copione), tools, InMemorySaver(), max_steps)


def _osservazioni(stato: dict[str, Any]) -> list[str]:
    from langchain_core.messages import ToolMessage

    return [m.text for m in stato["messages"] if isinstance(m, ToolMessage)]


async def test_l_agente_non_vede_un_cliente_di_un_altro_portafoglio() -> None:
    from src.agents.deps import crea_deps
    from src.auth.deps import UserContext

    copione = Copione(
        [tool("find_customer_accounts", '{"customer_id": "C-20417"}'), completion("Non risulta.")]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        await AgentePrebuilt(chat(copione), MODELLO).rispondi(
            "i conti di C-20417", UserContext(**MARCO), deps
        )
    # il cliente è di un altro operatore: quello che il modello ha ricevuto dal tool lo dice, e
    # non porta né il nome né i conti del cliente. Il finto risponde comunque, quindi si guarda qui
    ritorno = json.dumps(copione.ricevute[1]["messages"][-1], ensure_ascii=False)
    assert "Non risulta nel portafoglio di questo operatore." in ritorno
    assert "IT60X0542811101000000789" not in ritorno and "Greco" not in ritorno


async def test_un_tool_che_si_rompe_torna_al_modello_come_osservazione() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from pydantic import BaseModel

    from src.agents.registry import Tool
    from src.lc.agente import come_e_finita, costruisci_agente
    from src.lc.strumenti import fallito

    class Niente(BaseModel):
        pass

    async def rotto(_: BaseModel) -> str:
        raise RuntimeError("database giù")

    copione = Copione([tool("rotto", "{}"), completion("Il dato ora non è disponibile.")])
    agente = costruisci_agente(
        chat(copione),
        [Tool("rotto", "Un tool che si rompe.", Niente, rotto, scrive=False)],
        InMemorySaver(),
    )
    stato = await agente.ainvoke(
        {"messages": [HumanMessage("prova")]}, {"configurable": {"thread_id": "mbianchi:rotto"}}
    )
    # come nel ciclo: l'eccezione diventa una frase per il modello, e l'esecuzione finisce bene
    assert _osservazioni(stato) == [fallito("rotto")]
    assert come_e_finita(stato)[0] == "model"


async def test_due_letture_nello_stesso_passo_vanno_in_fila() -> None:
    from langchain_core.messages import HumanMessage

    risparmio = "IT60X0542811101000000456"
    due = completion(
        None,
        tool_calls=[
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "arguments": json.dumps({"account_id": c}),
                },
            }
            for i, c in enumerate((CONTO_DI_MARCO, risparmio), start=1)
        ],
    )
    copione = Copione([due, completion("I saldi.")])
    s, agente = await _agente_su(copione)
    async with s:
        stato = await agente.ainvoke(
            {"messages": [HumanMessage("i saldi dei due conti")]},
            {"configurable": {"thread_id": "mbianchi:due"}},
        )
    # il nodo dei tool le lancia insieme; la sessione condivisa ne reggerebbe una: la fila le
    # rimette una dopo l'altra, come il ciclo del repository
    assert sorted(_osservazioni(stato)) == [
        f"Saldo di {CONTO_DI_MARCO}: 48200.00 EUR",
        f"Saldo di {risparmio}: 1200.00 EUR",
    ]


async def test_il_tetto_dell_agente_conta_anche_i_passi_prima_della_sosta() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from src.lc.agente import chiamate_vere

    letture = [
        tool("list_recent_movements", f'{{"account_id": "{CONTO_DI_MARCO}"}}', i)
        for i in range(2, 9)
    ]
    copione = Copione(
        [
            tool("list_recent_movements", f'{{"account_id": "{CONTO_DI_MARCO}"}}'),
            tool("apri_segnalazione_compliance", SEGNALA_25K, 9),
            *letture,
        ]
    )
    config: Any = {"configurable": {"thread_id": "mbianchi:tetto"}}
    s, agente = await _agente_su(copione, max_steps=3)
    async with s:
        stato = await agente.ainvoke({"messages": [HumanMessage("analizza e segnala")]}, config)
        assert "__interrupt__" in stato
        stato = await agente.ainvoke(
            Command(resume={"decisions": [{"type": "reject", "message": "no"}]}), config
        )
    # due chiamate prima della sosta, una dopo: il tetto è sull'esecuzione, non sull'invocazione
    assert len(chiamate_vere(stato["messages"])) == 3
