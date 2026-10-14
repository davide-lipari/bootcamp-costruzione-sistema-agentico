# tests/test_g3_dura.py — cinquanta turni, una memoria, un modello che ragiona, il tempo indietro
import asyncio
import json
import sys
from collections.abc import Callable, Coroutine
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx2
import psycopg
import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import ValidationError

from evals.durata import misura, rapporto
from src.agents.deps import crea_deps
from src.auth.deps import UserContext
from src.config import settings
from src.db.session import AsyncSessionLocal, engine
from src.lg import memoria
from src.lg.conversazione import (
    MAX_MESSAGGI,
    aggiorna_fatti,
    costruisci_conversazione,
    domanda_con_contesto,
)
from src.lg.grafo import PREZZI, costruisci_grafo, stato_iniziale
from src.lg.stato import Contesto
from src.lg.tempo import cronologia, prova_da
from tests.conftest import CONTO_DI_MARCO
from tests.finti import GIULIA, completion, embedder_finto, openai_finto
from tests.test_g1_langchain import MARCO, Copione, banca, chat, tool
from tests.test_g2_langgraph import alert, copione_segnalazione

__all__ = ["banca"]

MARCO_U = UserContext(**MARCO)
PAOLA = UserContext(username="pgalli", role="operator")


def un_thread() -> RunnableConfig:
    return {"configurable": {"thread_id": f"mbianchi:{uuid4()}"}}


def ragiona(testo: str, ragionamento: int, uscita: int) -> httpx2.Response:
    """Una risposta di un modello che ragiona: i token di ragionamento stanno nell'uscita."""
    return httpx2.Response(
        200,
        json={
            "id": "r1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": testo},
                    "finish_reason": "stop" if testo else "length",
                }
            ],
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": uscita,
                "total_tokens": 300 + uscita,
                "completion_tokens_details": {"reasoning_tokens": ragionamento},
            },
        },
    )


def chat_ragiona(copione: Copione) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.default_model,
        api_key="sk-test",
        base_url="http://finto/v1",
        max_retries=0,
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(copione)),
    )


# ---------------------------------------------------------------- i fatti e il contesto
def test_i_fatti_si_prendono_dal_testo_con_il_codice() -> None:
    fatti = aggiorna_fatti(None, "parliamo del cliente C-10234", f"Il conto è {CONTO_DI_MARCO}.")
    assert fatti == {"cliente": "C-10234", "conti": [CONTO_DI_MARCO]}
    dopo = aggiorna_fatti(fatti, "e il saldo?", "Il saldo è 48.200,00 EUR.")
    assert dopo["cliente"] == "C-10234"  # una domanda senza cliente non lo cancella


def conversazione_lunga(turni: int) -> list[AnyMessage]:
    messaggi: list[AnyMessage] = [HumanMessage("parliamo del cliente C-10234"), AIMessage("Ok.")]
    for i in range(2, turni + 1):
        messaggi += [HumanMessage(f"domanda {i} sul saldo"), AIMessage(f"risposta {i}")]
    return [*messaggi, HumanMessage("e il saldo dell'altro conto?")]


def test_al_turno_cinquanta_il_cliente_del_primo_turno_c_e_ancora() -> None:
    messaggi = conversazione_lunga(50)
    fatti = aggiorna_fatti(None, "parliamo del cliente C-10234", "")
    testo = domanda_con_contesto(messaggi, fatti, "e il saldo dell'altro conto?")
    assert "C-10234" in testo
    # la finestra fissa del repository, venti messaggi, al turno cinquanta l'ha perso
    assert not any("C-10234" in m.text for m in messaggi[-20:])
    # e il contesto ha un budget: raggiunto, smette di crescere
    all_ottanta = domanda_con_contesto(
        conversazione_lunga(80), fatti, "e il saldo dell'altro conto?"
    )
    assert abs(len(testo) - len(all_ottanta)) <= 50


def test_il_contesto_non_comincia_mai_con_una_risposta_senza_domanda() -> None:
    # con risposte di lunghezza diversa il taglio del budget cade a volte su una risposta
    for coda in range(0, 120, 7):
        messaggi: list[AnyMessage] = [
            m if isinstance(m, HumanMessage) else AIMessage(m.text + " dettaglio" * (coda // 10))
            for m in conversazione_lunga(50)
        ]
        prima_riga = domanda_con_contesto(messaggi, None, "e il saldo?").splitlines()[1]
        assert prima_riga.startswith("operatore:"), (coda, prima_riga)


# ---------------------------------------------------------------- la conversazione intera
async def test_cinquanta_turni_lo_stato_resta_piccolo_e_il_cliente_resta() -> None:
    giri = []
    for _ in range(50):  # ascolto, triage, lo specialista delle policy, la sintesi
        giri += [
            completion("nessuna"),
            completion("policy"),
            completion("Niente da aggiungere."),
            completion("Ok."),
        ]
    copione = Copione(giri)
    config = un_thread()
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), InMemoryStore(), InMemorySaver()
        )
        await conv.ainvoke({"messages": [HumanMessage("parliamo del cliente C-10234")]}, config)
        for i in range(2, 51):
            out = await conv.ainvoke({"messages": [HumanMessage(f"domanda {i}")]}, config)
    assert out["turni"] == 50
    assert len(out["messages"]) <= MAX_MESSAGGI
    assert out["fatti"]["cliente"] == "C-10234"
    al_triage = json.dumps(copione.ricevute[-3], ensure_ascii=False)  # il triage del turno 50
    assert "C-10234" in al_triage
    # il triage del turno dieci, con lo stesso contesto di allora: la differenza è il tetto
    al_decimo = json.dumps(copione.ricevute[37], ensure_ascii=False)
    assert len(al_triage) < len(al_decimo) + 300


async def test_la_misura_della_durata_legge_tutto_dallo_stato() -> None:
    giri = []
    for turno in range(1, 21):  # una sonda ogni dieci turni: alla prima il cliente c'è, poi no
        sintesi = "Il cliente è C-10234." if turno == 10 else "Ok."
        giri += [
            completion("nessuna"),
            completion("policy"),
            completion("Niente."),
            completion(sintesi),
        ]
    copione = Copione(giri)
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), InMemoryStore(), InMemorySaver()
        )
        righe = await misura(conv, un_thread(), turni=20)
    assert [r.turno for r in righe] == list(range(1, 21))
    assert max(r.messaggi for r in righe) <= MAX_MESSAGGI
    assert righe[-1].cliente == "C-10234"
    assert [r.sonda for r in righe if r.sonda is not None] == [True, False]
    testo = rapporto(righe)
    assert len(testo.splitlines()) == 4  # l'intestazione, due blocchi, le sonde
    assert "1 su 2" in testo
    blocchi = [Decimal(riga.split("|")[3]) for riga in testo.splitlines()[1:3]]
    assert blocchi[0] == blocchi[1] > 0  # turni uguali, blocchi uguali: il blocco non è il totale


async def test_la_sosta_di_un_turno_ferma_la_conversazione_e_riprende() -> None:
    copione = Copione(
        [
            completion("nessuna"),  # un turno prima: la conversazione è già cominciata
            completion("policy"),
            completion("Niente."),
            completion("Ok."),
            completion("nessuna"),
            *copione_segnalazione(),
            completion("Segnalazione aperta."),
            completion("nessuna"),  # e un turno dopo la ripresa, sullo stesso thread
            completion("policy"),
            completion("Niente."),
            completion("Terzo turno."),
        ]
    )
    config = un_thread()
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), InMemoryStore(), InMemorySaver()
        )
        await conv.ainvoke({"messages": [HumanMessage("regole sul cliente C-10234?")]}, config)
        out = await conv.ainvoke({"messages": [HumanMessage("apri una segnalazione")]}, config)
        assert out["__interrupt__"][0].value["conto"] == CONTO_DI_MARCO
        assert await alert() == 0
        out = await conv.ainvoke(
            Command(resume={"approvato": True, "da": GIULIA.username, "motivo": None}), config
        )
        assert await alert() == 1
        assert out["messages"][-1].text == "Segnalazione aperta."
        out = await conv.ainvoke({"messages": [HumanMessage("e adesso?")]}, config)
    assert out["messages"][-1].text == "Terzo turno."
    assert out["turni"] == 3 and out["fatti"]["cliente"] == "C-10234"
    assert await alert() == 1  # il turno dopo non riscrive niente


# ---------------------------------------------------------------- la memoria
async def test_le_preferenze_sono_dell_operatore_e_si_dimenticano() -> None:
    store = InMemoryStore()
    await memoria.salva(store, MARCO_U, memoria.Preferenze(formato="dettagliato"))
    assert (await memoria.leggi(store, MARCO_U)).formato == "dettagliato"
    assert (await memoria.leggi(store, PAOLA)).formato == "breve"  # il namespace isola
    await memoria.dimentica(store, MARCO_U)
    assert (await memoria.leggi(store, MARCO_U)).formato == "breve"


def test_nella_memoria_un_cliente_non_ha_dove_entrare() -> None:
    with pytest.raises(ValidationError):
        memoria.Preferenze.model_validate({"formato": "breve", "cliente": "C-10234"})


async def test_la_preferenza_arriva_alla_sintesi() -> None:
    copione = Copione(
        [
            completion("nessuna"),
            completion("policy"),
            completion("Niente."),
            completion("Ok."),
        ]
    )
    store = InMemoryStore()
    await memoria.salva(store, MARCO_U, memoria.Preferenze(formato="dettagliato"))
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), store, InMemorySaver()
        )
        await conv.ainvoke({"messages": [HumanMessage("regole?")]}, un_thread())
    alla_sintesi = copione.ricevute[-1]["messages"][0]["content"]
    assert "in un elenco" in alla_sintesi


# ------------------------------------------------------ la strada B: la preferenza a richiesta
async def test_quando_l_operatore_chiede_una_preferenza_si_salva_e_si_vede_subito() -> None:
    copione = Copione(
        [
            completion("dettagliato"),  # l'ascolto: una parola, e la frase non entra nello store
            completion("entrambi"),
            completion("Conti trovati."),
            completion("Regole lette."),
            completion("Tutto qui."),
        ]
    )
    store = InMemoryStore()
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), store, InMemorySaver()
        )
        out = await conv.ainvoke(
            {"messages": [HumanMessage("d'ora in poi rispondimi in modo dettagliato")]},
            un_thread(),
        )
    voce = await store.asearch(memoria.namespace(MARCO_U))
    assert voce and voce[0].value == {"formato": "dettagliato"}  # solo lo schema, mai la frase
    # la conferma arriva nella risposta di quel turno: il cambio si vede subito
    assert "rispondo in modo dettagliato" in out["messages"][-1].text
    # e il turno già risponde nello stile nuovo: la preferenza non aspetta il turno dopo
    assert "in un elenco" in copione.ricevute[-1]["messages"][0]["content"]


async def test_dimentica_cancella_subito_e_l_operatore_lo_ve_de() -> None:
    store = InMemoryStore()
    await memoria.salva(store, MARCO_U, memoria.Preferenze(formato="dettagliato"))
    copione = Copione(
        [
            completion("dimentica"),  # l'ascolto capisce la cancellazione
            completion("entrambi"),
            completion("Conti trovati."),
            completion("Regole lette."),
            completion("Tutto qui."),
        ]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), store, InMemorySaver()
        )
        out = await conv.ainvoke(
            {"messages": [HumanMessage("dimentica le mie preferenze")]}, un_thread()
        )
    assert (await memoria.leggi(store, MARCO_U)).formato == "breve"  # cancellata, non scaduta
    assert await store.asearch(memoria.namespace(MARCO_U)) == []  # la voce non c'è più
    assert "dimenticato" in out["messages"][-1].text  # e l'operatore lo sa
    # niente stile: la risposta di quel turno è già quella di prima
    assert "in un elenco" not in copione.ricevute[-1]["messages"][0]["content"]


async def test_una_preferenza_in_una_domanda_non_porta_il_cliente_nella_memoria() -> None:
    copione = Copione(
        [
            completion("breve"),  # la domanda parla del cliente, ma la preferenza è solo «breve»
            completion("dati_conto"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            completion(f"Il conto è {CONTO_DI_MARCO}."),
            completion(f"Il conto è {CONTO_DI_MARCO}."),
            completion("nessuna"),  # e la domanda dopo, senza richieste, non scrive nulla
            completion("dati_conto"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            completion("Saldo 48.200,00 EUR."),
            completion("Saldo 48.200,00 EUR."),
        ]
    )
    store = InMemoryStore()
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        conv = costruisci_conversazione(
            MARCO_U, deps, chat(copione), chat(copione), store, InMemorySaver()
        )
        config = un_thread()
        await conv.ainvoke(
            {"messages": [HumanMessage("del cliente C-10234 quali conti ha? E da ora in breve")]},
            config,
        )
        await conv.ainvoke({"messages": [HumanMessage("e il saldo?")]}, config)
    voce = await store.asearch(memoria.namespace(MARCO_U))
    assert voce and voce[0].value == {"formato": "breve"}  # il cliente non ha dove entrare
    assert "C-10234" not in json.dumps(voce[0].value)  # e la domanda normale dopo non ha scritto


# ---------------------------------------------------------------- il modello che ragiona
async def test_l_istruttoria_usa_il_budget_dell_invocazione_e_paga_il_ragionamento() -> None:
    copione = Copione(
        [completion("istruttoria"), completion("Conto principale."), completion("Ok.")]
    )
    pensatore = Copione([ragiona("Due bonifici verso Panama da segnalare.", 850, 900)])
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            MARCO_U, deps, chat(copione), chat(copione), None, llm_ragiona=chat_ragiona(pensatore)
        )
        out = await g.ainvoke(
            stato_iniziale("analizza i movimenti di C-10234", MARCO_U),
            context=Contesto(sforzo="medium", max_token_istruttoria=2000),
        )
    richiesta = pensatore.ricevute[0]
    assert richiesta["reasoning_effort"] == "medium"
    assert richiesta["max_completion_tokens"] == 2000
    # i soli suoi tool, e i conti già trovati: è per questo che la riga passa prima dai conti
    offerti = sorted(t["function"]["name"] for t in richiesta.get("tools", []))
    assert offerti == ["list_recent_movements", "search_documents"]
    assert "Conto principale." in json.dumps(richiesta["messages"], ensure_ascii=False)
    [contributo] = [c for c in out["contributi"] if c["specialista"] == "istruttoria"]
    assert contributo["completo"] is True
    ingresso, uscita = PREZZI[settings.default_model]
    pensiero = (300 * ingresso + 900 * uscita) / 1000  # i 850 di ragionamento stanno nei 900
    ti, tu = PREZZI[settings.triage_model]
    triage = (100 * ti + 20 * tu) / 1000
    una = (100 * ingresso + 20 * uscita) / 1000  # lo specialista dei conti, e la sintesi
    # esatto, non «almeno»: il ragionamento contato una seconda volta farebbe un numero più alto
    assert out["costo_eur"] == triage + 2 * una + pensiero


async def test_a_budget_esaurito_il_ragionamento_non_e_una_risposta() -> None:
    copione = Copione(
        [completion("istruttoria"), completion("Conto principale."), completion("Ok.")]
    )
    pensatore = Copione([ragiona("", 2000, 2000)])  # pagato tutto, e la risposta non c'è
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(
            MARCO_U, deps, chat(copione), chat(copione), None, llm_ragiona=chat_ragiona(pensatore)
        )
        out = await g.ainvoke(
            stato_iniziale("analizza i movimenti di C-10234", MARCO_U),
            context=Contesto(max_token_istruttoria=2000),
        )
    [contributo] = [c for c in out["contributi"] if c["specialista"] == "istruttoria"]
    assert contributo["completo"] is False
    assert "budget di ragionamento" in contributo["contenuto"]


async def test_senza_un_modello_che_ragiona_lo_sforzo_non_parte() -> None:
    copione = Copione(
        [
            completion("istruttoria"),
            completion("Conto principale."),
            completion("Niente."),
            completion("Ok."),
        ]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(MARCO_U, deps, chat(copione), chat(copione), None)
        await g.ainvoke(
            stato_iniziale("analizza i movimenti di C-10234", MARCO_U),
            context=Contesto(sforzo="high"),
        )
    assert all("reasoning_effort" not in r for r in copione.ricevute)


# ---------------------------------------------------------------- cosa vede l'operatore
async def test_le_righe_di_avanzamento_arrivano_in_ordine_e_senza_dati() -> None:
    copione = Copione(
        [
            completion("dati_conto"),
            tool("find_customer_accounts", '{"customer_id": "C-10234"}'),
            completion(f"Conto {CONTO_DI_MARCO}."),
            completion(f"Il conto è {CONTO_DI_MARCO}."),
        ]
    )
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(MARCO_U, deps, chat(copione), chat(copione), None)
        righe = [
            pezzo["fase"]
            async for pezzo in g.astream(
                stato_iniziale("i conti di C-10234", MARCO_U), stream_mode="custom"
            )
        ]
    assert righe == [
        "Sto capendo la richiesta",
        "Sto leggendo i conti",
        "Sto scrivendo la risposta",
    ]
    assert not any(CONTO_DI_MARCO in r for r in righe)


# ---------------------------------------------------------------- il tempo all'indietro
async def test_un_ramo_di_prova_si_ferma_prima_di_scrivere_e_la_storia_resta() -> None:
    sotto_soglia = json.dumps(
        {"account_id": CONTO_DI_MARCO, "motivo": "Due bonifici verso Panama", "importo": "100"}
    )
    copione = Copione(
        [
            completion("policy"),  # l'esecuzione vera: una domanda di policy
            completion("Niente."),
            completion("Ok."),
            completion("Conto principale."),  # il ramo: dati, policy, proposta sotto soglia
            completion("Policy letta."),
            tool("SegnalazioneArgs", sotto_soglia),
        ]
    )
    saver, config = InMemorySaver(), un_thread()
    async with AsyncSessionLocal() as s:
        deps = crea_deps(s, embedder=embedder_finto(), openai=openai_finto(copione))
        g = costruisci_grafo(MARCO_U, deps, chat(copione), chat(copione), saver)
        await g.ainvoke(stato_iniziale("regole?", MARCO_U), config)
        storia = await cronologia(g, config)
        dopo_il_triage = next(p for p in storia if p.visitati == ["triage"])
        ramo = await prova_da(g, dopo_il_triage, {"instradamento": "segnalazione"})
        assert ramo["visitati"][-1] == "chiedi"  # sotto soglia nessuna sosta: si ferma il ramo
        assert await alert() == 0  # e prima di scrivere
        dopo = await cronologia(g, config)  # riletta: la storia originale c'è ancora, intera
        ids = {p.config["configurable"]["checkpoint_id"] for p in dopo}
        assert {p.config["configurable"]["checkpoint_id"] for p in storia} <= ids
        assert len(dopo) > len(storia)
        finale = next(p for p in dopo if p.config == storia[-1].config)
        assert finale.visitati[-1] == "sintesi"


# ---------------------------------------------------------------- la memoria su Postgres
def nel_loop_di_psycopg(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    fabbrica = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    asyncio.run(scenario(), loop_factory=fabbrica)


async def test_la_memoria_su_postgres_isola_solo_con_la_chiave_intera() -> None:
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def scenario() -> None:
        try:
            async with memoria.store_postgres() as store:
                await memoria.salva(store, MARCO_U, memoria.Preferenze(formato="dettagliato"))
                await memoria.salva(store, PAOLA, memoria.Preferenze(formato="breve"))
                assert (await memoria.leggi(store, MARCO_U)).formato == "dettagliato"
                assert (await memoria.leggi(store, PAOLA)).formato == "breve"
                # il prefisso da solo restituisce tutti: è per questo che la chiave è intera
                tutti = await store.asearch(("preferenze",))
                assert {"mbianchi", "pgalli"} <= {v.namespace[1] for v in tutti}
                # la conservazione dichiarata dallo store: novanta giorni, sulla riga
                async with await psycopg.AsyncConnection.connect(dsn) as c:
                    cur = await c.execute(
                        "select ttl_minutes from store where prefix = %s and key = %s",
                        ("preferenze.mbianchi", memoria.CHIAVE),
                    )
                    riga = await cur.fetchone()
                assert riga is not None and riga[0] == memoria.GIORNI * 24 * 60
                await memoria.dimentica(store, MARCO_U)
                await memoria.dimentica(store, PAOLA)
        finally:
            await engine.dispose()

    await asyncio.to_thread(nel_loop_di_psycopg, scenario)


async def test_una_preferenza_scaduta_non_si_legge_piu() -> None:
    async def scenario() -> None:
        try:
            # una scadenza di poco più di un secondo: lo spazzino passa ogni cinque minuti
            async with memoria.store_postgres(scadenza_minuti=0.02) as store:
                await memoria.salva(store, MARCO_U, memoria.Preferenze(formato="dettagliato"))
                assert (await memoria.leggi(store, MARCO_U)).formato == "dettagliato"
                await asyncio.sleep(2.5)
                # scaduta: la lettura torna al predefinito, prima che lo spazzino la tolga
                assert (await memoria.leggi(store, MARCO_U)).formato == "breve"
                await memoria.dimentica(store, MARCO_U)
        finally:
            await engine.dispose()

    await asyncio.to_thread(nel_loop_di_psycopg, scenario)


def test_anche_le_tabelle_della_memoria_restano_fuori_dalle_migration() -> None:
    from src.lg.grafo import fuori_dalle_migration

    # lo store se le crea da sé come il checkpointer: un autogenerate le cancellerebbe
    for nome in ("store", "store_migrations"):
        assert not fuori_dalle_migration(None, nome, "table", True, None)
