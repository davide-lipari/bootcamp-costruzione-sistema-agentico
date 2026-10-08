# src/agents/supervisor.py — un triage piccolo, due specialisti, una sintesi che non inventa
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from src.agents.deps import Deps
from src.agents.loop import RunResult, costo_chiamata, run_agent
from src.agents.tools import build_tools_for
from src.auth.deps import UserContext
from src.config import settings

# il modello del triage è configurabile: stesso fornitore del resto, ma il più piccolo che c'è

PROMPT_TRIAGE = """Classifica la domanda di un operatore di filiale. Rispondi con una parola sola:
dati_conto  se riguarda clienti, conti, saldi o movimenti;
policy      se riguarda regole, soglie, procedure o adempimenti;
entrambi    se servono tutte e due le cose, o se non sei sicuro."""

PROMPT_DATI_CONTO = """Sei lo specialista dei dati di conto di LipariBank.
Leggi i dati con i tool e riporta importi e date esatti, mai stime né arrotondamenti.
Non interpretare le policy: non sono il tuo dominio. Se la domanda non riguarda dati di
clienti, conti o movimenti, dillo in una frase e fermati."""

PROMPT_POLICY = """Sei lo specialista delle policy di LipariBank.
Rispondi solo da ciò che il tool ti restituisce, e cita ogni documento con il suo identificativo
fra parentesi quadre. Se il documento non c'è, dillo: non ricostruire regole a memoria.
Se la domanda non riguarda regole o procedure, dillo in una frase e fermati."""

PROMPT_SINTESI = """Componi la risposta per l'operatore a partire dai contributi in JSON.
Usa solo quello che i contributi contengono, con gli importi e le citazioni così come sono.
Se un contributo ha completo=false, di' che quella parte della risposta non c'è."""


@dataclass(frozen=True)
class Specialista:
    nome: Literal["dati_conto", "policy"]
    system_prompt: str
    tool_names: tuple[str, ...]


SPECIALISTI = {
    "dati_conto": Specialista(
        "dati_conto",
        PROMPT_DATI_CONTO,
        ("find_customer_accounts", "get_account_balance", "list_recent_movements"),
    ),
    "policy": Specialista("policy", PROMPT_POLICY, ("search_documents",)),
}
INSTRADAMENTO = {
    "dati_conto": ["dati_conto"],
    "policy": ["policy"],
    "entrambi": ["dati_conto", "policy"],
}


class Contributo(BaseModel):
    """Quello che uno specialista consegna: dati che il codice sa, non impressioni."""

    specialista: Literal["dati_conto", "policy"]
    completo: bool  # ha risposto il modello, non il tetto: lo dice stopped_by
    stopped_by: str
    contenuto: str
    tool_calls: list[str]


@dataclass
class SupervisorResult:
    risposta: str
    instradamento: str
    contributi: list[Contributo]
    cost_eur: Decimal


async def run_supervisor(user: UserContext, domanda: str, deps: Deps) -> SupervisorResult:
    """Sceglie chi deve rispondere, li fa lavorare ognuno con i suoi tool, e ricompone."""
    costo = Decimal("0")

    # 1. il triage: una chiamata, un insieme chiuso di risposte. Non è un agente, è una
    #    classificazione; e se risponde altro, si prende la strada più prudente
    triage = await deps.openai.chat.completions.create(
        model=settings.triage_model,
        max_tokens=5,
        messages=[
            {"role": "system", "content": PROMPT_TRIAGE},
            {"role": "user", "content": domanda},
        ],
    )
    costo += costo_chiamata(triage.usage, settings.triage_model)
    scelta = (triage.choices[0].message.content or "").strip().lower()
    if scelta not in INSTRADAMENTO:
        scelta = "entrambi"  # al più due specialisti: il tetto è nella tabella

    # 2. ognuno col SUO prompt e i SUOI tool, già costruiti per l'utente
    tutti = build_tools_for(user, deps)
    contributi: list[Contributo] = []
    for nome in INSTRADAMENTO[scelta]:
        s = SPECIALISTI[nome]
        run: RunResult = await run_agent(
            messaggi=[
                {"role": "system", "content": s.system_prompt},
                {"role": "user", "content": domanda},
            ],
            tools=[t for t in tutti if t.name in s.tool_names],
            client=deps.openai,
            model=deps.model,
        )
        costo += run.cost_eur
        contributi.append(
            Contributo(
                specialista=s.nome,
                completo=run.stopped_by == "model",
                stopped_by=run.stopped_by,
                contenuto=run.reply,
                tool_calls=run.tool_calls,
            )
        )

    # 3. la sintesi riceve dati, non prosa incollata: sa chi ha detto cosa e se ha finito
    materiale = json.dumps(
        {"domanda": domanda, "contributi": [c.model_dump() for c in contributi]}, ensure_ascii=False
    )
    sintesi = await deps.openai.chat.completions.create(
        model=deps.model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": PROMPT_SINTESI},
            {"role": "user", "content": materiale},
        ],
    )
    costo += costo_chiamata(sintesi.usage, deps.model)
    return SupervisorResult(
        risposta=sintesi.choices[0].message.content or "",
        instradamento=scelta,
        contributi=contributi,
        cost_eur=costo,
    )
