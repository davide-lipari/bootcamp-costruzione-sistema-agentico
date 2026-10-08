# src/agents/loop.py — Giorno 8: il ciclo che sa fermarsi prima di un'azione da approvare
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import get_close_matches
from typing import cast
from uuid import uuid4

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError

from src.agents.registry import Tool, impronta
from src.auth.deps import UserContext
from src.db.runs import RunRepository

logger = logging.getLogger(__name__)

MAX_STEPS = 6  # il tetto dell'endpoint: la commessa ti chiede di misurarlo
BUDGET_PER_RUN = Decimal("0.05")  # euro
PREZZI_PER_1K = {  # euro per mille token (ingresso, uscita), come al Giorno 4
    "gpt-4o-mini": (Decimal("0.00014"), Decimal("0.00056")),
    "gpt-4o": (Decimal("0.0023"), Decimal("0.0091")),
    "openai/gpt-oss-120b": (Decimal("0.00009"), Decimal("0.00046")),  # Groq
}
SENZA_APPROVAZIONE = (
    "Questa azione richiede l'approvazione di un responsabile, e da qui non si può chiedere: "
    "non è stata eseguita. Non ritentarla: riferisci all'utente che va richiesta dal flusso "
    "con approvazione."
)


@dataclass
class RunResult:
    run_id: str
    reply: str = ""
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    stopped_by: str = ""  # "model" | "max_steps" | "budget" | "awaiting_approval"
    cost_eur: Decimal = Decimal("0")


async def run_agent(
    messaggi: list[ChatCompletionMessageParam],
    tools: list[Tool],
    client: AsyncOpenAI,
    model: str,
    max_steps: int = MAX_STEPS,
    *,
    runs: RunRepository | None = None,
    user: UserContext | None = None,
    run: RunResult | None = None,
) -> RunResult:
    """Chiama il modello finché chiede tool, poi ritorna la sua risposta.

    Dal Giorno 8 sa anche fermarsi: se un tool del passo va approvato, salva lo stato in
    `runs` per conto di `user` ed esce con stopped_by="awaiting_approval", prima che
    qualunque tool di quel passo sia partito. Senza `runs` e `user` quel tool non parte.
    `run` è il run da continuare quando riprende: stesso id, stessi passi, stesso costo.
    """
    if model not in PREZZI_PER_1K:  # meglio fermarsi qui che scoprirlo dopo aver pagato
        raise ValueError(f"Nessun prezzo per {model}: il budget non sarebbe calcolabile.")
    messaggi = list(messaggi)  # la lista di chi chiama non si tocca
    by_name = {t.name: t for t in tools}  # il dispatch vede SOLO i tool offerti
    schemi = [t.to_openai_schema() for t in tools]
    run = run or RunResult(run_id=str(uuid4()))  # alla ripresa il run arriva da fuori

    for _ in range(max_steps - run.steps):  # il tetto conta anche i passi di prima
        run.steps += 1
        risposta = await client.chat.completions.create(
            model=model,
            messages=messaggi,
            max_tokens=500,
            tools=schemi or omit,  # l'API rifiuta una lista vuota
        )
        run.cost_eur += costo_chiamata(risposta.usage, model)
        choice = risposta.choices[0].message

        if not choice.tool_calls:  # uscita 1: ha risposto
            run.stopped_by = "model"
            run.reply = choice.content or ""
            return run
        if run.cost_eur > BUDGET_PER_RUN:  # uscita 3: il budget, prima di altri tool
            run.stopped_by = "budget"
            run.reply = (
                "Ho interrotto l'elaborazione perché la richiesta ha superato il "
                "budget previsto. Prova a formularla in modo più circoscritto."
            )
            return run

        # PRIMA l'assistant con le chiamate, poi i risultati: altrimenti l'API rifiuta
        assistente = choice.model_dump(exclude_none=True)
        messaggi.append(cast(ChatCompletionAssistantMessageParam, assistente))
        chiamate = [
            c for c in choice.tool_calls if isinstance(c, ChatCompletionMessageFunctionToolCall)
        ]
        in_attesa = [c for c in chiamate if richiede_approvazione(by_name, c)]

        if in_attesa and runs is not None and user is not None:
            # uscita 4: nessun tool di QUESTO passo è partito, nemmeno quelli in lettura.
            # Alla ripresa partono tutti, nell'ordine: il modello li aveva chiesti insieme
            descrizione = descrivi_attesa(by_name, in_attesa, user, run.run_id)
            await runs.sospendi(
                run_id=run.run_id,
                username=user.username,
                role=user.role,
                messages=[dict(m) for m in messaggi],
                pending_calls=[c.model_dump() for c in chiamate],
                description=descrizione,
                steps=run.steps,
                cost_eur=run.cost_eur,
                tool_calls=run.tool_calls,
            )
            run.stopped_by = "awaiting_approval"
            run.reply = descrizione
            return run

        for call in chiamate:
            if call in in_attesa:  # nessuno a cui chiedere: non parte
                logger.warning("tool_senza_approvazione", extra={"tool": call.function.name})
                esito = SENZA_APPROVAZIONE
            else:
                esito = await esegui_e_registra(by_name, call, run)
            messaggi.append({"role": "tool", "tool_call_id": call.id, "content": esito})

    run.stopped_by = "max_steps"  # uscita 2: tetto raggiunto, NON è una risposta
    run.reply = (
        "Non ho completato la richiesta entro i passi previsti: la risposta non c'è. "
        "Riformula la domanda in modo più specifico, o dividila in domande più semplici."
    )
    return run


def richiede_approvazione(
    by_name: dict[str, Tool], call: ChatCompletionMessageFunctionToolCall
) -> bool:
    """Decide sugli argomenti VALIDATI, e nel dubbio chiede."""
    tool = by_name.get(call.function.name)
    if tool is None or not tool.scrive:
        return False  # un nome inventato o un tool in lettura
    try:
        args = tool.args_model.model_validate_json(call.function.arguments)
    except ValidationError:
        return False  # non partirà: _execute rimanda l'errore
    return tool.serve_approvazione is None or tool.serve_approvazione(args)


def descrivi_attesa(
    by_name: dict[str, Tool],
    calls: list[ChatCompletionMessageFunctionToolCall],
    user: UserContext,
    run_id: str,
) -> str:
    """Cosa sta per succedere, in una forma che una persona può approvare o respingere."""
    righe = []
    for call in calls:
        args = by_name[call.function.name].args_model.model_validate_json(call.function.arguments)
        campi = ", ".join(
            f"{k}={v}" for k, v in args.model_dump(mode="json").items() if v is not None
        )
        righe.append(f"- {call.function.name}: {campi}")
    return (
        f"Serve l'approvazione di un responsabile, richiesta da {user.username}, per:\n"
        + "\n".join(righe)
        + f"\nPratica in attesa: {run_id}. Nulla è stato ancora eseguito."
    )


async def esegui_e_registra(
    by_name: dict[str, Tool], call: ChatCompletionMessageFunctionToolCall, run: RunResult
) -> str:
    """Esegue un tool e ne lascia la riga di traccia. La usa anche la ripresa."""
    run.tool_calls.append(call.function.name)
    tool = by_name.get(call.function.name)
    inizio = time.perf_counter()
    esito = await _execute(by_name, call)
    tipo = tipo_esito(tool, esito)
    logger.info(
        "agent_step",
        extra={
            "run_id": run.run_id,
            "step": run.steps,
            "tool": call.function.name,
            "tool_args": argomenti_per_il_log(tool, call.function.arguments),
            "result_kind": tipo,
            # di un tool che scrive resta la ricevuta, con il numero di pratica: un audit risale da
            # lì al run. Solo se è riuscito: il messaggio di un errore riporta gli argomenti
            "receipt": esito if tool and tool.scrive and tipo == "ok" else None,
            "result_len": len(esito),
            "duration_ms": int((time.perf_counter() - inizio) * 1000),
        },
    )
    return esito


def argomenti_per_il_log(tool: Tool | None, argomenti: str) -> object:
    """Quali campi, quanto lunghi e se si ripetono: non il loro valore."""
    if tool is None or tool.scrive:
        # quelli di un tool che scrive stanno nella sua tabella, con i suoi permessi
        return "<omessi>"
    try:
        valori = json.loads(argomenti)
    except ValueError:
        return "<non leggibili>"
    if not isinstance(valori, dict):
        return "<non leggibili>"
    return {
        campo: v if isinstance(v, int | float) else f"{len(str(v))} car. #{impronta(str(v))}"
        for campo, v in valori.items()
    }


def tipo_esito(tool: Tool | None, esito: str) -> str:
    """Il tipo del risultato al posto del suo testo: basta a separare tool e dato sbagliati."""
    if tool is None or esito.startswith("ERRORE"):
        return "errore"
    return "vuoto" if esito in tool.senza_dati else "ok"


def costo_chiamata(usage: CompletionUsage | None, model: str) -> Decimal:
    """Il `_costo` del Giorno 7, ora pubblico: lo usa anche il supervisor."""
    if usage is None:
        return Decimal("0")
    ingresso, uscita = PREZZI_PER_1K[model]
    return (usage.prompt_tokens * ingresso + usage.completion_tokens * uscita) / 1000


async def _execute(by_name: dict[str, Tool], call: ChatCompletionMessageFunctionToolCall) -> str:
    tool = by_name.get(call.function.name)
    if tool is None:
        # Il modello ha inventato un nome. Glielo diciamo, non solleviamo — e lo
        # registriamo: un nome inventato con insistenza è un segnale da guardare
        logger.warning("tool_inesistente", extra={"tool": call.function.name[:80]})
        vicini = get_close_matches(call.function.name, list(by_name), n=1)
        suggerimento = f"Forse intendevi '{vicini[0]}'. " if vicini else ""
        elenco = f"Tool disponibili: {', '.join(by_name)}." if len(by_name) <= 10 else ""
        return f"ERRORE: il tool '{call.function.name}' non esiste. {suggerimento}{elenco}".strip()
    try:
        args = tool.args_model.model_validate_json(call.function.arguments)
    except ValidationError as e:
        return f"ERRORE di validazione degli argomenti: {e.errors(include_url=False)}"
    try:
        return await tool.run(args)
    except Exception as e:
        # un tool che fallisce (database lento, vincolo violato) è un'osservazione, non un 500;
        # nel log il tipo dell'errore e non il traceback, che porta con sé l'SQL e i parametri
        logger.error("tool_fallito", extra={"tool": tool.name, "errore": type(e).__name__})
        return (
            f"ERRORE: il tool '{tool.name}' non ha potuto completare l'operazione. "
            "Non riprovare la stessa chiamata: riferisci all'utente che il dato ora non è "
            "disponibile."
        )
