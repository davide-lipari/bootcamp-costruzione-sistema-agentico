# src/agents/approval.py — riprendere un run sospeso, dopo che una persona ha deciso
from decimal import Decimal
from typing import cast

from openai.types.chat import ChatCompletionMessageFunctionToolCall, ChatCompletionMessageParam

from src.agents.deps import Deps
from src.agents.loop import RunResult, esegui_e_registra, richiede_approvazione, run_agent
from src.agents.tools import build_tools_for
from src.auth.deps import UserContext
from src.db.models import AgentRunState

RIFIUTO = (
    "Il responsabile {da} ha respinto questa azione, che non è stata eseguita. Motivo: {motivo}. "
    "Non ritentarla e non cercare un'altra strada per ottenere lo stesso risultato: riferisci "
    "all'utente che la richiesta è stata respinta, e perché."
)


async def riprendi(
    stato: AgentRunState, *, approvato: bool, da: str, motivo: str | None, deps: Deps
) -> RunResult:
    """Rimette in piedi la conversazione salvata e fa ripartire il ciclo da dove si era fermato."""
    # i tool si rifanno per CHI HA CHIESTO, mai per chi approva: sono closure, non dati
    richiedente = UserContext(username=stato.username, role=stato.role)
    tools = build_tools_for(richiedente, deps)
    by_name = {t.name: t for t in tools}
    run = RunResult(
        run_id=stato.id,
        steps=stato.steps,
        tool_calls=list(stato.tool_calls),
        cost_eur=Decimal(stato.cost_eur),
    )
    # lo stato è una lista di dizionari, e tale resta: il cast dice solo di che forma sono
    messaggi = [cast(ChatCompletionMessageParam, m) for m in stato.messages]

    # le chiamate del passo sospeso, tutte e nell'ordine: ognuna ha diritto alla sua risposta
    for dati in stato.pending_calls:
        call = ChatCompletionMessageFunctionToolCall.model_validate(dati)
        if approvato or not richiede_approvazione(by_name, call):
            esito = await esegui_e_registra(by_name, call, run)
        else:
            esito = RIFIUTO.format(da=da, motivo=motivo)
        messaggi.append({"role": "tool", "tool_call_id": call.id, "content": esito})

    # e il ciclo riparte: stesso run, stessi passi già contati, stesso costo già speso
    return await run_agent(
        messaggi=messaggi,
        tools=tools,
        client=deps.openai,
        model=deps.model,
        runs=deps.runs,
        user=richiedente,
        run=run,
    )
