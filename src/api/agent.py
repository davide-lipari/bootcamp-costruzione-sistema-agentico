# src/api/agent.py — Giorno 8: l'agente, e le due decisioni che può aspettare
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.agents.approval import riprendi
from src.agents.deps import Deps, get_deps
from src.agents.loop import RunResult, run_agent
from src.agents.prompts import AGENT_SYSTEM
from src.agents.supervisor import run_supervisor
from src.agents.tools import build_tools_for
from src.auth.deps import UserContext, get_current_user, require_role
from src.observability.ledger import CostLedger

router = APIRouter(prefix="/api/ai", tags=["agent"])

RUOLI_APPROVATORI = ("compliance_lead", "risk_lead")  # un ruolo, non una persona


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    run_id: str
    reply: str
    steps: int
    tool_calls: list[str]
    stopped_by: str  # "model" | "max_steps" | "budget" | "awaiting_approval"
    cost_eur: float


class SupervisorResponse(BaseModel):
    risposta: str
    instradamento: str  # "dati_conto" | "policy" | "entrambi"
    specialisti_completi: dict[str, bool]
    cost_eur: float


class RejectRequest(BaseModel):
    motivo: str = Field(min_length=10, max_length=500)  # il modello lo riferirà a Marco


class RunStatus(BaseModel):
    run_id: str
    status: str
    requested_by: str
    description: str
    decided_by: str | None
    decisions: list[dict[str, Any]]  # ogni firma del run, anche quelle delle pause di prima


def _risposta(run: RunResult) -> AgentResponse:
    return AgentResponse(
        run_id=run.run_id,
        reply=run.reply,
        steps=run.steps,
        tool_calls=run.tool_calls,
        stopped_by=run.stopped_by,
        cost_eur=float(run.cost_eur),
    )


@router.post("/agent", response_model=AgentResponse)
async def agent(
    payload: AgentRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> AgentResponse:
    run = await run_agent(
        messaggi=[
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": payload.message},
        ],
        tools=build_tools_for(user, deps),  # ← i tool nascono qui, per lui
        client=deps.openai,
        model=deps.model,
        runs=deps.runs,
        user=user,  # Giorno 8: dove fermarsi, e per chi
    )
    await _registra(deps, "agent", user.username, run.run_id, run.cost_eur)
    return _risposta(run)


@router.post("/supervisor", response_model=SupervisorResponse)
async def supervisor(
    payload: AgentRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> SupervisorResponse:
    """La stessa domanda, divisa fra specialisti. Per confrontarla con /agent, costo compreso."""
    esito = await run_supervisor(user, payload.message, deps)
    await _registra(deps, "supervisor", user.username, None, esito.cost_eur)
    return SupervisorResponse(
        risposta=esito.risposta,
        instradamento=esito.instradamento,
        specialisti_completi={c.specialista: c.completo for c in esito.contributi},
        cost_eur=float(esito.cost_eur),
    )


@router.get("/agent/{run_id}", response_model=RunStatus)
async def stato_run(
    run_id: str,
    user: Annotated[UserContext, Depends(get_current_user)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> RunStatus:
    """Cosa si sta approvando. La vedono chi ha chiesto e chi può decidere, nessun altro."""
    stato = await deps.runs.get(run_id)
    if stato is None or (stato.username != user.username and user.role not in RUOLI_APPROVATORI):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Esecuzione non trovata")
    return RunStatus(
        run_id=stato.id,
        status=stato.status,
        requested_by=stato.username,
        description=stato.description,
        decided_by=stato.decided_by,
        decisions=stato.decisions,
    )


@router.post("/agent/{run_id}/approve", response_model=AgentResponse)
async def approve(
    run_id: str,
    approvatore: Annotated[UserContext, Depends(require_role(*RUOLI_APPROVATORI))],
    deps: Annotated[Deps, Depends(get_deps)],
) -> AgentResponse:
    """Autorizza l'azione sospesa e fa riprendere il lavoro. 404, 403, 409: vedi _decidi."""
    return await _decidi(run_id, approvatore, approvato=True, motivo=None, deps=deps)


@router.post("/agent/{run_id}/reject", response_model=AgentResponse)
async def reject(
    run_id: str,
    payload: RejectRequest,
    approvatore: Annotated[UserContext, Depends(require_role(*RUOLI_APPROVATORI))],
    deps: Annotated[Deps, Depends(get_deps)],
) -> AgentResponse:
    """Respinge l'azione sospesa: il run riprende, e l'agente riferisce il rifiuto."""
    return await _decidi(run_id, approvatore, approvato=False, motivo=payload.motivo, deps=deps)


async def _registra(
    deps: Deps, endpoint: str, username: str, run_id: str | None, costo: Decimal
) -> None:
    """Giorno 9: il costo nel registro. La sessione è quella dei servizi dell'agente."""
    CostLedger(deps.runs.session).aggiungi(
        endpoint=endpoint, username=username, model=deps.model, cost_eur=costo, run_id=run_id
    )
    await deps.runs.session.commit()


async def _decidi(
    run_id: str, approvatore: UserContext, *, approvato: bool, motivo: str | None, deps: Deps
) -> AgentResponse:
    # 1. lo stato sta nel database, non in memoria: un riavvio non lo perde
    stato = await deps.runs.get(run_id)
    if stato is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Esecuzione non trovata")

    # 2. chi chiede non decide
    if stato.username == approvatore.username:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Chi ha richiesto l'azione non può deciderla"
        )

    # 3. la decisione si scrive una volta sola: l'UPDATE condizionato fa vincere un clic,
    #    e da qui la riga dice chi ha deciso e quando — prima che l'azione parta
    decisa = await deps.runs.decidi(
        run_id, da=approvatore.username, stato="running" if approvato else "rejected", motivo=motivo
    )
    if not decisa:
        raise HTTPException(status.HTTP_409_CONFLICT, "L'esecuzione non è più in attesa")

    # 4. si riprende, con i tool di chi aveva chiesto
    ripreso = await riprendi(
        stato, approvato=approvato, da=approvatore.username, motivo=motivo, deps=deps
    )

    # il costo di questa ripresa, non del run intero: quello prima era già nel registro
    await _registra(
        deps, "agent", stato.username, run_id, ripreso.cost_eur - Decimal(stato.cost_eur)
    )

    # 5. e si chiude, a meno che il run non si sia fermato di nuovo
    if ripreso.stopped_by != "awaiting_approval":
        await deps.runs.chiudi(run_id, "done" if approvato else "rejected")
    return _risposta(ripreso)
