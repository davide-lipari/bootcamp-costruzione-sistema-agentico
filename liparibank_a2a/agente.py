# liparibank_a2a/agente.py — l'istruttoria come agente A2A: chi chiama chiede COSA, lei decide COME
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import jwt
from a2a.auth.user import UnauthenticatedUser, User
from a2a.helpers import get_message_text, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    DefaultServerCallContextBuilder,
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    Part,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from src.agents.deps import crea_deps
from src.auth.deps import UserContext
from src.auth.mcp_tokens import EMITTENTE, RISORSA_A2A
from src.auth.tokens import ALGORITHM
from src.config import settings
from src.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)
RPC = "/a2a"
SCOPE = "istruttoria:chiedi"
IBAN = re.compile(r"\bIT\d{2}[A-Z]\d{10,23}\b")
NON_RIUSCITA = "L'istruttoria non è riuscita. Riprova più tardi o chiedi all'assistenza."

# chi la pubblica decide come rispondere: il grafo dell'istruttoria in servizio, un finto nei test
Rispondi = Callable[[UserContext, str], Awaitable[str]]


async def istruttoria_del_grafo(user: UserContext, testo: str) -> str:
    """In servizio: il grafo dell'assistente, costruito per l'operatore del token. Il triage
    sceglie la riga dell'istruttoria, che lavora sul portafoglio di chi chiama. Il modello che
    ragiona si passa come `llm_ragiona`, come nel giorno prima; senza, lavora il modello del
    repository."""
    from src.agents.supervisor import MODELLO_TRIAGE
    from src.lc.modelli import modello_chat
    from src.lg.grafo import costruisci_grafo, stato_iniziale

    async with AsyncSessionLocal() as db:
        grafo = costruisci_grafo(
            user,
            crea_deps(db),
            modello_chat(settings.default_model),
            modello_chat(MODELLO_TRIAGE),
            None,
        )
        out = await grafo.ainvoke(stato_iniziale(testo, user))
    return str(out["risposta"])


def carta(base_url: str = RISORSA_A2A) -> AgentCard:
    """La carta 1.0: dove si raggiunge, cosa sa fare, come ci si autentica. La dichiarazione non
    applica niente: l'autenticazione la fa il middleware di questa app."""
    return AgentCard(
        name="istruttoria-lipari",
        description=(
            "Analizza i movimenti di un conto del portafoglio di chi chiama e dice se ci sono "
            "elementi da segnalare, con date, importi e la policy che li riguarda."
        ),
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=f"{base_url}{RPC}", protocol_binding="JSONRPC", protocol_version="1.0"
            )
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="istruttoria",
                name="Istruttoria sui movimenti di un conto",
                description=(
                    "Dato l'IBAN di un conto e una domanda, legge movimenti e policy e dice "
                    "se c'è qualcosa da segnalare. Non apre segnalazioni: le propone."
                ),
                tags=["compliance", "aml", "movimenti"],
                examples=["Ci sono bonifici da segnalare sul conto IT60X0542811101000000123?"],
            )
        ],
        security_schemes={
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="Bearer", bearer_format="JWT"
                )
            )
        },
        # gli schemi dicono quali esistono, i requisiti quali servono: senza, la carta non chiede
        # niente, e un client che la legge non manda il token
        security_requirements=[SecurityRequirement(schemes={"bearer": StringList()})],
    )


class Operatore(User):
    """L'operatore del token, come l'SDK lo vuole. Viene dal middleware, mai dal messaggio."""

    def __init__(self, utente: UserContext) -> None:
        self.utente = utente

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.utente.username


class DalToken(DefaultServerCallContextBuilder):
    """Il contesto di sempre (gli header servono all'SDK: la versione del protocollo sta lì),
    con l'utente preso da dove l'ha messo il middleware, e da nessun altro posto."""

    def build_user(self, request: Request) -> User:
        operatore = getattr(request.state, "operatore", None)
        return operatore if isinstance(operatore, Operatore) else UnauthenticatedUser()


class Istruttoria(AgentExecutor):
    def __init__(self, rispondi: Rispondi) -> None:
        self.rispondi = rispondi

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        operatore = context.call_context.user if context.call_context else None
        if not isinstance(operatore, Operatore):  # il middleware non l'avrebbe fatto passare
            raise PermissionError("chiamata senza operatore")
        if context.message is None:
            raise ValueError("richiesta senza messaggio")
        attuale = context.current_task
        if attuale is None:
            attuale = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(attuale)
            up = TaskUpdater(event_queue, attuale.id, attuale.context_id)
            await up.start_work()
        else:
            up = TaskUpdater(event_queue, attuale.id, attuale.context_id)
        # la richiesta è tutto quello che l'operatore ha scritto sul task, una volta e in ordine
        detti = [m for m in attuale.history if m.role == Role.ROLE_USER]
        if context.message.message_id not in {m.message_id for m in detti}:
            detti.append(context.message)
        richiesta = " ".join(get_message_text(m) for m in detti)
        if not IBAN.search(richiesta):
            # il dato che manca si chiede, non si indovina: il task resta aperto e aspetta
            domanda = up.new_agent_message([Part(text="Su quale conto? Serve l'IBAN.")])
            await up.requires_input(domanda)
            return
        try:
            esito = await self.rispondi(operatore.utente, richiesta)
        except Exception as e:
            # al chiamante una frase che non dice niente di noi; nel log il tipo dell'errore, non
            # il traceback, che porterebbe con sé l'SQL e i parametri, cioè dati dei clienti
            logger.error(
                "istruttoria_fallita",
                extra={"username": operatore.user_name, "errore": type(e).__name__},
            )
            await up.failed(up.new_agent_message([Part(text=NON_RIUSCITA)]))
            return
        await up.add_artifact([Part(text=esito)], name="istruttoria")
        await up.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id is None or context.context_id is None:
            raise ValueError("annullamento senza task")
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def _operatore_dal_token(intestazione: str, risorsa: str) -> UserContext | None:
    if not intestazione.startswith("Bearer "):
        return None
    try:
        dati: dict[str, Any] = jwt.decode(
            intestazione.removeprefix("Bearer "),
            settings.jwt_secret,
            algorithms=[ALGORITHM],
            audience=risorsa,  # un token per il server MCP qui non entra
            issuer=EMITTENTE,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},  # senza scadenza non entra
        )
    except jwt.InvalidTokenError:
        return None
    if SCOPE not in str(dati.get("scope", "")).split():
        return None
    return UserContext(username=str(dati["sub"]), role=str(dati.get("role", "public")))


def crea_app(
    rispondi: Rispondi, base_url: str = RISORSA_A2A, store: TaskStore | None = None
) -> FastAPI:
    """La carta è pubblica, perché la scoperta deve funzionare; il punto RPC no."""
    c = carta(base_url)
    handler = DefaultRequestHandler(
        agent_executor=Istruttoria(rispondi),
        task_store=store or InMemoryTaskStore(),  # in servizio, DatabaseTaskStore: un task dura
        agent_card=c,
    )
    app = FastAPI()

    @app.middleware("http")
    async def autentica(request: Request, avanti: RequestResponseEndpoint) -> Response:
        if request.url.path == RPC:
            utente = _operatore_dal_token(request.headers.get("authorization", ""), base_url)
            if utente is None:
                return JSONResponse(
                    {"error": "token non valido per questo agente"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.operatore = Operatore(utente)
        return await avanti(request)

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(c),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=RPC, context_builder=DalToken()),
    )
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(crea_app(istruttoria_del_grafo), host="127.0.0.1", port=8300)
