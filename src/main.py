# src/main.py — Giorno 2: la sonda di ieri, gli handler degli errori, i middleware, i router
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import accounts, admin, advice, agent, auth, categorize, chat, glossary, movements
from src.config import settings
from src.exceptions import AppError, RateLimitError
from src.observability.json_log import configura_log, request_id
from src.types.error import ErrorResponse

configura_log()  # Giorno 9: le righe di log diventano JSON, con il request_id
logger = logging.getLogger(__name__)
ID_VALIDO = re.compile(r"[A-Za-z0-9-]{8,64}")


class Credentials(BaseModel):
    """Quali chiavi ci sono. Un sì o un no: mai il valore, e mai un pezzo del valore."""

    openai: bool
    anthropic: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    app_name: str
    version: str
    environment: str
    credentials: Credentials


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="LipariBank AI Assistant — Bootcamp Python AI Powered, Lipari Consulting",
)


def _busta(
    req: Request, status: int, codice: str, messaggio: str, dettagli: list[str] | None = None
) -> dict[str, object]:
    """La forma unica di ogni errore: la stessa per il 404, il 422, il 429 e il 500."""
    return ErrorResponse(
        timestamp=datetime.now(UTC),
        status=status,
        error=codice,
        message=messaggio,
        path=req.url.path,
        details=dettagli,
    ).model_dump(mode="json")


@app.exception_handler(AppError)
async def handle_app_error(req: Request, exc: AppError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if isinstance(exc, RateLimitError) else None
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content=_busta(req, exc.status_code, exc.code, exc.message),
    )


CODICI_HTTP = {
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
}


@app.exception_handler(StarletteHTTPException)
async def handle_http(req: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Gli errori del framework: una rotta che non esiste, un metodo sbagliato, l'HTTPException
    # di una dipendenza. Senza questo handler escono come {"detail": ...}, una forma in più.
    # Gli header di chi l'ha sollevata restano: il 401 deve portare WWW-Authenticate
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content=_busta(
            req, exc.status_code, CODICI_HTTP.get(exc.status_code, "HTTP_ERROR"), str(exc.detail)
        ),
    )


def _campo(loc: tuple[int | str, ...]) -> str:
    """Da ("body", "amount") ad "amount": il campo come lo ha scritto il chiamante."""
    return ".".join(str(p) for p in loc[1:]) or str(loc[0])


@app.exception_handler(RequestValidationError)
async def handle_validation(req: Request, exc: RequestValidationError) -> JSONResponse:
    # Un dettaglio per campo, «dove: cosa». Il valore ricevuto (e["input"]) non si rimanda
    # indietro: può essere lungo un file intero, o contenere quello che non doveva uscire.
    # Un corpo che non è JSON non ha un campo: il suo loc è ("body", 0), e si dice «corpo».
    dettagli = [
        f"{'corpo' if e['type'] == 'json_invalid' else _campo(e['loc'])}: {e['msg']}"
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=_busta(req, 422, "VALIDATION_ERROR", "La richiesta non è valida", dettagli),
    )


@app.exception_handler(Exception)
async def handle_unexpected(req: Request, exc: Exception) -> JSONResponse:
    logger.exception("errore_non_gestito", extra={"path": req.url.path})  # il dettaglio nel log
    return JSONResponse(
        status_code=500,  # e fuori, niente
        # il middleware non arriva a scriverlo, perché call_next ha sollevato: lo mette l'handler
        headers={"X-Request-Id": getattr(req.state, "request_id", "-")},
        content=_busta(req, 500, "INTERNAL_ERROR", "Errore inatteso"),
    )


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Se il chiamante ne manda uno, lo stesso id attraversa i due sistemi; se non lo manda,
    # o manda qualcosa che non ha la forma di un id, se ne genera uno nuovo.
    ricevuto = request.headers.get("X-Request-Id", "")
    rid = ricevuto if ID_VALIDO.fullmatch(ricevuto) else str(uuid.uuid4())
    request_id.set(rid)  # Giorno 9: da qui ogni riga di log di questa richiesta lo porta
    request.state.request_id = rid  # lo legge l'handler del 500
    inizio = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    response.headers["X-Process-Time"] = f"{time.perf_counter() - inizio:.4f}"
    return response


app.add_middleware(  # registrato per ultimo: è il più esterno
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="UP",
        timestamp=datetime.now(UTC).isoformat(),
        app_name=settings.app_name,
        version="1.0.0",
        environment=settings.environment,
        credentials=Credentials(
            openai=bool(settings.openai_api_key),
            anthropic=bool(settings.anthropic_api_key),
        ),
    )


app.include_router(chat.router)
app.include_router(glossary.router)
app.include_router(categorize.router)
app.include_router(movements.router)
app.include_router(accounts.router)
app.include_router(advice.router)
app.include_router(auth.router)
app.include_router(agent.router)
app.include_router(admin.router)

app.state.limiter = advice.limiter  # slowapi lo cerca qui


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(req: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "60"},
        content=_busta(req, 429, "RATE_LIMIT", "Troppe richieste: riprova fra un minuto"),
    )
