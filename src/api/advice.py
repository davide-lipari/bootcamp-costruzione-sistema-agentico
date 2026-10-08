# src/api/advice.py — l'advisor, da oggi autenticato, filtrato per ruolo e limitato per utente
from decimal import Decimal
from functools import cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.deps import UserContext, get_current_user, require_role
from src.auth.tokens import decode_token
from src.config import settings
from src.db.repos import ChatRepository
from src.db.session import get_db
from src.llm.client import LLMProvider
from src.llm.embedding_client import EmbeddingClient
from src.llm.factory import get_embedder, get_llm_provider
from src.llm.prompt import load_prompt
from src.llm.rewriter import QueryRewriter
from src.observability.cost_tracker import CostTracker
from src.observability.ledger import CostLedger
from src.services.ingest_service import IngestService
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService
from src.types.advice import AdviceRequest, AdviceResponse, IngestRequest, IngestResponse

router = APIRouter(prefix="/api/ai", tags=["Advice"])
PROMPT = load_prompt("advice_system_v1")


def per_utente(request: Request) -> str:
    """La chiave del limite: l'utente del token, non l'indirizzo IP."""
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    try:
        return str(decode_token(token)["sub"])
    except HTTPException:
        return get_remote_address(request)  # senza un token valido arriverà comunque un 401


limiter = Limiter(key_func=per_utente)


@cache  # uno per processo: la sua cache delle riscritture deve sopravvivere alla richiesta
def get_rewriter() -> QueryRewriter:
    return QueryRewriter(get_llm_provider(), load_prompt("rewrite_system_v1"))


def get_rag_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    embedder: Annotated[EmbeddingClient, Depends(get_embedder)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    rewriter: Annotated[QueryRewriter, Depends(get_rewriter)],
) -> RAGService:
    return RAGService(RetrievalService(session), embedder, llm, rewriter, PROMPT)


def get_ingest_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    embedder: Annotated[EmbeddingClient, Depends(get_embedder)],
) -> IngestService:
    return IngestService(session, embedder)


@router.post("/advice", response_model=AdviceResponse)
@limiter.limit("15/minute")
async def advice(
    request: Request,  # slowapi la pretende: è da lì che legge la chiave
    req: AdviceRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[RAGService, Depends(get_rag_service)],
    session: Annotated[AsyncSession, Depends(get_db)],  # la stessa sessione del servizio
) -> AdviceResponse:
    # il tetto della chat vale anche qui: ogni chiamata al modello spende, non solo la chat
    await CostTracker(ChatRepository(session), settings.daily_budget_eur).verifica()
    risposta = await service.advise(req.question, user)
    # Giorno 9: la generazione nel registro. La riscrittura qui non c'è: vedi il Code Blueprint
    CostLedger(session).aggiungi(
        endpoint="advice",
        username=user.username,
        model=settings.default_model,
        tokens=risposta.tokens_used,
        cost_eur=Decimal(str(risposta.cost_eur)),
    )
    await session.commit()
    return risposta


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    user: Annotated[UserContext, Depends(require_role("compliance_lead", "admin"))],
    service: Annotated[IngestService, Depends(get_ingest_service)],
) -> IngestResponse:
    """Scrive l'indice: da oggi, solo la compliance e gli admin."""
    quanti = await service.ingest(req.document_id, req.content, req.metadata, req.visibility)
    return IngestResponse(chunk_count=quanti, embedding_dim=1536)
