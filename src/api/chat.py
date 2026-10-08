# src/api/chat.py — la chat, con il modello, intera o in streaming
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.deps import UserContext, get_current_user
from src.config import settings
from src.db.session import get_db
from src.llm.client import LLMProvider
from src.llm.factory import get_llm_provider
from src.llm.prompt import load_prompt
from src.services.chat_service import ChatService
from src.types.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/ai", tags=["Chat"])

SYSTEM_PROMPT = load_prompt("chat_system_v1")


def get_chat_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> ChatService:
    return ChatService(session, llm, SYSTEM_PROMPT, settings.daily_budget_eur)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    return await service.chat(req, user_id=user.username)  # da oggi, l'utente del token


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    pezzi = await service.chat_stream(
        req, user_id=user.username
    )  # un 404 esce qui, prima dello stream

    async def eventi() -> AsyncIterator[str]:
        async for pezzo in pezzi:
            # ogni evento è JSON: un a capo dentro il testo non rompe il formato
            yield f"data: {json.dumps({'text': pezzo}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(eventi(), media_type="text/event-stream")
