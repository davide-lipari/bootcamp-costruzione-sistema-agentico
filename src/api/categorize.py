# src/api/categorize.py — la categorizzazione, dal modello e validata
from typing import Annotated

import instructor
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.repos import ChatRepository
from src.db.session import get_db
from src.llm.factory import get_instructor
from src.llm.prompt import load_prompt
from src.observability.cost_tracker import CostTracker
from src.observability.ledger import CostLedger
from src.services.categorize_service import CategorizeService
from src.types.categorize import CategorizeRequest, CategorizeResponse

router = APIRouter(prefix="/api/ai", tags=["Categorize"])

CATEGORIZE_PROMPT = load_prompt("categorize_system_v1")


def get_categorize_service(
    client: Annotated[instructor.AsyncInstructor, Depends(get_instructor)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CategorizeService:
    # il tetto della chat vale anche qui: ogni chiamata al modello spende, non solo la chat
    costi = CostTracker(ChatRepository(session), settings.daily_budget_eur)
    return CategorizeService(
        client, settings.default_model, CATEGORIZE_PROMPT, costi, CostLedger(session)
    )


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize(
    req: CategorizeRequest, service: Annotated[CategorizeService, Depends(get_categorize_service)]
) -> CategorizeResponse:
    return await service.categorize(req)
