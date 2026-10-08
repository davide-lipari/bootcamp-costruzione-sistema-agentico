# tests/conftest.py
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.agents.deps import Deps
from src.auth.deps import UserContext
from src.db.models import Account, Customer, Movement
from src.db.repos import AccountRepository, MovementRepository
from src.db.runs import RunRepository
from src.db.session import Base
from src.services.alerts import AlertService
from src.services.retrieval_service import RetrievalService
from tests.finti import embedder_finto

CLIENTE_DI_MARCO = "C-10234"  # nel portafoglio di mbianchi
CLIENTE_ALTRUI = "C-20417"  # nel portafoglio di un collega, pgalli
CONTO_DI_MARCO = "IT60X0542811101000000123"
CONTO_ALTRUI = "IT60X0542811101000000789"
TABELLE_DEL_GIORNO = [
    "customers",
    "accounts",
    "movements",
    "compliance_alerts",
    "agent_runs",
    "llm_calls",  # dal Giorno 9: gli endpoint dell'agente registrano il costo
]


def risposta(
    *,
    tool: str | None = None,
    argomenti: str = "{}",
    testo: str | None = None,
    prompt_tokens: int = 10,
) -> ChatCompletion:
    """Una risposta del modello: finta, ma con la forma vera dell'SDK."""
    messaggio: dict[str, Any] = {"role": "assistant", "content": testo}
    if tool:
        messaggio["tool_calls"] = [
            {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": argomenti}}
        ]
    return ChatCompletion.model_validate(
        {
            "id": "finto",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": messaggio,
                    "finish_reason": "tool_calls" if tool else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 5,
                "total_tokens": prompt_tokens + 5,
            },
        }
    )


@pytest.fixture
def marco() -> UserContext:
    return UserContext(username="mbianchi", role="operator")


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        # solo le tabelle di oggi: quelle dei giorni precedenti usano tipi di PostgreSQL
        await conn.run_sync(
            lambda c: Base.metadata.create_all(
                c, tables=[Base.metadata.tables[nome] for nome in TABELLE_DEL_GIORNO]
            )
        )
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        s.add_all(
            [
                Customer(id=CLIENTE_DI_MARCO, full_name="Paolo Ferri", operator="mbianchi"),
                Customer(id=CLIENTE_ALTRUI, full_name="Anna Greco", operator="pgalli"),
                Account(
                    id=CONTO_DI_MARCO,
                    customer_id=CLIENTE_DI_MARCO,
                    label="principale",
                    balance=Decimal("48200.00"),
                ),
                Account(
                    id=CONTO_ALTRUI,
                    customer_id=CLIENTE_ALTRUI,
                    label="principale",
                    balance=Decimal("15300.00"),
                ),
                Movement(
                    account_id=CONTO_DI_MARCO,
                    booking_date=date(2026, 9, 20),
                    description="Bonifico estero",
                    amount=Decimal("-9800.00"),
                ),
            ]
        )
        await s.commit()
        yield s
    await engine.dispose()


@pytest.fixture
def deps_reali(session: AsyncSession) -> Deps:
    return Deps(
        accounts=AccountRepository(session),
        movements=MovementRepository(session),
        alerts=AlertService(session),
        runs=RunRepository(session),
        retrieval=RetrievalService(session),
        embedder=embedder_finto(),
        openai=AsyncMock(),
        model="gpt-4o-mini",
    )


@pytest.fixture
def deps_spia() -> MagicMock:
    deps = MagicMock()
    deps.accounts.of_user = AsyncMock(return_value=None)
    deps.accounts.of_customer = AsyncMock(return_value=[])
    deps.movements.recent = AsyncMock(return_value=[])
    deps.alerts.apri = AsyncMock()
    return deps
