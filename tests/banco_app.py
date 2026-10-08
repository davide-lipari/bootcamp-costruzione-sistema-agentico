# tests/banco_app.py — l'app della giornata su un database vero e un modello finto
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock

from fastapi import FastAPI, Header
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.agents.deps import Deps, crea_deps, get_deps
from src.api.agent import router
from src.auth.deps import UserContext, get_current_user
from src.db.models import Account, Customer
from src.db.session import Base
from tests.conftest import (
    CLIENTE_ALTRUI,
    CLIENTE_DI_MARCO,
    CONTO_ALTRUI,
    CONTO_DI_MARCO,
    TABELLE_DEL_GIORNO,
)
from tests.finti import embedder_finto

UTENTI = {"mbianchi": "operator", "grossi": "compliance_lead", "lverdi": "risk_lead"}


async def prepara_db(percorso: Path) -> async_sessionmaker[AsyncSession]:
    """Un database su file: sopravvive al processo che l'ha creato, come quello vero."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{percorso}")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Base.metadata.create_all(
                c, tables=[Base.metadata.tables[nome] for nome in TABELLE_DEL_GIORNO]
            )
        )
    fabbrica = async_sessionmaker(engine, expire_on_commit=False)
    async with fabbrica() as s:
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
            ]
        )
        await s.commit()
    return fabbrica


def apri_db(percorso: Path) -> async_sessionmaker[AsyncSession]:
    """Lo stesso database, riaperto: è quello che fa il processo dopo un riavvio."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{percorso}")
    return async_sessionmaker(engine, expire_on_commit=False)


def crea_app(fabbrica: async_sessionmaker[AsyncSession], modello: AsyncMock) -> FastAPI:
    """L'app del corso: stessi router, stessi servizi. Cambiano solo l'utente e il modello."""
    app = FastAPI()
    app.include_router(router)

    async def utente(x_utente: Annotated[str, Header()]) -> UserContext:
        return UserContext(username=x_utente, role=UTENTI[x_utente])

    async def servizi() -> AsyncIterator[Deps]:
        async with fabbrica() as s:  # una sessione per richiesta, come get_db
            yield crea_deps(s, embedder=embedder_finto(), openai=modello)

    app.dependency_overrides[get_current_user] = utente
    app.dependency_overrides[get_deps] = servizi
    return app
