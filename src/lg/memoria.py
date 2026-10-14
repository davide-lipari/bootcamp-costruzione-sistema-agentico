# src/lg/memoria.py — cosa l'assistente ricorda di un operatore fra una conversazione e l'altra
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel, ConfigDict

from src.auth.deps import UserContext
from src.config import settings

GIORNI = 90  # la conservazione è una decisione dichiarata: dopo, la preferenza scade
CHIAVE = "operatore"


class Preferenze(BaseModel):
    """Cosa si ricorda: uno schema chiuso. Un cliente, un IBAN, un saldo non hanno dove entrare."""

    model_config = ConfigDict(extra="forbid")

    formato: Literal["breve", "dettagliato"] = "breve"


def namespace(user: UserContext) -> tuple[str, str]:
    """La chiave dell'isolamento, costruita dal codice con l'utente autenticato. Sempre intera:
    una ricerca sul solo prefisso restituirebbe le preferenze di tutti."""
    return ("preferenze", user.username)


async def leggi(store: BaseStore, user: UserContext) -> Preferenze:
    voce = await store.aget(namespace(user), CHIAVE)
    return Preferenze.model_validate(voce.value) if voce else Preferenze()


async def salva(store: BaseStore, user: UserContext, preferenze: Preferenze) -> None:
    # la scadenza non si passa qui: la dichiara lo store, una volta, con il suo TTL predefinito
    await store.aput(namespace(user), CHIAVE, preferenze.model_dump())


async def dimentica(store: BaseStore, user: UserContext) -> None:
    """Su richiesta dell'operatore: la preferenza sparisce subito, non alla scadenza."""
    await store.adelete(namespace(user), CHIAVE)


@asynccontextmanager
async def store_postgres(
    scadenza_minuti: float = GIORNI * 24 * 60,
) -> AsyncIterator[AsyncPostgresStore]:
    """Come il checkpointer: si apre una volta, nel ciclo di vita del servizio. La scadenza si
    cambia solo nei test, per vederla scattare in pochi secondi."""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    ttl = {
        "default_ttl": scadenza_minuti,
        "refresh_on_read": False,  # leggere una preferenza non la allunga
        "omit_expired": True,  # scaduta non si legge più, anche prima che passi lo spazzino
    }
    async with AsyncPostgresStore.from_conn_string(dsn, ttl=ttl) as store:  # type: ignore[arg-type]
        await store.setup()
        await store.start_ttl_sweeper()  # chi toglie dal disco le voci scadute
        try:
            yield store
        finally:
            await store.stop_ttl_sweeper()
