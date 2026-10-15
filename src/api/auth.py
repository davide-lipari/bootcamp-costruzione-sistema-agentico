# src/api/auth.py — l'ingresso: credenziali in, token fuori
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.deps import UserContext, get_current_user
from src.auth.mcp_tokens import DURATA, autentica_client, emetti
from src.auth.passwords import hash_password, verify_password
from src.auth.tokens import create_access_token
from src.db.models import AppUser
from src.db.session import get_db

router = APIRouter(prefix="/api/auth", tags=["Auth"])

# un hash vero di una password qualunque: serve a far durare uguale il ramo «utente inesistente»
DUMMY_HASH = hash_password("nessuno-ha-questa-password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Verifica le credenziali ed emette un access token a scadenza.

    Credenziali sbagliate: 401, e il messaggio NON dice quale delle due era sbagliata.
    """
    utente = await db.scalar(select(AppUser).where(AppUser.username == form.username))
    # il confronto si fa comunque, anche senza utente: due percorsi di durata diversa
    # dicono a chi prova quali username esistono
    atteso = utente.password_hash if utente else DUMMY_HASH
    valida = verify_password(form.password, atteso)

    if utente is None or not valida:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token(utente.username, utente.role))


class RichiestaTokenServizio(BaseModel):
    client_id: str
    client_secret: str  # il client prova chi è: il nome da solo lo può scrivere chiunque
    resource: str  # l'URI canonico del servizio, come nella specifica: il token vale solo lì


class TokenServizio(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str


@router.post("/token-mcp", response_model=TokenServizio)
async def token_per_un_servizio(
    richiesta: RichiestaTokenServizio,
    user: Annotated[UserContext, Depends(get_current_user)],
) -> TokenServizio:
    """Scambia il token dell'operatore con uno per un servizio pubblicato.

    Il token di questa API non apre il server MCP, e quello del server MCP non apre questa API:
    ognuno ha la sua audience. Il client si autentica con il suo segreto, nel corpo come in
    `client_secret_post` di OAuth: un client che non lo prova riceve 401. Un rifiuto dello
    scambio è un 403 con il motivo, che non contiene dati.
    """
    autentica_client(richiesta.client_id, richiesta.client_secret)
    token, scope = emetti(user, richiesta.client_id, richiesta.resource)
    return TokenServizio(
        access_token=token, expires_in=int(DURATA.total_seconds()), scope=" ".join(sorted(scope))
    )
