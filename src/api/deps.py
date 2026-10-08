# src/api/deps.py — le dipendenze che più endpoint chiedono
from typing import Annotated

from fastapi import Header, HTTPException, status

from src.config import Settings, settings


async def get_settings() -> Settings:
    return settings


async def get_current_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    # Un'intestazione assente o malformata è un 401: con Header(...) obbligatorio sarebbe un
    # 422, cioè «hai compilato male la richiesta» invece di «non so chi sei».
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Serve un token", headers={"WWW-Authenticate": "Bearer"}
        )
    # Oggi il token non si decodifica: l'autenticazione vera arriva al Giorno 6.
    # Lo user_id resta finto e dichiaratamente tale.
    return "user-id-hardcoded-demo"
