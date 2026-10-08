# src/auth/deps.py — chi sta chiedendo, e con quale ruolo
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.auth.tokens import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class UserContext(BaseModel):
    username: str
    role: str


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserContext:
    payload = decode_token(token)
    username = payload.get("sub")
    if not isinstance(username, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token senza subject")
    return UserContext(username=username, role=payload.get("role", "public"))


def require_role(*allowed: str) -> Callable[[UserContext], Awaitable[UserContext]]:
    """Dependency factory: consente l'accesso solo ai ruoli indicati."""

    async def _check(
        user: Annotated[UserContext, Depends(get_current_user)],
    ) -> UserContext:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accesso riservato a: {', '.join(allowed)}",
            )
        return user

    return _check
