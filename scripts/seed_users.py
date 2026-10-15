# scripts/seed_users.py — i tre utenti di sviluppo, con la password che sta nel .env
"""I tre utenti di sviluppo. Script di sviluppo, si lancia a mano.

uv run python -m scripts.seed_users           # crea gli utenti che mancano
uv run python -m scripts.seed_users --ruota   # e riscrive la password di quelli che ci sono

La password viene da `SEED_PASSWORD` nel `.env`. Fino a ieri era scritta qui, quindi sta nella
storia del repository pubblico: toglierla dal file non la toglie da lì. Per questo quella vecchia
si rifiuta, e cambiare il `.env` non basta: gli utenti già creati hanno l'hash di prima.
"""

import argparse
import asyncio

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import select

from src.auth.passwords import hash_password
from src.db.models import AppUser
from src.db.session import AsyncSessionLocal

# la password che stava qui fino a ieri: è pubblica, quindi bruciata
BRUCIATA = "bootcamp"
MINIMO = 12

UTENTI = [
    ("mbianchi", "Marco Bianchi", "operator"),
    ("grossi", "Giulia Rossi", "compliance_lead"),
    ("lverdi", "Lucia Verdi", "risk_lead"),
]


class Credenziali(BaseSettings):
    """Solo quello che serve agli script di sviluppo: il resto della configurazione sta altrove."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    seed_password: SecretStr = SecretStr("")


def password_di_sviluppo() -> str:
    """La password del `.env`, controllata: assente, bruciata o corta, si ferma e dice perché."""
    valore = Credenziali().seed_password.get_secret_value()
    if not valore:
        raise SystemExit("Manca SEED_PASSWORD nel .env: scegline una di almeno 12 caratteri.")
    if valore == BRUCIATA:
        raise SystemExit("SEED_PASSWORD è quella che stava nel codice ed è pubblica: cambiala.")
    if len(valore) < MINIMO:
        raise SystemExit(f"SEED_PASSWORD è troppo corta: almeno {MINIMO} caratteri.")
    return valore


async def semina(password: str, *, ruota: bool = False) -> tuple[int, int]:
    """Crea gli utenti che mancano e, con `ruota`, riscrive l'hash di quelli che ci sono.

    Restituisce quanti ne ha creati e quanti ne ha ruotati.
    """
    creati = ruotati = 0
    async with AsyncSessionLocal() as session:
        for username, full_name, role in UTENTI:
            utente = await session.scalar(select(AppUser).where(AppUser.username == username))
            if utente is None:
                session.add(
                    AppUser(
                        username=username,
                        full_name=full_name,
                        password_hash=hash_password(password),
                        role=role,
                    )
                )
                creati += 1
            elif ruota:
                utente.password_hash = hash_password(password)
                ruotati += 1
        await session.commit()
    return creati, ruotati


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gli utenti di sviluppo.")
    parser.add_argument(
        "--ruota", action="store_true", help="riscrive la password degli utenti che ci sono già"
    )
    args = parser.parse_args(argv)
    creati, ruotati = await semina(password_di_sviluppo(), ruota=args.ruota)
    # la password non si stampa: un terminale condiviso o un log la porterebbero altrove
    print(f"{creati} utenti creati, {ruotati} password riscritte")
    if not args.ruota and creati < len(UTENTI):
        print("   chi c'era già ha ancora la password di prima: per cambiarla, --ruota")


if __name__ == "__main__":
    asyncio.run(main())
