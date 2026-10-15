# src/auth/mcp_tokens.py — i token per i server pubblicati: per chi, per quale client, per fare cosa
import hashlib
import hmac
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from src.auth.deps import UserContext
from src.auth.tokens import ALGORITHM
from src.config import settings
from src.exceptions import AppError

# l'applicazione della banca è l'emittente: il server MCP e l'agente A2A lo verificano nel token.
# Con la barra finale, la forma in cui i metadati del server lo pubblicano: `iss` deve coincidere
EMITTENTE = os.environ.get("LIPARI_EMITTENTE", "http://127.0.0.1:8000").rstrip("/") + "/"
# l'URI canonico di ciascun servizio pubblicato: è l'audience che il servizio pretende
RISORSA_MCP = os.environ.get("LIPARI_MCP_URL", "http://127.0.0.1:8200/mcp")
RISORSA_A2A = os.environ.get("LIPARI_A2A_URL", "http://127.0.0.1:8300")
DURATA = timedelta(minutes=10)  # breve: un token per un servizio non si conserva

# quello che un ruolo può fare, qualunque client usi: il limite della persona
SCOPE_DEL_RUOLO: dict[str, frozenset[str]] = {
    "operator": frozenset({"conti:leggi", "policy:leggi", "istruttoria:chiedi"}),
    "compliance_lead": frozenset({"policy:leggi", "istruttoria:chiedi"}),
    "risk_lead": frozenset({"policy:leggi"}),
}
# quello che ciascun servizio sa fare: un token per un servizio non porta gli scope di un altro
SCOPE_DELLA_RISORSA: dict[str, frozenset[str]] = {
    RISORSA_MCP: frozenset({"conti:leggi", "policy:leggi"}),
    RISORSA_A2A: frozenset({"istruttoria:chiedi"}),
}


@dataclass(frozen=True)
class ClienteRegistrato:
    """Un client preregistrato: per quali servizi, e il massimo che può chiedere."""

    descrizione: str
    risorse: frozenset[str]
    scope: frozenset[str]


CLIENTI: dict[str, ClienteRegistrato] = {
    "assistente-filiale": ClienteRegistrato(
        "il grafo dell'assistente di filiale",
        frozenset({RISORSA_MCP, RISORSA_A2A}),
        frozenset({"conti:leggi", "policy:leggi", "istruttoria:chiedi"}),
    ),
    "supporto-procedure": ClienteRegistrato(
        "l'assistente del supporto, che risponde sulle procedure e non vede i conti",
        frozenset({RISORSA_MCP}),
        frozenset({"policy:leggi"}),
    ),
    "agente-compliance": ClienteRegistrato(
        "l'agente della Compliance, di un altro fornitore: chiede istruttorie, non vede i conti",
        frozenset({RISORSA_A2A}),
        frozenset({"istruttoria:chiedi"}),
    ),
}


class TokenNegatoError(AppError):
    """Lo scambio non si fa: client sconosciuto, servizio non suo, o niente da concedere."""

    def __init__(self, message: str) -> None:
        super().__init__(403, "TOKEN_NEGATO", message)


class ClientNonAutenticatoError(AppError):
    """Chi chiama non ha dimostrato di essere il client che dice: il nome da solo non basta."""

    def __init__(self) -> None:
        super().__init__(401, "CLIENT_NON_AUTENTICATO", "Client non autenticato.")


def segreto_del_client(client_id: str) -> str:
    """Il segreto di un client registrato, derivato dalla chiave dell'emittente.

    La banca lo consegna a chi installa il client, e non lo conserva da nessuna parte: lo
    ricalcola. Cambiare la chiave dell'emittente cambia tutti i segreti insieme.
    """
    return hmac.new(
        settings.jwt_secret.encode(), f"client:{client_id}".encode(), hashlib.sha256
    ).hexdigest()


def autentica_client(client_id: str, segreto: str) -> None:
    """Il client prova chi è prima di ricevere un token: è la differenza fra un nome scritto nel
    corpo della richiesta e un'identità. Un client sconosciuto e un segreto sbagliato danno la
    stessa risposta, così il rifiuto non dice quali client esistono."""
    atteso = segreto_del_client(client_id) if client_id in CLIENTI else ""
    if not atteso or not hmac.compare_digest(segreto, atteso):
        raise ClientNonAutenticatoError()


def emetti(user: UserContext, client_id: str, risorsa: str) -> tuple[str, frozenset[str]]:
    """Il token dell'operatore diventa un token per UN servizio, con gli scope che servono.

    Gli scope sono l'intersezione fra quelli del client, quelli del ruolo e quelli del servizio:
    un client largo non allarga una persona, e una persona larga non allarga un client. Con il
    client autenticato prima (`autentica_client`, nell'endpoint), è il confused deputy chiuso
    nel punto in cui nasce il token, non in ciascun server.
    """
    cliente = CLIENTI.get(client_id)
    if cliente is None:
        raise TokenNegatoError("client non registrato")
    if risorsa not in cliente.risorse:
        raise TokenNegatoError("servizio non previsto per questo client")
    concessi = (
        cliente.scope
        & SCOPE_DEL_RUOLO.get(user.role, frozenset())
        & SCOPE_DELLA_RISORSA.get(risorsa, frozenset())
    )
    if not concessi:
        raise TokenNegatoError("nessuno scope concedibile")
    adesso = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": EMITTENTE,
        "sub": user.username,
        "role": user.role,
        "aud": risorsa,  # il servizio verifica che il token sia per lui, e per nessun altro
        "azp": client_id,  # il client a cui è emesso: chi lo registra sa quale client ha chiamato
        "scope": " ".join(sorted(concessi)),
        "iat": adesso,
        "exp": adesso + DURATA,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM), concessi


if __name__ == "__main__":
    # uv run python -m src.auth.mcp_tokens assistente-filiale: il segreto da consegnare al client
    nome = sys.argv[1] if len(sys.argv) > 1 else ""
    if nome not in CLIENTI:
        print(f"client sconosciuto; quelli registrati: {', '.join(sorted(CLIENTI))}")
        raise SystemExit(1)
    print(segreto_del_client(nome))
