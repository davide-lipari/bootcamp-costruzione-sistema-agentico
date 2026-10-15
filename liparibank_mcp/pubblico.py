# liparibank_mcp/pubblico.py — gli stessi tool, come servizio: HTTP, token, scope, versioni
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import RemoteAuthProvider, require_scopes
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError

from src.agents.deps import crea_deps
from src.agents.tools import (
    NIENTE_NEI_DOCUMENTI,
    ClienteArgs,
    RicercaArgs,
    SaldoArgs,
    build_tools_for,
)
from src.auth.deps import UserContext
from src.auth.mcp_tokens import EMITTENTE, RISORSA_MCP
from src.auth.tokens import ALGORITHM
from src.config import settings
from src.db.session import AsyncSessionLocal

VERSIONE = "2.0.0"  # quella che il server dice all'`initialize`, e che la voce di registry ripete
SOLA_LETTURA = {"readOnlyHint": True}
SCOPE = ["conti:leggi", "policy:leggi"]


class VerificaConScadenza(JWTVerifier):
    """Il verificatore JWT, che però non fa entrare un token senza scadenza: un token per un
    servizio è breve per contratto, e uno che non scade mai non lo è."""

    async def verify_token(self, token: str) -> AccessToken | None:
        accesso = await super().verify_token(token)
        return accesso if accesso is not None and accesso.expires_at is not None else None


class RicercaArgsV2(RicercaArgs):
    """La versione 2 aggiunge un parametro facoltativo: chi non lo manda ottiene quello di prima."""

    quanti: int = Field(default=3, ge=1, le=5, description="Quanti passaggi, da 1 a 5.")


def identita() -> UserContext:
    """Chi chiama, dal token che il transport ha già verificato. Senza token non c'è nessuno:
    capita al client in memoria, che non passa da HTTP, e qui si rifiuta come un estraneo."""
    token = get_access_token()
    if token is None:
        raise ToolError("Identità non verificata: serve un token.")
    sub, ruolo = token.claims.get("sub"), token.claims.get("role", "public")
    if not isinstance(sub, str) or not isinstance(ruolo, str):
        raise ToolError("Identità non verificata: token senza soggetto.")
    return UserContext(username=sub, role=ruolo)


async def _esegui(nome: str, args: BaseModel) -> str:
    utente = identita()
    async with AsyncSessionLocal() as db:
        tool = next(t for t in build_tools_for(utente, crea_deps(db)) if t.name == nome)
        return await tool.run(args)


def _argomenti[M: BaseModel](modello: type[M], **valori: object) -> M:
    try:
        return modello.model_validate(valori)
    except ValidationError as e:
        raise ToolError(f"Argomenti non validi: {e.errors(include_url=False)}") from e


def crea_server(base_url: str = RISORSA_MCP.removesuffix("/mcp")) -> FastMCP:
    """Il server pubblicato. `base_url` è dove gira: la risorsa è `{base_url}/mcp`, ed è
    l'audience che un token deve avere per entrare."""
    verificatore = VerificaConScadenza(
        public_key=settings.jwt_secret,  # HS256: la stessa chiave dell'emittente
        algorithm=ALGORITHM,
        issuer=EMITTENTE,
        audience=f"{base_url}/mcp",  # un token della API REST, o di un altro servizio, non entra
    )
    auth = RemoteAuthProvider(
        token_verifier=verificatore,
        authorization_servers=[AnyHttpUrl(EMITTENTE)],  # dove un client ottiene il token
        base_url=base_url,
        scopes_supported=SCOPE,
        resource_name="LipariBank, conti e procedure",
    )
    # mask_error_details: un'eccezione non prevista arriva al client come «Error calling tool»,
    # senza il suo messaggio; un ToolError, che scriviamo noi, passa com'è.
    # on_duplicate="error": due tool con lo stesso nome fermano l'avvio invece di sostituirsi
    mcp = FastMCP(
        "liparibank", version=VERSIONE, auth=auth, mask_error_details=True, on_duplicate="error"
    )

    @mcp.tool(
        name="search_policy",
        version="1",
        annotations=SOLA_LETTURA,
        auth=require_scopes("policy:leggi"),
    )
    async def search_policy_v1(query: str) -> str:
        """Cerca nelle procedure e nelle policy di LipariBank, con i permessi di chi chiama.

        Usalo quando la domanda riguarda regole, soglie, procedure o adempimenti della banca.
        Non usarlo per dati di clienti o di conti.

        Args:
            query: la domanda in forma completa, non una parola sola.
        """
        return await _esegui("search_documents", _argomenti(RicercaArgs, query=query))

    @mcp.tool(
        name="search_policy",
        version="2",
        annotations=SOLA_LETTURA,
        auth=require_scopes("policy:leggi"),
    )
    async def search_policy_v2(query: str, quanti: int = 3) -> str:
        """Cerca nelle procedure e nelle policy di LipariBank, con i permessi di chi chiama.

        Usalo quando la domanda riguarda regole, soglie, procedure o adempimenti della banca.
        Non usarlo per dati di clienti o di conti.

        Args:
            query: la domanda in forma completa, non una parola sola.
            quanti: quanti passaggi restituire, da 1 a 5. Di solito bastano 3.
        """
        args = _argomenti(RicercaArgsV2, query=query, quanti=quanti)
        utente = identita()
        async with AsyncSessionLocal() as db:
            deps = crea_deps(db)
            # lo stesso retrieval del tool del repository, con l'ACL del ruolo di chi chiama
            passaggi = await deps.retrieval.search_for_user(
                await deps.embedder.embed_one(args.query), utente.role, top_k=args.quanti
            )
        if not passaggi:
            return NIENTE_NEI_DOCUMENTI
        return "\n\n".join(f"[{r.document_id}] {r.content[:300]}" for r in passaggi)

    @mcp.tool(annotations=SOLA_LETTURA, auth=require_scopes("conti:leggi"))
    async def find_customer_accounts(customer_id: str) -> str:
        """I conti di un cliente, con IBAN ed etichetta, se il cliente è nel portafoglio di chi
        chiama.

        Usalo quando la domanda nomina un cliente e non un IBAN.

        Args:
            customer_id: il codice cliente, come compare nella sua scheda.
        """
        args = _argomenti(ClienteArgs, customer_id=customer_id)
        return await _esegui("find_customer_accounts", args)

    @mcp.tool(annotations=SOLA_LETTURA, auth=require_scopes("conti:leggi"))
    async def get_account_balance(account_id: str) -> str:
        """Saldo disponibile di un conto, se il conto è di un cliente nel portafoglio di chi
        chiama.

        Args:
            account_id: l'IBAN del conto, come lo restituisce find_customer_accounts.
        """
        return await _esegui("get_account_balance", _argomenti(SaldoArgs, account_id=account_id))

    return mcp


async def come_e_fatto(mcp: FastMCP) -> dict[str, Any]:
    """Quello che il server è davvero, letto dai tool registrati: gli scope che chiedono, se
    sono tutti in sola lettura, le versioni servite. La voce di registry deve dire questo."""
    tools = await mcp.local_provider.list_tools()
    scope: set[str] = set()
    versioni: dict[str, list[str]] = {}
    for t in tools:
        controlli = t.auth if isinstance(t.auth, list) else [t.auth]
        for c in controlli:
            scope |= set(getattr(c, "required_scopes", ()))
        if t.version is not None:
            versioni.setdefault(t.name, []).append(str(t.version))
    return {
        "version": mcp.version,
        "scope": sorted(scope),
        "sola_lettura": all(t.annotations and t.annotations.read_only_hint for t in tools),
        "versioni_servite": {
            n: sorted(v, reverse=True) for n, v in sorted(versioni.items()) if len(v) > 1
        },
    }


if __name__ == "__main__":
    crea_server().run(transport="streamable-http", host="127.0.0.1", port=8200)
