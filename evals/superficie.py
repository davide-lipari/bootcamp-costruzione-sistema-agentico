# evals/superficie.py — cosa vede ciascun client, per ciascun ruolo: la superficie pubblicata
import asyncio
import io
import sys
from dataclasses import dataclass

import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from liparibank_mcp.pubblico import crea_server
from src.auth.deps import UserContext
from src.auth.mcp_tokens import CLIENTI, RISORSA_A2A, RISORSA_MCP, SCOPE_DEL_RUOLO, emetti
from src.exceptions import AppError

BASE = RISORSA_MCP.removesuffix("/mcp")


@dataclass(frozen=True)
class Riga:
    client_id: str
    ruolo: str
    tool: tuple[str, ...]  # i tool visti, con le versioni servite dove ce n'è più d'una
    scrivono: int  # quanti di quelli non dichiarano di essere in sola lettura
    agente: bool  # se lo scambio concede un token per l'agente A2A


async def misura() -> list[Riga]:
    """Il server vero, su HTTP nello stesso processo: l'elenco lo dice lui, non un documento."""
    app = crea_server(BASE).http_app(path="/mcp")

    def fabbrica(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
        **_: object,
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url=BASE,
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    righe = []
    async with app.router.lifespan_context(app):
        for client_id in sorted(CLIENTI):
            for ruolo in sorted(SCOPE_DEL_RUOLO):
                persona = UserContext(username="misura", role=ruolo)
                try:
                    token, _ = emetti(persona, client_id, RISORSA_MCP)
                except AppError:
                    righe.append(Riga(client_id, ruolo, (), 0, _agente(persona, client_id)))
                    continue
                trasporto = StreamableHttpTransport(
                    RISORSA_MCP,
                    headers={"Authorization": f"Bearer {token}"},
                    httpx_client_factory=fabbrica,
                )
                async with Client(trasporto) as c:
                    tools = await c.list_tools()
                visti = []
                for t in sorted(tools, key=lambda t: t.name):
                    versioni = (t.meta or {}).get("fastmcp", {}).get("versions", [])
                    visti.append(f"{t.name} v{'/'.join(versioni)}" if len(versioni) > 1 else t.name)
                scrivono = sum(
                    1 for t in tools if not (t.annotations and t.annotations.read_only_hint)
                )
                righe.append(
                    Riga(client_id, ruolo, tuple(visti), scrivono, _agente(persona, client_id))
                )
    return righe


def _agente(persona: UserContext, client_id: str) -> bool:
    try:
        _, scope = emetti(persona, client_id, RISORSA_A2A)
    except AppError:
        return False
    return "istruttoria:chiedi" in scope


def rapporto(righe: list[Riga]) -> str:
    testo = ["client | ruolo | tool visti | che scrivono | agente A2A"]
    for r in righe:
        visti = ", ".join(r.tool) or "nessuno"
        testo.append(
            f"{r.client_id} | {r.ruolo} | {visti} | {r.scrivono} | {'sì' if r.agente else 'no'}"
        )
    totale = sum(len(r.tool) for r in righe)
    testo.append(f"coppie client-ruolo: {len(righe)} · tool esposti in tutto: {totale}")
    return "\n".join(testo)


async def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    print(rapporto(await misura()))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
