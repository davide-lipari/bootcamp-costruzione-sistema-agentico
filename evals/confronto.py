# evals/confronto.py — gli stessi casi su più implementazioni, con i controlli di runner.py
import asyncio
import io
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from evals.implementazioni import Esito, Implementazione
from evals.runner import CONTROLLI, DATASETS, SOGLIE, Metriche, leggi, rapporto
from src.agents.deps import crea_deps
from src.auth.deps import UserContext
from src.db.session import AsyncSessionLocal
from src.llm.embedding_client import EmbeddingClient

# il sesto controllo sta qui, non in runner.py: quel file resta quello del cancello
CONTROLLI_CON_VALORE = CONTROLLI + ("risposta_contiene",)
CIFRE = re.compile(r"\d+(?:[.,]\d+)?")


def valore_in(risposta: str, atteso: str) -> bool:
    """Il valore atteso c'è nella risposta: spazi e simbolo € non contano, virgola o punto
    è lo stesso separatore decimale, e 68,6 vale quanto 68,60.

    Se l'atteso non è un numero si guarda la sottostringa, sempre senza spazi né €.
    """
    atteso_n = atteso.replace(" ", "").replace("€", "").lower()
    if not atteso_n:
        return True
    try:
        valore = Decimal(atteso_n.replace(",", "."))
    except InvalidOperation:
        return atteso_n in risposta.replace(" ", "").replace("€", "").lower()
    return any(Decimal(c.replace(",", ".")) == valore for c in CIFRE.findall(risposta))


def controlla(atteso: dict[str, Any], esito: Esito) -> list[str]:
    """I sei controlli di eval_traiettorie, su un Esito invece che su un RunResult."""
    chiamati = set(esito.tool_calls)
    da_citare = atteso.get("documento_da_citare")
    valore = atteso.get("risposta_contiene")
    esiti = {
        "tool_richiesti": set(atteso["tool_richiesti"]) <= chiamati,
        "tool_vietati": not (set(atteso.get("tool_vietati", [])) & chiamati),
        "passi": esito.steps <= atteso["passi_max"],
        "terminazione": esito.stopped_by == atteso["terminazione"],
        "citazione": f"[{da_citare}]" in esito.reply if da_citare else True,
        # senza il campo il controllo passa: la maggior parte dei casi non misura il valore
        "risposta_contiene": valore_in(esito.reply, valore) if valore else True,
    }
    return [c for c in CONTROLLI_CON_VALORE if not esiti[c]]


async def misura(
    path: Path, impl: Implementazione, embedder: EmbeddingClient, openai: AsyncOpenAI
) -> Metriche:
    casi = await asyncio.to_thread(leggi, path)
    superati, fallimenti, costo, violazioni = 0, [], Decimal("0"), Counter[str]()
    per_caso: dict[str, str] = {}  # il costo di ogni caso, superato o no: il totale non dice dove
    ricerche: dict[str, int] = {}  # le ricerche eseguite per caso: dove finisce il retrieval
    for caso in casi:
        utente = UserContext(**caso["input"]["user"])
        async with AsyncSessionLocal() as s:  # una sessione per caso: nessun caso vede l'altro
            deps = crea_deps(s, embedder=embedder, openai=openai)
            try:
                esito = await impl.rispondi(caso["input"]["message"], utente, deps)
            except Exception as exc:  # noqa: BLE001 — un caso che esplode è un fallimento
                esito = Esito(f"ERRORE: {type(exc).__name__}: {exc}"[:200], stopped_by="errore")
        costo += esito.cost_eur
        per_caso[caso["id"]] = f"{esito.cost_eur:.6f}"
        ricerche[caso["id"]] = esito.ricerche
        violati = controlla(caso["expected"], esito)
        violazioni.update(violati)
        if violati:
            fallimenti.append(
                {
                    "id": caso["id"],
                    "violati": violati,
                    "tool_chiamati": sorted(set(esito.tool_calls)),
                    "passi": esito.steps,
                    "stopped_by": esito.stopped_by,
                    "costo_eur": f"{esito.cost_eur:.6f}",
                }
            )
        else:
            superati += 1
    return Metriche(
        nome="traiettorie",
        n=len(casi),
        punteggio=superati / len(casi) if casi else 0.0,
        costo_eur=costo,
        fallimenti=fallimenti,
        dettagli={
            "implementazione": impl.nome,
            "violazioni_per_controllo": dict(violazioni),
            "costo_per_caso": per_caso,
            "ricerche_per_caso": ricerche,
        },
    )


async def confronta(
    path: Path, impls: list[Implementazione], embedder: EmbeddingClient, openai: AsyncOpenAI
) -> dict[str, Metriche]:
    return {i.nome: await misura(path, i, embedder, openai) for i in impls}


async def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    from evals.implementazioni import AgentePrebuilt, Catena, CicloAMano
    from src.config import settings
    from src.lc.modelli import modello_chat
    from src.llm.factory import get_embedder, get_openai

    openai, modello = get_openai(), settings.default_model
    llm = modello_chat(modello)  # la chiave dal .env, come il client del ciclo
    impls: list[Implementazione] = [
        CicloAMano(openai, modello),
        Catena(llm, modello),
        AgentePrebuilt(llm, modello),
    ]
    risultati = await confronta(DATASETS / "agent_traiettorie.jsonl", impls, get_embedder(), openai)
    for nome, m in risultati.items():
        print(f"\n== {nome}")
        print(rapporto([m], SOGLIE))
        for caso, euro in m.dettagli["costo_per_caso"].items():
            print(f"  {caso:24} {m.dettagli['ricerche_per_caso'][caso]} ricerche  {euro} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
