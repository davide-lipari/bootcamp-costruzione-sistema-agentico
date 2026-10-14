# src/lg/tempo.py — tornare indietro su un'esecuzione: per capire, e per provare un'alternativa
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


@dataclass(frozen=True)
class Passo:
    """Un checkpoint, per chi fa supporto: dove era, chi aveva lavorato, quanto era costato."""

    config: RunnableConfig
    step: int
    prossimo: tuple[str, ...]
    visitati: list[str]
    costo_eur: Decimal


async def cronologia(
    grafo: CompiledStateGraph[Any, Any, Any, Any], config: RunnableConfig
) -> list[Passo]:
    """Dal primo all'ultimo. Contiene lo stato intero, quindi dati dei clienti: chi la legge ha
    l'accesso di chi può leggere quelle conversazioni, non di più."""
    passi = [
        Passo(
            config=s.config,
            step=int(s.metadata.get("step", -1)) if s.metadata else -1,
            prossimo=tuple(s.next),
            visitati=list(s.values.get("visitati", [])),
            costo_eur=s.values.get("costo_eur", Decimal("0")),
        )
        async for s in grafo.aget_state_history(config)
    ]
    return list(reversed(passi))


async def prova_da(
    grafo: CompiledStateGraph[Any, Any, Any, Any], passo: Passo, modifica: dict[str, Any]
) -> dict[str, Any]:
    """Un ramo nuovo da quel punto, con la modifica, fermato prima di `scrivi`.
    Vale sul grafo della banca: su una sessione quel grafo lavora dentro `turno`, e
    l'interruzione non lo raggiunge. Le chiamate al modello del ramo si ripagano; la storia
    originale resta."""
    ramo = await grafo.aupdate_state(passo.config, modifica)
    stato: dict[str, Any] = await grafo.ainvoke(None, ramo, interrupt_before=["scrivi"])
    return stato
