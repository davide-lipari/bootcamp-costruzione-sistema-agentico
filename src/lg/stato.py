# src/lg/stato.py — il contratto fra i nodi: cosa entra, cosa si accumula, cosa si decide
from decimal import Decimal
from operator import add
from typing import Annotated, Any, Literal, TypedDict

Instradamento = Literal["dati_conto", "policy", "entrambi", "segnalazione"]
Uscita = Literal["model", "budget", "max_steps"]  # l'attesa la riconosce lo strato HTTP
Specialista = Literal["dati_conto", "policy", "segnalazione"]


class Contributo(TypedDict):
    """Quello che un nodo consegna alla sintesi: dati che il codice sa leggere, non prosa."""

    specialista: Specialista
    completo: bool  # ha finito il lavoro, non l'ha fermato un tetto
    contenuto: str
    tool_calls: list[str]


class Decisione(TypedDict):
    approvato: bool
    da: str | None  # None quando sotto soglia non serve nessuno
    motivo: str | None


class StatoBanca(TypedDict):
    # --- input: li scrive il server all'ingresso, e non cambiano --------------------------
    domanda: str
    richiedente: str  # per il registro e per «chi chiede non decide», mai per i tool

    # --- accumulatori: storia, quindi un reducer che somma ----------------------------------
    contributi: Annotated[list[Contributo], add]
    visitati: Annotated[list[str], add]
    passi: Annotated[int, add]  # chiamate pagate al modello, in tutto il grafo
    costo_eur: Annotated[Decimal, add]

    # --- campi di lavoro: fotografie, l'ultimo che scrive vince -----------------------------
    instradamento: Instradamento
    proposta: dict[str, Any] | None  # gli argomenti della segnalazione, già validati
    decisione: Decisione
    risposta: str
    stopped_by: Uscita
