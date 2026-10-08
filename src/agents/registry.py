# src/agents/registry.py
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    """Un tool esposto al modello: contratto pubblico più implementazione."""

    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any], Awaitable[str]]  # riceve un'istanza di args_model, già validata
    scrive: bool = True  # chi non lo dichiara è trattato come chi scrive
    # le risposte che dicono «il dato non c'è»: nel log diventano result_kind="vuoto"
    senza_dati: frozenset[str] = frozenset()
    # Solo per chi scrive: dice, dagli argomenti validati, se serve l'approvazione.
    # None vuol dire «sempre»: il tool nuovo che nessuno ha pensato è protetto.
    serve_approvazione: Callable[[Any], bool] | None = None

    def to_openai_schema(self) -> ChatCompletionToolParam:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


def impronta(valore: str) -> str:
    """Uguale per due valori uguali: nel log dice se un valore si ripete, non quale è."""
    return hashlib.sha256(valore.encode()).hexdigest()[:8]
