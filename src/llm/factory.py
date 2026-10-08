# src/llm/factory.py — un solo posto decide quale fornitore, e con quali client
from functools import cache

import instructor
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.config import settings
from src.exceptions import AppError
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.client import LLMProvider
from src.llm.embedding_client import EmbeddingClient, EmbeddingClientLocale
from src.llm.openai_provider import OpenAIProvider


def _chiave(valore: str, nome: str) -> str:
    if not valore:  # l'SDK rifiuta una chiave vuota: meglio un errore che dica quale manca
        raise AppError(503, "LLM_NOT_CONFIGURED", f"Manca {nome}: il modello non è configurato")
    return valore


@cache  # un client per processo: tiene il suo pool di connessioni
def _openai() -> AsyncOpenAI:
    # timeout scelto, non ereditato (il default è di dieci minuti); max_retries è quello dell'SDK
    return AsyncOpenAI(
        api_key=_chiave(settings.openai_api_key, "OPENAI_API_KEY"),
        base_url=settings.openai_base_url or None,  # vuota = l'endpoint di OpenAI
        timeout=20.0,
        max_retries=2,
    )


@cache
def _anthropic() -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=_chiave(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        timeout=20.0,
        max_retries=2,
    )


def get_llm_provider() -> LLMProvider:
    modello = settings.default_model
    # «openai/» è il prefisso che usano i modelli OpenAI girati su altri fornitore (Groq)
    if modello.startswith(("gpt", "openai/")):
        return OpenAIProvider(_openai(), modello)
    if modello.startswith("claude"):
        return AnthropicProvider(_anthropic(), modello)
    raise ValueError(f"Modello sconosciuto: {modello}")


def get_openai() -> AsyncOpenAI:
    """Il client dell'SDK, per chi deve parlarci direttamente: lo stesso, con lo stesso timeout."""
    return _openai()


@cache
def get_instructor() -> instructor.AsyncInstructor:
    return instructor.from_openai(_openai())


def get_embedder() -> EmbeddingClient:
    # lo stesso modello per i documenti e per le domande: è la regola che non si viola.
    # È un modello locale: i vettori non hanno chiave, e la dimensione è quella della colonna
    return EmbeddingClientLocale(settings.embedding_model)
