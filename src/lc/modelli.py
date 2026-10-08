# src/lc/modelli.py — i modelli di LangChain, con la chiave e i tempi dei client del repository
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import settings
from src.exceptions import AppError


def modello_chat(model: str, max_retries: int = 2) -> ChatOpenAI:
    """Un ChatOpenAI come i client del repository.

    La chiave arriva da `settings`, cioè anche dal `.env`: ChatOpenAI da solo la cerca soltanto
    nell'ambiente del processo, e con la chiave scritta nel `.env` direbbe che manca. Timeout e
    tentativi sono quelli del client del ciclo, così un confronto mette le implementazioni nelle
    stesse condizioni anche quando il modello è lento. I tentativi si possono togliere: chi
    mette i ritentativi in un middleware non vuole che si moltiplichino con quelli dell'SDK.
    """
    if not settings.openai_api_key:
        raise AppError(
            503, "LLM_NOT_CONFIGURED", "Manca OPENAI_API_KEY: il modello non è configurato"
        )
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url or None,  # come il client del ciclo: vuota = OpenAI
        timeout=20.0,
        max_retries=max_retries,
    )
