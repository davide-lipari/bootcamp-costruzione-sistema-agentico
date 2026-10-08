# src/lc/catena.py — la catena di policy: il retrieval del repository, una risposta citata
from collections.abc import Awaitable, Callable
from operator import itemgetter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from src.services.retrieval_service import RetrievalResult


class RispostaCitata(BaseModel):
    risposta: str
    fonti: list[str]  # gli identificativi dei passaggi usati, come compaiono fra parentesi quadre


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sei l'assistente delle policy di LipariBank per gli operatori di filiale. Rispondi "
            "in italiano e solo da ciò che il contesto contiene. In `fonti` metti gli "
            "identificativi dei passaggi che hai usato. Se il contesto non contiene la "
            "risposta, dillo e lascia `fonti` vuoto.",
        ),
        ("human", "Contesto:\n{contesto}\n\nDomanda: {domanda}"),
    ]
)

Recupera = Callable[[str], Awaitable[list[RetrievalResult]]]


def formatta(passaggi: list[RetrievalResult]) -> str:
    if not passaggi:
        return "(nessun documento visibile a questo ruolo)"
    return "\n\n".join(f"[{p.document_id}] {p.content[:300]}" for p in passaggi)


def costruisci_catena(llm: BaseChatModel, recupera: Recupera) -> Runnable[dict[str, Any], Any]:
    """{"domanda"} → {"passaggi", "uscita": {"raw", "parsed", "parsing_error"}}.

    I passaggi restano nell'uscita perché chi riceve la risposta deve poter controllare che
    ogni fonte citata sia fra quelle recuperate: il modello può scriverne una che non c'era.
    """
    con_passaggi = RunnablePassthrough.assign(
        passaggi=itemgetter("domanda") | RunnableLambda(recupera)
    )
    risposta = (
        RunnablePassthrough.assign(contesto=itemgetter("passaggi") | RunnableLambda(formatta))
        | PROMPT
        # function_calling, non il json_schema predefinito: con json_schema un'uscita fuori
        # schema solleva dentro la chiamata, e include_raw non la trasforma in parsed=None
        | llm.with_structured_output(RispostaCitata, include_raw=True, method="function_calling")
    )
    return con_passaggi | {"passaggi": itemgetter("passaggi"), "uscita": risposta}


def fonti_verificate(risposta: RispostaCitata, passaggi: list[RetrievalResult]) -> list[str]:
    """Le sole fonti che il retrieval ha dato davvero: una che non c'era è inventata."""
    visti = {p.document_id for p in passaggi}
    return [f for f in risposta.fonti if f in visti]
