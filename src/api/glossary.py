# src/api/glossary.py — lo stesso schema su un caso che non è la commessa
from fastapi import APIRouter

from src.exceptions import NotFoundError
from src.types.glossary import GlossaryRequest, GlossaryResponse

router = APIRouter(prefix="/api/ai", tags=["Glossario"])

GLOSSARIO: dict[str, tuple[str, str]] = {
    "iban": (
        "Codice che identifica un conto in modo univoco a livello internazionale.",
        "Circolare interna 12/2025",
    ),
    "sepa": (
        "Area unica dei pagamenti in euro: un bonifico interno all'area segue regole e "
        "tempi comuni.",
        "Circolare interna 04/2024",
    ),
    "mav": (
        "Bollettino di pagamento mediante avviso, incassabile presso qualunque sportello.",
        "Manuale operativo, cap. 7",
    ),
}


@router.post("/glossary", response_model=GlossaryResponse, summary="Spiega un termine interno")
async def glossary(req: GlossaryRequest) -> GlossaryResponse:
    voce = GLOSSARIO.get(req.term.strip().lower())
    if voce is None:
        raise NotFoundError(f"Il termine «{req.term}» non è nel glossario interno")
    definizione, fonte = voce
    return GlossaryResponse(term=req.term, definition=definizione, source=fonte)
