# src/api/movements.py — l'estratto dei movimenti, caricato in blocco
from fastapi import APIRouter, UploadFile

from src.services.movement_import import importa_csv
from src.types.movements import ImportResult

router = APIRouter(prefix="/api/ai", tags=["Movements"])


@router.post(
    "/movements/import",
    response_model=ImportResult,
    summary="Importa un CSV di movimenti",
    description="Importa le righe valide e rende conto, per numero di riga, di ognuna scartata.",
)
async def import_movements(file: UploadFile) -> ImportResult:
    return importa_csv(await file.read())
