# src/services/movement_import.py — il CSV riga per riga: ogni riga importata o spiegata
import csv

from pydantic import ValidationError

from src.exceptions import InvalidFileError
from src.types.movements import ImportProblem, ImportResult, MovementRow

INTESTAZIONE = ["date", "description", "amount", "currency"]


def motivo(errore: ValidationError) -> str:
    """Il primo campo che non va, e perché: una riga, leggibile da chi ha preparato il file."""
    e = errore.errors()[0]
    campo = ".".join(str(p) for p in e["loc"]) or "riga"
    return f"{campo}: {e['msg']}"


def importa_csv(contenuto: bytes) -> ImportResult:
    if not contenuto.strip():
        raise InvalidFileError("Il file è vuoto")
    try:
        testo = contenuto.decode("utf-8-sig")  # -sig: toglie il BOM che Excel mette in testa
    except UnicodeDecodeError:
        raise InvalidFileError("Il file non è testo UTF-8: non è un CSV") from None

    # Un movimento per riga del file, e ogni riga si legge da sola: una virgoletta aperta e mai
    # chiusa, o un campo oltre il limite del modulo csv, rovinano la loro riga e non le successive
    righe = [r.rstrip("\r") for r in testo.split("\n")]
    if next(csv.reader(righe[:1]), []) != INTESTAZIONE:
        raise InvalidFileError(f"L'intestazione deve essere {','.join(INTESTAZIONE)}")

    importate, problemi = 0, []
    for numero, linea in enumerate(righe[1:], start=2):  # la riga nel file: la 1 è l'intestazione
        if not linea.strip():
            continue  # una riga vuota non è un movimento
        try:
            campi = next(csv.reader([linea], strict=True))
        except csv.Error:  # sollevato nel codice: se nessuno lo cattura, è un 500
            problemi.append(ImportProblem(row=numero, reason="riga illeggibile come CSV"))
            continue
        if len(campi) != len(INTESTAZIONE):
            quanti = "più" if len(campi) > len(INTESTAZIONE) else "meno"
            problemi.append(ImportProblem(row=numero, reason=f"{quanti} di quattro campi"))
            continue
        try:
            MovementRow.model_validate(dict(zip(INTESTAZIONE, campi, strict=True)))
        except ValidationError as e:  # costruito nel codice: se nessuno lo cattura, è un 500
            problemi.append(ImportProblem(row=numero, reason=motivo(e)))
            continue
        importate += 1
    return ImportResult(imported_count=importate, problems=problemi)
