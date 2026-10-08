# src/services/categorize_rules.py — la categoria da parole chiave, finché non arriva il modello
import re

from src.types.categorize import CategorizeRequest, CategorizeResponse, Category

# (categoria, sottocategoria, parole che la indicano). Parole intere, non pezzi di parola:
# «eni» non deve scattare dentro «servizi», né «bar» dentro «barbiere».
REGOLE: list[tuple[Category, str, frozenset[str]]] = [
    ("UTILITIES", "ENERGY", frozenset({"enel", "edison", "a2a", "hera", "iren", "luce", "gas"})),
    ("UTILITIES", "TELECOM", frozenset({"tim", "vodafone", "windtre", "fastweb", "iliad"})),
    ("GROCERIES", "SUPERMARKET", frozenset({"esselunga", "coop", "conad", "carrefour", "lidl"})),
    ("TRANSPORT", "FUEL", frozenset({"eni", "q8", "tamoil", "carburante", "benzina"})),
    ("TRANSPORT", "PUBLIC", frozenset({"trenitalia", "italo", "atm", "atac", "metro"})),
    ("RESTAURANTS", "RESTAURANT", frozenset({"ristorante", "pizzeria", "trattoria", "osteria"})),
    ("ENTERTAINMENT", "STREAMING", frozenset({"netflix", "spotify", "disney", "dazn"})),
]


def categorizza(req: CategorizeRequest) -> CategorizeResponse:
    parole = set(re.findall(r"\w+", req.description.lower()))
    trovate = [(cat, sub, sorted(parole & ch)) for cat, sub, ch in REGOLE if parole & ch]
    if not trovate:
        return CategorizeResponse(
            category="OTHER",
            subcategory="UNKNOWN",
            confidence=0.2,
            reasoning="Nessuna parola chiave riconosciuta nella descrizione.",
        )
    categoria, sotto, indizi = trovate[0]
    # Due categorie diverse sulla stessa riga: la regola sceglie la prima, e lo dichiara
    in_conflitto = len({cat for cat, _, _ in trovate}) > 1
    return CategorizeResponse(
        category=categoria,
        subcategory=sotto,
        confidence=0.5 if in_conflitto else 0.8,
        reasoning=f"Parola chiave «{', '.join(indizi)}»"
        + (": ma la descrizione ne contiene anche di un'altra categoria." if in_conflitto else "."),
    )
