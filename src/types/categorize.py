# src/types/categorize.py — il contratto della categorizzazione
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["UTILITIES", "GROCERIES", "TRANSPORT", "RESTAURANTS", "ENTERTAINMENT", "OTHER"]


class CategorizeRequest(BaseModel):
    # gli spazi ai bordi si tolgono prima dei vincoli: «   » non passa per una descrizione
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=200, description="Il testo del movimento")
    amount: Decimal = Field(gt=0, description="L'importo, maggiore di zero")
    currency: str = Field(default="EUR", pattern=r"^[A-Z]{3}$", description="Tre lettere, es. EUR")


class CategorizeResponse(BaseModel):
    category: Category = Field(description="La categoria, una sola fra le sei ammesse")
    subcategory: str = Field(description="Un'etichetta più precisa, per esempio ENERGY o FUEL")
    confidence: float = Field(ge=0.0, le=1.0, description="Quanto è sicura la scelta, da 0 a 1")
    reasoning: str = Field(description="Una o due frasi che spiegano la scelta")
