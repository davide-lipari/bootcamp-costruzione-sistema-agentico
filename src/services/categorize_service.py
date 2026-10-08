# src/services/categorize_service.py — Giorno 9: lo schema imposto al modello, e il suo costo
import instructor
import openai
from instructor.core import InstructorRetryException

from src.agents.loop import costo_chiamata
from src.exceptions import LLMProviderError
from src.observability.cost_tracker import CostTracker
from src.observability.ledger import CostLedger
from src.types.categorize import CategorizeRequest, CategorizeResponse


class CategorizeService:
    def __init__(
        self,
        client: instructor.AsyncInstructor,
        model: str,
        system_prompt: str,
        costi: CostTracker | None = None,
        registro: CostLedger | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.costi = costi  # il tetto della chat: l'endpoint lo passa sempre
        self.registro = registro  # Giorno 9: dove va il costo, con quello di chat, advice, agente

    async def categorize(self, req: CategorizeRequest) -> CategorizeResponse:
        if self.costi is not None:
            await self.costi.verifica()  # prima di spendere: ogni chiamata al modello spende
        try:
            risposta, completion = await self.client.chat.completions.create_with_completion(
                model=self.model,
                response_model=CategorizeResponse,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Descrizione: {req.description}\n"
                        f"Importo: {req.amount} {req.currency}",
                    },
                ],
                max_retries=2,  # ogni nuovo tentativo è una chiamata pagata
                max_tokens=300,
                temperature=0.0,  # la variabilità più bassa: non vuol dire deterministico
            )
        except InstructorRetryException as e:
            # lo schema non rispettato nemmeno dopo i tentativi, già pagati: è un guasto del
            # fornitore, non un bug nostro, e il client riceve il 502 della chat
            raise LLMProviderError("openai", f"risposta fuori schema: {e}") from e
        except openai.APIError as e:  # fornitore giù, modello sconosciuto, chiave rifiutata
            raise LLMProviderError("openai", str(e)) from e
        if self.registro is not None:
            # l'usage che Instructor restituisce somma anche i tentativi respinti: pagati tutti
            uso = completion.usage
            self.registro.aggiungi(
                endpoint="categorize",
                username="-",  # la categorizzazione non chiede chi sei
                model=self.model,
                tokens=uso.total_tokens if uso else 0,
                cost_eur=costo_chiamata(uso, self.model),
            )
            await self.registro.session.commit()
        return risposta
