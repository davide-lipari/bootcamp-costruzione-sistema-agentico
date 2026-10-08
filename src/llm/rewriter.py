# src/llm/rewriter.py — la domanda dell'utente, in una forma che il recupero sa cercare
import hashlib
import logging

from src.llm.client import LLMProvider
from src.llm.types import Message

log = logging.getLogger(__name__)


class QueryRewriter:
    """Porta la domanda dell'utente in una forma che il recupero sa cercare."""

    def __init__(self, llm: LLMProvider, prompt: str, enabled: bool = True) -> None:
        self.llm = llm
        self.prompt = prompt
        self.enabled = enabled  # l'interruttore: spento, il recupero cerca sull'originale
        self._cache: dict[str, str] = {}  # in memoria di processo: al Giorno 10 diventa Redis

    async def _riscrivi(self, question: str, history: list[str] | None) -> str | None:
        """La riscrittura, o None se il modello non risponde o risponde male."""
        coda = "\n".join(history[-3:]) if history else ""
        try:
            risposta = await self.llm.complete(
                [
                    Message(role="system", content=self.prompt),
                    Message(role="user", content=f"{coda}\n\nDomanda: {question}".strip()),
                ],
                max_tokens=80,
            )
        except Exception:
            log.warning("rewriter.fallito", extra={"domanda": question})
            return None

        riscritta = risposta.content.strip().strip('"')
        # una riscrittura vuota, o molto più lunga della domanda, è una risposta
        # travestita: si scarta, e il recupero prosegue sull'originale
        if not riscritta or len(riscritta) > 4 * len(question) + 80:
            log.warning("rewriter.scartata", extra={"uscita": riscritta[:120]})
            return None
        return riscritta

    async def rewrite(self, question: str, history: list[str] | None = None) -> str:
        """Riformula la domanda; se non ci riesce, ritorna l'originale.

        Una riscrittura mancata non deve impedire una risposta.
        """
        if not self.enabled:
            return question
        return await self._riscrivi(question, history) or question

    async def rewrite_cached(self, question: str) -> str:
        """Come rewrite, ma la stessa domanda si riscrive una volta sola."""
        if not self.enabled:
            return question
        key = f"rewrite:{hashlib.sha256(question.encode()).hexdigest()}"
        if (cached := self._cache.get(key)) is not None:
            return cached
        riscritta = await self._riscrivi(question, None)
        if riscritta is None:
            return question  # il ripiego non va in cache: la prossima volta si riprova
        self._cache[key] = riscritta
        return riscritta
