# src/exceptions.py — gli errori di dominio, e la natura di ognuno
class AppError(Exception):
    """Un errore previsto. Porta il suo codice di stato: lo decide chi conosce il guasto."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class NotFoundError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(404, "NOT_FOUND", message)


class ChatSessionNotFoundError(NotFoundError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Sessione {session_id} non trovata")
        self.code = "CHAT_SESSION_NOT_FOUND"


class RateLimitError(AppError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            429, "RATE_LIMIT", f"Limite raggiunto. Riprova fra {retry_after_seconds} secondi"
        )
        self.retry_after = retry_after_seconds


class LLMProviderError(AppError):
    def __init__(self, provider: str, original: str) -> None:
        super().__init__(502, "LLM_PROVIDER_ERROR", f"Il fornitore {provider} non risponde")
        self.original = original  # per il log, non per il client


class InvalidFileError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(422, "INVALID_FILE", message)


class AccountNotFoundError(NotFoundError):
    def __init__(self, account_id: str) -> None:
        # «non trovato» anche quando il conto esiste ma è di un altro portafoglio:
        # la risposta non deve dire a chi chiede quali conti esistono
        super().__init__(f"Il conto {account_id} non è nel tuo portafoglio")
        self.code = "ACCOUNT_NOT_FOUND"


class InvalidPeriodError(AppError):
    def __init__(self) -> None:
        super().__init__(422, "INVALID_PERIOD", "La data di inizio viene dopo quella di fine")
