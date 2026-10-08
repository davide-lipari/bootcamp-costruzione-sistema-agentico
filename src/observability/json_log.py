# src/observability/json_log.py — righe di log che una macchina sa interrogare
import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

# l'identificativo della richiesta in corso: lo imposta il middleware, lo legge il formatter,
# e nessuna funzione in mezzo lo deve passare come parametro
request_id: ContextVar[str] = ContextVar("request_id", default="-")

CAMPI_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        riga: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id.get(),
        }
        # i campi di extra=: tutto ciò che il LogRecord standard non ha
        riga.update({k: v for k, v in record.__dict__.items() if k not in CAMPI_STANDARD})
        if record.exc_info:  # logger.exception: il traceback c'è, in un campo suo
            riga["traceback"] = self.formatException(record.exc_info)
        # default=str: un Decimal o un datetime fra i campi non fa cadere il logging
        return json.dumps(riga, default=str, ensure_ascii=False)


def configura_log(livello: int = logging.INFO) -> None:
    """Un handler solo, sul logger radice, con le righe in JSON. Si chiama una volta, all'avvio."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    radice = logging.getLogger()
    radice.handlers = [handler]
    radice.setLevel(livello)
