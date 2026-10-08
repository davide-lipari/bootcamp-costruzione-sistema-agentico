# tests/riavvio_altro_processo.py — la seconda metà del test del riavvio, in un altro processo.
# Un interprete nuovo non ha niente in memoria: se lo stato del run stava in un dizionario,
# qui non c'è più, e l'approvazione risponde 404.
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from tests.banco_app import apri_db, crea_app
from tests.conftest import risposta


async def main(percorso: Path, run_id: str) -> None:
    modello = AsyncMock()
    modello.chat.completions.create.return_value = risposta(testo="Segnalazione aperta.")
    app = crea_app(apri_db(percorso), modello)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.post(f"/api/ai/agent/{run_id}/approve", headers={"X-Utente": "grossi"})
    print(
        json.dumps(
            {
                "status": r.status_code,
                "stopped_by": r.json().get("stopped_by") if r.is_success else None,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
