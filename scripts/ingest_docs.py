# scripts/ingest_docs.py — il corpus in data/docs/, con il livello preso dalla sottocartella
import asyncio
import os
from pathlib import Path

import httpx

API = "http://localhost:8000"
# da oggi l'ingestione vuole un token di chi può scrivere l'indice: Giulia, della compliance.
# Sono le credenziali del seed di sviluppo; in un ambiente vero arrivano dall'ambiente.
UTENTE = os.environ.get("INGEST_USER", "grossi")
PASSWORD = os.environ.get("INGEST_PASSWORD", "bootcamp")


async def main() -> None:
    docs_dir = Path("data/docs")
    percorsi = await asyncio.to_thread(lambda: sorted(docs_dir.rglob("*.md")))
    async with httpx.AsyncClient(timeout=60.0) as client:
        login = await client.post(
            f"{API}/api/auth/login", data={"username": UTENTE, "password": PASSWORD}
        )
        login.raise_for_status()  # senza token non si va avanti: meglio fermarsi qui
        token = login.json()["access_token"]
        for path in percorsi:
            # leggere un file è bloccante: dentro una coroutine si sposta su un thread
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            visibility = path.parent.name if path.parent != docs_dir else "public"
            response = await client.post(
                f"{API}/api/ai/documents/ingest",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "document_id": path.stem,
                    "content": content,
                    "visibility": visibility,
                    "metadata": {"source": str(path), "title": path.stem.replace("_", " ")},
                },
            )
            print(f"{path.name} [{visibility}]: {response.status_code} {response.json()}")


if __name__ == "__main__":
    asyncio.run(main())
