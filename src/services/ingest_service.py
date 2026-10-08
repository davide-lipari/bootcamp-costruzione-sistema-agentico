# src/services/ingest_service.py — dal documento ai passaggi nell'indice
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentChunk
from src.lib.chunking import chunk_text
from src.llm.embedding_client import EmbeddingClient


def senza_ripetizioni(testo: str) -> str:
    """Un paragrafo ripetuto, come una pagina incollata due volte, resta una volta sola.

    Prima del taglio, non dopo: con la sovrapposizione i pezzi di due pagine uguali cominciano in
    punti diversi, e non sono mai identici.
    """
    paragrafi = [p.strip() for p in testo.split("\n\n") if p.strip()]
    return "\n\n".join(dict.fromkeys(paragrafi))


class IngestService:
    """Porta un documento dentro l'indice: taglia, vettorizza, salva."""

    LOTTO = 64  # quanti passaggi per chiamata, sotto il tetto di token del modello

    def __init__(self, session: AsyncSession, embedder: EmbeddingClient) -> None:
        self.session = session
        self.embedder = embedder

    async def ingest(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
        visibility: str = "public",
    ) -> int:
        """Taglia, vettorizza e salva. Ritorna quanti passaggi sono stati scritti.

        Ri-ingerire lo stesso `document_id` sostituisce: i passaggi precedenti vengono
        cancellati prima, nella stessa transazione.
        """
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

        pezzi = chunk_text(senza_ripetizioni(text))
        if not pezzi:
            await self.session.commit()
            return 0

        for inizio in range(0, len(pezzi), self.LOTTO):
            lotto = pezzi[inizio : inizio + self.LOTTO]
            vettori = await self.embedder.embed(lotto)
            self.session.add_all(
                [
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=inizio + i,
                        content=testo,
                        embedding=vettore,
                        chunk_metadata=metadata or {},
                        visibility=visibility,  # ogni passaggio eredita il livello del documento
                    )
                    for i, (testo, vettore) in enumerate(zip(lotto, vettori, strict=True))
                ]
            )

        await self.session.commit()  # o il documento è nell'indice tutto, o non c'è
        return len(pezzi)
