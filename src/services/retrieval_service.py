# src/services/retrieval_service.py — la ricerca vettoriale sui passaggi dei documenti
from pydantic import BaseModel
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.acl import visible_to

SOGLIA_PREDEFINITA = 0.35  # da tarare sui propri documenti: non è un numero universale


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    similarity: float


class RetrievalService:
    """La ricerca vettoriale sui passaggi dei documenti."""

    SQL = sql(
        """
        SELECT id, document_id, content,
               1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM document_chunks
        WHERE 1 - (embedding <=> CAST(:q AS vector)) >= :soglia
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )

    # la SQL di ieri con UNA riga in più: AND visibility = ANY(:livelli)
    SQL_PER_RUOLO = sql(
        """
        SELECT id, document_id, content,
               1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM document_chunks
        WHERE 1 - (embedding <=> CAST(:q AS vector)) >= :soglia
          AND visibility = ANY(:livelli)                     -- <- la riga
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self, query_vec: list[float], top_k: int = 5, soglia: float = SOGLIA_PREDEFINITA
    ) -> list[RetrievalResult]:
        """Ricerca vettoriale SENZA filtro di visibilità.

        Uso consentito: script di ingestione, job amministrativi, analisi sull'intero corpus.
        NON usare in un percorso che origina da una richiesta HTTP: lì serve
        search_for_user(), che applica l'ACL nel WHERE.

        Prende il **vettore**, non la domanda: vettorizzare è compito di chi chiama, e
        così la stessa ricerca serve l'advisor e gli eval del Giorno 9 senza pagare un
        embedding che qualcuno ha già fatto.

        Sotto la soglia non torna niente: una lista vuota è una risposta, e il chiamante
        deve poterla distinguere da cinque passaggi irrilevanti.
        """
        righe = await self.session.execute(
            self.SQL, {"q": str(query_vec), "k": top_k, "soglia": soglia}
        )
        return [
            RetrievalResult(
                chunk_id=str(r.id),
                document_id=r.document_id,
                content=r.content,
                similarity=float(r.similarity),
            )
            for r in righe
        ]

    async def search_for_user(
        self,
        query_vec: list[float],
        role: str,
        top_k: int = 5,
        soglia: float = SOGLIA_PREDEFINITA,
    ) -> list[RetrievalResult]:
        """I passaggi più vicini FRA QUELLI che questo ruolo può vedere.

        Il filtro sui livelli sta DENTRO la query, non dopo: il database non deve
        nemmeno restituire i passaggi che questo ruolo non può vedere.
        """
        righe = await self.session.execute(
            self.SQL_PER_RUOLO,
            {"q": str(query_vec), "k": top_k, "soglia": soglia, "livelli": visible_to(role)},
        )
        return [
            RetrievalResult(
                chunk_id=str(r.id),
                document_id=r.document_id,
                content=r.content,
                similarity=float(r.similarity),
            )
            for r in righe
        ]
