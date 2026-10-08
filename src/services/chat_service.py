# src/services/chat_service.py — la conversazione, con un modello che risponde e uno strumento
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ChatSession
from src.db.repos import AccountRepository, ChatRepository
from src.exceptions import AccountNotFoundError, ChatSessionNotFoundError, LLMProviderError
from src.llm.client import LLMProvider
from src.llm.tools import TOOLS_SCHEMA, costruisci_strumenti
from src.llm.types import LLMResponse, Message
from src.observability.cost_tracker import CostTracker
from src.observability.ledger import CostLedger
from src.types.chat import ChatRequest, ChatResponse, ToolCallInfo

FINESTRA = 20  # quanti messaggi di storia entrano nel prompt
GIRI_MASSIMI = 3  # le chiamate al modello concesse a una domanda: di solito ne bastano due
ARRESO = "Non sono riuscito a completare la richiesta con gli strumenti che ho: riprova più tardi."


def _argomenti(testo: str) -> dict[str, str]:
    """Gli argomenti di una chiamata, per la risposta: il JSON del modello, se è un oggetto."""
    try:
        valori = json.loads(testo)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in valori.items()} if isinstance(valori, dict) else {}


class ChatService:
    def __init__(
        self, session: AsyncSession, llm: LLMProvider, system_prompt: str, tetto_eur: Decimal
    ) -> None:
        self.session = session
        self.repo = ChatRepository(session)
        self.accounts = AccountRepository(session)
        self.costi = CostTracker(self.repo, tetto_eur)
        self.llm = llm
        self.system_prompt = system_prompt

    async def _conversazione(
        self, req: ChatRequest, user_id: str
    ) -> tuple[ChatSession, list[Message]]:
        """La conversazione di chi chiede, e i messaggi da mandare al modello."""
        storia: list[Message] = []
        if req.session_id != "new":
            chat = await self.repo.find_session(req.session_id, user_id)
            if chat is None:
                raise ChatSessionNotFoundError(req.session_id)
            # la storia si legge PRIMA di aggiungere il messaggio di adesso: altrimenti
            # la domanda dell'utente arriva al modello due volte
            storia = [Message(role=m.role, content=m.content) for m in chat.messages[-FINESTRA:]]
        else:
            chat = await self.repo.create_session(user_id=user_id)
        return chat, [
            Message(role="system", content=self.system_prompt),
            *storia,
            Message(role="user", content=req.message),
        ]

    async def _salva(self, chat: ChatSession, domanda: str, risposta: LLMResponse) -> None:
        await self.repo.add_message(chat.id, "user", domanda)
        await self.repo.add_message(
            chat.id,
            "assistant",
            risposta.content,
            tokens=risposta.tokens_used,
            cost_eur=risposta.cost_eur,
            model_used=risposta.model,
        )
        CostLedger(self.session).aggiungi(  # Giorno 9: il costo anche nel registro
            endpoint="chat",
            username=chat.user_id,
            model=risposta.model,
            tokens=risposta.tokens_used,
            cost_eur=risposta.cost_eur,
        )
        await self.session.commit()  # una volta, a lavoro finito: le tre scritture valgono insieme

    async def _saldo(self, user_id: str, account_id: str) -> Decimal:
        conto = await self.accounts.of_user(user_id, account_id)
        if conto is None:  # lo strumento ha già controllato: qui è una seconda difesa
            raise AccountNotFoundError(account_id)
        return conto.balance

    async def chat(self, req: ChatRequest, user_id: str) -> ChatResponse:
        """Una risposta del modello, con gli strumenti che chiede, il costo di tutti i giri."""
        await self.costi.verifica()  # prima di spendere: dopo, la spesa è già fatta
        chat, messaggi = await self._conversazione(req, user_id)

        # gli strumenti di QUESTA richiesta, con i conti del portafoglio letti una volta sola
        conti = await self.accounts.ids_in_portfolio(user_id)

        async def leggi_saldo(account_id: str) -> Decimal:
            return await self._saldo(user_id, account_id)

        strumenti = costruisci_strumenti(user_id, conti, leggi_saldo)

        risposta = await self.llm.complete(messaggi, tools=TOOLS_SCHEMA)
        token, costo = risposta.tokens_used, risposta.cost_eur
        eseguite: list[ToolCallInfo] = []
        giri = 1
        while risposta.tool_calls and giri < GIRI_MASSIMI:
            # l'assistente che chiede rientra nella conversazione PRIMA dei risultati
            messaggi.append(
                Message(role="assistant", content=risposta.content, tool_calls=risposta.tool_calls)
            )
            for richiesta in risposta.tool_calls:  # una risposta a ogni richiesta del giro
                esegui = strumenti.get(richiesta.name)
                esito = (
                    await esegui(richiesta.arguments)
                    if esegui
                    else f"ERRORE: lo strumento {richiesta.name} non esiste."
                )
                messaggi.append(Message(role="tool", content=esito, tool_call_id=richiesta.id))
                eseguite.append(
                    ToolCallInfo(
                        name=richiesta.name, arguments=_argomenti(richiesta.arguments), result=esito
                    )
                )
            risposta = await self.llm.complete(messaggi, tools=TOOLS_SCHEMA)
            token, costo = token + risposta.tokens_used, costo + risposta.cost_eur
            giri += 1

        # la risposta salvata porta il costo di TUTTI i giri, non solo dell'ultimo
        finale = risposta.model_copy(
            update={
                "content": ARRESO if risposta.tool_calls else risposta.content,
                "tokens_used": token,
                "cost_eur": costo,
            }
        )
        await self._salva(chat, req.message, finale)
        return ChatResponse(
            session_id=chat.id,
            reply=finale.content,
            tool_calls=eseguite,
            tokens_used=finale.tokens_used,
            cost_eur=float(finale.cost_eur),  # nella risposta JSON il costo è un numero
            model_used=finale.model,
            created_at=datetime.now(UTC),
        )

    async def chat_stream(self, req: ChatRequest, user_id: str) -> AsyncIterator[str]:
        """Prepara la conversazione subito, e restituisce i pezzi da mandare al client.

        La preparazione avviene prima che la risposta parta: una sessione che non esiste, o il
        tetto superato, sono ancora un 404 e un 429, non un errore a metà di uno stream già
        cominciato con un 200. Lo streaming non usa strumenti: il giro sta nella risposta intera.
        """
        await self.costi.verifica()
        chat, messaggi = await self._conversazione(req, user_id)
        return self._pezzi(chat, req.message, messaggi)

    async def _pezzi(
        self, chat: ChatSession, domanda: str, messaggi: list[Message]
    ) -> AsyncIterator[str]:
        finale: LLMResponse | None = None
        async for pezzo in self.llm.stream(messaggi):
            if isinstance(pezzo, LLMResponse):
                finale = pezzo  # l'ultimo: il testo intero, i token e il costo
            else:
                yield pezzo
        if finale is None:
            raise LLMProviderError("stream", "lo stream si è chiuso senza la risposta finale")
        await self._salva(chat, domanda, finale)  # dopo il ciclo, una volta sola
