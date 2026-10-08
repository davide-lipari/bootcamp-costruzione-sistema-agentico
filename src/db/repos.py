# src/db/repos.py — l'unico posto che sa che esiste un database
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import Account, ChatMessage, ChatSession, Customer, Movement


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, user_id: str) -> ChatSession:
        chat = ChatSession(user_id=user_id)
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def find_session(self, session_id: str, user_id: str) -> ChatSession | None:
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        cost_eur: Decimal = Decimal("0"),
        model_used: str | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tokens=tokens,
            cost_eur=cost_eur,
            model_used=model_used,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def list_messages(self, session_id: str) -> list[ChatMessage]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---- le due forme di interrogazione: la JOIN e l'aggregazione

    async def messaggi_con_sessione(
        self, user_id: str, giorni: int = 7
    ) -> list[tuple[ChatMessage, datetime]]:
        da = datetime.now(UTC) - timedelta(days=giorni)
        stmt = (
            select(ChatMessage, ChatSession.started_at)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)  # ← la JOIN
            .where(ChatSession.user_id == user_id, ChatMessage.created_at >= da)
            .order_by(ChatMessage.created_at.desc())
        )
        return list((await self.session.execute(stmt)).all())

    async def costo_per_sessione(self, user_id: str) -> list[tuple[str, int, Decimal | None]]:
        stmt = (
            select(
                ChatSession.id,
                func.count(ChatMessage.id).label("messaggi"),
                func.sum(ChatMessage.cost_eur).label("costo"),
            )
            .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id)  # ← il GROUP BY
            .order_by(func.sum(ChatMessage.cost_eur).desc())
        )
        return list((await self.session.execute(stmt)).all())

    async def cost_between(self, inizio: datetime, fine: datetime) -> Decimal:
        """Quanto sono costati i messaggi scritti in un intervallo: la somma, o zero."""
        speso = await self.session.scalar(
            select(func.coalesce(func.sum(ChatMessage.cost_eur), 0)).where(
                ChatMessage.created_at >= inizio, ChatMessage.created_at < fine
            )
        )
        return Decimal(speso or 0)


# ---- conti e movimenti: si leggono sempre filtrando per il portafoglio di chi chiede


class AccountRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def of_user(self, username: str, account_id: str) -> Account | None:
        """Il conto, solo se è di un cliente nel portafoglio di questo operatore."""
        return await self.session.scalar(
            select(Account)
            .join(Customer, Customer.id == Account.customer_id)
            .where(Account.id == account_id, Customer.operator == username)
        )

    async def ids_in_portfolio(self, username: str) -> set[str]:
        """Gli IBAN dei conti dei clienti nel portafoglio di questo operatore."""
        stmt = (
            select(Account.id)
            .join(Customer, Customer.id == Account.customer_id)
            .where(Customer.operator == username)
        )
        return set((await self.session.scalars(stmt)).all())

    async def of_customer(self, username: str, customer_id: str) -> list[Account]:
        """I conti di un cliente, solo se il cliente è nel portafoglio di questo operatore."""
        stmt = (
            select(Account)
            .join(Customer, Customer.id == Account.customer_id)
            .where(Customer.id == customer_id, Customer.operator == username)
            .order_by(Account.label)
        )
        return list((await self.session.scalars(stmt)).all())


class MovementRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def recent(self, username: str, account_id: str, limite: int) -> list[Movement]:
        """Gli ultimi movimenti di un conto del portafoglio, dal più recente."""
        stmt = (
            select(Movement)
            .join(Account, Account.id == Movement.account_id)
            .join(Customer, Customer.id == Account.customer_id)
            .where(Account.id == account_id, Customer.operator == username)
            .order_by(Movement.booking_date.desc())
            .limit(limite)
        )
        return list((await self.session.scalars(stmt)).all())

    async def with_owner(
        self, username: str, account_id: str, dal: date, al: date
    ) -> list[tuple[Movement, str]]:
        """I movimenti di un conto in un periodo, ognuno con il nome dell'intestatario."""
        stmt = (
            select(Movement, Customer.full_name)
            .join(Account, Account.id == Movement.account_id)  # il movimento, al suo conto
            .join(Customer, Customer.id == Account.customer_id)  # il conto, al suo cliente
            .where(
                Account.id == account_id,
                Customer.operator == username,
                Movement.booking_date.between(dal, al),
            )
            .order_by(Movement.booking_date, Movement.id)
        )
        return list((await self.session.execute(stmt)).all())

    async def spent_by_month(
        self, username: str, account_id: str, dal: date, al: date
    ) -> list[tuple[str, Decimal]]:
        """Quanto è uscito dal conto, mese per mese: la somma delle uscite, cambiata di segno."""
        mese = func.to_char(Movement.booking_date, "YYYY-MM")
        stmt = (
            select(mese, func.sum(-Movement.amount))
            .join(Account, Account.id == Movement.account_id)
            .join(Customer, Customer.id == Account.customer_id)
            .where(
                Account.id == account_id,
                Customer.operator == username,
                Movement.booking_date.between(dal, al),
                Movement.amount < 0,  # solo le uscite: gli accrediti non sono spesa
            )
            .group_by(mese)
            .order_by(mese)
        )
        return list((await self.session.execute(stmt)).all())
