# src/services/account_service.py — i conti del portafoglio: i movimenti e la spesa di un periodo
import datetime as dt
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repos import AccountRepository, MovementRepository
from src.exceptions import AccountNotFoundError, InvalidPeriodError
from src.types.accounts import MonthlySpending, MovementOut, Spending


class AccountService:
    def __init__(self, session: AsyncSession) -> None:
        self.accounts = AccountRepository(session)
        self.movements = MovementRepository(session)

    async def _verifica(self, username: str, account_id: str, dal: dt.date, al: dt.date) -> None:
        if dal > al:
            raise InvalidPeriodError()
        if await self.accounts.of_user(username, account_id) is None:
            raise AccountNotFoundError(account_id)

    async def movements_with_owner(
        self, username: str, account_id: str, dal: dt.date, al: dt.date
    ) -> list[MovementOut]:
        await self._verifica(username, account_id, dal, al)
        righe = await self.movements.with_owner(username, account_id, dal, al)
        return [
            MovementOut(
                booking_date=m.booking_date,
                description=m.description,
                amount=m.amount,
                owner=intestatario,
            )
            for m, intestatario in righe
        ]

    async def spending(self, username: str, account_id: str, dal: dt.date, al: dt.date) -> Spending:
        await self._verifica(username, account_id, dal, al)
        mesi = await self.movements.spent_by_month(username, account_id, dal, al)
        return Spending(
            account_id=account_id,
            date_from=dal,
            date_to=al,
            months=[MonthlySpending(month=m, spent=s) for m, s in mesi],
            total=sum((s for _, s in mesi), Decimal("0")),  # Decimal su Decimal: al centesimo
        )
