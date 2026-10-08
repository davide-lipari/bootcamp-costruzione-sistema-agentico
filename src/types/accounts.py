# src/types/accounts.py — i movimenti di un conto, e la spesa per mese
import datetime as dt
from decimal import Decimal

from pydantic import BaseModel


class MovementOut(BaseModel):
    booking_date: dt.date
    description: str
    amount: Decimal  # nel JSON esce come stringa, "-87.40": un importo non passa da un float
    owner: str  # l'intestatario del conto, dalla JOIN


class MonthlySpending(BaseModel):
    month: str  # "2026-07"
    spent: Decimal


class Spending(BaseModel):
    account_id: str
    date_from: dt.date
    date_to: dt.date
    months: list[MonthlySpending]
    total: Decimal
