# scripts/seed_accounts.py — i clienti, i conti e i movimenti della banca, per sviluppo e prove
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from src.db.models import Account, Customer, Movement
from src.db.session import AsyncSessionLocal

CLIENTI = [  # (codice cliente, nome, operatore che lo ha in portafoglio)
    ("C-10234", "Paolo Ferri", "mbianchi"),
    ("C-20417", "Anna Greco", "pgalli"),  # un collega di un'altra filiale
]
CONTI = [  # (iban, codice cliente, etichetta, saldo)
    ("IT60X0542811101000000123", "C-10234", "principale", Decimal("48200.00")),
    ("IT60X0542811101000000456", "C-10234", "risparmio", Decimal("1200.00")),
    ("IT60X0542811101000000789", "C-20417", "principale", Decimal("15300.00")),
]
PRINCIPALE, RISPARMIO, ANNA = (c[0] for c in CONTI)
MOVIMENTI = [  # (iban, giorni fa, descrizione, importo): negativo = uscita
    (PRINCIPALE, 3, "Bonifico estero verso Panama", Decimal("-9800.00")),
    (PRINCIPALE, 11, "Bonifico estero verso Panama", Decimal("-9500.00")),
    (PRINCIPALE, 15, "Accredito stipendio", Decimal("2850.00")),
    (RISPARMIO, 20, "Giroconto da principale", Decimal("500.00")),
    (PRINCIPALE, 45, "Accredito stipendio", Decimal("2850.00")),
    (PRINCIPALE, 75, "Accredito stipendio", Decimal("2850.00")),
    (PRINCIPALE, 5, "Esselunga Milano", Decimal("-87.40")),
    (PRINCIPALE, 12, "Esselunga Milano", Decimal("-64.15")),
    (PRINCIPALE, 26, "Esselunga Milano", Decimal("-102.33")),
    (PRINCIPALE, 40, "Coop Lombardia", Decimal("-45.90")),
    (PRINCIPALE, 58, "Esselunga Milano", Decimal("-71.08")),
    (PRINCIPALE, 83, "Coop Lombardia", Decimal("-38.60")),
    (PRINCIPALE, 8, "Bolletta Enel Energia", Decimal("-87.40")),
    (PRINCIPALE, 38, "Bolletta Enel Energia", Decimal("-92.15")),
    (PRINCIPALE, 68, "Bolletta Enel Energia", Decimal("-79.99")),
    (PRINCIPALE, 18, "Fastweb abbonamento", Decimal("-29.95")),
    (PRINCIPALE, 48, "Fastweb abbonamento", Decimal("-29.95")),
    (PRINCIPALE, 78, "Fastweb abbonamento", Decimal("-29.95")),
    (PRINCIPALE, 9, "Trenitalia Milano-Roma", Decimal("-89.90")),
    (PRINCIPALE, 33, "ATM abbonamento mensile", Decimal("-39.00")),
    (PRINCIPALE, 63, "ATM abbonamento mensile", Decimal("-39.00")),
    (PRINCIPALE, 6, "Pizzeria da Gino", Decimal("-32.50")),
    (PRINCIPALE, 29, "Trattoria Milanese", Decimal("-58.00")),
    (PRINCIPALE, 52, "Osteria del Ponte", Decimal("-44.70")),
    (PRINCIPALE, 14, "Netflix", Decimal("-12.99")),
    (PRINCIPALE, 44, "Netflix", Decimal("-12.99")),
    (PRINCIPALE, 74, "Netflix", Decimal("-12.99")),
    # le commissioni: tanti 0.10 e 0.20, cioè i valori che in virgola mobile non tornano
    (PRINCIPALE, 2, "Commissione bonifico", Decimal("-0.10")),
    (PRINCIPALE, 4, "Commissione bonifico", Decimal("-0.20")),
    (PRINCIPALE, 10, "Commissione bonifico", Decimal("-0.10")),
    (PRINCIPALE, 16, "Commissione bonifico", Decimal("-0.20")),
    (PRINCIPALE, 22, "Commissione bonifico", Decimal("-0.10")),
    (PRINCIPALE, 34, "Commissione bonifico", Decimal("-0.20")),
    (PRINCIPALE, 46, "Commissione bonifico", Decimal("-0.10")),
    (PRINCIPALE, 56, "Commissione bonifico", Decimal("-0.20")),
    (PRINCIPALE, 66, "Commissione bonifico", Decimal("-0.10")),
    (RISPARMIO, 50, "Giroconto da principale", Decimal("500.00")),
    (RISPARMIO, 80, "Interessi trimestrali", Decimal("0.37")),
    (ANNA, 7, "Conad Bergamo", Decimal("-54.20")),
    (ANNA, 30, "Accredito pensione", Decimal("1640.00")),
]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        if await session.scalar(select(Customer.id).limit(1)) is not None:
            print("Clienti già presenti: nessuna azione.")
            return
        # Un flush per gruppo, nell'ordine delle chiavi esterne: i modelli non dichiarano
        # relationship(), e senza la sessione non sa che un conto va scritto dopo il suo cliente
        session.add_all(Customer(id=c, full_name=n, operator=o) for c, n, o in CLIENTI)
        await session.flush()
        session.add_all(Account(id=i, customer_id=c, label=e, balance=s) for i, c, e, s in CONTI)
        await session.flush()
        oggi = date.today()
        session.add_all(
            Movement(account_id=i, booking_date=oggi - timedelta(days=g), description=d, amount=a)
            for i, g, d, a in MOVIMENTI
        )
        await session.commit()
    print(f"{len(CLIENTI)} clienti, {len(CONTI)} conti e {len(MOVIMENTI)} movimenti creati.")


if __name__ == "__main__":
    asyncio.run(main())
