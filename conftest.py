# conftest.py — nella radice: l'ambiente dei test, prima che qualcosa importi src
import os
from collections.abc import Iterator

import pytest
from alembic.config import Config

from alembic import command

# Dal Giorno 3 i test parlano con un database vero: quello di test, mai quello di sviluppo.
# Qui non c'è setdefault: un DATABASE_URL rimasto nell'ambiente porterebbe i test a svuotare
# il database su cui lavori.
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://lipari:lipari@localhost:5432/lipari_ai_test"
)
os.environ.setdefault("JWT_SECRET", "segreto-dei-test-lungo-almeno-trentadue-caratteri")
# Dal Giorno 9: chiavi finte, anche qui senza setdefault. La tua chiave vera sta nel .env, e un test
# che dimentica il suo finto la userebbe: con questa riceve un 401, che si vede, invece di una
# fattura, che no. Deve stare qui in cima, prima che l'import di src costruisca Settings.
os.environ["OPENAI_API_KEY"] = "sk-test-mai-valida"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-mai-valida"


@pytest.fixture(scope="session", autouse=True)
def schema() -> Iterator[None]:
    """Lo schema del database di test, dalle migration e da zero: come lo avrà chi clona."""
    cfg = Config("alembic.ini")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield


@pytest.fixture(scope="session", autouse=True)
def modelli_finti() -> Iterator[None]:
    """Dal Giorno 4 nessun test chiama un modello vero: costerebbe, e risponderebbe ogni volta
    in un modo diverso. La chat parla con un finto che ripete la domanda, la categorizzazione
    usa la regola del Giorno 2, gli embedding sono finti ma coerenti (dal Giorno 5). E dal Giorno
    6 chi chiede è Marco, operatore, senza token: i test di autenticazione rimettono il vero.
    Un test che vuole un modello preciso, o un altro utente, mette il suo."""
    from src.api.advice import get_rewriter
    from src.api.categorize import get_categorize_service
    from src.auth.deps import get_current_user
    from src.llm.factory import get_embedder, get_llm_provider
    from src.main import app
    from tests.finti import (
        MARCO,
        CategorizzazioneARegole,
        ModelloEco,
        embedder_finto,
        riscrittore_spento,
    )

    app.dependency_overrides[get_llm_provider] = ModelloEco
    app.dependency_overrides[get_categorize_service] = CategorizzazioneARegole
    app.dependency_overrides[get_embedder] = embedder_finto
    app.dependency_overrides[get_rewriter] = riscrittore_spento  # dal Giorno 6
    app.dependency_overrides[get_current_user] = lambda: MARCO  # dal Giorno 6: chi chiede
    yield
    app.dependency_overrides.clear()
