# tests/test_g1.py — la configurazione che si rifiuta di partire, e la sonda che dice la verità
import os
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.config import Settings, settings, spiega
from src.main import app

RADICE = Path(__file__).resolve().parents[1]


def avvia_senza(tmp_path: Path, *variabili: str) -> subprocess.CompletedProcess[str]:
    """Importa la configurazione in un processo nuovo, da una cartella senza .env."""
    ambiente = {k: v for k, v in os.environ.items() if k not in variabili}
    ambiente["PYTHONPATH"] = str(RADICE)
    return subprocess.run(
        [sys.executable, "-c", "import src.config"],
        cwd=tmp_path,  # il .env si cerca nella cartella corrente: qui non c'è
        env=ambiente,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_senza_variabili_obbligatorie_il_processo_si_ferma_e_dice_quali(tmp_path):
    r = avvia_senza(tmp_path, "DATABASE_URL", "JWT_SECRET")
    assert r.returncode == 1
    assert "DATABASE_URL: manca" in r.stderr
    assert "JWT_SECRET: manca" in r.stderr
    assert "secrets.token_urlsafe" in r.stderr  # dice anche cosa ci va
    assert "Traceback" not in r.stderr  # e non lo dice in mezzo a un traceback


def test_un_valore_del_tipo_sbagliato_nomina_il_campo():
    with pytest.raises(ValidationError) as e:
        Settings(
            _env_file=None,
            database_url="postgresql+asyncpg://x:y@localhost:5432/z",
            jwt_secret="s" * 40,
            max_tokens_per_request="duemila",
        )
    assert "MAX_TOKENS_PER_REQUEST: valore non valido" in spiega(e.value)


def test_un_segreto_corto_non_basta():
    with pytest.raises(ValidationError) as e:
        Settings(_env_file=None, database_url="postgresql+asyncpg://x", jwt_secret="corto")
    assert "JWT_SECRET: valore non valido" in spiega(e.value)


async def test_la_sonda_dice_quali_chiavi_ci_sono_senza_mostrarle(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-prova-segreta-9876")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["credentials"] == {"openai": True, "anthropic": False}
    assert r.json()["environment"] == settings.environment
    assert "sk-prova" not in r.text and "9876" not in r.text  # né il valore né un suo pezzo
