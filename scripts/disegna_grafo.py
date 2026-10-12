# scripts/disegna_grafo.py — il grafo del repository, disegnato dal codice che lo costruisce
#
# Non è un disegno fatto a mano: nodi e collegamenti vengono dal grafo compilato, quindi non
# può andarsi a guardare. In Mermaid, in docs/grafo.md:  uv run python -m scripts.disegna_grafo
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.agents.deps import Deps
from src.auth.deps import UserContext
from src.lg.grafo import PERCORSI, costruisci_grafo

# solo la forma: nessun nodo gira, e a costruire il grafo serve soltanto un LLM costruibile
SERVIZI_FINTI = cast(Deps, MagicMock())
CHI_CHIEDE = UserContext(username="mbianchi", role="operator")
LLM = ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr("non-usata"), base_url="http://finto/v1")


def disegna() -> str:
    """Il grafo compilato, in Mermaid: dal codice che lo costruisce, con la tabella dei percorsi."""
    g = costruisci_grafo(CHI_CHIEDE, SERVIZI_FINTI, LLM, LLM, None)
    mermaid = g.get_graph().draw_mermaid()
    tabella = "\n".join(f"| `{k}` | {' → '.join(v)} |" for k, v in PERCORSI.items())
    return (
        "# Il grafo del supervisor\n\n"
        "Generato da `scripts/disegna_grafo.py` dal grafo compilato: non è un disegno a mano.\n\n"
        "```mermaid\n"
        f"{mermaid}\n"
        "```\n\n"
        "## I percorsi\n\n"
        "| instradamento | nodi |\n|---|---|\n"
        f"{tabella}\n"
    )


def main() -> None:
    Path("docs").mkdir(exist_ok=True)  # la cartella della documentazione generata
    Path("docs/grafo.md").write_text(disegna(), encoding="utf-8")
    print("docs/grafo.md scritto")


if __name__ == "__main__":
    main()
