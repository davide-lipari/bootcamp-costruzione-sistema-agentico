# src/llm/prompt.py — i prompt sono file versionati in src/llm/prompts/
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
