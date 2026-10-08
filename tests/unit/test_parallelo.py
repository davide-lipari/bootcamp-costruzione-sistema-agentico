import asyncio
import time

from openai.types.chat import ChatCompletionMessageFunctionToolCall
from pydantic import BaseModel

from src.agents.parallelo import execute_all
from src.agents.registry import Tool


class Vuoti(BaseModel):
    pass


def _call(id_: str, nome: str) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall.model_validate(
        {"id": id_, "type": "function", "function": {"name": nome, "arguments": "{}"}}
    )


async def test_chi_legge_insieme_chi_scrive_dopo() -> None:
    eventi: list[str] = []

    def lento(nome: str) -> Tool:
        async def run(_: Vuoti) -> str:
            eventi.append(f"inizio {nome}")
            await asyncio.sleep(0.05)
            eventi.append(f"fine {nome}")
            return nome

        return Tool(nome, nome, Vuoti, run, scrive=(nome == "scrive"))

    tools = [lento("saldo"), lento("movimenti"), lento("scrive")]
    inizio = time.perf_counter()
    esiti = await execute_all(
        {t.name: t for t in tools},
        [_call("1", "scrive"), _call("2", "saldo"), _call("3", "movimenti")],
    )
    durata = time.perf_counter() - inizio

    assert esiti == {"1": "scrive", "2": "saldo", "3": "movimenti"}
    assert eventi[:2] == ["inizio saldo", "inizio movimenti"]  # partono insieme
    assert eventi.index("inizio scrive") > eventi.index("fine movimenti")
    assert durata < 0.14  # due attese, non tre
