"""Offline test: a fake LLM pretends to call tools, so no API key is needed.
Run:  python test_agent.py"""
import json
from types import SimpleNamespace as NS

from agent import Agent


def tool_call(id_, name, args):
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class FakeClient:
    def __init__(self, script):
        self.script = iter(script)
        self.chat = NS(completions=NS(create=lambda **kw: next(self.script)))


def test_tool_flow():
    fake = FakeClient([reply(calls=[tool_call("1", "calculator", {"expression": "12*(3+4)"})]),
                       reply("The answer is 84.")])
    agent = Agent(client=fake, model="fake")
    assert agent.chat("What is 12*(3+4)?", verbose=False) == "The answer is 84."
    assert any(m["role"] == "tool" and m["content"] == "84" for m in agent.messages)


def test_todo_tools():
    fake = FakeClient([reply(calls=[tool_call("1", "add_todo", {"task": "test task"})]),
                       reply("Added.")])
    agent = Agent(client=fake, model="fake")
    agent.chat("add test task", verbose=False)
    assert "test task" in agent.run_tool("list_todos", {})


def test_unknown_tool_and_bad_args():
    agent = Agent(client=FakeClient([]), model="fake")
    assert "unknown tool" in agent.run_tool("hack", {})
    assert "wrong arguments" in agent.run_tool("calculator", {"wrong": 1})


def test_step_limit():
    loop = [reply(calls=[tool_call(str(i), "get_datetime", {})]) for i in range(10)]
    agent = Agent(client=FakeClient(loop), model="fake")
    assert "could not finish" in agent.chat("loop forever", verbose=False)


if __name__ == "__main__":
    for t in (test_tool_flow, test_todo_tools, test_unknown_tool_and_bad_args, test_step_limit):
        t()
        print("PASS", t.__name__)
