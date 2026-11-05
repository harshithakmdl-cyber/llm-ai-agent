"""Tools the agent can use. Each tool is a normal Python function.
TOOL_SPECS describes them to the AI model so it knows when to call each one."""
import ast
import json
import operator
import os
from datetime import datetime

import requests

DATA_DIR = "data"
TODO_FILE = os.path.join(DATA_DIR, "todos.json")
NOTES_FILE = os.path.join(DATA_DIR, "notes.json")
os.makedirs(DATA_DIR, exist_ok=True)


def _load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


# ---- calculator (safe: never uses eval) ----
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 6))
    except Exception:
        return "Error: could not calculate that expression."


def get_datetime() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


# ---- to-do list ----
def add_todo(task: str) -> str:
    items = _load(TODO_FILE)
    items.append(task)
    _save(TODO_FILE, items)
    return f"Added task #{len(items)}: {task}"


def list_todos() -> str:
    items = _load(TODO_FILE)
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)) or "The to-do list is empty."


def complete_todo(number: int) -> str:
    items = _load(TODO_FILE)
    try:
        done = items.pop(int(number) - 1)
    except (ValueError, IndexError):
        return "Error: no task with that number."
    _save(TODO_FILE, items)
    return f"Completed: {done}"


# ---- notes ----
def save_note(text: str) -> str:
    notes = _load(NOTES_FILE)
    notes.append({"time": get_datetime(), "text": text})
    _save(NOTES_FILE, notes)
    return "Note saved."


def read_notes() -> str:
    notes = _load(NOTES_FILE)
    return "\n".join(f"- [{n['time']}] {n['text']}" for n in notes) or "No notes saved yet."


# ---- Wikipedia (free, no key needed) ----
def wikipedia_summary(topic: str) -> str:
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(topic.replace(" ", "_")),
            headers={"User-Agent": "llm-ai-agent/1.0"}, timeout=10)
        if r.status_code != 200:
            return f"No Wikipedia page found for '{topic}'."
        return r.json().get("extract", "No summary available.")
    except requests.RequestException:
        return "Error: could not reach Wikipedia."


FUNCTIONS = {
    "calculator": calculator, "get_datetime": get_datetime, "add_todo": add_todo,
    "list_todos": list_todos, "complete_todo": complete_todo, "save_note": save_note,
    "read_notes": read_notes, "wikipedia_summary": wikipedia_summary,
}


def _spec(name, desc, props=None):
    props = props or {}
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(props)}}}


TOOL_SPECS = [
    _spec("calculator", "Do exact maths. Use for ANY calculation.",
          {"expression": {"type": "string", "description": "e.g. '12*(3+4)/5'"}}),
    _spec("get_datetime", "Get the current date and time."),
    _spec("add_todo", "Add a task to the user's to-do list.", {"task": {"type": "string"}}),
    _spec("list_todos", "Show all tasks in the to-do list."),
    _spec("complete_todo", "Mark a to-do task done by its number.", {"number": {"type": "integer"}}),
    _spec("save_note", "Save a note for the user.", {"text": {"type": "string"}}),
    _spec("read_notes", "Read all saved notes."),
    _spec("wikipedia_summary", "Look up a factual summary of a topic on Wikipedia.",
          {"topic": {"type": "string"}}),
]
