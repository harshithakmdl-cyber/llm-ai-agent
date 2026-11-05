"""LLM-powered AI Agent (free API).
Loop:  user message -> LLM decides -> (calls tools) -> sees results -> final answer.
Uses any OpenAI-compatible API (Groq / Gemini / OpenRouter), set in the .env file."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import FUNCTIONS, TOOL_SPECS

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful personal assistant agent. You have tools for maths, date/time, "
    "a to-do list, notes and Wikipedia. Use a tool whenever it gives a more reliable "
    "answer than guessing (always use the calculator for maths, Wikipedia for facts). "
    "Keep answers short and simple."
)
MAX_STEPS = 6  # safety limit so the agent can never loop forever


class Agent:
    def __init__(self, client=None, model=None):
        self.client = client or OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # memory

    def run_tool(self, name, args):
        func = FUNCTIONS.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'."
        try:
            return str(func(**args))
        except TypeError:
            return "Error: wrong arguments for this tool."

    def chat(self, user_text, verbose=True):
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_STEPS):
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages,
                tools=TOOL_SPECS, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:                       # model gave the final answer
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content
            self.messages.append({                       # model wants to use tools
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in msg.tool_calls]})
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.run_tool(call.function.name, args)
                if verbose:
                    print(f"  [tool] {call.function.name}({args}) -> {result[:80]}")
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "Sorry, I could not finish that request. Please try again."


if __name__ == "__main__":
    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("Missing LLM_API_KEY. Copy .env.example to .env and add your free key.")
    agent = Agent()
    print("AI Agent ready! Type 'exit' to quit.")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit", "bye"):
            break
        if text:
            try:
                print("Agent:", agent.chat(text))
            except Exception as e:  # network / key / rate-limit problems
                print("Agent: Something went wrong:", e)
