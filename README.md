# LLM AI Agent (free API)

A personal assistant agent powered by a **free LLM API** (Groq by default). The AI model decides by itself when to use tools such as a calculator, to-do list, notes, date/time and Wikipedia.

## How it works
1. You send a message.
2. The LLM decides: answer directly, or call a **tool** (function calling).
3. The tool runs in Python and its result goes back to the LLM.
4. The LLM writes the final answer. A step limit prevents endless loops.
The conversation history is kept, so the agent remembers earlier messages.

## Tools
| Tool | What it does |
|---|---|
| calculator | safe maths (no `eval`) |
| get_datetime | current date and time |
| add_todo / list_todos / complete_todo | to-do list saved in `data/` |
| save_note / read_notes | notes saved in `data/` |
| wikipedia_summary | facts from Wikipedia (no key needed) |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # then open .env and paste your free API key
python agent.py
```
Get a free key: https://console.groq.com (or Gemini / OpenRouter, see `.env.example`).
Never upload your `.env` file to GitHub.

## Test without an API key
```bash
python test_agent.py
```

## Try
- `What is 15% of 2400?`
- `Add submit report to my to-do list`
- `Tell me about the Internet of Things`
- `What day is it today?`

## Tech
Python, OpenAI-compatible API (Groq / Gemini / OpenRouter), function calling, requests

Author: Harshitha N
