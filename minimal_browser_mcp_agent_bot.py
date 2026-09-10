#!/usr/bin/env python3
"""
minimal_browser_mcp_agent_bot.py

Telegram AI agent with explicit memory + browser MCP tool calls.

Behavior:
- /save_memory  -> persists this chat to disk and enables browser MCP tools
- /clear_memory -> wipes this chat from disk/RAM and disables tools
- Before /save_memory: context is RAM-only, no disk write, no MCP tools
- After /clear_memory: nothing is kept

Commands:
  /save_memory or /save
  /clear_memory or /clear
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

try:
    from mcp import ClientSession
except Exception:
    try:
        from mcp.client.session import ClientSession
    except Exception:
        ClientSession = None

try:
    from mcp.client.streamable_http import streamablehttp_client
except Exception:
    try:
        from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    except Exception:
        streamablehttp_client = None

MCP_IMPORT_OK = bool(ClientSession and streamablehttp_client)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("browser-mcp-agent")

# -----------------------------
# Config
# -----------------------------

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Comma or space separated numeric Telegram user IDs.
# Example: "123456789" or "123456789,987654321"
ALLOWED_USER_IDS = {
    int(x)
    for x in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").replace(",", " ").split()
    if x.strip()
}

# OpenAI-compatible LLM config.
# For NVIDIA NIM / other OpenAI-compatible endpoints, set LLM_BASE_URL.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""

# Your browser MCP server URL.
# Example: http://127.0.0.1:8765/mcp
MCP_URL = os.getenv("MCP_URL", "")

MEMORY_FILE = Path(os.getenv("MEMORY_FILE", "agent_memory.json"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are Serge's private operator assistant. Be terse and precise. "
        "Use browser MCP tools only when the user explicitly asks to fetch, read, "
        "open, extract, click, navigate, or otherwise act on browser data."
    ),
)

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "120"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "40"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "12000"))

if not TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN")
if not LLM_API_KEY:
    raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY")
if not LLM_MODEL:
    raise RuntimeError("Set LLM_MODEL")

llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

MCP_ENABLED = bool(MCP_URL and MCP_IMPORT_OK)
if MCP_URL and not MCP_IMPORT_OK:
    log.warning("MCP_URL is set but MCP SDK import failed; MCP tools disabled.")


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def split_text(text: str, limit: int = 4000) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return ["(empty response)"]
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def safe_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "tool"


def message_groups(messages: list[dict]) -> list[list[dict]]:
    """
    Groups assistant tool_calls together with their matching tool responses.
    Prevents truncation from leaving orphan tool messages.
    """
    groups: list[list[dict]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            i += 1
            continue

        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            group = [msg]
            needed = set()

            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("id"):
                    needed.add(tc["id"])

            i += 1

            while i < len(messages) and needed:
                nxt = messages[i]
                if (
                    isinstance(nxt, dict)
                    and nxt.get("role") == "tool"
                    and nxt.get("tool_call_id") in needed
                ):
                    group.append(nxt)
                    needed.discard(nxt.get("tool_call_id"))
                    i += 1
                else:
                    break

            # Keep only complete tool-call/tool-result groups.
            if not needed:
                groups.append(group)
        else:
            # Skip orphan tool messages.
            if role != "tool":
                groups.append([msg])
            i += 1

    return groups


def safe_tail(messages: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    if len(messages) <= limit:
        return messages

    groups = message_groups(messages)
    out: list[dict] = []
    count = 0

    for group in reversed(groups):
        if count + len(group) > limit:
            break
        out = group + out
        count += len(group)

    if not out:
        out = messages[-limit:]

    return out


def sanitize_schema(schema: Any) -> dict:
    """
    Minimal OpenAI-compatible JSON schema sanitizer.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    out = {}
    allowed = {"type", "properties", "required", "items", "enum", "description"}

    for k, v in schema.items():
        if k in allowed:
            out[k] = v

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {
            k: sanitize_schema(v)
            for k, v in out["properties"].items()
            if isinstance(v, dict)
        }

    if "items" in out:
        out["items"] = sanitize_schema(out["items"])

    out.setdefault("type", "object")
    return out


def mcp_tools_to_openai_tools(tools: list) -> tuple[list[dict], dict[str, str]]:
    """
    Converts MCP tools to OpenAI function tools.
    Returns:
      tools, name_map
      name_map maps sanitized OpenAI function name -> original MCP tool name.
    """
    out: list[dict] = []
    name_map: dict[str, str] = {}
    used: set[str] = set()

    for t in tools:
        original_name = getattr(t, "name", None)
        if not original_name:
            continue

        base = safe_tool_name(original_name)
        candidate = base
        suffix = 1

        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"[:64]

        used.add(candidate)
        name_map[candidate] = original_name

        description = getattr(t, "description", "") or ""
        if candidate != original_name:
            description = f"MCP tool: {original_name}\n{description}".strip()

        out.append(
            {
                "type": "function",
                "function": {
                    "name": candidate,
                    "description": description,
                    "parameters": sanitize_schema(getattr(t, "inputSchema", None) or {}),
                },
            }
        )

    return out, name_map


def serialize_mcp_result(result: Any) -> str:
    prefix = "Tool returned error:\n" if getattr(result, "isError", False) else ""
    content = getattr(result, "content", None)

    if content is None:
        return prefix + str(result)

    parts: list[str] = []

    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        elif isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))

    return prefix + ("\n".join(parts) if parts else str(result))


# -----------------------------
# Memory
# -----------------------------

class Memory:
    """
    RAM context for all chats.
    Disk persistence only for chats where active == True.
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                self.data = raw
        except Exception as e:
            log.error("Failed to load memory file: %s", e)
            self.data = {}

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Persist only active chats.
            persistent = {
                chat_id: state
                for chat_id, state in self.data.items()
                if state.get("active")
            }

            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(persistent, indent=1, ensure_ascii=False))
            tmp.replace(self.path)
        except Exception as e:
            log.error("Failed to save memory file: %s", e)

    def chat(self, chat_id: int) -> dict:
        key = str(chat_id)
        if key not in self.data:
            self.data[key] = {"active": False, "messages": []}

        state = self.data[key]
        if "active" not in state:
            state["active"] = False
        if "messages" not in state:
            state["messages"] = []

        return state

    def set_active(self, chat_id: int, active: bool):
        state = self.chat(chat_id)
        state["active"] = bool(active)
        self.save()

    def add(self, chat_id: int, message: dict):
        state = self.chat(chat_id)
        state["messages"].append(message)

        if len(state["messages"]) > MAX_MEMORY_MESSAGES:
            state["messages"] = safe_tail(state["messages"], MAX_MEMORY_MESSAGES)

        if state.get("active"):
            self.save()

    def clear(self, chat_id: int):
        self.data[str(chat_id)] = {"active": False, "messages": []}
        self.save()


memory = Memory(MEMORY_FILE)


# -----------------------------
# MCP client
# -----------------------------

@asynccontextmanager
async def mcp_session():
    if not MCP_ENABLED:
        raise RuntimeError("MCP disabled")

    async with streamablehttp_client(MCP_URL) as transport:
        if isinstance(transport, (tuple, list)):
            read = transport[0]
            write = transport[1]
        else:
            read = transport.read
            write = transport.write

        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_mcp_tool(session, name: str, arguments: dict) -> str:
    result = await session.call_tool(name, arguments)
    text = serialize_mcp_result(result)

    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"

    return text


# -----------------------------
# Agent loop
# -----------------------------

async def complete_with_tools(
    chat_id: int,
    messages: list[dict],
    tools: list[dict],
    tool_name_map: dict[str, str],
    session,
    active: bool,
) -> str:
    for _ in range(MAX_TOOL_ROUNDS):
        kwargs = {}
        if tools:
            kwargs["tools"] = tools

        try:
            resp = await llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            # If provider rejects tool schema, retry once without tools.
            if tools:
                log.warning("LLM rejected tools; retrying without tools: %s", e)
                tools = []
                tool_name_map = {}
                session = None
                continue
            raise

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": normalize_text(msg.content),
                "tool_calls": [
                    {
                        "id": getattr(tc, "id", None) or "",
                        "type": "function",
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", "") or "",
                            "arguments": getattr(getattr(tc, "function", None), "arguments", "") or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            }

            messages.append(assistant_msg)
            memory.add(chat_id, assistant_msg)

            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue

                function_name = getattr(fn, "name", "") or ""
                raw_args = getattr(fn, "arguments", "") or "{}"
                tool_call_id = getattr(tc, "id", None) or ""

                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        args = {"value": args}
                except Exception:
                    args = {"raw_arguments": raw_args}

                original_mcp_tool_name = tool_name_map.get(function_name, function_name)

                try:
                    if session is None:
                        result = (
                            "Browser MCP tools are disabled. "
                            "Send /save_memory to enable browser MCP for this chat."
                        )
                    else:
                        result = await call_mcp_tool(
                            session,
                            original_mcp_tool_name,
                            args,
                        )
                except Exception as e:
                    log.exception("MCP tool call failed: %s", original_mcp_tool_name)
                    result = f"Tool call failed: {e}"

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }

                messages.append(tool_msg)
                memory.add(chat_id, tool_msg)

            continue

        content = normalize_text(msg.content)
        if not content.strip():
            content = "(empty response)"

        assistant_msg = {"role": "assistant", "content": content}
        memory.add(chat_id, assistant_msg)
        return content

    return "Stopped: too many tool rounds."


async def run_agent(chat_id: int, text: str) -> str:
    state = memory.chat(chat_id)
    active = bool(state.get("active"))

    memory.add(chat_id, {"role": "user", "content": text})

    history = safe_tail(state["messages"], MAX_CONTEXT_MESSAGES)

    tools_enabled = active and MCP_ENABLED
    tool_note = (
        "Browser MCP tools are ENABLED for this request."
        if tools_enabled
        else (
            "Browser MCP tools are DISABLED for this request. "
            "If the user asks for browser/data, tell them to send /save_memory."
        )
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": tool_note},
    ] + history

    if tools_enabled:
        try:
            async with mcp_session() as session:
                listed = await session.list_tools()
                mcp_tools = getattr(listed, "tools", []) or []
                tools, tool_name_map = mcp_tools_to_openai_tools(mcp_tools)

                return await complete_with_tools(
                    chat_id=chat_id,
                    messages=messages,
                    tools=tools,
                    tool_name_map=tool_name_map,
                    session=session,
                    active=active,
                )
        except Exception as e:
            log.exception("MCP session failed")
            messages.append(
                {
                    "role": "system",
                    "content": f"MCP unavailable: {e}. Browser tools are disabled for this response.",
                }
            )
            answer = await complete_with_tools(
                chat_id=chat_id,
                messages=messages,
                tools=[],
                tool_name_map={},
                session=None,
                active=active,
            )
            return f"MCP unavailable: {e}\n{answer}"

    return await complete_with_tools(
        chat_id=chat_id,
        messages=messages,
        tools=[],
        tool_name_map={},
        session=None,
        active=active,
    )


# -----------------------------
# Telegram authorization
# -----------------------------

def authorized(update: Update) -> bool:
    if not update.effective_user:
        return False

    # If allowlist is empty, bot answers anyone. Set TELEGRAM_ALLOWED_USER_IDS.
    if not ALLOWED_USER_IDS:
        return True

    return update.effective_user.id in ALLOWED_USER_IDS


# -----------------------------
# Telegram handlers
# -----------------------------

async def cmd_help(update: Update, context):
    if not authorized(update):
        return

    await update.message.reply_text(
        "Commands:\n"
        "/save_memory - persist memory + enable browser MCP tools\n"
        "/clear_memory - wipe memory + disable browser MCP tools\n\n"
        "Without /save_memory: RAM-only context, no disk write, no browser MCP tools."
    )


async def cmd_save(update: Update, context):
    if not authorized(update):
        return

    chat_id = update.effective_chat.id
    memory.set_active(chat_id, True)

    if MCP_ENABLED:
        await update.message.reply_text(
            "Memory on. Browser MCP tools enabled for this chat."
        )
    else:
        await update.message.reply_text(
            "Memory on. MCP tools disabled: set MCP_URL and install MCP SDK."
        )


async def cmd_clear(update: Update, context):
    if not authorized(update):
        return

    chat_id = update.effective_chat.id
    memory.clear(chat_id)

    await update.message.reply_text(
        "Memory cleared. Browser MCP tools disabled for this chat."
    )


async def on_text(update: Update, context):
    if not authorized(update):
        return

    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id

    try:
        await update.message.reply_chat_action("typing")
    except Exception:
        pass

    try:
        answer = await run_agent(chat_id, update.message.text)
    except Exception as e:
        log.exception("Agent run failed")
        answer = f"Error: {e}"

    for chunk in split_text(answer):
        await update.message.reply_text(chunk)


def main():
    if not ALLOWED_USER_IDS:
        log.warning(
            "TELEGRAM_ALLOWED_USER_IDS is empty; bot will answer anyone. "
            "Set it for privacy."
        )

    if not MCP_ENABLED:
        log.warning(
            "MCP disabled. Set MCP_URL and install MCP SDK to enable browser tools."
        )

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(CommandHandler("help", cmd_help))

    app.add_handler(CommandHandler("save_memory", cmd_save))
    app.add_handler(CommandHandler("save", cmd_save))

    app.add_handler(CommandHandler("clear_memory", cmd_clear))
    app.add_handler(CommandHandler("clear", cmd_clear))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
