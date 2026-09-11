#!/usr/bin/env python3
"""
minimal_browser_mcp_agent_bot.py

Telegram AI agent with explicit memory + browser MCP tool calls.

Behavior:
- /save_memory  -> persists this chat to disk and enables browser MCP tools
- /clear_memory -> wipes this chat from disk/RAM and disables tools
- Before /save_memory: context is RAM-only, no disk write, no MCP tools
- After /clear_memory: nothing is kept
- Sends Telegram debug/status messages while thinking and using tools

Commands:
  /save_memory or /save
  /clear_memory or /clear
"""

import os
import re
import json
import logging
import threading
import time
import asyncio
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

ALLOWED_USER_IDS = {
    int(x)
    for x in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").replace(",", " ").split()
    if x.strip()
}

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""

MCP_URL = os.getenv("MCP_URL", "")

MEMORY_FILE = Path(os.getenv("MEMORY_FILE", "agent_memory.json"))

HEALTH_HOST = os.getenv("HEALTH_HOST", "0.0.0.0")
HEALTH_PORT = int(os.getenv("HEALTH_PORT") or os.getenv("PORT") or "10000")
START_TIME = time.time()

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are Serge's private operator assistant. Be terse and precise. "
        "Use browser MCP tools only when the user explicitly asks to fetch, read, "
        "open, extract, click, navigate, or otherwise act on browser data.\n\n"
        "Tool usage rules:\n"
        "- Always validate arguments match the tool schema before calling.\n"
        "- If a tool returns InvalidParams, inspect the error, correct the arguments, and retry.\n"
        "- Pass only valid JSON types (string, number, boolean, array, object, null).\n"
        "- Never pass undefined, NaN, or malformed JSON.\n"
        "- If unsure about tool parameters, ask the user for clarification instead of guessing."
    ),
)

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "30"))
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "220"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "80"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "12000"))
MAX_INVALID_PARAMS_RETRIES = int(os.getenv("MAX_INVALID_PARAMS_RETRIES", "2"))

TELEGRAM_DEBUG = os.getenv("TELEGRAM_DEBUG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
DEBUG_TOOL_RESULTS = os.getenv("DEBUG_TOOL_RESULTS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
DEBUG_TOOL_RESULT_PREVIEW_CHARS = int(
    os.getenv("DEBUG_TOOL_RESULT_PREVIEW_CHARS", "700")
)
DEBUG_TOOL_ARGS_PREVIEW_CHARS = int(
    os.getenv("DEBUG_TOOL_ARGS_PREVIEW_CHARS", "300")
)

MCP_TOOL_CALL_TIMEOUT = int(os.getenv("MCP_TOOL_CALL_TIMEOUT", "60"))

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


async def send_debug(bot, chat_id: int, text: str):
    if not TELEGRAM_DEBUG or bot is None or not text:
        return

    try:
        await bot.send_message(chat_id=chat_id, text=str(text)[:3900])
    except Exception as e:
        log.warning("Debug send failed: %s", e)


def split_text(text: str, limit: int = 4000) -> list:
    text = str(text or "").strip()
    if not text:
        return ["(empty response)"]
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def safe_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "tool"


def message_groups(messages: list) -> list:
    """
    Groups assistant tool_calls together with their matching tool responses.
    Prevents truncation from leaving orphan tool messages.
    """
    groups = []
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


def safe_tail(messages: list, limit: int) -> list:
    if limit <= 0:
        return []
    if len(messages) <= limit:
        return messages

    groups = message_groups(messages)
    out = []
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


def mcp_tools_to_openai_tools(tools: list) -> tuple:
    """
    Converts MCP tools to OpenAI function tools.
    Returns:
      tools, name_map
      name_map maps sanitized OpenAI function name -> original MCP tool name.
    """
    out = []
    name_map = {}
    used = set()

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

    parts = []

    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        elif isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))

    return prefix + ("\n".join(parts) if parts else str(result))


def truncate_preview(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


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
        self.data = {}
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
# Health endpoint
# -----------------------------

class HealthHandler(BaseHTTPRequestHandler):
    server_version = "AgentHealth/1.0"

    def log_message(self, format, *args):
        pass

    def _path(self) -> str:
        path = self.path.split("?", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/")
        return path or "/"

    def _payload(self) -> dict:
        return {
            "ok": True,
            "service": "telegram-browser-mcp-agent",
            "uptime_sec": int(time.time() - START_TIME),
            "mcp_enabled": MCP_ENABLED,
            "mcp_import_ok": MCP_IMPORT_OK,
            "mcp_url_configured": bool(MCP_URL),
            "llm_model_configured": bool(LLM_MODEL),
            "telegram_allowlist_configured": bool(ALLOWED_USER_IDS),
        }

    def _send(self, code: int, payload=None, head_only: bool = False):
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if not head_only and body:
            self.wfile.write(body)

    def do_GET(self):
        path = self._path()

        if path == "/health":
            self._send(200, self._payload())
        elif path == "/":
            self._send(200, {"ok": True, "endpoints": ["/health"]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_HEAD(self):
        path = self._path()

        if path == "/health":
            self._send(200, head_only=True)
        elif path == "/":
            self._send(200, head_only=True)
        else:
            self._send(404, head_only=True)


def start_health_server():
    try:
        ThreadingHTTPServer.daemon_threads = True
        server = ThreadingHTTPServer((HEALTH_HOST, HEALTH_PORT), HealthHandler)
    except OSError as e:
        log.error("Health server failed to bind %s:%s: %s", HEALTH_HOST, HEALTH_PORT, e)
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    log.info("Health endpoint listening on http://%s:%s/health", HEALTH_HOST, HEALTH_PORT)


# -----------------------------
# MCP client
# -----------------------------

@asynccontextmanager
async def mcp_session():
    if not MCP_ENABLED:
        raise RuntimeError("MCP disabled")

    # NOTE: removed sse_read_timeout kwarg to support older mcp SDK versions.
    # The tool call timeout is enforced via asyncio.wait_for in call_mcp_tool.
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
    try:
        result = await asyncio.wait_for(
            session.call_tool(name, arguments),
            timeout=MCP_TOOL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"MCP tool call '{name}' timed out after {MCP_TOOL_CALL_TIMEOUT}s"
        )

    text = serialize_mcp_result(result)

    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"

    return text


# -----------------------------
# Agent loop
# -----------------------------

async def complete_with_tools(
    chat_id: int,
    messages: list,
    tools: list,
    tool_name_map: dict,
    session,
    active: bool,
    bot=None,
) -> str:
    invalid_params_count = {}
    
    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        await send_debug(bot, chat_id, f"Thinking... (round {round_num}/{MAX_TOOL_ROUNDS})")

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
                await send_debug(bot, chat_id, f"Tool schema rejected. Retrying without tools: {e}")
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

            await send_debug(
                bot,
                chat_id,
                f"Agent requested {len(tool_calls)} tool call(s).",
            )

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

                await send_debug(bot, chat_id, f"Using tool: {original_mcp_tool_name}")

                if DEBUG_TOOL_RESULTS:
                    args_preview = truncate_preview(
                        normalize_text(args),
                        DEBUG_TOOL_ARGS_PREVIEW_CHARS,
                    )
                    await send_debug(bot, chat_id, f"Tool args:\n{args_preview}")

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
                except asyncio.TimeoutError:
                    log.exception("MCP tool call timed out: %s", original_mcp_tool_name)
                    result = f"Tool call timed out after {MCP_TOOL_CALL_TIMEOUT}s"
                except Exception as e:
                    error_str = str(e)
                    log.error("MCP tool call failed: %s | args=%s | error=%s", 
                             original_mcp_tool_name, args, error_str)
                    
                    # Track retries per tool to prevent infinite loops
                    tool_key = original_mcp_tool_name
                    invalid_params_count[tool_key] = invalid_params_count.get(tool_key, 0) + 1
                    
                    if invalid_params_count[tool_key] > MAX_INVALID_PARAMS_RETRIES:
                        result = (
                            f"Tool '{original_mcp_tool_name}' has failed {MAX_INVALID_PARAMS_RETRIES} times with invalid parameters. "
                            f"Last error: {error_str}. Please verify the correct tool and parameters, or ask the user for help."
                        )
                    elif "InvalidParams" in error_str or "invalid params" in error_str.lower():
                        result = (
                            f"Error: Invalid parameters for tool '{original_mcp_tool_name}'. "
                            f"The MCP server rejected your arguments: {args}. "
                            f"Server error: {error_str}. "
                            f"Retry attempt {invalid_params_count[tool_key]}/{MAX_INVALID_PARAMS_RETRIES}. "
                            f"Please check the tool schema and retry with corrected arguments."
                        )
                    else:
                        result = f"Tool call failed: {error_str}"

                await send_debug(
                    bot,
                    chat_id,
                    f"Tool finished: {original_mcp_tool_name} ({len(result)} chars)",
                )

                if DEBUG_TOOL_RESULTS:
                    result_preview = truncate_preview(
                        result,
                        DEBUG_TOOL_RESULT_PREVIEW_CHARS,
                    )
                    await send_debug(bot, chat_id, f"Tool result:\n{result_preview}")

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

        await send_debug(bot, chat_id, "Final answer ready.")
        return content

    return "Stopped: too many tool rounds."


async def run_agent(chat_id: int, text: str, bot=None) -> str:
    state = memory.chat(chat_id)
    active = bool(state.get("active"))

    await send_debug(bot, chat_id, "Received message. Preparing agent...")

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
            await send_debug(bot, chat_id, "Connecting to browser MCP...")
            async with mcp_session() as session:
                await send_debug(bot, chat_id, "Listing MCP tools...")
                listed = await session.list_tools()
                mcp_tools = getattr(listed, "tools", []) or []
                tools, tool_name_map = mcp_tools_to_openai_tools(mcp_tools)

                await send_debug(
                    bot,
                    chat_id,
                    f"MCP connected. {len(tools)} tools available.",
                )

                return await complete_with_tools(
                    chat_id=chat_id,
                    messages=messages,
                    tools=tools,
                    tool_name_map=tool_name_map,
                    session=session,
                    active=active,
                    bot=bot,
                )
        except Exception as e:
            log.exception("MCP session failed")
            await send_debug(bot, chat_id, f"MCP unavailable: {e}")

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
                bot=bot,
            )
            return f"MCP unavailable: {e}\n{answer}"

    return await complete_with_tools(
        chat_id=chat_id,
        messages=messages,
        tools=[],
        tool_name_map={},
        session=None,
        active=active,
        bot=bot,
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
        answer = await run_agent(chat_id, update.message.text, bot=context.bot)
    except Exception as e:
        log.exception("Agent run failed")
        answer = f"Error: {e}"
        await send_debug(context.bot, chat_id, f"Error: {e}")

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

    start_health_server()

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
