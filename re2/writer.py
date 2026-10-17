"""The text model that writes digests and wiki pages ([wiki] in config, defaulting to [chat]).

Requests go through ask.py's streaming helpers (llama.cpp sends progress while it reads a long
prompt, so no read timeout fires on a slow server). Batches are sized from the configured context.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from types import SimpleNamespace

import httpx

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.0      # conservative for English mixed with names, handles and URLs
PROMPT_OVERHEAD_TOKENS = 1200


class WriterError(Exception):
    pass


def input_chars(cfg, max_tokens: int, overhead_tokens: int = PROMPT_OVERHEAD_TOKENS) -> int:
    """How many characters of material fit in one request next to the instructions and the reply."""
    return max(4000, int((cfg.wiki_context_tokens - max_tokens - overhead_tokens) * CHARS_PER_TOKEN))


def _as_chat(cfg) -> SimpleNamespace:
    """A view of cfg in which ask.py's chat helpers address the wiki model."""
    return SimpleNamespace(chat_base_url=cfg.wiki_base_url, chat_model=cfg.wiki_model)


def payload(cfg, system: str, user: str, max_tokens: int, schema: dict | None = None,
            temperature: float = 0.3) -> dict:
    from re2 import ask

    extra = {
        "temperature": temperature,
        # Servers set a presence penalty for chat; it punishes the repeated "[m:" of citations and
        # the numbers of member lists, so the wiki turns it off.
        "presence_penalty": 0.0,
    }
    if schema is not None:
        extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    return ask._payload(_as_chat(cfg), [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens, **extra)


def complete(client: httpx.Client, cfg, body: dict) -> str:
    """The whole reply, streamed with ask.py's helper; WriterError on HTTP failures."""
    from re2 import ask

    try:
        text = "".join(ask._stream_chat(client, _as_chat(cfg), body, None))
    except httpx.HTTPError as exc:
        raise WriterError(f"{type(exc).__name__}: {exc}") from exc
    from re2.vlm import scrub

    return scrub(re.sub(r"^\s*<think>.*?</think>", "", text, flags=re.DOTALL).strip())


def complete_json(client: httpx.Client, cfg, body: dict) -> dict:
    text = complete(client, cfg, body)
    try:
        out = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WriterError(f"reply is not JSON: {text[:200]!r}") from exc
    if not isinstance(out, dict):
        raise WriterError("reply is not a JSON object")
    return out


class Clients:
    """One httpx client per worker thread, all closed at the end."""

    def __init__(self, factory=httpx.Client):
        self.factory = factory
        self.local = threading.local()
        self.all: list[httpx.Client] = []
        self.lock = threading.Lock()

    def get(self) -> httpx.Client:
        client = getattr(self.local, "client", None)
        if client is None:
            client = self.local.client = self.factory()
            with self.lock:
                self.all.append(client)
        return client

    def close(self) -> None:
        for client in self.all:
            client.close()
