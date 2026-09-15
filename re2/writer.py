"""The text model that writes digests and wiki pages ([wiki] in config, defaulting to [chat]).

Requests stream, like ask.py's, so llama.cpp sends progress while it reads a long prompt and
no read timeout fires on a slow server. Batches are sized from the configured context.
"""
from __future__ import annotations

import json
import logging
import re
import threading

import httpx

log = logging.getLogger(__name__)

TIMEOUT = 300.0
CHARS_PER_TOKEN = 3.0      # conservative for English mixed with names, handles and URLs
PROMPT_OVERHEAD_TOKENS = 1200


class WriterError(Exception):
    pass


class LengthError(WriterError):
    """The model ran out of output tokens."""


def input_chars(cfg, max_tokens: int, overhead_tokens: int = PROMPT_OVERHEAD_TOKENS) -> int:
    """How many characters of material fit in one request next to the instructions and the reply."""
    return max(4000, int((cfg.wiki_context_tokens - max_tokens - overhead_tokens) * CHARS_PER_TOKEN))


def payload(cfg, system: str, user: str, max_tokens: int, schema: dict | None = None,
            temperature: float = 0.3) -> dict:
    body = {
        "model": cfg.wiki_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Servers set a presence penalty for chat; it punishes the repeated "[m:" of citations and
        # the numbers of member lists, so the wiki turns it off.
        "presence_penalty": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    return body


def complete(client: httpx.Client, cfg, body: dict) -> str:
    """The whole reply; raises LengthError when it was cut off, WriterError on HTTP failures."""
    body = {**body, "stream": True, "return_progress": True}
    parts: list[str] = []
    finish = None
    try:
        with client.stream("POST", f"{cfg.wiki_base_url}/chat/completions", json=body, timeout=TIMEOUT) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise WriterError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                finish = choices[0].get("finish_reason") or finish
    except httpx.HTTPError as exc:
        raise WriterError(f"{type(exc).__name__}: {exc}") from exc
    text = re.sub(r"^\s*<think>.*?</think>", "", "".join(parts), flags=re.DOTALL).strip()
    if finish == "length":
        err = LengthError(f"reply cut off at {body['max_tokens']} tokens")
        err.raw = text
        raise err
    return text


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
