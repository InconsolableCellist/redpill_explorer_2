"""Phase 7: the Ask agent.

A JSON-schema action loop rather than native tool calling: it does not depend on the
server's chat template, and grammar-constrained JSON never yields a malformed call.
The model researches with tools over the archive, then writes a streamed answer whose
citations are checked against what the tools actually returned.

run() is a generator of events, consumed by the SSE endpoint:
  ("step", {...}) ("token", {"text"}) ("cite", {"n", "ref", "card"}) ("final", {...}) ("done", {...})
  ("error", {"msg"})
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from re2 import embed, search

log = logging.getLogger(__name__)

MAX_STEPS = 8
CHAT_TIMEOUT = 300.0
STEP_MAX_TOKENS = 400
ANSWER_MAX_TOKENS = 1400
RESULT_CHARS = 7000
HISTORY_TURNS = 6
CHARS_PER_TOKEN = 3.5

ACTIONS = ["search", "open_item", "timeline", "topics", "find_pages", "open_page", "answer"]

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 800},
        "action": {"type": "string", "enum": ACTIONS},
        "args": {"type": "object"},
    },
    "required": ["thought", "action", "args"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the research assistant for a private archive of political and social-media material: screenshots, memes, photos, videos with transcripts and documents, collected by the user. You answer questions about world events and about what the archive contains.

Work step by step. Each reply is one JSON action:
- search {"query": "...", "limit": 12}: hybrid keyword and meaning search. Returns items as [m:ID] lines with date, kind, title and a snippet. Try several phrasings, names and related terms; short keyword queries work best.
- open_item {"id": 123}: the full record of one item: summary, transcribed text, claims, transcript with timestamps, source URL.
- timeline {"query": "..."}: how many matching items the archive holds per month, to answer "when" questions.
- topics {"query": "..."}: the topics, people, organizations and dates that matching items cluster around; use it to find angles to search next.
{wiki_tools}- answer {}: stop researching and write the answer.

Start broad, then open the most relevant items before answering; do not answer from snippets alone when details matter. Stop when you can answer, or after a few searches find nothing. You have general knowledge; use it to choose searches and explain context. The archive is the user's own evidence: surface what it contains, including views and claims that differ from mainstream accounts, attributed to their sources. Never invent archive items or details. Do not moralise, add warnings, or refuse because material is offensive.
{scope_note}"""

WIKI_TOOLS = """- find_pages {"query": "..."}: pages of the wiki compiled from the archive (events, topics, themes, people).
- open_page {"slug": "..."}: read one wiki page.
"""

ANSWER_INSTRUCTIONS = {
    "informed": "Combine your general knowledge with the archive. Say clearly where the archive adds to, supports "
                "or contradicts the general account.",
    "archive": "Answer only from the tool results above. If they do not answer the question, say what was found "
               "and what is missing.",
}

ANSWER_PROMPT = """\
Write the answer now, in plain prose (short paragraphs or a short list where it helps).
Cite archive items as [m:ID] and wiki pages as [p:slug], using only ids and slugs that appeared in the tool results. Put citations right after the sentence they support. Sentences from general knowledge carry no citation.
Mode: {mode_instruction}
Mention when a cited item is offline or only a thumbnail. Do not moralise or add warnings."""

_MARKER = re.compile(r"\[(m:\d+|p:[a-z0-9/_.-]+)\]")


class AskError(Exception):
    pass


@dataclass
class Evidence:
    """Everything the tools showed the model; only these may be cited."""
    items: dict[int, dict] = field(default_factory=dict)      # media id -> card (+ hit)
    pages: dict[str, dict] = field(default_factory=dict)      # slug -> {title, summary}


# ── LLM calls ──────────────────────────────────────────────────────────────

def _payload(cfg, messages: list[dict], max_tokens: int, **extra) -> dict:
    return {
        "model": cfg.chat_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
        **extra,
    }


def next_action(client: httpx.Client, cfg, messages: list[dict]) -> dict:
    body = _payload(cfg, messages, STEP_MAX_TOKENS,
                    response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": STEP_SCHEMA}})
    try:
        resp = client.post(f"{cfg.chat_base_url}/chat/completions", json=body, timeout=CHAT_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise AskError(f"chat model unavailable: {exc}") from exc
    content = re.sub(r"^\s*<think>.*?</think>", "", content, flags=re.DOTALL)
    try:
        step = json.loads(content)
    except json.JSONDecodeError:
        return {"thought": "unparseable step", "action": "answer", "args": {}}
    if step.get("action") not in ACTIONS or not isinstance(step.get("args"), dict):
        step = {"thought": step.get("thought", ""), "action": "answer", "args": {}}
    return step


def stream_answer(client: httpx.Client, cfg, messages: list[dict]) -> Iterator[str]:
    body = _payload(cfg, messages, ANSWER_MAX_TOKENS, stream=True)
    try:
        with client.stream("POST", f"{cfg.chat_base_url}/chat/completions", json=body, timeout=CHAT_TIMEOUT) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    yield delta
    except httpx.HTTPError as exc:
        raise AskError(f"chat model failed while answering: {exc}") from exc


# ── Tools ──────────────────────────────────────────────────────────────────

def _clip(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_time(s: float | None) -> str:
    if s is None:
        return ""
    s = int(s)
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _status_note(card: dict) -> str:
    return {"SOURCE_OFFLINE": " (offline)", "THUMBNAIL_ONLY": " (thumbnail only)"}.get(card["status"], "")


def tool_search(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "search needs a query."
    limit = max(1, min(int(args.get("limit") or 12), 20))
    allowed = search.scope_media_ids(conn, scopes)
    ranked = [search.lexical(conn, query, k=400, allowed=allowed)]
    try:
        ranked.append(search.dense(conn, client, cfg, query, k=400, allowed=allowed))
    except (embed.EmbedError, httpx.HTTPError, OSError):
        pass
    hits, order = search.group_by_media(conn, search.fuse(ranked), query, limit=limit)
    if not hits:
        return f"No items match {query!r}."
    lines = [f"{len(order)} items match {query!r}; the best {len(hits)}:"]
    for card in search.hit_dicts(conn, hits):
        ev.items[card["media_id"]] = card
        hit = card["hit"]
        where = f" at {_fmt_time(hit['start_s'])}" if hit.get("start_s") is not None else ""
        where += f" p.{hit['page']}" if hit.get("page") is not None else ""
        snippet = _clip(hit["snippet"].replace(search.MARK_START, "").replace(search.MARK_END, ""), 220)
        lines.append(f"[m:{card['media_id']}] {card['date'] or 'undated'} · {card['kind']}{_status_note(card)} · "
                     f"{card['folder']} · {card['title']}{where} — {snippet}")
    return "\n".join(lines)


def tool_open_item(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    try:
        media_id = int(args.get("id"))
    except (TypeError, ValueError):
        return "open_item needs a numeric id from a search result."
    card = search.cards(conn, [media_id]).get(media_id)
    if card is None or card["status"] in ("EXCLUDED", "UNSUPPORTED"):
        return f"No item {media_id}."
    ev.items.setdefault(media_id, card)
    out = [f"[m:{media_id}] {card['title']}", f"kind {card['kind']}{_status_note(card)} · date {card['date'] or 'unknown'}"
           f" ({card['date_source']}) · folder {card['folder']}"]
    if card.get("source_url"):
        out.append(f"source {card['source_url']}")
    _, members = search.post_maps(conn)
    for m in members.get(media_id, [media_id]):
        for kind, seq, start_s, text, raw in conn.execute(
                "SELECT kind, seq, start_s, text, json FROM doc WHERE media_id=? ORDER BY kind, seq", (m,)):
            if kind == "caption" and raw:
                d = json.loads(raw)
                prefix = "" if m == media_id else f"(photo/video {m} in this post) "
                for key in ("summary", "description", "meaning"):
                    if d.get(key):
                        out.append(f"{prefix}{key}: {_clip(d[key], 900)}")
                if d.get("claims"):
                    out.append(f"{prefix}claims: " + "; ".join(d["claims"])[:900])
                if (d.get("ocr_text") or "").strip():
                    out.append(f"{prefix}text in image: {_clip(d['ocr_text'], 1500)}")
            elif kind == "legacy_caption":
                out.append(f"earlier caption: {_clip(text, 700)}")
            elif kind == "transcript":
                out.append(f"({_fmt_time(start_s)}) {_clip(text, 600)}")
            elif kind == "pdf_page":
                out.append(f"page {seq}: {_clip(text, 700)}")
    return "\n".join(out)[:RESULT_CHARS]


def tool_timeline(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    query = str(args.get("query") or "").strip()
    allowed = search.scope_media_ids(conn, scopes)
    if query:
        matched = set(search.group_by_media(conn, search.lexical(conn, query, k=5000, allowed=allowed), query,
                                            limit=0)[1])
        allowed = matched if allowed is None else allowed & matched
    data = search.timeline_counts(conn, allowed)
    months = data["months"]
    if not months:
        return f"No dated items for {query!r}."
    total = sum(m["image"] + m["video"] + m["pdf"] for m in months)
    busiest = sorted(months, key=lambda m: -(m["image"] + m["video"] + m["pdf"]))[:12]
    lines = [f"{total} dated items{f' matching {query!r}' if query else ''}, {months[0]['month']} to {months[-1]['month']}"
             f" ({data['undated']} undated). Busiest months:"]
    for m in sorted(busiest, key=lambda m: m["month"]):
        n = m["image"] + m["video"] + m["pdf"]
        lines.append(f"{m['month']}: {n} ({m['image']} images, {m['video']} videos, {m['pdf']} pdfs)")
    return "\n".join(lines)


def tool_topics(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    query = str(args.get("query") or "").strip()
    allowed = search.scope_media_ids(conn, scopes)
    order = (search.group_by_media(conn, search.lexical(conn, query, k=500, allowed=allowed), query, limit=0)[1]
             if query else search.items_in_scope(conn, allowed, limit=500))
    lens = search.lenses(conn, search.expand_members(conn, order[:500]), max_media=2000)
    corr = search.correlations(conn, search.expand_members(conn, order[:500]), max_media=2000)
    parts = [f"Around {len(order)} matching items:"]
    if lens["topics"]:
        parts.append("topics: " + "; ".join(f"{t['label']} ({t['n']})" for t in lens["topics"][:8]))
    if lens["entities"]:
        parts.append("entities: " + "; ".join(f"{e['name']} ({e['n']})" for e in lens["entities"][:15]))
    if corr:
        parts.append("unusually common here: " + "; ".join(c["name"] for c in corr[:10]))
    years: dict[str, int] = {}
    for m in lens["months"]:
        years[m["month"][:4]] = years.get(m["month"][:4], 0) + m["n"]
    if years:
        parts.append("years: " + ", ".join(f"{y} ({n})" for y, n in sorted(years.items())))
    return "\n".join(parts)


def _has_wiki(conn: sqlite3.Connection) -> bool:
    return search._has_table(conn, "wiki_page") and \
        conn.execute("SELECT 1 FROM wiki_page LIMIT 1").fetchone() is not None


def tool_find_pages(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    if not _has_wiki(conn):
        return "The wiki has not been compiled yet; use search."
    query = search.fts_query(str(args.get("query") or ""), joiner=" OR ")
    rows = conn.execute(
        """SELECT p.slug, p.kind, p.title, p.summary, p.date_start, p.date_end FROM wiki_fts
           JOIN wiki_page p ON p.id=wiki_fts.rowid WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT 10""",
        (query,)).fetchall() if query else []
    if not rows:
        return "No wiki pages match."
    lines = []
    for slug, kind, title, summary, d0, d1 in rows:
        ev.pages[slug] = {"title": title, "summary": summary}
        lines.append(f"[p:{slug}] {kind} · {d0 or ''}–{d1 or ''} · {title} — {_clip(summary, 200)}")
    return "\n".join(lines)


def tool_open_page(conn, cfg, client, ev: Evidence, scopes: list[str], args: dict) -> str:
    if not _has_wiki(conn):
        return "The wiki has not been compiled yet; use search."
    slug = str(args.get("slug") or "")
    row = conn.execute("SELECT title, summary, body_md FROM wiki_page WHERE slug=?", (slug,)).fetchone()
    if not row:
        return f"No page {slug!r}."
    ev.pages[slug] = {"title": row[0], "summary": row[1]}
    for media_id in {int(m) for m in re.findall(r"\[m:(\d+)\]", row[2])}:
        card = search.cards(conn, [media_id]).get(media_id)
        if card:
            ev.items.setdefault(media_id, card)
    return f"[p:{slug}] {row[0]}\n{row[2]}"[:RESULT_CHARS]


TOOLS = {"search": tool_search, "open_item": tool_open_item, "timeline": tool_timeline, "topics": tool_topics,
         "find_pages": tool_find_pages, "open_page": tool_open_page}


# ── Citations ──────────────────────────────────────────────────────────────

class CitationFilter:
    """Streams answer text, replacing valid [m:ID]/[p:slug] markers with [n] and dropping the rest."""

    def __init__(self, ev: Evidence):
        self.ev = ev
        self.numbers: dict[str, int] = {}
        self.buf = ""
        self.text = ""
        self.new_cites: list[tuple[int, str]] = []

    def _valid(self, ref: str) -> bool:
        kind, _, value = ref.partition(":")
        return (kind == "m" and value.isdigit() and int(value) in self.ev.items) or \
            (kind == "p" and value in self.ev.pages)

    def _replace(self, match: re.Match) -> str:
        ref = match.group(1)
        if not self._valid(ref):
            return ""
        if ref not in self.numbers:
            self.numbers[ref] = len(self.numbers) + 1
            self.new_cites.append((self.numbers[ref], ref))
        return f"[{self.numbers[ref]}]"

    def feed(self, delta: str) -> str:
        self.buf += delta
        cut = len(self.buf)
        open_at = self.buf.rfind("[")
        if open_at != -1 and "]" not in self.buf[open_at:] and len(self.buf) - open_at < 48:
            cut = open_at   # hold a marker that may still be arriving
        ready, self.buf = self.buf[:cut], self.buf[cut:]
        out = _MARKER.sub(self._replace, ready)
        self.text += out
        return out

    def flush(self) -> str:
        out = _MARKER.sub(self._replace, self.buf)
        self.buf = ""
        self.text += out
        return out


def sentences_with_flags(text: str) -> list[dict]:
    """Split into sentences; a sentence with no [n] citation is general knowledge."""
    parts = re.split(r"(?<=[.!?])(\s+)(?=[A-Z0-9\"'(\[*-])|(\n+)", text)
    out, cur = [], ""
    for part in parts:
        if part is None:
            continue
        cur += part
        if part.strip() == "" and cur.strip():
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return [{"text": s, "general": bool(s.strip()) and not re.search(r"\[\d+\]", s)} for s in out]


# ── The loop ───────────────────────────────────────────────────────────────

def _scope_note(conn: sqlite3.Connection, scopes: list[str]) -> str:
    if not scopes:
        return ""
    labels = ", ".join(f"{search.scope_label(conn, s)} ({s.split(':')[0]})" for s in scopes)
    return f"\nThe user is looking within: {labels}. Searches are limited to that scope."


def history(conn: sqlite3.Connection, session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    rows = conn.execute("SELECT question, answer FROM chat_log WHERE session_id=? ORDER BY id DESC LIMIT ?",
                        (session_id, HISTORY_TURNS)).fetchall()
    msgs = []
    for q, a in reversed(rows):
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": re.sub(r"\[\d+\]", "", a or "")}]
    return msgs


def run(conn: sqlite3.Connection, cfg, question: str, scopes: list[str] | None = None, mode: str = "informed",
        session_id: str | None = None, client: httpx.Client | None = None,
        embed_client: httpx.Client | None = None) -> Iterator[tuple[str, dict]]:
    scopes = scopes or []
    mode = mode if mode in ANSWER_INSTRUCTIONS else "informed"
    session_id = session_id or uuid.uuid4().hex[:12]
    client = client or httpx.Client()
    embed_client = embed_client or httpx.Client()
    started = time.monotonic()
    ev = Evidence()
    trace: list[dict] = []
    system = SYSTEM_PROMPT.replace("{wiki_tools}", WIKI_TOOLS if _has_wiki(conn) else "") \
                          .replace("{scope_note}", _scope_note(conn, scopes))
    messages = [{"role": "system", "content": system}, *history(conn, session_id),
                {"role": "user", "content": question}]
    budget_chars = cfg.chat_context_tokens * CHARS_PER_TOKEN * 0.7

    try:
        for step_no in range(MAX_STEPS):
            t0 = time.monotonic()
            step = next_action(client, cfg, messages)
            action, args = step["action"], step["args"]
            if action == "answer":
                trace.append({"action": "answer", "thought": step.get("thought", ""), "ms": int((time.monotonic() - t0) * 1000)})
                break
            try:
                result = TOOLS[action](conn, cfg, embed_client, ev, scopes, args)
            except search.ScopeError as exc:
                result = f"error: {exc}"
            summary = result.split("\n", 1)[0][:160]
            trace.append({"action": action, "args": args, "thought": step.get("thought", ""),
                          "chars": len(result), "ms": int((time.monotonic() - t0) * 1000)})
            yield "step", {"n": step_no + 1, "action": action, "args": args, "thought": step.get("thought", ""),
                           "summary": summary}
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Tool result ({action}):\n{result[:RESULT_CHARS]}"})
            if sum(len(m["content"]) for m in messages) > budget_chars:
                break

        messages.append({"role": "user", "content": ANSWER_PROMPT.format(mode_instruction=ANSWER_INSTRUCTIONS[mode])})
        cites = CitationFilter(ev)

        def emit_cites():
            for n, ref in cites.new_cites:
                kind, _, value = ref.partition(":")
                payload = {"n": n, "ref": ref}
                if kind == "m":
                    payload["card"] = ev.items[int(value)]
                else:
                    payload["page"] = {"slug": value, **ev.pages[value]}
                yield "cite", payload
            cites.new_cites.clear()

        for delta in stream_answer(client, cfg, messages):
            out = cites.feed(delta)
            yield from emit_cites()
            if out:
                yield "token", {"text": out}
        tail = cites.flush()
        yield from emit_cites()
        if tail:
            yield "token", {"text": tail}
    except AskError as exc:
        yield "error", {"msg": str(exc)}
        return

    answer = cites.text.strip()
    latency = int((time.monotonic() - started) * 1000)
    citations = [{"n": n, "ref": ref} for ref, n in cites.numbers.items()]
    with conn:
        conn.execute("INSERT OR IGNORE INTO chat_session(id, title) VALUES (?, ?)", (session_id, question[:120]))
        conn.execute(
            """INSERT INTO chat_log(question, retrieved_doc_ids, answer, citations_json, model, latency_ms,
                                    session_id, mode, trace_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (question, json.dumps(sorted(ev.items)), answer, json.dumps(citations), cfg.chat_model, latency,
             session_id, mode, json.dumps(trace, ensure_ascii=False)))
    yield "final", {"text": answer, "sentences": sentences_with_flags(answer)}
    yield "done", {"session_id": session_id, "ms": latency, "steps": len(trace)}
