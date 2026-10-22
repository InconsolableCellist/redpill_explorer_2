"""Phase 7: the Ask agent.

A JSON-schema action loop rather than native tool calling: it does not depend on the
server's chat template, and grammar-constrained JSON never yields a malformed call.
The model researches with tools over the archive, then writes a streamed answer whose
citations are checked against what the tools actually returned.

run() is a generator of events, consumed by the SSE endpoint:
  ("step", {...}) ("token", {"text"}) ("cite", {"n", "ref", "card"}) ("final", {...}) ("done", {...})
  ("error", {"msg"})
While a model call or tool is under way, Control.live reports what it is doing; a Run keeps only the latest
and sends it as ("live", {"what", "phase", "elapsed_ms", ...}).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from datetime import date, timedelta
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from re2 import db, embed, search

log = logging.getLogger(__name__)

MAX_STEPS = 8          # at the default depth; deeper research gets more
CHAT_TIMEOUT = 300.0
STEP_MAX_TOKENS = 400
ANSWER_MAX_TOKENS = 1400
RESULT_CHARS = 7000
LIVE_EVERY_S = 0.5     # at most this often, a model call reports its progress
LIVE_PREVIEW_CHARS = 240
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

Start broad, then open the two or three most relevant items before answering; do not answer from snippets alone when details matter. Rephrasing the same search rarely finds more: after three or four searches, answer with what you have. You have general knowledge; use it to choose searches and explain context. The archive is the user's own evidence: surface what it contains, including views and claims that differ from mainstream accounts, attributed to their sources. Never invent archive items or details. Do not moralise, add warnings, or refuse because material is offensive.
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

MAX_SEARCHES = 4
MAX_OPENS = 5

ANSWER_SYSTEM = """\
You are the research assistant for a private archive of political and social-media material collected by the user. You write clear, factual answers from research notes. Do not moralise, add warnings, or refuse because material is offensive."""

ANSWER_PROMPT = """\
Question: {question}

Research notes (tool results from the archive):
{notes}

Write the answer now, in plain prose (short paragraphs or a short list where it helps). Do not output JSON.
Cite archive items as [m:ID] and wiki pages as [p:slug], using only ids and slugs that appeared in the tool results. Put citations right after the sentence they support. Sentences from general knowledge carry no citation.
Mode: {mode_instruction}
Mention when a cited item is offline or only a thumbnail. Do not moralise or add warnings."""

_REF = r"(?:m:\d+|p:[a-z0-9/_.-]+)"
# One citation or a group of them: [m:12], [m:12, m:40], [m:12; 40], [m:12, p:topic/x].
_MARKER = re.compile(rf"\[({_REF}(?:\s*[,;]\s*(?:{_REF}|\d+))*)\]")
_HOLD_CHARS = 240   # a "[" with no "]" yet, this close to the end, may be a citation still arriving
# Models sometimes cite a tool instead of an item: "[timeline]", "[search results]".
_TOOL_MARKER = re.compile(r"\s?\[(?:timeline|search|topics|open_item|find_pages|open_page|tool)[^\]\n]{0,40}\]", re.IGNORECASE)


class AskError(Exception):
    pass


class Stopped(Exception):
    """The user pressed stop."""


class Control:
    """A stop flag the model calls check between streamed chunks, and where they report what they are doing."""

    def __init__(self):
        self.stop = threading.Event()
        self.report = None   # Run.set_live, when someone may be watching

    def check(self) -> None:
        if self.stop.is_set():
            raise Stopped()

    def live(self, what: str, phase: str, **status) -> None:
        if self.report:
            self.report({"what": what, "phase": phase, **status})


# ── Depth: how much the agent reads and how much of it reaches the answer ──

DEPTH_DEFAULT = 0.35
PROMPT_OVERHEAD_TOKENS = 2500     # system prompt, question and answer instructions


@dataclass
class Plan:
    depth: float
    notes_tokens: int      # research notes handed to the answer model
    searches: int
    opens: int
    search_limit: int
    related: int           # related items gathered around each attached item
    steps: int


def plan(cfg, depth: float | None = None) -> Plan:
    """0 is a quick answer from a few items; 1 fills the chat model's context window with evidence."""
    d = DEPTH_DEFAULT if depth is None else max(0.0, min(1.0, float(depth)))
    usable = max(2000, cfg.chat_context_tokens - ANSWER_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS)
    searches, opens = 3 + round(3 * d), 3 + round(5 * d)
    return Plan(depth=d, notes_tokens=int(usable * (0.12 + 0.88 * d)), searches=searches, opens=opens,
                search_limit=10 + round(10 * d), related=4 + round(20 * d), steps=searches + opens + 2)


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


def step_schema(actions: list[str]) -> dict:
    schema = json.loads(json.dumps(STEP_SCHEMA))
    schema["properties"]["action"]["enum"] = actions
    return schema


def _partial_thought(text: str) -> str:
    """The "thought" of a step's JSON while it is still being written."""
    m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    return re.sub(r"\\(.)", lambda e: {"n": " ", "t": " "}.get(e.group(1), e.group(1)), m.group(1)) if m else ""


def _stream_chat(client: httpx.Client, cfg, body: dict, control: Control | None, what: str = "",
                 preview=None) -> Iterator[str]:
    """Content deltas from a streamed completion. Every call streams, so a stop takes effect between chunks;
    return_progress makes llama.cpp send chunks while it reads a long prompt, too. Along the way it reports
    to control.live: waiting for the server, reading the prompt (tokens of it so far), then writing (tokens
    so far and preview(text), the end of the text by default)."""
    body = {**body, "stream": True, "return_progress": True}
    t0 = last = time.monotonic()
    text, tokens = "", 0

    def tell(phase: str, force: bool = False, **status) -> None:
        nonlocal last
        now = time.monotonic()
        if control and (force or now - last >= LIVE_EVERY_S):
            last = now
            control.live(what, phase, elapsed_ms=int((now - t0) * 1000), **status)

    tell("waiting", force=True)
    with client.stream("POST", f"{cfg.chat_base_url}/chat/completions", json=body, timeout=CHAT_TIMEOUT) as resp:
        if resp.status_code >= 400:
            resp.read()
            # llama.cpp explains itself in the body ("Context size has been exceeded"); keep that in the error.
            raise httpx.HTTPStatusError(f"HTTP {resp.status_code}: {resp.text[:300]}", request=resp.request,
                                        response=resp)
        for line in resp.iter_lines():
            if control:
                control.check()   # leaving the with block closes the connection, which cancels the request
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            progress = chunk.get("prompt_progress")
            if progress and progress.get("total"):
                tell("reading", progress["processed"] >= progress["total"], processed=progress["processed"],
                     total=progress["total"], cache=progress.get("cache", 0))
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}).get("content") if choices else None
            if delta:
                text += delta
                tokens += 1
                tell("writing", tokens == 1, tokens=tokens,
                     preview=(preview(text) if preview else text)[-LIVE_PREVIEW_CHARS:])
                yield delta


def next_action(client: httpx.Client, cfg, messages: list[dict], actions: list[str] | None = None,
                control: Control | None = None, what: str = "choosing the next step") -> dict:
    schema = step_schema(actions or ACTIONS)
    body = _payload(cfg, messages, STEP_MAX_TOKENS,
                    response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": schema}})
    try:
        content = "".join(_stream_chat(client, cfg, body, control, what, _partial_thought))
    except httpx.HTTPError as exc:
        raise AskError(f"chat model unavailable: {exc}") from exc
    content = re.sub(r"^\s*<think>.*?</think>", "", content, flags=re.DOTALL)
    try:
        step = json.loads(content)
    except json.JSONDecodeError:
        return {"thought": "unparseable step", "action": "answer", "args": {}}
    if step.get("action") not in (actions or ACTIONS) or not isinstance(step.get("args"), dict):
        step = {"thought": step.get("thought", ""), "action": "answer", "args": {}}
    return step


def stream_answer(client: httpx.Client, cfg, messages: list[dict], control: Control | None = None,
                  max_tokens: int = ANSWER_MAX_TOKENS) -> Iterator[str]:
    try:
        yield from _stream_chat(client, cfg, _payload(cfg, messages, max_tokens), control, "writing the answer")
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

    def _number(self, ref: str) -> str:
        if not self._valid(ref):
            return ""
        if ref not in self.numbers:
            self.numbers[ref] = len(self.numbers) + 1
            self.new_cites.append((self.numbers[ref], ref))
        return f"[{self.numbers[ref]}]"

    def _replace(self, match: re.Match) -> str:
        # A bare number in a group ("m:12, 40") is another item.
        refs = [r if ":" in r else f"m:{r}" for r in re.split(r"\s*[,;]\s*", match.group(1))]
        return "".join(self._number(r) for r in dict.fromkeys(refs))

    def feed(self, delta: str) -> str:
        self.buf += delta
        cut = len(self.buf)
        open_at = self.buf.rfind("[")
        if open_at != -1 and "]" not in self.buf[open_at:] and len(self.buf) - open_at < _HOLD_CHARS:
            cut = open_at   # hold a marker that may still be arriving
        ready, self.buf = self.buf[:cut], self.buf[cut:]
        out = _MARKER.sub(self._replace, _TOOL_MARKER.sub("", ready))
        self.text += out
        return out

    def flush(self) -> str:
        out = _MARKER.sub(self._replace, _TOOL_MARKER.sub("", self.buf))
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


# ── Attached items, related items, and filling the notes to the depth's budget ──

RELATED_DAYS = 30
RECORD_CHARS = 1800        # one item's record when filling the notes; attached items get RESULT_CHARS
FILL_SEARCH_LIMIT = 80


def _day(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text[:10]) if text and len(text) >= 10 else None
    except ValueError:
        return None


def related_items(conn: sqlite3.Connection, cfg, media_id: int, k: int) -> list[int]:
    """Items around one item: close in meaning and from the same weeks first, then items from those weeks
    sharing its people, places and organizations, then the closest in meaning from any time."""
    try:
        similar = search.similar(conn, cfg, media_id, k=k * 4)
    except Exception:  # noqa: BLE001 - no embedding index yet; the other two sources still work
        log.warning("similar items unavailable for %s", media_id, exc_info=True)
        similar = []
    to_primary, members = search.post_maps(conn)
    primary = to_primary.get(media_id, media_id)
    own = set(members.get(primary, [])) | {primary, media_id}
    card = search.cards(conn, [media_id]).get(media_id) or {}
    near: set[int] = set()
    sharing: list[int] = []
    day = _day(card.get("date"))
    if day:
        lo, hi = day - timedelta(days=RELATED_DAYS), day + timedelta(days=RELATED_DAYS)
        near = search.scope_media_ids(conn, [f"date:{lo.isoformat()}..{hi.isoformat()}"]) or set()
        sharing = [r[0] for r in conn.execute(
            """SELECT b.media_id FROM media_entity a
               JOIN media_entity b ON b.entity_id=a.entity_id AND b.media_id != a.media_id
               WHERE a.media_id IN (SELECT value FROM json_each(?))
               GROUP BY b.media_id ORDER BY COUNT(*) DESC LIMIT 2000""", (json.dumps(sorted(own)),))
            if r[0] in near]
    out: list[int] = []
    for m in [m for m in similar if m in near] + sharing + similar:
        m = to_primary.get(m, m)
        if m not in own and m not in out:
            out.append(m)
            if len(out) >= k:
                break
    return out


def _one_line(card: dict) -> str:
    return (f"[m:{card['media_id']}] {card['date'] or 'undated'} · {card['kind']}{_status_note(card)} · {card['title']}"
            f" — {_clip(card.get('summary') or card.get('description') or '', 200)}")


def match_set(conn: sqlite3.Connection, cfg, embed_client, queries: list[str], question: str,
              scopes: list[str]) -> tuple[list[int], int]:
    """(candidate item ids, how many of them matched a keyword search). Candidates are every item any of the
    research's search phrasings matched by keyword, as many again of the closest in meaning to the question
    and of the question's best loose keyword matches
    (at least 50; similarity scores are too compressed for a cutoff, so reading decides relevance), and, when
    a scope is set and no bigger than EXHAUSTIVE_MAX, everything in it. Ordered by fused rank."""
    allowed = search.scope_media_ids(conn, scopes)
    # Every word of a phrasing must appear: falling back to any word would match most of the archive.
    lists = [search.lexical(conn, q, k=5000, allowed=allowed, any_terms=False)
             for q in dict.fromkeys(q.strip() for q in [*queries, question]) if q]
    keyword = search.group_by_media(conn, search.fuse(lists), question, limit=0)[1] if any(lists) else []
    loose = search.group_by_media(conn, search.lexical(conn, question, k=400, allowed=allowed), question,
                                  limit=0)[1][:max(50, len(keyword))]
    meaning: list[int] = []
    try:
        dn = search.dense(conn, embed_client, cfg, question, k=max(400, 6 * len(keyword)), allowed=allowed)
        meaning = search.group_by_media(conn, dn, question, limit=0)[1][:max(50, len(keyword))]
    except (embed.EmbedError, httpx.HTTPError, OSError):
        pass
    ordered = [m for m, _ in search.fuse([[(m, 0.0) for m in keyword], [(m, 0.0) for m in meaning],
                                          [(m, 0.0) for m in loose]])]
    if allowed is not None and len(allowed) <= EXHAUSTIVE_MAX:
        ordered += search.items_in_scope(conn, allowed, limit=EXHAUSTIVE_MAX)
    return list(dict.fromkeys(ordered)), len(keyword)


EXHAUSTIVE_MAX = 1500          # items one exhaustive answer reads at most
EXHAUSTIVE_RECORD_CHARS = 2400
MAP_MAX_TOKENS = 1200
SPLIT_PAUSE_S = 2.0            # after an overflow, give the server a moment before the halves
EXHAUSTIVE_ANSWER_MAX_TOKENS = 2600

MAP_SYSTEM = """You are the research assistant for a private archive of political and social-media material collected by the user. You read archive items in batches and write precise research notes. Do not moralise, add warnings, or refuse because material is offensive."""

MAP_PROMPT = """Question: {question}

Below are {n} archive items (batch {i} of {total}). Write research notes for answering the question from these items only:
- one bullet per distinct fact, claim, event, quote or date that bears on the question, saying who claims it and what kind of item it is, ending with its citation [m:ID];
- merge near-identical items into one bullet carrying all their citations;
- note contradictions, and items that undercut or debunk other claims;
- leave out items unrelated to the question. If none are related, write only: NONE

Items:
{records}"""

REDUCE_PROMPT = """Question: {question}

These are research notes from several batches of archive items. Merge them into one set of notes: keep every distinct point and every [m:ID] citation, combine duplicates (keeping all their citations), keep contradictions visible, and group the bullets under short headings by sub-topic or period. Do not add anything that is not in the notes.

Notes:
{notes}"""


def _batch_chars(cfg) -> int:
    # llama.cpp's automatic slots share one context between concurrent requests, and dense text can run
    # nearer 3 characters a token, so a batch takes well under the whole window.
    return max(8000, int((cfg.chat_context_tokens - PROMPT_OVERHEAD_TOKENS - MAP_MAX_TOKENS) * 3.0 * 0.6))


def _complete(client, cfg, system: str, prompt: str, max_tokens: int, control: Control, what: str = "") -> str:
    body = _payload(cfg, [{"role": "system", "content": system}, {"role": "user", "content": prompt}], max_tokens)
    try:
        return "".join(_stream_chat(client, cfg, body, control, what)).strip()
    except httpx.HTTPError as exc:
        raise AskError(f"chat model failed while reading: {exc}") from exc


def _read_all(conn, cfg, client, embed_client, ev: Evidence, question: str, order: list[int], notes_budget: int,
              control: Control, trace: list[dict]):
    """Map-reduce over every candidate: batches of item records sized to the model's window become cited notes,
    merged until they fit the answer's notes budget. Yields step events; returns (items read, batches, notes)."""
    batch_chars = _batch_chars(cfg)
    ids = order[:EXHAUSTIVE_MAX]
    # Pack lazily, so a stop between batches doesn't wait for records nobody will read.
    batches: list[list[str]] = [[]]
    size = 0
    records_ready = []
    for m in ids:
        record = tool_open_item(conn, cfg, embed_client, ev, [], {"id": m})[:EXHAUSTIVE_RECORD_CHARS]
        if record.startswith("No item"):
            continue
        if size + len(record) > batch_chars and batches[-1]:
            batches.append([])
            size = 0
        batches[-1].append(record)
        size += len(record) + 2
        records_ready.append(m)
    conn.commit()
    batches = [b for b in batches if b]
    notes: list[str] = []
    i = 0
    while i < len(batches):
        batch = batches[i]
        i += 1
        control.check()
        t0 = time.monotonic()
        try:
            text = _complete(client, cfg, MAP_SYSTEM, MAP_PROMPT.format(question=question, n=len(batch), i=i,
                                                                        total=len(batches), records="\n\n".join(batch)),
                             MAP_MAX_TOKENS, control, f"reading batch {i} of {len(batches)} ({len(batch)} items)")
        except AskError as exc:
            # Too big for what the server has free right now: read the batch as two halves instead.
            if len(batch) < 2 or not re.search(r"context|exceed|HTTP 5\d\d", str(exc), re.IGNORECASE):
                raise
            log.warning("batch of %d items failed (%s); splitting it", len(batch), exc)
            batches[i - 1:i] = [batch[:len(batch) // 2], batch[len(batch) // 2:]]
            i -= 1
            time.sleep(SPLIT_PAUSE_S)
            continue
        cited = len(set(re.findall(r"\[m:(\d+)\]", text)))
        if text and text.strip().upper() != "NONE":
            notes.append(text)
        trace.append({"action": "read_batch", "args": {"batch": i, "items": len(batch), "cited": cited},
                      "ms": int((time.monotonic() - t0) * 1000)})
        yield "step", {"n": len(trace), "action": "read", "args": {"id": f"batch {i}/{len(batches)}"}, "thought": "",
                       "summary": f"{len(batch)} items · {cited} cited"}
    merged = "\n\n".join(notes)
    rounds = 0
    while len(merged) > notes_budget and len(notes) > 1 and rounds < 4:
        rounds += 1
        groups: list[list[str]] = [[]]
        size = 0
        for n in notes:
            if size + len(n) > batch_chars and groups[-1]:
                groups.append([])
                size = 0
            groups[-1].append(n)
            size += len(n)
        notes = []
        for j, group in enumerate(groups, 1):
            control.check()
            notes.append(_complete(client, cfg, MAP_SYSTEM, REDUCE_PROMPT.format(question=question, notes="\n\n".join(group)),
                                   MAP_MAX_TOKENS * 2, control, f"merging notes, round {rounds}: {j} of {len(groups)}"))
            yield "step", {"n": len(trace), "action": "merge", "args": {"id": f"round {rounds}, {j}/{len(groups)}"},
                           "thought": "", "summary": ""}
        merged = "\n\n".join(notes)
    trace.append({"action": "reduce", "args": {"rounds": rounds}, "chars": len(merged)})
    return len(records_ready), len(batches), merged[:notes_budget]


def run(conn: sqlite3.Connection, cfg, question: str, scopes: list[str] | None = None, mode: str = "informed",
        session_id: str | None = None, client: httpx.Client | None = None,
        embed_client: httpx.Client | None = None, items: list[int] | None = None, related: bool = False,
        depth: float | None = None, control: Control | None = None,
        debate: dict | None = None, exhaustive: bool = False) -> Iterator[tuple[str, dict]]:
    """One turn. items are what the user attached ("discuss this"); related gathers items around them;
    depth (0..1) sets how much the agent reads and how much of the window the answer's notes may fill.
    debate (argue.py) makes it one speaker's turn: {role, system, answer_system, instruction, label, cited}.
    The question is then the debate prompt, the transcript replaces the session history, and the turn is
    logged under the speaker's label with mode = role."""
    scopes = scopes or []
    mode = mode if mode in ANSWER_INSTRUCTIONS else "informed"
    mode_instruction = ANSWER_INSTRUCTIONS[mode]
    session_id = session_id or uuid.uuid4().hex[:12]
    client = client or httpx.Client()
    embed_client = embed_client or httpx.Client()
    control = control or Control()
    pl = plan(cfg, depth)
    started = time.monotonic()
    ev = Evidence()
    trace: list[dict] = []
    context = {"items": list(items or []), "related": bool(related), "depth": pl.depth}
    if exhaustive:
        context["exhaustive"] = True
    with conn:
        conn.execute("INSERT OR IGNORE INTO chat_session(id, title) VALUES (?, ?)", (session_id, question[:120]))
    system = SYSTEM_PROMPT.replace("{wiki_tools}", WIKI_TOOLS if _has_wiki(conn) else "") \
                          .replace("{scope_note}", _scope_note(conn, scopes))
    past = history(conn, session_id)
    answer_system, log_question = ANSWER_SYSTEM, question
    if debate:
        system += "\n\n" + debate["system"]
        answer_system, log_question, past = debate["answer_system"], debate["label"], []
        mode_instruction = f"{mode_instruction} {debate['instruction']}"
        mode = debate["role"]
        # Items cited earlier in the debate stay citable without being searched again.
        for card in search.cards(conn, list(debate.get("cited") or [])).values():
            ev.items.setdefault(card["media_id"], card)
    loop_budget_chars = cfg.chat_context_tokens * CHARS_PER_TOKEN * 0.7
    # Exhaustive answers read everything in batches, so their notes may fill the whole window.
    notes_tokens = plan(cfg, 1).notes_tokens if exhaustive else pl.notes_tokens
    notes_budget = max(4000, int(notes_tokens * CHARS_PER_TOKEN) - sum(len(m["content"]) for m in past))

    # Attached items go in whole, ahead of the question; related items as one line each, to open if they matter.
    attached = [r for r in (tool_open_item(conn, cfg, embed_client, ev, [], {"id": m}) for m in context["items"])
                if not r.startswith("No item")]
    related_ids: list[int] = []
    if related and context["items"]:
        for m in context["items"]:
            related_ids += [x for x in related_items(conn, cfg, m, pl.related) if x not in related_ids]
        for card in search.cards(conn, related_ids).values():
            ev.items.setdefault(card["media_id"], card)
    conn.commit()
    lead = ""
    if attached:
        lead = "The user is asking about these archive items:\n\n" + "\n\n".join(attached) + "\n\n"
        if related_ids:
            lead += "Related items (near in time or meaning; open_item any that matter):\n" + "\n".join(
                _one_line(ev.items[m]) for m in related_ids if m in ev.items) + "\n\n"
        lead += "Question: "
        yield "step", {"n": 0, "action": "attached", "args": {"id": ", ".join(map(str, context["items"]))},
                       "thought": "", "summary": f"{len(attached)} item(s)" +
                       (f", {len(related_ids)} related" if related_ids else "")}
    messages = [{"role": "system", "content": system}, *past, {"role": "user", "content": lead + question}]

    cites = CitationFilter(ev)
    stopped = False
    try:
        notes: list[str] = []
        opened: set[int] = set(context["items"])
        used = {"search": 0, "open_item": 0}
        for step_no in range(pl.steps):
            t0 = time.monotonic()
            # Steer by narrowing the grammar, not by pleading: searches and opens run out,
            # and the last step can only be an answer.
            actions = [a for a in ACTIONS
                       if not (a == "search" and used["search"] >= pl.searches)
                       and not (a == "open_item" and used["open_item"] >= pl.opens)
                       and not (a in ("find_pages", "open_page") and not _has_wiki(conn))]
            if step_no == pl.steps - 1:
                actions = ["answer"]
            step = next_action(client, cfg, messages, actions, control,
                               f"step {step_no + 1}: choosing what to do" if actions != ["answer"] else "deciding to answer")
            action, args = step["action"], step["args"]
            if action == "answer":
                trace.append({"action": "answer", "thought": step.get("thought", ""), "ms": int((time.monotonic() - t0) * 1000)})
                break
            if action == "search":
                args["limit"] = min(int(args.get("limit") or pl.search_limit), pl.search_limit)
            detail = args.get("query") or args.get("id") or args.get("slug") or ""
            control.live(f"{action.replace('_', ' ')} {detail}".strip(), "tool")
            try:
                result = TOOLS[action](conn, cfg, embed_client, ev, scopes, args)
            except (search.ScopeError, ValueError, TypeError) as exc:
                result = f"error: {exc}"
            # Search helpers fill temp tables, which opens a transaction; end it so the snapshot
            # stays fresh while other stages (ASR, extraction) commit, or the final log write fails.
            conn.commit()
            if action == "open_item" and str(args.get("id", "")).isdigit():
                opened.add(int(args["id"]))
            summary = result.split("\n", 1)[0][:160]
            used[action] = used.get(action, 0) + 1
            notes.append(f"## {action} {json.dumps(args, ensure_ascii=False)}\n{result[:RESULT_CHARS]}")
            trace.append({"action": action, "args": args, "thought": step.get("thought", ""),
                          "chars": len(result), "ms": int((time.monotonic() - t0) * 1000)})
            yield "step", {"n": step_no + 1, "action": action, "args": args, "thought": step.get("thought", ""),
                           "summary": summary}
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            remaining = pl.steps - step_no - 1
            nudge = f"\n\n({remaining} steps left.)" if remaining <= 3 else ""
            messages.append({"role": "user", "content": f"Tool result ({action}):\n{result[:RESULT_CHARS]}{nudge}"})
            if sum(len(m["content"]) for m in messages) > loop_budget_chars:
                break

        notes = [f"## attached by the user\n{r}" for r in attached] + notes
        t0 = time.monotonic()
        queries = [str(t["args"]["query"]) for t in trace
                   if t["action"] in ("search", "timeline", "topics") and (t.get("args") or {}).get("query")]
        control.live("gathering every item the research matched", "tool")
        candidates, n_keyword = match_set(conn, cfg, embed_client, queries, question, scopes)
        conn.commit()
        order = list(dict.fromkeys(related_ids + [m for m in ev.items if m not in related_ids] + candidates))
        order = [m for m in order if m not in context["items"]]
        coverage = {"candidates": len(set(candidates) | set(related_ids) | set(ev.items)), "keyword": n_keyword}
        if exhaustive:
            read, map_notes, reduced = yield from _read_all(conn, cfg, client, embed_client, ev, question, order, notes_budget,
                                                            control, trace)
            # The agent's own research stays, but the batch notes carry the answer.
            loop_notes = "\n\n".join(notes)[: notes_budget // 4]
            notes = [loop_notes, "## notes from reading every candidate item\n" + reduced] if loop_notes else [reduced]
            coverage |= {"read": read + len(attached), "batches": map_notes, "capped": len(order) > EXHAUSTIVE_MAX}
        else:
            # Fill the notes up to the depth's budget: related items first, then what the searches found, then the rest.
            used_chars = sum(len(n) for n in notes)
            extra = []
            for m in order:
                control.check()
                if used_chars >= notes_budget:
                    break
                if m in opened:
                    continue
                record = tool_open_item(conn, cfg, embed_client, ev, [], {"id": m})[:RECORD_CHARS]
                if record.startswith("No item") or used_chars + len(record) > notes_budget:
                    continue
                opened.add(m)
                extra.append(record)
                used_chars += len(record) + 2
            conn.commit()
            if extra:
                notes.append("## more items from the archive (related, found, or in scope)\n" + "\n\n".join(extra))
                trace.append({"action": "fill", "args": {"items": len(extra)}, "chars": sum(map(len, extra)),
                              "ms": int((time.monotonic() - t0) * 1000)})
                yield "step", {"n": len(trace), "action": "read", "args": {"id": f"{len(extra)} more items"},
                               "thought": "", "summary": f"~{used_chars // CHARS_PER_TOKEN / 1000:.1f}k tokens of notes"}
            coverage["read"] = len(opened)
        coverage["candidates"] = max(coverage["candidates"], coverage["read"])
        trace.append({"action": "coverage", "args": coverage})
        context["coverage"] = coverage
        yield "coverage", coverage

        # A fresh conversation for the answer: after a run of JSON steps the model otherwise
        # keeps writing JSON. The research becomes notes in a single message.
        answer_messages = [{"role": "system", "content": answer_system}, *past,
                           {"role": "user", "content": ANSWER_PROMPT.format(
                               question=question, notes="\n\n".join(notes)[:notes_budget] or "(no research)",
                               mode_instruction=mode_instruction)}]

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

        for delta in stream_answer(client, cfg, answer_messages, control,
                                   EXHAUSTIVE_ANSWER_MAX_TOKENS if exhaustive else ANSWER_MAX_TOKENS):
            out = cites.feed(delta)
            yield from emit_cites()
            if out:
                yield "token", {"text": out}
        tail = cites.flush()
        yield from emit_cites()
        if tail:
            yield "token", {"text": tail}
    except Stopped:
        stopped = True
    except AskError as exc:
        yield "error", {"msg": str(exc)}
        return

    answer = search.scrub_text(cites.text.strip())
    latency = int((time.monotonic() - started) * 1000)
    conn.commit()
    if stopped:
        trace.append({"action": "stopped", "ms": latency})
        if not answer:
            yield "stopped", {}
            yield "done", {"session_id": session_id, "ms": latency, "steps": len(trace), "stopped": True}
            return
        answer += " (stopped)"
    citations = [{"n": n, "ref": ref} for ref, n in cites.numbers.items()]
    with conn:
        conn.execute(
            """INSERT INTO chat_log(question, retrieved_doc_ids, answer, citations_json, model, latency_ms,
                                    session_id, mode, trace_json, context_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log_question, json.dumps(sorted(ev.items)), answer, json.dumps(citations), cfg.chat_model, latency,
             session_id, mode, json.dumps(trace, ensure_ascii=False), json.dumps(context)))
    if stopped:
        yield "stopped", {}
    yield "final", {"text": answer, "sentences": sentences_with_flags(answer)}
    yield "done", {"session_id": session_id, "ms": latency, "steps": len(trace), "stopped": stopped}


# ── Runs: a turn keeps going when the browser looks away, and can be watched again ──

KEEP_FINISHED_S = 600


class Run:
    def __init__(self, session_id: str, question: str, context: dict):
        self.id = uuid.uuid4().hex[:12]
        self.session_id = session_id
        self.question = question
        self.context = context
        self.started = time.time()
        self.finished: float | None = None
        self.control = Control()
        self.control.report = self.set_live
        self.events: list[tuple[str, dict]] = []
        self.live: dict | None = None   # what the run is doing right now; only the latest is kept
        self._live_seq = 0
        self._cond = threading.Condition()

    def add(self, name: str, data: dict) -> None:
        with self._cond:
            self.events.append((name, data))
            self._cond.notify_all()

    def set_live(self, status: dict) -> None:
        with self._cond:
            self.live = status
            self._live_seq += 1
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.finished = time.time()
            self._cond.notify_all()

    def follow(self, start: int = 0, heartbeat_s: float = 15.0) -> Iterator[tuple[str, dict] | None]:
        """Every event from start on, as it happens, then ("live", ...) whenever what it is doing changes;
        None now and then while nothing does."""
        i, seen = start, -1
        while True:
            with self._cond:
                if i >= len(self.events) and self.finished is None and self._live_seq == seen:
                    self._cond.wait(heartbeat_s)
                batch, finished = self.events[i:], self.finished is not None
                live = self.live if self._live_seq != seen and self.live else None
                seen = self._live_seq
            if not batch and not finished and live is None:
                yield None
            for event in batch:
                yield event
            if live is not None and not finished:
                yield "live", live
            i += len(batch)
            if finished and i >= len(self.events):
                return

    def summary(self) -> dict:
        # A debate logs each speaker's turn as it finishes; a returning viewer resumes at the current speaker.
        turn_start = max((i for i, (name, _) in enumerate(self.events) if name == "speaker"), default=0)
        return {"run_id": self.id, "session_id": self.session_id, "question": self.question,
                "context": self.context, "started": self.started, "done": self.finished is not None,
                "turn_start": turn_start}


RUNS: dict[str, Run] = {}
_runs_lock = threading.Lock()


def start_run(cfg, question: str, scopes: list[str], mode: str, session_id: str | None,
              items: list[int] | None = None, related: bool = False, depth: float | None = None,
              client: httpx.Client | None = None, embed_client: httpx.Client | None = None,
              exhaustive: bool = False) -> Run:
    """Run one turn on a thread of its own, so it finishes (and is logged) whoever is watching."""
    session_id = session_id or uuid.uuid4().hex[:12]
    context = {"items": list(items or []), "related": related, "depth": depth, "scopes": scopes, "mode": mode,
               "exhaustive": exhaustive}
    return start_background(cfg, session_id, question, context, lambda conn, control: run(
        conn, cfg, question, scopes=scopes, mode=mode, session_id=session_id, client=client,
        embed_client=embed_client, items=items, related=related, depth=depth, control=control, exhaustive=exhaustive))


def start_background(cfg, session_id: str, question: str, context: dict, events) -> Run:
    """events(conn, control) yields (name, data); it runs on a thread with its own connection."""
    r = Run(session_id, question, context)
    with _runs_lock:
        for rid in [k for k, v in RUNS.items() if v.finished and time.time() - v.finished > KEEP_FINISHED_S]:
            del RUNS[rid]
        RUNS[r.id] = r
    r.add("start", {"run_id": r.id, "session_id": session_id})

    def work():
        conn = db.connect(cfg.db_path)
        try:
            for name, data in events(conn, r.control):
                r.add(name, data)
        except Exception as exc:  # noqa: BLE001 - whoever watches must see an end, not silence
            log.exception("ask run %s failed", r.id)
            r.add("error", {"msg": str(exc)})
        finally:
            conn.close()
            r.finish()

    threading.Thread(target=work, name=f"ask-{r.id}", daemon=True).start()
    return r


def active_runs() -> list[Run]:
    with _runs_lock:
        return [r for r in RUNS.values() if r.finished is None]
