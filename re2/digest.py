"""Phase 6.3: a short title, summary and event hint for every item, and the records the wiki reads.

Items captioned with the current image prompt already carry a title and summary; those are copied,
with no model call. Everything else (images with only a 2024 caption, videos with transcripts,
PDFs) goes to the [wiki] text model in batches, one JSON reply per batch. A digest remembers the
hash of what it was made from, so when a new caption or transcript arrives the item is due again
(and a new current caption replaces a model digest for free).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field

from re2 import db, runner, search, writer

log = logging.getLogger(__name__)

STAGE = "digest"
DIGEST_VERSION = 1        # bump when the input or prompt changes enough to redo model digests
BATCH_ITEMS = 10
INPUT_CHARS = 2400        # one item's input
TOKENS_PER_ITEM = 170
RECORD_CHARS = 900        # one item's record in a wiki prompt

SYSTEM_PROMPT = """\
You index items in a private archive of political and social-media material (screenshots, memes, photos, videos, documents). For each numbered item, write:
- title: at most 12 words, specific and factual, naming the key people, places and events (e.g. "Trudeau says Ukraine may use Canadian long-range missiles in Russia"). No clickbait, no editorialising.
- summary: one to three sentences stating what the item shows and claims. Attribute claims to their source ("the post claims ..."). Do not soften offensive content or add warnings.
- event_hint: a short name for the specific real-world event or story the item is about, or "none" if it is not about a specific event.
Some descriptions come from an older captioner that hedges about race and gender; ignore those hedges."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "title": {"type": "string", "maxLength": 160},
                       "summary": {"type": "string", "maxLength": 900},
                       "event_hint": {"type": "string", "maxLength": 160}},
        "required": ["n", "title", "summary", "event_hint"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


def _clip(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


@dataclass
class ItemText:
    """What the archive holds about one item (a post's photos and videos included)."""
    media_id: int
    kind: str
    date: str | None = None
    date_source: str | None = None
    folder: str = "."
    source_url: str | None = None
    caption: dict | None = None          # the item's own caption JSON, any prompt version
    member_captions: list[dict] = field(default_factory=list)
    legacy: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    pdf: list[str] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    digest: dict | None = None           # item_digest row

    def has_text(self) -> bool:
        return bool(self.caption or self.member_captions or self.legacy or self.transcript or self.pdf
                    or self.keyframes)

    def current_caption(self) -> dict | None:
        from re2.extract_images import PROMPT_VERSION

        c = self.caption
        if c and c.get("title") and c.get("summary") and (c.get("prompt_version") or 0) >= PROMPT_VERSION:
            return c
        return None


def gather(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, ItemText]:
    """ItemText for each item id (a post primary or a standalone media), in a handful of queries."""
    if not item_ids:
        return {}
    _, members = search.post_maps(conn)
    owner = {}
    for item in item_ids:
        for m in members.get(item, [item]):
            owner.setdefault(m, item)
    table = search._temp_ids(conn, "digest_media", list(owner))
    items: dict[int, ItemText] = {}
    for mid, kind in conn.execute(f"SELECT id, kind FROM media WHERE id IN (SELECT id FROM {table})"):
        if owner[mid] == mid:
            items[mid] = ItemText(mid, kind)
    has_date = search._has_column(conn, "media_meta", "item_date")
    if has_date:
        for mid, d, src in conn.execute(
                f"SELECT media_id, item_date, date_source FROM media_meta WHERE media_id IN (SELECT id FROM {table})"):
            if mid in items:
                items[mid].date, items[mid].date_source = d, src
    for mid, url, date in conn.execute(
            f"SELECT media_id, url, date FROM source WHERE media_id IN (SELECT id FROM {table}) ORDER BY id"):
        it = items.get(owner[mid])
        if it and not it.source_url:
            it.source_url = url
        if it and not it.date and date:
            it.date, it.date_source = date[:10], "source"
    for mid, rel_path in conn.execute(
            f"SELECT media_id, MIN(rel_path) FROM media_path WHERE media_id IN (SELECT id FROM {table}) GROUP BY media_id"):
        if mid in items:
            items[mid].folder = os.path.dirname(rel_path) or "."
    for mid, kind, seq, text, raw in conn.execute(
            f"""SELECT media_id, kind, seq, text, json FROM doc WHERE media_id IN (SELECT id FROM {table})
                AND kind IN ('caption', 'legacy_caption', 'transcript', 'pdf_page', 'keyframe')
                ORDER BY media_id, kind, seq"""):
        it = items.get(owner[mid])
        if it is None:
            continue
        if kind == "caption":
            d = json.loads(raw) if raw else {"description": text}
            if mid == it.media_id:
                it.caption = d
            else:
                it.member_captions.append(d)
        elif kind == "legacy_caption":
            it.legacy.append(text)
        elif kind == "transcript":
            it.transcript.append(text)
        elif kind == "pdf_page":
            it.pdf.append(text)
        else:
            it.keyframes.append(text)
    for mid, name in conn.execute(
            f"""SELECT me.media_id, e.name FROM media_entity me JOIN entity e ON e.id=me.entity_id
                WHERE me.media_id IN (SELECT id FROM {table}) ORDER BY e.name"""):
        it = items.get(owner[mid])
        if it and name not in it.entities:
            it.entities.append(name)
    if search._has_table(conn, "item_digest"):
        for r in conn.execute(
                f"SELECT media_id, title, summary, digest_json, source, input_hash FROM item_digest "
                f"WHERE media_id IN (SELECT id FROM {table})"):
            if r[0] in items:
                items[r[0]].digest = {"title": r[1], "summary": r[2], "source": r[4], "input_hash": r[5],
                                      **json.loads(r[3] or "{}")}
    return items


def digest_input(it: ItemText) -> str:
    captions = [c for c in [it.caption, *it.member_captions] if c]
    lines = [f"Kind: {it.kind}   Date: {it.date or 'unknown'}   Folder: {it.folder}",
             f"Source URL: {it.source_url or 'none'}"]
    desc = " | ".join(_clip(c.get("summary") or c.get("description"), 700) for c in captions)
    if not desc and it.legacy:
        desc = _clip(" ".join(it.legacy), 900)
    if desc:
        lines.append(f"Description: {desc}")
    ocr = " | ".join(_clip(c.get("ocr_text"), 600) for c in captions if (c.get("ocr_text") or "").strip())
    if ocr:
        lines.append(f"Text in image: {ocr}")
    claims = "; ".join(x for c in captions for x in (c.get("claims") or []))
    if claims:
        lines.append(f"Claims: {_clip(claims, 500)}")
    if it.transcript:
        lines.append(f"Transcript (start): {_clip(' '.join(it.transcript), 1200)}")
    if it.pdf:
        lines.append(f"Document text (start): {_clip(' '.join(it.pdf), 1200)}")
    if it.keyframes:
        lines.append(f"Keyframes: {_clip(' | '.join(it.keyframes), 500)}")
    if it.entities:
        lines.append(f"Entities: {', '.join(it.entities[:15])}")
    return "\n".join(lines)[:INPUT_CHARS]


def input_hash(it: ItemText) -> str:
    return hashlib.sha1(f"v{DIGEST_VERSION}\n{digest_input(it)}".encode()).hexdigest()


def title_of(it: ItemText) -> str:
    cap = it.current_caption()
    return (it.digest or {}).get("title") or (cap or {}).get("title") or \
        search.title_from_text((it.caption or {}).get("description")) or \
        search.title_from_text(" ".join(it.legacy)) or \
        search.title_from_text(" ".join(it.transcript)) or f"{it.kind} in {it.folder}"


def record(it: ItemText) -> str:
    """One item as the wiki's writer reads it: [m:ID], date, kind, title, summary and the telling details."""
    cap = it.current_caption()
    title = title_of(it)
    summary = (it.digest or {}).get("summary") or (cap or {}).get("summary") or \
        _clip((it.caption or {}).get("description") or " ".join(it.legacy), 500)
    weak = "~" if it.date_source == "mtime" else ""
    lines = [f"[m:{it.media_id}] {weak}{it.date or 'undated'} · {it.kind} · {title}", _clip(summary, 500)]
    claims = "; ".join(x for c in [it.caption, *it.member_captions] if c for x in (c.get("claims") or []))
    if claims:
        lines.append(f"Claims: {_clip(claims, 260)}")
    ocr = (it.caption or {}).get("ocr_text")
    if ocr and ocr.strip():
        lines.append(f"Text: {_clip(ocr, 220)}")
    if it.transcript:
        lines.append(f"Says: {_clip(' '.join(it.transcript), 260)}")
    elif it.pdf:
        lines.append(f"Document: {_clip(' '.join(it.pdf), 260)}")
    if it.source_url:
        lines.append(f"Source: {it.source_url}")
    return "\n".join(lines)[:RECORD_CHARS]


def record_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def event_hint(it: ItemText) -> str:
    if it.digest and it.digest.get("event_hint"):
        return it.digest["event_hint"]
    events = ((it.current_caption() or {}).get("entities") or {}).get("events") or []
    return events[0] if events else "none"


# ── Selecting and running ─────────────────────────────────────────────────

def item_ids(conn: sqlite3.Connection) -> list[int]:
    """Every item (standalone media and post primaries): topic members first, grouped by topic and date,
    so a batch holds related items and the wiki's material is ready first."""
    to_primary, _ = search.post_maps(conn)
    rows = conn.execute(
        f"""SELECT m.id, {search.item_date_sql('m') if search._has_column(conn, 'media_meta', 'item_date') else 'NULL'}
            FROM media m WHERE m.kind IN ('image', 'video', 'pdf') AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED')""").fetchall()
    run = db.kv_get(conn, "cluster_run") if search._has_table(conn, "cluster") else None
    topic_of = dict(conn.execute(
        """SELECT cm.media_id, c.id FROM cluster_member cm JOIN cluster c ON c.id=cm.cluster_id
           WHERE c.run_id=? AND c.level='topic'""", (run,)).fetchall()) if run else {}
    items = [(m, d) for m, d in rows if to_primary.get(m, m) == m]
    items.sort(key=lambda r: (r[0] not in topic_of, topic_of.get(r[0], 0), r[1] or "9999", r[0]))
    return [m for m, _ in items]


@dataclass
class DigestStats:
    items: int = 0
    copied: int = 0
    due: int = 0
    selected: int = 0
    done: int = 0
    failed: int = 0
    batches: int = 0
    interrupted: bool = False
    started: float = 0.0

    def summary(self) -> str:
        rate = self.done * 60 / max(time.monotonic() - self.started, 1e-9)
        return (f"items={self.items} copied_from_captions={self.copied} due={self.due} selected={self.selected} "
                f"done={self.done} failed={self.failed} batches={self.batches} rate={rate:.1f}/min"
                + (" INTERRUPTED" if self.interrupted else ""))


def store(conn: sqlite3.Connection, it: ItemText, title: str, summary: str, hint: str, source: str,
          model: str | None) -> None:
    conn.execute(
        """INSERT INTO item_digest(media_id, title, summary, item_date, date_source, digest_json, source, input_hash, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(media_id) DO UPDATE SET title=excluded.title, summary=excluded.summary,
             item_date=excluded.item_date, date_source=excluded.date_source, digest_json=excluded.digest_json,
             source=excluded.source, input_hash=excluded.input_hash, model=excluded.model, created_at=datetime('now')""",
        (it.media_id, " ".join(title.split())[:200], " ".join(summary.split()), it.date, it.date_source or "none",
         json.dumps({"event_hint": " ".join((hint or "none").split())[:160]}, ensure_ascii=False), source,
         input_hash(it), model))


def batches(cfg, items: list[ItemText]) -> list[list[ItemText]]:
    budget = writer.input_chars(cfg, TOKENS_PER_ITEM * BATCH_ITEMS + 200)
    out: list[list[ItemText]] = [[]]
    size = 0
    for it in items:
        n = len(digest_input(it)) + 12
        if out[-1] and (len(out[-1]) >= BATCH_ITEMS or size + n > budget):
            out.append([])
            size = 0
        out[-1].append(it)
        size += n
    return [b for b in out if b]


def digest_batch(client, cfg, batch: list[ItemText]) -> dict[int, dict]:
    user = "\n\n".join(f"### {i}\n{digest_input(it)}" for i, it in enumerate(batch, 1))
    user += f"\n\nWrite one entry for each of the {len(batch)} items above, in order, with n = the item's number."
    body = writer.payload(cfg, SYSTEM_PROMPT, user, TOKENS_PER_ITEM * len(batch) + 200, schema=SCHEMA, temperature=0.2)
    out: dict[int, dict] = {}
    for entry in writer.complete_json(client, cfg, body).get("items") or []:
        n = entry.get("n") if isinstance(entry, dict) else None
        if isinstance(n, int) and 1 <= n <= len(batch) and n not in out and entry.get("title") and entry.get("summary"):
            out[n] = entry
    return out


def pending(conn: sqlite3.Connection) -> tuple[int, int]:
    """(items run() would copy from a current caption, items it would send to the model), without writing."""
    ids = item_ids(conn)
    texts = gather(conn, ids)
    conn.commit()
    skip = db.pending_failures(conn, STAGE)
    copy = due = 0
    for m in ids:
        it = texts.get(m)
        if it is None or not it.has_text() or (it.digest and it.digest["input_hash"] == input_hash(it)):
            continue
        if it.current_caption():
            copy += 1
        elif m not in skip:
            due += 1
    return copy, due


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, client_factory=None,
        concurrency: int | None = None) -> DigestStats:
    import httpx

    stats = DigestStats(started=time.monotonic())
    ids = item_ids(conn)
    texts = gather(conn, ids)
    conn.commit()
    skip = db.pending_failures(conn, STAGE)
    due: list[ItemText] = []
    with conn:
        for m in ids:
            it = texts.get(m)
            if it is None or not it.has_text():
                continue
            stats.items += 1
            if it.digest and it.digest["input_hash"] == input_hash(it):
                continue
            cap = it.current_caption()
            if cap:
                store(conn, it, cap["title"], cap["summary"], event_hint(it), "caption", None)
                stats.copied += 1
            elif m not in skip:
                due.append(it)
    stats.due = len(due)
    if limit is not None:
        due = due[:limit]
    stats.selected = len(due)
    jobs = batches(cfg, due)
    log.info("%d items copied from current captions; %d due for the model in %d batches (%s at %s)",
             stats.copied, len(due), len(jobs), cfg.wiki_model, cfg.wiki_base_url)
    clients = writer.Clients(client_factory or httpx.Client)

    def on_result(batch: list[ItemText], entries: dict[int, dict]) -> None:
        with conn:
            for n, it in enumerate(batch, 1):
                e = entries.get(n)
                if e is None:
                    db.upsert_failure(conn, STAGE, it.media_id, None, "no entry in the model's reply")
                    stats.failed += 1
                    continue
                store(conn, it, e["title"], e["summary"], e.get("event_hint"), "model", cfg.wiki_model)
                db.clear_failure(conn, STAGE, it.media_id)
                stats.done += 1
        stats.batches += 1
        if stats.batches % 10 == 0:
            log.info("progress: %s", stats.summary())

    def on_error(batch: list[ItemText], exc: BaseException) -> None:
        with conn:
            for it in batch:
                db.upsert_failure(conn, STAGE, it.media_id, None, f"{type(exc).__name__}: {exc}"[:2000])
        stats.failed += len(batch)
        log.warning("digest batch of %d failed: %s", len(batch), exc)

    try:
        stats.interrupted = not runner.run_jobs(jobs, lambda b: digest_batch(clients.get(), cfg, b), on_result,
                                                on_error, concurrency or cfg.wiki_concurrency, total=len(jobs),
                                                desc="digests")
    finally:
        clients.close()
    return stats

