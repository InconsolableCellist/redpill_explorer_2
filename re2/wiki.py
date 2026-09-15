"""Phase 6.6–6.8: the compiled wiki — events, page writing, incremental updates and Markdown export.

The hierarchy is overview → theme → topic → event → item, with entity pages across it. Topics and
themes come from the latest clustering run; the text model groups each topic's items into events.

Pages cover every member, not a sample. An event reads all its items' records: directly when they
fit one request, else in batches that become cited notes (map), merged if the notes are long
(reduce), then written. A topic reads its written event pages in full plus notes on the items no
event holds; a theme reads its topic pages. So a page's coverage is exact and recorded:
members_read of members_total, and per member the hash of the record its writer read, which is
how later captions and digests make pages stale.

Commands: `re2 wiki-update` (no model: sync pages with clusters and entities, attach new items,
mark stale pages), `re2 wiki-build` (update, split topics into events, write stale pages) and
`re2 wiki-export`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from re2 import db, digest, runner, search, writer

log = logging.getLogger(__name__)

STAGE = "wiki"
NOTES_VERSION = 1
ENTITY_MIN_ITEMS = 8
JACCARD_KEEP = 0.5          # a recomputed cluster or event keeps an existing page's slug and row at this overlap
STALE_FRACTION = 0.10       # captions arrive for days: rewrite a page once a tenth of what it read has changed
WAIT_SPLIT = 10             # new items waiting in a topic before they are grouped into events
ATTACH_DAYS = 14
DIRECT_MAX_ITEMS = 40       # up to this many records are written from directly; more are read in batches first
MAP_BATCH_ITEMS = 40
MAP_MAX_TOKENS = 1800
REDUCE_MAX_TOKENS = 2400
WRITE_MAX_TOKENS = 2500
SPLIT_LINES = 400
KINDS = ("event", "topic", "theme", "entity", "overview")
LENGTHS = {"event": "250-700", "topic": "300-900", "entity": "300-900", "theme": "400-1000", "overview": "600-1200"}

SYSTEM_PROMPT = """\
You are writing a page of a private wiki that documents what a personal archive of political and social-media material contains. Readers use it to recall what they saved and to find the original items.

Rules:
- Write in plain, neutral encyclopedic English. Describe what the archived items show and claim; attribute claims to their sources ("a post by @user claims ..."). Say when items contradict each other.
- Every sentence that states something from the archive ends with one or more citations in the form [m:1234], using only the item ids provided. Link to other wiki pages only as [p:slug], using only the slugs provided.
- You may add brief, clearly general background (who a person is, when an event happened) without citations, but never invent details about the archived items.
- Do not moralise, add warnings, or soften offensive material; quote it where it matters.
- Structure: a one-paragraph lead, then sections with ## headings as the material warrants. End with no conclusion or summary section."""

NOTES_SYSTEM = """\
You read items from a private archive of political and social-media material and take precise notes for a wiki page about them. Plain and neutral: describe and attribute, do not moralise, add warnings or soften offensive material."""

MAP_PROMPT = """\
Wiki page: {kind} "{title}"{dates}
Below are items {first}-{last} of the {total} items on this page (batch {i} of {batches}).

{records}

Write notes for the page from these items:
- one bullet per distinct fact, claim, quote, date or framing, saying who claims it and what kind of item shows it, ending with its citations [m:ID];
- merge near-identical items into one bullet carrying all their citations; every item should be cited at least once;
- keep contradictions and items that dispute other items visible;
- after the bullets, one line "Off-topic: [m:ID] ..." for items that do not belong on this page, if any."""

REDUCE_PROMPT = """\
Wiki page: {kind} "{title}"{dates}
These are notes from several batches of the page's items. Merge them into one set of notes: keep every distinct point and every [m:ID] citation, combine duplicates (keeping all their citations), keep contradictions visible, and group the bullets under short headings by sub-topic or period. Do not add anything that is not in the notes.

{notes}"""

KIND_GUIDE = {
    "event": "Sections as the material warrants, such as: What happened, Claims and framing, Timeline, Reactions.",
    "topic": "Give an overview of the topic, its main events and how the archive's items frame them. Link each event page where you discuss it.",
    "theme": "Give an overview of the theme across its topics, what connects them, and how coverage changes over time. Link each topic page where you discuss it.",
    "entity": "Describe how the archive's items present this entity: roles, statements attributed to it, events it appears in, and how different items portray it. Link event pages where relevant.",
    "overview": "Describe what the archive as a whole contains: its themes, periods and kinds of material. Link each theme page.",
}

WRITE_PROMPT = """\
Write the wiki page for this {kind}.
{title_line}{dates}
It is about {total} archive items{coverage}.
{children}{material}
{guide}
Length: {length} words. Output Markdown only. {start}"""

SPLIT_SYSTEM = """\
You organise a private archive of political and social-media material into a wiki. Be specific and neutral; do not moralise."""

SPLIT_PROMPT = """\
These are {n} items of one topic, one per line: number | date (~ marks a weak date from a file's timestamp) | title | people and organisations | event hint.

{lines}

Group the lines into specific real-world events or stories: the same incident, statement, controversy or news cycle, normally within a few weeks.
- Every line number appears exactly once: in one event's members or in not_events.
- An event needs at least 2 members. Items about a general subject rather than a specific story go in not_events.
- Event titles: at most 12 words, specific and factual, naming the key people, places and events. Summaries: one or two sentences on what the event is and what the items show.
- topic_title: a specific, factual title of at most 8 words for everything above."""

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_title": {"type": "string", "maxLength": 120},
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string", "maxLength": 160},
                           "summary": {"type": "string", "maxLength": 700},
                           "members": {"type": "array", "items": {"type": "integer"}}},
            "required": ["title", "summary", "members"], "additionalProperties": False}},
        "not_events": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["topic_title", "events", "not_events"], "additionalProperties": False,
}


# ── Slugs and small helpers ────────────────────────────────────────────────

def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_len].rstrip("-") or "untitled"


def unique_slug(conn: sqlite3.Connection, base: str, page_id: int | None = None) -> str:
    slug, n = base, 2
    while True:
        row = conn.execute("SELECT id FROM wiki_page WHERE slug=?", (slug,)).fetchone()
        if row is None or row[0] == page_id:
            return slug
        slug, n = f"{base}-{n}", n + 1


def _hash(*parts) -> str:
    return hashlib.sha1("\n".join(map(str, parts)).encode()).hexdigest()


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _members(conn: sqlite3.Connection, page_id: int) -> dict[int, dict]:
    return {r[0]: {"origin": r[1], "read_hash": r[2], "added_at": r[3]} for r in conn.execute(
        "SELECT media_id, origin, read_hash, added_at FROM wiki_member WHERE page_id=?", (page_id,))}


def _pages(conn: sqlite3.Connection, where: str = "1=1", args=()) -> list[dict]:
    return [dict(r) for r in conn.execute(f"SELECT * FROM wiki_page WHERE {where} ORDER BY id", args)]


def date_range(items: list[digest.ItemText]) -> tuple[str | None, str | None]:
    """First and last member date, ignoring weak file-timestamp dates unless there are no others."""
    strong = [it.date for it in items if it.date and it.date_source != "mtime"]
    dates = strong or [it.date for it in items if it.date]
    return (min(dates), max(dates)) if dates else (None, None)


def _set_members(conn: sqlite3.Connection, page_id: int, wanted: set[int], origin: str,
                 keep_other_origins: bool = True) -> None:
    """Members of one origin become exactly `wanted`; others stay (items attached by wiki-update)."""
    have = _members(conn, page_id)
    for m in wanted - set(have):
        conn.execute("INSERT INTO wiki_member(page_id, media_id, origin) VALUES (?, ?, ?)", (page_id, m, origin))
    for m, info in have.items():
        if m not in wanted and (info["origin"] == origin or not keep_other_origins):
            conn.execute("DELETE FROM wiki_member WHERE page_id=? AND media_id=?", (page_id, m))


def delete_page(conn: sqlite3.Connection, page_id: int) -> None:
    for table, col in (("wiki_member", "page_id"), ("wiki_citation", "page_id"), ("wiki_link", "from_page"),
                       ("wiki_link", "to_page"), ("wiki_note", "page_id")):
        conn.execute(f"DELETE FROM {table} WHERE {col}=?", (page_id,))
    conn.execute("DELETE FROM wiki_page WHERE id=?", (page_id,))


# ── Sync with clusters and entities (no model) ─────────────────────────────

@dataclass
class UpdateStats:
    topics: int = 0
    themes: int = 0
    entities: int = 0
    created: int = 0
    deleted: int = 0
    attached: int = 0
    attached_to_events: int = 0
    stale: int = 0
    by_kind: dict = field(default_factory=dict)

    def summary(self) -> str:
        kinds = " ".join(f"{k}={v['pages']}/stale:{v['stale']}" for k, v in self.by_kind.items())
        return (f"topics={self.topics} themes={self.themes} entities={self.entities} created={self.created} "
                f"deleted={self.deleted} attached={self.attached} (to events {self.attached_to_events}) "
                f"stale={self.stale} {kinds}")


def _match_clusters(conn: sqlite3.Connection, kind: str, clusters: dict[int, set[int]]) -> dict[int, int]:
    """cluster id -> existing page id: the same cluster, else the page whose members overlap most (Jaccard ≥ 0.5)."""
    pages = {p["id"]: p for p in _pages(conn, "kind=?", (kind,))}
    member_sets = {pid: {r[0] for r in conn.execute("SELECT media_id FROM wiki_member WHERE page_id=?", (pid,))}
                   for pid in pages}
    out: dict[int, int] = {}
    used: set[int] = set()
    for cid in clusters:
        for pid, p in pages.items():
            if p["cluster_id"] == cid and pid not in used:
                out[cid] = pid
                used.add(pid)
                break
    for cid, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if cid in out:
            continue
        best = max(((_jaccard(members, member_sets[pid]), pid) for pid in pages if pid not in used), default=(0, None))
        if best[1] is not None and best[0] >= JACCARD_KEEP:
            out[cid] = best[1]
            used.add(best[1])
    return out


def sync(conn: sqlite3.Connection, stats: UpdateStats | None = None) -> UpdateStats:
    stats = stats or UpdateStats()
    run = db.kv_get(conn, "cluster_run") if search._has_table(conn, "cluster") else None
    with conn:
        if not conn.execute("SELECT 1 FROM wiki_page WHERE slug='overview'").fetchone():
            conn.execute("INSERT INTO wiki_page(slug, kind, title, named) VALUES ('overview', 'overview', 'Overview', 1)")
            stats.created += 1
        if run:
            rows = conn.execute("SELECT id, level, parent_id, label FROM cluster WHERE run_id=?", (run,)).fetchall()
            members: dict[int, set[int]] = {r[0]: set() for r in rows}
            for cid, m in conn.execute(
                    "SELECT cm.cluster_id, cm.media_id FROM cluster_member cm JOIN cluster c ON c.id=cm.cluster_id "
                    "WHERE c.run_id=?", (run,)):
                members[cid].add(m)
            clustered = {m for r in rows if r[1] == "topic" for m in members[r[0]]}
            slug_of: dict[int, str] = {}
            for level in ("theme", "topic"):
                level_rows = [r for r in rows if r[1] == level]
                matched = _match_clusters(conn, level, {r[0]: members[r[0]] for r in level_rows})
                keep = set(matched.values())
                for cid, _, parent, label in level_rows:
                    parent_slug = slug_of.get(parent) if level == "topic" else "overview"
                    pid = matched.get(cid)
                    if pid is None:
                        slug = unique_slug(conn, f"{level}/{slugify(label or f'cluster {cid}')}")
                        pid = conn.execute(
                            "INSERT INTO wiki_page(slug, kind, title, cluster_id, parent_slug) VALUES (?, ?, ?, ?, ?)",
                            (slug, level, label or f"{level} {cid}", cid, parent_slug)).lastrowid
                        stats.created += 1
                        keep.add(pid)
                    else:
                        conn.execute("UPDATE wiki_page SET cluster_id=?, parent_slug=? WHERE id=?", (cid, parent_slug, pid))
                    slug_of[cid] = conn.execute("SELECT slug FROM wiki_page WHERE id=?", (pid,)).fetchone()[0]
                    _set_members(conn, pid, members[cid], "cluster")
                    # An item attached by wiki-update that a new clustering placed elsewhere leaves.
                    for m, info in _members(conn, pid).items():
                        if info["origin"] == "attached" and m in clustered and m not in members[cid]:
                            conn.execute("DELETE FROM wiki_member WHERE page_id=? AND media_id=?", (pid, m))
                    setattr(stats, level + "s", getattr(stats, level + "s") + 1)
                # Topics and themes the new clustering no longer has go; their events are re-parented below.
                for p in _pages(conn, "kind=?", (level,)):
                    if p["id"] not in keep:
                        delete_page(conn, p["id"])
                        stats.deleted += 1
            topic_members = {p["slug"]: set(_members(conn, p["id"])) for p in _pages(conn, "kind='topic'")}
            for ev in _pages(conn, "kind='event'"):
                em = set(_members(conn, ev["id"]))
                best = max(((len(em & tm), slug) for slug, tm in topic_members.items()), default=(0, None))
                parent = best[1] if best[0] else None
                if parent != ev["parent_slug"]:
                    conn.execute("UPDATE wiki_page SET parent_slug=? WHERE id=?", (parent, ev["id"]))
        _sync_entities(conn, stats)
        _refresh_dates(conn)
    return stats


def _sync_entities(conn: sqlite3.Connection, stats: UpdateStats) -> None:
    to_primary, _ = search.post_maps(conn)
    aliased = {r[0] for r in conn.execute("SELECT alias_id FROM entity_alias")} \
        if search._has_table(conn, "entity_alias") else set()
    live = {r[0] for r in conn.execute("SELECT id FROM media WHERE status NOT IN ('EXCLUDED', 'UNSUPPORTED')")}
    by_entity: dict[int, set[int]] = {}
    for eid, m in conn.execute("SELECT entity_id, media_id FROM media_entity"):
        if eid not in aliased and m in live:
            by_entity.setdefault(eid, set()).add(to_primary.get(m, m))
    names = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT id, name, type FROM entity")}
    pages = {p["entity_id"]: p for p in _pages(conn, "kind='entity'")}
    for eid, items in by_entity.items():
        if len(items) < ENTITY_MIN_ITEMS or eid not in names:
            continue
        page = pages.pop(eid, None)
        if page is None:
            pid = conn.execute(
                "INSERT INTO wiki_page(slug, kind, title, entity_id, named) VALUES (?, 'entity', ?, ?, 1)",
                (unique_slug(conn, f"entity/{slugify(names[eid][0])}"), names[eid][0], eid)).lastrowid
            stats.created += 1
        else:
            pid = page["id"]
        _set_members(conn, pid, items, "entity")
        stats.entities += 1
    for page in pages.values():   # entities merged away or below the threshold
        if not page["built_at"]:
            delete_page(conn, page["id"])
            stats.deleted += 1


def _refresh_dates(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id FROM wiki_page WHERE kind != 'overview'").fetchall()
    all_ids = {r[0] for r in conn.execute("SELECT DISTINCT media_id FROM wiki_member")}
    texts = digest.gather(conn, sorted(all_ids))
    for (pid,) in rows:
        items = [texts[m] for m in _members(conn, pid) if m in texts]
        d0, d1 = date_range(items)
        conn.execute("UPDATE wiki_page SET date_start=?, date_end=? WHERE id=? "
                     "AND (date_start IS NOT ? OR date_end IS NOT ?)", (d0, d1, pid, d0, d1))


# ── Attaching new items (no model) ─────────────────────────────────────────

def attach_new(conn: sqlite3.Connection, cfg, stats: UpdateStats) -> None:
    """Items with vectors that no topic page holds join the nearest topic when they are as close to it as its
    own typical members (the rule cluster.py uses), and the nearest of its events when also dated within its
    range ± 14 days. Needs only the local docs index, not the embedding server."""
    import numpy as np

    from re2 import cluster

    topics = _pages(conn, "kind='topic'")
    if not topics:
        return
    try:
        ids, vecs = cluster.item_vectors(conn, cfg)
    except (OSError, ValueError) as exc:
        log.info("no item vectors, nothing attached: %s", exc)
        return
    if len(ids) == 0:
        return
    row_of = {int(m): i for i, m in enumerate(ids.tolist())}
    held = {r[0] for r in conn.execute(
        "SELECT wm.media_id FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id WHERE p.kind='topic'")}
    new = [m for m in row_of if m not in held]
    if not new:
        return

    def centroid(members: list[int]):
        rows = [row_of[m] for m in members if m in row_of]
        if not rows:
            return None, None
        c = vecs[rows].mean(axis=0)
        c /= max(float(np.linalg.norm(c)), 1e-12)
        sims = vecs[rows] @ c
        return c, float(np.percentile(sims, 10)) if len(rows) >= 5 else float(sims.min())

    t_cent, t_floor, t_pages = [], [], []
    for p in topics:
        c, floor = centroid(list(_members(conn, p["id"])))
        if c is not None:
            t_cent.append(c)
            t_floor.append(floor)
            t_pages.append(p)
    if not t_pages:
        return
    t_cent = np.stack(t_cent)
    new_vecs = vecs[[row_of[m] for m in new]]
    sims = t_cent @ new_vecs.T
    best = sims.argmax(axis=0)
    texts = digest.gather(conn, new)
    theme_of = {p["slug"]: p["id"] for p in _pages(conn, "kind='theme'")}
    events_by_topic: dict[str, list] = {}
    with conn:
        for j, m in enumerate(new):
            k = int(best[j])
            if sims[k, j] < t_floor[k]:
                continue
            topic = t_pages[k]
            conn.execute("INSERT OR IGNORE INTO wiki_member(page_id, media_id, origin) VALUES (?, ?, 'attached')",
                         (topic["id"], m))
            if topic["parent_slug"] in theme_of:
                conn.execute("INSERT OR IGNORE INTO wiki_member(page_id, media_id, origin) VALUES (?, ?, 'attached')",
                             (theme_of[topic["parent_slug"]], m))
            stats.attached += 1
            if topic["slug"] not in events_by_topic:
                events_by_topic[topic["slug"]] = [(e, *centroid(list(_members(conn, e["id"]))))
                                                  for e in _pages(conn, "kind='event' AND parent_slug=?", (topic["slug"],))]
            it = texts.get(m)
            fits = []
            for ev, c, floor in events_by_topic[topic["slug"]]:
                if c is None or float(c @ vecs[row_of[m]]) < floor or not _within(it, ev):
                    continue
                fits.append((float(c @ vecs[row_of[m]]), ev["id"]))
            if fits:
                conn.execute("INSERT OR IGNORE INTO wiki_member(page_id, media_id, origin) VALUES (?, ?, 'attached')",
                             (max(fits)[1], m))
                stats.attached_to_events += 1


def _within(it: digest.ItemText | None, ev: dict) -> bool:
    from datetime import date, timedelta

    def day(s: str | None, end: bool = False) -> date | None:
        if not s:
            return None
        try:
            if len(s) == 7:
                y, mo = map(int, s.split("-"))
                return date(y, mo, 28 if end else 1)
            return date.fromisoformat(s[:10])
        except ValueError:
            return None

    d = day(it.date) if it and it.date_source != "mtime" else None
    lo, hi = day(ev["date_start"]), day(ev["date_end"], end=True)
    if d is None or lo is None or hi is None:
        return True
    return lo - timedelta(days=ATTACH_DAYS) <= d <= hi + timedelta(days=ATTACH_DAYS)


# ── Staleness ───────────────────────────────────────────────────────────────

def current_hashes(conn: sqlite3.Connection, ids) -> dict[int, str]:
    texts = digest.gather(conn, sorted(set(ids)))
    return {m: digest.record_hash(digest.record(it)) for m, it in texts.items()}


def changes(conn: sqlite3.Connection, page: dict, hashes: dict[int, str]) -> int:
    """Members the page's writer never read, read in a different form, or read and since removed."""
    mem = _members(conn, page["id"])
    changed = sum(1 for m, info in mem.items() if info["read_hash"] is None or info["read_hash"] != hashes.get(m))
    still = sum(1 for info in mem.values() if info["read_hash"] is not None)
    return changed + max(0, (page["members_read"] or 0) - still)


def mark_stale(conn: sqlite3.Connection, stats: UpdateStats | None = None) -> UpdateStats:
    stats = stats or UpdateStats()
    pages = _pages(conn)
    built = [p for p in pages if p["built_at"]]
    ids = {r[0] for r in conn.execute(
        "SELECT wm.media_id FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id WHERE p.built_at IS NOT NULL")}
    hashes = current_hashes(conn, ids)
    by_slug = {p["slug"]: p for p in pages}
    with conn:
        for p in pages:
            stale = not p["built_at"]
            if not stale:
                n = changes(conn, p, hashes)
                total = p["members_total"] or 0
                stale = total > 0 and n >= max(1, math.ceil(STALE_FRACTION * total))
                # A child rewritten since, or a child not written yet that this page could now read.
                stale = stale or any(c["built_at"] and c["built_at"] > p["built_at"]
                                     for c in pages if c["parent_slug"] == p["slug"] and c["kind"] != "entity")
            if stale and not p["stale"]:
                conn.execute("UPDATE wiki_page SET stale=1 WHERE id=?", (p["id"],))
                p["stale"] = 1
    stats.by_kind = {}
    for p in pages:
        k = stats.by_kind.setdefault(p["kind"], {"pages": 0, "stale": 0, "written": 0})
        k["pages"] += 1
        k["stale"] += p["stale"]
        k["written"] += bool(p["built_at"])
    stats.stale = sum(p["stale"] for p in pages)
    del by_slug
    return stats


def update(conn: sqlite3.Connection, cfg) -> UpdateStats:
    stats = sync(conn)
    attach_new(conn, cfg, stats)
    mark_stale(conn, stats)
    return stats


# ── Events ──────────────────────────────────────────────────────────────────

def split_lines(texts: list[digest.ItemText]) -> list[str]:
    out = []
    for n, it in enumerate(texts, 1):
        weak = "~" if it.date_source == "mtime" else ""
        ents = ", ".join(it.entities[:5]) or "-"
        out.append(f"{n} | {weak}{it.date or '?'} | {digest._clip(digest.title_of(it), 140)} | {ents} | "
                   f"hint: {digest._clip(digest.event_hint(it), 80)}")
    return out


def validate_split(reply: dict, n: int) -> tuple[list[dict], list[int]]:
    """Events with 1-based member numbers: unknown numbers dropped, a number used twice kept only the first
    time, events left with fewer than 2 members dissolved, anything unaccounted for goes to not_events."""
    seen: set[int] = set()
    events = []
    for ev in reply.get("events") or []:
        if not isinstance(ev, dict):
            continue
        mem = []
        for x in ev.get("members") or []:
            if isinstance(x, int) and 1 <= x <= n and x not in seen:
                seen.add(x)
                mem.append(x)
        if len(mem) >= 2 and str(ev.get("title") or "").strip():
            events.append({"title": " ".join(str(ev["title"]).split()), "summary": " ".join(str(ev.get("summary") or "").split()),
                           "members": mem})
        else:
            seen -= set(mem)
    return events, [x for x in range(1, n + 1) if x not in seen]


def split_topic(client: httpx.Client, cfg, texts: list[digest.ItemText]) -> dict:
    """Ask the model to group one topic's items (sorted by date) into events, 400 lines a call at most."""
    texts = sorted(texts, key=lambda it: (it.date or "9999", it.media_id))
    lines = split_lines(texts)
    per_line = max(len(x) for x in lines) + 1 if lines else 1
    max_lines = max(20, min(SPLIT_LINES, writer.input_chars(cfg, 6000) // per_line))
    result = {"topic_title": "", "events": [], "not_events": []}
    for start in range(0, len(texts), max_lines):
        chunk, chunk_lines = texts[start:start + max_lines], lines[start:start + max_lines]
        numbered = [line.split(" | ", 1)[1] for line in chunk_lines]
        body_lines = "\n".join(f"{i} | {rest}" for i, rest in enumerate(numbered, 1))
        max_tokens = min(6000, 700 + 10 * len(chunk))
        reply = writer.complete_json(client, cfg, writer.payload(
            cfg, SPLIT_SYSTEM, SPLIT_PROMPT.format(n=len(chunk), lines=body_lines), max_tokens, schema=SPLIT_SCHEMA))
        events, rest = validate_split(reply, len(chunk))
        result["topic_title"] = result["topic_title"] or " ".join(str(reply.get("topic_title") or "").split())
        result["events"] += [{**e, "members": [chunk[x - 1].media_id for x in e["members"]]} for e in events]
        result["not_events"] += [chunk[x - 1].media_id for x in rest]
    return result


def apply_split(conn: sqlite3.Connection, cfg, topic: dict, result: dict) -> dict:
    """Create or update event pages under the topic; name the topic if the model has not named it yet."""
    counts = {"events": 0, "reused": 0}
    with conn:
        title = result.get("topic_title")
        if title and not topic["named"] and not topic["built_at"]:
            rename(conn, topic["id"], title)
        topic = _pages(conn, "id=?", (topic["id"],))[0]
        existing = [(e, set(_members(conn, e["id"]))) for e in _pages(conn, "kind='event'")]
        texts = digest.gather(conn, [m for e in result["events"] for m in e["members"]])
        for ev in result["events"]:
            members = set(ev["members"])
            best = max(((_jaccard(members, em), e) for e, em in existing), default=(0, None), key=lambda x: x[0])
            d0, d1 = date_range([texts[m] for m in members if m in texts])
            if best[1] is not None and best[0] >= JACCARD_KEEP:
                pid = best[1]["id"]
                conn.execute("UPDATE wiki_page SET parent_slug=?, stale=1 WHERE id=?", (topic["slug"], pid))
                counts["reused"] += 1
            else:
                slug = unique_slug(conn, f"event/{(d0 or 'undated')[:7]}-{slugify(ev['title'])}")
                pid = conn.execute(
                    """INSERT INTO wiki_page(slug, kind, title, summary, parent_slug, named, date_start, date_end)
                       VALUES (?, 'event', ?, ?, ?, 1, ?, ?)""",
                    (slug, ev["title"], ev["summary"], topic["slug"], d0, d1)).lastrowid
                counts["events"] += 1
            for m in members:
                conn.execute("INSERT OR IGNORE INTO wiki_member(page_id, media_id, origin) VALUES (?, ?, 'split')", (pid, m))
        conn.execute("UPDATE wiki_page SET events_split_at=datetime('now') WHERE id=?", (topic["id"],))
    return counts


def rename(conn: sqlite3.Connection, page_id: int, title: str) -> str:
    """Title a page the model has named, with a slug to match; children follow. Only before it is first written,
    so nothing written links to the old slug."""
    page = _pages(conn, "id=?", (page_id,))[0]
    title = " ".join(title.split())[:160]
    slug = unique_slug(conn, f"{page['kind']}/{slugify(title)}", page_id) if page["kind"] in ("topic", "theme") \
        else page["slug"]
    conn.execute("UPDATE wiki_page SET title=?, slug=?, named=1 WHERE id=?", (title, slug, page_id))
    if slug != page["slug"]:
        conn.execute("UPDATE wiki_page SET parent_slug=? WHERE parent_slug=?", (slug, page["slug"]))
    return slug


def topics_to_split(conn: sqlite3.Connection) -> list[tuple[dict, list[int]]]:
    """(topic, items to group): never split topics with all their members, split ones with ≥ WAIT_SPLIT new
    items no event holds."""
    out = []
    for t in _pages(conn, "kind='topic'"):
        mem = _members(conn, t["id"])
        if not t["events_split_at"]:
            out.append((t, sorted(mem)))
            continue
        in_events = {r[0] for r in conn.execute(
            "SELECT wm.media_id FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id "
            "WHERE p.kind='event' AND p.parent_slug=?", (t["slug"],))}
        waiting = [m for m, info in mem.items() if m not in in_events and info["added_at"] > t["events_split_at"]]
        if len(waiting) >= WAIT_SPLIT:
            out.append((t, sorted(waiting)))
    return out


# ── Writing pages ───────────────────────────────────────────────────────────

_CITE_GROUP = re.compile(r"\[m:\s*(\d+(?:\s*[,;]\s*(?:m:)?\s*\d+)+)\s*\]")
_CITE = re.compile(r"\[m:(\d+)\]")
_PAGE = re.compile(r"\[p:([a-z0-9][a-z0-9/_.-]*)\]")
_WIKILINK = re.compile(r"\[\[([^\]|\n]+)(?:\|([^\]\n]*))?\]\]")


def clean_body(text: str, allowed_ids: set[int], slugs: dict[str, str]) -> tuple[str, dict[int, int], set[str]]:
    """Normalise the model's markers, drop citations of ids it was not given and links to unknown slugs
    ([[slug|text]] keeps its text). Returns (body, citation counts, linked slugs)."""
    text = _CITE_GROUP.sub(lambda m: "".join(f"[m:{x}]" for x in re.findall(r"\d+", m.group(1))), text)

    def wikilink(m: re.Match) -> str:
        target = m.group(1).strip()
        if target.startswith("p:"):
            target = target[2:]
        return f"[p:{target}]" if target in slugs else (m.group(2) or slugs.get(target) or target)

    text = _WIKILINK.sub(wikilink, text)
    cites: dict[int, int] = {}
    links: set[str] = set()

    def cite(m: re.Match) -> str:
        media_id = int(m.group(1))
        if media_id not in allowed_ids:
            return ""
        cites[media_id] = cites.get(media_id, 0) + 1
        return m.group(0)

    def page(m: re.Match) -> str:
        if m.group(1) not in slugs:
            return ""
        links.add(m.group(1))
        return m.group(0)

    text = _PAGE.sub(page, _CITE.sub(cite, text))
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), cites, links


def strip_markers(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _PAGE.sub("", _CITE.sub("", text))).replace(" .", ".").strip()


def extract_summary(body: str, max_chars: int = 300) -> str:
    """The lead: the first paragraph that is not a heading, list or quote, without markers."""
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block or block.startswith(("#", "-", "*", ">", "|")) or re.match(r"^\d+[.)]\s", block):
            continue
        text = strip_markers(" ".join(block.split()))
        text = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", text)
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        return cut[:end + 1] if end > max_chars // 3 else cut.rsplit(" ", 1)[0] + "…"
    return ""


def split_title(text: str) -> tuple[str | None, str]:
    """A leading '# Title' line, and the rest."""
    m = re.match(r"^\s*#\s+(.+?)\s*\n", text + "\n")
    if not m:
        return None, text
    return m.group(1).strip().strip("*"), text[m.end():].lstrip("\n")


@dataclass
class PagePlan:
    page: dict
    total: int
    records: dict[int, str]                   # members read directly: id -> record
    inherited: dict[int, str]                 # members read through child pages: id -> the child's read hash
    children: list[dict]                      # child pages shown to the writer
    batches: list[list[int]] = field(default_factory=list)
    extra: str = ""                           # the overview's counts
    waiting_for: int = 0                      # child pages not written yet

    def dates(self) -> str:
        p = self.page
        if not p["date_start"]:
            return ""
        return f" ({p['date_start']} to {p['date_end']})" if p["date_end"] != p["date_start"] else f" ({p['date_start']})"


def _children(conn: sqlite3.Connection, page: dict) -> list[dict]:
    kind = {"topic": "event", "theme": "topic", "overview": "theme"}.get(page["kind"])
    if not kind:
        return []
    return _pages(conn, "kind=? AND parent_slug=?", (kind, page["slug"]))


def plan_page(conn: sqlite3.Connection, cfg, page: dict, texts: dict[int, digest.ItemText]) -> PagePlan:
    mem = _members(conn, page["id"])
    children = _children(conn, page)
    inherited: dict[int, str] = {}
    written = [c for c in children if c["built_at"]]
    for c in written:
        for m, info in _members(conn, c["id"]).items():
            if m in mem and info["read_hash"]:
                inherited.setdefault(m, info["read_hash"])
    if page["kind"] == "entity":
        # Entities read every item themselves; event pages that share items are context and link targets.
        inherited = {}
        children = [dict(r) for r in conn.execute(
            """SELECT p.*, COUNT(*) AS shared FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id
               WHERE p.kind='event' AND p.built_at IS NOT NULL AND wm.media_id IN (SELECT media_id FROM wiki_member WHERE page_id=?)
               GROUP BY p.id ORDER BY shared DESC LIMIT 30""", (page["id"],))]
        written = children
    records = {m: digest.record(texts[m]) for m in sorted(mem, key=lambda m: ((texts[m].date or "9999") if m in texts else "9999", m))
               if m not in inherited and m in texts}
    total = len(mem)
    extra = ""
    if page["kind"] == "overview":
        total = conn.execute("SELECT COUNT(DISTINCT wm.media_id) FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id "
                             "WHERE p.kind='theme'").fetchone()[0]
        extra = _overview_counts(conn)
    plan = PagePlan(page, total, records, inherited, written, waiting_for=len(children) - len(written))
    ids = list(records)
    material_budget = writer.input_chars(cfg, WRITE_MAX_TOKENS) - _children_chars(plan, cfg)
    direct = sum(len(r) + 2 for r in records.values())
    if len(ids) > DIRECT_MAX_ITEMS or direct > material_budget:
        budget = writer.input_chars(cfg, MAP_MAX_TOKENS)
        batch: list[int] = []
        size = 0
        for m in ids:
            n = len(records[m]) + 2
            if batch and (len(batch) >= MAP_BATCH_ITEMS or size + n > budget):
                plan.batches.append(batch)
                batch, size = [], 0
            batch.append(m)
            size += n
        if batch:
            plan.batches.append(batch)
    return plan


def _children_chars(plan: PagePlan, cfg) -> int:
    return min(sum(len(c["body_md"]) + 200 for c in plan.children), int(writer.input_chars(cfg, WRITE_MAX_TOKENS) * 0.7))


def _overview_counts(conn: sqlite3.Connection) -> str:
    years = conn.execute(
        f"""SELECT substr(d, 1, 4), COUNT(*) FROM (SELECT {search.item_date_sql('m')} AS d FROM media m
            WHERE m.status NOT IN ('EXCLUDED', 'UNSUPPORTED') AND m.kind IN ('image', 'video', 'pdf'))
            WHERE d IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall()
    kinds = conn.execute("SELECT kind, COUNT(*) FROM media WHERE status NOT IN ('EXCLUDED', 'UNSUPPORTED') "
                         "AND kind IN ('image', 'video', 'pdf') GROUP BY 1").fetchall()
    return ("Items by kind: " + ", ".join(f"{k} {n}" for k, n in kinds) + "\nItems by year (best-guess dates): "
            + ", ".join(f"{y} {n}" for y, n in years))


def map_prompt(plan: PagePlan, i: int) -> str:
    batch = plan.batches[i]
    first = sum(len(b) for b in plan.batches[:i]) + 1
    return MAP_PROMPT.format(kind=plan.page["kind"], title=plan.page["title"], dates=plan.dates(), first=first,
                             last=first + len(batch) - 1, total=len(plan.records), i=i + 1, batches=len(plan.batches),
                             records="\n\n".join(plan.records[m] for m in batch))


def note_hash(cfg, plan: PagePlan, i: int) -> str:
    return _hash(NOTES_VERSION, cfg.wiki_model, plan.page["id"], map_prompt(plan, i))


def _clip_body(body: str, n: int) -> str:
    if len(body) <= n:
        return body
    cut = body[:n]
    return cut[:cut.rfind("\n\n")] if cut.rfind("\n\n") > n // 2 else cut.rsplit(" ", 1)[0] + " …"


def write_prompt(cfg, plan: PagePlan, material: str) -> tuple[str, set[str]]:
    p = plan.page
    slugs: set[str] = set()
    children = ""
    if plan.children:
        per = max(600, _children_chars(plan, cfg) // max(1, len(plan.children)))
        label = {"topic": "Event pages in this topic", "theme": "Topic pages in this theme",
                 "overview": "Theme pages", "entity": "Event pages with items about it"}[p["kind"]]
        blocks = []
        for c in plan.children:
            slugs.add(c["slug"])
            d = f" · {c['date_start'] or ''}–{c['date_end'] or ''}" if c["date_start"] else ""
            body = c["summary"] if p["kind"] in ("entity", "overview") else _clip_body(c["body_md"], per)
            blocks.append(f"### [p:{c['slug']}] {c['title']}{d}\n{body}")
        children = f"\n{label} (link them as [p:slug]):\n\n" + "\n\n".join(blocks) + "\n"
    naming = p["kind"] in ("topic", "theme") and not p["named"]
    title_line = "Choose its title." if naming else f"Title: {p['title']}"
    covered = len(plan.inherited) + len(plan.records)
    coverage = f"; the pages and notes below cover {covered} of them" if p["kind"] != "event" else ""
    start = ("Start with one line '# <a specific, factual title of at most 8 words>', then the lead paragraph."
             if naming else "Start with the lead paragraph; do not repeat the title as a heading.")
    return WRITE_PROMPT.format(kind=p["kind"], title_line=title_line, dates=plan.dates(), total=plan.total,
                               coverage=coverage, children=children, material=material, guide=KIND_GUIDE[p["kind"]],
                               length=LENGTHS[p["kind"]], start=start), slugs


@dataclass
class BuildStats:
    split: int = 0
    events_created: int = 0
    map_calls: int = 0
    pages: int = 0
    failed: int = 0
    skipped_waiting: int = 0
    interrupted: bool = False
    started: float = 0.0
    written: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"topics_split={self.split} events_created={self.events_created} map_calls={self.map_calls} "
                f"pages_written={self.pages} failed={self.failed} waiting_for_children={self.skipped_waiting} "
                f"elapsed={time.monotonic() - self.started:.0f}s" + (" INTERRUPTED" if self.interrupted else ""))


def reduce_notes(client, cfg, plan: PagePlan, notes: list[str], budget: int) -> tuple[str, int]:
    rounds = 0
    group_budget = writer.input_chars(cfg, REDUCE_MAX_TOKENS)
    while sum(len(n) + 2 for n in notes) > budget and len(notes) > 1 and rounds < 3:
        rounds += 1
        groups: list[list[str]] = [[]]
        size = 0
        for n in notes:
            if groups[-1] and size + len(n) > group_budget:
                groups.append([])
                size = 0
            groups[-1].append(n)
            size += len(n) + 2
        notes = [writer.complete(client, cfg, writer.payload(
            cfg, NOTES_SYSTEM, REDUCE_PROMPT.format(kind=plan.page["kind"], title=plan.page["title"], dates=plan.dates(),
                                                   notes="\n\n".join(g)), REDUCE_MAX_TOKENS)) for g in groups]
    merged = "\n\n".join(notes)
    return (merged if len(merged) <= budget else _clip_body(merged, budget)), rounds


def write_page(client, cfg, conn_notes: dict[str, str], plan: PagePlan) -> dict:
    """The reduce and write calls for one page (its map notes are already made). Runs on a worker thread."""
    t0 = time.monotonic()
    budget = writer.input_chars(cfg, WRITE_MAX_TOKENS) - _children_chars(plan, cfg) - 1500
    rounds = 0
    if plan.batches:
        notes = [conn_notes[note_hash(cfg, plan, i)] for i in range(len(plan.batches))]
        merged, rounds = reduce_notes(client, cfg, plan, notes, budget)
        material = f"\nNotes from all {len(plan.records)} items read for this page:\n\n{merged}\n"
    elif plan.records:
        material = f"\nThe items:\n\n" + "\n\n".join(plan.records.values()) + "\n"
    else:
        material = ""
    if plan.extra:
        material += f"\n{plan.extra}\n"
    prompt, slugs = write_prompt(cfg, plan, material)
    text = writer.complete(client, cfg, writer.payload(cfg, SYSTEM_PROMPT, prompt, WRITE_MAX_TOKENS))
    return {"text": text, "prompt": prompt, "slugs": slugs, "rounds": rounds, "seconds": time.monotonic() - t0}


def save_page(conn: sqlite3.Connection, cfg, plan: PagePlan, out: dict, hashes: dict[int, str]) -> dict:
    p = plan.page
    title, text = split_title(out["text"])
    slug_titles = {c["slug"]: c["title"] for c in plan.children if c["slug"] in out["slugs"]}
    allowed = {int(x) for x in _CITE.findall(out["prompt"])} | set(plan.records)
    if p["kind"] != "overview":
        allowed &= set(_members(conn, p["id"])) | set(plan.records)
    body, cites, links = clean_body(text, allowed, slug_titles)
    if not body:
        raise writer.WriterError("the model wrote an empty page")
    read = {m: hashes.get(m) or digest.record_hash(r) for m, r in plan.records.items()}
    read.update({m: h for m, h in plan.inherited.items() if m not in read})
    member_hash = _hash(*sorted(f"{m}:{h}" for m, h in read.items()), *sorted(f"{c['slug']}:{c['built_at']}" for c in plan.children))
    build = {"direct": not plan.batches and bool(plan.records), "batches": len(plan.batches), "reduce_rounds": out["rounds"],
             "items_read_directly": len(plan.records), "items_read_via_pages": len(plan.inherited),
             "child_pages_read": len(plan.children), "seconds": round(out["seconds"], 1)}
    members_read = len(read) if p["kind"] != "overview" else \
        len({m for c in plan.children for m in _members(conn, c["id"]) if _members(conn, c["id"])[m]["read_hash"]})
    with conn:
        if title and p["kind"] in ("topic", "theme") and not p["named"]:
            rename(conn, p["id"], title)
        conn.execute(
            """UPDATE wiki_page SET body_md=?, summary=?, stale=0, model=?, members_read=?, members_total=?, members_cited=?,
               member_hash=?, build_json=?, built_at=strftime('%Y-%m-%d %H:%M:%f', 'now'), updated_at=datetime('now'), named=1 WHERE id=?""",
            (body, extract_summary(body), cfg.wiki_model, members_read, plan.total, len(cites), member_hash,
             json.dumps(build), p["id"]))
        conn.execute("UPDATE wiki_member SET read_hash=NULL WHERE page_id=?", (p["id"],))
        conn.executemany("UPDATE wiki_member SET read_hash=? WHERE page_id=? AND media_id=?",
                         [(h, p["id"], m) for m, h in read.items()])
        conn.execute("DELETE FROM wiki_citation WHERE page_id=?", (p["id"],))
        conn.executemany("INSERT INTO wiki_citation(page_id, media_id, n) VALUES (?, ?, ?)",
                         [(p["id"], m, n) for m, n in cites.items()])
        conn.execute("DELETE FROM wiki_link WHERE from_page=?", (p["id"],))
        for slug in links:
            conn.execute("INSERT OR IGNORE INTO wiki_link(from_page, to_page) SELECT ?, id FROM wiki_page WHERE slug=?",
                         (p["id"], slug))
        used = {note_hash(cfg, plan, i) for i in range(len(plan.batches))}
        for (h,) in conn.execute("SELECT hash FROM wiki_note WHERE page_id=?", (p["id"],)).fetchall():
            if h not in used:
                conn.execute("DELETE FROM wiki_note WHERE hash=?", (h,))
        db.clear_failure(conn, STAGE, None, p["id"])
    return build


def build(conn: sqlite3.Connection, cfg, kinds: list[str] | None = None, limit: int | None = None,
          topics: list[int] | None = None, client_factory=httpx.Client) -> BuildStats:
    """Update, group topics into events, then write stale pages kind by kind: events, topics, themes, entities,
    overview. --limit N works on N topics end to end (their split, events and page) and N pages of other kinds."""
    stats = BuildStats(started=time.monotonic())
    kinds = [k for k in KINDS if k in (kinds or KINDS)]
    update(conn, cfg)
    clients = writer.Clients(client_factory)
    skip = db.pending_failures(conn, STAGE, "doc_id")   # page ids ride in failure.doc_id
    try:
        in_play = _topics_in_play(conn, topics, limit)
        in_play_slugs = {t["slug"] for t in in_play}
        if "event" in kinds:
            if not _run_splits(conn, cfg, clients, [(t, ids) for t, ids in topics_to_split(conn)
                                                     if t["id"] in {x["id"] for x in in_play}], stats):
                return stats
            in_play = [_pages(conn, "id=?", (t["id"],))[0] for t in in_play]   # splitting renames topics
            in_play_slugs = {t["slug"] for t in in_play}
        for kind in kinds:
            mark_stale(conn)
            pages = _pages(conn, "kind=? AND stale=1", (kind,))
            pages = [p for p in pages if p["id"] not in skip]
            if kind == "event" and (topics or limit):
                pages = [p for p in pages if p["parent_slug"] in in_play_slugs]
            elif kind == "topic" and (topics or limit):
                pages = [p for p in pages if p["slug"] in in_play_slugs]
            elif topics:
                continue
            if kind == "topic":
                # A topic waits until it has been grouped into events.
                pages = [p for p in pages if p["events_split_at"]]
            if kind not in ("event", "topic") and limit is not None:
                pages = pages[:limit]
            if not pages:
                continue
            if not _write_kind(conn, cfg, clients, kind, pages, stats):
                return stats
    finally:
        clients.close()
    return stats


def _topics_in_play(conn: sqlite3.Connection, topics: list[int] | None, limit: int | None) -> list[dict]:
    pages = _pages(conn, "kind='topic'")
    if topics:
        return [p for p in pages if p["cluster_id"] in set(topics) or p["id"] in set(topics)]
    stale_parents = {r[0] for r in conn.execute("SELECT parent_slug FROM wiki_page WHERE kind='event' AND stale=1")}
    due = [p for p in pages if not p["events_split_at"] or p["stale"] or p["slug"] in stale_parents]
    if limit is None:
        return pages
    sizes = dict(conn.execute("SELECT page_id, COUNT(*) FROM wiki_member GROUP BY page_id").fetchall())
    due.sort(key=lambda p: -sizes.get(p["id"], 0))
    return due[:limit]


def _run_splits(conn, cfg, clients, jobs, stats: BuildStats) -> bool:
    if not jobs:
        return True
    log.info("grouping %d topics into events with %s", len(jobs), cfg.wiki_model)

    def work(job):
        topic, ids = job
        texts = digest.gather(db.connect(cfg.db_path) if cfg.db_path.exists() else conn, ids)
        return split_topic(clients.get(), cfg, list(texts.values()))

    def on_result(job, result):
        counts = apply_split(conn, cfg, job[0], result)
        stats.split += 1
        stats.events_created += counts["events"]
        log.info("topic %s: %d events (%d reused), %d items in no event", job[0]["slug"], counts["events"],
                 counts["reused"], len(result["not_events"]))

    def on_error(job, exc):
        stats.failed += 1
        with conn:
            db.upsert_failure(conn, STAGE + "_split", None, job[0]["id"], f"{type(exc).__name__}: {exc}"[:2000])
        log.warning("splitting topic %s failed: %s", job[0]["slug"], exc)

    # Gathering reads the database on worker threads, so it happens here instead.
    prepared = [(t, ids) for t, ids in jobs]
    texts_by_job = {t["id"]: list(digest.gather(conn, ids).values()) for t, ids in prepared}
    conn.commit()
    finished = runner.run_jobs(prepared, lambda job: split_topic(clients.get(), cfg, texts_by_job[job[0]["id"]]),
                               on_result, on_error, cfg.wiki_concurrency, total=len(prepared), desc="wiki events")
    del work
    stats.interrupted = not finished
    return finished


def _write_kind(conn, cfg, clients, kind: str, pages: list[dict], stats: BuildStats) -> bool:
    ids = {m for p in pages for m in _members(conn, p["id"])}
    texts = digest.gather(conn, sorted(ids))
    hashes = {m: digest.record_hash(digest.record(it)) for m, it in texts.items()}
    plans = []
    for p in pages:
        plan = plan_page(conn, cfg, p, texts)
        if plan.waiting_for and kind in ("topic", "theme", "overview"):
            stats.skipped_waiting += 1
            log.info("%s waits for %d child pages to be written", p["slug"], plan.waiting_for)
            continue
        plans.append(plan)
    conn.commit()
    notes = {r[0]: r[1] for r in conn.execute("SELECT hash, notes FROM wiki_note")}
    map_jobs = [(plan, i) for plan in plans for i in range(len(plan.batches)) if note_hash(cfg, plan, i) not in notes]
    log.info("%s: %d pages to write, %d batches of items to read first", kind, len(plans), len(map_jobs))

    def map_work(job):
        plan, i = job
        return writer.complete(clients.get(), cfg, writer.payload(cfg, NOTES_SYSTEM, map_prompt(plan, i), MAP_MAX_TOKENS))

    def map_result(job, text):
        plan, i = job
        valid = set(plan.batches[i])
        text = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "",
                         _CITE_GROUP.sub(lambda m: "".join(f"[m:{x}]" for x in re.findall(r"\d+", m.group(1))), text))
        h = note_hash(cfg, plan, i)
        notes[h] = text
        with conn:
            conn.execute("INSERT OR REPLACE INTO wiki_note(hash, page_id, notes, model) VALUES (?, ?, ?, ?)",
                         (h, plan.page["id"], text, cfg.wiki_model))
        stats.map_calls += 1

    failed_pages: set[int] = set()

    def map_error(job, exc):
        plan, i = job
        failed_pages.add(plan.page["id"])
        log.warning("reading batch %d of %s failed: %s", i + 1, plan.page["slug"], exc)

    if map_jobs and not runner.run_jobs(map_jobs, map_work, map_result, map_error, cfg.wiki_concurrency,
                                        total=len(map_jobs), desc=f"wiki {kind} notes"):
        stats.interrupted = True
        return False
    for pid in failed_pages:
        stats.failed += 1
        with conn:
            db.upsert_failure(conn, STAGE, None, pid, "a batch of items could not be read")
    plans = [p for p in plans if p.page["id"] not in failed_pages]

    def on_result(plan, out):
        build_info = save_page(conn, cfg, plan, out, hashes)
        stats.pages += 1
        stats.written.append(plan.page["slug"])
        log.info("wrote %s: %s", plan.page["slug"], build_info)

    def on_error(plan, exc):
        stats.failed += 1
        with conn:
            db.upsert_failure(conn, STAGE, None, plan.page["id"], f"{type(exc).__name__}: {exc}"[:2000])
        log.warning("writing %s failed: %s", plan.page["slug"], exc)

    finished = runner.run_jobs(plans, lambda plan: write_page(clients.get(), cfg, notes, plan), on_result, on_error,
                               cfg.wiki_concurrency, total=len(plans), desc=f"wiki {kind} pages")
    stats.interrupted = not finished
    return finished


# ── Reading pages (API) ─────────────────────────────────────────────────────

def render(conn: sqlite3.Connection, page: dict) -> dict:
    """HTML for a page body: markdown-it with raw HTML escaped; [m:ID] become numbered citation chips and
    [p:slug] links to pages. Markers are swapped for placeholder words first so Markdown leaves them alone."""
    from markdown_it import MarkdownIt
    from html import escape

    body = page["body_md"] or ""
    order: list[int] = []
    for m in _CITE.findall(body):
        if int(m) not in order:
            order.append(int(m))
    cards = search.cards(conn, order)
    slugs = set(_PAGE.findall(body))
    titles = dict(conn.execute(f"SELECT slug, title FROM wiki_page WHERE slug IN ({','.join('?' * len(slugs))})",
                               tuple(slugs)).fetchall()) if slugs else {}
    text = _CITE.sub(lambda m: f"WIKICITE{m.group(1)}Z", body)
    text = _PAGE.sub(lambda m: f"WIKIPAGE{order_slug(m.group(1))}Z", text)
    html = MarkdownIt("commonmark", {"html": False}).enable("table").render(text)

    def cite(m: re.Match) -> str:
        media_id = int(m.group(1))
        card = cards.get(media_id)
        if card is None:
            return ""
        n = order.index(media_id) + 1
        tip = f"{card['title']} — {card['date'] or 'undated'}"
        return (f'<a class="cite" data-media="{media_id}" href="#/wiki/{escape(page["slug"])}?open={media_id}" '
                f'title="{escape(tip)}">{n}</a>')

    def link(m: re.Match) -> str:
        slug = unorder_slug(m.group(1))
        if slug not in titles:
            return ""
        return f'<a class="wikilink" data-slug="{escape(slug)}" href="#/wiki/{escape(slug)}">{escape(titles[slug])}</a>'

    html = re.sub(r"WIKICITE(\d+)Z", cite, html)
    html = re.sub(r"WIKIPAGE([0-9a-f]+)Z", link, html)
    return {"html": html, "cited": [cards[m] for m in order if m in cards]}


def order_slug(slug: str) -> str:
    return slug.encode().hex()


def unorder_slug(hexed: str) -> str:
    return bytes.fromhex(hexed).decode()


# ── Export ──────────────────────────────────────────────────────────────────

def export(conn: sqlite3.Connection, cfg, out: Path) -> int:
    """One Markdown file per written page at <out>/<slug>.md, Obsidian style: front matter, [[slug|Title]] links,
    and citations as numbered footnotes with the item's title, date, source and dashboard link. Written to a
    temporary directory and swapped in, so a reader never sees half an export."""
    out = Path(out)
    host = "127.0.0.1" if cfg.server_host in ("0.0.0.0", "::") else cfg.server_host
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pages = _pages(conn, "body_md != ''")
    titles = {p["slug"]: p["title"] for p in pages}
    for p in pages:
        order: list[int] = []
        for m in _CITE.findall(p["body_md"]):
            if int(m) not in order:
                order.append(int(m))
        cards = search.cards(conn, order)
        body = _CITE.sub(lambda m: f"[^{order.index(int(m.group(1))) + 1}]" if int(m.group(1)) in cards else "", p["body_md"])
        body = _PAGE.sub(lambda m: f"[[{m.group(1)}|{titles[m.group(1)]}]]" if m.group(1) in titles else "", body)
        front = ["---", f"title: {json.dumps(p['title'], ensure_ascii=False)}", f"kind: {p['kind']}",
                 f"date_start: {p['date_start'] or ''}", f"date_end: {p['date_end'] or ''}",
                 f"updated: {p['built_at'] or p['updated_at']}",
                 f"members_read: {p['members_read'] or 0}", f"members_total: {p['members_total'] or 0}"]
        if p["parent_slug"] in titles:
            front.append(f"parent: \"[[{p['parent_slug']}|{titles[p['parent_slug']]}]]\"")
        front.append("---")
        notes = []
        for n, m in enumerate(order, 1):
            c = cards.get(m)
            if c:
                notes.append(f"[^{n}]: {c['title']} — {c['date'] or 'undated'}"
                             + (f" — {c['source_url']}" if c["source_url"] else "")
                             + f" — http://{host}:{cfg.server_port}/#/item/{m}")
        path = tmp / f"{p['slug']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(front) + f"\n\n# {p['title']}\n\n{body}\n" + ("\n" + "\n".join(notes) + "\n" if notes else ""),
                        encoding="utf-8")
    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    return len(pages)
