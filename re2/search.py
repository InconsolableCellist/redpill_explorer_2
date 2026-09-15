"""Hybrid search: scopes, lexical (FTS5), dense, fusion, lenses and correlations."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from re2 import embed

log = logging.getLogger(__name__)

SCOPE_TYPES = ("text", "entity", "tag", "folder", "kind", "status", "date", "page", "media", "cluster", "caption")
VALID_KINDS = {"image", "video", "pdf"}
VALID_STATUSES = {"READY", "THUMBNAIL_ONLY", "SOURCE_OFFLINE"}
MARK_START, MARK_END = "\x02", "\x03"


class ScopeError(ValueError):
    pass


# ── Scopes ─────────────────────────────────────────────────────────────────

def parse_scope(scope: str) -> tuple[str, str]:
    kind, sep, value = scope.partition(":")
    if not sep or kind not in SCOPE_TYPES or not value.strip():
        raise ScopeError(f"bad scope {scope!r}")
    return kind, value.strip()


def _date_bounds(value: str) -> tuple[str | None, str | None]:
    lo, sep, hi = value.partition("..")
    if not sep:
        hi = lo
    pattern = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
    for part in (lo, hi):
        if part and not pattern.match(part):
            raise ScopeError(f"bad date bound {part!r}")
    # "2024-06" as an upper bound means the end of June: compare on the prefix length.
    return lo or None, (hi + "￿") if hi else None


def item_date_sql(alias: str = "m") -> str:
    """Best available date for a media row, as TEXT starting YYYY[-MM[-DD]]."""
    return (f"COALESCE((SELECT mm.item_date FROM media_meta mm WHERE mm.media_id={alias}.id), "
            f"(SELECT substr(MIN(s.date), 1, 10) FROM source s WHERE s.media_id={alias}.id))")


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is not None


def _scope_ids(conn: sqlite3.Connection, kind: str, value: str) -> set[int]:
    q = lambda sql, *args: {r[0] for r in conn.execute(sql, args)}  # noqa: E731
    if kind == "text":
        match = fts_query(value, prefix_last=False)
        if not match:
            return q("SELECT id FROM media")
        return q("SELECT DISTINCT d.media_id FROM doc_fts JOIN doc d ON d.id=doc_fts.rowid WHERE doc_fts MATCH ?",
                 match)
    if kind == "entity":
        return q("SELECT media_id FROM media_entity WHERE entity_id=?", int(value))
    if kind == "tag":
        return q("SELECT media_id FROM media_tag WHERE tag_id=?", int(value))
    if kind == "folder":
        if value in (".", "/"):
            return q("SELECT media_id FROM media_path WHERE rel_path NOT LIKE '%/%'")
        prefix = value.strip("/")
        return q("SELECT media_id FROM media_path WHERE rel_path LIKE ? ESCAPE '\\'",
                 prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%")
    if kind == "kind":
        if value not in VALID_KINDS:
            raise ScopeError(f"bad kind {value!r}")
        return q("SELECT id FROM media WHERE kind=?", value)
    if kind == "status":
        if value not in VALID_STATUSES:
            raise ScopeError(f"bad status {value!r}")
        return q("SELECT id FROM media WHERE status=?", value)
    if kind == "date":
        lo, hi = _date_bounds(value)
        date = item_date_sql("m") if _has_column(conn, "media_meta", "item_date") else \
            "(SELECT substr(MIN(s.date), 1, 10) FROM source s WHERE s.media_id=m.id)"
        return q(f"SELECT id FROM (SELECT m.id, {date} AS d FROM media m) "
                 "WHERE d IS NOT NULL AND (? IS NULL OR d >= ?) AND (? IS NULL OR d <= ?)", lo, lo, hi, hi)
    if kind == "cluster":
        # Clusters hold post primaries; include each post's photos and videos so scopes intersect cleanly.
        ids = q("SELECT media_id FROM cluster_member WHERE cluster_id=?", int(value))
        return set(expand_members(conn, list(ids)))
    if kind == "caption":
        # caption:current, images captioned with the current image prompt; caption:older, with an earlier one.
        from re2.extract_images import PROMPT_VERSION

        if value not in ("current", "older"):
            raise ScopeError(f"bad caption scope {value!r}")
        op = ">=" if value == "current" else "<"
        return q("SELECT media_id FROM doc WHERE kind='caption' "
                 f"AND coalesce(json_extract(json, '$.prompt_version'), 0) {op} ?", PROMPT_VERSION)
    if kind == "page":
        if not _has_table(conn, "wiki_member"):
            return set()
        return q("SELECT wm.media_id FROM wiki_member wm JOIN wiki_page p ON p.id=wm.page_id WHERE p.slug=?", value)
    if kind == "media":
        media_id = int(value)
        ids = {media_id}
        if _has_table(conn, "wiki_member"):
            ids |= q("SELECT m2.media_id FROM wiki_member m1 JOIN wiki_member m2 ON m2.page_id=m1.page_id "
                     "JOIN wiki_page p ON p.id=m1.page_id WHERE m1.media_id=? AND p.kind='event'", media_id)
        return ids
    raise ScopeError(kind)


def scope_media_ids(conn: sqlite3.Connection, scopes: list[str]) -> set[int] | None:
    """Media ids inside every scope, or None when there are no scopes. Never includes EXCLUDED media."""
    if not scopes:
        return None
    result: set[int] | None = None
    for scope in scopes:
        ids = _scope_ids(conn, *parse_scope(scope))
        result = ids if result is None else result & ids
        if not result:
            return set()
    excluded = {r[0] for r in conn.execute("SELECT id FROM media WHERE status IN ('EXCLUDED', 'UNSUPPORTED')")}
    return result - excluded


def scope_label(conn: sqlite3.Connection, scope: str) -> str:
    kind, value = parse_scope(scope)
    row = None
    if kind == "entity":
        row = conn.execute("SELECT name FROM entity WHERE id=?", (int(value),)).fetchone()
    elif kind == "tag":
        row = conn.execute("SELECT name FROM tag WHERE id=?", (int(value),)).fetchone()
    elif kind == "page" and _has_table(conn, "wiki_page"):
        row = conn.execute("SELECT title FROM wiki_page WHERE slug=?", (value,)).fetchone()
    elif kind == "media":
        return f"#{value}"
    elif kind == "caption":
        from re2.extract_images import PROMPT_VERSION

        return f"current prompt (v{PROMPT_VERSION})" if value == "current" else f"older than v{PROMPT_VERSION}"
    elif kind == "cluster" and _has_table(conn, "cluster"):
        row = conn.execute("SELECT label FROM cluster WHERE id=?", (int(value),)).fetchone()
    return row[0] if row else value


# ── Lexical ────────────────────────────────────────────────────────────────

_PHRASE = re.compile(r'"([^"]+)"')
_TOKEN = re.compile(r"[\w][\w'.@-]*", re.UNICODE)


def fts_query(text: str, prefix_last: bool = True, joiner: str = " ") -> str:
    """User text to an FTS5 query: quoted tokens, "phrases" kept, last token prefix-matched."""
    parts: list[str] = []
    for phrase in _PHRASE.findall(text):
        words = _TOKEN.findall(phrase)
        if words:
            parts.append('"' + " ".join(words) + '"')
    rest = _PHRASE.sub(" ", text)
    tokens = [t for t in _TOKEN.findall(rest) if len(t) >= 2]
    ends_with_space = text[-1:].isspace() or text.rstrip().endswith('"')
    for i, tok in enumerate(tokens):
        quoted = '"' + tok.replace('"', "") + '"'
        if prefix_last and i == len(tokens) - 1 and not ends_with_space:
            quoted += "*"
        parts.append(quoted)
    return joiner.join(parts)


def _temp_ids(conn: sqlite3.Connection, name: str, ids) -> str:
    conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} (id INTEGER PRIMARY KEY)")
    conn.execute(f"DELETE FROM {name}")
    conn.executemany(f"INSERT INTO {name}(id) VALUES (?)", ((int(i),) for i in ids))
    return name


def lexical(conn: sqlite3.Connection, q: str, k: int = 200, allowed: set[int] | None = None) -> list[tuple[int, float]]:
    """(doc_id, score) with higher scores better."""
    restrict = ""
    if allowed is not None:
        restrict = f"AND d.media_id IN (SELECT id FROM {_temp_ids(conn, 'scope_media', allowed)})"
    sql = f"""SELECT doc_fts.rowid, bm25(doc_fts) AS s FROM doc_fts
              JOIN doc d ON d.id=doc_fts.rowid JOIN media m ON m.id=d.media_id
              WHERE doc_fts MATCH ? AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED') {restrict}
              ORDER BY s LIMIT ?"""
    results: list[tuple[int, float]] = []
    for joiner in (" ", " OR "):
        match = fts_query(q, joiner=joiner)
        if not match:
            return []
        try:
            results = [(r[0], -r[1]) for r in conn.execute(sql, (match, k))]
        except sqlite3.OperationalError as exc:  # malformed query from odd input
            log.debug("fts query %r failed: %s", match, exc)
            return []
        if len(results) >= 5 or joiner != " " or len(fts_query(q).split()) < 2:
            break
    return results


# ── Dense ──────────────────────────────────────────────────────────────────

def dense(conn: sqlite3.Connection, client, cfg, q: str, k: int = 200,
          allowed: set[int] | None = None) -> list[tuple[int, float]]:
    index = embed.get_index(cfg, "docs")
    allowed_docs = None
    if allowed is not None:
        table = _temp_ids(conn, "scope_media", allowed)
        allowed_docs = [r[0] for r in conn.execute(f"SELECT id FROM doc WHERE media_id IN (SELECT id FROM {table})")]
    return index.search(embed.query_vector(client, cfg, q), k=k, allowed_ids=allowed_docs)


def fuse(lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion."""
    scores: dict[int, float] = {}
    for results in lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ── Hits and cards ─────────────────────────────────────────────────────────

@dataclass
class Hit:
    media_id: int
    doc_id: int
    doc_kind: str
    score: float
    snippet: str
    start_s: float | None = None
    end_s: float | None = None
    page: int | None = None
    member_id: int | None = None   # the photo or video inside a post that actually matched


def _snippet(conn: sqlite3.Connection, doc_id: int, q: str, text: str) -> str:
    match = fts_query(q, joiner=" OR ")
    if match:
        try:
            row = conn.execute(
                f"SELECT snippet(doc_fts, 0, '{MARK_START}', '{MARK_END}', '…', 24) FROM doc_fts "
                "WHERE doc_fts MATCH ? AND rowid=?", (match, doc_id)).fetchone()
            if row and MARK_START in row[0]:
                return row[0]
        except sqlite3.OperationalError:
            pass
    text = " ".join(text.split())
    return text[:240] + ("…" if len(text) > 240 else "")


# ── Posts ──────────────────────────────────────────────────────────────────

_post_cache: dict[str, tuple[float, dict[int, int], dict[int, list[int]]]] = {}
POST_TTL_S = 60


def post_maps(conn: sqlite3.Connection) -> tuple[dict[int, int], dict[int, list[int]]]:
    """(member -> post's primary media, primary -> all member media in order), cached briefly."""
    if not _has_table(conn, "post_member"):
        return {}, {}
    db_file = conn.execute("PRAGMA database_list").fetchone()[2] or f"memory:{id(conn)}"
    now = time.monotonic()
    cached = _post_cache.get(db_file)
    if cached and now - cached[0] < POST_TTL_S:
        return cached[1], cached[2]
    to_primary: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    for post_id, primary, media_id in conn.execute(
            """SELECT p.id, p.primary_media_id, pm.media_id FROM post p JOIN post_member pm ON pm.post_id=p.id
               ORDER BY p.id, pm.seq"""):
        members.setdefault(primary, []).append(media_id)
        to_primary.setdefault(media_id, primary)   # a photo reused by two posts belongs to the first
    to_primary = {m: p for m, p in to_primary.items() if m != p}
    _post_cache[db_file] = (now, to_primary, members)
    return to_primary, members


def expand_members(conn: sqlite3.Connection, media_ids: list[int]) -> list[int]:
    """Posts' primaries followed by their members, so lenses see every photo's entities and tags."""
    _, members = post_maps(conn)
    out, seen = [], set()
    for m in media_ids:
        for x in members.get(m, [m]):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


SORTS = ("relevance", "newest", "oldest", "recent")


def caption_progress(conn: sqlite3.Connection) -> dict:
    """How far captioning with the current image prompt has got, for watching a run live."""
    from re2.extract_images import PROMPT_VERSION

    row = conn.execute(
        """SELECT count(*), max(created_at), max(id),
                  sum(created_at >= datetime('now', '-10 minutes')) FROM doc
           WHERE kind='caption' AND coalesce(json_extract(json, '$.prompt_version'), 0) >= ?""",
        (PROMPT_VERSION,)).fetchone()
    images = conn.execute(
        "SELECT count(*) FROM media WHERE kind='image' AND status IN ('READY', 'THUMBNAIL_ONLY')").fetchone()[0]
    return {"version": PROMPT_VERSION, "done": row[0], "images": images, "last_at": row[1],
            "last_doc": row[2], "per_min_10": round((row[3] or 0) / 10, 1)}


def sort_media(conn: sqlite3.Connection, media_ids: list[int], sort: str) -> list[int]:
    """Reorder ids: newest/oldest by item date (undated last), recent by when the models last read them."""
    if sort == "relevance" or len(media_ids) < 2:
        return media_ids
    table = _temp_ids(conn, "sort_media", media_ids)
    if sort == "recent":
        key_sql = ("(SELECT MAX(d.created_at) FROM doc d WHERE d.media_id=m.id "
                   "AND d.kind IN ('caption', 'transcript', 'keyframe', 'pdf_page'))")
        direction = "DESC"
    else:
        key_sql = item_date_sql("m") if _has_column(conn, "media_meta", "item_date") else \
            "(SELECT substr(MIN(s.date), 1, 10) FROM source s WHERE s.media_id=m.id)"
        direction = "DESC" if sort == "newest" else "ASC"
    rank = {m: i for i, m in enumerate(media_ids)}
    keyed = conn.execute(f"SELECT m.id, {key_sql} FROM media m WHERE m.id IN (SELECT id FROM {table})").fetchall()
    present = sorted((r for r in keyed if r[1] is not None), key=lambda r: (r[1], -rank[r[0]]),
                     reverse=(direction == "DESC"))
    missing = sorted((r for r in keyed if r[1] is None), key=lambda r: rank[r[0]])
    return [r[0] for r in present] + [r[0] for r in missing]


def group_by_media(conn: sqlite3.Connection, ranked: list[tuple[int, float]], q: str,
                   limit: int = 40, offset: int = 0, sort: str = "relevance") -> tuple[list[Hit], list[int]]:
    """Best doc per item, in rank order, with a post's photos and videos folded into the post.

    Returns (hits for the page, all item ids in order).
    """
    if not ranked:
        return [], []
    table = _temp_ids(conn, "rank_docs", [d for d, _ in ranked])
    meta = {r[0]: r for r in conn.execute(
        f"SELECT id, media_id, kind, start_s, end_s, page, text FROM doc WHERE id IN (SELECT id FROM {table})")}
    to_primary, _ = post_maps(conn)
    order: list[int] = []
    best: dict[int, tuple[int, float]] = {}
    for doc_id, score in ranked:
        row = meta.get(doc_id)
        if row is None:
            continue
        item = to_primary.get(row[1], row[1])
        if item not in best:
            best[item] = (doc_id, score)
            order.append(item)
    order = sort_media(conn, order, sort)
    hits = []
    for item in order[offset:offset + limit]:
        doc_id, score = best[item]
        _, media_id, kind, start_s, end_s, page, text = meta[doc_id]
        hit = Hit(item, doc_id, kind, round(score, 5), _snippet(conn, doc_id, q, text), start_s, end_s, page)
        hit.member_id = media_id if media_id != item else None
        hits.append(hit)
    return hits, order


_BOILERPLATE = re.compile(
    r"^((the |this )?(image|picture|photo|photograph|screenshot)\s+(is|shows|appears to be|appears to show|"
    r"contains|depicts|displays|features)|this is|it is|here is)\s+(a |an )?"
    r"((screenshot|photo|photograph|picture|image|capture) (of|from)\s+)?(a |an |the )?",
    re.IGNORECASE)
MIN_TITLE = 30


# The 2024 captioner's disclaimers ("The race and gender of anyone in the picture cannot be
# determined") make useless titles; such sentences are skipped.
_DISCLAIMER = re.compile(r"\b(race|gender|ethnicity)\b.*\b(cannot|can't|not|no|unclear|difficult|unable)\b|"
                         r"\b(cannot|can't|not|no|unclear|difficult|unable)\b.*\b(race|gender|ethnicity)\b",
                         re.IGNORECASE)


def title_from_text(text: str | None, max_len: int = 90) -> str | None:
    if not text:
        return None
    text = _BOILERPLATE.sub("", " ".join(text.split()))
    sentences = [x for x in re.split(r"(?<=[.!?])\s", text) if not _DISCLAIMER.search(x)]
    if not sentences:
        return None
    sentences[0] = _BOILERPLATE.sub("", sentences[0])
    sentence = sentences[0]
    for extra in sentences[1:]:
        if len(sentence) >= MIN_TITLE:
            break
        sentence = sentence.rstrip(".") + ": " + _BOILERPLATE.sub("", extra)
    sentence = sentence.rstrip(".")
    if not sentence:
        return None
    sentence = sentence[0].upper() + sentence[1:]
    return sentence if len(sentence) <= max_len else sentence[: max_len - 1].rsplit(" ", 1)[0] + "…"


def cards(conn: sqlite3.Connection, media_ids: list[int]) -> dict[int, dict]:
    if not media_ids:
        return {}
    table = _temp_ids(conn, "card_media", media_ids)
    has_digest = _has_table(conn, "item_digest")
    has_item_date = _has_column(conn, "media_meta", "item_date")
    digest_sql = ("(SELECT title FROM item_digest g WHERE g.media_id=m.id)" if has_digest else "NULL")
    date_sql = item_date_sql("m") if has_item_date else \
        "(SELECT substr(MIN(s.date), 1, 10) FROM source s WHERE s.media_id=m.id)"
    rows = conn.execute(f"""
        SELECT m.id, m.kind, m.status, m.thumb_path, m.width, m.height, m.duration, m.legacy_caption,
               mm.sensitivity, m.legacy_spicy, mm.media_type, mm.low_res,
               {digest_sql} AS digest_title, {date_sql} AS item_date,
               {"(SELECT date_source FROM media_meta x WHERE x.media_id=m.id)" if has_item_date else "NULL"} AS date_source,
               (SELECT json_extract(d.json, '$.description') FROM doc d
                 WHERE d.media_id=m.id AND d.kind='caption') AS description,
               (SELECT json_extract(d.json, '$.title') FROM doc d
                 WHERE d.media_id=m.id AND d.kind='caption') AS caption_title,
               (SELECT json_extract(d.json, '$.summary') FROM doc d
                 WHERE d.media_id=m.id AND d.kind='caption') AS summary,
               (SELECT d.text FROM doc d WHERE d.media_id=m.id AND d.kind != 'caption'
                 ORDER BY CASE d.kind WHEN 'legacy_caption' THEN 0 WHEN 'keyframe' THEN 1 WHEN 'pdf_page' THEN 2
                          WHEN 'transcript' THEN 3 ELSE 4 END, d.seq LIMIT 1) AS other_text,
               (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id ORDER BY p.readable DESC, p.id LIMIT 1) AS rel_path,
               (SELECT s.url FROM source s WHERE s.media_id=m.id ORDER BY s.id LIMIT 1) AS source_url
        FROM media m LEFT JOIN media_meta mm ON mm.media_id=m.id
        WHERE m.id IN (SELECT id FROM {table})""").fetchall()
    posts: dict[int, dict] = {}
    if _has_table(conn, "post"):
        for pr in conn.execute(
                f"""SELECT p.primary_media_id, p.url, pm.media_id, pm.role, pm.seq, m2.kind, m2.thumb_path,
                           m2.duration, m2.status
                    FROM post p JOIN post_member pm ON pm.post_id=p.id JOIN media m2 ON m2.id=pm.media_id
                    WHERE p.primary_media_id IN (SELECT id FROM {table}) ORDER BY p.id, pm.seq"""):
            post = posts.setdefault(pr[0], {"url": pr[1], "members": []})
            if post["url"] != pr[1]:
                continue  # a screenshot shared by two posts: keep the first
            post["members"].append({"media_id": pr[2], "role": pr[3], "seq": pr[4], "kind": pr[5],
                                    "thumb_url": f"/media/{pr[2]}/thumb" if pr[6] else None,
                                    "duration": pr[7], "status": pr[8]})
    out = {}
    for r in rows:
        rel_path = r["rel_path"] or ""
        folder = os.path.dirname(rel_path) or "."
        title = (r["caption_title"] or r["digest_title"] or title_from_text(r["description"]) or title_from_text(r["legacy_caption"])
                 or title_from_text(r["other_text"]) or os.path.basename(rel_path) or f"#{r['id']}")
        out[r["id"]] = {
            "media_id": r["id"],
            "kind": r["kind"],
            "status": r["status"],
            "title": title,
            "filename": os.path.basename(rel_path),
            "folder": folder,
            "date": r["item_date"],
            "date_source": r["date_source"] or ("source" if r["item_date"] else None),
            "sensitivity": r["sensitivity"],    # from the vision model; the only value used to veil
            "legacy_spicy": r["legacy_spicy"],  # the 2024 tagger's guess, shown but not trusted
            "media_type": r["media_type"],
            "low_res": bool(r["low_res"]),
            "width": r["width"], "height": r["height"], "duration": r["duration"],
            "thumb_url": f"/media/{r['id']}/thumb" if r["thumb_path"] else None,
            "media_url": f"/media/{r['id']}/file" if r["status"] == "READY" else None,
            "source_url": r["source_url"],
            "summary": r["summary"],
            "post": posts.get(r["id"]) if len(posts.get(r["id"], {}).get("members", [])) > 1 else None,
        }
    return out


def hit_dicts(conn: sqlite3.Connection, hits: list[Hit]) -> list[dict]:
    card_map = cards(conn, [h.media_id for h in hits])
    return [{**card_map[h.media_id], "hit": asdict(h)} for h in hits if h.media_id in card_map]


# ── Lenses and correlations ────────────────────────────────────────────────

LENS_MEDIA = 500      # search results: lenses describe the best matches
SCOPE_LENS_MEDIA = 20000  # browsing a scope with no query: lenses describe everything in it


def lenses(conn: sqlite3.Connection, media_ids: list[int], max_media: int = LENS_MEDIA) -> dict:
    ids = media_ids[:max_media]
    if not ids:
        return {"n": 0, "topics": [], "themes": [], "entities": [], "tags": [], "folders": [], "kinds": {},
                "statuses": {}, "months": []}
    table = _temp_ids(conn, "lens_media", ids)
    entities = [dict(r) for r in conn.execute(
        f"""SELECT e.id, e.name, e.type, COUNT(*) AS n FROM media_entity me JOIN entity e ON e.id=me.entity_id
            WHERE me.media_id IN (SELECT id FROM {table}) GROUP BY e.id ORDER BY n DESC, e.name LIMIT 20""")]
    tag_rows = [dict(r) for r in conn.execute(
        f"""SELECT t.id, t.name, COUNT(DISTINCT mt.media_id) AS n FROM media_tag mt JOIN tag t ON t.id=mt.tag_id
            WHERE mt.media_id IN (SELECT id FROM {table}) GROUP BY t.id ORDER BY n DESC, t.name LIMIT 20""")]
    folders: Counter[str] = Counter()
    for media_id, rel_path in conn.execute(
            f"""SELECT media_id, MIN(rel_path) FROM media_path WHERE media_id IN (SELECT id FROM {table})
                GROUP BY media_id"""):
        folders[rel_path.split("/", 1)[0] if "/" in rel_path else "."] += 1
    kinds = dict(conn.execute(f"SELECT kind, COUNT(*) FROM media WHERE id IN (SELECT id FROM {table}) GROUP BY kind").fetchall())
    statuses = dict(conn.execute(
        f"SELECT status, COUNT(*) FROM media WHERE id IN (SELECT id FROM {table}) GROUP BY status").fetchall())
    date = item_date_sql("m") if _has_column(conn, "media_meta", "item_date") else \
        "(SELECT substr(MIN(s.date), 1, 10) FROM source s WHERE s.media_id=m.id)"
    months = [{"month": r[0], "n": r[1]} for r in conn.execute(
        f"""SELECT substr(d, 1, 7) AS month, COUNT(*) FROM (SELECT {date} AS d FROM media m
            WHERE m.id IN (SELECT id FROM {table})) WHERE length(d) >= 7 GROUP BY month ORDER BY month""")]
    clusters = {"topics": [], "themes": []}
    run = conn.execute("SELECT value FROM kv WHERE key='cluster_run'").fetchone() if _has_table(conn, "cluster") else None
    if run:
        for r in conn.execute(
                f"""SELECT c.id, c.label, c.level, COUNT(*) AS n FROM cluster_member cm JOIN cluster c ON c.id=cm.cluster_id
                    WHERE c.run_id=? AND cm.media_id IN (SELECT id FROM {table})
                    GROUP BY c.id ORDER BY n DESC""", (run[0],)):
            bucket = clusters["topics" if r[2] == "topic" else "themes"]
            if len(bucket) < 12:
                bucket.append({"id": r[0], "label": r[1], "n": r[3]})
    return {
        "n": len(ids),
        "topics": clusters["topics"],
        "themes": clusters["themes"],
        "entities": entities,
        "tags": tag_rows,
        "folders": [{"folder": f, "n": n} for f, n in folders.most_common(20)],
        "kinds": kinds,
        "statuses": statuses,
        "months": months,
    }


_overall_cache: dict[tuple[str, str], tuple[float, int, dict]] = {}
OVERALL_TTL_S = 120


def _overall_counts(conn: sqlite3.Connection, what: str) -> tuple[int, dict[int, int]]:
    now = time.monotonic()
    db_file = conn.execute("PRAGMA database_list").fetchone()[2] or f"memory:{id(conn)}"
    key = (what, db_file)
    cached = _overall_cache.get(key)
    if cached and now - cached[0] < OVERALL_TTL_S:
        return cached[1], cached[2]
    n_items = conn.execute("SELECT COUNT(*) FROM media WHERE status NOT IN ('EXCLUDED', 'UNSUPPORTED')").fetchone()[0]
    if what == "entity":
        counts = dict(conn.execute("SELECT entity_id, COUNT(*) FROM media_entity GROUP BY entity_id").fetchall())
    else:
        counts = dict(conn.execute("SELECT tag_id, COUNT(DISTINCT media_id) FROM media_tag GROUP BY tag_id").fetchall())
    _overall_cache[key] = (now, n_items, counts)
    return n_items, counts


def correlations(conn: sqlite3.Connection, media_ids: list[int], limit: int = 15, min_count: int = 3,
                 max_media: int = LENS_MEDIA) -> list[dict]:
    """Entities and tags over-represented in the result set, scored by lift * log(count)."""
    ids = media_ids[:max_media]
    if len(ids) < min_count:
        return []
    table = _temp_ids(conn, "corr_media", ids)
    out = []
    for what, sql in (
        ("entity", f"""SELECT e.id, e.name, e.type, COUNT(*) FROM media_entity me JOIN entity e ON e.id=me.entity_id
                       WHERE me.media_id IN (SELECT id FROM {table}) GROUP BY e.id HAVING COUNT(*) >= ?"""),
        ("tag", f"""SELECT t.id, t.name, 'tag', COUNT(DISTINCT mt.media_id) FROM media_tag mt JOIN tag t ON t.id=mt.tag_id
                    WHERE mt.media_id IN (SELECT id FROM {table}) GROUP BY t.id HAVING COUNT(DISTINCT mt.media_id) >= ?"""),
    ):
        n_items, overall = _overall_counts(conn, what)
        for item_id, name, typ, n in conn.execute(sql, (min_count,)):
            base = overall.get(item_id, n) / max(n_items, 1)
            lift = (n / len(ids)) / max(base, 1e-9)
            if lift <= 1.5:
                continue
            out.append({"scope": f"{what}:{item_id}", "name": name, "type": typ, "n": n,
                        "lift": round(lift, 2), "score": round(lift * math.log(n), 3)})
    out.sort(key=lambda d: -d["score"])
    return out[:limit]


# ── Convenience ────────────────────────────────────────────────────────────

def items_in_scope(conn: sqlite3.Connection, allowed: set[int] | None, limit: int = 40,
                   offset: int = 0, sort: str = "newest") -> list[int]:
    """Item ids for an empty query, posts folded to their screenshot. Default: newest dated first."""
    restrict = ""
    if allowed is not None:
        restrict = f"AND m.id IN (SELECT id FROM {_temp_ids(conn, 'scope_media', allowed)})"
    ids = [r[0] for r in conn.execute(
        f"""SELECT m.id FROM media m
            WHERE m.status NOT IN ('EXCLUDED', 'UNSUPPORTED') AND m.kind IN ('image', 'video', 'pdf') {restrict}
            ORDER BY m.id DESC""")]
    to_primary, _ = post_maps(conn)
    items, seen = [], set()
    for m in ids:
        item = to_primary.get(m, m)
        if item not in seen:
            seen.add(item)
            items.append(item)
    items = sort_media(conn, items, "newest" if sort == "relevance" else sort)
    return items[offset:offset + limit]


def similar(conn: sqlite3.Connection, cfg, media_id: int, k: int = 12) -> list[int]:
    """Media whose docs are closest to the mean vector of this media's docs."""
    doc_ids = [r[0] for r in conn.execute("SELECT id FROM doc WHERE media_id=?", (media_id,))]
    index = embed.get_index(cfg, "docs")
    vecs, _ = index.vectors_for(doc_ids)
    if len(vecs) == 0:
        return []
    q = vecs.mean(axis=0)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    ranked = index.search(q, k=k * 6)
    table = _temp_ids(conn, "sim_docs", [d for d, _ in ranked])
    doc_media = dict(conn.execute(
        f"""SELECT d.id, d.media_id FROM doc d JOIN media m ON m.id=d.media_id
            WHERE d.id IN (SELECT id FROM {table}) AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED')""").fetchall())
    out: list[int] = []
    for doc_id, _ in ranked:
        m = doc_media.get(doc_id)
        if m is not None and m != media_id and m not in out:
            out.append(m)
        if len(out) >= k:
            break
    return out


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ── Co-occurrence graph ────────────────────────────────────────────────────

def cooccurrence(conn: sqlite3.Connection, allowed: set[int] | None, kind: str = "entity", limit: int = 150,
                 min_pair: int = 3, min_npmi: float = 0.05) -> dict:
    """Top entities (or tags) in scope as nodes; edges weighted by normalized PMI of sharing an item.

    npmi = log(p_ab / (p_a * p_b)) / -log(p_ab), in [-1, 1]; 0 means independent.
    """
    if kind == "entity":
        link, key, label_sql = "media_entity", "entity_id", "SELECT id, name, type FROM entity WHERE id IN"
    else:
        link, key, label_sql = "media_tag", "tag_id", "SELECT id, name, 'tag' FROM tag WHERE id IN"
    # Excluded and unsupported files never get entities or tags, so only a scope needs a media filter.
    restrict, pair_filter = "", ""
    if allowed is not None:
        table = _temp_ids(conn, "graph_media", allowed)
        restrict = f"WHERE l.media_id IN (SELECT id FROM {table})"
        pair_filter = f"AND a.media_id IN (SELECT id FROM {table})"
    n_items = conn.execute(f"SELECT COUNT(DISTINCT l.media_id) FROM {link} l {restrict}").fetchone()[0]
    counts = conn.execute(
        f"""SELECT l.{key}, COUNT(DISTINCT l.media_id) AS n FROM {link} l {restrict}
            GROUP BY l.{key} HAVING n >= 2 ORDER BY n DESC LIMIT ?""", (limit,)).fetchall()
    if not counts or not n_items:
        return {"kind": kind, "n_items": n_items, "nodes": [], "edges": []}
    node_counts = {r[0]: r[1] for r in counts}
    nodes_table = _temp_ids(conn, "graph_nodes", node_counts)
    labels = {r[0]: (r[1], r[2]) for r in conn.execute(f"{label_sql} (SELECT id FROM {nodes_table})")}
    # Pair counts as M.T @ M over an items x nodes incidence matrix: milliseconds, where the
    # equivalent SQL self-join over the most common tags takes over a second.
    node_ids = list(node_counts)
    col = {node: j for j, node in enumerate(node_ids)}
    rows = conn.execute(
        f"""SELECT DISTINCT l.media_id, l.{key} FROM {link} l
            WHERE l.{key} IN (SELECT id FROM {nodes_table}) {pair_filter.replace("a.media_id", "l.media_id")}""").fetchall()
    item_index: dict[int, int] = {}
    ii = np.fromiter((item_index.setdefault(m, len(item_index)) for m, _ in rows), dtype=np.int64, count=len(rows))
    jj = np.fromiter((col[n] for _, n in rows), dtype=np.int64, count=len(rows))
    incidence = np.zeros((len(item_index), len(node_ids)), dtype=np.float32)
    incidence[ii, jj] = 1.0
    together = incidence.T @ incidence
    edges = []
    for a_idx, b_idx in zip(*np.nonzero(np.triu(together, k=1) >= min_pair)):
        n_ab = int(together[a_idx, b_idx])
        a, b = node_ids[a_idx], node_ids[b_idx]
        p_ab = n_ab / n_items
        p_a, p_b = node_counts[a] / n_items, node_counts[b] / n_items
        npmi = 1.0 if p_ab >= 1 else math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)
        if npmi >= min_npmi:
            edges.append({"a": a, "b": b, "n": n_ab, "w": round(npmi, 3)})
    edges.sort(key=lambda e: (-e["w"], -e["n"]))
    edges = edges[: limit * 4]
    prefix = "entity" if kind == "entity" else "tag"
    nodes = [{"id": i, "scope": f"{prefix}:{i}", "name": labels.get(i, (str(i), ""))[0],
              "type": labels.get(i, ("", kind))[1], "n": n} for i, n in node_counts.items()]
    return {"kind": kind, "n_items": n_items, "nodes": nodes, "edges": edges}


def timeline_counts(conn: sqlite3.Connection, allowed: set[int] | None) -> dict:
    """Items per month by kind; `weak` counts dates that come only from file times."""
    restrict = ""
    if allowed is not None:
        restrict = f"AND m.id IN (SELECT id FROM {_temp_ids(conn, 'timeline_media', allowed)})"
    months: dict[str, dict] = {}
    undated = 0
    for month, kind, weak, n in conn.execute(
            f"""SELECT substr(mm.item_date, 1, 7), m.kind, mm.date_source = 'mtime', COUNT(*)
                FROM media m LEFT JOIN media_meta mm ON mm.media_id=m.id
                WHERE m.kind IN ('image', 'video', 'pdf')
                  AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED') {restrict}
                GROUP BY 1, 2, 3"""):
        if not month:
            undated += n
            continue
        row = months.setdefault(month, {"month": month, "image": 0, "video": 0, "pdf": 0, "weak": 0})
        row[kind] += n
        if weak:
            row["weak"] += n
    return {"months": [months[k] for k in sorted(months)], "undated": undated, "events": []}
