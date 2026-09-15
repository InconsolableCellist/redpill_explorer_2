"""Tag relatedness: what co-occurs in the archive, and what is related in the world.

Two scores per pair of the most used tags, kept separately in `tag_relation`:
- observed: normalized PMI of two tags appearing on the same items (-1..1; 0 = independent),
  computed from the archive in seconds.
- semantic: an LLM's 0..1 judgement of how related the concepts are (israelis/jews ~0.95,
  gaza/election interference ~0.3). For each anchor tag the model sees the whole tag list and
  names the most related tags; a pair scored from both sides keeps the mean.

The LLM pass is resumable (`tag_relation_done`) and uses the [chat] model settings.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass

import httpx
import numpy as np

from re2 import runner

log = logging.getLogger(__name__)

MAX_TAGS = 400
MIN_ITEMS = 15
MAX_RELATED = 25
MIN_PAIR = 3
TIMEOUT = 600.0

# Tags that describe the format or platform rather than the content; they relate to everything.
FORMAT_TAGS = {
    "screenshot", "social media", "twitter", "tweet", "video", "news article", "photography", "photo", "graph",
    "data visualization", "news", "media", "meme", "memes", "image", "text", "comparison", "political commentary",
    "social commentary", "political meme", "infographic", "chart", "document", "article", "headline", "post",
    "social media post", "screenshot of tweet", "twitter screenshot", "social media screenshot", "x", "x twitter",
    "reddit", "facebook", "youtube", "instagram", "tiktok", "online", "internet", "website", "collage",
    # Catch-alls that the model rates as related to nearly everything.
    "politics", "political", "controversy", "usa", "united states", "government", "social issues", "health issues",
    "legal issues", "international relations", "online conflict", "allegations", "statistics", "culture", "society",
    "history", "world", "public opinion", "current events",
}

TABLES = """
CREATE TABLE IF NOT EXISTS tag_relation (
  a INTEGER NOT NULL REFERENCES tag(id),      -- a < b
  b INTEGER NOT NULL REFERENCES tag(id),
  observed REAL,                              -- normalized PMI in the archive
  n_ab INTEGER,                               -- items carrying both
  semantic_ab REAL,                           -- the model's score when asked about a
  semantic_ba REAL,                           -- ... and when asked about b
  PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS ix_tag_relation_b ON tag_relation(b);
CREATE TABLE IF NOT EXISTS tag_relation_done (
  tag_id INTEGER PRIMARY KEY REFERENCES tag(id),
  model TEXT, done_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SYSTEM_PROMPT = """\
You rate how related search tags are, for a private archive of political and social-media material. Two tags are related when someone looking into one would want items tagged with the other, through shared people, groups, places, events, history, causes or ideology. Judge real-world relatedness, not wording.

Scale: 1.0 the same thing or inseparable; 0.9 very closely bound (israelis and jews; gaza and israel hamas war; gaza and jews); 0.7 strongly linked; 0.5 often connected; 0.3 occasionally connected (gaza and election interference). Leave out anything below 0.3. Do not moralise; offensive tags are rated like any other.

The tags:
{tags}"""

USER_PROMPT = "Tag: {anchor}\nName up to {n} tags from the list that are most related to it, most related first, with scores."


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(TABLES)


def candidate_tags(conn: sqlite3.Connection, max_tags: int = MAX_TAGS, min_items: int = MIN_ITEMS) -> list[tuple]:
    """(tag id, name, item count) for the most used content tags."""
    rows = conn.execute(
        """SELECT t.id, t.name, COUNT(DISTINCT mt.media_id) AS n FROM tag t JOIN media_tag mt ON mt.tag_id=t.id
           GROUP BY t.id HAVING n >= ? ORDER BY n DESC""", (min_items,)).fetchall()
    return [tuple(r) for r in rows if r[1] not in FORMAT_TAGS][:max_tags]


def observed_scores(conn: sqlite3.Connection, tags: list[tuple]) -> dict[tuple[int, int], tuple[float, int]]:
    """(a, b) with a < b -> (normalized PMI, items with both), for pairs seen together MIN_PAIR+ times."""
    ids = [t[0] for t in tags]
    col = {tag_id: j for j, tag_id in enumerate(ids)}
    n_items = conn.execute("SELECT COUNT(DISTINCT media_id) FROM media_tag").fetchone()[0] or 1
    rows = conn.execute(
        f"SELECT DISTINCT media_id, tag_id FROM media_tag WHERE tag_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    items: dict[int, int] = {}
    ii = np.fromiter((items.setdefault(m, len(items)) for m, _ in rows), dtype=np.int64, count=len(rows))
    jj = np.fromiter((col[t] for _, t in rows), dtype=np.int64, count=len(rows))
    incidence = np.zeros((max(len(items), 1), len(ids)), dtype=np.float32)
    incidence[ii, jj] = 1.0
    together = incidence.T @ incidence
    counts = np.diag(together).copy()
    out = {}
    for x, y in zip(*np.nonzero(np.triu(together, k=1) >= MIN_PAIR)):
        n_ab = float(together[x, y])
        p_ab = n_ab / n_items
        npmi = 1.0 if p_ab >= 1 else float(np.log(p_ab / ((counts[x] / n_items) * (counts[y] / n_items))) / -np.log(p_ab))
        a, b = sorted((ids[x], ids[y]))
        out[(a, b)] = (round(npmi, 4), int(n_ab))
    return out


def schema_for(names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"related": {
            "type": "array", "maxItems": MAX_RELATED,
            "items": {"type": "object",
                      "properties": {"tag": {"type": "string", "enum": names}, "score": {"type": "number"}},
                      "required": ["tag", "score"], "additionalProperties": False}}},
        "required": ["related"], "additionalProperties": False,
    }


def score_anchor(client: httpx.Client, cfg, anchor: str, names: list[str], system: str, schema: dict) -> list[dict]:
    body = {
        "model": cfg.chat_model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": USER_PROMPT.format(anchor=anchor, n=MAX_RELATED)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "related", "schema": schema}},
        "temperature": 0.2, "max_tokens": 900,
        "chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 0,
    }
    resp = client.post(f"{cfg.chat_base_url}/chat/completions", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"] or "{}").get("related", [])


@dataclass
class RelationStats:
    tags: int = 0
    observed_pairs: int = 0
    anchors_selected: int = 0
    anchors_done: int = 0
    semantic_pairs: int = 0
    failed: int = 0
    interrupted: bool = False
    started: float = 0.0

    def summary(self) -> str:
        rate = self.anchors_done * 60 / max(time.monotonic() - self.started, 1e-9)
        return (f"tags={self.tags} observed_pairs={self.observed_pairs} anchors={self.anchors_done}/"
                f"{self.anchors_selected} semantic_pairs={self.semantic_pairs} failed={self.failed} "
                f"rate={rate:.1f}/min" + (" INTERRUPTED" if self.interrupted else ""))


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, observed_only: bool = False,
        concurrency: int = 2, client_factory=httpx.Client, rescore: bool = False) -> RelationStats:
    ensure_tables(conn)
    if rescore:
        # After the tag vocabulary changes (e.g. a full re-extraction), every anchor is asked again.
        with conn:
            conn.execute("DELETE FROM tag_relation_done")
            conn.execute("UPDATE tag_relation SET semantic_ab=NULL, semantic_ba=NULL")
    stats = RelationStats(started=time.monotonic())
    tags = candidate_tags(conn)
    stats.tags = len(tags)
    if len(tags) < 2:
        return stats
    observed = observed_scores(conn, tags)
    with conn:
        conn.execute("UPDATE tag_relation SET observed=NULL, n_ab=NULL")
        conn.executemany(
            """INSERT INTO tag_relation(a, b, observed, n_ab) VALUES (?, ?, ?, ?)
               ON CONFLICT(a, b) DO UPDATE SET observed=excluded.observed, n_ab=excluded.n_ab""",
            [(a, b, score, n) for (a, b), (score, n) in observed.items()])
    stats.observed_pairs = len(observed)
    if observed_only:
        return stats

    by_name = {name: tag_id for tag_id, name, _ in tags}
    names = [name for _, name, _ in tags]
    system = SYSTEM_PROMPT.format(tags=", ".join(names))
    schema = schema_for(names)
    done = {r[0] for r in conn.execute("SELECT tag_id FROM tag_relation_done")}
    anchors = [(tag_id, name) for tag_id, name, _ in tags if tag_id not in done]
    if limit is not None:
        anchors = anchors[:limit]
    stats.anchors_selected = len(anchors)
    local = threading.local()
    clients: list[httpx.Client] = []
    lock = threading.Lock()

    def work(anchor):
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        return score_anchor(client, cfg, anchor[1], names, system, schema)

    def on_result(anchor, related):
        anchor_id = anchor[0]
        rows = []
        for r in related:
            other = by_name.get(r.get("tag"))
            if other is None or other == anchor_id:
                continue
            score = min(1.0, max(0.0, float(r.get("score") or 0)))
            a, b = sorted((anchor_id, other))
            rows.append((a, b, score if anchor_id == a else None, score if anchor_id == b else None))
        with conn:
            conn.executemany(
                """INSERT INTO tag_relation(a, b, semantic_ab, semantic_ba) VALUES (?, ?, ?, ?)
                   ON CONFLICT(a, b) DO UPDATE SET semantic_ab=COALESCE(excluded.semantic_ab, semantic_ab),
                                                   semantic_ba=COALESCE(excluded.semantic_ba, semantic_ba)""", rows)
            conn.execute("INSERT OR REPLACE INTO tag_relation_done(tag_id, model) VALUES (?, ?)",
                         (anchor_id, cfg.chat_model))
        stats.anchors_done += 1
        stats.semantic_pairs += len(rows)
        if stats.anchors_done % 20 == 0:
            log.info("progress: %s", stats.summary())

    def on_error(anchor, exc):
        stats.failed += 1
        log.warning("relatedness for %r failed: %s", anchor[1], exc)

    try:
        stats.interrupted = not runner.run_jobs(anchors, work, on_result, on_error, concurrency,
                                                total=len(anchors), desc="tag-relations")
    finally:
        for client in clients:
            client.close()
    return stats


def related(conn: sqlite3.Connection, tag_id: int, limit: int = 20) -> list[dict]:
    """Tags related to one tag: semantic score (mean of both directions) and observed co-occurrence."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='tag_relation'").fetchone():
        return []
    rows = conn.execute(
        """SELECT CASE WHEN r.a=? THEN r.b ELSE r.a END AS other, t.name, r.observed, r.n_ab,
                  CASE WHEN r.semantic_ab IS NOT NULL AND r.semantic_ba IS NOT NULL
                       THEN (r.semantic_ab + r.semantic_ba) / 2 ELSE COALESCE(r.semantic_ab, r.semantic_ba) END AS semantic
           FROM tag_relation r JOIN tag t ON t.id = CASE WHEN r.a=? THEN r.b ELSE r.a END
           WHERE r.a=? OR r.b=?""", (tag_id, tag_id, tag_id, tag_id)).fetchall()
    out = [{"id": r[0], "name": r[1], "observed": r[2], "n_ab": r[3], "semantic": r[4],
            "score": round(r[4] if r[4] is not None else max(r[2] or 0, 0), 3)} for r in rows]
    out.sort(key=lambda d: (-d["score"], -(d["n_ab"] or 0)))
    return out[:limit]


def related_graph(conn: sqlite3.Connection, allowed: set[int] | None, limit: int = 150, min_score: float = 0.5) -> dict:
    """Graph of the scored tags present in scope; edges are semantic relatedness >= min_score."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='tag_relation'").fetchone():
        return {"kind": "related", "n_items": 0, "nodes": [], "edges": []}
    scored = [r[0] for r in conn.execute("SELECT tag_id FROM tag_relation_done")]
    if not scored:
        return {"kind": "related", "n_items": 0, "nodes": [], "edges": []}
    marks = ",".join("?" * len(scored))
    params: list = list(scored)
    restrict = ""
    if allowed is not None:
        from re2 import search
        restrict = f"AND mt.media_id IN (SELECT id FROM {search._temp_ids(conn, 'related_graph_media', allowed)})"
    counts = conn.execute(
        f"""SELECT t.id, t.name, COUNT(DISTINCT mt.media_id) AS n FROM tag t JOIN media_tag mt ON mt.tag_id=t.id
            WHERE t.id IN ({marks}) {restrict} GROUP BY t.id HAVING n > 0 ORDER BY n DESC LIMIT ?""",
        params + [limit]).fetchall()
    ids = {r[0] for r in counts}
    n_items = conn.execute(f"SELECT COUNT(DISTINCT mt.media_id) FROM media_tag mt WHERE 1=1 {restrict}").fetchone()[0]
    edges = []
    for a, b, sab, sba, observed in conn.execute("SELECT a, b, semantic_ab, semantic_ba, observed FROM tag_relation"):
        if a not in ids or b not in ids:
            continue
        scores = [x for x in (sab, sba) if x is not None]
        if not scores:
            continue
        score = sum(scores) / len(scores)
        if score >= min_score:
            edges.append({"a": a, "b": b, "w": round(score, 3), "n": 0, "observed": observed})
    edges.sort(key=lambda e: -e["w"])
    return {"kind": "related", "n_items": n_items,
            "nodes": [{"id": r[0], "scope": f"tag:{r[0]}", "name": r[1], "type": "tag", "n": r[2]} for r in counts],
            "edges": edges[: limit * 4]}
