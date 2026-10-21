"""Phase 6.5: topics, themes and the 2D map, from item vectors (local CPU).

Item vectors are the mean of the item's document vectors in the `docs` index (a
post's photos and videos count toward the post). Topics come from HDBSCAN on a
15-d UMAP projection; themes group topic centroids; the map is a separate 2-d
UMAP. Labels are provisional keywords (class-based TF-IDF) until the wiki step
names clusters with the LLM.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from re2 import embed, search

log = logging.getLogger(__name__)

MIN_TOPIC_SIZE = 8
NOISE_JOIN_PERCENTILE = 10   # a leftover item joins a topic if it is as close as that topic's 10th-percentile member
THEME_RANGE = (12, 25)
KEEP_RUNS = 2

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves image screenshot shows appears shown text post posts user named handle picture
photo contains tweet tweets twitter social media account caption top bottom left right side background visible reads
white black blue red green yellow wearing man woman people person two one three also like seems possibly likely image's
lower upper center middle large small title header profile icon icons logo displays displayed featuring features includes
including various several section page website article screenshot's image. there's it's received garnered likes
retweets reposts replies views comments shares dated content video videos platform notification notifications miss embed
permalink thumbnail category categories cannot determine determined information headline news reads reading following
posted posting shared sharing mentions mentioned discusses discussing appears appear seen showing depicts depicting
""".split())
# The 2024 captioner often apologised that race or gender "cannot be determined"; those
# sentences say nothing about the item and would otherwise label whole topics.
_BOILERPLATE = re.compile(r"[^.\n]*\b(cannot|can not|can't|unable to|not possible to|no way to)\b[^.\n]*"
                          r"\b(race|gender|ethnicity|categor)[^.\n]*[.\n]?", re.IGNORECASE)
_WORD = re.compile(r"[a-z][a-z'-]{2,}")


@dataclass
class ClusterStats:
    items: int = 0
    topics: int = 0
    noise: int = 0
    themes: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (f"items={self.items} topics={self.topics} unclustered={self.noise} themes={self.themes} "
                f"elapsed={self.seconds:.0f}s")


def item_rows(conn: sqlite3.Connection, doc_ids: list[int]) -> dict[int, list[int]]:
    """Item id -> rows of doc_ids that belong to it: post members fold into their post, excluded media drop out."""
    to_primary, _ = search.post_maps(conn)
    doc_media = dict(conn.execute("SELECT id, media_id FROM doc").fetchall())
    groups: dict[int, list[int]] = defaultdict(list)
    for row, doc_id in enumerate(doc_ids):
        media_id = doc_media.get(doc_id)
        if media_id is not None:
            groups[to_primary.get(media_id, media_id)].append(row)
    excluded = {r[0] for r in conn.execute("SELECT id FROM media WHERE status IN ('EXCLUDED', 'UNSUPPORTED')")}
    return {m: rows for m, rows in groups.items() if m not in excluded}


def item_vectors(conn: sqlite3.Connection, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(item ids, L2-normalized mean doc vectors) for items with at least one indexed doc."""
    vecs, doc_ids = embed.get_index(cfg, "docs").load()
    if len(doc_ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, cfg.embed_dims), dtype=np.float32)
    groups = item_rows(conn, doc_ids.tolist())
    ids = np.array(sorted(groups), dtype=np.int64)
    out = np.zeros((len(ids), vecs.shape[1]), dtype=np.float32)
    for i, media_id in enumerate(ids.tolist()):
        out[i] = vecs[groups[media_id]].mean(axis=0)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return ids, out


def keyword_labels(texts_by_cluster: dict[int, list[str]], top: int = 4) -> dict[int, str]:
    """Class-based TF-IDF: words frequent in one cluster and rare across the others."""
    counts = {c: Counter(w for t in texts for w in _WORD.findall(_BOILERPLATE.sub(" ", t).lower())
                         if w not in STOPWORDS)
              for c, texts in texts_by_cluster.items()}
    df = Counter(w for cnt in counts.values() for w in cnt)
    n = max(len(counts), 1)
    labels = {}
    for c, cnt in counts.items():
        total = sum(cnt.values()) or 1
        scored = sorted(((cnt[w] / total) * math.log(1 + n / df[w]), w) for w in cnt if cnt[w] >= 2)
        labels[c] = " · ".join(w for _, w in reversed(scored[-top:])) or f"cluster {c}"
    return labels


def _item_texts(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    """A short text per item for labelling: caption titles and summaries when present, else doc starts."""
    _, members = search.post_maps(conn)
    wanted = {m: item for item in ids for m in members.get(item, [item])}
    out: dict[int, list[str]] = defaultdict(list)
    for media_id, kind, text, raw in conn.execute("SELECT media_id, kind, text, json FROM doc"):
        item = wanted.get(media_id)
        if item is None:
            continue
        if kind == "caption" and raw:
            d = json.loads(raw)
            out[item].append(" ".join(str(d.get(k) or "") for k in ("title", "summary", "meaning", "tags")))
        else:
            out[item].append(text[:600])
    return {k: " ".join(v) for k, v in out.items()}


def recluster(conn: sqlite3.Connection, cfg, seed: int = 42) -> ClusterStats:
    import umap
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering

    started = time.monotonic()
    stats = ClusterStats()
    ids, vecs = item_vectors(conn, cfg)
    stats.items = len(ids)
    if len(ids) < MIN_TOPIC_SIZE * 3:
        raise ValueError(f"only {len(ids)} items have vectors; run `re2 embed` first")

    log.info("UMAP 15-d over %d items", len(ids))
    reduced = umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine",
                        random_state=seed).fit_transform(vecs)
    labels = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=4).fit_predict(reduced)
    topic_ids = sorted(set(labels) - {-1})
    centroids = np.stack([vecs[labels == t].mean(axis=0) for t in topic_ids])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    # Unclustered items join the nearest topic when they are as close to it as its own
    # typical members. Embedding cosines here sit in a narrow high band, so a fixed
    # threshold either joins everything or nothing.
    floor = np.array([np.percentile(vecs[labels == t] @ centroids[k], NOISE_JOIN_PERCENTILE)
                      for k, t in enumerate(topic_ids)])
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        sims = centroids @ vecs[noise_idx].T           # topics x noise items
        best = sims.argmax(axis=0)
        joined = sims[best, np.arange(len(noise_idx))] >= floor[best]
        labels[noise_idx[joined]] = np.array(topic_ids)[best[joined]]
    stats.topics = len(topic_ids)
    stats.noise = int((labels == -1).sum())

    # Themes: Ward linkage on the unit-length centroids (Euclidean there tracks cosine).
    # Average linkage put two thirds of the archive into one theme; Ward keeps them comparable.
    n_themes = int(min(THEME_RANGE[1], max(THEME_RANGE[0], round(1.3 * math.sqrt(len(topic_ids))))))
    n_themes = min(n_themes, len(topic_ids))
    theme_of = AgglomerativeClustering(n_clusters=n_themes, linkage="ward").fit_predict(centroids)
    stats.themes = len(set(theme_of))

    log.info("UMAP 2-d map")
    coords = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=seed).fit_transform(vecs).astype(np.float32)
    index_dir = Path(cfg.data_dir) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "map2d.npy", coords)
    np.save(index_dir / "map2d_ids.npy", ids)
    embed.get_index(cfg, "items")._write(vecs, ids)

    texts = _item_texts(conn, ids.tolist())
    by_topic: dict[int, list[str]] = defaultdict(list)
    for media_id, t in zip(ids.tolist(), labels.tolist()):
        if t != -1:
            by_topic[t].append(texts.get(media_id, ""))
    topic_labels = keyword_labels(by_topic)
    by_theme: dict[int, list[str]] = defaultdict(list)
    for t_idx, t in enumerate(topic_ids):
        by_theme[int(theme_of[t_idx])].extend(by_topic[t])
    theme_labels = keyword_labels(by_theme, top=3)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    with conn:
        theme_row = {}
        for theme in sorted(set(theme_of.tolist())):
            theme_row[theme] = conn.execute(
                "INSERT INTO cluster(level, parent_id, run_id, label, label_source) VALUES ('theme', NULL, ?, ?, 'keywords')",
                (run_id, theme_labels[theme])).lastrowid
        topic_row = {}
        for t_idx, t in enumerate(topic_ids):
            topic_row[t] = conn.execute(
                "INSERT INTO cluster(level, parent_id, run_id, label, label_source) VALUES ('topic', ?, ?, ?, 'keywords')",
                (theme_row[int(theme_of[t_idx])], run_id, topic_labels[t])).lastrowid
        members = [(topic_row[t], m) for m, t in zip(ids.tolist(), labels.tolist()) if t != -1]
        conn.executemany("INSERT INTO cluster_member(cluster_id, media_id) VALUES (?, ?)", members)
        conn.executemany("INSERT INTO cluster_member(cluster_id, media_id) VALUES (?, ?)",
                         [(conn.execute("SELECT parent_id FROM cluster WHERE id=?", (c,)).fetchone()[0], m)
                          for c, m in members])
        old_runs = [r[0] for r in conn.execute(
            "SELECT DISTINCT run_id FROM cluster ORDER BY run_id DESC LIMIT -1 OFFSET ?", (KEEP_RUNS,))]
        for run in old_runs:
            conn.execute("DELETE FROM cluster_member WHERE cluster_id IN (SELECT id FROM cluster WHERE run_id=?)", (run,))
            conn.execute("DELETE FROM cluster WHERE run_id=?", (run,))
        conn.execute("INSERT INTO kv(key, value) VALUES ('cluster_run', ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (run_id,))
    stats.seconds = time.monotonic() - started
    return stats
