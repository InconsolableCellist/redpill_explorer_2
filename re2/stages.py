"""The ingestion pipeline: every stage in order, what it still has to do, and what it has done.

Each pending count follows the stage's own selection, so "533 pending" is what a run would take.
Local stages need no model and may run on their own (the Pipeline page's auto run); model stages
call a model server or the GPU and run only when started, on the service or preset chosen then.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from re2 import db


@dataclass(frozen=True)
class Stage:
    name: str                       # job_run.stage
    label: str
    command: str                    # the `re2` subcommand
    unit: str
    service: str | None             # vision | wiki | asr | embed | None (no model)
    auto: bool                      # runs on its own by default (local stages)
    failure_stages: tuple[str, ...]  # failure.stage values this stage writes
    pending: Callable[[sqlite3.Connection, object], int]
    done: Callable[[sqlite3.Connection, object], int | None] = lambda conn, cfg: None
    after: tuple[str, ...] = ()
    options: tuple[str, ...] = ("limit",)   # start options the command takes: limit, concurrency, preset
    help: str = ""

    @property
    def local(self) -> bool:
        return self.service is None


def _one(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> int:
    return conn.execute(sql, args).fetchone()[0] or 0


def _without_failures(conn, stage: str, sql: str) -> int:
    skip = db.pending_failures(conn, stage)
    return sum(1 for (mid,) in conn.execute(sql) if mid not in skip)


READABLE = "EXISTS (SELECT 1 FROM media_path p WHERE p.media_id=m.id AND p.readable=1)"


def scan_changes(conn: sqlite3.Connection, cfg) -> int:
    """Files that are new or changed since the last scan, plus known files that are gone (no hashing)."""
    from re2 import scan

    root = Path(cfg.corpus_root)
    if not root.is_dir():
        return 0
    known = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT rel_path, size, mtime_ns FROM media_path WHERE readable=1")}
    changed = seen = 0
    for entry in scan.walk(root):
        rel = os.path.relpath(entry.path, root)
        if scan.is_excluded(entry.name):
            known.pop(rel, None)
            continue
        try:
            st = os.stat(entry.path)   # follows symlinks, as scan does
        except OSError:
            continue
        seen += 1
        if known.pop(rel, None) != (st.st_size, st.st_mtime_ns):
            changed += 1
    return changed + len(known)


def _extract_pending(conn, cfg) -> int:
    from re2 import extract_images

    return len(extract_images.select_jobs(conn, cfg, None, True))


def _extract_done(conn, cfg) -> int:
    from re2.extract_images import PROMPT_VERSION

    return _one(conn, """SELECT COUNT(DISTINCT d.media_id) FROM doc d JOIN media m ON m.id=d.media_id
                         WHERE m.kind='image' AND d.kind='caption'
                           AND coalesce(json_extract(d.json, '$.prompt_version'), 0) >= ?""", (PROMPT_VERSION,))


def _ocr_pending(conn, cfg) -> int:
    failed = {(r[0], r[1]) for r in conn.execute(
        "SELECT media_id, doc_id FROM failure WHERE stage='pdf_ocr' AND attempts>=?", (db.MAX_ATTEMPTS,))}
    return sum(1 for r in conn.execute("SELECT media_id, page FROM pdf_ocr_queue") if (r[0], r[1]) not in failed)


def _sources_pending(conn, cfg) -> int:
    from re2 import legacy

    return legacy.sources_pending(conn, cfg)


def _asr_pending(conn, cfg) -> int:
    from re2 import asr

    return len(asr.select_jobs(conn, None))


ITEMS = "m.kind IN ('image', 'video', 'pdf') AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED')"


def _index_ids(cfg) -> set[int]:
    path = Path(cfg.data_dir) / "index" / "docs_ids.npy"
    try:
        return set(np.load(path).tolist())
    except (OSError, ValueError):
        return set()


def _embed_pending(conn, cfg) -> int:
    from re2 import embed

    have = _index_ids(cfg)
    return sum(1 for doc_id, _ in embed._docs_to_index(conn) if doc_id not in have)


def _recluster_pending(conn, cfg) -> int:
    """Items with vectors that the last recluster did not take in. Items it left out of every topic count as
    taken in, or this never reaches 0 and the supervisor reruns it every cycle."""
    from re2 import cluster

    have = _index_ids(cfg)
    if not have:
        return 0
    path = Path(cfg.data_dir) / "index" / "items_ids.npy"
    try:
        done = set(np.load(path).tolist()) if db.kv_get(conn, "cluster_run") else set()
    except (OSError, ValueError):
        done = set()
    return len(set(cluster.item_rows(conn, sorted(have))) - done)


def _digest_pending(conn, cfg) -> int:
    from re2 import digest

    copy, due = digest.pending(conn)
    return copy + due


def _wiki_pending(conn, cfg) -> int:
    return _one(conn, "SELECT COUNT(*) FROM wiki_page WHERE stale=1 OR (kind='topic' AND events_split_at IS NULL)")


STAGES: list[Stage] = [
    Stage("scan", "Find new files", "scan", "files", None, True, ("scan",), scan_changes,
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM media_path WHERE readable=1"),
          help="Reads the corpus folder and records new, changed and missing files. Nothing else sees a file until this has run."),
    Stage("sources", "Link tweets", "reconcile-sources", "files", None, True, (),
          lambda c, cfg: _sources_pending(c, cfg),
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM post"), after=("scan",), options=(),
          help="Groups each saved tweet's screenshot, photos and videos into one post, from archive_metadata.json."),
    Stage("thumbs", "Thumbnails", "thumbs", "images", None, True, ("thumb",),
          lambda c, cfg: _without_failures(c, "thumb", f"""SELECT m.id FROM media m WHERE m.kind='image'
              AND m.status='READY' AND m.thumb_path IS NULL AND m.sha256 IS NOT NULL AND {READABLE}"""),
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM media WHERE kind='image' AND thumb_path IS NOT NULL"),
          after=("scan",)),
    Stage("video_probe", "Video keyframes", "video-probe", "videos", None, True, ("keyframe",),
          lambda c, cfg: _without_failures(c, "keyframe", f"""SELECT m.id FROM media m WHERE m.kind='video'
              AND m.status='READY' AND m.duration IS NULL AND {READABLE}"""),
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM media WHERE kind='video' AND duration IS NOT NULL"),
          after=("scan",)),
    Stage("pdfs", "PDF text", "pdfs", "PDFs", None, True, ("pdf",),
          lambda c, cfg: _without_failures(c, "pdf", """SELECT m.id FROM media m WHERE m.kind='pdf'
              AND m.status='READY' AND NOT EXISTS (SELECT 1 FROM kv WHERE kv.key='pdf_read:' || m.id)"""),
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM kv WHERE key LIKE 'pdf_read:%'"),
          after=("scan",)),
    Stage("extract_images", "Image captions", "extract-images", "images", "vision", False, ("vlm_image",),
          _extract_pending, _extract_done, after=("scan",), options=("limit", "concurrency", "preset"),
          help="Captions, OCR, tags and entities for each image."),
    Stage("pdf_ocr", "Scanned PDF pages", "pdf-ocr", "pages", "vision", False, ("pdf_ocr",), _ocr_pending,
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM doc WHERE kind='pdf_page' AND json IS NOT NULL"),
          after=("pdfs",), options=("limit", "preset")),
    Stage("asr", "Speech to text", "asr", "videos", "asr", False, ("asr",), _asr_pending,
          lambda c, cfg: _one(c, "SELECT COUNT(DISTINCT media_id) FROM doc WHERE kind='transcript'"),
          after=("video_probe",), help="Parakeet on this machine's GPU, when the router leaves room. Other languages are skipped."),
    Stage("dates", "Item dates", "dates", "items", None, True, (),
          lambda c, cfg: _one(c, f"""SELECT COUNT(*) FROM media m WHERE {ITEMS}
              AND NOT EXISTS (SELECT 1 FROM media_meta mm WHERE mm.media_id=m.id AND mm.date_source IS NOT NULL)"""),
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM media_meta WHERE date_source IS NOT NULL"),
          after=("scan",), options=()),
    Stage("embed", "Search index", "embed", "texts", "embed", False, (), _embed_pending,
          lambda c, cfg: len(_index_ids(cfg)), help="Embeds new captions, transcripts and text for meaning search."),
    Stage("recluster", "Topics and map", "recluster", "items", None, False, (), _recluster_pending,
          after=("embed",), options=(), help="Regroups every item into topics and themes and redraws the map."),
    Stage("digests", "Item digests", "digests", "items", "wiki", False, ("digest",), _digest_pending,
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM item_digest"), options=("limit", "concurrency", "preset")),
    Stage("wiki_build", "Wiki pages", "wiki-build", "pages", "wiki", False, ("wiki", "wiki_split"), _wiki_pending,
          lambda c, cfg: _one(c, "SELECT COUNT(*) FROM wiki_page WHERE stale=0"), after=("digests",),
          options=("limit", "preset")),
]
BY_NAME = {s.name: s for s in STAGES}
BY_COMMAND = {s.command: s for s in STAGES}


def counts(conn: sqlite3.Connection, cfg, names: list[str] | None = None) -> dict[str, dict]:
    """{stage: {pending, done, error}}; a stage whose count fails reports the error instead."""
    out = {}
    for s in STAGES:
        if names and s.name not in names:
            continue
        try:
            out[s.name] = {"pending": s.pending(conn, cfg), "done": s.done(conn, cfg), "error": None}
        except Exception as exc:  # noqa: BLE001 - one broken count must not hide the others
            out[s.name] = {"pending": None, "done": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def failures(conn: sqlite3.Connection, stage: Stage) -> dict:
    if not stage.failure_stages:
        return {"total": 0, "gave_up": 0}
    marks = ",".join("?" * len(stage.failure_stages))
    row = conn.execute(f"""SELECT COUNT(*), SUM(attempts >= ? AND coalesce(kind, 'item') != 'transient')
                           FROM failure WHERE stage IN ({marks})""", (db.MAX_ATTEMPTS, *stage.failure_stages)).fetchone()
    return {"total": row[0] or 0, "gave_up": row[1] or 0}


def runs(conn: sqlite3.Connection) -> dict[str, dict]:
    """The latest run of each stage, and its latest finished one."""
    out: dict[str, dict] = {}
    for r in conn.execute("SELECT * FROM job_run WHERE id IN (SELECT MAX(id) FROM job_run GROUP BY stage)"):
        d = dict(r)
        d["args"] = json.loads(d.pop("args_json") or "{}")
        out[d["stage"]] = {"latest": d}
    for stage, at in conn.execute("SELECT stage, MAX(finished_at) FROM job_run WHERE state='finished' GROUP BY stage"):
        out.setdefault(stage, {})["last_finished"] = at
    return out
