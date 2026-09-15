"""Phase 2: run the vision model over every image and store docs, tags, entities."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from re2 import db, runner, tags, vlm
from re2 import entities as entities_mod
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

STAGE = "vlm_image"
# Bump when the image prompt or schema changes enough that every caption should be redone.
# Captions made with an older version are extracted again after the uncaptioned images.
PROMPT_VERSION = 2
LOW_RES_SIDE = 500
USER_TEXT = "Catalogue this image."
LOW_RES_TEXT = " The image is a low-resolution thumbnail; transcribe what is legible."
ENTITY_TYPES = {
    "people": "person",
    "organizations": "organization",
    "places": "place",
    "events": "event",
    "platforms": "platform",
}


def caption_text(d: dict) -> str:
    # Headline first: snippets, embeddings and the wiki all read the start of the doc most.
    lede = "\n".join(x for x in (d.get("title"), d.get("summary")) if x)
    return (
        (f"{lede}\n\n" if lede else "") +
        f"{d['description']}\n\n"
        f"Subjects: {', '.join(d.get('subjects') or [])}\n"
        f"Actions: {d.get('actions', '')}\n"
        f"Meaning: {d.get('meaning', '')}\n"
        f"Claims: {'; '.join(d.get('claims') or [])}\n"
        f"Intent: {d.get('intent', '')}\n"
        f"Saved as: {d.get('saved_as', '')}\n"
        f"Later context: {d.get('later_context', '')}\n"
        f"Humor: {d.get('humor', '')}"
    )


def store_entities(conn: sqlite3.Connection, media_id: int, entities: dict) -> None:
    for key, etype in ENTITY_TYPES.items():
        for name in (entities or {}).get(key) or []:
            name = " ".join(str(name).split())
            if not name or entities_mod.is_placeholder(name):
                continue
            conn.execute("INSERT OR IGNORE INTO entity(name, type) VALUES (?, ?)", (name, etype))
            entity_id = conn.execute("SELECT id FROM entity WHERE name=? AND type=?", (name, etype)).fetchone()[0]
            entity_id = entities_mod.canonical_id(conn, entity_id)   # names merged earlier stay merged
            conn.execute("INSERT OR IGNORE INTO media_entity(media_id, entity_id) VALUES (?, ?)",
                         (media_id, entity_id))


def store_image_result(conn: sqlite3.Connection, cfg, media_id: int, d: dict, low_res: bool) -> None:
    """Write one extraction in a single transaction."""
    payload = {k: v for k, v in d.items() if not k.startswith("_")}
    payload["prompt_version"] = PROMPT_VERSION
    with conn:
        # Replace rather than update, so a redone caption gets a new doc id and the embed step,
        # which tracks doc ids, embeds the new text.
        conn.execute("DELETE FROM doc WHERE media_id=? AND kind IN ('caption', 'ocr') AND seq=0", (media_id,))
        conn.execute(
            "INSERT INTO doc(media_id, kind, seq, text, json, model) VALUES (?, 'caption', 0, ?, ?, ?)",
            (media_id, caption_text(d), json.dumps(payload, ensure_ascii=False), cfg.vision_model),
        )
        ocr = (d.get("ocr_text") or "").strip()
        if ocr:
            conn.execute("INSERT INTO doc(media_id, kind, seq, text, model) VALUES (?, 'ocr', 0, ?, ?)",
                         (media_id, ocr, cfg.vision_model))
        conn.execute(
            """INSERT INTO media_meta(media_id, sensitivity, media_type, language, low_res)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(media_id) DO UPDATE SET sensitivity=excluded.sensitivity,
                 media_type=excluded.media_type, language=excluded.language, low_res=excluded.low_res""",
            (media_id, d.get("sensitivity"), d.get("media_type"), d.get("language"), int(low_res)),
        )
        tags.set_media_tags(conn, media_id, {t: 1.0 for t in d.get("tags") or []}, "vlm")
        store_entities(conn, media_id, d.get("entities"))
        db.clear_failure(conn, STAGE, media_id)


def save_raw(cfg, media_id: int, exc: BaseException) -> None:
    """Keep the model's unparseable output for inspection: data/logs/vlm_raw/<media_id>.txt."""
    raw = getattr(exc, "raw", None)
    if not raw:
        return
    out = Path(cfg.data_dir) / "logs" / "vlm_raw" / f"{media_id}.txt"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# {exc}\n{raw}", encoding="utf-8")
    except OSError as err:
        log.debug("could not save raw output for %s: %s", media_id, err)


@dataclass
class Job:
    media_id: int
    path: Path
    low_res: bool
    long_output: bool = False  # tall image, or ran out of tokens before
    folder: str | None = None
    rel_path: str | None = None
    item_date: str | None = None
    redo: bool = False  # has a caption from an older prompt version


@dataclass
class ExtractStats:
    selected: int = 0
    done: int = 0
    failed: int = 0
    tiled: int = 0
    retried_long: int = 0
    redone: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    interrupted: bool = False
    started: float = 0.0

    def rate_per_min(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.done * 60 / elapsed if elapsed > 0 else 0.0

    def summary(self) -> str:
        avg_out = self.completion_tokens / self.done if self.done else 0
        avg_in = self.prompt_tokens / self.done if self.done else 0
        return (f"selected={self.selected} done={self.done} failed={self.failed} tiled={self.tiled} "
                f"retried_long={self.retried_long} redone={self.redone} "
                f"rate={self.rate_per_min():.1f}/min avg_prompt_tokens={avg_in:.0f} "
                f"avg_completion_tokens={avg_out:.0f}" + (" INTERRUPTED" if self.interrupted else ""))


def select_jobs(conn: sqlite3.Connection, cfg, limit: int | None, include_thumbnail_only: bool) -> list[Job]:
    statuses = ("READY", "THUMBNAIL_ONLY") if include_thumbnail_only else ("READY",)
    skip = db.pending_failures(conn, STAGE)
    ran_out = {r[0] for r in conn.execute(
        "SELECT media_id FROM failure WHERE stage=? AND error LIKE '%finish_reason=length%'", (STAGE,))}
    rows = conn.execute(
        f"""SELECT m.id, m.status, m.width, m.height, m.thumb_path,
                   (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                    ORDER BY p.id LIMIT 1) AS rel_path,
                   (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id ORDER BY p.id LIMIT 1) AS any_path,
                   (SELECT mm.item_date FROM media_meta mm WHERE mm.media_id=m.id) AS item_date,
                   EXISTS (SELECT 1 FROM doc d WHERE d.media_id=m.id AND d.kind='caption') AS redo
            FROM media m
            WHERE m.kind='image' AND m.status IN ({','.join('?' * len(statuses))})
              AND NOT EXISTS (SELECT 1 FROM doc d WHERE d.media_id=m.id AND d.kind='caption'
                              AND coalesce(json_extract(d.json, '$.prompt_version'), 0) >= ?)
            ORDER BY redo, m.status='THUMBNAIL_ONLY', m.id""",
        (*statuses, PROMPT_VERSION),
    ).fetchall()
    jobs = []
    for r in rows:
        if r["id"] in skip:
            continue
        if r["status"] == "READY" and r["rel_path"]:
            small = r["width"] is not None and max(r["width"], r["height"] or 0) < LOW_RES_SIDE
            path, low_res = Path(cfg.corpus_root) / r["rel_path"], small
        elif r["thumb_path"]:
            path, low_res = REPO_ROOT / r["thumb_path"], True
        else:
            continue
        long_output = vlm.is_tall(r["width"], r["height"]) or r["id"] in ran_out
        name = r["rel_path"] or r["any_path"]
        folder = os.path.dirname(name) if name else None
        jobs.append(Job(r["id"], path, low_res, long_output, folder, name, r["item_date"], bool(r["redo"])))
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


def sized_for_server(cfg, client_factory=httpx.Client, requested: int = 0):
    """cfg with vision_concurrency and vision_max_tokens_tall fitted to the server's slots and context."""
    import dataclasses

    with client_factory() as client:
        limits = vlm.server_limits(client, cfg)
    concurrency, tall, notes = vlm.plan_concurrency(cfg, limits, requested or cfg.vision_concurrency)
    for note in notes:
        log.info("vision server: %s", note)
    log.info("vision server: running %d requests at once; long retries up to %d tokens", concurrency, tall)
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, vision_concurrency=concurrency, vision_max_tokens_tall=tall)
    cfg.vision_concurrency, cfg.vision_max_tokens_tall = concurrency, tall
    return cfg


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, include_thumbnail_only: bool = True,
        client_factory=httpx.Client, concurrency: int = 0) -> ExtractStats:
    if concurrency or not cfg.vision_concurrency:
        cfg = sized_for_server(cfg, client_factory, concurrency)
    jobs = select_jobs(conn, cfg, limit, include_thumbnail_only)
    stats = ExtractStats(selected=len(jobs), started=time.monotonic())
    redo = sum(j.redo for j in jobs)
    log.info("%d images to extract: %d new, %d redone with prompt version %d",
             len(jobs), len(jobs) - redo, redo, PROMPT_VERSION)
    local = threading.local()
    clients: list[httpx.Client] = []
    lock = threading.Lock()

    def work(job: Job, max_tokens: int | None = None) -> dict:
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        images = vlm.prepare_images(job.path, cfg.vision_max_side, cfg.vision_max_tiles, dense=job.long_output)
        text = (USER_TEXT + (LOW_RES_TEXT if job.low_res else "") + vlm.folder_note(job.folder)
                + vlm.filename_note(job.rel_path) + vlm.date_note(job.item_date) + vlm.tiles_note(len(images)))
        result = vlm.extract(client, cfg, images, vlm.IMAGE_SCHEMA, vlm.IMAGE_SYSTEM_PROMPT, text,
                             max_tokens=max_tokens, sampling=vlm.LOOP_BREAKER if job.long_output else None)
        result["_tiles"] = len(images)
        return result

    def on_result(job: Job, result: dict) -> None:
        store_image_result(conn, cfg, job.media_id, result, job.low_res)
        usage = result.get("_usage") or {}
        stats.done += 1
        stats.redone += job.redo
        stats.tiled += result["_tiles"] > 1
        stats.completion_tokens += int(usage.get("completion_tokens") or 0)
        stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        if stats.done % 50 == 0:
            log.info("progress: %s", stats.summary())

    # Tall images (sliced into tiles) and text-dense images that ran out of
    # tokens need long outputs. The vision model shares one small context across
    # concurrent requests, so these run last, one at a time, with a larger budget.
    normal = [j for j in jobs if not j.long_output]
    long_jobs = [j for j in jobs if j.long_output]

    def on_error(job: Job, exc: BaseException) -> None:
        with conn:
            db.upsert_failure(conn, STAGE, job.media_id, None, f"{type(exc).__name__}: {exc}"[:2000])
        save_raw(cfg, job.media_id, exc)
        if isinstance(exc, vlm.VLMLengthError) and not job.long_output:
            job.long_output = True
            long_jobs.append(job)
            stats.retried_long += 1
            log.info("media %s ran out of tokens; will retry with %d", job.media_id, cfg.vision_max_tokens_tall)
            return
        stats.failed += 1
        log.warning("media %s failed: %s", job.media_id, exc)

    try:
        finished = runner.run_jobs(normal, work, on_result, on_error, cfg.vision_concurrency,
                                   total=len(normal), desc="extract-images")
        if finished and long_jobs:
            log.info("extracting %d long-output images one at a time", len(long_jobs))
            finished = runner.run_jobs(long_jobs, lambda j: work(j, cfg.vision_max_tokens_tall), on_result,
                                       on_error, 1, total=len(long_jobs), desc="extract-images (long)")
        stats.interrupted = not finished
    finally:
        for client in clients:
            client.close()
    return stats
