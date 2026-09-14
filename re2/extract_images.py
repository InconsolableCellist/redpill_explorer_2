"""Phase 2: run the vision model over every image and store docs, tags, entities."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

from re2 import db, tags, vlm
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

STAGE = "vlm_image"
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
    return (
        f"{d['description']}\n\n"
        f"Subjects: {', '.join(d.get('subjects') or [])}\n"
        f"Actions: {d.get('actions', '')}\n"
        f"Meaning: {d.get('meaning', '')}\n"
        f"Claims: {'; '.join(d.get('claims') or [])}\n"
        f"Intent: {d.get('intent', '')}\n"
        f"Humor: {d.get('humor', '')}"
    )


def store_entities(conn: sqlite3.Connection, media_id: int, entities: dict) -> None:
    for key, etype in ENTITY_TYPES.items():
        for name in (entities or {}).get(key) or []:
            name = " ".join(str(name).split())
            if not name:
                continue
            conn.execute("INSERT OR IGNORE INTO entity(name, type) VALUES (?, ?)", (name, etype))
            entity_id = conn.execute("SELECT id FROM entity WHERE name=? AND type=?", (name, etype)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO media_entity(media_id, entity_id) VALUES (?, ?)",
                         (media_id, entity_id))


def store_image_result(conn: sqlite3.Connection, cfg, media_id: int, d: dict, low_res: bool) -> None:
    """Write one extraction in a single transaction."""
    payload = {k: v for k, v in d.items() if not k.startswith("_")}
    with conn:
        conn.execute(
            """INSERT INTO doc(media_id, kind, seq, text, json, model) VALUES (?, 'caption', 0, ?, ?, ?)
               ON CONFLICT(media_id, kind, seq) DO UPDATE SET
                 text=excluded.text, json=excluded.json, model=excluded.model""",
            (media_id, caption_text(d), json.dumps(payload, ensure_ascii=False), cfg.vision_model),
        )
        ocr = (d.get("ocr_text") or "").strip()
        if ocr:
            conn.execute(
                """INSERT INTO doc(media_id, kind, seq, text, model) VALUES (?, 'ocr', 0, ?, ?)
                   ON CONFLICT(media_id, kind, seq) DO UPDATE SET text=excluded.text, model=excluded.model""",
                (media_id, ocr, cfg.vision_model),
            )
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


@dataclass
class Job:
    media_id: int
    path: Path
    low_res: bool


@dataclass
class ExtractStats:
    selected: int = 0
    done: int = 0
    failed: int = 0
    completion_tokens: int = 0
    started: float = 0.0

    def rate_per_min(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.done * 60 / elapsed if elapsed > 0 else 0.0

    def summary(self) -> str:
        avg = self.completion_tokens / self.done if self.done else 0
        return (f"selected={self.selected} done={self.done} failed={self.failed} "
                f"rate={self.rate_per_min():.1f}/min avg_completion_tokens={avg:.0f}")


def select_jobs(conn: sqlite3.Connection, cfg, limit: int | None, include_thumbnail_only: bool) -> list[Job]:
    statuses = ("READY", "THUMBNAIL_ONLY") if include_thumbnail_only else ("READY",)
    skip = db.pending_failures(conn, STAGE)
    rows = conn.execute(
        f"""SELECT m.id, m.status, m.width, m.height, m.thumb_path,
                   (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                    ORDER BY p.id LIMIT 1) AS rel_path
            FROM media m
            WHERE m.kind='image' AND m.status IN ({','.join('?' * len(statuses))})
              AND NOT EXISTS (SELECT 1 FROM doc d WHERE d.media_id=m.id AND d.kind='caption')
            ORDER BY m.status='THUMBNAIL_ONLY', m.id""",
        statuses,
    ).fetchall()
    jobs = []
    for r in rows:
        if r["id"] in skip:
            continue
        if r["status"] == "READY" and r["rel_path"]:
            small = r["width"] is not None and max(r["width"], r["height"] or 0) < LOW_RES_SIDE
            jobs.append(Job(r["id"], Path(cfg.corpus_root) / r["rel_path"], small))
        elif r["thumb_path"]:
            jobs.append(Job(r["id"], REPO_ROOT / r["thumb_path"], True))
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, include_thumbnail_only: bool = True,
        client_factory=httpx.Client) -> ExtractStats:
    jobs = select_jobs(conn, cfg, limit, include_thumbnail_only)
    stats = ExtractStats(selected=len(jobs), started=time.monotonic())
    local = threading.local()
    clients: list[httpx.Client] = []
    lock = threading.Lock()

    def work(job: Job) -> dict:
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        image = vlm.prepare_image(job.path, cfg.vision_max_side)
        text = USER_TEXT + (LOW_RES_TEXT if job.low_res else "")
        return vlm.extract(client, cfg, image, vlm.IMAGE_SCHEMA, vlm.IMAGE_SYSTEM_PROMPT, text)

    try:
        with ThreadPoolExecutor(cfg.vision_concurrency) as pool:
            futures = {pool.submit(work, job): job for job in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="extract-images", unit="img"):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 - VLMError, PIL and OS errors all count as failures
                    stats.failed += 1
                    log.warning("media %s failed: %s", job.media_id, exc)
                    with conn:
                        db.upsert_failure(conn, STAGE, job.media_id, None, f"{type(exc).__name__}: {exc}"[:2000])
                    continue
                store_image_result(conn, cfg, job.media_id, result, job.low_res)
                stats.done += 1
                stats.completion_tokens += int((result.get("_usage") or {}).get("completion_tokens") or 0)
                if stats.done % 50 == 0:
                    log.info("progress: %s", stats.summary())
    finally:
        for client in clients:
            client.close()
    return stats
