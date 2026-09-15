"""Phase 5: PDF pages as searchable docs; scanned pages through the vision model."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from re2 import db, extract_images, runner, thumbs, vlm
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

MAX_PAGES = 300
MIN_TEXT_CHARS = 50
RENDER_DPI = 150
OCR_STAGE = "pdf_ocr"   # failure.doc_id holds the page number for this stage
TEXT_STAGE = "pdf"
PAGE_MODEL = "pymupdf-text"
OCR_CONCURRENCY = 2      # page transcripts are long; two requests fit the vision model's shared context
OCR_MAX_TOKENS = 2800


def _first_readable(conn: sqlite3.Connection, media_id: int) -> str | None:
    row = conn.execute("SELECT rel_path FROM media_path WHERE media_id=? AND readable=1 ORDER BY id LIMIT 1",
                       (media_id,)).fetchone()
    return row[0] if row else None


@dataclass
class PdfText:
    media_id: int
    n_pages: int
    text_pages: list[tuple[int, str]]
    scanned_pages: list[int]
    thumb: str | None


def read_pdf(cfg, media_id: int, sha256: str | None, path: Path, needs_thumb: bool) -> PdfText:
    text_pages, scanned = [], []
    thumb = None
    with pymupdf.open(path) as doc:
        n = doc.page_count
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_CHARS:
                text_pages.append((i + 1, text))
            else:
                scanned.append(i + 1)
        if needs_thumb and sha256 and n:
            pix = doc[0].get_pixmap(dpi=72)
            rel = thumbs.thumb_rel_path(cfg, sha256)
            tmp = (REPO_ROOT / rel).with_suffix(".page1.png")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(tmp))
            try:
                thumbs.make_thumbnail(tmp, REPO_ROOT / rel)
                thumb = str(rel)
            finally:
                tmp.unlink(missing_ok=True)
    return PdfText(media_id, n, text_pages, scanned, thumb)


def store_text(conn: sqlite3.Connection, r: PdfText) -> None:
    with conn:
        conn.executemany(
            """INSERT INTO doc(media_id, kind, seq, page, text, model) VALUES (?, 'pdf_page', ?, ?, ?, ?)
               ON CONFLICT(media_id, kind, seq) DO UPDATE SET text=excluded.text, model=excluded.model""",
            [(r.media_id, page, page, text, PAGE_MODEL) for page, text in r.text_pages])
        conn.executemany("INSERT OR IGNORE INTO pdf_ocr_queue(media_id, page) VALUES (?, ?)",
                         [(r.media_id, page) for page in r.scanned_pages])
        if r.thumb:
            conn.execute("UPDATE media SET thumb_path=COALESCE(thumb_path, ?) WHERE id=?", (r.thumb, r.media_id))
        # A PDF with no text layer at all still needs a marker so it is not re-read on every run.
        conn.execute("INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (f"pdf_read:{r.media_id}", json.dumps({"pages": r.n_pages, "scanned": len(r.scanned_pages)})))
        db.clear_failure(conn, TEXT_STAGE, r.media_id)


@dataclass
class PdfStats:
    selected: int = 0
    done: int = 0
    text_pages: int = 0
    queued: int = 0
    failed: int = 0

    def summary(self) -> str:
        return (f"selected={self.selected} done={self.done} text_pages={self.text_pages} "
                f"queued_for_ocr={self.queued} failed={self.failed}")


def run_text(conn: sqlite3.Connection, cfg, limit: int | None = None) -> PdfStats:
    skip = db.pending_failures(conn, TEXT_STAGE)
    rows = conn.execute(
        """SELECT m.id, m.sha256, m.thumb_path FROM media m
           WHERE m.kind='pdf' AND m.status='READY'
             AND NOT EXISTS (SELECT 1 FROM kv WHERE kv.key='pdf_read:' || m.id)
           ORDER BY m.id""").fetchall()
    jobs = [r for r in rows if r["id"] not in skip][: limit if limit is not None else None]
    stats = PdfStats(selected=len(jobs))
    for row in jobs:
        rel = _first_readable(conn, row["id"])
        try:
            if rel is None:
                raise FileNotFoundError("no readable path")
            result = read_pdf(cfg, row["id"], row["sha256"], Path(cfg.corpus_root) / rel, row["thumb_path"] is None)
        except Exception as exc:  # noqa: BLE001 - PyMuPDF raises many types on broken files
            stats.failed += 1
            log.warning("pdf %s (%s) failed: %s", row["id"], rel, exc)
            with conn:
                db.upsert_failure(conn, TEXT_STAGE, row["id"], None, f"{type(exc).__name__}: {exc}"[:2000])
            continue
        store_text(conn, result)
        stats.done += 1
        stats.text_pages += len(result.text_pages)
        stats.queued += len(result.scanned_pages)
    return stats


# ── Scanned pages ─────────────────────────────────────────────────────────

def render_page(path: Path, page: int) -> bytes:
    with pymupdf.open(path) as doc:
        return doc[page - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")


def page_text(d: dict) -> str:
    ocr = (d.get("ocr_text") or "").strip()
    return ocr if ocr else (d.get("description") or "").strip()


@dataclass
class PageJob:
    media_id: int
    page: int
    n_pages: int
    path: Path
    long_output: bool = False


@dataclass
class OcrStats:
    selected: int = 0
    done: int = 0
    failed: int = 0
    retried_long: int = 0
    interrupted: bool = False
    started: float = 0.0

    def summary(self) -> str:
        rate = self.done * 60 / max(time.monotonic() - self.started, 1e-9)
        return (f"selected={self.selected} done={self.done} failed={self.failed} retried_long={self.retried_long} "
                f"rate={rate:.1f}/min" + (" INTERRUPTED" if self.interrupted else ""))


def run_ocr(conn: sqlite3.Connection, cfg, limit: int | None = None, client_factory=httpx.Client) -> OcrStats:
    failed_pages = {(r[0], r[1]) for r in conn.execute(
        "SELECT media_id, doc_id FROM failure WHERE stage=? AND attempts>=?", (OCR_STAGE, db.MAX_ATTEMPTS))}
    ran_out = {(r[0], r[1]) for r in conn.execute(
        "SELECT media_id, doc_id FROM failure WHERE stage=? AND error LIKE '%finish_reason=length%'", (OCR_STAGE,))}
    pages: dict[int, int] = {}
    jobs: list[PageJob] = []
    for media_id, page in conn.execute("SELECT media_id, page FROM pdf_ocr_queue ORDER BY media_id, page"):
        if (media_id, page) in failed_pages:
            continue
        rel = _first_readable(conn, media_id)
        if rel is None:
            continue
        if media_id not in pages:
            info = db.kv_get(conn, f"pdf_read:{media_id}")
            pages[media_id] = json.loads(info)["pages"] if info else 0
        jobs.append(PageJob(media_id, page, pages[media_id], Path(cfg.corpus_root) / rel,
                            (media_id, page) in ran_out))
        if limit is not None and len(jobs) >= limit:
            break
    stats = OcrStats(selected=len(jobs), started=time.monotonic())
    local = threading.local()
    clients: list[httpx.Client] = []
    lock = threading.Lock()

    def work(job: PageJob, max_tokens: int = OCR_MAX_TOKENS) -> dict:
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        images = vlm.prepare_images(render_page(job.path, job.page), cfg.vision_max_side, cfg.vision_max_tiles,
                                    dense=job.long_output)
        text = f"Transcribe this document page (page {job.page} of {job.n_pages})." + vlm.tiles_note(len(images))
        return vlm.extract(client, cfg, images, vlm.PAGE_SCHEMA, vlm.PAGE_SYSTEM_PROMPT, text, max_tokens=max_tokens)

    def on_result(job: PageJob, d: dict) -> None:
        payload = {k: v for k, v in d.items() if not k.startswith("_")}
        with conn:
            conn.execute(
                """INSERT INTO doc(media_id, kind, seq, page, text, json, model) VALUES (?, 'pdf_page', ?, ?, ?, ?, ?)
                   ON CONFLICT(media_id, kind, seq) DO UPDATE SET text=excluded.text, json=excluded.json,
                                                                  model=excluded.model""",
                (job.media_id, job.page, job.page, page_text(d), json.dumps(payload, ensure_ascii=False),
                 cfg.vision_model))
            # Page tags are not stored: set_media_tags replaces per media, so pages would overwrite each other.
            extract_images.store_entities(conn, job.media_id, d.get("entities"))
            conn.execute("DELETE FROM pdf_ocr_queue WHERE media_id=? AND page=?", (job.media_id, job.page))
            db.clear_failure(conn, OCR_STAGE, job.media_id, job.page)
        stats.done += 1

    long_jobs: list[PageJob] = []

    def on_error(job: PageJob, exc: BaseException) -> None:
        with conn:
            db.upsert_failure(conn, OCR_STAGE, job.media_id, job.page, f"{type(exc).__name__}: {exc}"[:2000])
        if isinstance(exc, vlm.VLMLengthError) and not job.long_output:
            job.long_output = True
            long_jobs.append(job)
            stats.retried_long += 1
            return
        stats.failed += 1
        log.warning("pdf %s page %s failed: %s", job.media_id, job.page, exc)

    normal = [j for j in jobs if not j.long_output]
    long_jobs.extend(j for j in jobs if j.long_output)
    try:
        finished = runner.run_jobs(normal, work, on_result, on_error, OCR_CONCURRENCY,
                                   total=len(normal), desc="pdf-ocr")
        if finished and long_jobs:
            finished = runner.run_jobs(long_jobs, lambda j: work(j, cfg.vision_max_tokens_tall), on_result,
                                       on_error, 1, total=len(long_jobs), desc="pdf-ocr (long)")
        stats.interrupted = not finished
    finally:
        for client in clients:
            client.close()
    return stats
