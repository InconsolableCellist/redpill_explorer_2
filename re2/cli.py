"""`re2` command-line entry point."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from re2 import config, db


def _table(title: str, rows, headers: tuple[str, ...]) -> None:
    rows = [tuple("" if v is None else str(v) for v in r) for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print(f"\n{title}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows or [("(none)",) + ("",) * (len(headers) - 1)]:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)))


def _index_size() -> int:
    from re2 import embed

    return len(embed.get_index(config.load(), "docs"))


def status_counts(conn) -> dict:
    """Progress counts shared by `re2 status` and /api/status."""
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "media": [dict(r) for r in conn.execute(
            "SELECT status, kind, COUNT(*) AS n FROM media GROUP BY 1, 2 ORDER BY 1, 2")],
        "paths": [dict(r) for r in conn.execute(
            "SELECT readable, is_symlink, COUNT(*) AS n FROM media_path GROUP BY 1, 2 ORDER BY 1, 2")],
        "docs": [dict(r) for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM doc GROUP BY 1 ORDER BY 1")],
        "failures": [dict(r) for r in conn.execute(
            "SELECT stage, COUNT(*) AS n, SUM(attempts >= 3) AS gave_up FROM failure GROUP BY 1 ORDER BY 1")],
        "tags": one("SELECT COUNT(*) FROM tag"),
        "media_tags": [dict(r) for r in conn.execute(
            "SELECT origin, COUNT(DISTINCT media_id) AS media, COUNT(*) AS n FROM media_tag GROUP BY 1")],
        "entities": one("SELECT COUNT(*) FROM entity"),
        "sources": one("SELECT COUNT(*) FROM source"),
        "source_media": one("SELECT COUNT(DISTINCT media_id) FROM source"),
        "legacy_captions": one("SELECT COUNT(*) FROM media WHERE legacy_caption IS NOT NULL"),
        "thumbnails": one("SELECT COUNT(*) FROM media WHERE thumb_path IS NOT NULL"),
        "embedding_index": _index_size(),
        "last_scan_at": db.kv_get(conn, "last_scan_at"),
    }


def cmd_status(cfg: config.Config, args: argparse.Namespace) -> int:
    for field in dataclasses.fields(cfg):
        print(f"{field.name:20} {getattr(cfg, field.name)}")
    print(f"{'corpus_root exists':20} {cfg.corpus_root.is_dir()}")
    print(f"{'db exists':20} {cfg.db_path.exists()}")
    if not cfg.db_path.exists():
        return 0

    s = status_counts(db.open_db(cfg))
    _table("Media by status and kind", [tuple(r.values()) for r in s["media"]], ("status", "kind", "count"))
    _table("Paths", [tuple(r.values()) for r in s["paths"]], ("readable", "symlink", "count"))
    _table("Docs by kind", [tuple(r.values()) for r in s["docs"]], ("kind", "count"))
    _table("Tags by origin", [tuple(r.values()) for r in s["media_tags"]], ("origin", "media", "rows"))
    _table("Failures by stage", [tuple(r.values()) for r in s["failures"]], ("stage", "count", "gave_up"))
    _table("Totals", [
        ("distinct tags", s["tags"]),
        ("entities", s["entities"]),
        ("source rows", s["sources"]),
        ("media with source", s["source_media"]),
        ("legacy captions", s["legacy_captions"]),
        ("thumbnails", s["thumbnails"]),
        ("embedding index", s["embedding_index"]),
        ("last scan", s["last_scan_at"]),
    ], ("item", "value"))
    return 0


def cmd_scan(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import scan

    conn = db.open_db(cfg)
    stats = scan.scan(conn, cfg, limit=args.limit, rehash=args.rehash)
    print(f"scan: {stats.summary()}")
    return 0


def cmd_reconcile_legacy(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import legacy

    stats = legacy.reconcile(db.open_db(cfg), cfg)
    print(f"reconcile-legacy: {stats.summary()}")
    return 0


def cmd_reconcile_sources(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import legacy

    stats = legacy.reconcile_sources(db.open_db(cfg), cfg)
    print(f"reconcile-sources: {stats.summary()}")
    return 0


def cmd_thumbs(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import thumbs

    stats = thumbs.run(db.open_db(cfg), cfg, limit=args.limit)
    print(f"thumbs: {stats.summary()}")
    return 0


def cmd_extract_images(cfg: config.Config, args: argparse.Namespace) -> int:
    import dataclasses

    from re2 import extract_images

    if args.server or args.model:
        # Another GPU host alongside the configured one; runs claim images so they never read the same one.
        cfg = dataclasses.replace(cfg, base_url=(args.server or cfg.base_url).rstrip("/"),
                                  vision_model=args.model or cfg.vision_model)

    if args.check:
        sized = extract_images.sized_for_server(cfg, requested=args.concurrency or cfg.vision_concurrency)
        print(f"extract-images --check: model {cfg.vision_model} at {cfg.base_url}: concurrency "
              f"{sized.vision_concurrency}, long retries up to {sized.vision_max_tokens_tall} tokens")
        return 0

    stats = extract_images.run(db.open_db(cfg), cfg, limit=args.limit,
                               include_thumbnail_only=not args.no_thumbnail_only, concurrency=args.concurrency or 0)
    print(f"extract-images: {stats.summary()}")
    return 0


def cmd_embed(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import embed

    embed.ensure_local_server(cfg)
    stats = embed.build_docs(db.open_db(cfg), cfg, limit=args.limit, rebuild=args.rebuild)
    print(f"embed: {stats.summary()}")
    return 0


def cmd_serve(cfg: config.Config, args: argparse.Namespace) -> int:
    import uvicorn

    from re2 import api, embed

    db.open_db(cfg).close()  # create or migrate the schema before read-only connections open it
    embed.ensure_local_server(cfg)
    # Ask streams stay open while a tab watches a run, so don't wait on them at shutdown; runs don't survive it anyway.
    uvicorn.run(api.create_app(cfg), host=args.host or cfg.server_host, port=args.port or cfg.server_port,
                log_level="info", timeout_graceful_shutdown=3)
    return 0


def cmd_dates(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import dates

    counts = dates.run(db.open_db(cfg))
    print("dates: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def cmd_video_probe(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import video

    stats = video.run(db.open_db(cfg), cfg, limit=args.limit)
    print(f"video-probe: {stats.summary()}")
    return 0


def cmd_asr(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import asr

    stats = asr.run(db.open_db(cfg), cfg, limit=args.limit)
    print(f"asr: {stats.summary()}")
    return 0


def cmd_pdfs(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import pdf

    stats = pdf.run_text(db.open_db(cfg), cfg, limit=args.limit)
    print(f"pdfs: {stats.summary()}")
    return 0


def cmd_pdf_ocr(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import pdf

    stats = pdf.run_ocr(db.open_db(cfg), cfg, limit=args.limit)
    print(f"pdf-ocr: {stats.summary()}")
    return 0


def cmd_review(cfg: config.Config, args: argparse.Namespace) -> int:
    """Print the most recent vision-model captions: title, summary, and the start of the description."""
    import json
    import textwrap

    conn = db.open_db(cfg)
    rows = conn.execute(
        """SELECT d.media_id, d.created_at, d.json,
                  (SELECT rel_path FROM media_path p WHERE p.media_id=d.media_id ORDER BY p.id LIMIT 1)
           FROM doc d WHERE d.kind='caption' ORDER BY d.created_at DESC, d.id DESC LIMIT ?""", (args.n,)).fetchall()
    for media_id, created, raw, rel_path in rows:
        d = json.loads(raw or "{}")
        print(f"\n#{media_id}  {created}  {rel_path}")
        print(f"  link     http://{cfg.server_host}:{cfg.server_port}/#/item/{media_id}")
        print(f"  title    {d.get('title') or '(none: extracted before titles existed)'}")
        for label, text in (("summary", d.get("summary")), ("meaning", d.get("meaning")),
                            ("describe", (d.get("description") or "")[:300])):
            if text:
                print(textwrap.fill(text, width=110, initial_indent=f"  {label:8} ", subsequent_indent=" " * 11))
    return 0


COVERAGE_QUESTIONS = [
    "What does the archive say about COVID-19 vaccines?",
    "What does the archive contain about Israel and Gaza?",
    "What evidence does the archive hold about fraud in the 2020 US election?",
    "What does the archive say about Justin Trudeau?",
]


def cmd_ask_coverage(cfg: config.Config, args: argparse.Namespace) -> int:
    """Ask broad questions and report how much of what matched each answer actually read and cited."""
    import time

    from re2 import ask

    conn = db.open_db(cfg)
    questions = args.questions or COVERAGE_QUESTIONS
    print(f"{'question':52} {'cands':>6} {'kw':>5} {'read':>5} {'%':>4} {'cited':>5} {'batch':>5} {'secs':>5}")
    for q in questions:
        started, cov, cited, error = time.monotonic(), {}, 0, ""
        for name, data in ask.run(conn, cfg, q, depth=args.depth, exhaustive=args.exhaustive):
            if name == "coverage":
                cov = data
            elif name == "cite":
                cited += 1
            elif name == "error":
                error = data["msg"]
        if error:
            print(f"{q[:52]:52} error: {error}")
            continue
        pct = round(100 * cov.get("read", 0) / cov["candidates"]) if cov.get("candidates") else 0
        print(f"{q[:52]:52} {cov.get('candidates', 0):6} {cov.get('keyword', 0):5} {cov.get('read', 0):5} {pct:4} "
              f"{cited:5} {cov.get('batches', '-'):>5} {time.monotonic() - started:5.0f}")
    print("cands: items the research's phrasings matched by keyword plus the closest in meaning; kw: keyword matches;"
          " read: items whose records reached the answer's model. Answers are logged as Ask conversations.")
    return 0


def cmd_recluster(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import cluster

    stats = cluster.recluster(db.open_db(cfg), cfg)
    print(f"recluster: {stats.summary()}")
    return 0


def cmd_merge_entities(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import entities

    stats = entities.merge(db.open_db(cfg), dry_run=args.dry_run)
    for line in stats.examples:
        print(f"  {line}")
    print(f"merge-entities{' (dry run)' if args.dry_run else ''}: {stats.summary()}")
    return 0


def cmd_normalize_tags(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import tags

    if args.dry_run:
        plan = tags.renormalize(db.open_db(cfg), dry_run=True)
        for old, new, n in plan:
            print(f"  {old!r} -> {new!r}  ({n} items)")
        print(f"normalize-tags (dry run): {len(plan)} tags would be renamed or merged")
        return 0
    changed, moved = tags.renormalize(db.open_db(cfg))
    print(f"normalize-tags: renamed_or_merged={changed} rows_moved={moved}")
    return 0


def cmd_tag_relations(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import relations

    stats = relations.run(db.open_db(cfg), cfg, limit=args.limit, observed_only=args.observed_only,
                          concurrency=args.concurrency, rescore=args.rescore)
    print(f"tag-relations: {stats.summary()}")
    return 0


def cmd_digests(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import digest

    stats = digest.run(db.open_db(cfg), cfg, limit=args.limit, concurrency=args.concurrency)
    print(f"digests: {stats.summary()}")
    return 0


def cmd_wiki_update(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import wiki

    stats = wiki.update(db.open_db(cfg), cfg)
    print(f"wiki-update: {stats.summary()}")
    return 0


def cmd_wiki_events(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import wiki

    stats = wiki.build(db.open_db(cfg), cfg, kinds=["event"], topics=args.topic, limit=args.limit, write=False)
    print(f"wiki-events: {stats.summary()}")
    return 0


def cmd_wiki_build(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import wiki

    stats = wiki.build(db.open_db(cfg), cfg, kinds=args.kind, limit=args.limit, topics=args.topic)
    for slug in stats.written:
        print(f"  wrote {slug}")
    print(f"wiki-build: {stats.summary()}")
    return 0


def cmd_wiki_export(cfg: config.Config, args: argparse.Namespace) -> int:
    from pathlib import Path

    from re2 import wiki

    out = Path(args.out) if args.out else cfg.data_dir / "wiki"
    n = wiki.export(db.open_db(cfg), cfg, out)
    print(f"wiki-export: {n} pages to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="re2")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show config and progress").set_defaults(func=cmd_status)

    p = sub.add_parser("scan", help="build the read-only manifest of the corpus")
    p.add_argument("--limit", type=int, help="hash at most N new or changed files")
    p.add_argument("--rehash", action="store_true", help="re-hash files even if size/mtime match")
    p.set_defaults(func=cmd_scan)

    sub.add_parser("reconcile-legacy", help="import legacy captions, tags, thumbnails, output.json") \
        .set_defaults(func=cmd_reconcile_legacy)
    sub.add_parser("reconcile-sources", help="attach archive_metadata.json URLs to media") \
        .set_defaults(func=cmd_reconcile_sources)

    p = sub.add_parser("thumbs", help="make 400-px thumbnails for readable images")
    p.add_argument("--limit", type=int, help="process at most N images")
    p.set_defaults(func=cmd_thumbs)

    p = sub.add_parser("extract-images", help="caption, OCR and tag images with the vision model on cube")
    p.add_argument("--limit", type=int, help="process at most N images")
    p.add_argument("--no-thumbnail-only", action="store_true", help="skip items that only have a thumbnail")
    p.add_argument("--concurrency", type=int,
                   help="requests at once (default: sized from the server's slots and context)")
    p.add_argument("--check", action="store_true", help="print what the server allows and exit")
    p.add_argument("--server", help="OpenAI-compatible base URL of another vision server (default [llm] base_url)")
    p.add_argument("--model", help="vision model name on that server (default [llm] vision_model)")
    p.set_defaults(func=cmd_extract_images)

    p = sub.add_parser("embed", help="add new docs to the dense embedding index")
    p.add_argument("--limit", type=int, help="embed at most N docs")
    p.add_argument("--rebuild", action="store_true", help="drop the index and embed everything again")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("video-probe", help="ffprobe videos, extract keyframes and thumbnails (local CPU)")
    p.add_argument("--limit", type=int, help="process at most N videos")
    p.set_defaults(func=cmd_video_probe)

    p = sub.add_parser("asr", help="transcribe videos with faster-whisper on the local GPU")
    p.add_argument("--limit", type=int, help="transcribe at most N videos")
    p.set_defaults(func=cmd_asr)

    p = sub.add_parser("pdfs", help="index PDF text layers and queue scanned pages (local)")
    p.add_argument("--limit", type=int, help="read at most N PDFs")
    p.set_defaults(func=cmd_pdfs)

    p = sub.add_parser("pdf-ocr", help="transcribe queued scanned PDF pages with the vision model on cube")
    p.add_argument("--limit", type=int, help="transcribe at most N pages")
    p.set_defaults(func=cmd_pdf_ocr)

    p = sub.add_parser("review", help="print the most recent vision-model titles and summaries")
    p.add_argument("-n", type=int, default=10, help="how many (default 10)")
    p.set_defaults(func=cmd_review)

    sub.add_parser("recluster", help="topics, themes and the 2D map from item vectors (local CPU)") \
        .set_defaults(func=cmd_recluster)

    p = sub.add_parser("merge-entities", help="merge entity names that are obviously the same")
    p.add_argument("--dry-run", action="store_true", help="print the merges without applying them")
    p.set_defaults(func=cmd_merge_entities)

    p = sub.add_parser("normalize-tags", help="re-apply tag normalization and tag_synonyms.toml to stored tags")
    p.add_argument("--dry-run", action="store_true", help="print the renames and merges without applying them")
    p.set_defaults(func=cmd_normalize_tags)

    p = sub.add_parser("tag-relations", help="observed and semantic relatedness between the most used tags")
    p.add_argument("--limit", type=int, help="score at most N anchor tags with the model")
    p.add_argument("--observed-only", action="store_true", help="only recompute co-occurrence (no model)")
    p.add_argument("--concurrency", type=int, default=4, help="requests at once (default 4; match the server's slots)")
    p.add_argument("--rescore", action="store_true", help="forget earlier model scores and ask about every tag again")
    p.set_defaults(func=cmd_tag_relations)

    p = sub.add_parser("digests", help="a title, summary and event hint per item ([wiki] text model)")
    p.add_argument("--limit", type=int, help="send at most N items to the model")
    p.add_argument("--concurrency", type=int, help="requests at once (default [wiki] concurrency)")
    p.set_defaults(func=cmd_digests)

    sub.add_parser("wiki-update", help="sync wiki pages with topics and entities, attach new items, mark stale "
                                       "pages (no model)").set_defaults(func=cmd_wiki_update)

    p = sub.add_parser("wiki-events", help="group topics' items into event pages (no page writing)")
    p.add_argument("--topic", type=int, action="append", help="a topic's cluster id (repeatable)")
    p.add_argument("--limit", type=int, help="at most N topics, largest first")
    p.set_defaults(func=cmd_wiki_events)

    p = sub.add_parser("wiki-build", help="update, group topics into events, write stale wiki pages")
    p.add_argument("--kind", action="append", choices=["event", "topic", "theme", "entity", "overview"],
                   help="only these kinds (repeatable)")
    p.add_argument("--topic", type=int, action="append",
                   help="only this topic (cluster id), its events and its page (repeatable)")
    p.add_argument("--limit", type=int, help="at most N topics end to end, and N pages of each other kind")
    p.set_defaults(func=cmd_wiki_build)

    p = sub.add_parser("wiki-export", help="write the wiki as Obsidian-style Markdown")
    p.add_argument("--out", help="directory (default data/wiki)")
    p.set_defaults(func=cmd_wiki_export)

    sub.add_parser("dates", help="recompute item dates from sources, filenames, text and mtimes") \
        .set_defaults(func=cmd_dates)

    p = sub.add_parser("ask-coverage", help="ask broad questions; report candidates, items read and cited")
    p.add_argument("questions", nargs="*", help="questions (default: a few broad ones)")
    p.add_argument("--exhaustive", action="store_true", help="read every candidate in batches")
    p.add_argument("--depth", type=float, default=None, help="research depth 0..1 (default 0.35)")
    p.set_defaults(func=cmd_ask_coverage)

    p = sub.add_parser("serve", help="run the dashboard and API server")
    p.add_argument("--host", help="bind address (default from config.toml)")
    p.add_argument("--port", type=int, help="port (default from config.toml)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = config.load(args.config)
    if args.command != "status":
        log_dir = cfg.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / f"{args.command}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info("start: re2 %s", " ".join(sys.argv[1:]))
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
