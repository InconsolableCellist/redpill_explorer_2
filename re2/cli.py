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
    from re2 import extract_images

    stats = extract_images.run(db.open_db(cfg), cfg, limit=args.limit,
                               include_thumbnail_only=not args.no_thumbnail_only)
    print(f"extract-images: {stats.summary()}")
    return 0


def cmd_embed(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import embed

    stats = embed.build_docs(db.open_db(cfg), cfg, limit=args.limit, rebuild=args.rebuild)
    print(f"embed: {stats.summary()}")
    return 0


def cmd_serve(cfg: config.Config, args: argparse.Namespace) -> int:
    import uvicorn

    from re2 import api

    db.open_db(cfg).close()  # create or migrate the schema before read-only connections open it
    uvicorn.run(api.create_app(cfg), host=args.host or cfg.server_host, port=args.port or cfg.server_port,
                log_level="info")
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
    p.set_defaults(func=cmd_extract_images)

    p = sub.add_parser("embed", help="add new docs to the dense embedding index")
    p.add_argument("--limit", type=int, help="embed at most N docs")
    p.add_argument("--rebuild", action="store_true", help="drop the index and embed everything again")
    p.set_defaults(func=cmd_embed)

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
