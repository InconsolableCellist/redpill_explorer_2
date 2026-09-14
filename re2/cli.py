"""`re2` command-line entry point."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from re2 import config, db


def cmd_status(cfg: config.Config, args: argparse.Namespace) -> int:
    for field in dataclasses.fields(cfg):
        print(f"{field.name:20} {getattr(cfg, field.name)}")
    print(f"{'corpus_root exists':20} {cfg.corpus_root.is_dir()}")
    print(f"{'db exists':20} {cfg.db_path.exists()}")
    return 0


def cmd_scan(cfg: config.Config, args: argparse.Namespace) -> int:
    from re2 import scan

    conn = db.open_db(cfg)
    stats = scan.scan(conn, cfg, limit=args.limit, rehash=args.rehash)
    print(f"scan: {stats.summary()}")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = config.load(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
