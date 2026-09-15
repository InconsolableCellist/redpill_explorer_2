import json
import os
from types import SimpleNamespace

from PIL import Image

from re2 import db, legacy, scan, tags

SHA_OFFLINE = "b" * 64


def setup(tmp_path):
    root = tmp_path / "corpus"
    (root / "race").mkdir(parents=True)
    Image.new("RGB", (10, 10), "blue").save(root / "race" / "shot.png")
    os.symlink("/mnt/hanoi_data/storage/redpills/race/lost.jpg", root / "race" / "lost.jpg")

    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(
        corpus_root=root,
        legacy_dir=tmp_path / "legacy",
        thumbs_dir=tmp_path / "thumbs",
        legacy_output_json=tmp_path / "output.json",
        archive_metadata=tmp_path / "archive_metadata.json",
    )
    scan.scan(conn, cfg)
    sha_real = conn.execute("SELECT sha256 FROM media WHERE kind='image' AND sha256 IS NOT NULL").fetchone()[0]

    cfg.legacy_dir.mkdir()
    (cfg.legacy_dir / "images_captioned.json").write_text(json.dumps({
        sha_real: {"filename": "shot.png", "description": "a blue square"},
        SHA_OFFLINE: {"filename": "old.jpg", "description": "an old meme"},
    }))
    (cfg.legacy_dir / "images_captioned_tagged.json").write_text(json.dumps({
        sha_real: {"filename": "shot.png", "description": "a blue square",
                   "tags": {"USA Today": 0.5, "usa_today": 0.9, "x": 1.0}, "spicy": 0.2},
    }))
    lost_sha = "c" * 64
    cfg.legacy_output_json.write_text(json.dumps({
        lost_sha: {"filename": "lost.jpg", "path": "/mnt/hanoi_data/storage/redpills/race/lost.jpg",
                   "description": "a lost screenshot"},
    }))
    thumb = cfg.thumbs_dir / "b" / "b" / "b" / ("b" * 61 + ".jpg")
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"jpg")
    cfg.archive_metadata.write_text(json.dumps([
        {"url": "https://x.com/a/status/1", "date": "2026-01-01", "folder": "race",
         "screenshot": "shot.png", "photos": [], "videos": [], "restricted": False},
        {"url": "https://x.com/a/status/2", "date": "2026-01-02", "folder": ".",
         "screenshot": "shot.png", "photos": ["nowhere.png"], "videos": [], "restricted": False},
    ]))
    return conn, cfg, sha_real, lost_sha


def test_reconcile(tmp_path):
    conn, cfg, sha_real, lost_sha = setup(tmp_path)
    legacy.reconcile(conn, cfg)

    real = conn.execute("SELECT * FROM media WHERE uid=?", (sha_real,)).fetchone()
    assert (real["status"], real["legacy_caption"], real["legacy_spicy"]) == ("READY", "a blue square", 0.2)
    tag_rows = conn.execute(
        "SELECT t.name, mt.weight FROM media_tag mt JOIN tag t ON t.id=mt.tag_id WHERE media_id=?",
        (real["id"],)).fetchall()
    assert [tuple(r) for r in tag_rows] == [("usa today", 0.9)]

    offline = conn.execute("SELECT status, thumb_path FROM media WHERE uid=?", (SHA_OFFLINE,)).fetchone()
    assert offline["status"] == "THUMBNAIL_ONLY" and offline["thumb_path"]

    rekeyed = conn.execute(
        "SELECT m.uid, m.legacy_caption FROM media_path p JOIN media m ON m.id=p.media_id "
        "WHERE p.rel_path='race/lost.jpg'").fetchone()
    assert tuple(rekeyed) == (lost_sha, "a lost screenshot")
    assert conn.execute("SELECT count(*) FROM media WHERE uid LIKE 'missing:%'").fetchone()[0] == 0

    assert conn.execute("SELECT count(*) FROM doc WHERE kind='legacy_caption'").fetchone()[0] == 3
    # Idempotent.
    legacy.reconcile(conn, cfg)
    assert conn.execute("SELECT count(*) FROM doc WHERE kind='legacy_caption'").fetchone()[0] == 3


def test_reconcile_sources(tmp_path):
    conn, cfg, sha_real, _ = setup(tmp_path)
    stats = legacy.reconcile_sources(conn, cfg)
    assert (stats.linked, stats.by_basename, stats.not_found) == (2, 1, 1)
    urls = [r[0] for r in conn.execute("SELECT url FROM source ORDER BY url")]
    assert urls == ["https://x.com/a/status/1", "https://x.com/a/status/2"]


def test_normalize():
    assert tags.normalize("USA Today") == tags.normalize("usa_today") == "usa today"
    assert tags.normalize("  #Covid-19 ") == "covid 19"
    assert tags.normalize("Anti-Semitism") == "antisemitism" and tags.normalize("Jewish") == "jews"
    assert tags.normalize("Coronavirus") == "covid 19"
    assert tags.normalize("x") == ""


def test_renormalize_merges_synonyms():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'image', 'READY'), (2, 'b', 'image', 'READY');
        INSERT INTO tag(id, name) VALUES (1, 'jews'), (2, 'jewish'), (3, 'African American'), (4, 'x');
        INSERT INTO media_tag(media_id, tag_id, weight, origin) VALUES
          (1, 1, 0.5, 'legacy'), (1, 2, 0.9, 'legacy'), (2, 2, 1.0, 'vlm'), (2, 3, 1.0, 'vlm'), (2, 4, 1.0, 'vlm');
    """)
    from re2 import relations
    relations.ensure_tables(conn)
    conn.execute("INSERT INTO tag_relation(a, b, semantic_ab) VALUES (1, 2, 1.0)")   # a merged-away tag has scores
    conn.execute("INSERT INTO tag_relation_done(tag_id) VALUES (2)")
    assert {p[:2] for p in tags.renormalize(conn, dry_run=True)} == {
        ("jewish", "jews"), ("African American", "black people"), ("x", "")}
    changed, _ = tags.renormalize(conn)
    assert changed == 3
    assert conn.execute("SELECT count(*) FROM tag_relation").fetchone()[0] == 0
    rows = {tuple(r) for r in conn.execute(
        "SELECT mt.media_id, t.name, mt.weight, mt.origin FROM media_tag mt JOIN tag t ON t.id=mt.tag_id")}
    assert rows == {(1, "jews", 0.9, "legacy"), (2, "jews", 1.0, "vlm"), (2, "black people", 1.0, "vlm")}
