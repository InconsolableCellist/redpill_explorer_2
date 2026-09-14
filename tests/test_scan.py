import os
from types import SimpleNamespace

from PIL import Image

from re2 import db, scan


def make_corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / ".Trash-1000" / "files").mkdir(parents=True)
    Image.new("RGB", (40, 30), "red").save(root / "a.png")
    (root / "sub" / "copy of a.png").write_bytes((root / "a.png").read_bytes())
    os.symlink("/mnt/hanoi_data/storage/redpills/gone.mp4", root / "gone.mp4")
    (root / ".Trash-1000" / "files" / "deleted.png").write_bytes((root / "a.png").read_bytes())
    (root / "cookies.txt").write_text("# Netscape HTTP Cookie File\n")
    return root


def run_scan(root, **kw):
    conn = db.connect(":memory:")
    db.init_schema(conn)
    stats = scan.scan(conn, SimpleNamespace(corpus_root=root), **kw)
    return conn, stats


def test_scan_manifest(tmp_path):
    root = make_corpus(tmp_path)
    conn, stats = run_scan(root)

    images = conn.execute("SELECT id, status, width, height FROM media WHERE kind='image'").fetchall()
    assert len(images) == 1
    assert tuple(images[0])[1:] == ("READY", 40, 30)
    paths = conn.execute("SELECT rel_path FROM media_path WHERE media_id=? ORDER BY rel_path",
                         (images[0]["id"],)).fetchall()
    assert [p[0] for p in paths] == ["a.png", "sub/copy of a.png"]

    offline = conn.execute("SELECT kind FROM media WHERE status='SOURCE_OFFLINE'").fetchall()
    assert [r[0] for r in offline] == ["video"]
    assert conn.execute(
        "SELECT count(*) FROM media_path WHERE is_symlink=1 AND readable=0").fetchone()[0] == 1

    assert conn.execute(
        "SELECT count(*) FROM media_path WHERE rel_path LIKE '%.Trash-1000%'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM media WHERE status='EXCLUDED'").fetchone()[0] == 1
    assert stats.hashed == 2


def test_rescan_is_incremental_and_tracks_removal(tmp_path):
    root = make_corpus(tmp_path)
    conn, _ = run_scan(root)
    stats = scan.scan(conn, SimpleNamespace(corpus_root=root))
    assert stats.hashed == 0
    assert stats.unchanged == 2

    (root / "a.png").unlink()
    (root / "sub" / "copy of a.png").unlink()
    scan.scan(conn, SimpleNamespace(corpus_root=root))
    assert conn.execute("SELECT status FROM media WHERE kind='image'").fetchone()[0] == "SOURCE_OFFLINE"


def test_broken_symlink_replaced_by_real_file(tmp_path):
    root = make_corpus(tmp_path)
    conn, _ = run_scan(root)
    (root / "gone.mp4").unlink()
    (root / "gone.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    scan.scan(conn, SimpleNamespace(corpus_root=root))
    assert conn.execute("SELECT count(*) FROM media WHERE uid LIKE 'missing:%'").fetchone()[0] == 0
    row = conn.execute(
        "SELECT m.status, p.is_symlink FROM media_path p JOIN media m ON m.id=p.media_id "
        "WHERE p.rel_path='gone.mp4'").fetchone()
    assert tuple(row) == ("READY", 0)


def test_limit_defers_hashing(tmp_path):
    root = make_corpus(tmp_path)
    conn, stats = run_scan(root, limit=1)
    assert stats.hashed == 1
    assert stats.deferred == 1


def test_exclusion_rules():
    assert scan.is_excluded("cookies.txt")
    assert scan.is_excluded("youtube_cookies")
    assert scan.is_excluded(".netrc")
    assert scan.is_excluded("download.part")
    assert not scan.is_excluded("tokenism_meme.png")
    assert not scan.is_excluded("photo.jpg")
