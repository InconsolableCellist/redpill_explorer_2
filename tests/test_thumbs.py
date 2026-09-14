from types import SimpleNamespace

from PIL import Image

from re2 import db, scan, thumbs


def test_thumbnails(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    Image.new("RGB", (1000, 500), "green").save(root / "wide.png")
    Image.new("RGBA", (100, 80), (0, 0, 255, 128)).save(root / "small.png")
    frames = [Image.new("P", (600, 600), i) for i in range(3)]
    frames[0].save(root / "anim.gif", save_all=True, append_images=frames[1:])
    (root / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")

    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, thumbs_dir=tmp_path / "thumbs")
    scan.scan(conn, cfg)
    stats = thumbs.run(conn, cfg)

    assert (stats.made, stats.failed) == (3, 1)
    sizes = {}
    for r in conn.execute("SELECT m.width, m.thumb_path FROM media m WHERE thumb_path IS NOT NULL"):
        with Image.open(r["thumb_path"]) as im:
            sizes[r["width"]] = im.size
    assert sizes == {1000: (400, 200), 100: (100, 80), 600: (400, 400)}
    assert conn.execute("SELECT stage, attempts FROM failure").fetchall()[0][:] == ("thumb", 1)

    # Re-run touches only the failed item, and gives up after three attempts.
    thumbs.run(conn, cfg)
    thumbs.run(conn, cfg)
    assert db.pending_failures(conn, "thumb")
    assert thumbs.run(conn, cfg).selected == 0
