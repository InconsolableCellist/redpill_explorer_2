import dataclasses
import json
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from re2 import api, config, db, scan


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "corpus"
    (root / "ukraine").mkdir(parents=True)
    Image.new("RGB", (64, 48), "red").save(root / "ukraine" / "shot.png")
    (root / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 40)
    secret = tmp_path / "outside.txt"
    secret.write_text("do not serve")
    os.symlink(secret, root / "sneaky.png")
    (root / "cookies.txt").write_text("secret")

    cfg = dataclasses.replace(
        config.load(), corpus_root=root, data_dir=tmp_path / "data", thumbs_dir=tmp_path / "thumbs",
        embed_url="http://127.0.0.1:9/v1/embeddings")  # closed port: dense search unavailable
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    scan.scan(conn, cfg)
    ids = {r["rel_path"]: r["media_id"] for r in conn.execute("SELECT rel_path, media_id FROM media_path")}
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'legacy_caption', ?)",
                 (ids["ukraine/shot.png"], "A tweet about Ukraine and long range missiles"))
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'legacy_caption', ?)",
                 (ids["cookies.txt"], "missiles cookie secret"))
    conn.commit()
    conn.close()
    return TestClient(api.create_app(cfg)), ids


def lines(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_stream_sections_in_order_with_dense_fallback(env):
    client, ids = env
    resp = client.get("/api/stream", params={"q": "missi"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("application/x-ndjson")
    msgs = lines(resp)
    assert [m["t"] for m in msgs] == ["scope", "lexical", "lenses", "pages", "warn", "correlations", "done"]
    hits = msgs[1]["hits"]
    assert [h["media_id"] for h in hits] == [ids["ukraine/shot.png"]]  # the EXCLUDED cookies doc never appears
    assert "\x02missiles\x03" in hits[0]["hit"]["snippet"]


def test_stream_empty_query_lists_scope(env):
    client, ids = env
    msgs = lines(client.get("/api/stream", params=[("s", "folder:ukraine")]))
    assert [m["t"] for m in msgs] == ["scope", "items", "lenses", "correlations", "done"]
    assert [h["media_id"] for h in msgs[1]["hits"]] == [ids["ukraine/shot.png"]]
    assert msgs[0]["n_in_scope"] == 1


def test_bad_scope_is_400(env):
    client, _ = env
    assert client.get("/api/stream", params=[("s", "bogus:1")]).status_code == 400


def test_media_file_confinement_and_ranges(env):
    client, ids = env
    shot = client.get(f"/media/{ids['ukraine/shot.png']}/file")
    assert shot.status_code == 200 and shot.headers["content-type"] == "image/png"
    # A symlink inside the corpus pointing outside it is never served.
    assert client.get(f"/media/{ids['sneaky.png']}/file").status_code == 404
    assert client.get(f"/media/{ids['cookies.txt']}/file").status_code == 404
    ranged = client.get(f"/media/{ids['clip.mp4']}/file", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100


def test_confined_path_rejects_traversal(env, tmp_path):
    cfg = dataclasses.replace(config.load(), corpus_root=tmp_path / "corpus")
    assert api.confined_path(cfg, "ukraine/shot.png") is not None
    assert api.confined_path(cfg, "../outside.txt") is None
    assert api.confined_path(cfg, "/etc/passwd") is None


def test_item_detail(env):
    client, ids = env
    detail = client.get(f"/api/item/{ids['ukraine/shot.png']}").json()
    assert detail["docs"]["legacy_caption"][0]["text"].startswith("A tweet")
    assert detail["paths"][0]["rel_path"] == "ukraine/shot.png"
    assert detail["title"] == "A tweet about Ukraine and long range missiles"
    assert client.get(f"/api/item/{ids['cookies.txt']}").status_code == 404


def test_legacy_routes(env):
    client, _ = env
    assert client.get("/legacy/").status_code == 200
    assert client.get("/legacy/matrix").status_code == 200
    assert client.get("/legacy/script.js").status_code == 200
    assert client.get("/hash-to-path/" + "0" * 64).status_code == 404


def test_timeline_counts_months(env):
    client, ids = env
    data = client.get("/api/timeline").json()
    assert sum(m["image"] + m["video"] for m in data["months"]) + data["undated"] == 2
    assert client.get("/api/timeline", params={"q": "missiles"}).json()["undated"] + \
        sum(m["image"] for m in client.get("/api/timeline", params={"q": "missiles"}).json()["months"]) == 1


def test_posts_fold_photos_into_the_screenshot(tmp_path):
    from re2 import legacy, search

    root = tmp_path / "corpus"
    root.mkdir()
    for name, color in (("screenshot_20260914_150141.jpg", "white"), ("screenshot_20260914_150141_photo1.jpg", "red"),
                        ("screenshot_20260914_150141_photo2.png", "blue"), ("other.png", "green")):
        Image.new("RGB", (60, 40), color).save(root / name)
    (tmp_path / "archive_metadata.json").write_text(json.dumps([{
        "url": "https://x.com/a/status/1", "date": "2026-09-14T15:01:48-06:00", "folder": ".",
        "screenshot": "screenshot_20260914_150141.jpg",
        "photos": ["screenshot_20260914_150141_photo1.jpg", "screenshot_20260914_150141_photo2.png"],
        "videos": [], "restricted": False}]))
    cfg = dataclasses.replace(config.load(), corpus_root=root, data_dir=tmp_path / "data",
                              thumbs_dir=tmp_path / "thumbs", archive_metadata=tmp_path / "archive_metadata.json",
                              embed_url="http://127.0.0.1:9/v1/embeddings")
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    scan.scan(conn, cfg)
    assert legacy.reconcile_sources(conn, cfg).posts == 1
    ids = {r[0]: r[1] for r in conn.execute("SELECT rel_path, media_id FROM media_path")}
    shot, photo2 = ids["screenshot_20260914_150141.jpg"], ids["screenshot_20260914_150141_photo2.png"]
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'caption', 'a goat in a meatpacking plant')", (photo2,))
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'caption', 'a goat on a farm')", (ids["other.png"],))
    conn.commit()
    search._post_cache.clear()
    client = TestClient(api.create_app(cfg))

    msgs = lines(client.get("/api/stream", params={"q": "goat"}))
    hits = next(m for m in msgs if m["t"] == "lexical")["hits"]
    post_hit = next(h for h in hits if h["media_id"] == shot)
    assert post_hit["hit"]["member_id"] == photo2           # the photo matched, the post is shown
    assert [m["role"] for m in post_hit["post"]["members"]] == ["screenshot", "photo", "photo"]
    assert photo2 not in [h["media_id"] for h in hits]

    browse = lines(client.get("/api/stream"))
    listed = [h["media_id"] for h in next(m for m in browse if m["t"] == "items")["hits"]]
    assert sorted(listed) == sorted([shot, ids["other.png"]])  # photos never appear on their own

    detail = client.get(f"/api/item/{photo2}").json()
    assert detail["post"]["primary_media_id"] == shot and len(detail["post"]["members"]) == 3

    oldest = lines(client.get("/api/stream", params={"sort": "oldest"}))
    assert next(m for m in oldest if m["t"] == "items")["hits"]
    assert client.get("/api/stream", params={"sort": "sideways"}).status_code == 400
