import json
from types import SimpleNamespace

import httpx
from PIL import Image

from re2 import db, extract_images, scan

from tests.test_vlm import sample_result


def test_extract_images_pipeline(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    root.mkdir()
    Image.new("RGB", (800, 600), "white").save(root / "good.png")
    Image.new("RGB", (300, 200), "black").save(root / "small.png")
    Image.new("RGB", (900, 900), "gray").save(root / "fails.png")

    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, base_url="http://vlm.test/v1", vision_model="gemma-test",
                          vision_max_tokens=100, vision_max_side=1024, vision_concurrency=2)
    scan.scan(conn, cfg)

    def handler(request):
        body = json.loads(request.content)
        user_text = body["messages"][1]["content"][0]["text"]
        if b"data:image/jpeg;base64,RkFJTA==" in request.content:  # b"FAIL"
            return httpx.Response(400, text="rejected")
        result = sample_result()
        result["tags"] = ["Joe_Biden", "joe biden", "x"]
        result["entities"]["people"] = ["Joe Biden", " "]
        result["description"] = "low" if "low-resolution" in user_text else "full"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}],
                                         "usage": {"completion_tokens": 10}})

    real_prepare = extract_images.vlm.prepare_image
    monkeypatch.setattr(extract_images.vlm, "prepare_image",
                        lambda path, side: b"FAIL" if path.name == "fails.png" else real_prepare(path, side))
    stats = extract_images.run(conn, cfg, client_factory=lambda: httpx.Client(
        transport=httpx.MockTransport(handler)))

    assert (stats.done, stats.failed) == (2, 1)
    rows = conn.execute(
        "SELECT m.width, d.text, mm.low_res FROM doc d JOIN media m ON m.id=d.media_id "
        "JOIN media_meta mm ON mm.media_id=m.id WHERE d.kind='caption' ORDER BY m.width").fetchall()
    assert [(r[0], r[1].split("\n")[0], r[2]) for r in rows] == [(300, "low", 1), (800, "full", 0)]
    assert conn.execute("SELECT count(*) FROM doc WHERE kind='ocr'").fetchone()[0] == 2
    assert [r[0] for r in conn.execute("SELECT name FROM tag")] == ["joe biden"]
    assert conn.execute("SELECT count(*) FROM media_entity").fetchone()[0] == 2
    assert conn.execute("SELECT stage FROM failure").fetchone()[0] == "vlm_image"
    assert conn.execute(
        "SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'hello'").fetchone()[0] == 2
    # Re-run selects only the failed item.
    assert len(extract_images.select_jobs(conn, cfg, None, True)) == 1
