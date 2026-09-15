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
    Image.new("RGB", (600, 3000), "red").save(root / "tall.png")
    Image.new("RGB", (700, 700), "yellow").save(root / "dense.png")

    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, base_url="http://vlm.test/v1", vision_model="gemma-test",
                          vision_max_tokens=100, vision_max_side=1024, vision_concurrency=2,
                          vision_max_tiles=6, vision_max_tokens_tall=300, data_dir=tmp_path / "data")
    scan.scan(conn, cfg)

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        user_text = body["messages"][1]["content"][0]["text"]
        if b"data:image/jpeg;base64,RkFJTA==" in request.content:  # b"FAIL"
            return httpx.Response(400, text="rejected")
        if b"data:image/jpeg;base64,REVOU0U=" in request.content and body["max_tokens"] < 300:  # b"DENSE"
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"description": "trunc'},
                                                           "finish_reason": "length"}]})
        result = sample_result()
        result["tags"] = ["Joe_Biden", "joe biden", "x"]
        result["entities"]["people"] = ["Joe Biden", " "]
        result["description"] = "low" if "low-resolution" in user_text else "full"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}],
                                         "usage": {"completion_tokens": 10}})

    real_prepare = extract_images.vlm.prepare_images
    monkeypatch.setattr(extract_images.vlm, "prepare_images",
                        lambda path, side, tiles, dense=False: {"fails.png": [b"FAIL"], "dense.png": [b"DENSE"]}.get(path.name)
                        or real_prepare(path, side, tiles, dense))
    stats = extract_images.run(conn, cfg, client_factory=lambda: httpx.Client(
        transport=httpx.MockTransport(handler)))

    assert (stats.done, stats.failed, stats.tiled, stats.retried_long) == (4, 1, 1, 1)
    # Normal images first; then the tall image (as tiles) and the image that ran
    # out of tokens, one at a time with the larger budget.
    budgets = [r["max_tokens"] for r in requests]
    assert budgets == [100] * 4 + [300] * 2
    tall_request = next(r for r in requests[-2:] if sum(p["type"] == "image_url" for p in r["messages"][1]["content"]) > 1)
    assert tall_request["max_tokens"] == 300
    assert db.pending_failures(conn, "vlm_image") == set()
    assert list((tmp_path / "data" / "logs" / "vlm_raw").glob("*.txt"))  # the truncated output was kept
    rows = conn.execute(
        "SELECT m.width, d.text, mm.low_res FROM doc d JOIN media m ON m.id=d.media_id "
        "JOIN media_meta mm ON mm.media_id=m.id WHERE d.kind='caption' ORDER BY m.width").fetchall()
    assert [(r[0], r[1].split("\n")[0], r[2]) for r in rows] == [
        (300, "low", 1), (600, "full", 0), (700, "full", 0), (800, "full", 0)]
    assert conn.execute("SELECT count(*) FROM doc WHERE kind='ocr'").fetchone()[0] == 4
    assert [r[0] for r in conn.execute("SELECT name FROM tag")] == ["joe biden"]
    assert conn.execute("SELECT count(*) FROM media_entity").fetchone()[0] == 4
    errors = [r[0] for r in conn.execute("SELECT error FROM failure")]
    assert len(errors) == 1 and "400" in errors[0]
    assert conn.execute(
        "SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'hello'").fetchone()[0] == 4
    # Re-run selects only the failed item.
    assert len(extract_images.select_jobs(conn, cfg, None, True)) == 1


def test_previous_length_failures_go_to_long_phase(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    Image.new("RGB", (800, 600)).save(root / "a.png")
    Image.new("RGB", (800, 601)).save(root / "b.png")
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root)
    scan.scan(conn, cfg)
    ids = [r[0] for r in conn.execute("SELECT id FROM media ORDER BY id")]
    db.upsert_failure(conn, "vlm_image", ids[0], None,
                      "VLMError: unparseable response (finish_reason=length, max_tokens=1600): x")
    jobs = {j.media_id: j.long_output for j in extract_images.select_jobs(conn, cfg, None, True)}
    assert jobs == {ids[0]: True, ids[1]: False}
