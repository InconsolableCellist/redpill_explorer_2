import json
from pathlib import Path
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
    assert all("dry_multiplier" in r for r in requests[-2:]) and not any("dry_multiplier" in r for r in requests[:4])
    tall_request = next(r for r in requests[-2:] if sum(p["type"] == "image_url" for p in r["messages"][1]["content"]) > 1)
    assert tall_request["max_tokens"] == 300
    assert db.pending_failures(conn, "vlm_image") == set()
    assert list((tmp_path / "data" / "logs" / "vlm_raw").glob("*.txt"))  # the truncated output was kept
    rows = conn.execute(
        "SELECT m.width, d.text, mm.low_res FROM doc d JOIN media m ON m.id=d.media_id "
        "JOIN media_meta mm ON mm.media_id=m.id WHERE d.kind='caption' ORDER BY m.width").fetchall()
    # Caption text leads with the headline, then the description.
    assert all(r[1].startswith("Headline\nLede first.\n\n") for r in rows)
    assert [(r[0], r[1].split("\n\n")[1], r[2]) for r in rows] == [
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


def test_captions_from_older_prompt_are_redone_after_new_images(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for shade, name in enumerate(("old.png", "current.png", "new.png")):
        Image.new("RGB", (500, 400), (shade * 60, 0, 0)).save(root / name)
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, vision_model="gemma-test")
    scan.scan(conn, cfg)
    ids = {r[1]: r[0] for r in conn.execute("SELECT media_id, rel_path FROM media_path")}
    conn.execute("INSERT INTO doc(media_id, kind, text, json) VALUES (?, 'caption', 'old', '{}')", (ids["old.png"],))
    extract_images.store_image_result(conn, cfg, ids["current.png"], sample_result(), False)
    jobs = extract_images.select_jobs(conn, cfg, None, True)
    assert [(j.media_id, j.redo) for j in jobs] == [(ids["new.png"], False), (ids["old.png"], True)]
    old_doc = conn.execute("SELECT id FROM doc WHERE media_id=? AND kind='caption'", (ids["old.png"],)).fetchone()[0]
    extract_images.store_image_result(conn, cfg, ids["old.png"], sample_result(), False)
    new_doc = conn.execute("SELECT id FROM doc WHERE media_id=? AND kind='caption'", (ids["old.png"],)).fetchone()[0]
    assert new_doc != old_doc  # a new doc id, so the embed step picks up the new text
    assert conn.execute("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'old'").fetchone()[0] == 0
    assert [j.media_id for j in extract_images.select_jobs(conn, cfg, None, True)] == [ids["new.png"]]


def test_newest_images_are_read_first(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for shade, name in enumerate(("2019.png", "2026.png", "undated.png", "2024.png")):
        Image.new("RGB", (500, 400), (shade * 60, 0, 0)).save(root / name)
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root)
    scan.scan(conn, cfg)
    ids = {Path(r[1]).name: r[0] for r in conn.execute("SELECT media_id, rel_path FROM media_path")}
    for name, when in (("2019.png", "2019-03-01"), ("2026.png", "2026-09-14"), ("2024.png", "2024-11")):
        conn.execute("INSERT INTO media_meta(media_id, item_date) VALUES (?, ?)", (ids[name], when))
    order = [j.media_id for j in extract_images.select_jobs(conn, cfg, None, True)]
    assert order == [ids["2026.png"], ids["2024.png"], ids["2019.png"], ids["undated.png"]]


def _recaption_env(tmp_path):
    root = tmp_path / "corpus"
    (root / "protests").mkdir(parents=True)
    Image.new("RGB", (800, 600), "gray").save(root / "protests" / "statue.png")
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, base_url="http://vlm.test/v1", vision_model="gemma-test",
                          vision_max_tokens=100, vision_max_side=1024, vision_concurrency=1,
                          vision_max_tiles=6, vision_max_tokens_tall=300, data_dir=tmp_path / "data")
    scan.scan(conn, cfg)
    media_id = conn.execute("SELECT media_id FROM media_path").fetchone()[0]
    generic = sample_result()
    generic["tags"], generic["entities"]["places"] = ["toppled statue"], ["Somewhere"]
    extract_images.store_image_result(conn, cfg, media_id, generic, False)
    return conn, cfg, media_id


def caption_json(conn, media_id):
    return json.loads(conn.execute("SELECT json FROM doc WHERE media_id=? AND kind='caption'", (media_id,)).fetchone()[0])


def test_recaption_sends_the_owners_note_and_saves_it(tmp_path):
    conn, cfg, media_id = _recaption_env(tmp_path)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        result = sample_result()
        result["tags"] = ["christopher columbus", "st paul"]
        result["entities"]["people"] = ["Christopher Columbus"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    note = "This is the Columbus statue toppled at the Minnesota State Capitol, 10 June 2020"
    extract_images.recaption(conn, cfg, media_id, f"  {note}  ",
                             client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    text = requests[0]["messages"][1]["content"][0]["text"]
    assert text.startswith(extract_images.USER_TEXT)
    assert 'folder "protests"' in text  # the normal hints still go along
    assert f"The archive's owner says about this image: {note}. Treat this as reliable context" in text
    assert requests[0]["messages"][0]["content"] == extract_images.vlm.IMAGE_SYSTEM_PROMPT
    saved = caption_json(conn, media_id)
    assert saved["user_context"] == note and saved["prompt_version"] == extract_images.PROMPT_VERSION
    assert sorted(r[0] for r in conn.execute(
        "SELECT t.name FROM media_tag mt JOIN tag t ON t.id=mt.tag_id WHERE mt.media_id=?", (media_id,))) == [
        "christopher columbus", "st paul"]
    assert [r[0] for r in conn.execute(
        "SELECT e.name FROM media_entity me JOIN entity e ON e.id=me.entity_id WHERE me.media_id=?", (media_id,))] == [
        "Christopher Columbus"]
    assert extract_images.select_jobs(conn, cfg, None, True) == []  # current version: the batch leaves it alone


def test_redo_of_an_older_caption_keeps_the_owners_note(tmp_path):
    conn, cfg, media_id = _recaption_env(tmp_path)
    old = {**sample_result(), "user_context": "The Columbus statue", "prompt_version": 1}
    conn.execute("UPDATE doc SET json=? WHERE media_id=? AND kind='caption'", (json.dumps(old), media_id))
    texts = []

    def handler(request):
        texts.append(json.loads(request.content)["messages"][1]["content"][0]["text"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(sample_result())}}]})

    [job] = extract_images.select_jobs(conn, cfg, None, True)
    assert job.redo and job.user_context == "The Columbus statue"
    extract_images.run(conn, cfg, client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert "owner says about this image: The Columbus statue." in texts[0]
    saved = caption_json(conn, media_id)
    assert saved["user_context"] == "The Columbus statue" and saved["prompt_version"] == extract_images.PROMPT_VERSION


def test_batch_does_not_overwrite_a_recaption_made_while_queued(tmp_path):
    conn, cfg, media_id = _recaption_env(tmp_path)
    conn.execute("UPDATE doc SET json=? WHERE media_id=? AND kind='caption'", (json.dumps(sample_result()), media_id))

    def handler(request):
        # The dashboard recaptions the image while the batch's request for it is in flight.
        extract_images.store_image_result(conn, cfg, media_id, {**sample_result(), "user_context": "note"}, False)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(sample_result())}}]})

    stats = extract_images.run(conn, cfg, client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert stats.selected == 1 and caption_json(conn, media_id)["user_context"] == "note"


def test_parallel_runs_claim_images_so_none_is_read_twice(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for shade in range(4):
        Image.new("RGB", (500, 400), (shade * 60, 0, 0)).save(root / f"{shade}.png")
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, base_url="http://cube.test/v1", vision_model="gemma-test",
                          vision_max_tokens=100, vision_max_side=1024, vision_concurrency=1,
                          vision_max_tiles=6, vision_max_tokens_tall=300, data_dir=tmp_path / "data")
    scan.scan(conn, cfg)
    ids = [r[0] for r in conn.execute("SELECT id FROM media ORDER BY id")]
    # Another run (say on a second GPU host) holds one image; a third claim has lapsed.
    assert db.claim(conn, extract_images.STAGE, ids[0], "whiteglove:1:http://wg/v1", 600)
    assert not db.claim(conn, extract_images.STAGE, ids[0], "cube:2:http://cube/v1", 600)
    assert db.claim(conn, extract_images.STAGE, ids[1], "crashed:3:http://x/v1", -1)
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(sample_result())}}]})

    stats = extract_images.run(conn, cfg, client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert stats.done == 3 and stats.taken_elsewhere == 1 and len(sent) == 3
    captioned = {r[0] for r in conn.execute("SELECT media_id FROM doc WHERE kind='caption'")}
    assert captioned == set(ids[1:])
    # Finished items give their claims back; the other run's claim is untouched.
    assert [r[0] for r in conn.execute("SELECT media_id FROM work_claim")] == [ids[0]]
