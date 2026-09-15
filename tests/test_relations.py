import json

import httpx

from re2 import db, relations


def archive():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executemany("INSERT INTO media(id, uid, kind, status) VALUES (?, ?, 'image', 'READY')",
                     [(i, f"u{i}") for i in range(1, 41)])
    conn.executemany("INSERT INTO tag(id, name) VALUES (?, ?)",
                     [(1, "jews"), (2, "israelis"), (3, "gaza"), (4, "election interference"), (5, "screenshot")])
    rows = []
    for m in range(1, 41):
        rows.append((m, 5))                      # format tag everywhere
        if m <= 20:
            rows += [(m, 1), (m, 2)]             # jews and israelis always together
        if 10 < m <= 30:
            rows.append((m, 3))
        if m > 25:
            rows.append((m, 4))
    conn.executemany("INSERT INTO media_tag(media_id, tag_id, origin) VALUES (?, ?, 'vlm')", rows)
    return conn


def test_observed_and_semantic(monkeypatch):
    monkeypatch.setattr(relations, "MIN_ITEMS", 5)
    conn = archive()
    cfg = type("Cfg", (), {"chat_base_url": "http://chat.test/v1", "chat_model": "qwen"})()
    replies = {"jews": [["israelis", 95], ["gaza", 90], ["jews", 100]],
               "israelis": [["jews", 90]],
               "gaza": [["election interference", 30], ["nonsense", 80], ["malformed"]],
               "election interference": []}
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        anchor = body["messages"][1]["content"].split("\n")[0].removeprefix("Tag: ")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"related": replies[anchor]})}}]})

    stats = relations.run(conn, cfg, client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert stats.tags == 4 and stats.anchors_done == 4               # the format tag is never scored
    assert "screenshot" not in seen[0]["messages"][0]["content"]
    rel = {r["name"]: r for r in relations.related(conn, 1)}
    assert rel["israelis"]["observed"] == 1.0 and round(rel["israelis"]["semantic"], 3) == 0.925   # mean of both sides
    assert rel["gaza"]["semantic"] == 0.9 * relations.ONE_SIDED    # gaza never named jews back
    assert "dry_multiplier" in seen[0]
    assert relations.run(conn, cfg).anchors_selected == 0             # resumable: nothing left to score
