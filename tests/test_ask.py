import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from re2 import api, ask, config, db


@pytest.fixture
def archive(tmp_path):
    cfg = dataclasses.replace(config.load(), data_dir=tmp_path / "data", corpus_root=tmp_path,
                              chat_base_url="http://chat.test/v1", chat_model="qwen-test", chat_context_tokens=32000,
                              embed_url="http://127.0.0.1:9/v1/embeddings")
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'video', 'READY'), (2, 'b', 'image', 'SOURCE_OFFLINE');
        INSERT INTO doc(media_id, kind, seq, start_s, end_s, text) VALUES
          (1, 'transcript', 0, 35, 80, 'Trudeau said Ukraine can use Canadian long range missiles inside Russia'),
          (2, 'legacy_caption', 0, NULL, NULL, 'A meme about Trudeau and missiles');
    """)
    conn.commit()
    return conn, cfg


def scripted_model(steps, answer_chunks, seen):
    replies = iter(steps)

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in answer_chunks)
            return httpx.Response(200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_agent_loop_cites_only_what_tools_returned(archive):
    conn, cfg = archive
    seen = []
    client = scripted_model(
        steps=[{"thought": "look", "action": "search", "args": {"query": "trudeau missiles"}},
               {"thought": "read it", "action": "open_item", "args": {"id": 1}},
               {"thought": "enough", "action": "answer", "args": {}}],
        answer_chunks=["Trudeau said Ukraine may strike Russia with Canadian missiles [m:", "1]. ",
                       "A meme mocked it [m:2]. An invented item [m:999]. ", "NATO debated this for months."],
        seen=seen)
    events = list(ask.run(conn, cfg, "What did Trudeau say about missiles?", client=client))
    names = [e[0] for e in events]
    assert names[:2] == ["step", "step"] and names[-2:] == ["final", "done"]
    assert "[m:1]" in events[0][1]["summary"] or "items match" in events[0][1]["summary"]
    cites = [e[1] for e in events if e[0] == "cite"]
    assert [(c["n"], c["ref"]) for c in cites] == [(1, "m:1"), (2, "m:2")]
    final = dict(events)["final"]
    assert "[m:" not in final["text"] and "[1]" in final["text"] and "[2]" in final["text"] and "999" not in final["text"]
    assert [s["general"] for s in final["sentences"] if s["text"].strip()] == [False, False, True, True]
    # The open_item result reached the model, with the transcript timestamp.
    assert any("(0:35)" in m["content"] for m in seen[2]["messages"] if m["role"] == "user")
    row = conn.execute("SELECT question, session_id, trace_json FROM chat_log").fetchone()
    assert row[0].startswith("What did Trudeau") and len(json.loads(row[2])) == 3


def test_step_cap_and_unavailable_model(archive):
    conn, cfg = archive
    looping = scripted_model(steps=[{"thought": "again", "action": "timeline", "args": {"query": "trudeau"}}] * 20,
                             answer_chunks=["Nothing much."], seen=[])
    events = list(ask.run(conn, cfg, "When?", client=looping))
    assert sum(1 for e in events if e[0] == "step") == ask.MAX_STEPS

    def down(request):
        raise httpx.ConnectError("no route")
    events = list(ask.run(conn, cfg, "Anything?", client=httpx.Client(transport=httpx.MockTransport(down))))
    assert events[-1][0] == "error"


def test_citation_filter_holds_split_markers():
    ev = ask.Evidence(items={5: {}}, pages={})
    f = ask.CitationFilter(ev)
    assert f.feed("A claim [m") == "A claim "
    assert f.feed(":5] and more") == "[1] and more"
    assert f.flush() == ""


def test_ask_endpoint_streams_sse(archive, monkeypatch):
    conn, cfg = archive
    client = scripted_model(steps=[{"thought": "", "action": "answer", "args": {}}], answer_chunks=["No evidence."],
                            seen=[])
    real_run = ask.run
    monkeypatch.setattr(ask, "run", lambda *a, **kw: real_run(*a, **{**kw, "client": client}))
    app = TestClient(api.create_app(cfg))
    resp = app.post("/api/ask", json={"question": "Hello?"})
    assert resp.status_code == 200 and "event: final" in resp.text and "event: done" in resp.text
    session = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ") and "session_id" in line][0]
    turns = app.get(f"/api/ask/session/{session['session_id']}").json()["turns"]
    assert turns[0]["answer"] == "No evidence."
    assert app.get("/api/ask/sessions").json()["sessions"][0]["turns"] == 1
    assert app.post("/api/ask", json={"question": " "}).status_code == 400
