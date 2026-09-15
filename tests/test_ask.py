import dataclasses
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from re2 import api, ask, config, db, search


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
        # Every call streams; the research steps are the ones constrained to the step schema.
        chunks = [json.dumps(next(replies))] if body.get("response_format") else answer_chunks
        sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
        return httpx.Response(200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

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
    # The answer call is a fresh conversation with the research as notes, not a JSON transcript.
    answer_call = seen[-1]
    assert not answer_call.get("response_format") and len(answer_call["messages"]) == 2
    assert "Research notes" in answer_call["messages"][1]["content"] and "(0:35)" in answer_call["messages"][1]["content"]
    row = conn.execute("SELECT question, session_id, trace_json FROM chat_log").fetchone()
    trace = json.loads(row[2])
    assert row[0].startswith("What did Trudeau") and [t["action"] for t in trace] == ["search", "open_item", "answer", "fill", "coverage"]
    # The notes had room, so the item the search found but the agent did not open was read in too.
    assert trace[-2]["args"] == {"items": 1} and "A meme about Trudeau" in answer_call["messages"][1]["content"]
    # Coverage: both items the keyword search matched were read.
    assert dict(events)["coverage"] == {"candidates": 2, "keyword": 2, "read": 2}


def test_step_cap_and_unavailable_model(archive):
    conn, cfg = archive
    seen = []
    pl = ask.plan(cfg)
    looping = scripted_model(steps=[{"thought": "again", "action": "search", "args": {"query": "trudeau"}}] * pl.searches +
                             [{"thought": "again", "action": "timeline", "args": {"query": "trudeau"}}] * 30,
                             answer_chunks=["Nothing much."], seen=seen)
    events = list(ask.run(conn, cfg, "When?", client=looping))
    assert sum(1 for e in events if e[0] == "step" and e[1]["action"] != "read") == pl.steps - 1
    enums = [c["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] for c in seen if c.get("response_format")]
    assert "search" in enums[pl.searches - 1] and "search" not in enums[pl.searches]   # searches run out
    assert enums[-1] == ["answer"]                                                     # the last step can only answer
    # Deeper research gets more searches, more opens, and more of the window for notes.
    shallow, deep = ask.plan(cfg, 0), ask.plan(cfg, 1)
    assert shallow.searches < deep.searches and shallow.opens < deep.opens and shallow.notes_tokens * 5 < deep.notes_tokens
    assert deep.notes_tokens <= cfg.chat_context_tokens - ask.ANSWER_MAX_TOKENS

    def down(request):
        raise httpx.ConnectError("no route")
    events = list(ask.run(conn, cfg, "Anything?", client=httpx.Client(transport=httpx.MockTransport(down))))
    assert events[-1][0] == "error"


def test_attached_item_related_items_and_depth(archive):
    conn, cfg = archive
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (3, 'c', 'image', 'READY'), (4, 'd', 'image', 'READY');
        INSERT INTO media_meta(media_id, item_date) VALUES (1, '2024-09-12'), (3, '2024-09-20'), (4, '2019-01-01');
        INSERT INTO entity(id, name, type) VALUES (7, 'Justin Trudeau', 'person');
        INSERT INTO media_entity(media_id, entity_id) VALUES (1, 7), (3, 7), (4, 7);
        INSERT INTO doc(media_id, kind, seq, text) VALUES (3, 'legacy_caption', 0, 'Trudeau at a podium in Ottawa'),
                                                           (4, 'legacy_caption', 0, 'An old Trudeau photo');
    """)
    conn.commit()
    # Same person within a month counts as related; the same person five years earlier does not.
    assert ask.related_items(conn, cfg, 1, k=5) == [3]
    # Comma lists in a scope mean any of them.
    conn.execute("INSERT INTO entity(id, name, type) VALUES (8, 'Ottawa', 'place')")
    conn.execute("INSERT INTO media_entity(media_id, entity_id) VALUES (2, 8)")
    assert search.scope_media_ids(conn, ["entity:8"]) == {2} and search.scope_media_ids(conn, ["entity:7,8"]) == {1, 2, 3, 4}
    assert search.scope_label(conn, "entity:7,8") == "Justin Trudeau or Ottawa"
    seen = []
    client = scripted_model(steps=[{"thought": "", "action": "answer", "args": {}}], answer_chunks=["It shows [m:3]."], seen=seen)
    events = list(ask.run(conn, cfg, "Discuss this", client=client, items=[1], related=True, depth=0))
    first = seen[0]["messages"][-1]["content"]
    assert first.startswith("The user is asking about these archive items") and "(0:35)" in first and "[m:3]" in first
    assert events[0][1]["action"] == "attached" and dict(events)["final"]["text"] == "It shows [1]."
    ctx = json.loads(conn.execute("SELECT context_json FROM chat_log").fetchone()[0])
    assert {k: ctx[k] for k in ("items", "related", "depth")} == {"items": [1], "related": True, "depth": 0.0}
    assert ctx["coverage"]["read"] >= 2


def test_exhaustive_reads_every_candidate_in_batches(archive, monkeypatch):
    conn, cfg = archive
    rows = ", ".join(f"({i}, 'u{i}', 'image', 'READY')" for i in range(10, 40))
    filler = "x" * 900
    docs = ", ".join(f"({i}, 'legacy_caption', 0, 'A post about Trudeau and missiles number {i} {filler}')" for i in range(10, 40))
    conn.executescript(f"INSERT INTO media(id, uid, kind, status) VALUES {rows}; INSERT INTO doc(media_id, kind, seq, text) VALUES {docs};")
    conn.commit()
    monkeypatch.setattr(ask, "_batch_chars", lambda cfg: 9000)       # about nine records a batch
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        last = body["messages"][-1]["content"]
        if body.get("response_format"):
            chunks = [json.dumps({"thought": "", "action": "search", "args": {"query": "trudeau missiles"}})
                      if sum(1 for b in seen if b.get("response_format")) == 1 else json.dumps({"thought": "", "action": "answer", "args": {}})]
        elif last.startswith("Question:") and "archive items (batch" in last:
            ids = re.findall(r"^\[m:(\d+)\]", last, flags=re.M)
            chunks = ["- Trudeau and missiles " + " ".join(f"[m:{i}]" for i in ids)]
        else:
            chunks = ["Everything read [m:10] [m:39]."]
        sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
        return httpx.Response(200, text=sse + "data: [DONE]\n\n")

    events = list(ask.run(conn, cfg, "What does the archive say about Trudeau and missiles?",
                          client=httpx.Client(transport=httpx.MockTransport(handler)), exhaustive=True))
    cov = dict(events)["coverage"]
    assert cov["keyword"] == 32 and cov["read"] == 32 and cov["batches"] >= 3 and not cov["capped"]
    batch_calls = [b for b in seen if "archive items (batch" in b["messages"][-1]["content"]]
    read_ids = {m for b in batch_calls for m in re.findall(r"^\[m:(\d+)\]", b["messages"][-1]["content"], flags=re.M)}
    assert len(batch_calls) == cov["batches"] and len(read_ids) == 32
    assert dict(events)["final"]["text"] == "Everything read [1] [2]."


def test_stop_ends_a_turn_between_chunks(archive):
    conn, cfg = archive
    control = ask.Control()

    def slow(request):
        body = json.loads(request.content)
        if body.get("response_format"):
            return httpx.Response(200, text=f"data: {json.dumps({'choices': [{'delta': {'content': json.dumps({'thought': '', 'action': 'answer', 'args': {}})}}]})}\n\n")
        def chunks():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Partial answer. '}}]})}\n\n".encode()
            control.stop.set()   # the user presses stop while the answer streams
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Never shown.'}}]})}\n\n".encode()
        return httpx.Response(200, content=chunks())

    events = list(ask.run(conn, cfg, "Stop me", client=httpx.Client(transport=httpx.MockTransport(slow)), control=control))
    names = [e[0] for e in events]
    assert "stopped" in names and names[-1] == "done" and events[-1][1]["stopped"]
    assert conn.execute("SELECT answer FROM chat_log").fetchone()[0] == "Partial answer. (stopped)"


def test_citation_filter_holds_split_markers():
    ev = ask.Evidence(items={5: {}}, pages={})
    f = ask.CitationFilter(ev)
    assert f.feed("A claim [m") == "A claim "
    assert f.feed(":5] and more") == "[1] and more"
    assert f.flush() == ""
    assert f.feed("It peaked in September [timeline]. ") == "It peaked in September. "


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
    # A finished run can be watched again from the start, and a conversation can be deleted.
    start = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ") and "run_id" in line][0]
    again = app.get(f"/api/ask/run/{start['run_id']}/events")
    assert "event: final" in again.text and app.get("/api/ask/runs").json()["runs"] == []
    plan = app.get("/api/ask/plan", params={"depth": 1, "s": "kind:video"}).json()
    assert plan["in_scope"] == 1 and plan["notes_tokens"] > app.get("/api/ask/plan", params={"depth": 0}).json()["notes_tokens"]
    assert app.delete(f"/api/ask/session/{session['session_id']}").json() == {"deleted_turns": 1}
    assert app.get("/api/ask/sessions").json()["sessions"] == []


def test_settings_save_to_local_file_and_apply_live(archive, monkeypatch, tmp_path):
    conn, cfg = archive
    local = tmp_path / "config.local.toml"
    local.write_text('# my notes\n[server]\nhost = "0.0.0.0"\n')
    monkeypatch.setattr(config, "LOCAL_OVERRIDES", local)
    app = TestClient(api.create_app(cfg))
    change = {"services": {"chat": {"model": "gemma-big", "context_tokens": 160000}, "asr": {"device": "cpu"}}}
    assert app.post("/api/settings", json=change).status_code == 403      # no header: refused
    resp = app.post("/api/settings", json=change, headers={"X-RE2-Settings": "1"})
    assert resp.status_code == 200 and resp.json()["services"]["chat"]["model"] == "gemma-big"
    assert (cfg.chat_model, cfg.chat_context_tokens, cfg.asr_device) == ("gemma-big", 160000, "cpu")
    text = local.read_text()
    assert text.startswith("# my notes") and 'model = "gemma-big"' in text and "context_tokens = 160000" in text
    assert app.get("/api/ask/plan").json()["context_tokens"] == 160000
    bad = {"services": {"chat": {"base_url": "file:///etc/passwd"}}}
    assert app.post("/api/settings", json=bad, headers={"X-RE2-Settings": "1"}).status_code == 400
