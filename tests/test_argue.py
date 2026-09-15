import json

import httpx
import pytest
from fastapi.testclient import TestClient

from re2 import api, argue, ask
from tests.test_ask import archive  # noqa: F401 - the shared fixture


def debate_model(seen):
    """Research steps always answer at once; each speaker's text names its role and cites an item."""
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("response_format"):
            chunks = [json.dumps({"thought": "", "action": "answer", "args": {}})]
        else:
            role = "skeptic" if "You are the SKEPTIC" in body["messages"][-1]["content"] else "steelman"
            chunks = [f"As {role}, the transcript shows it [m:1]. ", "Where I stand: holding."]
        sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
        return httpx.Response(200, text=sse + "data: [DONE]\n\n")
    return httpx.Client(transport=httpx.MockTransport(handler))


def run_all(conn, cfg, **kw):
    return list(argue.events(conn, cfg, kw.pop("session_id", "deb1"), topic=kw.pop("topic", "Trudeau armed Ukraine"),
                             user_role=kw.pop("user_role", None), user_text=kw.pop("user_text", ""), rounds=1, seed=None,
                             scopes=[], mode="informed", depth=0, items=[], related=False, client=kw.pop("client"),
                             embed_client=None, control=ask.Control()))


def test_watching_a_round_runs_both_sides_and_passes_the_transcript(archive):
    conn, cfg = archive
    seen = []
    events = run_all(conn, cfg, client=debate_model(seen))
    assert [d["role"] for n, d in events if n == "speaker"] == ["steelman", "skeptic"]
    rows = conn.execute("SELECT question, mode, answer FROM chat_log WHERE session_id='deb1' ORDER BY id").fetchall()
    assert [r[1] for r in rows] == ["steelman", "skeptic"] and rows[0][0] == "Steelman · opening"
    assert "[1]" in rows[1][2] and "Where I stand" in rows[1][2]
    # The skeptic saw the steelman's turn, with its citation restored to [m:1].
    skeptic_step = [b for b in seen if b.get("response_format")][1]
    prompt = skeptic_step["messages"][-1]["content"]
    assert "Claim under debate: Trudeau armed Ukraine" in prompt and "STEELMAN:\nAs steelman" in prompt and "[m:1]" in prompt
    assert "You are the SKEPTIC" in prompt and "SKEPTIC in a structured debate" in skeptic_step["messages"][0]["content"]
    assert conn.execute("SELECT kind, title FROM chat_session WHERE id='deb1'").fetchone()[:] == ("argue", "Trudeau armed Ukraine")
    # Next round starts with the steelman again.
    assert argue.plan_speakers(argue.turns(conn, "deb1"), None, "", 1) == ["steelman", "skeptic"]


def test_taking_a_side(archive):
    conn, cfg = archive
    seen = []
    events = run_all(conn, cfg, session_id="deb2", user_role="steelman", user_text="Canada sent missiles [m:1].",
                     client=debate_model(seen))
    speakers = [(d["role"], d["by"]) for n, d in events if n == "speaker"]
    assert speakers == [("steelman", "user"), ("skeptic", "ai")]
    rows = conn.execute("SELECT question, mode, context_json FROM chat_log WHERE session_id='deb2' ORDER BY id").fetchall()
    assert rows[0][0] == "Steelman (you)" and json.loads(rows[0][2]) == {"by": "user"}
    assert "STEELMAN (WRITTEN BY THE USER):" in [b for b in seen if b.get("response_format")][0]["messages"][-1]["content"]
    # A user steelman with nothing written cannot proceed; a user skeptic lets the AI open.
    with pytest.raises(ValueError):
        argue.plan_speakers(argue.turns(conn, "deb2"), "steelman", "", 1)
    assert argue.plan_speakers([], "skeptic", "", 1) == ["steelman"]


def test_argue_endpoint(archive, monkeypatch):
    conn, cfg = archive
    app = TestClient(api.create_app(cfg))
    argue_client = debate_model([])
    resp_bad = app.post("/api/argue", json={"user_role": "steelman", "topic": "x"})
    assert resp_bad.status_code == 400
    real_start = argue.start
    monkeypatch.setattr(argue, "start", lambda cfg_, **kw: real_start(cfg_, **{**kw, "client": argue_client}))
    resp = app.post("/api/argue", json={"topic": "A claim", "seed": {"question": "q", "answer": "It was so [m:1]."}})
    assert resp.status_code == 200 and resp.text.count("event: speaker") == 2
    sid = [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ") and "run_id" in l][0]["session_id"]
    data = app.get(f"/api/ask/session/{sid}").json()
    assert data["kind"] == "argue" and data["seed"]["answer"].startswith("It was so") and [t["mode"] for t in data["turns"]] == ["steelman", "skeptic"]
    assert app.get("/api/ask/sessions").json()["sessions"][0]["kind"] == "argue"
