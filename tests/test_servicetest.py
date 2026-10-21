import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from re2 import api, config, servicetest

URL = "http://llm.test/v1"


def server(reply):
    """A llama.cpp router with one loaded model that reads images; chat answers with reply(body)."""
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m", "status": {
                "value": "loaded", "args": ["--ctx-size", "32768", "--mmproj", "x"]}}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"total_slots": 4, "modalities": {"vision": True}})
        if request.url.path == "/slots":
            return httpx.Response(200, json=[{"n_ctx": 32768}] * 4)
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": reply(body)}}],
                                             "timings": {"predicted_per_second": 40.0}})
        return httpx.Response(404)
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def cfg():
    return config.load()


def test_chat_ready(cfg):
    d = servicetest.run(cfg, "chat", {"base_url": URL, "model": "m"}, server(lambda b: "<think>hm</think>READY"))
    assert d["ok"] and [s["name"] for s in d["steps"]] == ["server", "reply"]
    assert "40 tokens/s" in d["steps"][1]["detail"]
    d = servicetest.run(cfg, "wiki", {"base_url": URL, "model": "m"}, server(lambda b: "Sure! Here you go."))
    assert not d["ok"] and "expected READY" in d["steps"][1]["detail"]
    d = servicetest.run(cfg, "chat", {"base_url": URL, "model": "missing"}, server(lambda b: "READY"))
    assert not d["ok"] and len(d["steps"]) == 1         # no request to a model the server does not have


def test_vision_must_read_the_number(cfg, monkeypatch):
    monkeypatch.setattr(servicetest.random, "randint", lambda a, b: 731)
    vals = {"base_url": URL, "vision_model": "m"}
    assert servicetest.run(cfg, "vision", vals, server(lambda b: "731"))["ok"]
    bad = servicetest.run(cfg, "vision", vals, server(lambda b: "I see a box."))
    assert not bad["ok"] and "wrong answer" in bad["steps"][-1]["detail"]
    assert servicetest.test_image("731")[:8] == b"\x89PNG\r\n\x1a\n"


def test_bad_values_and_the_api(cfg, tmp_path):
    with pytest.raises(ValueError):
        servicetest.run(cfg, "vision", {"base_url": "file:///etc"})
    with pytest.raises(ValueError):
        servicetest.run(cfg, "nope", {})
    app = TestClient(api.create_app(dataclasses.replace(cfg, data_dir=tmp_path / "data")))
    assert app.post("/api/settings/test", json={"service": "chat"}).status_code == 403
    r = app.post("/api/settings/test", json={"service": "chat", "values": {"base_url": "ftp://x"}},
                 headers={"X-RE2-Settings": "1"})
    assert r.status_code == 400
