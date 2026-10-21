import argparse
import dataclasses

import httpx
import pytest
from fastapi.testclient import TestClient

from re2 import api, config, presets

MLBOY = "https://mlboy.test:5000/v1"
ROUTER_MODELS = {"data": [
    {"id": "gemma4-31b-it-mtp-mlboy", "status": {"value": "loaded", "args": ["--ctx-size", "262144", "--mmproj", "x"]}},
    {"id": "gemma4-31b-it-heretic-ara-q8-mlboy", "status": {"value": "unloaded", "args": ["--ctx-size", "262144", "--mmproj", "x"]}},
    {"id": "qwen38-27b-heretic-uncensored-q8-mlboy", "status": {"value": "unloaded", "args": ["--ctx-size", "262144"]}},
]}


def router(request):
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json=ROUTER_MODELS)
    if request.url.path == "/props":
        return httpx.Response(200, json={"total_slots": 4, "modalities": {"vision": True}})
    if request.url.path == "/slots":
        return httpx.Response(200, json=[{"n_ctx": 262144}] * 4)
    return httpx.Response(404)


def client_factory():
    return httpx.Client(transport=httpx.MockTransport(router))


@pytest.fixture
def local(tmp_path, monkeypatch):
    path = tmp_path / "config.local.toml"
    path.write_text('# my notes\n[server]\nhost = "0.0.0.0"\n')
    monkeypatch.setattr(config, "LOCAL_OVERRIDES", path)
    return path


@pytest.fixture
def cfg():
    return dataclasses.replace(config.load(config.REPO_ROOT / "config.toml"), base_url="https://cube.test:5000/v1",
                               vision_model="gemma-cube", vision_concurrency=0)


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_save_load_delete_keeps_comments(local):
    presets.save("mlboy-mtp", {"base_url": MLBOY + "/", "model": "gemma4-31b-it-mtp-mlboy", "concurrency": "auto"})
    presets.save("mlboy-qwen", {"base_url": MLBOY, "model": "qwen", "context_tokens": "32000", "concurrency": 2})
    text = local.read_text()
    assert text.startswith("# my notes") and 'concurrency = "auto"' in text
    assert presets.load_all() == {
        "mlboy-mtp": {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy", "concurrency": 0},
        "mlboy-qwen": {"base_url": MLBOY, "model": "qwen", "concurrency": 2, "context_tokens": 32000}}
    assert presets.delete("mlboy-qwen") and not presets.delete("mlboy-qwen")
    assert list(presets.load_all()) == ["mlboy-mtp"]
    for bad in [("x y", {"base_url": MLBOY, "model": "m"}), ("ok", {"base_url": "file:///etc", "model": "m"}),
                ("ok", {"base_url": MLBOY, "model": " "})]:
        with pytest.raises(ValueError):
            presets.save(*bad)
    local.write_text(local.read_text() + '\n[presets.broken]\nmodel = "m"\n')
    assert list(presets.load_all()) == ["mlboy-mtp"]   # a half-written preset is skipped


def test_apply_matching_and_service_values(cfg):
    p = {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy", "concurrency": 3}
    run = presets.apply(cfg, "vision", p)
    assert (run.base_url, run.vision_model, run.vision_concurrency) == (MLBOY, p["model"], 3)
    assert cfg.base_url == "https://cube.test:5000/v1"                       # a copy, not the same cfg
    assert presets.matching(run, "vision", {"a": p}) == "a" and presets.matching(cfg, "vision", {"a": p}) is None
    assert presets.service_values("vision", p) == {"base_url": MLBOY, "vision_model": p["model"], "vision_concurrency": 3}
    wiki = presets.apply(cfg, "wiki", {"base_url": MLBOY, "model": "q", "concurrency": 0, "context_tokens": 9000})
    assert wiki.wiki_concurrency == cfg.wiki_concurrency and wiki.wiki_context_tokens == 9000


def test_probe(cfg):
    with client_factory() as client:
        ok = presets.probe(client, cfg, "vision", {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy"})
        assert ok.ok and ok.entry["vision"] and ok.limits.slots == 4 and ok.at_once == 4
        assert "reads images" in presets.describe(ok) and "4 at once" in presets.describe(ok)
        cold = presets.probe(client, cfg, "vision", {"base_url": MLBOY, "model": "gemma4-31b-it-heretic-ara-q8-mlboy"})
        assert cold.ok and "may unload gemma4-31b-it-mtp-mlboy" in presets.describe(cold)
        missing = presets.probe(client, cfg, "vision", {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-cube"})
        assert not missing.ok and missing.suggest[0] == "gemma4-31b-it-mtp-mlboy"
        text = presets.probe(client, cfg, "vision", {"base_url": MLBOY, "model": "qwen38-27b-heretic-uncensored-q8-mlboy"})
        assert not text.ok and "cannot read images" in text.error
        assert presets.probe(client, cfg, "chat", {"base_url": MLBOY, "model": "qwen38-27b-heretic-uncensored-q8-mlboy"}).ok

    def down(request):
        raise httpx.ConnectError("refused")

    with httpx.Client(transport=httpx.MockTransport(down)) as client:
        gone = presets.probe(client, cfg, "vision", {"base_url": MLBOY, "model": "m"})
        assert not gone.reachable and presets.describe(gone).startswith("✗ cannot reach")


def test_pick_existing_and_make_default(local, cfg):
    presets.save("mlboy-mtp", {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy", "concurrency": "auto"})
    out = []
    # 1 = the current (cube) settings, which do not answer; Enter keeps them only after "Run anyway?"
    run = presets.pick(cfg, "vision", "extract-images", client_factory, scripted("2", "n"), out.append)
    assert run.vision_model == "gemma4-31b-it-mtp-mlboy" and cfg.vision_model == "gemma-cube"
    assert any("(current settings)" in line for line in out)
    assert "vision_model" not in local.read_text()                       # not made the default
    run = presets.pick(cfg, "vision", "extract-images", client_factory, scripted("2", "y"), out.append)
    assert cfg.vision_model == "gemma4-31b-it-mtp-mlboy" and 'vision_model = "gemma4-31b-it-mtp-mlboy"' in local.read_text()


def test_pick_fixes_a_renamed_model(local, cfg):
    presets.save("mtp", {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-cube"})
    run = presets.pick(cfg, "vision", "extract-images", client_factory, scripted("2", "", "n"), lambda s: None)
    assert run.vision_model == "gemma4-31b-it-mtp-mlboy" and presets.load_all()["mtp"]["model"] == run.vision_model


def test_new_preset_walkthrough(local, cfg):
    out = []
    answers = scripted("n", "mlboy.test:5000", "3", "n", "1", "", "", "n")   # host:port; text model refused, then #1
    run = presets.pick(cfg, "vision", "extract-images", client_factory, answers, out.append)
    assert run.base_url == MLBOY and run.vision_model == "gemma4-31b-it-mtp-mlboy" and run.vision_concurrency == 0
    assert presets.load_all() == {"gemma4-31b-it-mtp-mlboy": {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy",
                                                              "concurrency": 0}}
    with pytest.raises(presets.Quit):
        presets.pick(cfg, "vision", "extract-images", client_factory, scripted("q"), out.append)


def test_choose(local, cfg):
    presets.save("q", {"base_url": MLBOY, "model": "qwen", "context_tokens": 20000})
    ns = lambda **kw: argparse.Namespace(**{"preset": None, "no_pick": False, **kw})  # noqa: E731
    assert presets.choose(cfg, ns(command="digests", preset="q")).wiki_model == "qwen"
    assert presets.choose(cfg, ns(command="tag-relations", preset="q")).chat_context_tokens == 20000
    assert presets.choose(cfg, ns(command="embed", preset="q")) is cfg
    assert presets.choose(cfg, ns(command="digests")) is cfg               # pytest's stdin is not a terminal
    with pytest.raises(SystemExit):
        presets.choose(cfg, ns(command="digests", preset="nope"))


def test_settings_api_presets(local, cfg, tmp_path):
    cfg = dataclasses.replace(cfg, data_dir=tmp_path / "data")
    app = TestClient(api.create_app(cfg))
    body = {"base_url": MLBOY, "model": "gemma4-31b-it-mtp-mlboy", "concurrency": 0}
    assert app.post("/api/settings/presets/mtp", json=body).status_code == 403
    d = app.post("/api/settings/presets/mtp", json=body, headers={"X-RE2-Settings": "1"}).json()
    assert d["presets"]["mtp"]["model"] == body["model"] and d["using"]["vision"] is None
    assert app.post("/api/settings/presets/bad name", json=body, headers={"X-RE2-Settings": "1"}).status_code == 400
    change = {"services": {"vision": presets.service_values("vision", d["presets"]["mtp"])}}
    assert app.post("/api/settings", json=change, headers={"X-RE2-Settings": "1"}).json()["using"]["vision"] == "mtp"
    assert app.delete("/api/settings/presets/mtp", headers={"X-RE2-Settings": "1"}).json()["presets"] == {}
    assert app.delete("/api/settings/presets/mtp", headers={"X-RE2-Settings": "1"}).status_code == 404
