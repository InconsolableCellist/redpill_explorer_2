import httpx

from re2 import routerguard, runner, stages


def router(states: dict[str, str]):
    """An httpx client factory answering /v1/models like llama.cpp's router with these model states."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": m, "status": {"value": s, "args": []}} for m, s in states.items()]})
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


URL = "https://mlboy:5000/v1"


def test_may_start_only_on_its_own_model_or_an_empty_router():
    assert routerguard.may_start(URL, "gemma", router({"gemma": "loaded", "qwen": "unloaded"})) is None
    assert routerguard.may_start(URL, "gemma", router({"gemma": "unloaded", "qwen": "unloaded"})) is None
    assert "qwen" in routerguard.may_start(URL, "gemma", router({"gemma": "unloaded", "qwen": "loaded"}))
    assert "qwen" in routerguard.may_start(URL, "gemma", router({"gemma": "unloaded", "qwen": "loading"}))
    plain = lambda: httpx.Client(transport=httpx.MockTransport(  # noqa: E731 - a server that is not a router
        lambda r: httpx.Response(200, json={"data": [{"id": "qwen"}]})))
    assert routerguard.may_start(URL, "gemma", plain) is None


def test_guard_stops_a_run_once_another_model_takes_the_router(monkeypatch):
    states = {"gemma": "loaded", "qwen": "unloaded"}
    guard = routerguard.Guard(URL, "gemma", router(states))
    monkeypatch.setattr(routerguard, "CHECK_EVERY_S", 0)
    done = []

    def work(n):
        if n == 2:
            states.update(gemma="unloaded", qwen="loading")   # someone opens a chat with qwen
        return n

    monkeypatch.setattr(runner, "guard", guard)
    monkeypatch.setattr(runner, "interrupted", False)
    monkeypatch.setattr(runner, "stopped_reason", None)
    finished = runner.run_jobs(range(10), work, lambda j, r: done.append(r), lambda j, e: None, 1)
    assert not finished and runner.interrupted
    assert runner.stopped_reason == "qwen took the router from gemma"
    assert {0, 1, 2} <= set(done) and len(done) < 10                   # the jobs in flight finish, no more


def test_a_queued_model_stage_waits_instead_of_unloading(monkeypatch, tmp_path):
    import dataclasses
    import sys

    from re2 import config, db, supervisor

    cfg = dataclasses.replace(config.load(), data_dir=tmp_path / "data")
    db.open_db(cfg).close()
    sup = supervisor.Supervisor(cfg, command=[sys.executable, "-c", "pass"])
    monkeypatch.setattr(stages, "counts", lambda c, cfg, names=None: {
        n: {"pending": 5 if n == "extract_images" else 0, "done": 0, "error": None} for n in (names or stages.BY_NAME)})
    monkeypatch.setattr(routerguard, "may_start", lambda url, model: "the router has qwen loaded")
    sup.set_auto({"stages": {"extract_images": True}})
    sup.run_local()
    sup._advance_chain()
    assert not sup.children and sup.waiting["extract_images"] == "the router has qwen loaded"
    assert next(s for s in sup.status()["stages"] if s["name"] == "extract_images")["waiting"]
