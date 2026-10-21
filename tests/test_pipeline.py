import dataclasses
import json
import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from re2 import api, cli, config, db, jobreport, runner, stages, supervisor

# Stands in for `python -m re2.cli <command>`: records itself in job_run like jobreport.Report, works for
# FAKE_SECONDS, and finishes, or is interrupted by SIGINT, or with FAKE_CRASH exits before recording.
FAKE = r'''
import json, os, socket, sqlite3, sys, time
if os.environ.get("FAKE_CRASH"):
    print("Traceback: something broke", flush=True)
    sys.exit(3)
stage = sys.argv[1].replace("-", "_")
conn = sqlite3.connect(os.environ["FAKE_DB"], timeout=10)
args = {"argv": sys.argv[1:], "base_url": os.environ.get("FAKE_URL"), "model": os.environ.get("FAKE_MODEL")}
rid = conn.execute("INSERT INTO job_run(stage, pid, host, state, args_json) VALUES (?, ?, ?, 'running', ?)",
                   (stage, os.getpid(), socket.gethostname(), json.dumps(args))).lastrowid
conn.commit()
state = "finished"
try:
    time.sleep(float(os.environ.get("FAKE_SECONDS", "0.2")))
except KeyboardInterrupt:
    state = "interrupted"
conn.execute("UPDATE job_run SET state=?, finished_at=datetime('now') WHERE id=?", (state, rid))
conn.commit()
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.jpg").write_bytes(b"x")
    cfg = dataclasses.replace(config.load(), corpus_root=corpus, data_dir=tmp_path / "data")
    conn = db.open_db(cfg)
    fake = tmp_path / "re2_fake.py"
    fake.write_text(FAKE)
    monkeypatch.setenv("FAKE_DB", str(cfg.db_path))
    sup = supervisor.Supervisor(cfg, command=[sys.executable, str(fake)])
    return cfg, conn, sup


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def latest(conn, stage):
    return conn.execute("SELECT * FROM job_run WHERE stage=? ORDER BY id DESC LIMIT 1", (stage,)).fetchone()


def test_counts_on_an_empty_archive(setup):
    cfg, conn, _ = setup
    counts = stages.counts(conn, cfg)
    assert set(counts) == set(stages.BY_NAME)
    assert all(c["error"] is None for c in counts.values()), counts
    assert counts["scan"]["pending"] == 1          # a.jpg has not been scanned
    assert counts["extract_images"]["pending"] == 0


def test_recluster_pending_counts_new_items_not_unclustered_ones(setup):
    import numpy as np

    cfg, conn, _ = setup
    conn.executemany("INSERT INTO media(id, uid, kind, status) VALUES (?, ?, 'image', ?)",
                     [(1, "a", "READY"), (2, "b", "READY"), (3, "c", "EXCLUDED")])
    conn.executemany("INSERT INTO doc(id, media_id, kind, text) VALUES (?, ?, 'caption', 't')", [(10, 1), (20, 2), (30, 3)])
    conn.commit()
    index = cfg.data_dir / "index"
    index.mkdir(parents=True, exist_ok=True)
    np.save(index / "docs_ids.npy", np.array([10, 30]))
    assert stages._recluster_pending(conn, cfg) == 1         # item 1; the excluded item 3 never counts
    np.save(index / "items_ids.npy", np.array([1]))          # the run took item 1 in but put it in no topic
    db.kv_set(conn, "cluster_run", "r1")
    assert stages._recluster_pending(conn, cfg) == 0
    np.save(index / "docs_ids.npy", np.array([10, 20, 30]))
    assert stages._recluster_pending(conn, cfg) == 1         # item 2 was embedded after the run


def test_scan_changes_sees_new_changed_and_gone_files(setup):
    cfg, conn, _ = setup
    from re2 import scan

    scan.scan(conn, cfg)
    assert stages.scan_changes(conn, cfg) == 0
    (cfg.corpus_root / "b.jpg").write_bytes(b"y")
    (cfg.corpus_root / "a.jpg").unlink()
    assert stages.scan_changes(conn, cfg) == 2


def test_report_and_progress(setup):
    cfg, conn, _ = setup
    report = jobreport.Report(cfg, "thumbs", {"model": "m"})
    seen = []
    assert runner.run_jobs(range(5), lambda j: j, lambda j, r: seen.append(r),
                           lambda j, e: None, concurrency=2, total=5)
    report.finish("finished")
    row = latest(conn, "thumbs")
    assert (row["state"], row["done"], row["selected"], json.loads(row["args_json"])) == ("finished", 5, 5, {"model": "m"})
    assert runner.progress_hook is None


def test_cli_records_the_run(setup, monkeypatch):
    cfg, conn, _ = setup
    monkeypatch.setattr(config, "load", lambda path=None: cfg)
    assert cli.main(["dates"]) == 0
    row = latest(conn, "dates")
    assert row["state"] == "finished" and row["pid"] == os.getpid()
    assert cli.main(["pipeline"]) == 0                      # not a stage: no row
    assert conn.execute("SELECT COUNT(*) FROM job_run").fetchone()[0] == 1


def test_start_stop_and_refusals(setup, monkeypatch):
    cfg, conn, sup = setup
    monkeypatch.setenv("FAKE_SECONDS", "30")
    started = sup.start("thumbs", limit=5)
    assert started["argv"] == ["thumbs", "--limit", "5"]
    assert wait_for(lambda: latest(conn, "thumbs") is not None)
    assert sup.running()["thumbs"]["state"] == "running"
    with pytest.raises(supervisor.Refused, match="already running"):
        sup.start("thumbs")
    sup.stop("thumbs")
    assert wait_for(lambda: latest(conn, "thumbs")["state"] == "interrupted")
    assert wait_for(lambda: sup._reap() == ["thumbs"] or not sup.children)
    assert "thumbs" not in sup.running()


def test_router_conflict(setup, monkeypatch):
    cfg, conn, sup = setup
    target = sup.service_target(stages.BY_NAME["extract_images"], None)
    monkeypatch.setenv("FAKE_SECONDS", "30")
    monkeypatch.setenv("FAKE_URL", target["base_url"])
    monkeypatch.setenv("FAKE_MODEL", "some-other-model")
    sup.start("digests")
    assert wait_for(lambda: latest(conn, "digests") is not None)
    with pytest.raises(supervisor.Refused, match="would unload it"):
        sup.start("extract_images")
    sup.kill("digests")
    assert latest(conn, "digests")["state"] == "killed"
    assert wait_for(lambda: sup.children["digests"].proc.poll() is not None)
    sup._reap()
    assert latest(conn, "digests")["state"] == "killed"


def test_a_run_that_dies_early_or_vanishes(setup, monkeypatch):
    cfg, conn, sup = setup
    monkeypatch.setenv("FAKE_CRASH", "1")
    sup.start("video_probe")
    assert wait_for(lambda: sup.children["video_probe"].proc.poll() is not None)
    sup._reap()
    row = latest(conn, "video_probe")
    assert row["state"] == "failed" and "something broke" in row["last_error"]
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    with conn:   # a terminal run whose process has exited without saying so
        conn.execute("INSERT INTO job_run(stage, pid, host, state) VALUES ('asr', ?, ?, 'running')", (gone.pid, sup.host))
    sup._mark_lost()
    assert latest(conn, "asr")["state"] == "lost"


def test_queued_stages_run_in_order_when_they_have_work(setup, monkeypatch):
    cfg, conn, sup = setup
    monkeypatch.setattr(stages, "counts", lambda c, cfg, names=None: {
        n: {"pending": {"scan": 1, "thumbs": 0, "video_probe": 2}.get(n, 0), "done": 0, "error": None}
        for n in (names or stages.BY_NAME)})
    sup.set_auto({"stages": {"dates": False}})
    assert sup.run_local() == ["scan", "sources", "thumbs", "video_probe", "pdfs"]
    sup._advance_chain()
    assert set(sup.children) == {"scan"}
    sup._advance_chain()                                   # scan still running: nothing else starts
    assert set(sup.children) == {"scan"}
    assert wait_for(lambda: sup.children["scan"].proc.poll() is not None)
    sup._reap()
    sup._advance_chain()                                   # thumbs has nothing to do; video_probe does
    assert set(sup.children) == {"video_probe"}
    assert wait_for(lambda: sup.children["video_probe"].proc.poll() is not None)
    sup._reap()
    sup._advance_chain()
    assert not sup.children and sup.chain == []


def test_pipeline_api(setup, monkeypatch):
    cfg, conn, sup = setup
    app = TestClient(api.create_app(cfg))
    app.app.state.supervisor.command = sup.command
    d = app.get("/api/pipeline").json()
    assert [s["name"] for s in d["stages"]] == [s.name for s in stages.STAGES]
    assert d["auto"]["enabled"] is True and d["stages"][0]["pending"] == 1
    assert app.post("/api/pipeline/thumbs/start", json={}).status_code == 403
    assert app.post("/api/pipeline/nope/start", json={}, headers={"X-RE2-Settings": "1"}).status_code == 404
    assert app.post("/api/pipeline/thumbs/stop", headers={"X-RE2-Settings": "1"}).status_code == 409
    r = app.post("/api/pipeline-auto", json={"interval_min": 5, "stages": {"embed": True}}, headers={"X-RE2-Settings": "1"})
    assert r.json()["interval_min"] == 5 and r.json()["stages"]["embed"] is True
    assert app.get("/api/pipeline/thumbs/failures").json() == {"failures": []}
    started = app.post("/api/pipeline/thumbs/start", json={"limit": 2}, headers={"X-RE2-Settings": "1"}).json()
    assert started["argv"] == ["thumbs", "--limit", "2"]
    assert wait_for(lambda: (latest(conn, "thumbs") or {"state": ""})["state"] == "finished")
    assert app.get("/api/pipeline/thumbs/log").json()["lines"]   # the console log has the start line
    sup2 = app.app.state.supervisor
    assert wait_for(lambda: sup2.children["thumbs"].proc.poll() is not None)
    assert sup2._reap() == ["thumbs"]
