"""Starts, stops and watches pipeline stages for the dashboard, and runs the local ones on its own.

Each stage runs as its own `python -m re2.cli <command>` process (start_new_session, so restarting the
server never stops it); the process records itself in job_run (jobreport.Report), which is how runs
started in a terminal show up too. A thread in `re2 serve` reaps finished runs, marks runs whose
process vanished as lost, refreshes the pending counts, and every few minutes looks for new files and
runs the stages marked auto that have work, one after another.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from re2 import config, db, presets, stages

log = logging.getLogger(__name__)

COUNT_EVERY_S = 30.0
SCAN_COUNT_EVERY_S = 300.0   # the new-file check walks the whole corpus (~5 s on the NFS mount)
TICK_S = 3.0
LOST_AFTER_S = 120        # a run on another host with no heartbeat for this long is lost
AUTO_KEY = "pipeline_auto"
SERVICE_STAGES = {"vision", "wiki", "chat"}


def default_auto() -> dict:
    return {"enabled": True, "interval_min": 10, "stages": {s.name: s.auto for s in stages.STAGES}}


def pid_alive(pid: int | None) -> bool:
    """True when pid is a running re2 process (a reused pid running something else is not)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True   # no /proc: trust the signal check
    return b"re2" in cmdline


@dataclass
class Child:
    proc: subprocess.Popen
    stage: str
    after_id: int          # job_run ids above this belong to this run
    killed: bool = False
    started: float = field(default_factory=time.monotonic)


class Refused(Exception):
    pass


class Supervisor:
    def __init__(self, cfg, command: list[str] | None = None):
        self.cfg = cfg
        self.command = command or [sys.executable, "-m", "re2.cli"]   # tests put a stand-in here
        self.host = socket.gethostname()
        self.children: dict[str, Child] = {}
        self.lock = threading.RLock()
        self.counts: dict[str, dict] = {}
        self.counted_at = 0.0
        self.scan_counted_at = 0.0
        self.chain: list[str] = []         # auto or "run local" stages still to consider, in order
        self.chain_current: str | None = None
        self.waiting: dict[str, str] = {}  # queued model stages that did not start on their own, and why
        self.next_auto = time.monotonic() + 60
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Settings ──────────────────────────────────────────────────────────

    def conn(self):
        return db.connect(self.cfg.db_path)

    def auto(self) -> dict:
        conn = self.conn()
        try:
            saved = json.loads(db.kv_get(conn, AUTO_KEY) or "{}")
        finally:
            conn.close()
        out = default_auto()
        out.update({k: saved[k] for k in ("enabled", "interval_min") if k in saved})
        out["stages"].update({k: bool(v) for k, v in (saved.get("stages") or {}).items() if k in stages.BY_NAME})
        return out

    def set_auto(self, changes: dict) -> dict:
        cur = self.auto()
        if "enabled" in changes:
            cur["enabled"] = bool(changes["enabled"])
        if "interval_min" in changes:
            cur["interval_min"] = max(1, min(24 * 60, int(changes["interval_min"])))
        for name, on in (changes.get("stages") or {}).items():
            if name not in stages.BY_NAME:
                raise ValueError(f"unknown stage {name!r}")
            cur["stages"][name] = bool(on)
        conn = self.conn()
        try:
            with conn:
                db.kv_set(conn, AUTO_KEY, json.dumps(cur))
        finally:
            conn.close()
        with self.lock:
            self.next_auto = min(self.next_auto, time.monotonic() + cur["interval_min"] * 60)
        return cur

    # ── Runs ──────────────────────────────────────────────────────────────

    def running(self) -> dict[str, dict]:
        """Stage -> its live run: a child still starting, or a job_run row whose process is alive."""
        out: dict[str, dict] = {}
        conn = self.conn()
        try:
            for r in conn.execute("SELECT * FROM job_run WHERE state='running' ORDER BY id"):
                d = dict(r)
                d["args"] = json.loads(d.pop("args_json") or "{}")
                if d["host"] != self.host or pid_alive(d["pid"]):
                    out[d["stage"]] = d
        finally:
            conn.close()
        with self.lock:
            for name, ch in self.children.items():
                if ch.proc.poll() is None and name not in out:
                    out[name] = {"stage": name, "state": "starting", "pid": ch.proc.pid, "args": {}}
        return out

    def service_target(self, stage: stages.Stage, preset_name: str | None) -> dict:
        """The base_url and model this stage would use (fresh config, so saved Settings count)."""
        cfg = config.load()
        if stage.service in SERVICE_STAGES:
            if preset_name:
                found = presets.load_all().get(preset_name)
                if found is None:
                    raise Refused(f"no preset {preset_name!r}")
                cfg = presets.apply(cfg, stage.service, found)
            cur = presets.of_service(cfg, stage.service)
            return {"base_url": cur["base_url"].rstrip("/"), "model": cur["model"]}
        if stage.service == "embed":
            return {"base_url": cfg.embed_url, "model": cfg.embed_model}
        if stage.service == "asr":
            return {"model": cfg.asr_model}
        return {}

    def start(self, name: str, preset: str | None = None, limit: int | None = None,
              concurrency: int | None = None) -> dict:
        stage = stages.BY_NAME.get(name)
        if stage is None:
            raise Refused(f"unknown stage {name!r}")
        live = self.running()
        if name in live:
            raise Refused(f"{stage.label} is already running")
        target = self.service_target(stage, preset if "preset" in stage.options else None)
        if target.get("base_url"):
            for other, run in live.items():
                a = run.get("args") or {}
                if a.get("base_url") == target["base_url"] and a.get("model") and a.get("model") != target["model"]:
                    raise Refused(f"{stages.BY_NAME[other].label if other in stages.BY_NAME else other} is using "
                                  f"{a['model']} on that server; starting {target['model']} there would unload it")
        argv = [*self.command, stage.command]
        if stage.command in presets.COMMAND_SERVICE:
            argv.append("--no-pick")
            if preset and "preset" in stage.options:
                argv += ["--preset", preset]
        if limit and "limit" in stage.options:
            argv += ["--limit", str(int(limit))]
        if concurrency and "concurrency" in stage.options:
            argv += ["--concurrency", str(int(concurrency))]
        log_dir = Path(self.cfg.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        conn = self.conn()
        try:
            after_id = conn.execute("SELECT coalesce(MAX(id), 0) FROM job_run").fetchone()[0]
        finally:
            conn.close()
        env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
        with open(log_dir / f"{stage.command}.console.log", "ab") as out:
            out.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} re2 {' '.join(argv[len(self.command):])}\n".encode())
            out.flush()
            proc = subprocess.Popen(argv, cwd=config.REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True, env=env)
        with self.lock:
            self.children[name] = Child(proc, name, after_id)
        log.info("pipeline: started %s (pid %d): %s", name, proc.pid, " ".join(argv[len(self.command):]))
        return {"stage": name, "pid": proc.pid, "argv": argv[len(self.command):], **target}

    def _signal(self, name: str, sig: int) -> int:
        run = self.running().get(name)
        if run is None:
            raise Refused(f"{name} is not running")
        if run.get("host") not in (None, self.host):
            raise Refused(f"{name} runs on {run['host']}; stop it there")
        os.kill(run["pid"], sig)
        return run["pid"]

    def stop(self, name: str) -> dict:
        """Ctrl-C: the run finishes the items in flight, saves them and exits."""
        self.chain = [s for s in self.chain if s != name]
        return {"stage": name, "pid": self._signal(name, signal.SIGINT), "signal": "SIGINT"}

    def kill(self, name: str) -> dict:
        """Ends the run at once; items in flight are redone next time."""
        self.chain = [s for s in self.chain if s != name]
        pid = self._signal(name, signal.SIGTERM)
        with self.lock:
            if name in self.children:
                self.children[name].killed = True
        conn = self.conn()
        try:
            with conn:
                conn.execute("UPDATE job_run SET state='killed', finished_at=datetime('now') "
                             "WHERE stage=? AND pid=? AND state='running'", (name, pid))
        finally:
            conn.close()
        return {"stage": name, "pid": pid, "signal": "SIGTERM"}

    def retry(self, name: str) -> int:
        """Forget the stage's failures so its next run tries those items again."""
        stage = stages.BY_NAME[name]
        if not stage.failure_stages:
            return 0
        conn = self.conn()
        try:
            with conn:
                n = conn.execute(f"DELETE FROM failure WHERE stage IN ({','.join('?' * len(stage.failure_stages))})",
                                 stage.failure_stages).rowcount
        finally:
            conn.close()
        self.refresh_counts([name])
        return n

    def run_local(self) -> list[str]:
        """Queue every auto stage (in pipeline order); each starts in turn if it has work."""
        auto = self.auto()
        with self.lock:
            self.chain = [s.name for s in stages.STAGES if auto["stages"].get(s.name)]
            return list(self.chain)

    # ── Watching ──────────────────────────────────────────────────────────

    def refresh_counts(self, names: list[str] | None = None) -> dict[str, dict]:
        conn = self.conn()
        try:
            fresh = stages.counts(conn, self.cfg, names)
        finally:
            conn.close()
        with self.lock:
            self.counts.update(fresh)
            if names is None or len(names) > 1:
                self.counted_at = time.time()
            if names is None or "scan" in names:
                self.scan_counted_at = time.time()
        return fresh

    def _reap(self) -> list[str]:
        finished = []
        with self.lock:
            done = [(n, ch) for n, ch in self.children.items() if ch.proc.poll() is not None]
            for n, _ in done:
                del self.children[n]
        for name, ch in done:
            finished.append(name)
            conn = self.conn()
            try:
                row = conn.execute("SELECT id, state FROM job_run WHERE stage=? AND pid=? AND id>? ORDER BY id DESC",
                                   (name, ch.proc.pid, ch.after_id)).fetchone()
                with conn:
                    if row is None:   # it died before recording itself: bad arguments, an import error
                        tail = _tail(Path(self.cfg.data_dir) / "logs" / f"{stages.BY_NAME[name].command}.console.log", 8)
                        conn.execute("INSERT INTO job_run(stage, pid, host, state, last_error, finished_at) "
                                     "VALUES (?, ?, ?, 'failed', ?, datetime('now'))",
                                     (name, ch.proc.pid, self.host, f"exit {ch.proc.returncode}: " + "\n".join(tail)))
                    elif row["state"] == "running":
                        conn.execute("UPDATE job_run SET state=?, finished_at=datetime('now') WHERE id=?",
                                     ("killed" if ch.killed else "lost", row["id"]))
            finally:
                conn.close()
        return finished

    def _mark_lost(self) -> None:
        conn = self.conn()
        try:
            with self.lock:
                ours = {ch.proc.pid for ch in self.children.values()}
            for r in conn.execute("SELECT id, pid, host, heartbeat_at FROM job_run WHERE state='running'").fetchall():
                if r["pid"] in ours:
                    continue
                gone = (not pid_alive(r["pid"])) if r["host"] == self.host else conn.execute(
                    "SELECT coalesce(?, '') < datetime('now', ?)", (r["heartbeat_at"], f"-{LOST_AFTER_S} seconds")).fetchone()[0]
                if gone:
                    with conn:
                        conn.execute("UPDATE job_run SET state='lost', finished_at=datetime('now') WHERE id=?", (r["id"],))
        finally:
            conn.close()

    def auto_blocker(self, stage: stages.Stage) -> str | None:
        """Why a queued model stage should not start on its own now (None: go ahead). Started by hand, it
        is not asked: the Pipeline page already says what a start would unload."""
        if stage.service == "asr":
            from re2 import asr

            try:
                asr.pick_card(config.load())
            except asr.Busy as exc:
                return str(exc)
            except (OSError, subprocess.SubprocessError) as exc:
                return f"nvidia-smi: {exc}"
        elif stage.service in SERVICE_STAGES:
            from re2 import routerguard

            target = self.service_target(stage, None)
            return routerguard.may_start(target["base_url"], target["model"])
        return None

    def _advance_chain(self) -> None:
        with self.lock:
            if not self.chain:
                return
            if self.chain_current and self.chain_current in self.running():
                return   # one queued stage at a time, in order
            while self.chain:
                name = self.chain.pop(0)
                pending = self.refresh_counts([name]).get(name, {}).get("pending")
                if pending:
                    blocker = self.auto_blocker(stages.BY_NAME[name])
                    if blocker:
                        log.info("pipeline: queued %s waits: %s", name, blocker)
                        self.waiting[name] = blocker
                        continue
                    self.waiting.pop(name, None)
                    try:
                        self.start(name)
                    except (Refused, OSError) as exc:
                        log.warning("pipeline: queued %s not started: %s", name, exc)
                        continue
                    self.chain_current = name
                    return
            self.chain_current = None

    def tick(self) -> None:
        finished = self._reap()
        self._mark_lost()
        if finished:
            self.refresh_counts()
        elif time.time() - self.counted_at > COUNT_EVERY_S:
            self.refresh_counts([s.name for s in stages.STAGES if s.name != "scan"])
        if time.time() - self.scan_counted_at > SCAN_COUNT_EVERY_S:
            self.refresh_counts(["scan"])
        auto = self.auto()
        if auto["enabled"] and time.monotonic() >= self.next_auto:
            self.next_auto = time.monotonic() + auto["interval_min"] * 60
            if not self.chain:
                self.run_local()
        self._advance_chain()

    def _loop(self) -> None:
        self.refresh_counts()
        while not self._stop.wait(TICK_S):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the watcher must outlive any one bad tick
                log.exception("pipeline: tick failed")

    def start_thread(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    # ── What the page shows ───────────────────────────────────────────────

    def status(self) -> dict:
        if not self.counts:
            self.refresh_counts()
        live = self.running()
        conn = self.conn()
        try:
            history = stages.runs(conn)
            rows = []
            for s in stages.STAGES:
                c = self.counts.get(s.name, {})
                run = live.get(s.name) or (history.get(s.name) or {}).get("latest")
                rows.append({
                    "name": s.name, "label": s.label, "command": s.command, "unit": s.unit, "service": s.service,
                    "local": s.local, "options": list(s.options), "help": s.help, "after": list(s.after),
                    "pending": c.get("pending"), "done": c.get("done"), "count_error": c.get("error"),
                    "running": s.name in live, "run": run,
                    "last_finished": (history.get(s.name) or {}).get("last_finished"),
                    "failures": stages.failures(conn, s), "queued": s.name in self.chain,
                    "waiting": self.waiting.get(s.name),
                })
        finally:
            conn.close()
        auto = self.auto()
        return {"stages": rows, "auto": {**auto, "next_in_s": max(0, round(self.next_auto - time.monotonic()))},
                "counted_at": self.counted_at, "presets": presets.load_all(),
                "services": config.services(config.load())}


def _tail(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64 * 1024))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    return [line.rstrip("\r") for line in lines[-n:]]


def log_tail(cfg, name: str, n: int = 40) -> list[str]:
    """The last lines of the stage's log (the console log when it is newer, e.g. a run that never started)."""
    stage = stages.BY_NAME[name]
    logs = Path(cfg.data_dir) / "logs"
    main, console = logs / f"{stage.command}.log", logs / f"{stage.command}.console.log"
    newest = max((p for p in (main, console) if p.exists()), key=lambda p: p.stat().st_mtime, default=None)
    return _tail(newest, n) if newest else []


def failure_list(cfg, name: str, limit: int = 200) -> list[dict]:
    stage = stages.BY_NAME[name]
    if not stage.failure_stages:
        return []
    conn = db.connect(cfg.db_path)
    try:
        marks = ",".join("?" * len(stage.failure_stages))
        return [dict(r) for r in conn.execute(
            f"""SELECT f.stage, f.media_id, f.doc_id, f.attempts, f.kind, f.error, f.updated_at,
                       (SELECT p.rel_path FROM media_path p WHERE p.media_id=f.media_id ORDER BY p.id LIMIT 1) AS rel_path
                FROM failure f WHERE f.stage IN ({marks}) ORDER BY f.updated_at DESC LIMIT ?""",
            (*stage.failure_stages, limit))]
    finally:
        conn.close()
