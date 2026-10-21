"""A stage run's row in job_run: live counts and a heartbeat, so the Pipeline page sees every run,
whether it was started there or in a terminal."""
from __future__ import annotations

import json
import os
import socket
import threading
import time

from re2 import db, runner

HEARTBEAT_S = 20.0


class Report:
    def __init__(self, cfg, stage: str, args: dict):
        # Its own connection: the stage's connection may be in the middle of a transaction.
        self.conn = db.connect(cfg.db_path)
        self.started = time.monotonic()
        with self.conn:
            self.id = self.conn.execute(
                "INSERT INTO job_run(stage, pid, host, state, args_json, heartbeat_at) "
                "VALUES (?, ?, ?, 'running', ?, datetime('now'))",
                (stage, os.getpid(), socket.gethostname(), json.dumps(args))).lastrowid
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()
        runner.progress_hook = self.progress

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self._update("heartbeat_at=datetime('now')")

    def _update(self, sets: str, params: tuple = ()) -> None:
        with self._lock:
            try:
                with self.conn:
                    self.conn.execute(f"UPDATE job_run SET {sets} WHERE id=?", (*params, self.id))
            except Exception:  # noqa: BLE001 - a busy database must never stop the stage itself
                pass

    def progress(self, done: int, failed: int, total: int | None) -> None:
        minutes = max(time.monotonic() - self.started, 1.0) / 60
        self._update("done=?, failed=?, selected=coalesce(?, selected), rate_per_min=?, heartbeat_at=datetime('now')",
                     (done, failed, total, round(done / minutes, 2)))

    def finish(self, state: str, error: str | None = None) -> None:
        self._stop.set()
        runner.progress_hook = None
        self._update("state=?, last_error=coalesce(?, last_error), finished_at=datetime('now'), "
                     "heartbeat_at=datetime('now')", (state, error[:2000] if error else None))
        self.conn.close()
