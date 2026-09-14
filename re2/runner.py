"""Interrupt-safe worker pool for long, resumable batch jobs.

Ctrl-C once: stop starting new work, save what is already in flight, exit.
Ctrl-C twice: abandon in-flight requests immediately. Every finished item is
committed as it completes, so re-running the command resumes where it stopped.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

from tqdm import tqdm

log = logging.getLogger(__name__)

DOUBLE_PRESS_S = 1.0  # a second Ctrl-C sooner than this is treated as the same keypress

J = TypeVar("J")
R = TypeVar("R")


def run_jobs(
    jobs: Iterable[J],
    work: Callable[[J], R],
    on_result: Callable[[J, R], None],
    on_error: Callable[[J, BaseException], None],
    concurrency: int,
    total: int | None = None,
    desc: str = "jobs",
) -> bool:
    """Run work(job) on a thread pool; on_result/on_error run on the calling thread.

    Returns True if every job was attempted, False if interrupted.
    """
    pool = ThreadPoolExecutor(concurrency)
    it = iter(jobs)
    pending: dict[Future, J] = {}
    bar = tqdm(total=total, desc=desc, unit="item", dynamic_ncols=True)
    exhausted = False

    def collect(done: Iterable[Future]) -> None:
        for fut in done:
            job = pending.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - any per-item failure is recorded, never fatal
                on_error(job, exc)
            else:
                on_result(job, result)
            bar.update(1)

    try:
        while True:
            while not exhausted and len(pending) < concurrency * 2:
                job = next(it, None)
                if job is None:
                    exhausted = True
                    break
                pending[pool.submit(work, job)] = job
            if not pending:
                return True
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            collect(done)
    except KeyboardInterrupt:
        in_flight = [f for f in pending if not f.cancel()]
        for f in list(pending):
            if f.cancelled():
                pending.pop(f)
        bar.clear()
        print(f"\nInterrupted: saving {len(in_flight)} in-flight item(s). "
              "Press Ctrl-C again to abandon them.", file=sys.stderr, flush=True)
        first_interrupt = time.monotonic()
        while pending:
            try:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                collect(done)
            except KeyboardInterrupt:
                # One keypress can arrive twice (terminal and `uv run` both deliver it).
                if time.monotonic() - first_interrupt < DOUBLE_PRESS_S:
                    continue
                print("Abandoned in-flight items; they will be retried next run.", file=sys.stderr, flush=True)
                bar.close()
                os._exit(130)  # worker threads may be blocked in HTTP calls; do not wait for them
        return False
    finally:
        bar.close()
        pool.shutdown(wait=False, cancel_futures=True)
