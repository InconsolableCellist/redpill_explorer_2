import threading
import time

from re2 import runner


def test_run_jobs_collects_results_and_errors():
    results, errors = [], []

    def work(n):
        if n == 3:
            raise ValueError("three")
        return n * 10

    ok = runner.run_jobs(range(6), work, lambda j, r: results.append(r),
                         lambda j, e: errors.append(j), concurrency=3, total=6)
    assert ok and sorted(results) == [0, 10, 20, 40, 50] and errors == [3]


def test_interrupt_saves_in_flight_and_stops_submitting():
    started, saved = [], []
    lock = threading.Lock()

    def work(n):
        with lock:
            started.append(n)
        time.sleep(0.05 if n == 0 else 0.3)
        return n

    def on_result(job, result):
        saved.append(job)
        if job == 0:
            raise KeyboardInterrupt  # simulates Ctrl-C arriving while collecting

    ok = runner.run_jobs(range(100), work, on_result, lambda j, e: None, concurrency=2, total=100)
    assert ok is False
    assert 0 in saved and len(saved) > 1           # in-flight work was saved
    assert set(saved) <= set(started)
    assert len(started) <= 5                         # no new work after the interrupt
