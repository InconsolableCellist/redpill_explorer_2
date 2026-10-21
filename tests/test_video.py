import subprocess
from types import SimpleNamespace

import pytest

from re2 import asr, db, scan, video


def make_video(path, seconds, audio=True, color_change=True):
    """Synthetic clip: test pattern (changes over time) plus an optional sine tone."""
    src = f"testsrc2=size=320x240:rate=10:duration={seconds}" if color_change else \
        f"color=c=gray:size=320x240:rate=10:duration={seconds}"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    make_video(root / "long.mp4", 40)
    make_video(root / "static.mp4", 40, audio=False, color_change=False)
    make_video(root / "short.mp4", 3)
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, data_dir=tmp_path / "data", thumbs_dir=tmp_path / "thumbs")
    scan.scan(conn, cfg)
    ids = {r[0]: r[1] for r in conn.execute("SELECT rel_path, media_id FROM media_path")}
    return conn, cfg, ids


def test_frame_times():
    assert video.frame_times(3) == [1.0]
    assert video.frame_times(1) == [0.5]
    assert video.frame_times(40) == [1.0, 15.0, 30.0]
    assert len(video.frame_times(3600)) == video.MAX_FRAMES


def test_probe_and_keyframes(env):
    conn, cfg, ids = env
    stats = video.run(conn, cfg)
    assert (stats.done, stats.failed) == (3, 0)
    rows = {r["id"]: r for r in conn.execute("SELECT id, duration, width, has_audio, thumb_path FROM media")}
    long, static = rows[ids["long.mp4"]], rows[ids["static.mp4"]]
    assert round(long["duration"]) == 40 and long["width"] == 320 and long["has_audio"] == 1
    assert static["has_audio"] == 0
    assert long["thumb_path"] is not None
    frames = lambda mid: conn.execute("SELECT seq, t_s FROM keyframe WHERE media_id=? ORDER BY seq", (mid,)).fetchall()  # noqa: E731
    assert [tuple(f) for f in frames(ids["long.mp4"])] == [(0, 1.0), (1, 15.0), (2, 30.0)]
    assert len(frames(ids["static.mp4"])) == 1  # identical frames are dropped by phash
    assert video.run(conn, cfg).selected == 0


class FakeWorker:
    """Stands in for asr.Worker: long.mp4 speaks English, short.mp4 is in a language Parakeet skips."""

    def __init__(self):
        self.files = []

    def transcribe(self, wav):
        assert wav.suffix == ".wav" and wav.exists()
        self.files.append(wav.name)
        if len(self.files) == 2:
            return "ja", 0.99, [asr.Segment(0.0, 2.0, "So partying a meeting")]
        return "en", 0.95, [asr.Segment(0.0, 2.5, "Hello there."), asr.Segment(70.0, 72.0, "Later words")]


def asr_cfg(cfg):
    return SimpleNamespace(**vars(cfg), asr_model="nvidia/parakeet-tdt-0.6b-v3", asr_device="cuda",
                           asr_need_gb=4.5, asr_floor_gb=1.0, base_url="http://127.0.0.1:5000/v1",
                           chat_base_url="http://127.0.0.1:5000/v1", wiki_base_url="http://127.0.0.1:5000/v1")


def test_asr_windows_and_storage(env):
    conn, cfg, ids = env
    cfg = asr_cfg(cfg)
    video.run(conn, cfg)
    segs = [asr.Segment(i * 10.0, i * 10.0 + 9, f"sentence number {i}") for i in range(14)]
    wins = asr.windows(segs)
    assert all(w.end - w.start <= asr.WINDOW_S for w in wins)
    assert " ".join(w.text for w in wins) == " ".join(s.text for s in segs)

    # static.mp4 has no audio stream, so only the two clips with audio are selected.
    stats = asr.run(conn, cfg, worker=FakeWorker())
    assert (stats.selected, stats.done, stats.other_language, stats.failed) == (2, 1, 1, 0)
    docs = conn.execute("SELECT seq, start_s, end_s, text, model FROM doc WHERE kind='transcript' AND media_id=? ORDER BY seq",
                        (ids["long.mp4"],)).fetchall()
    assert [tuple(d) for d in docs] == [(0, 0.0, 2.5, "Hello there.", "parakeet-tdt-0.6b-v3"),
                                        (1, 70.0, 72.0, "Later words", "parakeet-tdt-0.6b-v3")]
    skipped = conn.execute("SELECT error FROM failure WHERE stage='asr' AND media_id=?", (ids["short.mp4"],)).fetchone()
    assert skipped[0] == "language:ja"
    assert conn.execute("SELECT language FROM media_meta WHERE media_id=?", (ids["short.mp4"],)).fetchone()[0] == "ja"
    assert not list((cfg.data_dir / "audio").glob("*.wav"))  # temp audio removed
    assert asr.select_jobs(conn, None) == []                 # the skipped language is not selected again


def test_asr_redo_replaces_an_older_transcript(env):
    conn, cfg, ids = env
    cfg = asr_cfg(cfg)
    video.run(conn, cfg)
    long = ids["long.mp4"]
    asr.run(conn, cfg, worker=FakeWorker())
    conn.execute("UPDATE media_meta SET language='cy' WHERE media_id=?", (long,))   # as Whisper recorded it
    conn.commit()
    assert asr.redo_candidates(conn, "cy") == [long]
    assert asr.mark_redo(conn, asr.redo_candidates(conn, "cy") + asr.redo_candidates(conn, "ja")) == 2
    assert [r["id"] for r in asr.select_jobs(conn, None)] == sorted([long, ids["short.mp4"]])

    class Silent:
        def transcribe(self, wav):
            return "war", 0.3, []
    stats = asr.run(conn, cfg, worker=Silent())
    assert stats.no_speech == 2 and asr.redo_ids(conn) == set()
    assert not conn.execute("SELECT 1 FROM doc WHERE kind='transcript' AND media_id=?", (long,)).fetchone()
    assert conn.execute("SELECT language FROM media_meta WHERE media_id=?", (long,)).fetchone()[0] is None
    assert asr.select_jobs(conn, None) == []


def test_asr_stops_to_make_room(env, monkeypatch):
    conn, cfg, ids = env
    cfg = asr_cfg(cfg)
    video.run(conn, cfg)
    reasons = iter([None, "GPU 1 is down to 0.6 GB free"])
    monkeypatch.setattr(asr, "why_stop", lambda c, card: next(reasons))
    monkeypatch.setattr(asr.runner, "interrupted", False)
    stats = asr.run(conn, cfg, worker=FakeWorker())
    assert stats.done == 1 and stats.stopped == "GPU 1 is down to 0.6 GB free"
    assert asr.runner.interrupted                            # recorded as interrupted: resumable
    assert len(asr.select_jobs(conn, None)) == 1             # the other video waits for the next run


def test_asr_judge_language():
    assert asr.judge_language("ja", 0.99) == ("ja", True)       # sure, and not a Parakeet language
    assert asr.judge_language("iw", 1.0) == ("he", True)        # AmberNet's code for Hebrew
    assert asr.judge_language("war", 0.4) == (None, False)      # a guess: transcribe, record nothing
    assert asr.judge_language("cy", 0.99) == (None, False)      # British English, as AmberNet hears it
    assert asr.judge_language("de", 0.97) == ("de", False)
    assert asr.judge_language(None, 0.0) == (None, False)


def test_asr_pick_card(monkeypatch):
    cfg = asr_cfg(SimpleNamespace())
    monkeypatch.setattr(asr, "router_loading", lambda c: [])
    monkeypatch.setattr(asr, "gpu_free_gb", lambda: {0: 6.0, 1: 7.0})
    assert asr.pick_card(cfg) == 1
    cfg.asr_device = "cuda:0"
    assert asr.pick_card(cfg) == 0
    cfg.asr_need_gb = 6.5
    with pytest.raises(asr.Busy, match="GPU 0 has 6.0 GB free"):
        asr.pick_card(cfg)
    cfg.asr_device = "cpu"
    assert asr.pick_card(cfg) is None
    cfg.asr_device = "cuda"
    monkeypatch.setattr(asr, "router_loading", lambda c: ["gemma4-31b"])
    with pytest.raises(asr.Busy, match="loading gemma4-31b"):
        asr.pick_card(cfg)
    monkeypatch.setattr(asr, "router_loading", lambda c: [])
    monkeypatch.setattr(asr, "gpu_free_gb", lambda: {1: 0.5})
    assert asr.why_stop(cfg, 1).startswith("GPU 1 is down to 0.5 GB")
    assert asr.why_stop(cfg, None) is None


def test_asr_local_routers(monkeypatch):
    cfg = asr_cfg(SimpleNamespace())
    monkeypatch.setattr(asr.socket, "gethostname", lambda: "mlboy")
    cfg.base_url = "https://mlboy.tail99c274.ts.net:5000/v1"
    cfg.chat_base_url = "https://cube.tail99c274.ts.net:5000/v1"
    assert asr._local_routers(cfg) == ["http://127.0.0.1:5000", "https://mlboy.tail99c274.ts.net:5000"]


def test_asr_worker_cuts():
    import numpy as np

    from re2 import asr_worker

    sr = asr_worker.SR
    audio = np.ones(sr * 400, dtype=np.float32)
    audio[sr * 128: sr * 128 + sr // 10] = 0            # the quiet moment near the 120 s boundary
    b = asr_worker.cuts(audio, 120)
    assert b[0] == 0 and b[-1] == len(audio) and abs(b[1] - sr * 128) <= sr // 10
    assert all(z - a <= (120 + asr_worker.CUT_SEARCH_S) * sr for a, z in zip(b, b[1:]))
    assert asr_worker.cuts(audio[: sr * 60], 120) == [0, sr * 60]
