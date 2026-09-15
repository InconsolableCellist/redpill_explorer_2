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


def test_asr_windows_and_storage(env):
    conn, cfg, ids = env
    video.run(conn, cfg)
    segs = [asr.Segment(i * 10.0, i * 10.0 + 9, f"sentence number {i}") for i in range(14)]
    wins = asr.windows(segs)
    assert all(w.end - w.start <= asr.WINDOW_S for w in wins)
    assert " ".join(w.text for w in wins) == " ".join(s.text for s in segs)

    class FakeModel:
        def transcribe(self, wav, beam_size, vad_filter):
            assert wav.endswith(".wav") and beam_size == 1 and vad_filter
            info = SimpleNamespace(language="en")
            if "static" in self.current:
                return iter([]), info
            return iter([SimpleNamespace(start=0.0, end=2.5, text=" Hello there. "),
                         SimpleNamespace(start=70.0, end=72.0, text="Later words")]), info

    model = FakeModel()
    real = asr.transcribe

    def transcribe(m, c, media_id, path):
        model.current = path.name
        return real(m, c, media_id, path)

    asr.transcribe = transcribe
    try:
        # static.mp4 has no audio stream, so only the two clips with audio are selected.
        stats = asr.run(conn, cfg, model=model)
    finally:
        asr.transcribe = real
    assert (stats.selected, stats.done, stats.failed) == (2, 2, 0)
    docs = conn.execute("SELECT seq, start_s, end_s, text FROM doc WHERE kind='transcript' AND media_id=? ORDER BY seq",
                        (ids["long.mp4"],)).fetchall()
    assert [tuple(d) for d in docs] == [(0, 0.0, 2.5, "Hello there."), (1, 70.0, 72.0, "Later words")]
    assert not list((cfg.data_dir / "audio").glob("*.wav"))  # temp audio removed
    assert asr.select_jobs(conn, None) == []
