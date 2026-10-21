"""Speech-to-text worker: Parakeet (NeMo) plus AmberNet language ID, driven by re2/asr.py.

Runs under the Python env named in [asr] python (one with nemo_toolkit[asr]), not RE2's, so it imports
nothing from re2. Reads one JSON request per line on stdin, {"wav": path}, and answers one JSON line on
stdout: {"language": "en", "language_p": 0.97, "segments": [[start_s, end_s, text], ...]}. Deciding what
to do with a language is re2/asr.py's job. {"error": ...} reports a failed request; the worker carries on. The first line it writes is {"ready": ...}. EOF on stdin ends it, so it never outlives
the re2 process that started it.

Measured on a V100 (2026-10-21): the fp16 encoder with 120 s chunks peaks at ~2.7 GB for the whole process
and transcribes ~140x realtime. VRAM grows with the chunk, so long audio is cut at its quietest moment near
each chunk boundary rather than passed whole.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SR = 16000
LID_MODEL = "langid_ambernet"
LID_WINDOW_S = 30
CUT_SEARCH_S = 15          # look this far either side of a chunk boundary for the quietest 0.1 s
FRAME = SR // 10


def cuts(audio, chunk_s: float) -> list[int]:
    """Chunk boundaries in samples: about every chunk_s, each moved to the quietest 0.1 s near it."""
    import numpy as np

    out, pos, n = [0], 0, len(audio)
    while n - pos > (chunk_s + CUT_SEARCH_S) * SR:
        lo, hi = pos + int((chunk_s - CUT_SEARCH_S) * SR), pos + int((chunk_s + CUT_SEARCH_S) * SR)
        frames = audio[lo:hi][: (hi - lo) // FRAME * FRAME].reshape(-1, FRAME)
        pos = lo + int(np.argmin((frames ** 2).mean(axis=1))) * FRAME + FRAME // 2
        out.append(pos)
    return out + [n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-s", type=float, default=120.0)
    args = ap.parse_args()

    # NeMo and its dependencies print to stdout; keep the real stdout for answers and send the rest to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def send(obj) -> None:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    import soundfile as sf
    import torch
    import nemo.collections.asr as nemo_asr

    t = time.monotonic()
    cuda = args.device != "cpu"
    # Loaded on the CPU and halved there: loading straight onto the card peaks at the fp32 size (~5 GB).
    # Only the encoder is halved; NeMo decodes TDT in fp32, so the decoder and joint stay fp32.
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu").eval()
    if cuda:
        model.encoder.half()
    model = model.to(args.device)
    lid = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(LID_MODEL, map_location="cpu").eval().to(args.device)
    labels = list(lid._cfg.train_ds.labels)
    if cuda:
        torch.cuda.empty_cache()
    send({"ready": True, "load_s": round(time.monotonic() - t, 1), "device": args.device})

    def language(audio) -> tuple[str, float]:
        """The most likely language and its probability, averaged over up to three 30 s slices (start,
        middle, end). NeMo's get_label is not used: with num_segments > 1 it slices the already sliced
        array, so every slice after the first is empty."""
        n, w = len(audio), LID_WINDOW_S * SR
        starts = sorted({0, max(0, n // 2 - w // 2), max(0, n - w)}) if n > w else [0]
        probs = torch.stack([torch.softmax(lid.infer_segment(audio[s:s + w])[1], dim=1)[0] for s in starts]).mean(0)
        best = int(probs.argmax())
        return labels[best], round(float(probs[best]), 3)

    def transcribe(audio) -> list[list]:
        segments = []
        bounds = cuts(audio, args.chunk_s)
        for a, z in zip(bounds, bounds[1:]):
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                hyp = model.transcribe([audio[a:z]], timestamps=True, verbose=False)[0]
            off = a / SR
            segments += [[round(s["start"] + off, 2), round(s["end"] + off, 2), s["segment"].strip()]
                         for s in hyp.timestamp["segment"] if s["segment"].strip()]
        return segments

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            audio, sr = sf.read(req["wav"], dtype="float32")
            if sr != SR or audio.ndim != 1:
                raise ValueError(f"expected {SR} Hz mono, got {sr} Hz with shape {audio.shape}")
            if len(audio) < SR // 10:
                send({"language": None, "language_p": 0.0, "segments": []})
                continue
            with torch.inference_mode():
                lang, p = language(audio)
                send({"language": lang, "language_p": p, "segments": transcribe(audio)})
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the worker
            if cuda:
                torch.cuda.empty_cache()
            send({"error": f"{type(exc).__name__}: {exc}"[:2000]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
