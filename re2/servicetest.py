"""Settings → Test: prove a service works with the values in its card, before or after saving them.

Each test is a list of steps (name, ok, detail). Model services first check the server (reachable, model
listed, loaded, reads images), then make one real request: chat and wiki must answer a one-word prompt,
vision must read a number drawn in a generated image, embeddings must rank a paraphrase above an unrelated
sentence, and speech-to-text loads its model on the GPU in a separate process (so the memory is freed).
"""
from __future__ import annotations

import base64
import dataclasses
import io
import random
import re
import subprocess
import sys
import time

import httpx
import numpy as np

from re2 import config, presets

TIMEOUT = 300.0   # a router may have to load the model first


def with_values(cfg, service: str, values: dict):
    """cfg with the card's (possibly unsaved) values for one service."""
    if service not in config.SERVICE_FIELDS:
        raise ValueError(f"unknown service {service!r}")
    changes = {}
    for _, key, attr in config.SERVICE_FIELDS[service]:
        if key not in values or values[key] in (None, ""):
            continue
        value = values[key]
        if key in config.INT_KEYS:
            value = int(value)
        else:
            value = str(value).strip()
            if key in ("base_url", "url"):
                if not value.startswith(("http://", "https://")):
                    raise ValueError(f"{key} must be an http(s) URL")
                value = value.rstrip("/")
        changes[attr] = value
    return dataclasses.replace(cfg, **changes)


def step(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def _server_step(client, cfg, service: str) -> tuple[dict, presets.Probe]:
    preset = presets.of_service(cfg, service)
    p = presets.probe(client, cfg, service, preset)
    return step("server", p.ok, presets.describe(p).replace("\n      ", "; ").lstrip("✓✗ ")), p


def _chat(client, base_url: str, model: str, content, max_tokens: int) -> tuple[str, dict, float]:
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.monotonic()
    resp = client.post(f"{base_url}/chat/completions", json=body, timeout=TIMEOUT)
    seconds = time.monotonic() - t
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    d = resp.json()
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    text = re.sub(r"^\s*<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text, d, seconds


def _speed(d: dict, seconds: float) -> str:
    t = d.get("timings") or {}
    bits = [f"{seconds:.1f} s"]
    if t.get("predicted_per_second"):
        bits.append(f"{t['predicted_per_second']:.0f} tokens/s")
    return ", ".join(bits)


def test_image(number: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 240), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((12, 12, 467, 227), outline=(40, 60, 200), width=6)
    draw.text((240, 120), number, fill=(20, 20, 30), font=ImageFont.load_default(size=120), anchor="mm")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def test_text_model(cfg, service: str, client) -> list[dict]:
    base_url = cfg.chat_base_url if service == "chat" else cfg.wiki_base_url
    model = cfg.chat_model if service == "chat" else cfg.wiki_model
    first, probe = _server_step(client, cfg, service)
    steps = [first]
    if probe.entry is None:
        return steps
    try:
        text, d, seconds = _chat(client, base_url, model, "Reply with exactly one word: READY", 200)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        return steps + [step("reply", False, str(exc)[:300])]
    ok = "READY" in text.upper()
    return steps + [step("reply", ok, f"“{text[:120]}” in {_speed(d, seconds)}"
                                      + ("" if ok else "; expected READY"))]


def test_vision(cfg, client) -> list[dict]:
    first, probe = _server_step(client, cfg, "vision")
    steps = [first]
    if probe.entry is None:
        return steps
    number = str(random.randint(100, 999))
    content = [{"type": "text", "text": "What number is written in this image? Reply with only the digits."},
               {"type": "image_url", "image_url": {
                   "url": "data:image/png;base64," + base64.b64encode(test_image(number)).decode("ascii")}}]
    try:
        text, d, seconds = _chat(client, cfg.base_url, cfg.vision_model, content, 50)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        return steps + [step("reads an image", False, str(exc)[:300])]
    ok = number in re.sub(r"\D", "", text)
    return steps + [step("reads an image", ok, f"asked for the number {number}, it said “{text[:60]}” in "
                                              f"{_speed(d, seconds)}" + ("" if ok else "; wrong answer"))]


def test_embed(cfg, client) -> list[dict]:
    from re2 import embed

    texts = ["A cat is sleeping on the windowsill.", "A kitten naps by the window.", "Quarterly tax revenue fell."]
    t = time.monotonic()
    try:
        vecs = embed.embed_texts(client, cfg, texts, "search_document: ", timeout=60, retries=0)
    except Exception as exc:  # noqa: BLE001 - any failure is the answer
        return [step("embeds text", False, f"{type(exc).__name__}: {exc}"[:300])]
    seconds = time.monotonic() - t
    near, far = float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2])
    steps = [step("embeds text", True, f"{len(texts)} texts, {vecs.shape[1]} dimensions, {seconds:.2f} s")]
    steps.append(step("meaning", bool(near > far and np.isfinite(near)),
                      f"paraphrase {near:.2f} vs unrelated {far:.2f}" + ("" if near > far else "; the paraphrase should score higher")))
    if cfg.embed_model != config.load().embed_model:
        steps.append(step("index", False, "a different model than the search index was built with; "
                                          "searches would need embed --rebuild"))
    return steps


ASR_SCRIPT = """
import sys, time
from re2 import asr, config
cfg = config.load()
cfg.asr_model, cfg.asr_device, cfg.asr_compute_type = sys.argv[1:4]
t = time.monotonic()
model = asr.load_model(cfg)
loaded = time.monotonic() - t
import numpy as np
segments, info = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
list(segments)
print(f"loaded {cfg.asr_model} on {cfg.asr_device} ({cfg.asr_compute_type}) in {loaded:.1f} s, transcribed in {time.monotonic() - t - loaded:.1f} s")
"""


def test_asr(cfg) -> list[dict]:
    try:
        out = subprocess.run([sys.executable, "-c", ASR_SCRIPT, cfg.asr_model, cfg.asr_device, cfg.asr_compute_type],
                             cwd=config.REPO_ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return [step("loads the model", False, f"no answer in {TIMEOUT:.0f} s")]
    if out.returncode != 0:
        lines = [x for x in out.stderr.strip().splitlines() if x.strip()]
        return [step("loads the model", False, (lines[-1] if lines else f"exit {out.returncode}")[:300])]
    return [step("loads the model", True, out.stdout.strip().splitlines()[-1])]


def run(cfg, service: str, values: dict, client_factory=httpx.Client) -> dict:
    cfg = with_values(cfg, service, values or {})
    t = time.monotonic()
    if service == "asr":
        steps = test_asr(cfg)
    else:
        with client_factory() as client:
            steps = (test_vision(cfg, client) if service == "vision" else test_embed(cfg, client) if service == "embed"
                     else test_text_model(cfg, service, client))
    return {"service": service, "ok": all(s["ok"] for s in steps), "steps": steps,
            "seconds": round(time.monotonic() - t, 1)}
