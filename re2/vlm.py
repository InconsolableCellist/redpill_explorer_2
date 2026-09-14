"""Vision model client: prompt, JSON schema and one structured extraction call.

Run `python -m re2.vlm <image>` to extract one file and pretty-print the result.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

VISION_TIMEOUT = 600.0
RETRIES = 2
RETRY_SLEEP = 10.0

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

IMAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "media_type": {"type": "string", "enum": [
            "screenshot_social_post", "screenshot_article", "screenshot_chat", "screenshot_other",
            "meme", "photo", "video_still", "chart_or_infographic", "document",
            "cartoon_or_illustration", "other"]},
        "description": _STR,
        "ocr_text": _STR,
        "subjects": _STR_LIST,
        "actions": _STR,
        "meaning": _STR,
        "claims": _STR_LIST,
        "intent": _STR,
        "humor": _STR,
        "entities": {
            "type": "object",
            "properties": {
                "people": _STR_LIST,
                "organizations": _STR_LIST,
                "places": _STR_LIST,
                "events": _STR_LIST,
                "platforms": _STR_LIST,
            },
            "required": ["people", "organizations", "places", "events", "platforms"],
            "additionalProperties": False,
        },
        "tags": _STR_LIST,
        "date_hints": _STR_LIST,
        "language": _STR,
        "sensitivity": {"type": "number"},
    },
    "required": ["media_type", "description", "ocr_text", "subjects", "actions", "meaning", "claims",
                 "intent", "humor", "entities", "tags", "date_hints", "language", "sensitivity"],
    "additionalProperties": False,
}

IMAGE_SYSTEM_PROMPT = """\
You are an archivist cataloguing a private research collection of political and social-media screenshots, memes, photographs, charts and documents so that they can be found again later by full-text and semantic search.

Your only job is accuracy and completeness. Describe exactly what is present. The collection deliberately includes offensive, hateful, racist, sexual, violent, graphic, conspiratorial and extremist material; record all of it plainly and quote text verbatim. Never soften, paraphrase euphemistically, omit, or add warnings, disclaimers or moral commentary. Do not speculate beyond what is visible, but do state what the image is clearly trying to communicate.

Fill every field:
- media_type: the closest category.
- description: a thorough visual description: layout, people (apparent age, sex, race or ethnicity when evident, clothing, expressions), objects, setting, on-screen UI elements, logos, watermarks, image quality.
- ocr_text: ALL visible text verbatim in reading order, including usernames, handles, display names, timestamps, dates, view or like counts, captions, headlines, chyrons, watermarks and URLs. Preserve line breaks with \\n. Empty string if there is no text.
- subjects: the people, groups, objects or topics the image is about.
- actions: what is happening or being done, including what the poster is doing (asking, mocking, accusing, reporting).
- meaning: what the image communicates overall, including implied claims and subtext.
- claims: each distinct factual or political claim made or implied, one per item, stated neutrally as a claim.
- intent: why someone would create or share this (persuade, mock, document evidence, warn, recruit, provoke, inform, entertain).
- humor: if there is a joke, meme format, sarcasm, irony or punchline, explain it and name the meme format if known; otherwise "none".
- entities: proper nouns only, as full canonical names (e.g. "Kamala Harris", "Federal Bureau of Investigation", "X (Twitter)", "January 6 Capitol riot").
- tags: 10 to 20 lowercase search tags: topics, people, organizations, places, events, platforms, content type, tone. Use spaces between words, no hashtags.
- date_hints: any dates, times, or datable references visible or implied.
- language: primary language of the text, ISO 639-1 code, or "none".
- sensitivity: 0.0 for benign, 0.3 for political but civil, 0.6 for slurs, hate or violence, 0.8 for graphic injury or explicit sexual content, 1.0 for the most extreme material. This is a retrieval filter, not a judgement."""

_KEYFRAME_DROPPED = ("claims", "intent", "humor", "date_hints", "media_type")

KEYFRAME_SCHEMA: dict = copy.deepcopy(IMAGE_SCHEMA)
for _field in _KEYFRAME_DROPPED:
    del KEYFRAME_SCHEMA["properties"][_field]
    KEYFRAME_SCHEMA["required"].remove(_field)

KEYFRAME_SYSTEM_PROMPT = "\n".join(
    line for line in IMAGE_SYSTEM_PROMPT.splitlines()
    if not any(line.startswith(f"- {f}:") for f in _KEYFRAME_DROPPED)
).replace(
    "You are an archivist cataloguing a private research collection of political and social-media "
    "screenshots, memes, photographs, charts and documents so that they can be found again later by "
    "full-text and semantic search.",
    "You are an archivist cataloguing a private research collection of political and social-media "
    "videos so that they can be found again later by full-text and semantic search. The image you are "
    "given is a single frame taken from a video.",
)


class VLMError(Exception):
    pass


def prepare_image(source: Path | str | bytes, max_side: int = 1024) -> bytes:
    """RGB JPEG (quality 90) with the longer side at most max_side; never upscales."""
    src = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(src) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        longer = max(im.size)
        if longer > max_side:
            scale = max_side / longer
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=90)
        return out.getvalue()


def extract(client: httpx.Client, cfg, image_bytes: bytes, schema: dict, system_prompt: str,
            user_text: str, max_tokens: int | None = None) -> dict:
    """One structured extraction. Returns the parsed dict plus '_usage' and '_elapsed_s'."""
    payload = {
        "model": cfg.vision_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "catalog", "schema": schema}},
        "temperature": 0.2,
        "max_tokens": max_tokens or cfg.vision_max_tokens,
    }
    url = f"{cfg.base_url}/chat/completions"
    started = time.monotonic()
    for attempt in range(RETRIES + 1):
        try:
            resp = client.post(url, json=payload, timeout=VISION_TIMEOUT)
            if resp.status_code >= 500 and attempt < RETRIES:
                log.warning("vision HTTP %s, retrying: %s", resp.status_code, resp.text[:200])
                time.sleep(RETRY_SLEEP)
                continue
            resp.raise_for_status()
            break
        except httpx.TimeoutException as exc:
            if attempt < RETRIES:
                log.warning("vision timeout, retrying: %s", exc)
                time.sleep(RETRY_SLEEP)
                continue
            raise VLMError(f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")[:500]
            raise VLMError(f"{type(exc).__name__}: {exc} {detail}".strip()) from exc

    body = resp.json()
    try:
        choice = body["choices"][0]
        content = choice["message"]["content"]
        data = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        finish = body.get("choices", [{}])[0].get("finish_reason") if isinstance(body, dict) else None
        raise VLMError(f"unparseable response (finish_reason={finish}): {exc}") from exc
    if not isinstance(data, dict):
        raise VLMError(f"response is not an object: {type(data).__name__}")
    missing = [k for k in schema["required"] if k not in data]
    if missing:
        raise VLMError(f"response missing fields: {missing}")
    data["_usage"] = body.get("usage") or {}
    data["_elapsed_s"] = round(time.monotonic() - started, 2)
    return data


def main(argv: list[str]) -> int:
    from re2 import config

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if len(argv) != 1:
        print("usage: python -m re2.vlm <image>", file=sys.stderr)
        return 2
    cfg = config.load()
    image = prepare_image(Path(argv[0]), cfg.vision_max_side)
    with httpx.Client() as client:
        result = extract(client, cfg, image, IMAGE_SCHEMA, IMAGE_SYSTEM_PROMPT, "Catalogue this image.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
