"""Vision model client: prompt, JSON schema and one structured extraction call.

Run `python -m re2.vlm <image>` to extract one file and pretty-print the result.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
import math
import re
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
        # Last on purpose: the model writes the headline after it has transcribed and interpreted the image.
        "title": _STR,
        "summary": _STR,
    },
    "required": ["media_type", "description", "ocr_text", "subjects", "actions", "meaning", "claims",
                 "intent", "humor", "entities", "tags", "date_hints", "language", "sensitivity", "title", "summary"],
    "additionalProperties": False,
}

IMAGE_SYSTEM_PROMPT = """\
You are an archivist cataloguing a private research collection of political and social-media screenshots, memes, photographs, charts and documents so that they can be found again later by full-text and semantic search.

Your only job is accuracy and completeness. Describe exactly what is present. The collection deliberately includes offensive, hateful, racist, sexual, violent, graphic, conspiratorial and extremist material; record all of it plainly and quote text verbatim. Never soften, paraphrase euphemistically, omit, or add warnings, disclaimers or moral commentary. Do not speculate beyond what is visible, but do state what the image is clearly trying to communicate.

Fill every field:
- media_type: the closest category.
- description: a thorough visual description: layout, people (apparent age, sex, race or ethnicity when evident, clothing, expressions; name well-known public figures you recognize or that the text identifies), objects, setting, on-screen UI elements, logos, watermarks, image quality.
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
- sensitivity: 0.0 for benign, 0.3 for political but civil, 0.6 for slurs, hate or violence, 0.8 for graphic injury or explicit sexual content, 1.0 for the most extreme material. This is a retrieval filter, not a judgement.
- title: a headline of at most 14 words stating the substance: who is shown or quoted and what is claimed, shown or happening, naming people, places and events (e.g. "Netanyahu pictured beside the official promising unredacted JFK and Epstein files"). Never start with the medium ("Screenshot of", "Tweet from", "Image showing").
- summary: one to three sentences that lead with the most important content: the claim, event or implication and who makes it, then what the poster implies. The platform, account and engagement counts are secondary: mention them only after the substance, briefly, if at all."""

_KEYFRAME_DROPPED = ("claims", "intent", "humor", "date_hints", "media_type", "title", "summary")

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


# Captions extracted before title and summary existed get just those two fields,
# from the image plus the earlier extraction, which is much cheaper than a full re-read.
RETITLE_SCHEMA: dict = {
    "type": "object",
    "properties": {"title": _STR, "summary": _STR},
    "required": ["title", "summary"],
    "additionalProperties": False,
}

RETITLE_SYSTEM_PROMPT = """\
You write headlines and summaries for items in a private research archive of political and social-media material. You are given the image and an earlier, detailed catalogue entry for it.

Record content plainly, including offensive material, without softening, warnings or commentary.

- title: a headline of at most 14 words stating the substance: who is shown or quoted and what is claimed, shown or happening, naming people, places and events (e.g. "Netanyahu pictured beside the official promising unredacted JFK and Epstein files"). Name well-known public figures you recognize. Never start with the medium ("Screenshot of", "Tweet from", "Image showing").
- summary: one to three sentences that lead with the most important content: the claim, event or implication and who makes it, then what the poster implies. The platform, account and engagement counts are secondary: mention them only after the substance, briefly, if at all."""

# Scanned document pages: mostly verbatim text, so a slimmer schema keeps output tokens down.
PAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "description": _STR,
        "ocr_text": _STR,
        "entities": copy.deepcopy(IMAGE_SCHEMA["properties"]["entities"]),
        "tags": _STR_LIST,
        "language": _STR,
    },
    "required": ["description", "ocr_text", "entities", "tags", "language"],
    "additionalProperties": False,
}

PAGE_SYSTEM_PROMPT = """\
You are an archivist transcribing scanned pages of documents in a private research collection of political material (court filings, reports, tax forms, papers, leaked documents, meme compilations) so they can be found by full-text search.

Transcribe exactly and completely. The collection includes offensive and extremist material; record it plainly. Never soften, omit, summarise instead of transcribing, or add warnings or commentary.

Fill every field:
- description: one or two sentences on what the page is (form, letter, table, chart, slide, image) and what it is about.
- ocr_text: ALL text on the page verbatim in reading order, including headers, footers, page numbers, table cells (row by row, cells separated by " | "), captions, handwriting that is legible, stamps and redaction marks (write [REDACTED]). Preserve line breaks with \\n.
- entities: proper nouns only, as full canonical names.
- tags: 5 to 12 lowercase search tags.
- language: ISO 639-1 code of the main language, or "none"."""

THOUGHT_PREFIX = re.compile(r"^\s*<\|channel>thought.*?<channel\|>", re.DOTALL)


class VLMError(Exception):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw  # the model's text, when there was any


class VLMLengthError(VLMError):
    """The model ran out of output tokens before closing the JSON."""


def _open_rgb(source: Path | str | bytes) -> Image.Image:
    src = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(src) as im:
        im.seek(0)  # first frame of animations
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            return bg
        return im.convert("RGB")


def _jpeg(im: Image.Image) -> bytes:
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


def prepare_image(source: Path | str | bytes, max_side: int = 1024) -> bytes:
    """RGB JPEG (quality 90) with the longer side at most max_side; never upscales."""
    im = _open_rgb(source)
    longer = max(im.size)
    if longer > max_side:
        scale = max_side / longer
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    return _jpeg(im)


# Tall screenshots (long threads, articles) become unreadable when the whole
# image is shrunk to max_side, so they are cut into overlapping slices instead.
TALL_RATIO = 2.5          # height / width above which an image is sliced
TILE_WIDTH = 768          # slice width in pixels (never upscaled)
TILE_ASPECT = 1.25        # nominal slice height / width
TILE_OVERLAP = 0.10       # fraction of a slice shared with its neighbour


def is_tall(width: int | None, height: int | None) -> bool:
    return bool(width and height and height > width * TALL_RATIO)


def tile_boxes(width: int, height: int, max_tiles: int) -> list[tuple[int, int, int, int]]:
    """Crop boxes (left, top, right, bottom) of overlapping slices covering a tall image."""
    nominal_h = width * TILE_ASPECT
    step = nominal_h * (1 - TILE_OVERLAP)
    n = max(2, min(max_tiles, math.ceil((height - nominal_h * TILE_OVERLAP) / step)))
    # Solve for slice height so n slices with the overlap fraction exactly cover the image.
    tile_h = math.ceil(height / (n - (n - 1) * TILE_OVERLAP))
    stride = (height - tile_h) / (n - 1)
    return [(0, round(i * stride), width, min(height, round(i * stride) + tile_h)) for i in range(n)]


# Text-dense portrait images that overflowed once are sliced on retry, even
# when they are not "tall", so their small text stays legible.
DENSE_RATIO = 1.2
DENSE_MIN_HEIGHT = 1400


def prepare_images(source: Path | str | bytes, max_side: int = 1024, max_tiles: int = 6,
                   dense: bool = False) -> list[bytes]:
    """One JPEG for normal images; top-to-bottom overlapping slices for very tall (or dense portrait) ones."""
    im = _open_rgb(source)
    slice_it = is_tall(im.width, im.height) or (
        dense and im.height > im.width * DENSE_RATIO and im.height >= DENSE_MIN_HEIGHT)
    if not slice_it or max_tiles < 2:
        return [prepare_image(_jpeg(im), max_side)]
    if im.width > TILE_WIDTH:
        im = im.resize((TILE_WIDTH, max(1, round(im.height * TILE_WIDTH / im.width))), Image.LANCZOS)
    return [_jpeg(im.crop(box)) for box in tile_boxes(im.width, im.height, max_tiles)]


def tiles_note(n: int) -> str:
    if n < 2:
        return ""
    return (f" This is one tall image cut into {n} overlapping horizontal slices, given top to bottom."
            " Catalogue it as a single image and do not repeat text that appears twice in the overlaps.")


def extract(client: httpx.Client, cfg, images: bytes | list[bytes], schema: dict, system_prompt: str,
            user_text: str, max_tokens: int | None = None) -> dict:
    """One structured extraction over one image or its ordered slices.

    Returns the parsed dict plus '_usage' and '_elapsed_s'.
    """
    if isinstance(images, bytes):
        images = [images]
    payload = {
        "model": cfg.vision_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                *({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")}} for b in images),
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "catalog", "schema": schema}},
        "temperature": 0.2,
        "max_tokens": max_tokens or cfg.vision_max_tokens,
        # Gemma 4 thinks by default and the reasoning eats max_tokens before any JSON appears.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
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
    content = None
    try:
        choice = body["choices"][0]
        content = THOUGHT_PREFIX.sub("", choice["message"]["content"] or "")
        data = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        finish = body.get("choices", [{}])[0].get("finish_reason") if isinstance(body, dict) else None
        error = VLMLengthError if finish == "length" else VLMError
        raise error(f"unparseable response (finish_reason={finish}, "
                    f"max_tokens={payload['max_tokens']}): {exc}", raw=content) from exc
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
    images = prepare_images(Path(argv[0]), cfg.vision_max_side, cfg.vision_max_tiles)
    with httpx.Client() as client:
        result = extract(client, cfg, images, IMAGE_SCHEMA, IMAGE_SYSTEM_PROMPT,
                         "Catalogue this image." + tiles_note(len(images)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
