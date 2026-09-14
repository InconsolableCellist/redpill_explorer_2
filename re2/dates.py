"""Best-guess dates for items, from sources, filenames, on-image text and mtimes.

Rules run in order and the first that yields a plausible date wins:
source (archive_metadata.json) > filename > content (date hints, OCR,
captions) > mtime. Dates are strings: YYYY-MM-DD, or YYYY-MM when the day is
unknown.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

log = logging.getLogger(__name__)

EARLIEST = date(2005, 1, 1)
MTIME_SLACK = timedelta(days=2)
TWITTER_EPOCH_MS = 1288834974657

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _ymd(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _from_epoch(seconds: float) -> date | None:
    try:
        return datetime.fromtimestamp(seconds, UTC).date()
    except (OverflowError, OSError, ValueError):
        return None


# ── Filenames ──────────────────────────────────────────────────────────────

_FN_YMD = [
    # screenshot_20260131_132538.jpg, Screenshot_20200101-..., screenshot-2020_06_16-072406_PM.jpg,
    # Screenshot from 2020-01-01 ..., screenshot_20260206_203513_photo2.png
    re.compile(r"screen\s?shot\D{0,6}(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", re.IGNORECASE),
    # 2020-06-11_15-32.png, 2018-03-20_06-20-53-incarceration-rates.png
    re.compile(r"^(20\d{2})-(\d{2})-(\d{2})"),
    # 20200826_123336.jpg, IMG_20200101_..., VID_..., PXL_...
    re.compile(r"^(?:IMG|VID|PXL)?_?(20\d{2})(\d{2})(\d{2})_\d{6}", re.IGNORECASE),
]
_FN_EPOCH = [
    (re.compile(r"^(1[2-7]\d{14})(?!\d)"), 1e6),    # 4chan microseconds: 1684006713174249.png
    (re.compile(r"^(1[2-7]\d{11})(?!\d)"), 1e3),    # 4chan milliseconds: 1517112304894.png
    (re.compile(r"(?<!\d)(1[2-7]\d{8})(?!\d)"), 1),  # screenshot_race_1769449220.png
]
_TWEET_ID = re.compile(r"(?<!\d)(1\d{17,18})(?!\d)")  # X snowflake at the end of downloaded video names
_MEDIA_KEY = re.compile(r"^[A-Za-z0-9_-]{15}$")        # pbs.twimg.com media keys: GXjR0QdWwAARFz-.jpeg


def _snowflake_date(snowflake: int) -> date | None:
    return _from_epoch(((snowflake >> 22) + TWITTER_EPOCH_MS) / 1000)


def filename_dates(name: str) -> list[date]:
    """Candidate dates encoded in one file's basename."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    out: list[date] = []
    for pattern in _FN_YMD:
        m = pattern.search(stem)
        if m and (d := _ymd(int(m[1]), int(m[2]), int(m[3]))):
            out.append(d)
    for pattern, scale in _FN_EPOCH:
        m = pattern.search(stem)
        if m and (d := _from_epoch(int(m[1]) / scale)):
            out.append(d)
            break
    for m in _TWEET_ID.finditer(stem):
        if d := _snowflake_date(int(m[1])):
            out.append(d)
    if not out and _MEDIA_KEY.match(stem):
        try:
            raw = base64.urlsafe_b64decode(stem + "=")
            if d := _snowflake_date(int.from_bytes(raw[:8], "big")):
                out.append(d)
        except (binascii.Error, ValueError):
            pass
    return out


# ── Content ────────────────────────────────────────────────────────────────

_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_ORD = r"(?:st|nd|rd|th)?"
_CONTENT = [
    ("ymd", re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")),
    ("mdy_name", re.compile(rf"\b{_MON}\s+(\d{{1,2}}){_ORD}\)?,?\s+((?:19|20)\d{{2}}|'\d{{2}})\b", re.IGNORECASE)),
    ("dmy_name", re.compile(rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON},?\s+((?:19|20)\d{{2}}|\d{{2}})\b", re.IGNORECASE)),
    ("my_name", re.compile(rf"\b{_MON}\s+((?:19|20)\d{{2}})\b", re.IGNORECASE)),
    ("mdy_slash", re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2}|\d{2})\b")),
]


def _year(text: str) -> int:
    text = text.lstrip("'")
    return int(text) + 2000 if len(text) == 2 else int(text)


def content_dates(text: str) -> list[tuple[date, bool]]:
    """(date, has_day) candidates found in free text."""
    out: list[tuple[date, bool]] = []
    for kind, pattern in _CONTENT:
        for m in pattern.finditer(text):
            if kind == "ymd":
                d, has_day = _ymd(int(m[1]), int(m[2]), int(m[3])), True
            elif kind == "mdy_name":
                d, has_day = _ymd(_year(m[3]), MONTHS[m[1][:3].lower()], int(m[2])), True
            elif kind == "dmy_name":
                d, has_day = _ymd(_year(m[3]), MONTHS[m[2][:3].lower()], int(m[1])), True
            elif kind == "my_name":
                d, has_day = _ymd(int(m[2]), MONTHS[m[1][:3].lower()], 1), False
            else:
                d, has_day = _ymd(_year(m[3]), int(m[1]), int(m[2])), True
            if d:
                out.append((d, has_day))
    # "8 August 2016" also matches the month-only pattern; drop that duplicate.
    with_day = {(d.year, d.month) for d, has_day in out if has_day}
    return [(d, has_day) for d, has_day in out if has_day or (d.year, d.month) not in with_day]


# ── Item dates ─────────────────────────────────────────────────────────────

@dataclass
class ItemFacts:
    media_id: int
    source_dates: list[str]
    filenames: list[str]
    mtimes_ns: list[int]
    texts: list[str]


def _plausible(d: date, latest: date) -> bool:
    return EARLIEST <= d <= latest


def item_date(facts: ItemFacts, today: date | None = None) -> tuple[str | None, str]:
    today = today or date.today()
    newest_mtime = max((_from_epoch(ns / 1e9) for ns in facts.mtimes_ns), default=None, key=lambda d: d or date.min)
    latest = min(today, newest_mtime + MTIME_SLACK) if newest_mtime else today

    source = sorted(s[:10] for s in facts.source_dates if s and re.match(r"^\d{4}-\d{2}-\d{2}", s))
    if source:
        return source[0], "source"

    from_names = [d for name in facts.filenames for d in filename_dates(name) if _plausible(d, latest)]
    if from_names:
        return min(from_names).isoformat(), "filename"

    # Posts show their own date after any older dates they quote, so take the latest plausible one.
    found = [(d, has_day) for text in facts.texts for d, has_day in content_dates(text) if _plausible(d, latest)]
    if found:
        d, has_day = max(found, key=lambda x: (x[0], x[1]))
        return (d.isoformat() if has_day else d.strftime("%Y-%m")), "content"

    if newest_mtime:
        oldest = min(_from_epoch(ns / 1e9) for ns in facts.mtimes_ns)
        if oldest and _plausible(oldest, today + MTIME_SLACK):
            return oldest.isoformat(), "mtime"
    return None, "none"


def gather(conn: sqlite3.Connection) -> list[ItemFacts]:
    facts: dict[int, ItemFacts] = {
        r[0]: ItemFacts(r[0], [], [], [], [])
        for r in conn.execute("SELECT id FROM media WHERE kind IN ('image', 'video', 'pdf') "
                              "AND status NOT IN ('EXCLUDED', 'UNSUPPORTED')")}
    for media_id, d in conn.execute("SELECT media_id, date FROM source WHERE date IS NOT NULL"):
        if media_id in facts:
            facts[media_id].source_dates.append(d)
    for media_id, rel_path, mtime_ns in conn.execute("SELECT media_id, rel_path, mtime_ns FROM media_path"):
        if media_id in facts:
            facts[media_id].filenames.append(os.path.basename(rel_path))
            if mtime_ns:
                facts[media_id].mtimes_ns.append(mtime_ns)
    for media_id, kind, text, raw_json in conn.execute(
            "SELECT media_id, kind, text, json FROM doc WHERE kind IN ('caption', 'ocr', 'legacy_caption')"):
        if media_id not in facts:
            continue
        if kind == "caption" and raw_json:
            hints = json.loads(raw_json).get("date_hints") or []
            facts[media_id].texts.extend(str(h) for h in hints)
        else:
            facts[media_id].texts.append(text)
    return list(facts.values())


def run(conn: sqlite3.Connection, today: date | None = None) -> dict[str, int]:
    """Recompute every item's date; writes only rows that changed."""
    current = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT media_id, item_date, date_source FROM media_meta")}
    counts: dict[str, int] = {}
    changes = []
    for facts in gather(conn):
        d, src = item_date(facts, today)
        counts[src] = counts.get(src, 0) + 1
        if current.get(facts.media_id) != (d, src):
            changes.append((facts.media_id, d, src))
    for start in range(0, len(changes), 500):
        with conn:
            conn.executemany(
                """INSERT INTO media_meta(media_id, item_date, date_source) VALUES (?, ?, ?)
                   ON CONFLICT(media_id) DO UPDATE SET item_date=excluded.item_date,
                                                       date_source=excluded.date_source""",
                changes[start:start + 500])
    counts["changed"] = len(changes)
    return counts
