"""Phase 6.4 (deterministic part): merge entity names that are obviously the same.

Two entities merge when their keys match: case, punctuation, a leading "the" and
leading titles are ignored ("The White House" = "White House", "President Joe Biden"
= "Joe Biden"). Organizations and platforms merge across types, because vision
models file X, CNN or Reddit under either. Names that are placeholders ("none",
"unknown") are dropped. Pairs that need judgement ("Trump" / "Donald Trump") are
left for the LLM verification step.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PLACEHOLDER_NAMES = {"none", "n/a", "na", "unknown", "unidentified", "not visible", "not applicable", "null",
                     "anonymous", "unnamed", "various", "other", "", "-"}
TITLES = ("president", "vice president", "former president", "senator", "sen", "representative", "rep",
          "congressman", "congresswoman", "governor", "gov", "prime minister", "pm", "dr", "doctor", "mr", "mrs",
          "ms", "sir", "judge", "justice", "secretary", "general", "gen", "king", "queen", "prince", "princess",
          "pope", "mayor", "attorney general", "ag", "speaker")
_TITLE_RE = re.compile(r"^(?:(?:" + "|".join(sorted((re.escape(t) for t in TITLES), key=len, reverse=True)) +
                       r")\.?\s+)+", re.IGNORECASE)
MERGE_ACROSS = {"organization": "org", "platform": "org"}   # types that may merge with each other


def is_placeholder(name: str) -> bool:
    return name.strip().strip(".").lower() in PLACEHOLDER_NAMES


def entity_key(name: str, etype: str) -> tuple[str, str]:
    """(type group, normalized name) — equal keys mean the same entity."""
    key = name.casefold().strip()
    key = re.sub(r"^the\s+", "", key)
    if etype == "person":
        key = _TITLE_RE.sub("", key)
    key = re.sub(r"[^\w&+ ]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return MERGE_ACROSS.get(etype, etype), key


def ensure_alias_table(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS entity_alias (
                      alias_id INTEGER PRIMARY KEY REFERENCES entity(id),
                      canonical_id INTEGER NOT NULL REFERENCES entity(id))""")


def canonical_id(conn: sqlite3.Connection, entity_id: int) -> int:
    row = conn.execute("SELECT canonical_id FROM entity_alias WHERE alias_id=?", (entity_id,)).fetchone()
    return row[0] if row else entity_id


@dataclass
class MergeStats:
    groups: int = 0
    merged: int = 0
    placeholders: int = 0
    examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"groups={self.groups} merged={self.merged} placeholders_removed={self.placeholders}"


def merge(conn: sqlite3.Connection, dry_run: bool = False) -> MergeStats:
    ensure_alias_table(conn)
    stats = MergeStats()
    aliased = {r[0] for r in conn.execute("SELECT alias_id FROM entity_alias")}
    rows = conn.execute(
        """SELECT e.id, e.name, e.type, (SELECT COUNT(*) FROM media_entity me WHERE me.entity_id=e.id) AS n
           FROM entity e""").fetchall()
    placeholders = [r[0] for r in rows if is_placeholder(r[1])]
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for entity_id, name, etype, n in rows:
        if entity_id in aliased or entity_id in placeholders:
            continue
        groups[entity_key(name, etype)].append((entity_id, name, etype, n))

    merges: list[tuple[int, int]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # The most used spelling wins; ties go to the longer (usually fuller) name.
        members.sort(key=lambda m: (-m[3], -len(m[1]), m[0]))
        canon = members[0]
        stats.groups += 1
        for other in members[1:]:
            merges.append((other[0], canon[0]))
            if len(stats.examples) < 25:
                stats.examples.append(f"{other[1]} ({other[2]}) -> {canon[1]} ({canon[2]})")
    stats.merged = len(merges)
    stats.placeholders = len(placeholders)
    if dry_run:
        return stats

    with conn:
        for alias, canon in merges:
            conn.execute("INSERT OR REPLACE INTO entity_alias(alias_id, canonical_id) VALUES (?, ?)", (alias, canon))
            conn.execute("UPDATE entity_alias SET canonical_id=? WHERE canonical_id=?", (canon, alias))
            conn.execute("""INSERT OR IGNORE INTO media_entity(media_id, entity_id)
                            SELECT media_id, ? FROM media_entity WHERE entity_id=?""", (canon, alias))
            conn.execute("DELETE FROM media_entity WHERE entity_id=?", (alias,))
        for entity_id in placeholders:
            conn.execute("DELETE FROM media_entity WHERE entity_id=?", (entity_id,))
    return stats
