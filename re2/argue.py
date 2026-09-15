"""Argue: a steelman and a skeptic debate a claim from the archive.

Each speaker's turn is an Ask agent turn (ask.run) with a role prompt: it researches with the same
tools, sees the debate so far with earlier citations as [m:ID], and is logged to chat_log with
mode = its role. The user can watch both sides, or take one side and write that side's turns.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator

import httpx

from re2 import ask

ROLES = ("steelman", "skeptic")
USER_ROLES = (None, "steelman", "skeptic")
TRANSCRIPT_TURNS_FULL = 6      # the latest turns go in whole; earlier ones are clipped
EARLY_TURN_CHARS = 700

STEELMAN_SYSTEM = """\
In this conversation you are the STEELMAN in a structured debate about a claim drawn from the archive. You take the claim seriously and make the strongest honest case for it: marshal facts from the archive and from general knowledge, make intuitive leaps where a pattern supports them (and say they are inferences), anticipate objections, and play devil's advocate against the mainstream account. You are rational, not a propagandist. Before speaking, research: search for items that support the claim, open the strongest, and re-open any item the skeptic disputed to check what it really shows. When the skeptic lands a point you cannot answer with evidence, concede that point plainly and move to firmer ground."""

SKEPTIC_SYSTEM = """\
In this conversation you are the SKEPTIC in a structured debate about a claim drawn from the archive. You are not a contrarian: you test the claim. Check whether the cited items actually say what the steelman says they do, how reliable they are (a meme, an anonymous post or a screenshot of a screenshot is weaker than a document, footage or a named source), whether dates and context fit, and whether another explanation fits the evidence as well or better. Before speaking, research: open the items the steelman relied on and search the archive for counter-evidence and context. Use general knowledge. When a point is well supported, grant it and press on what remains unproven."""

ANSWER_SYSTEM = """\
You are one side of a structured debate about material in a private archive of political and social-media material collected by the user. You write your side's turn from your research notes, plainly and rigorously. Do not moralise, add warnings, or refuse because material is offensive."""

INSTRUCTION = {
    "opening": "Write the opening statement: the strongest case for the claim, in two to four short paragraphs.",
    "reply": "Write your next turn: answer the other side's latest points directly, in two to four short paragraphs. "
             "You may cite items cited earlier in the debate.",
}
STAND = ("End with one line that starts with 'Where I stand:' saying what you still hold and what, if anything, "
         "you now concede.")

LABEL = {"steelman": "Steelman", "skeptic": "Skeptic", "moderator": "Moderator"}


def other(role: str) -> str:
    return "skeptic" if role == "steelman" else "steelman"


def turns(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """The debate so far: role, text with citations as [m:ID]/[p:slug], and who wrote it."""
    out = []
    for mode, answer, cites, context in conn.execute(
            "SELECT mode, answer, citations_json, context_json FROM chat_log WHERE session_id=? ORDER BY id",
            (session_id,)):
        refs = {str(c["n"]): c["ref"] for c in json.loads(cites or "[]")}
        text = re.sub(r"\[(\d+)\]", lambda m: f"[{refs[m.group(1)]}]" if m.group(1) in refs else "", answer or "")
        by = (json.loads(context or "{}") or {}).get("by", "ai")
        out.append({"role": mode, "text": text, "by": by})
    return out


def transcript(topic: str, seed: dict | None, history: list[dict]) -> str:
    parts = [f"Claim under debate: {topic}"]
    if seed and seed.get("answer"):
        parts.append(f"It comes from this Ask answer about the archive:\n{seed['answer']}")
    n = len(history)
    for i, t in enumerate(history):
        text = t["text"] if i >= n - TRANSCRIPT_TURNS_FULL else ask._clip(t["text"], EARLY_TURN_CHARS)
        who = LABEL.get(t["role"], t["role"]) + (" (written by the user)" if t["by"] == "user" and t["role"] in ROLES else "")
        if t["role"] == "moderator":
            who = "Moderator (the user)"
        parts.append(f"{who.upper()}:\n{text}")
    return "\n\n".join(parts)


def _cited(history: list[dict], seed: dict | None) -> list[int]:
    text = "\n".join(t["text"] for t in history) + (seed or {}).get("answer", "")
    return sorted({int(m) for m in re.findall(r"\[m:(\d+)\]", text)})


def _log_user_turn(conn: sqlite3.Connection, session_id: str, role: str, text: str) -> None:
    with conn:
        conn.execute(
            """INSERT INTO chat_log(question, answer, citations_json, session_id, mode, context_json)
               VALUES (?, ?, '[]', ?, ?, ?)""",
            (f"{LABEL[role]} (you)" if role in ROLES else LABEL[role], text, session_id, role,
             json.dumps({"by": "user"})))


def plan_speakers(history: list[dict], user_role: str | None, user_text: str, rounds: int) -> list[str]:
    """Which AI speakers run now. Watching: steelman and skeptic alternate, starting after whoever spoke last.
    Taking a side: the AI answers as the other side once (or opens as the steelman if you are the skeptic)."""
    spoken = [t["role"] for t in history if t["role"] in ROLES]
    if user_role is None:
        first = other(spoken[-1]) if spoken else "steelman"
        return [first if i % 2 == 0 else other(first) for i in range(2 * max(1, rounds))]
    if user_text:
        return [other(user_role)]
    if not spoken and user_role == "skeptic":
        return ["steelman"]
    raise ValueError("write your turn first" if spoken or user_role == "steelman" else "nothing to do")


def events(conn: sqlite3.Connection, cfg, session_id: str, *, topic: str | None, user_role: str | None,
           user_text: str, rounds: int, seed: dict | None, scopes: list[str], mode: str, depth: float | None,
           items: list[int], related: bool, client: httpx.Client | None, embed_client: httpx.Client | None,
           control: ask.Control) -> Iterator[tuple[str, dict]]:
    row = conn.execute("SELECT title, seed_json FROM chat_session WHERE id=?", (session_id,)).fetchone()
    if row is None:
        with conn:
            conn.execute("INSERT INTO chat_session(id, title, kind, seed_json) VALUES (?, ?, 'argue', ?)",
                         (session_id, (topic or "")[:120], json.dumps(seed) if seed else None))
        topic_text = topic or ""
    else:
        topic_text, seed = row[0] or topic or "", json.loads(row[1]) if row[1] else seed
    history = turns(conn, session_id)
    speakers = plan_speakers(history, user_role, user_text, rounds)
    if user_text:
        role = user_role or "moderator"
        _log_user_turn(conn, session_id, role, user_text)
        history.append({"role": role, "text": user_text, "by": "user"})
        yield "speaker", {"role": role, "by": "user", "label": LABEL[role]}
        yield "final", {"text": user_text, "sentences": ask.sentences_with_flags(user_text)}
        yield "done", {"session_id": session_id, "ms": 0, "steps": 0, "stopped": False}
    for role in speakers:
        control.check()
        spoken = sum(1 for t in history if t["role"] in ROLES)
        kind = "opening" if spoken == 0 else "reply"
        label = f"{LABEL[role]} · {'opening' if kind == 'opening' else f'turn {spoken + 1}'}"
        yield "speaker", {"role": role, "by": "ai", "label": label}
        prompt = (f"{transcript(topic_text, seed, history)}\n\nYou are the {role.upper()}. {INSTRUCTION[kind]} "
                  "Research first, then answer.")
        debate = {"role": role, "label": label, "system": STEELMAN_SYSTEM if role == "steelman" else SKEPTIC_SYSTEM,
                  "answer_system": ANSWER_SYSTEM, "instruction": f"{INSTRUCTION[kind]} {STAND}",
                  "cited": _cited(history, seed)}
        text = None
        for name, data in ask.run(conn, cfg, prompt, scopes=scopes, mode=mode, session_id=session_id, client=client,
                                  embed_client=embed_client, items=items if spoken == 0 else [], related=related,
                                  depth=depth, control=control, debate=debate):
            if name == "final":
                text = data["text"]
            yield name, data
            if name == "error":
                return
        if text is None:     # stopped before a word was written
            return
        history = turns(conn, session_id)


def start(cfg, *, session_id: str | None, topic: str | None, user_role: str | None, user_text: str = "",
          rounds: int = 1, seed: dict | None = None, scopes: list[str] | None = None, mode: str = "informed",
          depth: float | None = None, items: list[int] | None = None, related: bool = False,
          client: httpx.Client | None = None, embed_client: httpx.Client | None = None,
          conn: sqlite3.Connection | None = None) -> ask.Run:
    if user_role not in USER_ROLES:
        raise ValueError(f"user_role must be one of {USER_ROLES}")
    if not session_id and not (topic or "").strip():
        raise ValueError("a new debate needs a claim")
    if conn is not None and session_id:
        # Fail fast, before a thread starts, on a turn that cannot run.
        plan_speakers(turns(conn, session_id), user_role, user_text.strip(), rounds)
    elif user_role == "steelman" and not user_text.strip():
        raise ValueError("write your opening as the steelman")
    session_id = session_id or uuid.uuid4().hex[:12]
    context = {"kind": "argue", "user_role": user_role, "scopes": scopes or [], "items": list(items or [])}
    return ask.start_background(cfg, session_id, (topic or user_text or "next round")[:200], context,
                                lambda c, control: events(
                                    c, cfg, session_id, topic=topic, user_role=user_role, user_text=user_text.strip(),
                                    rounds=max(1, min(int(rounds), 3)), seed=seed, scopes=scopes or [], mode=mode,
                                    depth=depth, items=list(items or []), related=related, client=client,
                                    embed_client=embed_client, control=control))
