import dataclasses
import json
import re

import httpx
import numpy as np
import pytest

from re2 import config, db, digest, embed, wiki


def sse(text: str) -> bytes:
    chunks = [text[i:i + 40] for i in range(0, len(text), 40)] or [""]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}, 'finish_reason': None}]})}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


class FakeModel:
    """Answers digest, split, notes, reduce and page requests the way a well-behaved model would."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        user = body["messages"][1]["content"]
        schema = (body.get("response_format") or {}).get("json_schema", {}).get("schema", {})
        props = schema.get("properties", {})
        if "topic_title" in props:
            n = len(re.findall(r"^\d+ \| ", user, flags=re.M))
            half = n // 2
            reply = {"topic_title": "Protests over the new law",
                     "events": [{"title": "First march on parliament", "summary": "A march.", "members": list(range(1, half + 1))},
                                {"title": "Second week of strikes", "summary": "Strikes.", "members": list(range(half + 1, n + 1))},
                                {"title": "Lonely", "summary": "One item.", "members": [1]}],
                     "not_events": []}
            text = json.dumps(reply)
        elif "items" in props:
            n = len(re.findall(r"^### \d+$", user, flags=re.M))
            text = json.dumps({"items": [{"n": i, "title": f"Digest title {i}", "summary": f"Digest summary {i}.",
                                          "event_hint": "the march"} for i in range(1, n + 1)]})
        elif "Write notes for the page" in user:
            ids = re.findall(r"^\[m:(\d+)\]", user, flags=re.M)
            text = "\n".join(f"- An item shows a claim [m:{m}]." for m in ids) + "\n- A made-up one [m:999999]."
        elif "Merge them into one set of notes" in user:
            ids = sorted(set(re.findall(r"\[m:(\d+)\]", user)))
            text = "## Merged\n" + "\n".join(f"- Point [m:{m}]" for m in ids)
        else:
            ids = list(dict.fromkeys(re.findall(r"\[m:(\d+)\]", user)))
            slugs = list(dict.fromkeys(re.findall(r"\[p:([a-z0-9/_.-]+)\]", user)))
            head = "# Named By Model\n\n" if "Choose its title" in user else ""
            links = " ".join(f"[p:{s}]" for s in slugs)
            text = (f"{head}The items document a protest [m:{ids[0] if ids else 1}] [m:999999] and more {links} [p:nowhere].\n\n"
                    "## What happened\n\n" + " ".join(f"Point {i} [m:{m}]." for i, m in enumerate(ids)))
        return httpx.Response(200, content=sse(text), headers={"content-type": "text/event-stream"})


@pytest.fixture
def env(tmp_path):
    cfg = dataclasses.replace(config.load(), data_dir=tmp_path / "data", wiki_base_url="http://wiki.test/v1",
                              wiki_model="fake", wiki_context_tokens=32000, wiki_concurrency=1, server_host="0.0.0.0")
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    model = FakeModel()
    return cfg, conn, model, (lambda: httpx.Client(transport=httpx.MockTransport(model)))


def add_item(conn, n, date, caption=True, legacy=None, folder="news", source="filename"):
    mid = conn.execute("INSERT INTO media(uid, kind, status) VALUES (?, 'image', 'READY')", (f"u{n}",)).lastrowid
    conn.execute("INSERT INTO media_path(media_id, rel_path, readable, first_seen, last_seen) VALUES (?, ?, 1, 'x', 'x')",
                 (mid, f"{folder}/{n}.png"))
    conn.execute("INSERT INTO media_meta(media_id, item_date, date_source) VALUES (?, ?, ?)", (mid, date, source))
    if caption:
        d = {"title": f"Caption title {n}", "summary": f"Caption summary {n}.", "description": "A photo.",
             "claims": [f"claim {n}"], "prompt_version": 2, "entities": {"events": ["the march"]}}
        conn.execute("INSERT INTO doc(media_id, kind, text, json) VALUES (?, 'caption', ?, ?)", (mid, d["summary"], json.dumps(d)))
    if legacy:
        conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'legacy_caption', ?)", (mid, legacy))
    return mid


def add_topic(conn, ids, run="r1"):
    theme = conn.execute("INSERT INTO cluster(level, run_id, label) VALUES ('theme', ?, 'unrest · law')", (run,)).lastrowid
    topic = conn.execute("INSERT INTO cluster(level, parent_id, run_id, label) VALUES ('topic', ?, ?, 'march · strike')",
                         (theme, run)).lastrowid
    conn.executemany("INSERT INTO cluster_member(cluster_id, media_id) VALUES (?, ?)",
                     [(c, m) for c in (theme, topic) for m in ids])
    db.kv_set(conn, "cluster_run", run)
    conn.commit()
    return topic


def page(conn, slug=None, kind=None):
    row = conn.execute("SELECT * FROM wiki_page WHERE " + ("slug=?" if slug else "kind=?"), (slug or kind,)).fetchone()
    return dict(row) if row else None


# ── Pure functions ────────────────────────────────────────────────────────

def test_clean_body_validates_citations_and_links():
    text = ("Lead [m:1, m:2] and [m:3; 4] bogus [m:77] . See [p:event/a] and [p:event/zzz] and [[event/a|the march]] "
            "and [[event/none|a strike]].")
    body, cites, links = wiki.clean_body(text, {1, 2, 3, 4}, {"event/a": "A"})
    assert "[m:77]" not in body and "[p:event/zzz]" not in body
    assert cites == {1: 1, 2: 1, 3: 1, 4: 1}
    assert links == {"event/a"}
    assert body.count("[p:event/a]") == 2 and "a strike" in body and "bogus." in body


def test_extract_summary_skips_headings_and_strips_markers():
    body = "## Heading\n\n- a list [m:1]\n\nThe **march** began on Monday [m:12][m:13] and ended [p:event/x] late. " + "x" * 400
    s = wiki.extract_summary(body)
    assert s.startswith("The march began on Monday and ended late.")
    assert "[m:" not in s and len(s) <= 300
    title, rest = wiki.split_title("# A title\n\nLead.")
    assert title == "A title" and rest == "Lead."


def test_validate_split_accounts_for_every_line():
    reply = {"events": [{"title": "A", "summary": "", "members": [1, 2, 2, 9]},
                        {"title": "B", "summary": "", "members": [2, 3]},     # 2 already used: B has one member
                        {"title": "C", "summary": "", "members": [4, 5]}], "not_events": [5]}
    events, rest = wiki.validate_split(reply, 6)
    assert [e["members"] for e in events] == [[1, 2], [4, 5]]
    assert rest == [3, 6]


def test_slugify():
    assert wiki.slugify("Trudeau: Ukraine may use “long-range” missiles!") == "trudeau-ukraine-may-use-long-range-missiles"
    assert wiki.slugify("Ämter über Straße") == "amter-uber-strae"


# ── Digests ───────────────────────────────────────────────────────────────

def test_digests_copy_captions_and_batch_the_rest(env):
    cfg, conn, model, factory = env
    captioned = add_item(conn, 1, "2024-01-01")
    legacy = [add_item(conn, n, "2024-01-02", caption=False, legacy=f"A tweet about the march number {n}.")
              for n in range(2, 25)]
    conn.commit()
    stats = digest.run(conn, cfg, client_factory=factory)
    assert stats.copied == 1 and stats.done == 23 and stats.failed == 0
    assert len(model.calls) == 3   # 23 items, 10 per request
    row = conn.execute("SELECT title, source, model FROM item_digest WHERE media_id=?", (captioned,)).fetchone()
    assert tuple(row) == ("Caption title 1", "caption", None)
    assert conn.execute("SELECT source FROM item_digest WHERE media_id=?", (legacy[0],)).fetchone()[0] == "model"
    # Nothing changed: nothing is due.
    again = digest.run(conn, cfg, client_factory=factory)
    assert again.due == 0 and len(model.calls) == 3
    # A current caption arrives for a legacy item: copied, no model call.
    d = {"title": "New caption", "summary": "Now captioned.", "prompt_version": 2}
    conn.execute("INSERT INTO doc(media_id, kind, text, json) VALUES (?, 'caption', 'x', ?)", (legacy[0], json.dumps(d)))
    conn.commit()
    third = digest.run(conn, cfg, client_factory=factory)
    assert third.copied == 1 and third.due == 0 and len(model.calls) == 3
    assert conn.execute("SELECT title FROM item_digest WHERE media_id=?", (legacy[0],)).fetchone()[0] == "New caption"


# ── Building ──────────────────────────────────────────────────────────────

def test_map_batches_cover_every_member(env, monkeypatch):
    cfg, conn, model, factory = env
    ids = [add_item(conn, n, f"2024-02-{1 + n % 28:02d}") for n in range(95)]
    conn.commit()
    add_topic(conn, ids)
    wiki.sync(conn)
    topic = page(conn, kind="topic")
    texts = digest.gather(conn, ids)
    plan = wiki.plan_page(conn, cfg, topic, texts)
    assert len(plan.batches) == 3 and all(len(b) <= wiki.MAP_BATCH_ITEMS for b in plan.batches)
    assert sorted(m for b in plan.batches for m in b) == sorted(ids)


def test_build_writes_events_topic_theme_overview_with_full_coverage(env, monkeypatch):
    cfg, conn, model, factory = env
    monkeypatch.setattr(wiki, "DIRECT_MAX_ITEMS", 20)    # so the 30-item events are read in batches
    monkeypatch.setattr(wiki, "MAP_BATCH_ITEMS", 12)
    ids = [add_item(conn, n, f"2024-03-{1 + n % 28:02d}") for n in range(60)]
    conn.commit()
    add_topic(conn, ids)
    stats = wiki.build(conn, cfg, client_factory=factory)
    assert stats.failed == 0 and not stats.interrupted, stats.summary()

    events = [dict(r) for r in conn.execute("SELECT * FROM wiki_page WHERE kind='event' ORDER BY id")]
    assert len(events) == 2        # the one-member event was dissolved
    topic = page(conn, kind="topic")
    assert topic["title"] == "Named By Model" or topic["title"] == "Protests over the new law"
    assert topic["slug"].startswith("topic/") and all(e["parent_slug"] == topic["slug"] for e in events)
    for p in events + [topic, page(conn, kind="theme")]:
        assert p["body_md"] and p["stale"] == 0
        assert p["members_read"] == p["members_total"] == conn.execute(
            "SELECT COUNT(*) FROM wiki_member WHERE page_id=?", (p["id"],)).fetchone()[0], p["slug"]
        assert "[m:999999]" not in p["body_md"] and "[p:nowhere]" not in p["body_md"]
        assert conn.execute("SELECT COUNT(*) FROM wiki_member WHERE page_id=? AND read_hash IS NULL", (p["id"],)).fetchone()[0] == 0
    # Every member of an event was in some notes request, and the events were read in batches.
    noted = {int(m) for c in model.calls for m in re.findall(r"^\[m:(\d+)\]", c["messages"][1]["content"], flags=re.M)
             if "Write notes for the page" in c["messages"][1]["content"]}
    assert noted == set(ids)
    assert json.loads(events[0]["build_json"])["batches"] == 3
    # Topics read their events (links recorded) and cite items.
    linked = {r[0] for r in conn.execute("SELECT p.slug FROM wiki_link l JOIN wiki_page p ON p.id=l.to_page "
                                         "WHERE l.from_page=?", (topic["id"],))}
    assert linked == {e["slug"] for e in events}
    assert conn.execute("SELECT COUNT(*) FROM wiki_citation WHERE page_id=?", (topic["id"],)).fetchone()[0] > 0
    overview = page(conn, "overview")
    assert overview["body_md"] and overview["members_read"] == 60
    # The FTS index follows the pages.
    assert conn.execute("SELECT COUNT(*) FROM wiki_fts WHERE wiki_fts MATCH 'protest'").fetchone()[0] >= 4

    # A second build has nothing to do.
    calls = len(model.calls)
    again = wiki.build(conn, cfg, client_factory=factory)
    assert again.pages == 0 and len(model.calls) == calls


def test_changed_items_make_pages_and_parents_stale(env, monkeypatch):
    cfg, conn, model, factory = env
    ids = [add_item(conn, n, "2024-04-01") for n in range(20)]
    conn.commit()
    add_topic(conn, ids)
    wiki.build(conn, cfg, client_factory=factory)
    event = dict(conn.execute("SELECT * FROM wiki_page WHERE kind='event' ORDER BY id LIMIT 1").fetchone())
    members = [r[0] for r in conn.execute("SELECT media_id FROM wiki_member WHERE page_id=?", (event["id"],))]
    assert wiki.mark_stale(conn).stale == 0

    # One changed caption in a ten-item event is a tenth: the event is stale, the topic not yet.
    conn.execute("UPDATE doc SET json=json_set(json, '$.summary', 'A different reading.') WHERE media_id=?", (members[0],))
    conn.commit()
    wiki.mark_stale(conn)
    assert page(conn, event["slug"])["stale"] == 1
    assert page(conn, kind="topic")["stale"] == 0

    # Rewriting the event makes its topic stale in turn, and the theme after that.
    wiki.build(conn, cfg, kinds=["event"], client_factory=factory)
    assert page(conn, event["slug"])["stale"] == 0
    wiki.mark_stale(conn)
    assert page(conn, kind="topic")["stale"] == 1
    wiki.build(conn, cfg, kinds=["topic"], client_factory=factory)
    wiki.mark_stale(conn)
    assert page(conn, kind="theme")["stale"] == 1


def test_update_attaches_new_items_and_resplits_waiting_ones(env, monkeypatch):
    cfg, conn, model, factory = env
    monkeypatch.setattr(wiki, "WAIT_SPLIT", 2)
    ids = [add_item(conn, n, "2024-05-01") for n in range(10)]
    near = [add_item(conn, 100 + n, "2024-05-03") for n in range(3)]
    far = add_item(conn, 200, "2019-01-01")
    conn.commit()
    add_topic(conn, ids)
    rng = np.random.default_rng(0)
    base = rng.normal(size=768).astype(np.float32)
    other = rng.normal(size=768).astype(np.float32)
    doc_ids, vecs = [], []
    for m in ids + near + [far]:
        doc_ids.append(conn.execute("SELECT id FROM doc WHERE media_id=?", (m,)).fetchone()[0])
        v = (other if m == far else base) + rng.normal(scale=0.05, size=768).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
    embed.get_index(cfg, "docs")._write(np.stack(vecs), np.array(doc_ids, dtype=np.int64))
    wiki.build(conn, cfg, kinds=["event"], write=False, client_factory=factory)
    topic = page(conn, kind="topic")
    assert topic["events_split_at"]
    held = {r[0] for r in conn.execute("SELECT media_id FROM wiki_member WHERE page_id=?", (topic["id"],))}
    assert set(near) <= held and far not in held       # attached during the build's update
    assert wiki.topics_to_split(conn) == []            # they were grouped with the rest on the first split

    # Close in meaning but years away from the events' dates: they wait in the topic.
    newer = [add_item(conn, 300 + n, "2020-01-02") for n in range(2)]
    conn.commit()
    for m in newer:
        doc_ids.append(conn.execute("SELECT id FROM doc WHERE media_id=?", (m,)).fetchone()[0])
        v = base + rng.normal(scale=0.05, size=768).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
    embed._indexes.clear()
    embed.get_index(cfg, "docs")._write(np.stack(vecs), np.array(doc_ids, dtype=np.int64))
    conn.execute("UPDATE wiki_page SET events_split_at='2000-01-01 00:00:00' WHERE id=?", (topic["id"],))
    conn.execute("UPDATE wiki_member SET added_at='1999-01-01 00:00:00'")
    conn.commit()
    stats = wiki.update(conn, cfg)
    assert stats.attached == 2 and stats.attached_to_events == 0
    waiting = wiki.topics_to_split(conn)
    assert [(t["id"], ids_) for t, ids_ in waiting] == [(topic["id"], sorted(newer))]
    wiki.build(conn, cfg, kinds=["event"], write=False, client_factory=factory)
    assert wiki.topics_to_split(conn) == []


def test_export_writes_obsidian_markdown(env, tmp_path):
    cfg, conn, model, factory = env
    ids = [add_item(conn, n, "2024-06-01") for n in range(6)]
    conn.execute("INSERT INTO source(media_id, url) VALUES (?, 'https://x.com/a/status/1')", (ids[0],))
    conn.commit()
    add_topic(conn, ids)
    wiki.build(conn, cfg, client_factory=factory)
    out = tmp_path / "wiki"
    (out).mkdir()
    (out / "old.md").write_text("gone after the swap")
    n = wiki.export(conn, cfg, out)
    assert n == conn.execute("SELECT COUNT(*) FROM wiki_page WHERE body_md != ''").fetchone()[0]
    assert not (out / "old.md").exists() and (out / "overview.md").is_file()
    topic = page(conn, kind="topic")
    text = (out / f"{topic['slug']}.md").read_text()
    assert text.startswith("---\ntitle: ") and "kind: topic" in text
    assert re.search(r"\[\[event/[^|\]]+\|[^\]]+\]\]", text)
    assert "[m:" not in text and "[^1]" in text
    assert re.search(r"^\[\^1\]: Caption title \d+ — 2024-06-01 .*http://127\.0\.0\.1:\d+/#/item/\d+$", text, flags=re.M)


def test_render_markdown_escapes_html_and_links_citations(env):
    cfg, conn, model, factory = env
    a = add_item(conn, 1, "2024-07-01")
    conn.execute("INSERT INTO wiki_page(slug, kind, title) VALUES ('event/2024-07-x', 'event', 'The <b>march</b>')")
    conn.commit()
    html, cites, slugs = wiki.render_markdown(
        conn, f"## Head\n\nA <script>x</script> claim [m:{a}] again [m:{a}] bogus [m:424242]; see [p:event/2024-07-x] [p:nope].")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert html.count(f'<a class="cite" data-media="{a}" href="#/item/{a}"') == 2 and ">1</a>" in html
    assert "424242" not in html and "nope" not in html
    assert '<a class="wikilink" data-slug="event/2024-07-x" href="#/wiki/event/2024-07-x">The &lt;b&gt;march&lt;/b&gt;</a>' in html
    assert cites == [{"n": 1, "media_id": a, "title": "Caption title 1", "date": "2024-07-01"}]
    assert slugs == ["event/2024-07-x"]
