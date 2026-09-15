import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from re2 import api, config, db, search


@pytest.fixture
def env(tmp_path):
    cfg = dataclasses.replace(config.load(), corpus_root=tmp_path / "corpus", data_dir=tmp_path / "data",
                              thumbs_dir=tmp_path / "thumbs", embed_url="http://127.0.0.1:9/v1/embeddings")
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    for i in range(1, 6):
        conn.execute("INSERT INTO media(id, uid, kind, status) VALUES (?, ?, 'image', 'READY')", (i, f"u{i}"))
        conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (?, 'legacy_caption', ?)",
                     (i, f"A tweet about long range missiles number {i}"))
    # Item 5 is a photo in the post whose screenshot is item 4.
    conn.execute("INSERT INTO post(id, url, primary_media_id) VALUES (1, 'https://x.com/a/status/1', 4)")
    conn.executemany("INSERT INTO post_member(post_id, media_id, role, seq) VALUES (1, ?, ?, ?)",
                     [(4, "screenshot", 0), (5, "photo", 1)])
    pages = [("overview", "overview", "The archive", None, "Everything.", ""),
             ("theme/war", "theme", "War and weapons", "overview", "Wars.", "Wars, see [p:topic/missiles]."),
             ("topic/missiles", "topic", "Long-range missiles", "theme/war", "Missiles.",
              "Missile debates [m:1][m:2].\n\n## Claims\n\nA post says <script>alert(1)</script> [m:4] "
              "and [p:event/2024-11-strike] and [p:nowhere]."),
             ("event/2024-11-strike", "event", "Strike on a depot", "topic/missiles", "A strike.", "")]
    ids = {}
    for slug, kind, title, parent, summary, body in pages:
        ids[slug] = conn.execute(
            """INSERT INTO wiki_page(slug, kind, title, parent_slug, summary, body_md, members_read, members_total,
                                     members_cited, built_at, stale)
               VALUES (?, ?, ?, ?, ?, ?, 3, 3, 3, CASE WHEN ?='' THEN NULL ELSE '2026-09-15 10:00:00' END, 0)""",
            (slug, kind, title, parent, summary, body, body)).lastrowid
    for m in (1, 2, 4):
        conn.execute("INSERT INTO wiki_member(page_id, media_id, read_hash) VALUES (?, ?, 'h')", (ids["topic/missiles"], m))
        conn.execute("INSERT INTO wiki_member(page_id, media_id) VALUES (?, ?)", (ids["theme/war"], m))
    conn.execute("INSERT INTO wiki_member(page_id, media_id) VALUES (?, 1), (?, 2)",
                 (ids["event/2024-11-strike"], ids["event/2024-11-strike"]))
    conn.execute("INSERT INTO wiki_citation(page_id, media_id, n) VALUES (?, 4, 1)", (ids["topic/missiles"],))
    conn.execute("INSERT INTO wiki_link(from_page, to_page) VALUES (?, ?)", (ids["theme/war"], ids["topic/missiles"]))
    conn.commit()
    conn.close()
    return TestClient(api.create_app(cfg)), cfg


def test_pages_list_counts_and_order(env):
    client, _ = env
    data = client.get("/api/wiki/pages").json()
    assert [p["kind"] for p in data["pages"]] == ["overview", "theme", "topic", "event"]
    assert data["counts"]["event"] == {"pages": 1, "written": 0, "stale": 0}
    topic = next(p for p in data["pages"] if p["slug"] == "topic/missiles")
    assert topic["members"] == 3 and topic["written"] == 1 and "body_md" not in topic
    assert client.get("/api/wiki/pages", params={"kind": "bogus"}).status_code == 400
    assert [p["slug"] for p in client.get("/api/wiki/pages", params={"kind": "event"}).json()["pages"]] == \
        ["event/2024-11-strike"]


def test_page_renders_citations_links_and_escapes_html(env):
    client, _ = env
    resp = client.get("/api/wiki/page/topic/missiles")
    assert resp.status_code == 200
    page = resp.json()
    html = page["html"]
    assert '<a class="cite" data-media="1" href="#/item/1"' in html and ">1</a>" in html
    assert 'data-media="4"' in html and ">3</a>" in html          # numbered in order of first use
    assert '<a class="wikilink" data-slug="event/2024-11-strike" href="#/wiki/event/2024-11-strike">Strike on a depot</a>' in html
    assert "nowhere" not in html                                   # unknown slugs vanish
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert [c["media_id"] for c in page["cited"]] == [1, 2, 4] and page["cited"][0]["card"]["media_id"] == 1
    assert [c["slug"] for c in page["breadcrumbs"]] == ["overview", "theme/war"]
    assert [c["slug"] for c in page["children"]] == ["event/2024-11-strike"]
    assert [b["slug"] for b in page["backlinks"]] == ["theme/war"]
    assert {c["media_id"] for c in page["member_cards"]} == {1, 2, 4}
    assert page["members_read"] == 3 and page["has_body"] is True and page["members_changed"] is not None
    assert client.get("/api/wiki/page/topic/nope").status_code == 404


def test_unwritten_page_has_no_body(env):
    client, _ = env
    page = client.get("/api/wiki/page/event/2024-11-strike").json()
    assert page["has_body"] is False and page["html"] == "" and page["written"] == 0


def test_item_lists_its_pages_including_through_its_post(env):
    client, _ = env
    pages = client.get("/api/item/1").json()["pages"]
    assert [p["slug"] for p in pages] == ["event/2024-11-strike", "topic/missiles", "theme/war"]
    photo = client.get("/api/item/5").json()["pages"]          # the photo's post (item 4) is on the topic page
    assert {p["slug"] for p in photo} == {"topic/missiles", "theme/war"}
    assert next(p for p in photo if p["slug"] == "topic/missiles")["cited"] == 1


def test_stream_pages_message_finds_written_pages(env):
    client, _ = env
    msgs = [json.loads(line) for line in client.get("/api/stream", params={"q": "missiles"}).text.splitlines() if line]
    pages = next(m for m in msgs if m["t"] == "pages")["hits"]
    assert pages[0]["slug"] == "topic/missiles"          # the title match ranks first
    assert "event/2024-11-strike" not in {p["slug"] for p in pages}   # unwritten pages are not offered


def test_page_scope_includes_post_members(env):
    _, cfg = env
    conn = db.connect(cfg.db_path)
    assert search.scope_media_ids(conn, ["page:topic/missiles"]) == {1, 2, 4, 5}
    assert search.scope_label(conn, "page:topic/missiles") == "Long-range missiles"
