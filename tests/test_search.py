import pytest

from re2 import db, search


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    c.executescript("""
        INSERT INTO media(id, uid, kind, status, legacy_caption) VALUES
          (1, 'a', 'image', 'READY', 'The image is a screenshot of a tweet about Ukraine missiles.'),
          (2, 'b', 'video', 'READY', NULL),
          (3, 'c', 'image', 'SOURCE_OFFLINE', 'Ballot boxes in Georgia.'),
          (4, 'd', 'image', 'READY', 'Georgia ballot drop box footage.'),
          (5, 'e', 'other', 'EXCLUDED', NULL);
        INSERT INTO media_path(media_id, rel_path, readable, first_seen, last_seen) VALUES
          (1, 'ukraine/a.png', 1, 'x', 'x'), (2, 'ukraine/clip.mp4', 1, 'x', 'x'),
          (3, 'voter fraud/c.jpg', 0, 'x', 'x'), (4, 'd.jpg', 1, 'x', 'x'), (5, 'cookies.txt', 1, 'x', 'x');
        INSERT INTO doc(id, media_id, kind, text) VALUES
          (10, 1, 'legacy_caption', 'A tweet about Ukraine using long range missiles in Russia'),
          (11, 2, 'transcript', 'Trudeau said Ukraine may use the missiles'),
          (12, 3, 'legacy_caption', 'Ballot boxes in Georgia with election workers'),
          (13, 4, 'legacy_caption', 'Georgia ballot drop box footage from the election'),
          (14, 5, 'legacy_caption', 'missiles secret cookie');
        UPDATE doc SET start_s=30, end_s=60 WHERE id=11;
        INSERT INTO entity(id, name, type) VALUES (1, 'Ukraine', 'place'), (2, 'Georgia', 'place'),
                                                 (3, 'Justin Trudeau', 'person');
        INSERT INTO media_entity(media_id, entity_id) VALUES (1, 1), (2, 1), (2, 3), (3, 2), (4, 2);
        INSERT INTO source(media_id, url, date, folder, restricted, role) VALUES
          (1, 'https://x.com/a/1', '2024-11-18T10:00:00-07:00', 'ukraine', 0, 'screenshot'),
          (4, 'https://x.com/b/2', '2021-01-05T10:00:00-07:00', '.', 0, 'screenshot');
    """)
    return c


def ids(results):
    return [r[0] for r in results]


def test_fts_query_prefix_and_phrases():
    assert search.fts_query("ukr") == '"ukr"*'
    assert search.fts_query("trudeau ukr") == '"trudeau" "ukr"*'
    assert search.fts_query("trudeau ") == '"trudeau"'
    assert search.fts_query('"long range" missi') == '"long range" "missi"*'
    assert search.fts_query('a " ') == ""


def test_lexical_prefix_match_and_exclusion(conn):
    assert set(ids(search.lexical(conn, "missi"))) == {10, 11}  # EXCLUDED doc 14 never appears
    assert ids(search.lexical(conn, '"drop box"')) == [13]
    search.lexical(conn, 'weird ( " AND NEAR(')  # malformed input must not raise


def test_lexical_falls_back_to_or(conn):
    assert set(ids(search.lexical(conn, "ballot trudeau"))) >= {11, 12, 13}


def test_scopes_intersect(conn):
    assert search.scope_media_ids(conn, []) is None
    assert search.scope_media_ids(conn, ["entity:1"]) == {1, 2}
    assert search.scope_media_ids(conn, ["entity:1", "kind:video"]) == {2}
    assert search.scope_media_ids(conn, ["folder:ukraine"]) == {1, 2}
    assert search.scope_media_ids(conn, ["folder:."]) == {4}
    assert search.scope_media_ids(conn, ["date:2024..2024-12"]) == {1}
    assert search.scope_media_ids(conn, ["date:..2022"]) == {4}
    assert search.scope_media_ids(conn, ["text:missiles"]) == {1, 2}
    assert search.scope_media_ids(conn, ["entity:2", "status:READY"]) == {4}
    with pytest.raises(search.ScopeError):
        search.scope_media_ids(conn, ["nope:1"])
    with pytest.raises(search.ScopeError):
        search.scope_media_ids(conn, ["date:yesterday"])


def test_lexical_within_scope(conn):
    allowed = search.scope_media_ids(conn, ["kind:video"])
    assert ids(search.lexical(conn, "missiles", allowed=allowed)) == [11]


def test_fuse_prefers_docs_in_both_lists():
    fused = search.fuse([[(1, 9.0), (2, 8.0), (3, 7.0)], [(3, 0.9), (4, 0.8), (1, 0.7)]])
    assert ids(fused)[:2] == [1, 3]


def test_group_by_media_snippets_and_cards(conn):
    ranked = search.lexical(conn, "missiles")
    hits, order = search.group_by_media(conn, ranked, "missiles")
    assert set(order) == {1, 2}
    by_media = {h.media_id: h for h in hits}
    assert search.MARK_START + "missiles" + search.MARK_END in by_media[1].snippet
    assert (by_media[2].start_s, by_media[2].end_s) == (30, 60)
    cards = search.cards(conn, [1, 3])
    assert cards[1]["title"] == "Tweet about Ukraine missiles"
    assert cards[1]["date"] == "2024-11-18" and cards[1]["folder"] == "ukraine"
    assert cards[1]["media_url"] == "/media/1/file"
    assert cards[3]["media_url"] is None  # offline items never get a file URL


def test_lenses_and_correlations(conn):
    lens = search.lenses(conn, [1, 2])
    assert lens["entities"][0]["name"] == "Ukraine" and lens["entities"][0]["n"] == 2
    assert {"folder": "ukraine", "n": 2} in lens["folders"]
    assert lens["kinds"] == {"image": 1, "video": 1}
    assert lens["months"] == [{"month": "2024-11", "n": 1}]
    corr = search.correlations(conn, [1, 2, 3, 4], min_count=2)
    # Ukraine and Georgia each cover half the results but also half the corpus: no lift.
    assert corr == []
    corr = search.correlations(conn, [1, 2], min_count=2)
    assert corr[0]["scope"] == "entity:1"


def test_items_in_scope_orders_by_date(conn):
    assert search.items_in_scope(conn, None)[:2] == [1, 4]


def test_cooccurrence_graph(conn):
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (6, 'f', 'image', 'READY'), (7, 'g', 'image', 'READY'),
                                                         (8, 'h', 'image', 'READY');
        INSERT INTO media_entity(media_id, entity_id) VALUES (1, 3), (6, 1), (6, 3), (7, 1), (7, 3), (8, 2);
    """)
    g = search.cooccurrence(conn, None, kind="entity", min_pair=2)
    names = {n["id"]: n["name"] for n in g["nodes"]}
    assert set(names.values()) == {"Ukraine", "Georgia", "Justin Trudeau"}
    assert len(g["edges"]) == 1
    edge = g["edges"][0]
    assert {names[edge["a"]], names[edge["b"]]} == {"Ukraine", "Justin Trudeau"} and edge["n"] == 4 and edge["w"] > 0
    scoped = search.cooccurrence(conn, {6, 7}, kind="entity", min_pair=2)
    assert scoped["n_items"] == 2


def test_selection_scope_expands_posts(conn):
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (6, 'f', 'image', 'READY');
        INSERT INTO post(id, url, primary_media_id) VALUES (1, 'https://x.com/a/1', 1);
        INSERT INTO post_member(post_id, media_id, role, seq) VALUES (1, 1, 'screenshot', 0), (1, 6, 'photo', 1);
    """)
    sel = search.save_selection(conn, [4, 1, 1])
    assert search.save_selection(conn, [1, 4]) == sel          # the same set reuses its row
    assert search.scope_media_ids(conn, [f"sel:{sel}"]) == {1, 4, 6}
    assert search.scope_media_ids(conn, [f"sel:{sel}", "entity:2"]) == {4}
    assert search.scope_label(conn, f"sel:{sel}") == "2 selected items"
    with pytest.raises(search.ScopeError):
        search.scope_media_ids(conn, ["sel:0000000000"])
    with pytest.raises(search.ScopeError):
        search.scope_media_ids(conn, ["sel:../../etc"])
    with pytest.raises(search.ScopeError):
        search.save_selection(conn, [])
    with pytest.raises(search.ScopeError):
        search.save_selection(conn, range(search.MAX_SELECTION + 1))


def test_lookup_tags_and_entities(conn):
    conn.executescript("""
        INSERT INTO tag(id, name) VALUES (1, 'georgia ballots'), (2, 'ukraine war'), (3, 'unused ukraine');
        INSERT INTO media_tag(media_id, tag_id, origin) VALUES (3, 1, 'vlm'), (1, 2, 'vlm');
    """)
    found = search.lookup(conn, "ukr")
    # The most used first; names nothing is tagged with are left out.
    assert [(r["scope"], r["n"]) for r in found] == [("entity:1", 2), ("tag:2", 1)]
    assert [r["name"] for r in search.lookup(conn, "GEORGIA")] == ["Georgia", "georgia ballots"]
    # An exact name beats a more common partial match.
    conn.execute("INSERT INTO tag(id, name) VALUES (4, 'georgi')")
    conn.execute("INSERT INTO media_tag(media_id, tag_id, origin) VALUES (4, 4, 'vlm')")
    search._overall_cache.clear()
    assert [r["name"] for r in search.lookup(conn, "georgi")] == ["georgi", "Georgia", "georgia ballots"]
    assert search.lookup(conn, "  ") == [] and search.lookup(conn, "%") == []


def test_post_author_and_tweet_from_ocr():
    from re2 import search

    assert search.post_author("https://x.com/AgroNationalism/status/2016898765425197350") == "AgroNationalism"
    assert search.post_author("https://twitter.com/jack/status/20") == "jack"
    assert search.post_author("https://x.com/i/status/2085688589975704030") is None
    assert search.post_author("https://t.me/somechannel/12") is None
    ocr = ("← Post\nVanRaalte, Agro-Nationalist\n@AgroNationalism\nThe kids have a snow day.\nYou kiss your wife.\n"
           "7:40 AM · Jan 29, 2026 · 1.2M Views\n12 replies")
    assert search.tweet_from_ocr(ocr) == ("VanRaalte, Agro-Nationalist", "AgroNationalism",
                                          "The kids have a snow day.\nYou kiss your wife.")
    assert search.tweet_from_ocr("A meme with no tweet in it") == (None, None, None)
    assert search.tweet_from_ocr(None) == (None, None, None)
