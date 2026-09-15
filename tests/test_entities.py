from re2 import db, entities, extract_images


def test_keys():
    assert entities.entity_key("The White House", "organization") == entities.entity_key("White House", "platform")
    assert entities.entity_key("President Joe Biden", "person") == entities.entity_key("Joe Biden", "person")
    assert entities.entity_key("Dr. Anthony Fauci", "person") == entities.entity_key("anthony fauci", "person")
    assert entities.entity_key("White House", "place") != entities.entity_key("White House", "organization")
    assert entities.entity_key("Trump", "person") != entities.entity_key("Donald Trump", "person")
    assert entities.is_placeholder("None") and not entities.is_placeholder("Nonesuch")


def test_merge_repoints_and_sticks():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'image', 'READY'), (2, 'b', 'image', 'READY'),
                                                         (3, 'c', 'image', 'READY');
        INSERT INTO entity(id, name, type) VALUES (1, 'CNN', 'organization'), (2, 'CNN', 'platform'),
                                                  (3, 'none', 'person'), (4, 'The White House', 'organization'),
                                                  (5, 'White House', 'organization');
        INSERT INTO media_entity(media_id, entity_id) VALUES (1, 1), (2, 1), (3, 2), (1, 2), (2, 3), (3, 4), (1, 5), (2, 5);
    """)
    dry = entities.merge(conn, dry_run=True)
    assert (dry.merged, dry.placeholders) == (2, 1)
    assert conn.execute("SELECT count(*) FROM media_entity").fetchone()[0] == 8  # dry run changes nothing
    entities.merge(conn)
    rows = {tuple(r) for r in conn.execute("SELECT media_id, entity_id FROM media_entity")}
    assert rows == {(1, 1), (2, 1), (3, 1), (1, 5), (2, 5), (3, 5)}
    # A later extraction naming the merged-away spelling lands on the canonical entity.
    extract_images.store_entities(conn, 3, {"platforms": ["CNN"], "people": ["None"]})
    assert conn.execute("SELECT count(*) FROM media_entity WHERE media_id=3").fetchone()[0] == 2
    assert entities.merge(conn).merged == 0
