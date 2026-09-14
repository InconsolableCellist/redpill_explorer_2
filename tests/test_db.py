from re2 import db


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_fts_finds_inserted_doc():
    conn = make_conn()
    conn.execute("INSERT INTO media(uid, kind, status) VALUES ('abc', 'image', 'READY')")
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (1, 'caption', 'hello world')")
    rows = conn.execute("SELECT rowid FROM doc_fts WHERE doc_fts MATCH 'hello'").fetchall()
    assert [r[0] for r in rows] == [1]


def test_fts_follows_update_and_delete():
    conn = make_conn()
    conn.execute("INSERT INTO media(uid, kind, status) VALUES ('abc', 'image', 'READY')")
    conn.execute("INSERT INTO doc(media_id, kind, text) VALUES (1, 'caption', 'hello world')")
    conn.execute("UPDATE doc SET text='goodbye world' WHERE id=1")
    assert conn.execute("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'hello'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'goodbye'").fetchone()[0] == 1
    conn.execute("DELETE FROM doc WHERE id=1")
    assert conn.execute("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'goodbye'").fetchone()[0] == 0


def test_failure_helpers():
    conn = make_conn()
    for _ in range(3):
        db.upsert_failure(conn, "thumb", 7, None, "boom")
    db.upsert_failure(conn, "thumb", 8, None, "boom")
    assert conn.execute("SELECT count(*) FROM failure").fetchone()[0] == 2
    assert db.pending_failures(conn, "thumb") == {7}
    db.clear_failure(conn, "thumb", 7)
    assert db.pending_failures(conn, "thumb") == set()
