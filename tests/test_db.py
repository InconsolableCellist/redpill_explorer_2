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


def test_a_service_being_down_never_retires_an_item():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'image', 'READY'), (2, 'b', 'image', 'READY')")
    # The vision server was refused three times for item 1; item 2's own file is unreadable.
    for _ in range(3):
        db.upsert_failure(conn, "vlm_image", 1, None, "VLMError: ConnectError: [Errno 111] Connection refused")
        db.upsert_failure(conn, "vlm_image", 2, None, "OSError: [Errno 5] Input/output error")
    kinds = dict(conn.execute("SELECT media_id, kind FROM failure WHERE stage='vlm_image'"))
    assert kinds == {1: "transient", 2: "item"}
    assert db.pending_failures(conn, "vlm_image") == {2}
    # Rows written before failures were classified are labelled on the next migration.
    conn.execute("UPDATE failure SET kind=NULL")
    assert db.classify_failures(conn) == 2
    assert db.pending_failures(conn, "vlm_image") == {2}
    # Each kind in turn, so the pattern list keeps covering what the services actually say.
    for text, kind in [("VLMError: ReadTimeout", "transient"),
                       ("HTTPStatusError: Server error '500 Internal Server Error'", "transient"),
                       ("{\"message\":\"Context size has been exceeded.\"}", "transient"),
                       ("AskError: chat model unavailable: connect", "transient"),
                       ("VLMLengthError: unparseable response (finish_reason=length)", "item"),
                       ("HTTPStatusError: Client error '400 Bad Request'", "item")]:
        assert db.failure_kind(text) == kind, text
