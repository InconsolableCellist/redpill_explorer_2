import json
import zlib
from types import SimpleNamespace

import httpx
import numpy as np

from re2 import db, embed


DIMS = 64


def fake_vector(text: str, dims: int = DIMS) -> list[float]:
    """Deterministic bag-of-words vector (crc32, not hash(), which is randomized per process)."""
    v = np.zeros(dims)
    for word in text.split():
        v[zlib.crc32(word.strip(".,").lower().encode()) % dims] += 1
    return v.tolist()


def mock_client(calls=None):
    def handler(request):
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body["input"])
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": fake_vector(t.split(": ", 1)[1])} for i, t in enumerate(body["input"])]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_cfg(tmp_path):
    return SimpleNamespace(data_dir=tmp_path, embed_url="http://embed.test/v1/embeddings",
                           embed_model="nomic", embed_dims=DIMS, embed_max_chars=50)


def test_embed_texts_normalizes_prefixes_and_truncates(tmp_path):
    calls = []
    vecs = embed.embed_texts(mock_client(calls), make_cfg(tmp_path), ["alpha beta", "x" * 100], "search_document: ")
    assert vecs.shape == (2, DIMS) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1)
    assert calls[0][0] == "search_document: alpha beta"
    assert len(calls[0][1]) == len("search_document: ") + 50


def test_index_append_remove_search(tmp_path):
    index = embed.Index(tmp_path, "t", dims=3)
    index.append([1, 2, 3], np.eye(3, dtype=np.float32))
    assert index.search(np.array([0, 1, 0.1]), k=2)[0][0] == 2
    assert [i for i, _ in index.search(np.array([0, 1, 0]), k=5, allowed_ids={1, 3})] in ([1, 3], [3, 1])
    index.append([2], np.array([[1, 0, 0]], dtype=np.float32))  # replaces id 2
    assert len(index) == 3
    assert index.remove([1, 99]) == 1
    fresh = embed.Index(tmp_path, "t", dims=3)  # another process sees the same files
    assert sorted(fresh.id_set()) == [2, 3]


def test_build_docs_skips_superseded_legacy_captions(tmp_path):
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript("""
        INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'image', 'READY'), (2, 'b', 'image', 'READY'),
                                                        (3, 'c', 'other', 'EXCLUDED');
        INSERT INTO doc(id, media_id, kind, text) VALUES
            (10, 1, 'legacy_caption', 'old words'), (11, 2, 'legacy_caption', 'ballot fraud claims'),
            (12, 3, 'legacy_caption', 'secret');
    """)
    cfg = make_cfg(tmp_path)
    stats = embed.build_docs(conn, cfg, client_factory=mock_client)
    index = embed.get_index(cfg, "docs")
    assert stats.embedded == 2 and index.id_set() == {10, 11}

    conn.execute("INSERT INTO doc(id, media_id, kind, text) VALUES (13, 1, 'caption', 'new words')")
    stats = embed.build_docs(conn, cfg, client_factory=mock_client)
    assert (stats.embedded, stats.removed) == (1, 1)
    assert index.id_set() == {11, 13}
    q = embed.query_vector(mock_client(), cfg, "ballot fraud")
    assert index.search(q, k=1)[0][0] == 11
