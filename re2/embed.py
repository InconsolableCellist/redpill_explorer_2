"""Dense embeddings: client for the nomic-embed server on cube and .npy indexes."""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from re2 import runner

log = logging.getLogger(__name__)

EMBED_TIMEOUT = 60.0
RETRIES = 2
RETRY_SLEEP = 5.0
BATCH = 32
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbedError(Exception):
    pass


def embed_texts(client: httpx.Client, cfg, texts: list[str], prefix: str) -> np.ndarray:
    """float32, L2-normalized, shape (len(texts), dims)."""
    out = np.zeros((len(texts), cfg.embed_dims), dtype=np.float32)
    for start in range(0, len(texts), BATCH):
        chunk = [prefix + (t or " ")[: cfg.embed_max_chars] for t in texts[start:start + BATCH]]
        for attempt in range(RETRIES + 1):
            try:
                resp = client.post(cfg.embed_url, json={"input": chunk, "model": cfg.embed_model},
                                   timeout=EMBED_TIMEOUT)
                if resp.status_code >= 500 and attempt < RETRIES:
                    time.sleep(RETRY_SLEEP)
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as exc:
                if attempt < RETRIES:
                    time.sleep(RETRY_SLEEP)
                    continue
                raise EmbedError(f"timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                raise EmbedError(f"{type(exc).__name__}: {exc}") from exc
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        vecs = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        if vecs.shape != (len(chunk), cfg.embed_dims):
            raise EmbedError(f"unexpected embedding shape {vecs.shape}")
        out[start:start + len(chunk)] = vecs
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


class Index:
    """A matrix of vectors plus a parallel id array, stored as two .npy files.

    Writes go to temp files that replace the originals atomically, so readers
    (the web server) never see a half-written index.
    """

    def __init__(self, directory: Path | str, name: str, dims: int = 768):
        self.dir = Path(directory)
        self.name = name
        self.dims = dims
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._vecs = np.zeros((0, dims), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)

    @property
    def vec_path(self) -> Path:
        return self.dir / f"{self.name}.npy"

    @property
    def ids_path(self) -> Path:
        return self.dir / f"{self.name}_ids.npy"

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            try:
                st = self.ids_path.stat()
            except FileNotFoundError:
                return self._vecs, self._ids
            stamp = (st.st_mtime_ns, st.st_ino)
            if stamp != self._stamp:
                ids = np.load(self.ids_path)
                vecs = np.load(self.vec_path)
                # A writer may have replaced one file but not yet the other; keep the old pair.
                if len(ids) == len(vecs):
                    self._vecs, self._ids, self._stamp = vecs, ids, stamp
            return self._vecs, self._ids

    def __len__(self) -> int:
        return len(self.load()[1])

    def id_set(self) -> set[int]:
        return set(self.load()[1].tolist())

    def _write(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_paths = []
        try:
            # Vectors first, ids last: load() keys off the ids file.
            for target, arr in ((self.vec_path, vecs), (self.ids_path, ids)):
                fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp-", suffix=".npy")  # np.save appends .npy otherwise
                os.close(fd)
                tmp_paths.append(tmp)
                np.save(tmp, arr)
                os.replace(tmp, target)
        finally:
            for tmp in tmp_paths:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        with self._lock:
            st = self.ids_path.stat()
            self._vecs, self._ids, self._stamp = vecs, ids, (st.st_mtime_ns, st.st_ino)

    def append(self, ids, vecs: np.ndarray) -> None:
        cur_vecs, cur_ids = self.load()
        ids = np.asarray(ids, dtype=np.int64)
        keep = ~np.isin(cur_ids, ids)
        self._write(np.vstack([cur_vecs[keep], vecs.astype(np.float32)]), np.concatenate([cur_ids[keep], ids]))

    def remove(self, ids) -> int:
        cur_vecs, cur_ids = self.load()
        keep = ~np.isin(cur_ids, np.asarray(list(ids), dtype=np.int64))
        removed = int((~keep).sum())
        if removed:
            self._write(cur_vecs[keep], cur_ids[keep])
        return removed

    def vectors_for(self, ids) -> tuple[np.ndarray, np.ndarray]:
        vecs, cur_ids = self.load()
        mask = np.isin(cur_ids, np.asarray(list(ids), dtype=np.int64))
        return vecs[mask], cur_ids[mask]

    def search(self, qvec: np.ndarray, k: int = 50, allowed_ids=None) -> list[tuple[int, float]]:
        vecs, ids = self.load()
        if allowed_ids is not None:
            mask = np.isin(ids, np.fromiter(allowed_ids, dtype=np.int64))
            vecs, ids = vecs[mask], ids[mask]
        if len(ids) == 0:
            return []
        scores = vecs @ qvec.astype(np.float32)
        k = min(k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top]


_indexes: dict[tuple[str, str], Index] = {}


def get_index(cfg, name: str) -> Index:
    key = (str(cfg.data_dir), name)
    if key not in _indexes:
        _indexes[key] = Index(Path(cfg.data_dir) / "index", name, cfg.embed_dims)
    return _indexes[key]


_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_query_lock = threading.Lock()
QUERY_CACHE_SIZE = 256


def query_vector(client: httpx.Client, cfg, text: str) -> np.ndarray:
    """Embedding for a search query; cached because typing re-sends prefixes."""
    with _query_lock:
        if text in _query_cache:
            _query_cache.move_to_end(text)
            return _query_cache[text]
    vec = embed_texts(client, cfg, [text], QUERY_PREFIX)[0]
    with _query_lock:
        _query_cache[text] = vec
        while len(_query_cache) > QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vec


@dataclass
class EmbedStats:
    selected: int = 0
    embedded: int = 0
    removed: int = 0
    failed_batches: int = 0
    interrupted: bool = False

    def summary(self) -> str:
        return (f"selected={self.selected} embedded={self.embedded} removed={self.removed} "
                f"failed_batches={self.failed_batches}" + (" INTERRUPTED" if self.interrupted else ""))


FLUSH_EVERY = 50  # batches


def _docs_to_index(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Docs that belong in the index: no legacy caption once a new caption exists, nothing EXCLUDED."""
    return [(r[0], r[1]) for r in conn.execute(
        """SELECT d.id, d.text FROM doc d JOIN media m ON m.id=d.media_id
           WHERE m.status != 'EXCLUDED' AND trim(d.text) != ''
             AND NOT (d.kind='legacy_caption' AND EXISTS (
                 SELECT 1 FROM doc c WHERE c.media_id=d.media_id AND c.kind='caption'))
           ORDER BY d.id""")]


def build_docs(conn: sqlite3.Connection, cfg, limit: int | None = None, rebuild: bool = False,
               client_factory=httpx.Client) -> EmbedStats:
    stats = EmbedStats()
    index = get_index(cfg, "docs")
    if rebuild:
        index.remove(index.id_set())
    wanted = _docs_to_index(conn)
    wanted_ids = {doc_id for doc_id, _ in wanted}
    stale = index.id_set() - wanted_ids  # superseded legacy captions, deleted docs
    if stale:
        stats.removed = index.remove(stale)
    have = index.id_set()
    todo = [(doc_id, text) for doc_id, text in wanted if doc_id not in have]
    if limit is not None:
        todo = todo[:limit]
    stats.selected = len(todo)

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    local = threading.local()
    clients: list[httpx.Client] = []
    lock = threading.Lock()
    pending_ids: list[np.ndarray] = []
    pending_vecs: list[np.ndarray] = []

    def flush() -> None:
        if pending_ids:
            index.append(np.concatenate(pending_ids), np.vstack(pending_vecs))
            pending_ids.clear()
            pending_vecs.clear()

    def work(batch):
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        return embed_texts(client, cfg, [t for _, t in batch], DOC_PREFIX)

    def on_result(batch, vecs):
        pending_ids.append(np.asarray([doc_id for doc_id, _ in batch], dtype=np.int64))
        pending_vecs.append(vecs)
        stats.embedded += len(batch)
        if len(pending_ids) >= FLUSH_EVERY:
            flush()

    def on_error(batch, exc):
        stats.failed_batches += 1
        log.warning("embedding batch starting at doc %s failed: %s", batch[0][0], exc)

    try:
        stats.interrupted = not runner.run_jobs(batches, work, on_result, on_error, 4,
                                                total=len(batches), desc="embed")
    finally:
        flush()
        for client in clients:
            client.close()
    return stats
