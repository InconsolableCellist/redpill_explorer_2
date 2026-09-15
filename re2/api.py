"""FastAPI server: streaming search, item detail, confined media files, legacy pages."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from re2 import config, db, embed, search
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

HITS_PER_PAGE = 40
DASHBOARD = REPO_ROOT / "dashboard" / "index.html"
PUBLIC = REPO_ROOT / "public"
TEMPLATES = REPO_ROOT / "templates"


def open_ro(cfg) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def confined_path(cfg, rel_path: str) -> Path | None:
    """Absolute path under the corpus root, or None if it resolves anywhere else."""
    root = os.path.realpath(cfg.corpus_root)
    real = os.path.realpath(os.path.join(root, rel_path))
    if not real.startswith(root + os.sep) or not os.path.isfile(real):
        return None
    return Path(real)


def _line(obj) -> bytes:
    return (search.dumps(obj) + "\n").encode("utf-8")


def create_app(cfg: config.Config | None = None) -> FastAPI:
    cfg = cfg or config.load()
    app = FastAPI(title="Redpill Explorer 2", docs_url="/api/docs", openapi_url="/api/openapi.json")
    embed_client = httpx.Client()
    jinja = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))

    def parse_scopes(conn: sqlite3.Connection, scopes: list[str]) -> tuple[set[int] | None, list[dict]]:
        try:
            allowed = search.scope_media_ids(conn, scopes)
            labels = [{"s": s, "label": search.scope_label(conn, s)} for s in scopes]
        except (search.ScopeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return allowed, labels

    # ── Search ────────────────────────────────────────────────────────────

    def lenses_for(conn: sqlite3.Connection, order: list[int]) -> dict:
        """Lenses over the best-matching items, counting every photo and video inside matched posts."""
        top = search.expand_members(conn, order[:search.LENS_MEDIA])
        return search.lenses(conn, top, max_media=len(top))

    def check_sort(sort: str) -> str:
        if sort not in search.SORTS:
            raise HTTPException(400, f"sort must be one of {', '.join(search.SORTS)}")
        return sort

    @app.get("/api/stream")
    def stream(q: str = "", s: list[str] = Query(default=[]), sort: str = "relevance"):
        check_sort(sort)
        conn = open_ro(cfg)
        try:
            allowed, labels = parse_scopes(conn, s)
        except HTTPException:
            conn.close()
            raise

        def generate():
            timings: dict[str, int] = {}
            started = time.perf_counter()

            def lap(name: str, since: float) -> float:
                now = time.perf_counter()
                timings[name] = round((now - since) * 1000)
                return now

            try:
                yield _line({"t": "scope", "scopes": labels,
                             "n_in_scope": None if allowed is None else len(allowed)})
                tag_scopes = [int(x.split(":", 1)[1]) for x in s if x.startswith("tag:")]
                if tag_scopes:
                    from re2 import relations
                    rel = relations.related(conn, tag_scopes[-1], limit=15)
                    if rel:
                        yield _line({"t": "related", "tag": labels[s.index(f"tag:{tag_scopes[-1]}")]["label"],
                                     "data": rel})
                query = q.strip()
                if not query:
                    order = search.items_in_scope(conn, allowed, limit=search.SCOPE_LENS_MEDIA, sort=sort)
                    card_map = search.cards(conn, order[:HITS_PER_PAGE])
                    yield _line({"t": "items", "hits": [card_map[m] for m in order[:HITS_PER_PAGE] if m in card_map],
                                 "more": len(order) > HITS_PER_PAGE})
                    everything = search.expand_members(conn, order)
                    yield _line({"t": "lenses", "data": search.lenses(conn, everything,
                                                                      max_media=search.SCOPE_LENS_MEDIA * 2)})
                    if allowed is not None:
                        yield _line({"t": "correlations", "data": search.correlations(
                            conn, everything, max_media=search.SCOPE_LENS_MEDIA * 2)})
                    lap("total", started)
                    yield _line({"t": "done", "ms": timings})
                    return

                t = time.perf_counter()
                lex = search.lexical(conn, query, allowed=allowed)
                hits, order = search.group_by_media(conn, lex, query, limit=HITS_PER_PAGE, sort=sort)
                t = lap("lexical", t)
                yield _line({"t": "lexical", "hits": search.hit_dicts(conn, hits),
                             "more": len(order) > HITS_PER_PAGE})
                yield _line({"t": "lenses", "data": lenses_for(conn, order)})
                yield _line({"t": "pages", "hits": []})

                try:
                    t = time.perf_counter()
                    dn = search.dense(conn, embed_client, cfg, query, allowed=allowed)
                    t = lap("dense", t)
                except (embed.EmbedError, httpx.HTTPError, OSError) as exc:
                    log.warning("dense search unavailable: %s", exc)
                    yield _line({"t": "warn", "msg": "dense search unavailable"})
                    dn = None
                if dn is not None:
                    fused = search.fuse([lex, dn])
                    hits, order = search.group_by_media(conn, fused, query, limit=HITS_PER_PAGE, sort=sort)
                    yield _line({"t": "fused", "hits": search.hit_dicts(conn, hits),
                                 "more": len(order) > HITS_PER_PAGE})
                    yield _line({"t": "lenses", "data": lenses_for(conn, order)})
                t = time.perf_counter()
                top = search.expand_members(conn, order[:search.LENS_MEDIA])
                yield _line({"t": "correlations", "data": search.correlations(conn, top, max_media=len(top))})
                lap("correlations", t)
                lap("total", started)
                yield _line({"t": "done", "ms": timings})
            finally:
                conn.close()

        return StreamingResponse(generate(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/items")
    def items(q: str = "", s: list[str] = Query(default=[]), offset: int = 0, limit: int = HITS_PER_PAGE,
              sort: str = "relevance"):
        check_sort(sort)
        limit = max(1, min(limit, 200))
        with closing(open_ro(cfg)) as conn:
            allowed, _ = parse_scopes(conn, s)
            query = q.strip()
            if not query:
                order = search.items_in_scope(conn, allowed, limit=limit + 1, offset=offset, sort=sort)
                card_map = search.cards(conn, order[:limit])
                return {"hits": [card_map[m] for m in order[:limit] if m in card_map], "more": len(order) > limit}
            ranked = [search.lexical(conn, query, k=1000, allowed=allowed)]
            try:
                ranked.append(search.dense(conn, embed_client, cfg, query, k=1000, allowed=allowed))
            except (embed.EmbedError, httpx.HTTPError, OSError):
                pass
            hits, order = search.group_by_media(conn, search.fuse(ranked), query, limit=limit, offset=offset,
                                                sort=sort)
            return {"hits": search.hit_dicts(conn, hits), "more": len(order) > offset + limit}

    @app.get("/api/timeline")
    def timeline(q: str = "", s: list[str] = Query(default=[])):
        """Items per month by kind, for the scope (and the query's lexical matches when given)."""
        with closing(open_ro(cfg)) as conn:
            allowed, _ = parse_scopes(conn, s)
            if q.strip():
                matched = {m for m in search.group_by_media(
                    conn, search.lexical(conn, q.strip(), k=20000, allowed=allowed), q, limit=0)[1]}
                allowed = matched if allowed is None else allowed & matched
            return search.timeline_counts(conn, allowed)

    @app.get("/api/graph")
    def graph(q: str = "", s: list[str] = Query(default=[]), kind: str = "entity", limit: int = 150):
        """Co-occurrence network of entities (or tags) over the items in scope, weighted by normalized PMI."""
        if kind not in ("entity", "tag", "related"):
            raise HTTPException(400, "kind must be entity, tag or related")
        limit = max(10, min(limit, 300))
        with closing(open_ro(cfg)) as conn:
            allowed, _ = parse_scopes(conn, s)
            if q.strip():
                matched = set(search.group_by_media(
                    conn, search.lexical(conn, q.strip(), k=20000, allowed=allowed), q, limit=0)[1])
                allowed = matched if allowed is None else allowed & matched
            if kind == "related":
                from re2 import relations
                return relations.related_graph(conn, allowed, limit=limit)
            return search.cooccurrence(conn, allowed, kind=kind, limit=limit)

    map_cache: dict = {}

    def load_map(conn: sqlite3.Connection) -> dict | None:
        """Points of the latest 2D map, normalized to 0..1, with topic and theme per point. Cached per run."""
        run = conn.execute("SELECT value FROM kv WHERE key='cluster_run'").fetchone()
        coords_path = Path(cfg.data_dir) / "index" / "map2d.npy"
        if not run or not coords_path.is_file():
            return None
        if map_cache.get("run") == run[0]:
            return map_cache
        import numpy as np

        coords = np.load(coords_path)
        ids = np.load(Path(cfg.data_dir) / "index" / "map2d_ids.npy")
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        norm = (coords - lo) / np.maximum(hi - lo, 1e-9)
        topic_of = dict(conn.execute(
            """SELECT cm.media_id, c.id FROM cluster_member cm JOIN cluster c ON c.id=cm.cluster_id
               WHERE c.run_id=? AND c.level='topic'""", (run[0],)).fetchall())
        clusters = {r[0]: {"id": r[0], "level": r[1], "parent": r[2], "label": r[3]} for r in conn.execute(
            "SELECT id, level, parent_id, label FROM cluster WHERE run_id=?", (run[0],))}
        index_of = {int(m): i for i, m in enumerate(ids.tolist())}
        topic_idx = [topic_of.get(int(m), 0) for m in ids.tolist()]
        # Label positions: the median of member points, and the member count.
        sums: dict[int, list] = {}
        for i, t in enumerate(topic_idx):
            if not t:
                continue
            for c in (t, clusters[t]["parent"]):
                sums.setdefault(c, []).append(i)
        for c, members in sums.items():
            pts = norm[members]
            clusters[c].update(x=round(float(np.median(pts[:, 0])), 4), y=round(float(np.median(pts[:, 1])), 4),
                               n=len(members))
        map_cache.clear()
        map_cache.update(run=run[0], index_of=index_of, payload={
            "run": run[0],
            "ids": ids.tolist(),
            "x": np.round(norm[:, 0], 4).tolist(),
            "y": np.round(norm[:, 1], 4).tolist(),
            "topic": topic_idx,
            "clusters": [c for c in clusters.values() if "n" in c],
        })
        return map_cache

    @app.get("/api/map")
    def map_points():
        with closing(open_ro(cfg)) as conn:
            m = load_map(conn)
        if m is None:
            raise HTTPException(404, "no map yet: run `re2 recluster`")
        return m["payload"]

    @app.get("/api/map/scope")
    def map_scope(q: str = "", s: list[str] = Query(default=[])):
        """Indices into /api/map's points for the current query and scopes (all points when neither)."""
        with closing(open_ro(cfg)) as conn:
            m = load_map(conn)
            if m is None:
                raise HTTPException(404, "no map yet: run `re2 recluster`")
            allowed, _ = parse_scopes(conn, s)
            if q.strip():
                matched = set(search.group_by_media(
                    conn, search.lexical(conn, q.strip(), k=20000, allowed=allowed), q, limit=0)[1])
                allowed = matched if allowed is None else allowed & matched
            if allowed is None:
                return {"all": True, "indices": []}
            to_primary, _ = search.post_maps(conn)
            items = {to_primary.get(x, x) for x in allowed}
            return {"all": False, "indices": sorted(m["index_of"][i] for i in items if i in m["index_of"])}

    # ── Ask ───────────────────────────────────────────────────────────────

    chat_client = httpx.Client()

    @app.post("/api/ask")
    def ask_endpoint(body: dict = Body(...)):
        from re2 import ask

        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question is required")
        scopes = [str(x) for x in body.get("scopes") or []]
        conn = db.connect(cfg.db_path)   # writable: the agent logs its turns
        try:
            search.scope_media_ids(conn, scopes)
        except (search.ScopeError, ValueError) as exc:
            conn.close()
            raise HTTPException(400, str(exc)) from exc

        def events():
            try:
                for name, data in ask.run(conn, cfg, question, scopes=scopes, mode=str(body.get("mode") or "informed"),
                                          session_id=body.get("session_id") or None, client=chat_client,
                                          embed_client=embed_client):
                    yield f"event: {name}\ndata: {search.dumps(data)}\n\n".encode("utf-8")
            except Exception as exc:  # noqa: BLE001 - the stream must end with an event, not a dropped connection
                log.exception("ask failed")
                yield f"event: error\ndata: {search.dumps({'msg': str(exc)})}\n\n".encode("utf-8")
            finally:
                conn.close()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/ask/sessions")
    def ask_sessions(limit: int = 30):
        with closing(open_ro(cfg)) as conn:
            if not search._has_table(conn, "chat_session"):
                return {"sessions": []}
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at, COUNT(l.id) AS turns FROM chat_session s
                   LEFT JOIN chat_log l ON l.session_id=s.id GROUP BY s.id ORDER BY MAX(l.id) DESC LIMIT ?""",
                (max(1, min(limit, 200)),)).fetchall()
            return {"sessions": [dict(r) for r in rows]}

    @app.get("/api/ask/session/{session_id}")
    def ask_session(session_id: str):
        with closing(open_ro(cfg)) as conn:
            rows = conn.execute(
                """SELECT question, answer, citations_json, trace_json, mode, ts, latency_ms FROM chat_log
                   WHERE session_id=? ORDER BY id""", (session_id,)).fetchall()
            turns = []
            for r in rows:
                cites = json.loads(r["citations_json"] or "[]")
                ids = [int(c["ref"][2:]) for c in cites if c["ref"].startswith("m:")]
                card_map = search.cards(conn, ids)
                for c in cites:
                    if c["ref"].startswith("m:") and int(c["ref"][2:]) in card_map:
                        c["card"] = card_map[int(c["ref"][2:])]
                from re2 import ask
                turns.append({"question": r["question"], "answer": r["answer"], "citations": cites,
                              "steps": json.loads(r["trace_json"] or "[]"), "mode": r["mode"], "ts": r["ts"],
                              "sentences": ask.sentences_with_flags(r["answer"] or "")})
            return {"session_id": session_id, "turns": turns}

    # ── Items ─────────────────────────────────────────────────────────────

    @app.get("/api/item/{media_id}")
    def item(media_id: int):
        with closing(open_ro(cfg)) as conn:
            card = search.cards(conn, [media_id]).get(media_id)
            if card is None or card["status"] in ("EXCLUDED", "UNSUPPORTED"):
                raise HTTPException(404, "not found")
            paths = [dict(r) for r in conn.execute(
                "SELECT rel_path, readable, size FROM media_path WHERE media_id=? ORDER BY id", (media_id,))]
            sources = [dict(r) for r in conn.execute(
                "SELECT url, date, folder, restricted, role FROM source WHERE media_id=? ORDER BY date", (media_id,))]
            meta = conn.execute("SELECT * FROM media_meta WHERE media_id=?", (media_id,)).fetchone()
            tags = [dict(r) for r in conn.execute(
                """SELECT t.id, t.name, mt.origin, mt.weight FROM media_tag mt JOIN tag t ON t.id=mt.tag_id
                   WHERE mt.media_id=? ORDER BY mt.origin DESC, mt.weight DESC, t.name""", (media_id,))]
            entities = [dict(r) for r in conn.execute(
                """SELECT e.id, e.name, e.type FROM media_entity me JOIN entity e ON e.id=me.entity_id
                   WHERE me.media_id=? ORDER BY e.type, e.name""", (media_id,))]
            docs: dict[str, list] = {}
            for r in conn.execute(
                    """SELECT id, kind, seq, start_s, end_s, page, text, json, model, created_at FROM doc
                       WHERE media_id=? ORDER BY kind, seq""", (media_id,)):
                d = dict(r)
                d["json"] = json.loads(d["json"]) if d["json"] else None
                docs.setdefault(d["kind"], []).append(d)
            keyframes = [dict(r) for r in conn.execute(
                "SELECT seq, t_s FROM keyframe WHERE media_id=? ORDER BY seq", (media_id,))]
            for k in keyframes:
                k["url"] = f"/media/{media_id}/keyframe/{k['seq']}"
            try:
                similar_ids = search.similar(conn, cfg, media_id)
            except Exception:  # noqa: BLE001 - similar items are optional garnish
                log.exception("similar items failed for %s", media_id)
                similar_ids = []
            sim_cards = search.cards(conn, similar_ids)
            post = None
            post_row = conn.execute(
                """SELECT p.id, p.url, p.date, p.primary_media_id FROM post_member pm JOIN post p ON p.id=pm.post_id
                   WHERE pm.media_id=? ORDER BY p.id LIMIT 1""", (media_id,)).fetchone()
            if post_row:
                member_rows = conn.execute(
                    "SELECT media_id, role, seq FROM post_member WHERE post_id=? ORDER BY seq", (post_row[0],)).fetchall()
                member_cards = search.cards(conn, [r[0] for r in member_rows])
                if len(member_rows) > 1:
                    post = {"url": post_row[1], "date": post_row[2], "primary_media_id": post_row[3],
                            "members": [{**member_cards[r[0]], "role": r[1], "seq": r[2]}
                                        for r in member_rows if r[0] in member_cards]}
            return {
                **card,
                "paths": paths,
                "sources": sources,
                "meta": dict(meta) if meta else None,
                "tags": tags,
                "entities": entities,
                "docs": docs,
                "keyframes": keyframes,
                "pages": [],
                "post": post,
                "similar": [sim_cards[m] for m in similar_ids if m in sim_cards],
            }

    def _first_readable(conn: sqlite3.Connection, media_id: int) -> Path | None:
        for (rel_path,) in conn.execute(
                """SELECT p.rel_path FROM media_path p JOIN media m ON m.id=p.media_id
                   WHERE p.media_id=? AND p.readable=1 AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED')
                   ORDER BY p.id""", (media_id,)):
            path = confined_path(cfg, rel_path)
            if path is not None:
                return path
        return None

    @app.get("/media/{media_id}/file")
    def media_file(media_id: int):
        with closing(open_ro(cfg)) as conn:
            path = _first_readable(conn, media_id)
            mime = conn.execute("SELECT mime FROM media WHERE id=?", (media_id,)).fetchone()
        if path is None:
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type=(mime[0] if mime and mime[0] else None),
                            headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/media/{media_id}/thumb")
    def media_thumb(media_id: int):
        with closing(open_ro(cfg)) as conn:
            row = conn.execute("SELECT thumb_path, status FROM media WHERE id=?", (media_id,)).fetchone()
        if not row or not row[0] or row[1] in ("EXCLUDED", "UNSUPPORTED"):
            raise HTTPException(404, "not found")
        thumbs_root = os.path.realpath(cfg.thumbs_dir)
        real = os.path.realpath(REPO_ROOT / row[0])
        if not real.startswith(thumbs_root + os.sep) or not os.path.isfile(real):
            raise HTTPException(404, "not found")
        return FileResponse(real, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/media/{media_id}/keyframe/{seq}")
    def media_keyframe(media_id: int, seq: int):
        with closing(open_ro(cfg)) as conn:
            row = conn.execute("SELECT path FROM keyframe WHERE media_id=? AND seq=?", (media_id, seq)).fetchone()
        keyframes_root = os.path.realpath(Path(cfg.data_dir) / "keyframes")
        real = os.path.realpath(REPO_ROOT / row[0]) if row else ""
        if not row or not real.startswith(keyframes_root + os.sep) or not os.path.isfile(real):
            raise HTTPException(404, "not found")
        return FileResponse(real, media_type="image/jpeg")

    @app.get("/api/captions/progress")
    def captions_progress():
        with closing(open_ro(cfg)) as conn:
            return search.caption_progress(conn)

    @app.get("/api/status")
    def status():
        from re2 import cli

        with closing(open_ro(cfg)) as conn:
            return cli.status_counts(conn)

    # ── Dashboard ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def home():
        if DASHBOARD.is_file():
            return FileResponse(DASHBOARD, media_type="text/html", headers={"Cache-Control": "no-cache"})
        return HTMLResponse('<!doctype html><title>Redpill</title><p>The dashboard is not built yet. '
                            '<a href="/legacy/">Legacy views</a></p>')

    # ── Legacy pages (frozen; retired in Phase 9) ─────────────────────────

    legacy_data = Path(cfg.data_dir)

    def _legacy_json(name: str):
        path = legacy_data / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="application/json")

    @app.get("/data")
    def legacy_data_json():
        return _legacy_json("images_captioned_tagged.json")

    @app.get("/tags")
    def legacy_tags_json():
        return _legacy_json("tags_with_sizes.json")

    @app.get("/tag_pairs_with_weights")
    def legacy_tag_pairs_json():
        return _legacy_json("tag_pairs_with_weights.json")

    for page in ("matrix", "spicy", "concept-map"):
        def _page(page=page):
            return FileResponse(PUBLIC / f"{page}.html", media_type="text/html")
        app.add_api_route(f"/legacy/{page}", _page, methods=["GET"], name=f"legacy_{page}")

    def _media_id_for_sha(conn: sqlite3.Connection, sha256: str) -> int | None:
        row = conn.execute("SELECT id FROM media WHERE sha256=? AND status NOT IN ('EXCLUDED', 'UNSUPPORTED')",
                           (sha256.lower(),)).fetchone()
        return row[0] if row else None

    @app.get("/hash-to-path/{sha256}")
    def hash_to_path(sha256: str):
        with closing(open_ro(cfg)) as conn:
            media_id = _media_id_for_sha(conn, sha256)
        if media_id is None:
            return JSONResponse({"error": "Hash not found"}, status_code=404)
        return {"url": f"/media/{media_id}/file"}

    @app.get("/node-info/{sha256}", response_class=HTMLResponse)
    def node_info(sha256: str):
        with closing(open_ro(cfg)) as conn:
            media_id = _media_id_for_sha(conn, sha256)
            if media_id is None:
                raise HTTPException(404, "Node not found")
            detail = item(media_id)
        caption = (detail["docs"].get("caption") or detail["docs"].get("legacy_caption") or [{}])[0]
        html = jinja.get_template("node_info.html").render(
            item=detail, description=caption.get("text", ""),
            sensitivity=detail["sensitivity"])
        return HTMLResponse(html)

    if (REPO_ROOT / "node_modules" / "three").is_dir():
        app.mount("/node_modules/three", StaticFiles(directory=REPO_ROOT / "node_modules" / "three"), name="three")
    app.mount("/legacy", StaticFiles(directory=PUBLIC, html=True), name="legacy")
    return app
