import json
from types import SimpleNamespace

import httpx
import pymupdf

from re2 import db, pdf, scan


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Affidavit of a poll watcher describing ballot handling in Detroit. " * 3)
    doc.new_page()  # page 2: no text layer, like a scan
    doc.save(path)


def test_text_pages_and_ocr_queue(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    make_pdf(root / "affidavit.pdf")
    conn = db.connect(":memory:")
    db.init_schema(conn)
    cfg = SimpleNamespace(corpus_root=root, thumbs_dir=tmp_path / "thumbs", data_dir=tmp_path / "data",
                          base_url="http://vlm.test/v1", vision_model="gemma-test", vision_max_side=1024,
                          vision_max_tiles=6, vision_max_tokens=100, vision_max_tokens_tall=300)
    scan.scan(conn, cfg)
    stats = pdf.run_text(conn, cfg)
    assert (stats.done, stats.text_pages, stats.queued) == (1, 1, 1)
    media_id = conn.execute("SELECT id FROM media WHERE kind='pdf'").fetchone()[0]
    assert conn.execute("SELECT page FROM doc WHERE kind='pdf_page'").fetchall()[0][0] == 1
    assert conn.execute("SELECT thumb_path FROM media WHERE id=?", (media_id,)).fetchone()[0]
    assert pdf.run_text(conn, cfg).selected == 0  # not re-read

    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        result = {"description": "A blank scanned page.", "ocr_text": "Page two text",
                  "entities": {"people": ["Jane Roe"], "organizations": [], "places": ["Detroit"],
                               "events": [], "platforms": []},
                  "tags": ["affidavit"], "language": "en"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    ocr = pdf.run_ocr(conn, cfg, client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert (ocr.done, ocr.failed) == (1, 0)
    assert "page 2 of 2" in sent[0]["messages"][1]["content"][0]["text"]
    assert sent[0]["response_format"]["json_schema"]["schema"]["required"][0] == "description"
    row = conn.execute("SELECT text, page FROM doc WHERE kind='pdf_page' AND seq=2").fetchone()
    assert tuple(row) == ("Page two text", 2)
    assert conn.execute("SELECT count(*) FROM pdf_ocr_queue").fetchone()[0] == 0
    assert {r[0] for r in conn.execute("SELECT name FROM entity")} == {"Jane Roe", "Detroit"}
