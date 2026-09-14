from datetime import date, datetime

from re2 import dates, db

TODAY = date(2026, 9, 14)


def ns(y, m, d):
    return int(datetime(y, m, d, 12).timestamp() * 1e9)


def facts(**kw):
    base = dict(media_id=1, source_dates=[], filenames=[], mtimes_ns=[ns(2026, 9, 1)], texts=[])
    base.update(kw)
    return dates.ItemFacts(**base)


def fn(name):
    return [d.isoformat() for d in dates.filename_dates(name)]


def test_filename_patterns():
    assert fn("screenshot_20260131_132538_photo1.jpg") == ["2026-01-31"]
    assert fn("screenshot-2022_12_10-084545_PM.jpg") == ["2022-12-10"]
    assert fn("Screenshot from 2020-01-02 10-11-12.png") == ["2020-01-02"]
    assert fn("2020-06-11_15-32.png") == ["2020-06-11"]
    assert fn("20200826_123336.jpg") == ["2020-08-26"]
    assert fn("1517112304894.png") == ["2018-01-28"]
    assert fn("1614994129034 (1).png") == ["2021-03-06"]
    assert fn("1684006713174249.png") == ["2023-05-13"]
    assert fn("screenshot_race_1769449220.png") == ["2026-01-26"]
    assert fn("EgUn8vuU8AERGy8.jpeg") == ["2020-08-26"]
    assert fn("The Hill - Biden speech-1449405021087469572.mp4") == ["2021-10-16"]
    assert fn("Ian Miles Cheong - We went to 54 states.mp4") == []


def test_content_patterns():
    found = {d.isoformat(): has_day for d, has_day in dates.content_dates(
        "Posted Monday 8 August 2016; quoted NOV 06, 2017 and Fri, Jun 9, 2017 (6/9/17). "
        "Updated May 31st), 2016. Report from March 2019. ISO 2021-01-05. 16th June 2020, 12 Jul 17")}
    assert found == {"2016-08-08": True, "2017-11-06": True, "2017-06-09": True, "2016-05-31": True,
                     "2019-03-01": False, "2021-01-05": True, "2020-06-16": True, "2017-07-12": True}


def test_rule_order_and_plausibility():
    assert dates.item_date(facts(source_dates=["2024-11-18T10:00:00-07:00"], filenames=["2020-01-01_x.png"]),
                           TODAY) == ("2024-11-18", "source")
    assert dates.item_date(facts(filenames=["2020-01-01_x.png"], texts=["January 3, 2019"]), TODAY) == \
        ("2020-01-01", "filename")
    # The latest plausible content date wins; dates after the file existed or before 2005 are rejected.
    assert dates.item_date(facts(texts=["Jan 1, 2004 then May 1, 2014 and Dec 25, 2030"]), TODAY) == \
        ("2014-05-01", "content")
    assert dates.item_date(facts(texts=["sometime in March 2019"]), TODAY) == ("2019-03", "content")
    assert dates.item_date(facts(), TODAY) == ("2026-09-01", "mtime")
    assert dates.item_date(facts(mtimes_ns=[]), TODAY) == (None, "none")
    # A filename date later than the file's mtime (+2 days) is impossible.
    assert dates.item_date(facts(filenames=["20300826_123336.jpg"]), TODAY) == ("2026-09-01", "mtime")


def test_run_writes_media_meta():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript(f"""
        INSERT INTO media(id, uid, kind, status) VALUES (1, 'a', 'image', 'READY'), (2, 'b', 'other', 'EXCLUDED');
        INSERT INTO media_path(media_id, rel_path, readable, mtime_ns, first_seen, last_seen)
          VALUES (1, 'race/1517112304894.png', 1, {ns(2018, 2, 1)}, 'x', 'x'),
                 (2, 'cookies.txt', 1, {ns(2018, 2, 1)}, 'x', 'x');
    """)
    counts = dates.run(conn, TODAY)
    assert counts == {"filename": 1, "changed": 1}
    assert conn.execute("SELECT item_date, date_source FROM media_meta WHERE media_id=1").fetchone()[:] == \
        ("2018-01-28", "filename")
    assert dates.run(conn, TODAY)["changed"] == 0
