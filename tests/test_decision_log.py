"""The decision log (``DECISION_LOG``): one JSON line per review decision and undo, written without ever failing
a request, and the report script that turns the log into pass rates per rule and a calibration CSV."""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import batch, config, main
from app.batch import BatchItem, BatchJob
from app.config import FIELDS
from app.models import Application
from app.pipeline import verify
from app.readers.base import LabelReading, OCRResult

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(main.app)
PARTIAL = {"X-Partial": "1"}
APP = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
       "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}
WARNING = ["GOVERNMENT WARNING: (1) According to the Surgeon General, women must not drink alcoholic",
           "beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
           "beverages impairs your ability to drive a car or operate machinery, and may cause health problems."]
LOG_KEYS = {"ts", "job", "source", "index", "application_id", "image", "decision", "key", "label_decision", "overall",
            "asked", "prompts",
            "fields", "warning", "read_confidence", "words_read", "reader", "timings"}


class FakeReader:
    """A label whose warning says 'must' (one difference to confirm) and whose brand is as given: in title case
    it is a second thing to confirm, and the queue asks about the brand first."""

    name = "fake"

    def __init__(self, brand: str = "Old Tom Distillery"):
        self.brand = brand

    def read(self, image):
        lines = [self.brand, "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)", "750 mL", *WARNING]
        return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, engine="fake"))


def make_job(n: int = 2, brand: str = "Old Tom Distillery", source: str = "apps.csv", job_id: str = "j1") -> BatchJob:
    """A finished batch of ``n`` labels that need a look, built without HTTP or Tesseract."""
    items = []
    for i in range(1, n + 1):
        app = Application(**APP, application_id=f"A{i}")
        result = verify(app, Image.new("RGB", (600, 800), "white"), FakeReader(brand), image_name=f"label_{i}.png")
        items.append(BatchItem(row=i + 1, application_id=f"A{i}", image_name=f"label_{i}.png", application=app,
                               result=result))
    return BatchJob(id=job_id, source=source, items=items, done=n, finished_ms=1.0)


@pytest.fixture
def log_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISION_LOG", str(path))
    return path


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- the log -------------------------------------------------------------------------------------------
def test_a_decision_writes_one_compact_line_with_the_documented_keys(log_path):
    job = make_job()
    job.decide(0, "pass")
    raw = log_path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1 and raw.endswith("\n")
    rec = json.loads(raw)
    assert set(rec) == LOG_KEYS
    assert json.dumps(rec, ensure_ascii=False, separators=(",", ":")) == raw.rstrip("\n")   # no indentation
    assert datetime.fromisoformat(rec["ts"]).tzinfo is not None
    assert (rec["job"], rec["source"], rec["index"], rec["application_id"], rec["image"]) == \
        ("j1", "apps.csv", 0, "A1", "label_1.png")
    assert rec["decision"] == "pass" and rec["overall"] == "REVIEW"
    # every question the label raised, the one the queue asked first marked
    assert [p["key"] for p in rec["prompts"]] == ["brand_name", "warning_wording", "warning_bold"]
    assert rec["asked"] == "brand_name" and [p["asked"] for p in rec["prompts"]] == [True, False, False]
    p = rec["prompts"][0]
    assert p["what"] == "Brand name" and p["verdict"] == "NEAR MATCH" and p["question"] == "Is this the same brand name?"
    assert p["left"] == ["", "OLD TOM DISTILLERY", ""] and p["right"] == ["", "Old Tom Distillery", ""]
    w = rec["prompts"][1]
    assert w["question"] == 'Does the label say "should"?' and w["left"][1] == "should" and w["right"][1] == "must"
    assert w["verdict"] == "1 DIFFERENCE" and rec["prompts"][2]["verdict"] == "COULD NOT TELL"
    # each field: key, verdict, expected, found, note
    assert [f["key"] for f in rec["fields"]] == [f.key for f in FIELDS]
    brand = rec["fields"][0]
    assert brand["verdict"] == "NEAR MATCH" and brand["expected"] == "OLD TOM DISTILLERY"
    assert brand["found"] == "Old Tom Distillery" and "capitalization" in brand["note"]
    assert rec["fields"][4] == {"key": "bottler_name_address", "verdict": "SKIPPED", "expected": "", "found": None,
                               "note": "Not provided on the application, so not checked."}
    # the four warning statuses and the wording score
    assert set(rec["warning"]) == {"present", "wording", "heading_caps", "heading_bold", "wording_score"}
    assert rec["warning"]["present"] is True and rec["warning"]["wording"] == "REVIEW"
    assert rec["warning"]["heading_bold"] == "REVIEW" and rec["warning"]["wording_score"] >= 97
    assert rec["reader"] == "fake" and set(rec["timings"]) == {"read_ms", "match_ms", "total_ms"}
    assert rec["read_confidence"] is None and rec["words_read"] == 0
    # never the image, its preview or the text read from it
    assert not any(k in raw for k in ("ocr_text", "label_text", "preview", "data:image", "operate machinery"))


def test_an_undo_is_logged_too_and_nothing_is_written_when_unset(log_path, monkeypatch):
    job = make_job()
    monkeypatch.setitem(batch.JOBS, job.id, job)
    r = client.post(f"/batch/{job.id}/decision", data={"index": 1, "value": "fail"}, headers=PARTIAL)
    assert r.status_code == 200 and r.json()["decision"] == "fail"
    r = client.post(f"/batch/{job.id}/decision", data={"index": 1, "value": "clear"}, headers=PARTIAL)
    assert r.status_code == 200 and r.json()["decision"] == ""
    recs = records(log_path)
    assert [(x["application_id"], x["decision"]) for x in recs] == [("A2", "fail"), ("A2", "clear")]
    assert recs[1]["overall"] == "REVIEW" and recs[1]["asked"] == "brand_name"    # an undo carries the same context
    monkeypatch.setattr(config, "DECISION_LOG", "")
    job.decide(0, "skip")
    assert len(records(log_path)) == 2 and job.items[0].decision == "skip"


def test_a_write_failure_never_fails_the_request(tmp_path, monkeypatch, caplog):
    path = tmp_path / "no-such-dir" / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISION_LOG", str(path))
    job = make_job()
    monkeypatch.setitem(batch.JOBS, job.id, job)
    with caplog.at_level(logging.WARNING, logger="labelcheck"):
        r = client.post(f"/batch/{job.id}/decision", data={"index": 0, "value": "pass"}, headers=PARTIAL)
    assert r.status_code == 200 and r.json()["decision"] == "pass" and job.items[0].decision == "pass"
    assert not path.exists()
    assert any("decision log" in m and "no-such-dir" in m for m in caplog.messages)
    # the plain form post (no JavaScript) is still sent on to the next label
    r = client.post(f"/batch/{job.id}/decision", data={"index": 1, "value": "fail"}, follow_redirects=False)
    assert r.status_code == 303 and job.items[1].decision == "fail"


def test_a_row_that_could_not_be_checked_is_logged_without_a_result(log_path):
    job = make_job(1)
    job.items.append(BatchItem(row=9, application_id="A9", image_name="blank.png", error="Unreadable image."))
    job.decide(1, "skip")
    rec = records(log_path)[0]
    assert rec["overall"] == "ERROR" and rec["prompts"] == [] and rec["fields"] == [] and rec["asked"] is None
    assert rec["warning"] is None and rec["timings"] is None and rec["reader"] == ""


def test_concurrent_decisions_each_write_a_whole_line(log_path):
    job = make_job(8)

    def work(i):
        for k in range(25):
            job.decide(i, ("pass", "fail", "skip", "clear")[k % 4])

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, range(8)))
    recs = records(log_path)   # every line parses on its own: no two writes were interleaved
    assert len(recs) == 200
    assert Counter(r["decision"] for r in recs) == {"pass": 56, "fail": 48, "skip": 48, "clear": 48}


# --- scripts/decisions_report.py -----------------------------------------------------------------------
def load_report():
    spec = importlib.util.spec_from_file_location("decisions_report", ROOT / "scripts" / "decisions_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_log(log_path: Path) -> None:
    """Two batches: in the first the brand is in title case (the queue asks about the brand), in the second it
    is exact (the queue asks about the warning wording). Answers, a re-decision and an undo."""
    a = make_job(4, job_id="jobA")
    for i, v in ((0, "pass"), (1, "pass"), (2, "fail"), (3, "skip")):
        a.decide(i, v)
    a.decide(1, "clear")           # undone: drops out
    a.decide(2, "pass")            # re-decided: the last answer counts
    b = make_job(3, brand="OLD TOM DISTILLERY", source="lot2.csv", job_id="jobB")
    for i, v in ((0, "fail"), (1, "fail"), (2, "pass")):
        b.decide(i, v)


def test_report_counts_the_standing_decisions_per_question(log_path):
    report = load_report()
    make_log(log_path)
    recs = report.read_log(log_path)
    assert len(recs) == 9
    decided = report.latest_decisions(recs)
    assert [(r["job"], r["application_id"], r["decision"]) for r in decided] == [
        ("jobA", "A1", "pass"), ("jobA", "A4", "skip"), ("jobA", "A3", "pass"),
        ("jobB", "A1", "fail"), ("jobB", "A2", "fail"), ("jobB", "A3", "pass")]
    rows = report.summarise(decided)
    assert [r["key"] for r in rows] == ["brand_name", "warning_wording", "warning_bold"]
    assert rows[0] == {"key": "brand_name", "raised": 3, "asked": 3, "pass": 2, "fail": 0, "skip": 1, "pass_rate": 1.0}
    assert rows[1]["pass_rate"] == pytest.approx(1 / 3)
    assert {k: v for k, v in rows[1].items() if k != "pass_rate"} == \
        {"key": "warning_wording", "raised": 6, "asked": 3, "pass": 1, "fail": 2, "skip": 0}
    assert rows[2] == {"key": "warning_bold", "raised": 6, "asked": 0, "pass": 0, "fail": 0, "skip": 0, "pass_rate": None}
    # the notes behind the answers: the field's note, or the question for a warning check
    assert report.top_notes(decided, "pass") == [
        ("brand_name", "Same words, but capitalization or punctuation differs. Please confirm.", 2),
        ("warning_wording", 'Does the label say "should"?', 1)]
    assert report.top_notes(decided, "fail") == [("warning_wording", 'Does the label say "should"?', 2)]
    assert report.top_notes(decided, "skip", n=1) == [
        ("brand_name", "Same words, but capitalization or punctuation differs. Please confirm.", 1)]


def test_calibration_csv_has_the_real_labels_layout_and_expect_from_the_decisions(log_path, tmp_path):
    report = load_report()
    make_log(log_path)
    decided = report.latest_decisions(report.read_log(log_path))
    out = tmp_path / "calibration.csv"
    report.write_csv(report.calibration_rows(decided), out)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    real_header = (ROOT / "scripts" / "real_labels.csv").read_text(encoding="utf-8").splitlines()[0]
    assert list(rows[0]) == real_header.split(",") == list(report.CSV_COLUMNS)
    assert report.FIELD_KEYS == tuple(f.key for f in FIELDS)
    assert [(r["ttbid"], r["expect"], r["notes"]) for r in rows] == [
        ("A1", "brand_name=review", "pass: Is this the same brand name?"),
        ("A3", "brand_name=review", "pass: Is this the same brand name?"),     # the skipped A4 is not calibration data
        ("A1", "", 'fail: Does the label say "should"?'),
        ("A2", "", 'fail: Does the label say "should"?'),
        ("A3", "", 'pass: Does the label say "should"?')]                      # a pass on a warning check names no field
    r = rows[0]
    assert (r["kind"], r["brand_name"], r["class_type"], r["alcohol_content"], r["net_contents"]) == (
        "", "OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL")
    assert r["bottler_name_address"] == "" and r["country_of_origin"] == ""
    # read the way scripts/real_labels.py reads it: the brand is expected to come back NEAR MATCH, the rest MATCH
    assert dict(e.split("=") for e in rows[0]["expect"].split(";") if e) == {"brand_name": "review"}
    assert dict(e.split("=") for e in rows[2]["expect"].split(";") if e) == {}


def test_report_script_prints_the_table_and_writes_the_csv(log_path, tmp_path):
    make_log(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("not json\n")                                # a damaged line is reported and skipped
    out = tmp_path / "cal.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "decisions_report.py"), str(log_path), "--csv", str(out)],
                       capture_output=True, text=True, check=False, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "not a JSON record" in r.stderr
    assert "9 records, 6 answers that stand" in r.stdout
    assert re.search(r"brand_name\s+3\s+3\s+2\s+0\s+1\s+100%", r.stdout)
    assert re.search(r"warning_wording\s+6\s+3\s+1\s+2\s+0\s+33%", r.stdout)
    assert re.search(r"warning_bold\s+6\s+0\s+0\s+0\s+0\s+-", r.stdout)
    assert 'Notes behind "pass" answers' in r.stdout and "Same words, but capitalization or punctuation differs" in r.stdout
    assert "wrote 5 rows" in r.stdout and out.exists()
