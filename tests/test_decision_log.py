"""The decision log (``DECISION_LOG``): one JSON line per review decision and undo, written without ever failing
a request, and the report script that turns the log into pass rates per rule and a calibration CSV."""

from __future__ import annotations

import json
import logging
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
LOG_KEYS = {"ts", "job", "source", "index", "application_id", "image", "decision", "overall", "asked", "prompts",
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
