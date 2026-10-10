"""Day 2 of the redesign: decisions kept with a batch, the review queue, scoped exports, printable reports."""

import csv
import io
import re
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.decisions import prompts_for
from app.models import Application
from app.pipeline import verify
from app.readers.base import LabelReading, OCRResult

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
client = TestClient(main.app)
PARTIAL = {"X-Partial": "1"}
APP = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
       "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}


class ReviewReader:
    """Reads a label whose brand is in title case and whose warning says 'must': two things to confirm."""

    name = "review"

    def read(self, image):
        lines = ["Old Tom Distillery", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)", "750 mL",
                 "GOVERNMENT WARNING: (1) According to the Surgeon General, women must not drink alcoholic",
                 "beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
                 "beverages impairs your ability to drive a car or operate machinery, and may cause health problems."]
        return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, engine="review"))


def _batch(monkeypatch, n=3):
    monkeypatch.setitem(main._readers, "tesseract", ReviewReader())
    rows = "\n".join(f"old_tom_clean.png,A{i},OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL"
                     for i in range(1, n + 1))
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n" + rows + "\n").encode()
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv")),
                                     ("files", ("old_tom_clean.png", (SAMPLES / "old_tom_clean.png").read_bytes(), "image/png"))],
                    headers=PARTIAL)
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        t = client.get(f"/batch/{job_id}", headers=PARTIAL).text
        if 'data-status="done"' in t:
            return job_id, t
        time.sleep(0.05)
    raise AssertionError("batch did not finish")


def test_prompts_ask_one_question_per_thing_to_confirm():
    r = verify(Application(**APP), Image.new("RGB", (600, 800), "white"), ReviewReader())
    ps = prompts_for(r)
    # no word boxes from this reader, so the heading's weight could not be measured: that asks too
    assert [p.key for p in ps] == ["brand_name", "warning_wording", "warning_bold"]
    assert ps[2].question == 'Is "GOVERNMENT WARNING:" printed in bold?'
    assert ps[0].question == "Is this the same brand name?" and ps[0].yes == "Yes, same brand"
    assert ps[1].question == 'Does the label say "should"?'
    assert ps[1].yes == 'Yes, it says "should": reading error' and ps[1].no == 'No, it says "must": misprint'
    assert ps[1].left[1] == "should" and ps[1].right[1] == "must"


def test_single_result_shows_the_prompts_and_prints_the_decision_block(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", ReviewReader())
    r = client.post("/verify", data={**APP, "sample": "old_tom_clean"}, headers=PARTIAL)
    assert r.status_code == 200
    assert 'data-decide="brand_name"' in r.text and "Is this the same brand name?" in r.text
    assert 'data-answer="pass"' in r.text and "Skip for now" in r.text
    assert "Reviewer decision" in r.text and "Your answer goes into the printed report" in r.text


def test_decisions_are_kept_with_the_batch_and_never_change_a_verdict(monkeypatch):
    job_id, t = _batch(monkeypatch)
    assert "Review the 3" in t and 'data-act="export"' in t
    # without a question key the answer covers every question the label raises
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass"}, headers=PARTIAL)
    assert r.status_code == 200
    j = r.json()
    assert {k: j[k] for k in ("index", "decision", "open", "decided", "queue", "remaining")} == {
        "index": 0, "decision": "pass", "open": [], "decided": 1, "queue": 3, "remaining": 2}
    assert j["answers"] == {"brand_name": "pass", "warning_wording": "pass", "warning_bold": "pass"}
    r = client.post(f"/batch/{job_id}/decision", data={"index": 1, "value": "maybe"}, headers=PARTIAL)
    assert r.status_code == 400
    t = client.get(f"/batch/{job_id}", headers=PARTIAL).text
    assert 'data-id="A1"' in t and re.search(r'data-id="A1"[^>]*data-status="REVIEW"', t)   # still REVIEW
    assert re.search(r'data-id="A1"[^>]*data-decision="pass"', t)
    j = client.get(f"/api/batch/{job_id}").json()
    assert j["items"][0]["decision"] == "pass" and j["items"][1]["decision"] == ""


def test_review_queue_walks_the_labels_that_need_a_look(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    r = client.get(f"/batch/{job_id}/review")
    assert r.status_code == 200
    assert "1 of 3 that need a look · 0 decided" in r.text
    assert "Is this the same brand name?" in r.text and 'class="qd cur"' in r.text
    assert "Mandatory label information" in r.text          # the rest of the result, under details
    assert 'href="/batch/' + job_id + '/review?n=2"' in r.text
    # a plain form post (no script) records the answer and moves on to the next label
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "fail"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/review?n=2")
    r = client.get(f"/batch/{job_id}/review?n=2")
    assert "2 of 3 that need a look · 1 decided" in r.text and 'class="qd fail"' in r.text
    r = client.get(f"/batch/{job_id}/review?n=99")
    assert "3 of 3 that need a look" in r.text


def test_review_queue_with_nothing_to_review(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", ReviewReader())
    r = client.post("/batch", data={"sample": "1"}, headers=PARTIAL)
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    deadline = time.monotonic() + 90
    while 'data-status="done"' not in client.get(f"/batch/{job_id}", headers=PARTIAL).text and time.monotonic() < deadline:
        time.sleep(0.1)
    page = client.get(f"/batch/{job_id}/review").text
    assert "Review queue" in page


def test_export_scopes_includes_and_file_name(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    client.post(f"/batch/{job_id}/decision", data={"index": 2, "value": "skip"}, headers=PARTIAL)
    r = client.get(f"/batch/{job_id}/export.csv")
    assert r.text.startswith("﻿")
    rows = list(csv.reader(io.StringIO(r.text.lstrip("﻿"))))
    assert rows[0][:5] == ["application_id", "image", "overall", "needs", "summary"]
    assert "decision" in rows[0] and "warning_diff" in rows[0] and "label_text" not in rows[0]
    assert rows[3][rows[0].index("decision")] == "skip"
    assert re.search(r'filename=label-check_apps_all_\d{4}-\d{2}-\d{2}\.csv', r.headers["content-disposition"])
    r = client.get(f"/batch/{job_id}/export.csv?ids=A2&include=decisions,ocr")
    rows = list(csv.reader(io.StringIO(r.text.lstrip("﻿"))))
    assert len(rows) == 2 and rows[0] == ["application_id", "image", "overall", "needs", "summary", "decision",
                                           "answers", "label_text", "reader", "error"]
    assert "GOVERNMENT WARNING" in rows[1][7]
    assert "_selected_" in r.headers["content-disposition"]
    r = client.get(f"/batch/{job_id}/export.csv?status=PASS")
    assert r.text.count("\n") == 1 and "_pass_" in r.headers["content-disposition"]


def test_printable_reports_one_per_label(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    r = client.get(f"/batch/{job_id}/report?ids=A1,A3")
    assert r.status_code == 200
    assert r.text.count('class="report-page"') == 2 and "2 printable reports" in r.text
    assert r.text.count("Reviewer decision") == 2


def test_shared_batch_link_opens_the_batch_tab(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    r = client.get(f"/batch/{job_id}")
    assert r.status_code == 200 and "<!doctype html>" in r.text
    assert 'data-open-tab="batch"' in r.text and 'id="batch-entry" hidden' in r.text
    assert "3 labels checked" in r.text
    assert client.get("/batch/nope").status_code == 404


def test_yes_always_means_the_label_is_fine_even_when_the_wording_fails():
    from app.config import MANDATED_WARNING
    from app.readers.base import LabelReading, OCRResult
    truncated = MANDATED_WARNING.split(" (2)")[0]

    class Truncated:
        name = "t"

        def read(self, image):
            return LabelReading(ocr=OCRResult(text=truncated, lines=[truncated]))
    r = verify(Application(**APP), Image.new("RGB", (60, 60)), Truncated())
    assert r.warning.wording.value == "FAIL"
    p = next(p for p in prompts_for(r) if p.key == "warning_wording")
    # The first button records "pass" (label fine) and carries the Y key: its text must say yes.
    assert p.question.startswith("Is ") and p.yes.startswith("Yes") and p.no.startswith("No")


def test_proof_contradiction_prompt_and_evidence():
    from app.evidence import evidence_for
    from app.matching import compare_alcohol
    r = verify(Application(**APP), Image.new("RGB", (60, 60), "white"), ReviewReader())
    near = compare_alcohol("45%", "45% Alc./Vol. (80 Proof)")
    r.fields = [near if f.key == "alcohol_content" else f for f in r.fields]
    p = next(p for p in prompts_for(r) if p.key == "alcohol_content")
    assert "proof agree with its percentage" in p.question and p.no == "No, the label contradicts itself"
    wrong = compare_alcohol("40%", "45% Alc./Vol. (80 Proof)")
    assert "proof and percent agree" not in evidence_for(wrong, r).text
    assert "proof and percent agree" in evidence_for(compare_alcohol("40%", "45% Alc./Vol. (90 Proof)"), r).text


# --- one answer settles one question, never the whole label -----------------------------------------

class FailReader(ReviewReader):
    """Reads the alcohol content as 40% against an application of 45%: a MISMATCH, so the label fails."""

    name = "fail"

    def read(self, image):
        lines = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "40% Alc./Vol.", "750 mL",
                 "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic",
                 "beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
                 "beverages impairs your ability to drive a car or operate machinery, and may cause health problems."]
        return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, engine="fail"))


def test_one_answer_leaves_the_other_questions_open(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "brand_name"},
                    headers=PARTIAL)
    j = r.json()
    assert j["decision"] == "partial" and j["open"] == ["warning_wording", "warning_bold"] and j["decided"] == 0
    item = main.batchmod.JOBS[job_id].items[0]
    assert item.decision == "partial" and item.answers == {"brand_name": "pass"}
    client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "warning_wording"}, headers=PARTIAL)
    j = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "fail", "key": "warning_bold"},
                    headers=PARTIAL).json()
    assert j["decision"] == "fail" and j["open"] == [] and j["decided"] == 1          # any "no" fails the label
    j = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "warning_bold"},
                    headers=PARTIAL).json()
    assert j["decision"] == "pass"                                                     # every "yes" clears it
    j = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "clear", "key": "brand_name"},
                    headers=PARTIAL).json()
    assert j["decision"] == "partial" and j["open"] == ["brand_name"]
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "country_of_origin"},
                    headers=PARTIAL)
    assert r.status_code == 400                                                        # not a question on this label
    row = next(x for x in csv.DictReader(io.StringIO(client.get(f"/batch/{job_id}/export.csv").text.lstrip("\ufeff")))
               if x["application_id"] == "A1")
    assert row["decision"] == "partial"
    assert row["answers"] == "brand_name: open; warning_wording: yes; warning_bold: yes"


def test_review_queue_asks_every_question_before_moving_on(monkeypatch):
    job_id, _ = _batch(monkeypatch)
    page = client.get(f"/batch/{job_id}/review").text
    assert "Question 1 of 3" in page and "Is this the same brand name?" in page
    assert "Everything else was checked and matches" not in page        # two other questions are open
    assert 'name="key" value="brand_name"' in page
    # a plain form post stays on the label while it has open questions
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "brand_name"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/review?n=1")
    page = client.get(f"/batch/{job_id}/review?n=1").text
    assert "Question 2 of 3" in page and 'Does the label say &#34;should&#34;?' in page and "0 decided" in page
    client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "warning_wording"})
    r = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "skip", "key": "warning_bold"},
                    follow_redirects=False)
    assert r.headers["location"].endswith("/review?n=2")                          # all answered: next label
    page = client.get(f"/batch/{job_id}/review?n=1&q=brand_name").text
    assert "Question 1 of 3" in page and "Decided by you" in page


def test_fails_can_be_checked_and_cleared_as_misreads(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", FailReader())
    rows = "old_tom_clean.png,F1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL"
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n" + rows + "\n").encode()
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv")),
                                     ("files", ("old_tom_clean.png", (SAMPLES / "old_tom_clean.png").read_bytes(), "image/png"))],
                    headers=PARTIAL)
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    deadline = time.monotonic() + 60
    while 'data-status="done"' not in (t := client.get(f"/batch/{job_id}", headers=PARTIAL).text):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert f'href="/batch/{job_id}/review?pile=fail"' in t and "Check the 1 fail" in t
    assert "send-back" not in t and "Ready to approve (0)" in t
    assert "Check the alcohol content" in t                    # never "Nothing: send back"
    page = client.get(f"/batch/{job_id}/review?pile=fail").text
    assert "Fails to check" in page and "1 of 1 that failed" in page
    assert 'Does the label say &#34;45% Alc./Vol.&#34;?' in page
    assert "data-neutral" in page                               # the application value is not the suggested answer
    j = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": "alcohol_content",
                                                       "pile": "fail"}, headers=PARTIAL).json()
    assert j["queue"] == 1 and j["decision"] == ("partial" if j["open"] else "pass")
    for key in j["open"]:                                       # the label's other questions (the heading's weight)
        j = client.post(f"/batch/{job_id}/decision", data={"index": 0, "value": "pass", "key": key, "pile": "fail"},
                        headers=PARTIAL).json()
    assert j["decision"] == "pass"
    out = client.get(f"/batch/{job_id}/export.csv?status=PASS,CLEARED").text.lstrip("\ufeff")
    rows = list(csv.DictReader(io.StringIO(out)))
    assert [(x["application_id"], x["overall"], x["decision"]) for x in rows] == [("F1", "FAIL", "pass")]
    assert "Ready to approve (1)" in client.get(f"/batch/{job_id}", headers=PARTIAL).text


def test_failures_ask_a_neutral_question():
    r = verify(Application(**APP), Image.new("RGB", (600, 800), "white"), FailReader())
    p = {q.key: q for q in prompts_for(r)}
    assert p["alcohol_content"].verdict == "MISMATCH" and p["alcohol_content"].lean is False
    assert p["alcohol_content"].question == 'Does the label say "45% Alc./Vol."?'
    assert p["alcohol_content"].yes == 'Yes, it says "45% Alc./Vol.": reading error'


def test_a_missing_warning_asks_whether_it_is_printed():
    class NoWarning(ReviewReader):
        def read(self, image):
            lines = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL"]
            return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, engine="nowarn"))
    r = verify(Application(**APP), Image.new("RGB", (600, 800), "white"), NoWarning())
    p = {q.key: q for q in prompts_for(r)}
    assert p["warning_present"].question == "Is the government warning printed on the label?"
    assert p["warning_present"].lean is False
