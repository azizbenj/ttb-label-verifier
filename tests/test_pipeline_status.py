"""The overall status from the strength of the evidence: a FAIL means something was read clearly and is wrong.

Fake readers with controlled word confidences, so every rule is pinned without Tesseract: a MISMATCH or
NOT FOUND keeps its verdict, but only a clearly read one fails the label; an unclear one asks for a look,
and when most required fields are unclear the result asks for a better image.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.batch import needs_phrase
from app.config import MANDATED_WARNING, THRESHOLDS
from app.decisions import prompt_map
from app.matching import NOTE_APPLICATION_UNREADABLE
from app.models import Application, FieldResult, Status, Verdict
from app.pipeline import POOR_IMAGE_NOTE, judge_clarity, verify
from app.readers.base import LabelReading, OCRResult, OCRWord

SURE = 95.0
UNSURE = THRESHOLDS.clear_min_conf - 30   # well under every clarity floor
WARNING_LINES = [
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic",
    "beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
    "beverages impairs your ability to drive a car or operate machinery, and may cause health problems.",
]
# The bottler does not repeat the brand, so a wrong brand line is a plain MISMATCH (not a small-print NEAR MATCH).
OLD_TOM = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)", "750 mL",
           "Bottled by Kettle Works, Bardstown, Kentucky 40004"]
APP = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                  alcohol_content="45% Alc./Vol. (90 Proof)", net_contents="750 mL",
                  bottler_name_address="Kettle Works, Bardstown, Kentucky 40004")
IMAGE = Image.new("RGB", (1600, 1200), "white")


class Reader:
    """Reads the given lines with word boxes; ``conf`` maps a line index to its words' confidence
    (default SURE). ``ink`` gives the result a blank page, so the warning statement gets a box and the
    confidence of its words can be measured."""

    name = "fake"

    def __init__(self, lines, conf=None, ink=True):
        self.lines, self.conf, self.ink = lines, conf or {}, ink

    def read(self, image):
        words = []
        for i, line in enumerate(self.lines):
            x = 20
            for t in line.split():
                words.append(OCRWord(text=t, left=x, top=40 + 40 * i, width=14 * len(t), height=24,
                                     conf=self.conf.get(i, SURE), line_index=i))
                x += 14 * len(t) + 12
        ink = np.zeros((1200, 1600), dtype=bool) if self.ink else None
        return LabelReading(ocr=OCRResult(text="\n".join(self.lines), lines=list(self.lines), words=words,
                                          engine="fake", ink=ink, mean_conf=SURE))


def run(lines, conf=None, app=APP, ink=True):
    return verify(app, IMAGE, Reader(lines, conf, ink), budget_s=0)


def field(r, key):
    return next(f for f in r.fields if f.key == key)


def smudges(n):
    """Lines of marks OCR could not read: a label that was not read well."""
    return [f"xq{i} zv{i} kw{i} pf{i}" for i in range(n)]


def test_a_clearly_read_wrong_brand_fails():
    r = run(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES)
    f = field(r, "brand_name")
    assert f.verdict == Verdict.MISMATCH and f.clear is True and f.read_conf == SURE
    assert r.overall == Status.FAIL and "1 problem: brand name" in r.summary


def test_a_wrong_brand_read_with_low_confidence_asks_for_a_look_and_keeps_its_verdict():
    r = run(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES, conf={0: UNSURE})
    f = field(r, "brand_name")
    assert f.verdict == Verdict.MISMATCH and f.found        # the field still says what was read
    assert f.clear is False and "low confidence" in f.clear_note
    assert r.overall == Status.REVIEW
    assert "Could not be read clearly: brand name" in r.summary and "problem" not in r.summary


def test_one_unsure_word_is_enough_to_doubt_a_text_mismatch():
    lines = ["OLD TOM DISTILLERY", "Kentucky Straight Rye Whiskey"] + OLD_TOM[2:] + WARNING_LINES
    sure = run(lines)
    assert field(sure, "class_type").verdict == Verdict.MISMATCH and sure.overall == Status.FAIL
    reader = Reader(lines)
    reading = reader.read(IMAGE)
    for w in reading.ocr.words:
        if w.text == "Rye":
            w.conf = THRESHOLDS.clear_min_conf - 1
    reader.read = lambda image: reading
    doubt = verify(APP, IMAGE, reader, budget_s=0)
    assert field(doubt, "class_type").clear is False and doubt.overall == Status.REVIEW


def _mismatch(key, found, **signals):
    return FieldResult(key=key, label=key, expected="Kentucky Straight Bourbon Whiskey", found=found,
                       verdict=Verdict.MISMATCH, signals={"mean_conf": SURE, "min_conf": SURE, "upright": True,
                                                          "plausible": 1.0, **signals})


WELL_READ = {"words": 40, "clear_share": 1.0, "low_share": 0.0, "warning_words": 9}


def test_implausible_or_sideways_text_is_not_clear_evidence():
    assert judge_clarity(_mismatch("class_type", "Kentucky Rye"), WELL_READ)[0] is True
    clear, note = judge_clarity(_mismatch("class_type", "Pee 7, \\ WHISKEY", plausible=0.6), WELL_READ)
    assert clear is False and "not plausible" in note
    assert judge_clarity(_mismatch("class_type", "ly"), WELL_READ)[0] is False          # two letters are no reading
    clear, note = judge_clarity(_mismatch("class_type", "Kentucky Rye", upright=False), WELL_READ)
    assert clear is False and "sideways" in note


def test_a_figure_the_application_cannot_state_is_the_applications_problem():
    f = FieldResult(key="alcohol_content", label="Alcohol content", expected="strong", verdict=Verdict.NOT_FOUND,
                    note=NOTE_APPLICATION_UNREADABLE + " as an alcohol content.", signals={})
    assert judge_clarity(f, {**WELL_READ, "clear_share": 0.2, "low_share": 0.5})[0] is True


def test_a_wrong_figure_fails_however_unsure_the_read():
    # Measured: confidence does not tell a misread figure from a wrong one, so a figure MISMATCH always fails.
    lines = OLD_TOM[:2] + ["40% Alc./Vol. (80 Proof)"] + OLD_TOM[3:] + WARNING_LINES
    r = run(lines, conf={2: 5})
    f = field(r, "alcohol_content")
    assert f.verdict == Verdict.MISMATCH and f.clear is True and f.read_conf == 5
    assert r.overall == Status.FAIL


def test_not_found_on_a_well_read_label_fails():
    r = run([l for l in OLD_TOM if l != "750 mL"] + WARNING_LINES)
    f = field(r, "net_contents")
    assert f.verdict == Verdict.NOT_FOUND and f.clear is True and r.overall == Status.FAIL


def test_not_found_on_a_poorly_read_label_asks_for_a_look():
    extra = smudges(4)
    lines = [l for l in OLD_TOM if l != "750 mL"] + WARNING_LINES + extra
    r = run(lines, conf={len(lines) - 1 - i: 10 for i in range(len(extra))})
    f = field(r, "net_contents")
    assert f.verdict == Verdict.NOT_FOUND and f.clear is False and "too small" in f.clear_note
    assert r.overall == Status.REVIEW and not r.poor_image   # one field: no request for a better image


def test_missing_warning_on_a_well_read_label_fails():
    r = run(OLD_TOM)
    assert not r.warning.present and r.warning.clear is True and r.overall == Status.FAIL


def test_missing_warning_on_a_poorly_read_label_asks_for_a_look():
    extra = smudges(4)
    lines = OLD_TOM + extra
    r = run(lines, conf={len(OLD_TOM) + i: 10 for i in range(len(extra))})
    w = r.warning
    assert not w.present and w.overall == Status.FAIL     # the warning's own result is unchanged
    assert w.clear is False and "sideways or too small" in w.clear_note
    assert r.overall == Status.REVIEW and "Could not be read clearly: government warning" in r.summary


def test_missing_warning_whose_words_were_read_elsewhere_asks_for_a_look():
    # Words of the statement were read but no statement could be assembled: it is probably there, unread.
    r = run(OLD_TOM + ["the Surgeon General xx pregnancy", "operate machinery qq"])
    assert not r.warning.present and r.warning.clear is False and r.overall == Status.REVIEW


def test_a_wording_failure_on_a_clearly_read_statement_fails():
    lines = OLD_TOM + WARNING_LINES[:1] + ["beverages during pregnancy because of the risk of birth defects."]
    r = run(lines)
    w = r.warning
    assert w.present and w.wording == Status.FAIL and w.clear is True and r.overall == Status.FAIL


def test_a_wording_failure_on_a_statement_read_with_low_confidence_asks_for_a_look():
    lines = OLD_TOM + WARNING_LINES[:1] + ["beverages during pregnancy because of the risk of birth defects."]
    r = run(lines, conf={len(OLD_TOM): UNSURE, len(OLD_TOM) + 1: UNSURE})
    w = r.warning
    assert w.present and w.wording == Status.FAIL and w.clear is False and "low confidence" in w.clear_note
    assert r.overall == Status.REVIEW


def test_one_clear_problem_fails_the_label_whatever_else_is_unclear():
    lines = ["RIVER BEND DISTILLERY"] + OLD_TOM[1:2] + ["40% Alc./Vol."] + OLD_TOM[3:] + WARNING_LINES
    r = run(lines, conf={0: UNSURE})
    assert field(r, "brand_name").clear is False and field(r, "alcohol_content").clear is True
    assert r.overall == Status.FAIL
    assert "1 problem: alcohol content" in r.summary and "Could not be read clearly: brand name" in r.summary
    assert needs_phrase(r)[0] == "Check the alcohol content"   # the clear problem first


def test_most_required_fields_unclear_asks_for_a_better_image():
    extra = smudges(5)
    lines = ["RIVER BEND DISTILLERY", "Tennessee Whiskey"] + OLD_TOM[2:3] + OLD_TOM[4:] + WARNING_LINES + extra
    conf = {0: UNSURE, 1: UNSURE, **{len(lines) - 1 - i: 10 for i in range(len(extra))}}
    r = run(lines, conf=conf)
    unclear = [f.key for f in r.fields if f.clear is False]
    assert len(unclear) >= THRESHOLDS.poor_image_unclear
    assert r.overall == Status.REVIEW and r.poor_image
    assert r.summary.startswith(POOR_IMAGE_NOTE.rstrip("."))
    assert needs_phrase(r)[0] == "Ask for a clearer image"
    assert not r.unreadable    # the grey card stays for images where nothing was read


def test_an_unclear_review_points_the_batch_row_at_the_field():
    r = run(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES, conf={0: UNSURE})
    assert needs_phrase(r)[0] == "Check the brand name"


def test_a_reader_without_word_confidences_fails_as_before():
    class NoWords:
        name = "plain"

        def read(self, image):
            lines = ["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES
            return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, engine="plain"))

    r = verify(APP, IMAGE, NoWords(), budget_s=0)
    f = field(r, "brand_name")
    assert f.verdict == Verdict.MISMATCH and f.clear is None and r.overall == Status.FAIL


def test_the_decision_prompt_for_an_unclear_mismatch_keeps_yes_for_a_fine_label():
    r = run(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES, conf={0: UNSURE})
    p = prompt_map(r)["brand_name"]
    assert p.yes.startswith("Yes") and "reading error" in p.yes and p.lean is False


def test_api_carries_clarity_but_not_the_raw_measurements():
    r = run(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES, conf={0: UNSURE})
    d = r.model_dump(mode="json")
    f = next(x for x in d["fields"] if x["key"] == "brand_name")
    assert f["clear"] is False and f["clear_note"] and f["read_conf"] == UNSURE and "signals" not in f
    assert "signals" not in d and d["poor_image"] is False and "clear" in d["warning"]


# --- the result page, without JavaScript ----------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    return TestClient(main.app), monkeypatch


FORM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)", "net_contents": "750 mL",
        "bottler_name_address": APP.bottler_name_address, "sample": "old_tom_clean"}


def test_banner_says_could_not_be_read_clearly_and_links_each_field(client):
    c, mp = client
    mp.setitem(main._readers, "tesseract", Reader(["RIVER BEND DISTILLERY"] + OLD_TOM[1:] + WARNING_LINES, {0: UNSURE}))
    t = c.post("/verify", data=FORM).text
    assert "Needs a quick look" in t
    assert 'Could not be read clearly: <a href="#row-brand_name">brand name</a>' in t
    assert "problem" not in t.split('class="banner')[1].split("</p>")[0]   # no "1 problem" for an unclear one
    assert "so this may be a reading error" in t   # the reason under the field's own note
    assert "ask for a sharper image" not in t


def test_banner_asks_for_a_sharper_image_when_most_fields_are_unclear(client):
    c, mp = client
    extra = smudges(5)
    lines = ["RIVER BEND DISTILLERY", "Tennessee Whiskey"] + OLD_TOM[2:3] + OLD_TOM[4:] + WARNING_LINES + extra
    conf = {0: UNSURE, 1: UNSURE, **{len(lines) - 1 - i: 10 for i in range(len(extra))}}
    mp.setitem(main._readers, "tesseract", Reader(lines, conf))
    t = c.post("/verify", data=FORM).text
    assert "Much of this label could not be read clearly. If possible, ask for a sharper image or the artwork file." in t


def test_banner_counts_only_clear_problems(client):
    c, mp = client
    lines = ["RIVER BEND DISTILLERY"] + OLD_TOM[1:2] + ["40% Alc./Vol."] + OLD_TOM[3:] + WARNING_LINES
    mp.setitem(main._readers, "tesseract", Reader(lines, {0: UNSURE}))
    t = c.post("/verify", data=FORM).text
    assert '1 problem: <a class="problem" href="#row-alcohol_content">alcohol content</a>.' in t
    assert 'Could not be read clearly: <a href="#row-brand_name">brand name</a>.' in t


def test_mandated_text_is_the_fixture():
    assert " ".join(WARNING_LINES) == MANDATED_WARNING
