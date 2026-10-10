"""The per-label time budget: optional passes run only while they fit, measured with a fake clock."""

import numpy as np
import pytest
from PIL import Image

from app import pipeline
from app.models import Application, Verdict
from app.pipeline import PassCost, verify
from app.readers.base import LabelReading, OCRResult, OCRWord, View

APP = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                  alcohol_content="45% Alc./Vol.", net_contents="750 mL")


def _ocr(lines):
    words, y = [], 100
    for li, line in enumerate(lines):
        x = 100
        for tok in line.split():
            words.append(OCRWord(text=tok, left=x, top=y, width=60, height=40, conf=85, line_index=li))
            x += 70
        y += 100
    ink = np.zeros((1000, 1600), dtype=bool)
    return OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink, engine="fake",
                     views=[View(rot=0, inverted=False, ink=ink, size=(1600, 1000))])


class FakeClock:
    """Advances by what each step says it costs."""

    def __init__(self, read_s: float):
        self.now, self.read_s = 0.0, read_s

    def __call__(self):
        return self.now


class StagedReader:
    """First pass misses the brand; the turned passes cost ``extend_s`` and find nothing; the second
    engine costs ``escalate_s`` and reads the brand."""

    name = "staged"

    def __init__(self, clock: FakeClock, extend_s: float, escalate_s: float):
        self.clock, self.extend_s, self.escalate_s = clock, extend_s, escalate_s
        self.ran = []

    def read(self, image):
        self.clock.now += self.clock.read_s
        return LabelReading(ocr=_ocr(["Kentucky Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL"]))

    def extend(self, reading):
        self.ran.append("extend")
        self.clock.now += self.extend_s
        return False

    def escalate(self, reading):
        self.ran.append("escalate")
        self.clock.now += self.escalate_s
        ocr = reading.ocr
        ocr.lines.append("OLD TOM DISTILLERY")
        ocr.words.append(OCRWord(text="OLD", left=100, top=500, width=60, height=40, conf=95, line_index=3))
        ocr.text = "\n".join(ocr.lines)
        ocr.engine += " + rapidocr pass"
        return True


@pytest.fixture(autouse=True)
def fresh_costs(monkeypatch):
    monkeypatch.setattr(pipeline, "PASS_COSTS", {"extend": PassCost(1.2), "escalate": PassCost(1.5)})


def _brand(r):
    return next(f for f in r.fields if f.key == "brand_name")


def test_both_passes_run_when_they_fit():
    clock = FakeClock(read_s=1.0)
    reader = StagedReader(clock, extend_s=1.0, escalate_s=1.0)
    r = verify(APP, Image.new("RGB", (10, 10)), reader, budget_s=5.0, clock=clock)
    assert reader.ran == ["extend", "escalate"]
    assert _brand(r).verdict == Verdict.MATCH and "skipped" not in r.reader
    # The process now knows what the passes cost here.
    assert pipeline.PASS_COSTS["extend"].expected == pytest.approx(1.1)
    assert pipeline.PASS_COSTS["escalate"].expected == pytest.approx(1.25)


def test_escalation_is_skipped_when_it_would_not_fit():
    clock = FakeClock(read_s=2.5)
    reader = StagedReader(clock, extend_s=1.2, escalate_s=1.0)   # 3.7 s spent + 1.5 s expected > 5 s
    r = verify(APP, Image.new("RGB", (10, 10)), reader, budget_s=5.0, clock=clock)
    assert reader.ran == ["extend"]
    assert _brand(r).verdict != Verdict.MATCH          # the first-pass verdict stands (the largest line is shown)
    assert r.reader.endswith("(RapidOCR escalation skipped for time)")


def test_turned_passes_are_skipped_too_when_the_first_read_ate_the_budget():
    clock = FakeClock(read_s=4.0)
    reader = StagedReader(clock, extend_s=1.0, escalate_s=1.0)
    r = verify(APP, Image.new("RGB", (10, 10)), reader, budget_s=5.0, clock=clock)
    assert reader.ran == []
    assert "turned and contrast passes skipped for time" in r.reader and "RapidOCR escalation skipped for time" in r.reader


def test_no_budget_runs_everything():
    clock = FakeClock(read_s=4.0)
    reader = StagedReader(clock, extend_s=3.0, escalate_s=3.0)
    r = verify(APP, Image.new("RGB", (10, 10)), reader, budget_s=0, clock=clock)
    assert reader.ran == ["extend", "escalate"] and _brand(r).verdict == Verdict.MATCH


def test_the_measured_cost_replaces_the_seed():
    clock = FakeClock(read_s=0.5)
    slow = StagedReader(clock, extend_s=0.1, escalate_s=4.0)           # fits by the seed (0.6 + 1.5 < 5)...
    verify(APP, Image.new("RGB", (10, 10)), slow, budget_s=5.0, clock=clock)
    assert pipeline.PASS_COSTS["escalate"].expected == pytest.approx(2.75)
    verify(APP, Image.new("RGB", (10, 10)), slow, budget_s=5.0, clock=clock)
    assert pipeline.PASS_COSTS["escalate"].expected == pytest.approx((1.5 + 4 + 4) / 3)
    clock2 = FakeClock(read_s=2.0)
    later = StagedReader(clock2, extend_s=0.1, escalate_s=4.0)        # ...but not once 3.2 s is the measured cost
    verify(APP, Image.new("RGB", (10, 10)), later, budget_s=5.0, clock=clock2)
    assert later.ran == ["extend"]


def test_an_instant_return_is_not_a_cost_sample():
    cost = PassCost(1.5)
    cost.add(0.0)
    cost.add(0.01)
    assert cost.expected == 1.5
    cost.add(0.5)
    assert cost.expected == pytest.approx(1.0)
