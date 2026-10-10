"""The RapidOCR reader: the real engine on one sample, the escalation with a fake engine (no model
needed), the reader registry and the health check."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.readers import rapid as R
from app.readers.base import LabelReading, OCRResult, OCRWord, ReaderError, View
from app.readers.extract import conflict_confidence

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
requires_rapid = pytest.mark.skipif(R.rapid_version() is None, reason="rapidocr-onnxruntime not installed")


# --- the real engine ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def old_tom_reading():
    return R.RapidOCRReader(escalate_with_tesseract=False).read(Image.open(SAMPLES / "old_tom_clean.png"))


@requires_rapid
def test_engine_reads_the_sample_with_boxes_inside_the_image(old_tom_reading):
    ocr = old_tom_reading.ocr
    text = ocr.text.upper()
    assert "OLD TOM DISTILLERY" in text and "GOVERNMENT WARNING" in text and "750" in text
    assert ocr.lines and len(ocr.words) >= len(ocr.lines)
    width, height = ocr.views[0].size
    assert ocr.ink is not None and ocr.ink.shape == (height, width)
    for w in ocr.words:
        assert 0 <= w.left < w.right <= width and 0 <= w.top < w.bottom <= height, w
        assert 0 <= w.conf <= 100 and w.engine == "rapid" and w.view == 0
        assert w.text in ocr.lines[w.line_index].split()
    assert ocr.mean_conf is not None and 50 <= ocr.mean_conf <= 100
    assert ocr.engine.startswith("rapidocr ") and "onnxruntime" in ocr.engine
    assert ocr.ms > 0 and ocr.source is not None


@requires_rapid
def test_rapid_reader_passes_the_sample_end_to_end(old_tom_reading):
    from app.models import Application, Status, Verdict
    from app.pipeline import verify
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="45% Alc./Vol.", net_contents="750 mL",
                      bottler_name_address="Old Tom Distillery, Bardstown, Kentucky 40004")
    result = verify(app, Image.open(SAMPLES / "old_tom_clean.png"), R.RapidOCRReader(escalate_with_tesseract=False))
    assert {f.verdict for f in result.fields if f.verdict != Verdict.SKIPPED} == {Verdict.MATCH}, result.fields
    assert result.warning.overall in (Status.PASS, Status.REVIEW), result.warning
    assert result.reader.startswith("rapidocr")


# --- the line -> word split, no model needed ------------------------------------------------------
def _quad(left, top, width, height):
    return [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]


def test_line_boxes_are_split_across_words_by_character_count():
    res = [[_quad(100, 50, 400, 40), "OLD TOM DISTILLERY", 0.97],      # 18 characters wide
           [_quad(10, 100, 30, 300), "GOVERNMENT WARNING", 0.9],        # printed sideways: split top to bottom
           [_quad(0, 0, 10, 10), "   ", 0.99], [_quad(0, 0, 10, 10), "low", 0.2]]
    lines, words, mean = R.lines_from_result(res, min_conf=60)
    assert lines == ["OLD TOM DISTILLERY", "GOVERNMENT WARNING"]
    old, tom, dist = words[:3]
    assert (old.left, old.width) == (100, round(400 * 3 / 18)) and old.top == 50 and old.height == 40
    assert tom.left == round(100 + 400 * 4 / 18) and dist.right == 500
    assert old.conf == 97.0 and old.engine == "rapid" and old.line_index == 0
    gov, warn = words[3:]
    assert gov.left == warn.left == 10 and gov.width == warn.width == 30
    assert gov.top == 100 and warn.bottom == 400 and gov.bottom <= warn.top
    assert mean == pytest.approx((97 * 3 + 90 * 2) / 5)


def test_input_array_is_padded_to_the_detector_grid_without_moving_pixels():
    img = Image.new("L", (1799, 1572), 200)
    arr = R.rapid_input(img, img, 0.0, (1799, 1572), mode="gray")
    assert arr.shape == (1600, 1824) and arr.dtype == np.uint8 and arr.flags["WRITEABLE"]
    assert arr[:1572, :1799].min() == 200 and arr[1572:, :].min() == 255 and arr[:, 1799:].min() == 255
    color = R.rapid_input(img, Image.new("RGB", (1799, 1572), (10, 20, 30)), 0.0, (1799, 1572), mode="color")
    assert color.shape == (1600, 1824, 3) and tuple(color[0, 0]) == (30, 20, 10)   # BGR, as the engine expects


# --- the escalation, with a fake engine -----------------------------------------------------------
def _tesseract_like(lines: list[str]) -> OCRResult:
    words, y = [], 100
    for li, line in enumerate(lines):
        x = 100
        for tok in line.split():
            words.append(OCRWord(text=tok, left=x, top=y, width=60, height=40, conf=85, line_index=li))
            x += 70
        y += 100
    ink = np.zeros((1000, 1600), dtype=bool)
    img = Image.new("L", (1600, 1000), 255)
    return OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink, mean_conf=85, engine="tesseract x",
                     views=[View(rot=0, inverted=False, ink=ink, size=img.size)], source=(img, img, 0.0, img.size))


def test_escalation_appends_rapid_lines_as_a_new_view(monkeypatch):
    ocr = _tesseract_like(["OLD TOM D1STILLERY", "750 mL"])
    fake = [[_quad(100, 100, 400, 40), "OLD TOM DISTILLERY", 0.98],
            [_quad(100, 200, 100, 40), "750 mL", 0.95],              # the same text again: kept (views are whole)
            [_quad(100, 300, 100, 40), "noise", 0.55]]               # under RAPID_MIN_LINE_CONF: dropped
    calls = []
    monkeypatch.setattr(R, "run_engine", lambda arr, timeout=0: calls.append(arr.shape) or fake)
    assert R.add_rapid_view(ocr) is True
    assert calls == [(1024, 1600)]
    assert ocr.lines == ["OLD TOM D1STILLERY", "750 mL", "OLD TOM DISTILLERY", "750 mL"]
    assert len(ocr.views) == 2 and ocr.views[1].rot == 0 and ocr.views[1].ink is ocr.views[0].ink
    new = [w for w in ocr.words if w.view == 1]
    assert [w.text for w in new] == ["OLD", "TOM", "DISTILLERY", "750", "mL"]
    assert all(w.engine == "rapid" and w.line_index in (2, 3) for w in new)
    assert ocr.text.endswith("750 mL") and ocr.escalated and "rapidocr" in ocr.engine
    # Runs once per label.
    assert R.add_rapid_view(ocr) is False and len(calls) == 1


def test_escalation_is_skipped_when_the_engine_fails(monkeypatch):
    ocr = _tesseract_like(["OLD TOM D1STILLERY"])

    def broken(arr, timeout=0):
        raise ReaderError("boom")
    monkeypatch.setattr(R, "run_engine", broken)
    assert R.add_rapid_view(ocr) is False
    assert ocr.lines == ["OLD TOM D1STILLERY"] and len(ocr.views) == 1 and ocr.escalated


def test_escalated_reading_resolves_a_field_the_first_pass_missed(monkeypatch):
    from app.models import Application, Verdict
    from app.pipeline import verify
    from app.readers.tesseract import TesseractReader

    class FirstPass(TesseractReader):
        def read(self, image):
            return LabelReading(ocr=_tesseract_like(["OLD TOM D1STILLERY", "Kentucky Straight Bourbon Whiskey",
                                                     "45% Alc./Vol.", "750 mL"]))

        def extend(self, reading):
            return False

    fake = [[_quad(100, 100, 400, 40), "OLD TOM DISTILLERY", 0.98]]
    monkeypatch.setattr(R, "run_engine", lambda arr, timeout=0: fake)
    monkeypatch.setattr("app.readers.tesseract.RAPID_ESCALATION", True)
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="45% Alc./Vol.", net_contents="750 mL")
    r = verify(app, Image.new("RGB", (10, 10)), FirstPass(), budget_s=0)
    brand = next(f for f in r.fields if f.key == "brand_name")
    # Tesseract's "D1STILLERY" folds to the same letters as the application value (1 -> l), so the
    # RapidOCR reading is a plain MATCH, not a disagreement.
    assert brand.verdict == Verdict.MATCH and brand.found == "OLD TOM DISTILLERY", brand
    assert brand.box is not None and "rapidocr" in r.reader


def test_rapid_reading_wins_a_conflict_only_when_confident():
    sure = OCRWord(text="BARK", left=0, top=0, width=10, height=10, conf=95, line_index=0, engine="rapid")
    unsure = OCRWord(text="BARK", left=0, top=0, width=10, height=10, conf=70, line_index=0, engine="rapid")
    tesseract = OCRWord(text="BARK", left=0, top=0, width=10, height=10, conf=95, line_index=0)
    assert conflict_confidence(sure) == 100 and conflict_confidence(unsure) == 70 and conflict_confidence(tesseract) == 95


def test_a_confident_rapid_reading_that_disagrees_forces_a_look(monkeypatch):
    from app.models import Application, Verdict
    from app.pipeline import verify
    from app.readers.tesseract import TesseractReader

    class FirstPass(TesseractReader):
        def read(self, image):
            ocr = _tesseract_like(["BARN BREW CO.", "Beer", "6% Alc./Vol.", "12 FL OZ", "Bottled by East Rock, New Haven"])
            return LabelReading(ocr=ocr)

        def extend(self, reading):
            return False

    fake = [[_quad(100, 100, 400, 40), "BARK BREW CO.", 0.97]]
    monkeypatch.setattr(R, "run_engine", lambda arr, timeout=0: fake)
    monkeypatch.setattr("app.readers.tesseract.RAPID_ESCALATION", True)
    app = Application(brand_name="BARN BREW CO.", class_type="Beer", alcohol_content="6% Alc./Vol.", net_contents="12 fl oz",
                      bottler_name_address="East Rock, New Haven, CT")
    r = verify(app, Image.new("RGB", (10, 10)), FirstPass(), budget_s=0)
    brand = next(f for f in r.fields if f.key == "brand_name")
    assert brand.verdict == Verdict.NEAR_MATCH and "BARK BREW" in brand.note, brand


def test_timeout_is_a_reader_error(monkeypatch):
    import time

    class Slow:
        def __call__(self, arr):
            time.sleep(0.3)
            return [], None
    monkeypatch.setattr(R, "engine", Slow)
    with pytest.raises(ReaderError, match="longer than"):
        R.run_engine(np.zeros((64, 64), dtype=np.uint8), timeout=0.05)


# --- registry and health ---------------------------------------------------------------------------
def test_get_reader_rapid(monkeypatch):
    monkeypatch.setattr(main, "rapid_version", lambda: "1.4.4")
    monkeypatch.setattr(main, "_readers", {})
    reader = main.get_reader("rapid")
    assert isinstance(reader, R.RapidOCRReader) and main.get_reader("rapid") is reader
    monkeypatch.setattr(main, "rapid_version", lambda: None)
    monkeypatch.setattr(main, "_readers", {})
    with pytest.raises(main.UserError, match="not installed"):
        main.get_reader("rapid")


def test_healthz_reports_rapidocr(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "rapid_version", lambda: "1.4.4")
    body = client.get("/healthz").json()
    assert body["rapidocr"] == "1.4.4" and body["rapid_escalation"] == main.RAPID_ESCALATION
    assert "RapidOCR 1.4.4" in main.reader_info()
    monkeypatch.setattr(main, "rapid_version", lambda: None)
    body = client.get("/healthz").json()
    assert body["rapidocr"] is None and body["rapid_escalation"] is False
    assert "RapidOCR" not in main.reader_info()


def test_run_together_rapid_lines_do_not_replace_tesseract_warning_lines():
    """RapidOCR often drops the spaces in small print. Its view is searched as a frame of its own, so a
    run-together reading of the same place never replaces Tesseract's word-for-word one."""
    from app.config import MANDATED_WARNING
    from app.models import Status
    from app.warning import check_warning
    text = MANDATED_WARNING.split()
    tess_lines = [" ".join(text[i:i + 9]) for i in range(0, len(text), 9)]
    ocr = _tesseract_like(tess_lines)
    rapid_lines = [line.replace(" ", "") if k % 2 else line for k, line in enumerate(tess_lines)]
    res = [[_quad(100, 100 + 100 * k, 600, 40), line, 0.97] for k, line in enumerate(rapid_lines)]
    lines2, words2, _ = R.lines_from_result(res)
    R.append_view(ocr, lines2, words2, View(rot=0, inverted=False, ink=ocr.views[0].ink, size=ocr.views[0].size,
                                           engine=R.ENGINE_KEY))
    w = check_warning(ocr.lines, ocr.words, ocr.ink, views=ocr.views)
    assert w.present and w.wording == Status.PASS, (w.wording_note, w.found_text)
