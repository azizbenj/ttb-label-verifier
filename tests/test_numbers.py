"""The second read of the figures: alcohol content and net contents lines re-read from their own crop
(app/readers/numbers.py), and what the matcher makes of a reading that corrects the first one."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.config import TESSERACT_EXTRA_TIMEOUT_S, THRESHOLDS
from app.matching import (NOTE_ALSO_READS, NOTE_CONFIRMED, NOTE_READ_ON_CROP, NOTE_SECOND_READ_SAME, compare_alcohol,
                          compare_volume, wants_second_read)
from app.models import Application, FieldResult, Status, Verdict
from app.normalize import alcohol_candidates, volume_candidates
from app.pipeline import verify
from app.readers import numbers as N
from app.readers import tesseract as T
from app.readers.base import LabelReading, OCRResult, OCRWord, View, text_height, upright_box
from app.readers.extract import extract_and_compare, line_heights, prominent_line_index

from .conftest import requires_tesseract

FONTS = Path(__file__).resolve().parents[1] / "data" / "fonts"


# --- the crop itself, read with Tesseract ---------------------------------------------------------
def _render(size: int, blur: float = 0.0) -> Image.Image:
    img = Image.new("L", (1600, 400), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), size)
    d.text((100, 60), "45% Alc./Vol.", font=f, fill=0)
    d.text((100, 60 + size * 3), "1.5 L", font=f, fill=0)
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


@requires_tesseract
@pytest.mark.parametrize("size,blur", [(14, 0.0), (22, 0.8), (60, 0.0)])
def test_small_or_degraded_figures_are_read_right_from_their_crop(size, blur):
    """14 px type, a blurred line, and 60 px type whose "1." the page pass reads as "L": the crop,
    scaled to the line height Tesseract reads best, must give the figure printed."""
    reader = T.TesseractReader()
    reading = reader.read(_render(size, blur))
    assert reader.reread_numbers(reading, {"alcohol", "volume"})
    ocr = reading.ocr
    volumes = [c for r in ocr.rereads if "volume" in r.kinds for c in volume_candidates(r.text)]
    alcohols = [c for r in ocr.rereads if "alcohol" in r.kinds for c in alcohol_candidates(r.text)]
    assert any(abs(c.ml - 1500) < 0.5 for c in volumes), [(r.text, r.mode) for r in ocr.rereads]
    assert all(abs(c.ml - 1500) < 0.5 or c.digits == "15" for c in volumes), [(r.text, r.mode) for r in ocr.rereads]
    assert any(abs(c.abv - 45) < 0.05 for c in alcohols), [(r.text, r.mode) for r in ocr.rereads]
    assert "figure crops" in ocr.engine
    # Each second read is a line of its own, in a view that puts its words back on the page.
    for r in ocr.rereads:
        if r.new_line is not None:
            words = [w for w in ocr.words if w.line_index == r.new_line]
            assert words and all(ocr.views[w.view].scale != 1.0 for w in words)
            box = upright_box(words[0], ocr.views)
            assert 0 <= box[0] < ocr.views[0].size[0] and 0 <= box[1] < ocr.views[0].size[1]


# --- the matcher's rule for a corrected reading ----------------------------------------------------
def test_one_edit_away_and_the_closer_read_agrees_is_a_confirmed_match():
    r = compare_volume("1.5 L", "ALC. 12% BY VOL.\n15L\n1.5L", rereads=[("15L", ["1.5L"])])
    assert r.verdict == Verdict.MATCH and NOTE_CONFIRMED in r.note and "'15L'" in r.note, r
    r = compare_volume("750 mL", "790 mL\n750 mL", rereads=[("790 mL", ["750 mL"])])
    assert r.verdict == Verdict.MATCH and "'790 mL'" in r.note, r
    r = compare_alcohol("57.7% Alc./Vol.", "07.7% ALC./VOL.\n57.7% ALC./VOL.", rereads=[("07.7% ALC./VOL.", ["57.7% ALC./VOL."])])
    assert r.verdict == Verdict.MATCH and NOTE_CONFIRMED in r.note and r.found == "57.7% ALC./VOL.", r
    # The page's proof disagreeing with its percentage, corrected by the closer read ("80" for "90").
    r = compare_alcohol("45%", "45% ALC./VOL. (80 PROOF)\nBottled by Old Tom\n45% ALC./VOL. (90 PROOF)",
                        rereads=[("45% ALC./VOL. (80 PROOF)", ["45% ALC./VOL. (90 PROOF)"])])
    assert r.verdict == Verdict.MATCH and NOTE_CONFIRMED in r.note, r


def test_a_page_reading_further_than_one_edit_away_stays_a_near_match_naming_both():
    r = compare_volume("1.75 L", "1 L\n1.75 L", rereads=[("1 L", ["1.75 L"])])
    assert r.verdict == Verdict.NEAR_MATCH and NOTE_ALSO_READS in r.note and "1 L" in r.found and "1.75 L" in r.found, r


def test_a_closer_read_that_disagrees_never_outvotes_anything():
    # The page agreed; the crop says something else: both are shown, as for any two readings.
    r = compare_volume("750 mL", "750 mL\n790 mL", rereads=[("750 mL", ["790 mL"])])
    assert r.verdict == Verdict.NEAR_MATCH and NOTE_ALSO_READS in r.note, r
    # Two closer reads of one line, one agreeing and one not: no confirmation, both readings named.
    r = compare_volume("750 mL", "790 mL\n750 mL\n760 mL", rereads=[("790 mL", ["750 mL", "760 mL"])])
    assert r.verdict == Verdict.NEAR_MATCH and NOTE_CONFIRMED not in r.note and NOTE_ALSO_READS in r.note, r
    r = compare_alcohol("45%", "40% ALC./VOL.\n45% ALC./VOL.\n46% ALC./VOL.",
                        rereads=[("40% ALC./VOL.", ["45% ALC./VOL.", "46% ALC./VOL."])])
    assert r.verdict == Verdict.NEAR_MATCH and NOTE_CONFIRMED not in r.note, r


def test_a_wrong_number_stays_a_mismatch_and_says_it_was_read_twice():
    r = compare_volume("750 mL", "700 mL", rereads=[("700 mL", ["700 mL"])])
    assert r.verdict == Verdict.MISMATCH and NOTE_SECOND_READ_SAME in r.note, r
    r = compare_alcohol("45%", "40% ALC./VOL.", rereads=[("40% ALC./VOL.", ["40% ALC./VOL."])])
    assert r.verdict == Verdict.MISMATCH and NOTE_SECOND_READ_SAME in r.note, r
    # A second read that agrees with the first on a dropped decimal point: still a look, never a match.
    r = compare_volume("1.5 L", "15L", rereads=[("15L", ["15 L"])])
    assert r.verdict == Verdict.NEAR_MATCH and NOTE_SECOND_READ_SAME in r.note, r


def test_a_figure_the_page_pass_never_read_is_taken_from_the_crop_and_says_so():
    r = compare_volume("1.75 L", "L751\n1.75 L", rereads=[("L751", ["1.75 L"])])
    assert r.verdict == Verdict.MATCH and NOTE_READ_ON_CROP in r.note, r
    r = compare_alcohol("45% Alc./Vol.", "4S% ALC./VOL.\n45% ALC./VOL.", rereads=[("4S% ALC./VOL.", ["45% ALC./VOL."])])
    assert r.verdict == Verdict.MATCH and NOTE_READ_ON_CROP in r.note, r


def test_a_closer_read_with_the_point_lost_beside_one_with_it_is_harmless():
    r = compare_volume("1.5 L", "ALC. 15% BY VOL. LSL\n15L\n1.5L", rereads=[("ALC. 15% BY VOL. LSL", ["15L", "1.5L"])])
    assert r.verdict == Verdict.MATCH and NOTE_CONFIRMED in r.note, r


def test_without_rereads_the_rules_are_unchanged():
    assert compare_volume("1.5 L", "15L").verdict == Verdict.NEAR_MATCH
    assert compare_volume("750 mL", "750 mL").verdict == Verdict.MATCH
    assert compare_alcohol("45%", "45% ALC./VOL.").note == "45% ABV on both."


# --- the trigger -----------------------------------------------------------------------------------
def test_second_read_is_wanted_only_for_doubtful_figures():
    assert wants_second_read(compare_volume("750 mL", "no volume here")) == "volume"
    assert wants_second_read(compare_volume("750 mL", "700 mL")) == "volume"
    assert wants_second_read(compare_volume("1.5 L", "15L")) == "volume"                 # point lost
    assert wants_second_read(compare_volume("750 mL", "760 mL")) == "volume"             # probable misread
    assert wants_second_read(compare_volume("750 mL", "750 mL\n790 mL")) == "volume"     # readings disagree
    assert wants_second_read(compare_alcohol("45%", "40% ALC./VOL.")) == "alcohol"
    assert wants_second_read(compare_alcohol("45%", "45% ALC./VOL. (80 PROOF)")) == "alcohol"   # proof disagrees
    assert wants_second_read(compare_volume("750 mL", "750 mL")) is None
    assert wants_second_read(compare_alcohol("13.5%", "Blend: 13.5% Petit Verdot")) is None    # not marked as alcohol
    assert wants_second_read(compare_volume("seven fifty", "750 mL")) is None            # the application is at fault
    assert wants_second_read(FieldResult(key="brand_name", label="Brand", expected="X", found=None,
                                         verdict=Verdict.NOT_FOUND)) is None


# --- the pipeline, with a fake reader -----------------------------------------------------------------
WARNING = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages "
           "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs "
           "your ability to drive a car or operate machinery, and may cause health problems.")


def make_ocr(lines: list[str], source=None) -> OCRResult:
    words, y = [], 100
    for li, line in enumerate(lines):
        x = 100
        for tok in line.split():
            words.append(OCRWord(text=tok, left=x, top=y, width=60, height=40 if li else 90, conf=90, line_index=li))
            x += 70
        y += 100
    ink = np.zeros((1000, 2000), dtype=bool)
    return OCRResult(text="\n".join(lines), lines=list(lines), words=words, ink=ink, mean_conf=90, engine="fake",
                     views=[View(rot=0, inverted=False, ink=ink, size=(2000, 1000))], source=source)


class FakeReader:
    """Reads fixed lines; a second read appends the text it was told to, as the real one would."""

    name = "fake"

    def __init__(self, lines: list[str], rereads: dict[str, list[tuple[int, str]]] | None = None):
        self.lines, self.rereads, self.calls = lines, rereads or {}, []

    def read(self, image):
        return LabelReading(ocr=make_ocr(self.lines))

    def reread_numbers(self, reading: LabelReading, kinds: set[str]) -> bool:
        self.calls.append(set(kinds))
        did = False
        for kind in sorted(kinds):
            for line_index, text in self.rereads.get(kind, []):
                view = View(rot=0, inverted=False, ink=np.zeros((30, 300), bool), size=(300, 30), scale=0.5,
                            offset=(100, 100 + 100 * line_index))
                words = [OCRWord(text=t, left=35 * i, top=0, width=30, height=20, conf=80, line_index=0)
                         for i, t in enumerate(text.split())]
                N.record(reading.ocr, line_index, (kind,), "fake", [text], words, view)
                did = True
        reading.ocr.text = "\n".join(reading.ocr.lines)
        return did


APP = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                  alcohol_content="45% Alc./Vol.", net_contents="1.5 L")
PAGE = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% ALC./VOL.", "L5L", WARNING]


def _field_verdicts(result) -> set[Verdict]:
    return {f.verdict for f in result.fields if f.verdict != Verdict.SKIPPED}


def test_a_clean_label_is_never_read_twice():
    reader = FakeReader(PAGE[:3] + ["1.5 L"] + PAGE[4:])
    assert _field_verdicts(verify(APP, Image.new("RGB", (10, 10)), reader)) == {Verdict.MATCH}
    assert reader.calls == []


def test_a_lost_decimal_point_is_confirmed_by_the_crop_and_passes():
    reader = FakeReader(PAGE, {"volume": [(3, "1.5 L")]})
    r = verify(APP, Image.new("RGB", (10, 10)), reader)
    assert reader.calls == [{"volume"}]
    by_key = {f.key: f for f in r.fields}
    assert by_key["net_contents"].verdict == Verdict.MATCH and NOTE_CONFIRMED in by_key["net_contents"].note
    assert _field_verdicts(r) == {Verdict.MATCH} and r.overall != Status.FAIL
    # The evidence box points at the crop's place on the page (line 3 sits at y = 400 of 1000).
    assert by_key["net_contents"].box is not None and by_key["net_contents"].box[1] == 40.0
    assert "1.5 L" in r.ocr_text


def test_a_second_read_for_one_figure_never_touches_a_clean_match_on_the_other():
    # The crop made for the volume also carries a (mis)read percentage: the alcohol content, a clean MATCH
    # on the page, must not see it.
    reader = FakeReader(PAGE, {"volume": [(3, "ALC. 40% BY VOL. 1.5 L")]})
    r = verify(APP, Image.new("RGB", (10, 10)), reader)
    by_key = {f.key: f for f in r.fields}
    assert by_key["alcohol_content"].verdict == Verdict.MATCH and by_key["alcohol_content"].note == "45% ABV on both."
    assert by_key["net_contents"].verdict == Verdict.MATCH
    assert reader.calls == [{"volume"}]


def test_a_wrong_number_stays_a_mismatch_after_the_second_read():
    reader = FakeReader(PAGE[:3] + ["700 mL"] + PAGE[4:], {"volume": [(3, "700 mL")]})
    r = verify(APP, Image.new("RGB", (10, 10)), reader)
    f = next(f for f in r.fields if f.key == "net_contents")
    assert f.verdict == Verdict.MISMATCH and NOTE_SECOND_READ_SAME in f.note and r.overall == Status.FAIL


def test_a_disagreeing_second_read_asks_for_a_look():
    reader = FakeReader(PAGE[:3] + ["1.5 L", "1.5 L"] + PAGE[4:])   # clean page ...
    reader.lines = PAGE[:2] + ["40% ALC./VOL."] + PAGE[3:]            # ... but the alcohol is wrong on the page
    reader.rereads = {"alcohol": [(2, "45% ALC./VOL.")], "volume": [(3, "1.5 L")]}
    r = verify(APP, Image.new("RGB", (10, 10)), reader)
    by_key = {f.key: f for f in r.fields}
    assert reader.calls == [{"alcohol", "volume"}]
    # "40%" for "45%" is one digit off and the closer read agrees with the application: confirmed.
    assert by_key["alcohol_content"].verdict == Verdict.MATCH and NOTE_CONFIRMED in by_key["alcohol_content"].note
    # The same with a closer read that says yet another figure: a look, naming both.
    reader = FakeReader(PAGE[:2] + ["40% ALC./VOL."] + PAGE[3:], {"alcohol": [(2, "46% ALC./VOL.")], "volume": [(3, "1.5 L")]})
    r = verify(APP, Image.new("RGB", (10, 10)), reader)
    f = next(f for f in r.fields if f.key == "alcohol_content")
    assert f.verdict == Verdict.MISMATCH and NOTE_CONFIRMED not in f.note and r.overall == Status.FAIL, f


def test_crop_lines_do_not_become_the_brand_line():
    # Words read from a crop are scaled back to their size on the page before the largest line is chosen.
    ocr = make_ocr(PAGE)
    view = View(rot=0, inverted=False, ink=np.zeros((60, 600), bool), size=(600, 60), scale=3.0, offset=(100, 400))
    words = [OCRWord(text="1500", left=0, top=0, width=90, height=120, conf=80, line_index=0),
             OCRWord(text="mL", left=100, top=0, width=90, height=120, conf=80, line_index=0)]
    N.record(ocr, 3, ("volume",), "fake", ["1500 mL"], words, view)
    heights = line_heights(ocr)
    assert prominent_line_index(ocr, heights) == 0 and heights[5] == 40.0
    assert text_height(words[0], ocr.views) == 40.0
    # A garbled line shrunk as far as the crop goes can come back as a few tall letters: the brand line
    # (and its fallback, shown when nothing resembles the brand) is still chosen among the page's lines.
    small = View(rot=0, inverted=False, ink=np.zeros((60, 600), bool), size=(600, 60), scale=0.25, offset=(100, 400))
    N.record(ocr, 2, ("alcohol",), "fake", ["I ee"], [OCRWord(text="ee", left=0, top=0, width=30, height=30, conf=40, line_index=0)], small)
    app = Application(brand_name="NOT ON THE LABEL", class_type="x", alcohol_content="45%", net_contents="1.5 L")
    brand = next(f for f in extract_and_compare(app, ocr) if f.key == "brand_name")
    assert brand.verdict == Verdict.MISMATCH and brand.found == "OLD TOM DISTILLERY"


# --- the reader's bookkeeping: crops, views, time limit -------------------------------------------------
def test_crop_view_words_map_back_onto_the_page():
    views = [View(rot=0, inverted=False, ink=np.zeros((10, 10), bool), size=(2000, 1000)),
             View(rot=0, inverted=False, ink=np.zeros((10, 10), bool), size=(500, 100), scale=2.5, offset=(100, 200)),
             View(rot=90, inverted=False, ink=np.zeros((10, 10), bool), size=(100, 400), scale=2.0, offset=(100, 200))]
    assert upright_box(OCRWord(text="x", left=50, top=10, width=100, height=40, conf=90, line_index=0, view=1), views) \
        == (120, 204, 40, 16)
    # A region 200 x 50 at (100, 200) turned counter-clockwise and doubled: a word drawn at (10, 5, 60, 20) of the
    # region sits at (5, 130, 20, 60) in the turned crop, (10, 260, 40, 120) once doubled.
    assert upright_box(OCRWord(text="x", left=10, top=260, width=40, height=120, conf=90, line_index=0, view=2), views) \
        == (110, 205, 60, 20)


def test_which_lines_are_worth_a_second_read():
    assert N.line_kinds("45% ALC./VOL. (90 PROOF)") == {"alcohol": True}
    assert N.line_kinds("12 FL OZ (355 mL)") == {"volume": True}
    assert N.line_kinds("ALC. 15% BY VOL. LSL") == {"alcohol": True, "volume": True}
    assert N.line_kinds("L751") == {"volume": False}                      # digits mixed with their look-alikes
    assert N.line_kinds("4S% ALC./VOL.") == {"alcohol": False}
    assert N.line_kinds("Milwaukee, WI 53202") == {}                      # a run of digits alone is not a figure
    assert N.line_kinds("GOVERNMENT WARNING: (1) According to") == {}
    ocr = make_ocr(["OLD TOM", "4S% ALC./VOL.", "750 mL", "L751", "Bardstown, KY 40004", "45% ALC./VOL."])
    # Outright figures first, then the look-alikes; only the kinds asked for; capped.
    assert N.lines_to_reread(ocr, {"volume"}, 4) == [(2, ("volume",)), (3, ("volume",))]
    assert N.lines_to_reread(ocr, {"alcohol", "volume"}, 2) == [(2, ("volume",)), (5, ("alcohol",))]
    N.record(ocr, 2, ("volume",), "fake", ["750 mL."], [], View(rot=0, inverted=False, ink=np.zeros((1, 1), bool), size=(1, 1)))
    assert all(i not in (2, 6) for i, _ in N.lines_to_reread(ocr, {"volume"}, 4))   # not twice, and not the second read


def _reading_with_source(lines: list[str]) -> LabelReading:
    page = Image.new("L", (2000, 1000), 255)
    return LabelReading(ocr=make_ocr(lines, source=(page, page, 0.0, page.size)))


def test_a_slow_or_failing_crop_pass_is_skipped_within_the_time_limit(monkeypatch):
    seen = []

    def slow(*a, **k):
        seen.append(k.get("timeout"))
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr(T.pytesseract, "image_to_data", slow)
    reading = _reading_with_source(PAGE)
    before = list(reading.ocr.lines)
    assert T.TesseractReader().reread_numbers(reading, {"volume"}) is False
    assert reading.ocr.lines == before and reading.ocr.rereads == []
    assert seen and all(t == TESSERACT_EXTRA_TIMEOUT_S for t in seen)


def test_at_most_a_handful_of_crops_each_read_once(monkeypatch):
    calls = []

    def fake(img, config="", output_type=None, timeout=0):
        calls.append((img.size, config))
        return {k: [] for k in ("text", "conf", "block_num", "par_num", "line_num", "left", "top", "width", "height")}
    monkeypatch.setattr(T.pytesseract, "image_to_data", fake)
    reading = _reading_with_source(["OLD TOM"] + [f"{n} mL" for n in range(700, 800, 10)])
    T.TesseractReader().reread_numbers(reading, {"volume"})
    assert len(calls) == THRESHOLDS.number_reread_max_crops
    assert all("--psm 7" in c for _, c in calls)
    # Each crop is scaled so its line is about number_reread_text_px tall (the words are 40 px tall).
    assert all(abs(size[1] - (40 + 2 * 20) * THRESHOLDS.number_reread_text_px / 40) <= 2 for size, _ in calls)
