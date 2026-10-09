from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.config import MANDATED_WARNING, THRESHOLDS
from app.models import Status
from app.readers.base import OCRWord
from app.warning import check_heading_caps, check_warning, check_wording, estimate_heading_bold, locate_warning

FONTS = Path(__file__).resolve().parents[1] / "data" / "fonts"

WARNING_LINES = [
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women",
    "should not drink alcoholic beverages during pregnancy because of the",
    "risk of birth defects. (2) Consumption of alcoholic beverages impairs",
    "your ability to drive a car or operate machinery, and may cause",
    "health problems.",
]
LABEL_LINES = ["OLD TOM DISTILLERY", "KENTUCKY STRAIGHT BOURBON WHISKEY", "45% ALC./VOL. (90 PROOF)", "750 mL"]


def test_locate_span_covers_statement_only():
    lines = LABEL_LINES + WARNING_LINES + ["Bottled by Old Tom Distillery, Bardstown KY"]
    span = locate_warning(lines)
    assert span.start == 4 and span.end == 8 and span.heading_line == 4


def test_locate_none_when_missing():
    assert locate_warning(LABEL_LINES) is None


def test_wording_exact_passes():
    status, score, _, diff = check_wording(MANDATED_WARNING)
    assert status == Status.PASS and score == 100 and diff == []


def test_wording_ignores_punctuation_and_case():
    text = MANDATED_WARNING.replace(",", "").replace("(1)", "1").lower()
    assert check_wording(text)[0] == Status.PASS


def test_wording_one_word_changed_needs_review_with_diff():
    text = MANDATED_WARNING.replace("may cause", "can cause")
    status, score, _, diff = check_wording(text)
    assert status == Status.REVIEW
    assert [(d.expected, d.found) for d in diff] == [("may", "can")]
    assert diff[0].expected_before.endswith("machinery, and") and diff[0].found_after.startswith("cause")


def test_wording_truncated_fails():
    text = MANDATED_WARNING.split("(2)")[0]
    assert check_wording(text)[0] == Status.FAIL


def test_heading_caps():
    assert check_heading_caps("GOVERNMENT WARNING: (1) According")[0] == Status.PASS
    assert check_heading_caps("Government Warning: (1) According")[0] == Status.FAIL
    assert check_heading_caps("GOVERNMENT WARNING (1) According")[0] == Status.REVIEW
    assert check_heading_caps(None)[0] == Status.FAIL


def test_check_warning_full_pass_with_vision_hint():
    r = check_warning(LABEL_LINES + WARNING_LINES, bold_hint=True)
    assert r.present and r.wording == Status.PASS and r.heading_caps == Status.PASS
    assert r.heading_bold == Status.PASS and r.overall == Status.PASS


def test_check_warning_missing():
    r = check_warning(LABEL_LINES)
    assert not r.present and r.overall == Status.FAIL


def test_check_warning_title_case_heading_fails():
    lines = LABEL_LINES + ["Government Warning: (1) According to the Surgeon General, women"] + WARNING_LINES[1:]
    r = check_warning(lines, bold_hint=True)
    assert r.heading_caps == Status.FAIL and r.overall == Status.FAIL


def test_check_warning_no_bold_info_asks_for_review():
    r = check_warning(LABEL_LINES + WARNING_LINES)
    assert r.heading_bold == Status.REVIEW and r.overall == Status.REVIEW


# --- bold heuristic on rendered text -----------------------------------------------------------
def _render(font_regular: str, font_bold: str, size: int, heading_bold: bool, body_bold: bool = False):
    """Render a two-line statement and return (ink array, heading words, body words)."""
    bold = ImageFont.truetype(str(FONTS / font_bold), size)
    reg = bold if body_bold else ImageFont.truetype(str(FONTS / font_regular), size)
    img = Image.new("L", (size * 40, size * 6), 255)
    draw = ImageDraw.Draw(img)
    words: list[tuple[str, ImageFont.FreeTypeFont, int]] = []
    heading = [("GOVERNMENT", bold if heading_bold else reg, 0), ("WARNING:", bold if heading_bold else reg, 0)]
    body0 = [(w, reg, 0) for w in "(1) According to the Surgeon General, women".split()]
    body1 = [(w, reg, 1) for w in "should not drink alcoholic beverages during pregnancy".split()]
    words = heading + body0 + body1
    boxes: list[OCRWord] = []
    x, line = 10, 0
    for text, font, line_no in words:
        if line_no != line:
            x, line = 10, line_no
        y = 10 + line_no * int(size * 1.6)
        draw.text((x, y), text, font=font, fill=0)
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        boxes.append(OCRWord(text=text, left=l, top=t, width=r - l, height=b - t, conf=95, line_index=line_no))
        x = r + int(size * 0.4)
    ink = np.array(img) < 128
    return ink, boxes[:2], boxes[2:]


@pytest.mark.parametrize("fonts", [("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf")])
@pytest.mark.parametrize("size", [22, 40])
def test_bold_heuristic_separates_bold_from_regular(fonts, size):
    ink, head, body = _render(fonts[0], fonts[1], size, heading_bold=True)
    status_b, ratio_b, _ = estimate_heading_bold(ink, head, body)
    ink, head, body = _render(fonts[0], fonts[1], size, heading_bold=False)
    status_r, ratio_r, _ = estimate_heading_bold(ink, head, body)
    assert status_b == Status.PASS and ratio_b >= THRESHOLDS.bold_ratio, (fonts, size, ratio_b)
    assert status_r == Status.REVIEW and ratio_r < THRESHOLDS.bold_ratio, (fonts, size, ratio_r)


def test_bold_heuristic_unmeasurable():
    ink = np.zeros((10, 10), dtype=bool)
    status, ratio, _ = estimate_heading_bold(ink, [], [])
    assert status == Status.REVIEW and ratio is None


def test_locate_ignores_short_noise_lines():
    lines = LABEL_LINES + ["a", "—"] + WARNING_LINES
    span = locate_warning(lines)
    assert span.heading_line == 6 and lines[span.start].startswith("GOVERNMENT")
    assert check_warning(lines, bold_hint=True).overall == Status.PASS


def test_heading_with_an_ocr_misread_letter_asks_for_review_not_fail():
    assert check_heading_caps("GOVERNMENT WARNlNG: (1) According")[0] == Status.REVIEW
    assert check_heading_caps("G0VERNMENT WARNING: (1) According")[0] == Status.REVIEW
    assert check_heading_caps("Government Warnlng: (1) According")[0] == Status.FAIL  # title case stays a failure
    assert check_heading_caps("WARNING: (1) According to the Surgeon General")[0] == Status.FAIL


@pytest.mark.parametrize("heading_bold", [True, False])
def test_bold_heuristic_on_light_text_over_a_dark_panel(heading_bold):
    ink, head, body = _render("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", 30, heading_bold=heading_bold)
    _, ratio, _ = estimate_heading_bold(ink, head, body)
    # The same statement printed light-on-dark: the global ink mask now marks the panel, not the letters.
    _, ratio_dark, _ = estimate_heading_bold(~ink, head, body)
    assert ratio_dark == pytest.approx(ratio, abs=0.05), (ratio, ratio_dark)


@pytest.mark.parametrize("fonts", [("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf")])
def test_all_bold_statement_asks_for_review(fonts):
    # 27 CFR 16.22(a)(2): the rest of the statement may not be bold. An all-bold statement measures ~1.0.
    ink, head, body = _render(fonts[0], fonts[1], 30, heading_bold=True, body_bold=True)
    assert estimate_heading_bold(ink, head, body)[0] == Status.REVIEW
