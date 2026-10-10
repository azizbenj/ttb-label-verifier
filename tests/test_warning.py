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


def test_statement_assembled_past_an_interleaved_line_and_a_neighbouring_column():
    lines = ["GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT",
             "DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH",
             "For Sale Only In Ohio",
             "DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A",
             "CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS. KEEP COLD AT ALL TIMES"]
    r = check_warning(lines, bold_hint=True)
    assert r.present and r.heading_caps == Status.PASS
    assert r.wording == Status.REVIEW and "KEEP COLD" in r.wording_note   # the beside-text is shown, never silently dropped


def test_unreadable_first_heading_word_asks_for_review():
    assert check_heading_caps("\\Noee WARNING: (1) ACCORDING TO THE SURGEON")[0] == Status.REVIEW


def test_a_clearly_read_wrong_heading_word_fails():
    for heading in ("HEALTH WARNING: (1) According", "SURGEON WARNING: (1) According", "GOVT WARNING: (1) According"):
        status, note = check_heading_caps(heading)
        assert status == Status.FAIL and "must read 'GOVERNMENT WARNING:'" in note, (heading, note)
    # Garbled by OCR (non-letters in the word) or a near spelling: a look, not a failure.
    assert check_heading_caps("\\Noee WARNING: (1) According")[0] == Status.REVIEW
    assert check_heading_caps("GOVERNMNT WARNING: (1) According")[0] == Status.REVIEW


def test_a_heading_cut_at_the_image_edge_is_a_misread_not_a_wrong_word():
    # A real light-on-purple can: OCR lost "GOV" at the edge and read N as M.
    assert check_heading_caps("ERNMENT WARMING: (1) ACCORDING TO THE SURGEON")[0] == Status.REVIEW


def test_words_left_out_as_beside_text_are_always_quoted():
    words = MANDATED_WARNING.split()
    lines = [" ".join(words[0:10]), " ".join(words[10:20]), " ".join(words[20:30]),
             " ".join(words[30:]) + " ENJOY OUR BEER RESPONSIBLY WITH FRIENDS"]
    r = check_warning(lines, bold_hint=True)
    assert r.wording == Status.REVIEW and "ENJOY OUR BEER RESPONSIBLY WITH FRIENDS" in r.wording_note
    # With a reading difference elsewhere as well, the left-out words are still named.
    lines[1] = lines[1].replace("drink", "drlnk")
    r = check_warning(lines, bold_hint=True)
    assert r.wording != Status.PASS and "ENJOY OUR BEER RESPONSIBLY WITH FRIENDS" in r.wording_note


# --- layout: word boxes decide what belongs to the statement ----------------------------------
from app.readers.base import View  # noqa: E402
from app.warning import join_hyphenated, sentence_structure  # noqa: E402

S1 = "(1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects."
S2 = "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems."
STATEMENT_ROWS = [
    "GOVERNMENT WARNING: (1) According to the Surgeon General,",
    "women should not drink alcoholic beverages during pregnancy",
    "because of the risk of birth defects. (2) Consumption of",
    "alcoholic beverages impairs your ability to drive a car or",
    "operate machinery, and may cause health problems.",
]


def _typeset(rows, size: int = 22, heading_bold: bool = True, gap: float = 0.4):
    """Render rows of text segments the way a label prints them and return (ink, lines, words).

    ``rows`` is a list of rows; each row is a list of (text, x) segments drawn on that row, or a
    (text, x, view) segment. All words of a row share one OCR line, as Tesseract reads across columns.
    "GOVERNMENT" and "WARNING:" are set bold when heading_bold is True; everything else regular."""
    reg = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), size)
    bold = ImageFont.truetype(str(FONTS / "DejaVuSans-Bold.ttf"), size)
    img = Image.new("L", (size * 60, int(size * 1.6 * (len(rows) + 1))), 255)
    draw = ImageDraw.Draw(img)
    lines, words = [], []
    for row_no, segments in enumerate(rows):
        y = 10 + int(row_no * size * 1.6)
        texts = []
        for seg in segments:
            text, x = seg[0], seg[1]
            view = seg[2] if len(seg) > 2 else 0
            for token in text.split():
                font = bold if heading_bold and token in ("GOVERNMENT", "WARNING:") else reg
                draw.text((x, y), token, font=font, fill=0)
                l, t, r, b = draw.textbbox((x, y), token, font=font)
                words.append(OCRWord(text=token, left=l, top=t, width=r - l, height=b - t, conf=95,
                                     line_index=row_no, view=view))
                texts.append(token)
                x = r + int(size * gap)
        lines.append(" ".join(texts))
    return np.array(img) < 128, lines, words


def _right_edge(words, rows):
    return max(w.right for w in words if w.line_index in rows)


def test_neighbouring_column_is_set_aside_by_its_boxes_and_quoted():
    # A keg collar: the statement in one column, "ATTENTION-READ BEFORE TAPPING" in the next, a normal
    # word gap apart; Tesseract reads each row across both columns.
    _, _, probe = _typeset([[(r, 10)] for r in STATEMENT_ROWS])
    x_col = _right_edge(probe, range(5)) + 9
    other = ["ATTENTION-READ BEFORE TAPPING", "THIS KEG MAY RUPTURE", "FOR SALE ONLY IN OHIO", "", "KEEP COLD"]
    ink, lines, words = _typeset([[(r, 10), (o, x_col)] for r, o in zip(STATEMENT_ROWS, other)])
    r = check_warning(lines, words, ink)
    assert r.wording == Status.PASS, (r.wording, r.wording_note, r.diff)
    assert "ATTENTION-READ BEFORE TAPPING" in r.wording_note and "KEEP COLD" in r.wording_note
    assert r.heading_caps == Status.PASS and r.heading_bold == Status.PASS and r.overall == Status.PASS
    # The same rows without boxes: wording alone cannot place the words, so a look is still asked for.
    r = check_warning(lines, bold_hint=True)
    assert r.wording == Status.REVIEW and "KEEP COLD" in r.wording_note


def test_a_column_on_the_left_is_set_aside_too():
    ink, lines, words = _typeset([[("12 FL OZ" if i in (1, 3) else "", 10), (r, 200)] for i, r in enumerate(STATEMENT_ROWS)])
    r = check_warning(lines, words, ink)
    assert r.wording == Status.PASS and "FL OZ" in r.wording_note, (r.wording_note, r.diff)


def test_words_sticking_out_on_a_single_line_are_judged_as_wording():
    # Only one row carries words beyond the statement's extent: they may be words added to the statement.
    _, _, probe = _typeset([[(r, 10)] for r in STATEMENT_ROWS])
    x_col = _right_edge(probe, range(5)) + 9
    ink, lines, words = _typeset([[(r, 10)] + ([("SAFELY", x_col)] if i == 3 else []) for i, r in enumerate(STATEMENT_ROWS)])
    r = check_warning(lines, words, ink)
    assert r.wording != Status.PASS
    assert any("SAFELY" in d.found for d in r.diff) or "SAFELY" in r.wording_note


def test_a_stray_mark_beyond_the_column_is_dropped_silently():
    # A real can: "4" read at the left of one row, outside the statement's left edge.
    ink, lines, words = _typeset([[("4" if i == 2 else "", 10), (r, 60)] for i, r in enumerate(STATEMENT_ROWS)])
    r = check_warning(lines, words, ink)
    assert r.wording == Status.PASS and "left out" not in r.wording_note, (r.wording_note, r.diff)


def _reread(words, row, new_line, garble=None, conf=60):
    """Another OCR reading of ``row`` on the same boxes, as the sparse pass produces."""
    return [OCRWord(text=garble.get(w.text, w.text) if garble else w.text, left=w.left, top=w.top, width=w.width,
                    height=w.height, conf=conf, line_index=new_line) for w in words if w.line_index == row]


def test_two_readings_of_one_printed_line_are_never_both_taken():
    ink, lines, words = _typeset([[(r, 10)] for r in STATEMENT_ROWS])
    # The sparse pass read row 1 again with one word garbled; its words sit on the same boxes.
    second = _reread(words, 1, 5, {"drink": "DRlNK"})
    r = check_warning(lines + [" ".join(w.text for w in second)], words + second, ink)
    assert r.wording == Status.PASS and r.overall == Status.PASS, (r.wording_note, r.diff)
    # The order the lines come in does not matter: the garbled reading first, the clean one second.
    garbled_first = _reread(words, 1, 1, {"drink": "DRlNK"}) + _reread(words, 1, 5, None, 95)
    rest = [w for w in words if w.line_index != 1]
    lines3 = list(lines)
    lines3[1] = " ".join(w.text for w in garbled_first if w.line_index == 1)
    lines3.append(" ".join(w.text for w in garbled_first if w.line_index == 5))
    r = check_warning(lines3, rest + garbled_first, ink)
    assert r.wording == Status.PASS, (r.wording_note, r.diff)
    # An identical second reading does not make a sentence appear twice.
    same = _reread(words, 2, 5)
    r = check_warning(lines + [" ".join(w.text for w in same)], words + same, ink)
    assert r.wording == Status.PASS, (r.wording_note, r.diff)


def test_a_surer_reading_that_disagrees_asks_for_a_look():
    # A misread that happens to agree with the mandated text must not hide a misprint the other reading
    # saw. Here OCR was surer of "DRINX" than of "drink".
    ink, lines, words = _typeset([[(r, 10)] for r in STATEMENT_ROWS])
    second = _reread(words, 1, 5, {"drink": "DRINX"}, conf=99)
    r = check_warning(lines + [" ".join(w.text for w in second)], words + second, ink)
    assert r.wording == Status.REVIEW and "DRINX" in r.wording_note, (r.wording, r.wording_note)


def test_heading_split_over_two_lines_is_one_heading():
    lines = LABEL_LINES + ["GOVERNMENT", "WARNING: (1) According to the Surgeon General, women"] + WARNING_LINES[1:]
    r = check_warning(lines, bold_hint=True)
    assert r.heading_caps == Status.PASS and r.wording == Status.PASS and r.overall == Status.PASS, r.heading_caps_note
    lines = LABEL_LINES + ["Government", "Warning: (1) According to the Surgeon General, women"] + WARNING_LINES[1:]
    r = check_warning(lines, bold_hint=True)
    assert r.heading_caps == Status.FAIL and r.overall == Status.FAIL


@pytest.mark.parametrize("heading_bold", [True, False])
def test_split_heading_is_measured_over_both_words(heading_bold):
    rows = [[("GOVERNMENT", 10)], [("WARNING: " + STATEMENT_ROWS[0].split(": ")[1], 10)]] + [[(r, 10)] for r in STATEMENT_ROWS[1:]]
    ink, lines, words = _typeset(rows, heading_bold=heading_bold)
    r = check_warning(lines, words, ink)
    assert r.heading_caps == Status.PASS and r.wording == Status.PASS
    assert r.heading_bold == (Status.PASS if heading_bold else Status.REVIEW), (r.heading_bold_note, r.bold_ratio)


@pytest.mark.parametrize("heading_bold", [True, False])
def test_heading_on_the_same_line_as_the_body_is_measured(heading_bold):
    ink, lines, words = _typeset([[(r, 10)] for r in STATEMENT_ROWS], heading_bold=heading_bold)
    r = check_warning(lines, words, ink)
    assert r.heading_caps == Status.PASS and r.wording == Status.PASS
    assert r.heading_bold == (Status.PASS if heading_bold else Status.REVIEW), (r.heading_bold_note, r.bold_ratio)


def test_misread_heading_words_are_still_the_ones_measured():
    # Light text cut at the image edge: the heading reads "ERNMENT WARMING:"; those boxes are the heading's.
    ink, lines, words = _typeset([[(r, 10)] for r in STATEMENT_ROWS])
    for w in words:
        if w.text == "GOVERNMENT":
            w.text = "ERNMENT"
        elif w.text == "WARNING:":
            w.text = "WARMING:"
    lines[0] = " ".join(w.text for w in words if w.line_index == 0)
    r = check_warning(lines, words, ink)
    assert r.heading_caps == Status.REVIEW and r.heading_bold == Status.PASS, (r.heading_caps_note, r.heading_bold_note)


def test_heading_and_body_read_in_two_views_of_one_frame_make_one_statement():
    # The contrast pass reads the body differently from the upright pass, which keeps only the heading line.
    ink, lines, words = _typeset([[(STATEMENT_ROWS[0], 10)]] + [[(r, 10, 1)] for r in STATEMENT_ROWS[1:]])
    size = (ink.shape[1], ink.shape[0])
    views = [View(rot=0, inverted=False, ink=ink, size=size), View(rot=0, inverted=False, ink=ink, size=size)]
    r = check_warning(lines, words, ink, views=views)
    assert r.wording == Status.PASS and r.heading_caps == Status.PASS and r.heading_bold == Status.PASS, r.wording_note
    # A turned view is its own frame: its lines cannot join the upright heading.
    views[1] = View(rot=90, inverted=False, ink=ink, size=(size[1], size[0]))
    assert check_warning(lines, words, ink, views=views).overall == Status.FAIL


def test_a_sentence_printed_twice_is_a_wording_difference():
    s1, s2 = S1.split(), S2.split()
    first = ["GOVERNMENT WARNING: " + " ".join(s1[:9]), " ".join(s1[9:])]
    second = [" ".join(s2[:9]), " ".join(s2[9:])]
    r = check_warning(LABEL_LINES + first + second + ["Bottled by Old Tom"] + first, bold_hint=True)
    assert r.wording == Status.REVIEW and "sentence (1) is printed again" in r.wording_note, r.wording_note
    assert any("Surgeon" in d.found for d in r.diff)
    # Sentence (1) twice and sentence (2) never: a failure, never a pass.
    r = check_warning(LABEL_LINES + first + ["Bottled by Old Tom"] + first, bold_hint=True)
    assert r.wording == Status.FAIL and r.overall == Status.FAIL
    # The statement printed whole a second time (front and back label) is not a wording problem.
    r = check_warning(LABEL_LINES + first + second + ["Bottled by Old Tom"] + first + second, bold_hint=True)
    assert r.wording == Status.PASS, r.wording_note


def test_sentence_structure_inside_the_statement():
    assert sentence_structure(MANDATED_WARNING) == []
    twice = MANDATED_WARNING.replace("(2)", S1 + " (2)")
    assert check_wording(twice)[0] == Status.FAIL and "appears 2 times" in check_wording(twice)[2]
    swapped = "GOVERNMENT WARNING: " + S2 + " " + S1
    assert check_wording(swapped)[0] == Status.FAIL and "before sentence (1)" in check_wording(swapped)[2]


def test_hyphenated_word_over_a_line_break_is_rejoined():
    assert join_hyphenated(["ACCORDING TO THE SUR-", "GEON GENERAL, WOMEN"]) == ["ACCORDING TO THE SURGEON", "GENERAL, WOMEN"]
    assert join_hyphenated(["ALCOHOLIC BEVER-", "[AGES IMPAIRS"]) == ["ALCOHOLIC BEVERAGES", "IMPAIRS"]
    assert join_hyphenated(["SUR-", "GEOM GENERAL"]) == ["SUR-", "GEOM GENERAL"]   # not a word of the statement
    lines = ["GOVERNMENT WARNING: (1) According to the Sur-", "geon General, women should not drink alcoholic bever-",
             "ages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
             "beverages impairs your ability to drive a car or operate machinery, and may cause health problems."]
    assert check_warning(lines, bold_hint=True).wording == Status.PASS
    lines[0] = lines[0].replace("Sur-", "Sor-")
    assert check_warning(lines, bold_hint=True).wording != Status.PASS
