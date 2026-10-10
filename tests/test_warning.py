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


# --- meaning-changing differences: they fail; an OCR slip still asks for a look -------------------------
import dataclasses  # noqa: E402

from app.models import Application  # noqa: E402
from app.normalize import normalize_loose  # noqa: E402
from app.readers.base import LabelReading, OCRResult  # noqa: E402
from app.warning import meaning_changes  # noqa: E402


def _statement(old: str | None = None, new: str | None = None, rows=STATEMENT_ROWS, conf: dict | None = None):
    """The statement typeset with real word boxes (confidence 95), ``old`` replaced by ``new`` where it first
    occurs; ``conf`` sets the confidence of the words it names (loose spelling)."""
    rows = list(rows)
    if old is not None:
        k = next(i for i, r in enumerate(rows) if old in r)
        rows[k] = rows[k].replace(old, new, 1)
    ink, lines, words = _typeset([[(r, 10)] for r in rows])
    for w in words:
        if conf and normalize_loose(w.text) in conf:
            w.conf = conf[normalize_loose(w.text)]
    return ink, lines, words


MEANING = [
    ("should not drink", "should drink", "The label says 'should drink' where the warning requires 'should not drink'"),
    ("may cause", "can cause", "The label says 'can cause' where the warning requires 'may cause'"),
    ("should not", "must not", "The label says 'must not' where the warning requires 'should not'"),
    ("women", "men", "The label says 'men should' where the warning requires 'women should'"),
    ("women", "woman", "The label says 'woman should' where the warning requires 'women should'"),
    ("impairs", "improves", "The label says 'improves your' where the warning requires 'impairs your'"),
    ("may cause", "may never cause", "The label says 'may never cause' where the warning requires 'may cause'"),
    ("Surgeon General", "Attorney General", "The label says 'attorney general' where the warning requires 'surgeon general'"),
    ("health problems", "health benefits", "The label says 'health benefits' where the warning requires 'health problems'"),
]


@pytest.mark.parametrize("old,new,note", MEANING)
def test_a_meaning_changing_difference_fails_and_says_so(old, new, note):
    ink, lines, words = _statement(old, new)
    r = check_warning(lines, words, ink)
    assert r.wording == Status.FAIL and r.overall == Status.FAIL, (r.wording_note, r.diff)
    assert note + ": this changes its meaning." in r.wording_note


@pytest.mark.parametrize("old,new", [("women", "wornen"), ("may", "rnay"), ("not", "n0t"), ("drink", "drlnk"),
                                     ("impairs", "impair"), ("defects", "defect"), ("drive a car", "drive a can"),
                                     ("not drink", "no t drink"), ("should not", "should no")])
def test_an_ocr_garble_or_a_change_of_form_still_asks_for_a_look(old, new):
    # "wornen", "rnay", "n0t" are the required word once OCR's usual confusions are undone; "impair" and
    # "defect" break the fixed text without changing what it warns about; "can" is not a word that
    # replaces "car"; "no t" is "not" split; "no" for "not" is a letter lost, not another word.
    ink, lines, words = _statement(old, new)
    r = check_warning(lines, words, ink)
    assert r.wording == Status.REVIEW, (r.wording_note, r.diff)
    assert "changes its meaning" not in r.wording_note


def test_a_meaning_changing_word_read_without_confidence_asks_for_a_look():
    ink, lines, words = _statement("may cause", "can cause", conf={"can": THRESHOLDS.meaning_conf - 1})
    r = check_warning(lines, words, ink)
    assert r.wording == Status.REVIEW, r.wording_note
    ink, lines, words = _statement("may cause", "can cause", conf={"can": THRESHOLDS.meaning_conf})
    assert check_warning(lines, words, ink).wording == Status.FAIL
    # The threshold is a setting: raising it turns the same reading back into a look.
    strict = dataclasses.replace(THRESHOLDS, meaning_conf=99)
    assert check_warning(lines, words, ink, th=strict).wording == Status.REVIEW


def test_a_missing_not_fails_only_where_the_label_leaves_no_room_for_it():
    # Printed "should drink": the two words sit a normal word gap apart.
    ink, lines, words = _statement("should not drink", "should drink")
    assert check_warning(lines, words, ink).wording == Status.FAIL
    # Printed "should not drink" and OCR dropped "not": the gap would hold a word, so it is a misread.
    ink, lines, words = _statement()
    dropped = [w for w in words if w.text != "not"]
    lines = [" ".join(w.text for w in dropped if w.line_index == i) for i in range(len(lines))]
    r = check_warning(lines, dropped, ink)
    assert r.wording == Status.REVIEW and [(d.expected, d.found) for d in r.diff] == [("not", "")], r.wording_note
    # Either neighbour read with low confidence: a look.
    ink, lines, words = _statement("should not drink", "should drink", conf={"drink": 40})
    assert check_warning(lines, words, ink).wording == Status.REVIEW
    # The gap falls at a line break: the word may have been lost at the edge of the line.
    rows = ["GOVERNMENT WARNING: (1) According to the Surgeon General, women should",
            "drink alcoholic beverages during pregnancy"] + STATEMENT_ROWS[2:]
    ink, lines, words = _statement(rows=rows)
    assert check_warning(lines, words, ink).wording == Status.REVIEW


def _justified(text: str, last_word: str, short_by: int = 0, indent: int = 0) -> list[OCRWord]:
    """Word boxes for ``text`` set justified, seven words to a line spread over 0-800 px, with a line break
    after ``last_word``; that line ends ``short_by`` px short of the edge, the next starts ``indent`` px in."""
    toks = normalize_loose(text).split()
    cut = toks.index(last_word) + 1
    starts = list(range(cut % 7, cut, 7)) if cut % 7 else list(range(0, cut, 7))
    starts = ([0] if starts[0] else []) + starts + list(range(cut, len(toks), 7))
    out = []
    for line, (s0, s1) in enumerate(zip(starts, starts[1:] + [len(toks)])):
        chunk, step = toks[s0:s1], 800 / (s1 - s0)
        for k, t in enumerate(chunk):
            left, right = int(k * step), int((k + 1) * step) - 12
            if s0 + k == cut - 1:
                right -= short_by
            if s0 + k == cut:
                left += indent
            out.append(OCRWord(t, left, 30 * line, right - left, 20, 95, line))
    return out


def test_a_missing_not_at_a_line_break_fails_only_in_justified_type():
    text = MANDATED_WARNING.replace("should not drink", "should drink")
    # "should" ends its line at the right edge and "drink" starts the next at the left edge: no room for "not".
    assert check_wording(text, token_words=_justified(text, "should"))[0] == Status.FAIL
    # The first line ends short of the edge (ragged, or a word dropped at its end): a look.
    assert check_wording(text, token_words=_justified(text, "should", short_by=60))[0] == Status.REVIEW
    # The next line starts indented (a word dropped at its start): a look.
    assert check_wording(text, token_words=_justified(text, "should", indent=60))[0] == Status.REVIEW


def test_an_inserted_word_counts_only_inside_a_line_of_the_statement():
    ink, lines, words = _statement("may cause", "may never cause")
    assert check_warning(lines, words, ink).wording == Status.FAIL
    # At the end of a line it may belong to text beside the statement ("FOR SALE ONLY IN OHIO").
    ink, lines, words = _statement("drive a car or", "drive a car or only")
    r = check_warning(lines, words, ink)
    assert r.wording == Status.REVIEW, r.wording_note


def test_without_word_boxes_a_meaning_change_asks_for_a_look():
    # The cloud reader gives no confidences: the change is classified but cannot be told from a misread.
    text = MANDATED_WARNING.replace("should not drink", "should drink")
    assert check_wording(text)[0] == Status.REVIEW
    assert [c.word for c in meaning_changes(normalize_loose(text).split())] == ["not"]
    lines = LABEL_LINES + [r.replace("should not drink", "should drink") for r in WARNING_LINES]
    assert check_warning(lines, bold_hint=True).wording == Status.REVIEW


def _sure(text: str) -> list[OCRWord]:
    """One confidently read word per token, a normal word gap apart on one line."""
    return [OCRWord(t, 60 * i, 0, 50, 20, 95, 0) for i, t in enumerate(normalize_loose(text).split())]


def test_words_are_judged_for_meaning_only_when_the_statement_was_read_closely():
    text = MANDATED_WARNING.replace("may cause", "can cause")
    status, _, note, _ = check_wording(text, token_words=_sure(text))
    assert status == Status.FAIL and "'can cause'" in note
    # Most of the statement missing: it fails on similarity, and one word in what is left is not judged.
    short = text.split(" (1)")[0] + " (2) Consumption of alcoholic beverages may cause health problems, and can cause"
    status, score, note, _ = check_wording(short, token_words=_sure(short))
    assert status == Status.FAIL and score < THRESHOLDS.meaning_min_score and "changes its meaning" not in note


def test_the_question_for_a_meaning_change_names_the_word_at_stake():
    from app.decisions import prompts_for
    from app.pipeline import verify

    ink, lines, words = _statement("should not drink", "should drink")

    class Typeset:
        name = "typeset"

        def read(self, image):
            return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink))
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Bourbon", alcohol_content="45%", net_contents="750 mL")
    r = verify(app, Image.new("RGB", (ink.shape[1], ink.shape[0]), "white"), Typeset())
    assert r.warning.wording == Status.FAIL
    p = next(p for p in prompts_for(r) if p.key == "warning_wording")
    assert p.question == 'Does the label print "not" in "should not drink"?'
    assert p.yes.startswith("Yes") and p.no.startswith("No") and p.lean is False   # Yes = the label is fine
    assert p.left == ("should ", "not", " drink") and p.right[1] == "(nothing)"
    assert "changes its meaning" in p.why
