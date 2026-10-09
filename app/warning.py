"""Government warning statement checks (27 CFR 16.21 / 16.22).

Four things are checked, each reported separately so an agent can see exactly what
is wrong:
  1. present      - a warning statement exists on the label
  2. wording      - the words match the mandated text exactly (punctuation ignored,
                    because OCR drops commas and periods unreliably)
  3. heading caps - "GOVERNMENT WARNING:" is in capital letters, as printed
  4. heading bold - the heading is printed bolder than the body text. This is a
                    heuristic (stroke-width comparison on the image); see README.
"""

from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from .config import MANDATED_WARNING, THRESHOLDS, Thresholds
from .models import DiffItem, Status, WarningResult
from .normalize import normalize_loose, normalize_strict
from .readers.base import OCRWord

_MANDATED_LOOSE = normalize_loose(MANDATED_WARNING)
_MANDATED_WORDS = _MANDATED_LOOSE.split()
_DESCENDER_CHARS = set("gjpqy,;()[]{}")
_HEIGHT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt0123456789")


# --- locating the statement -------------------------------------------------------------------
@dataclass(frozen=True)
class WarningSpan:
    start: int           # first line index
    end: int             # last line index (inclusive)
    heading_line: int | None  # line holding "GOVERNMENT WARNING", if recognised


def _contains_score(query: str, line: str) -> float:
    """How well ``query`` appears inside ``line``. Short OCR fragments ("a", "—") must not score high."""
    if len(line) < 0.8 * len(query):
        return fuzz.ratio(query, line)
    return fuzz.partial_ratio(query, line)


def locate_warning(lines: list[str], *, th: Thresholds = THRESHOLDS) -> WarningSpan | None:
    if not lines:
        return None
    loose = [normalize_loose(l) for l in lines]
    heading_line, best = None, 0
    for i, l in enumerate(loose):
        if not l:
            continue
        s = _contains_score("government warning", l)
        if s > best and s >= th.warning_locate:
            heading_line, best = i, s
    start = heading_line
    if start is None:  # heading unreadable or missing: look for the body of the statement
        for i, l in enumerate(loose):
            if _contains_score("according to the surgeon general", l) >= th.warning_locate:
                start = i
                break
    if start is None:
        return None
    # Grow the span line by line while the similarity to the mandated text keeps improving.
    acc = loose[start]
    best_ratio = fuzz.ratio(_MANDATED_LOOSE, acc)
    end = start
    for j in range(start + 1, min(len(lines), start + 15)):
        candidate = (acc + " " + loose[j]).strip()
        r = fuzz.ratio(_MANDATED_LOOSE, candidate)
        if r >= best_ratio:
            acc, best_ratio, end = candidate, r, j
        else:
            break
    return WarningSpan(start=start, end=end, heading_line=heading_line)


# --- wording ----------------------------------------------------------------------------------
def word_diff(found_text: str) -> list[DiffItem]:
    got = normalize_loose(found_text).split()
    sm = difflib.SequenceMatcher(a=_MANDATED_WORDS, b=got, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            out.append(DiffItem(expected=" ".join(_MANDATED_WORDS[i1:i2]), found=" ".join(got[j1:j2])))
    return out


def check_wording(found_text: str, *, th: Thresholds = THRESHOLDS) -> tuple[Status, int, str, list[DiffItem]]:
    got = normalize_loose(found_text)
    if got == _MANDATED_LOOSE:
        return Status.PASS, 100, "Wording matches the required statement word for word.", []
    score = int(round(fuzz.ratio(_MANDATED_LOOSE, got)))
    diff = word_diff(found_text)
    n = len(diff)
    if score >= th.warning_near:
        return (Status.REVIEW, score,
                f"{n} difference{'s' if n != 1 else ''} from the required wording. "
                "This may be a misprint on the label or a reading error. Please check the label.", diff)
    return Status.FAIL, score, "The wording does not match the required statement.", diff


# --- heading capitalization -------------------------------------------------------------------
_HEADING_RE = re.compile(r"(government)\s+(warning)(\s*:)?", re.IGNORECASE)


_LOOKALIKE_LETTERS = str.maketrans({"0": "o", "1": "i", "|": "i", "l": "i"})  # neither heading word has an "l"


def _misread_heading(text: str) -> tuple[str, str, bool] | None:
    """The two words that read most like 'GOVERNMENT WARNING' when OCR garbled a letter ("WARNlNG")."""
    tokens = text.split()
    best, best_score = None, 0.0
    for i in range(len(tokens) - 1):
        gov, warn = tokens[i], tokens[i + 1].rstrip(":;.")
        score = fuzz.ratio("government warning", f"{gov} {warn}".lower().translate(_LOOKALIKE_LETTERS))
        if score > best_score:
            colon = tokens[i + 1].endswith(":") or (i + 2 < len(tokens) and tokens[i + 2].startswith(":"))
            best, best_score = (gov, warn, colon), score
    return best if best_score >= 80 else None


def check_heading_caps(heading_line_text: str | None) -> tuple[Status, str]:
    if not heading_line_text:
        return Status.FAIL, "The words 'GOVERNMENT WARNING' were not found at the start of the statement."
    text = normalize_strict(heading_line_text)
    m = _HEADING_RE.search(text)
    if m:
        gov, warn, colon = m.group(1), m.group(2), bool(m.group(3))
    else:
        misread = _misread_heading(text)
        if misread is None:
            return Status.FAIL, "The words 'GOVERNMENT WARNING' were not found at the start of the statement."
        gov, warn, colon = misread
        lower = {ch for ch in gov + warn if ch.islower()}
        if lower <= {"l", "i", "o"}:  # capitals with a letter or two misread by OCR
            return Status.REVIEW, (f"The heading reads '{gov} {warn}': it looks like capitals, but a letter was not read "
                                   "clearly. Please check it by eye.")
    if not (gov.isupper() and warn.isupper()):
        return Status.FAIL, f"The heading is printed as '{gov} {warn}'. It must be all capitals: 'GOVERNMENT WARNING:'."
    if not colon:
        return Status.REVIEW, "Heading is in capitals, but the colon after 'WARNING' was not detected."
    return Status.PASS, "Heading is in capitals."


# --- heading boldness (heuristic) --------------------------------------------------------------
def _ink_area_and_boundary(ink: np.ndarray, word: OCRWord) -> tuple[int, int] | None:
    """Ink pixel count and boundary pixel count inside a word box (boundary = ink with a background 4-neighbour).

    The ink mask is global, so on a light-on-dark panel of an otherwise light label it marks the panel,
    not the letters. The ring of pixels just outside the word box is background either way: when it is
    mostly "ink", the polarity is flipped for this word.
    """
    h, w = ink.shape
    x0, y0 = max(0, word.left - 1), max(0, word.top - 1)
    x1, y1 = min(w, word.right + 1), min(h, word.bottom + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    rx0, ry0, rx1, ry1 = max(0, x0 - 3), max(0, y0 - 3), min(w, x1 + 3), min(h, y1 + 3)
    ring = ink[ry0:ry1, rx0:rx1].copy()
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    ring_px = (ry1 - ry0) * (rx1 - rx0) - (y1 - y0) * (x1 - x0)
    crop = ink[y0:y1, x0:x1]
    if ring_px and ring.sum() > 0.5 * ring_px:
        crop = ~crop
    crop = np.pad(crop, 1)
    inner = crop[1:-1, 1:-1]
    bg_neighbour = (~crop[:-2, 1:-1]) | (~crop[2:, 1:-1]) | (~crop[1:-1, :-2]) | (~crop[1:-1, 2:])
    area = int(inner.sum())
    boundary = int((inner & bg_neighbour).sum())
    if area < 20 or boundary == 0:
        return None
    return area, boundary


def stroke_width(ink: np.ndarray, words: list[OCRWord]) -> float | None:
    """Mean stroke width over a set of words: 2 x ink area / ink perimeter.

    For a stroke of width w and length L the area is w*L and the perimeter about 2L, so the
    estimate is orientation independent (diagonals and curves do not inflate it, unlike run
    lengths) and insensitive to letter case. Aggregating over all words makes it stable.
    """
    area = boundary = 0
    for w in words:
        ab = _ink_area_and_boundary(ink, w)
        if ab:
            area += ab[0]
            boundary += ab[1]
    if boundary == 0:
        return None
    return 2.0 * area / boundary


def _reference_height(words: list[OCRWord]) -> float | None:
    """Median height of words whose letters reach cap height but have no descenders."""
    hs = [w.height for w in words
          if w.text and not (set(w.text) & _DESCENDER_CHARS) and (set(w.text) & _HEIGHT_CHARS)]
    if not hs:
        hs = [w.height for w in words if w.text]
    return float(statistics.median(hs)) if hs else None


def estimate_heading_bold(ink: np.ndarray, heading_words: list[OCRWord], body_words: list[OCRWord],
                          *, th: Thresholds = THRESHOLDS) -> tuple[Status, float | None, str]:
    """Compare stroke width of the heading with the body text of the statement.

    Returns (status, ratio, note). ratio > 1 means the heading strokes are thicker than the
    body's after normalizing for text size (cap height of words without descenders). This is a
    heuristic: it assumes the body is set in regular weight at a similar size, which is how the
    statement is printed in practice.
    """
    head_sw = stroke_width(ink, heading_words)
    body_sw = stroke_width(ink, body_words)
    head_h, body_h = _reference_height(heading_words), _reference_height(body_words)
    if not head_sw or not body_sw or len(body_words) < 3 or not head_h or not body_h:
        return Status.REVIEW, None, "Could not measure the heading's weight. Please check it by eye."
    if min(head_h, body_h) < th.bold_min_text_px:
        return Status.REVIEW, None, "The statement is printed too small to measure its weight. Please check it by eye."
    ratio = round((head_sw / head_h) / (body_sw / body_h), 2)
    if ratio >= th.bold_ratio:
        return Status.PASS, ratio, f"Heading looks bold (strokes {ratio:.2f}x thicker than the text)."
    status = Status.FAIL if th.bold_failure_is_fail else Status.REVIEW
    return status, ratio, (f"Heading does not look bolder than the rest of the statement (strokes {ratio:.2f}x). "
                           "Please check by eye: the heading must be bold and the rest of the statement must not be.")


# --- putting it together -----------------------------------------------------------------------
_HEADING_TOKENS = ("government", "warning")


def check_warning(lines: list[str], words: list[OCRWord] | None = None, ink: np.ndarray | None = None,
                  *, bold_hint: bool | None = None, th: Thresholds = THRESHOLDS) -> WarningResult:
    span = locate_warning(lines, th=th)
    if span is None:
        return WarningResult(present=False, wording=Status.FAIL, wording_note="No government warning statement was found on the label.",
                             heading_caps=Status.FAIL, heading_caps_note="Not found.",
                             heading_bold=Status.FAIL, heading_bold_note="Not found.", overall=Status.FAIL)
    found_text = "\n".join(lines[span.start: span.end + 1])
    wording, score, wording_note, diff = check_wording(found_text, th=th)
    heading_text = lines[span.heading_line] if span.heading_line is not None else None
    caps, caps_note = check_heading_caps(heading_text)

    if bold_hint is not None:
        if bold_hint:
            bold, ratio, bold_note = Status.PASS, None, "Heading reported as bold by the vision model."
        else:
            bold = Status.FAIL if th.bold_failure_is_fail else Status.REVIEW
            ratio, bold_note = None, "Heading reported as NOT bold by the vision model. Please check it by eye."
    elif words is not None and ink is not None and span.heading_line is not None:
        heading_words = [w for w in words if w.line_index == span.heading_line
                         and normalize_loose(w.text) in _HEADING_TOKENS]
        body_words = [w for w in words if span.start <= w.line_index <= span.end
                      and w not in heading_words and len(normalize_loose(w.text)) >= 3]
        bold, ratio, bold_note = estimate_heading_bold(ink, heading_words, body_words, th=th)
    else:
        bold, ratio, bold_note = Status.REVIEW, None, "Could not measure the heading's weight. Please check it by eye."

    statuses = [wording, caps, bold]
    if Status.FAIL in statuses:
        overall = Status.FAIL
    elif Status.REVIEW in statuses:
        overall = Status.REVIEW
    else:
        overall = Status.PASS
    return WarningResult(present=True, found_text=found_text, wording=wording, wording_score=score,
                         wording_note=wording_note, diff=diff, heading_caps=caps, heading_caps_note=caps_note,
                         heading_bold=bold, heading_bold_note=bold_note, bold_ratio=ratio, overall=overall)
