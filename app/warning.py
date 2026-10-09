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
from rapidfuzz.distance import Levenshtein

from .config import MANDATED_WARNING, THRESHOLDS, Thresholds
from .models import DiffItem, Status, WarningResult
from .normalize import normalize_loose, normalize_strict
from .readers.base import OCRWord, upright_box

_MANDATED_LOOSE = normalize_loose(MANDATED_WARNING)
_MANDATED_WORDS = _MANDATED_LOOSE.split()
_DESCENDER_CHARS = set("gjpqy,;()[]{}")
_HEIGHT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt0123456789")


# --- locating the statement -------------------------------------------------------------------
@dataclass(frozen=True)
class WarningSpan:
    line_ids: tuple[int, ...]         # lines that make up the statement, in reading order
    heading_line: int | None          # line holding "GOVERNMENT WARNING", if recognised
    texts: tuple[str, ...] = ()       # those lines as read, minus text that sits beside the statement
    beside: str = ""                  # words left out because they belong to a neighbouring column

    @property
    def start(self) -> int:
        return self.line_ids[0]

    @property
    def end(self) -> int:
        return self.line_ids[-1]


def _contains_score(query: str, line: str) -> float:
    """How well ``query`` appears inside ``line``. Short OCR fragments ("a", "—") must not score high."""
    if len(line) < 0.8 * len(query):
        return fuzz.ratio(query, line)
    return fuzz.partial_ratio(query, line)


_MAX_SKIPS = 4        # lines in a row that may be passed over (another column, a duplicate read)
_MAX_LINES_AHEAD = 30


_MANDATED_VOCAB = set(_MANDATED_WORDS)


def _statement_share(line: str) -> float:
    """Share of a line's real words (3+ letters) that occur in the mandated statement."""
    toks = [t for t in line.split() if sum(ch.isalpha() for ch in t) >= 3]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _MANDATED_VOCAB) / len(toks)


def _assemble(order: list[int], loose: list[str], start_pos: int) -> tuple[list[int], float]:
    """Grow the statement from ``order[start_pos]``, taking each following line that is made mostly of
    the statement's own words and brings the text closer to it, and passing over lines that do not
    ("For Sale Only In Ohio" printed between two lines of a sideways warning)."""
    ids = [order[start_pos]]
    acc = loose[order[start_pos]]
    best = fuzz.ratio(_MANDATED_LOOSE, acc)
    skips = 0
    for pos in range(start_pos + 1, min(len(order), start_pos + _MAX_LINES_AHEAD)):
        cand = (acc + " " + loose[order[pos]]).strip()
        r = fuzz.ratio(_MANDATED_LOOSE, cand) if _statement_share(loose[order[pos]]) >= 0.5 else -1
        if r > best:
            ids.append(order[pos])
            acc, best, skips = cand, r, 0
        else:
            skips += 1
            if skips > _MAX_SKIPS:
                break
    return ids, best


def _trim_beside(texts: list[str]) -> tuple[list[str], str]:
    """Drop words at the start or end of a line that belong to a neighbouring column (a keg collar's
    "ATTENTION-READ BEFORE TAPPING" printed beside the warning). Returns the trimmed lines and the
    real words that were left out (short OCR noise such as "7" or "—c" is dropped silently)."""
    texts = list(texts)
    dropped: list[str] = []

    def score(ts):
        return fuzz.ratio(_MANDATED_LOOSE, normalize_loose(" ".join(ts)))

    for i, line in enumerate(texts):
        toks = line.split()
        if len(toks) < 2:
            continue
        best, best_cut = score(texts), (0, len(toks))
        for a in range(0, min(len(toks), 8)):
            for b in range(len(toks), max(a, len(toks) - 12), -1):
                if (a, b) == (0, len(toks)) or b <= a:
                    continue
                trial = texts[:i] + [" ".join(toks[a:b])] + texts[i + 1:]
                r = score(trial)
                if r > best + 0.5:
                    best, best_cut = r, (a, b)
        a, b = best_cut
        if (a, b) != (0, len(toks)):
            cut = toks[:a] + toks[b:]
            dropped += [t for t in cut if sum(ch.isalpha() for ch in t) >= 3]
            texts[i] = " ".join(toks[a:b])
    return texts, " ".join(dropped)


def locate_warning(lines: list[str], *, line_views: list[int] | None = None, line_tops: list[float] | None = None,
                   th: Thresholds = THRESHOLDS) -> WarningSpan | None:
    """Find the statement. Each view the reader produced (upright, turned, contrast) is tried on its
    own, its lines ordered top to bottom, and the reading that comes closest to the mandated text wins."""
    if not lines:
        return None
    loose = [normalize_loose(l) for l in lines]
    views = line_views or [0] * len(lines)
    tops = line_tops or [float(i) for i in range(len(lines))]
    best: tuple[float, WarningSpan] | None = None
    for view in sorted(set(views)):
        order = sorted((i for i in range(len(lines)) if views[i] == view and loose[i]), key=lambda i: (tops[i], i))
        heads = [(pos, _contains_score("government warning", loose[i])) for pos, i in enumerate(order)]
        heads = [(pos, sc) for pos, sc in heads if sc >= th.warning_locate]
        starts = [pos for pos, _ in sorted(heads, key=lambda h: -h[1])[:3]]
        if not starts:   # heading unreadable or missing: start from the body of the statement
            starts = [pos for pos, i in enumerate(order)
                      if _contains_score("according to the surgeon general", loose[i]) >= th.warning_locate][:2]
        for pos in starts:
            ids, ratio = _assemble(order, loose, pos)
            heading = order[pos] if any(h[0] == pos for h in heads) else None
            if best is None or ratio > best[0]:
                best = (ratio, WarningSpan(line_ids=tuple(ids), heading_line=heading,
                                           texts=tuple(lines[i] for i in ids)))
    if best is None:
        return None
    span = best[1]
    texts, beside = _trim_beside(list(span.texts))
    return WarningSpan(line_ids=span.line_ids, heading_line=span.heading_line, texts=tuple(texts), beside=beside)


# --- wording ----------------------------------------------------------------------------------
_MANDATED_RAW_WORDS = MANDATED_WARNING.split()   # same count as _MANDATED_WORDS (no punctuation-only tokens)


def word_diff(found_text: str, context: int = 5) -> list[DiffItem]:
    """Word-level differences from the mandated text, each with a few words of context on both sides."""
    raw_found = [t for t in found_text.split() if normalize_loose(t)]
    got = normalize_loose(found_text).split()
    show_found = raw_found if len(raw_found) == len(got) else got
    show_exp = _MANDATED_RAW_WORDS if len(_MANDATED_RAW_WORDS) == len(_MANDATED_WORDS) else _MANDATED_WORDS
    sm = difflib.SequenceMatcher(a=_MANDATED_WORDS, b=got, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append(DiffItem(
            expected=" ".join(show_exp[i1:i2]), found=" ".join(show_found[j1:j2]),
            expected_before=" ".join(show_exp[max(0, i1 - context):i1]), expected_after=" ".join(show_exp[i2:i2 + context]),
            found_before=" ".join(show_found[max(0, j1 - context):j1]), found_after=" ".join(show_found[j2:j2 + context])))
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
    """The two words that read most like 'GOVERNMENT WARNING' when OCR garbled a letter or two ("WARNlNG",
    "G0VERNMENT"), or cut at the edge of the image ("ERNMENT WARMING"). Each word must be a near spelling
    or a long fragment of its target: "GOVT WARNING" is an abbreviation printed on the label, not a
    misread, and must fail."""
    tokens = text.split()
    best, best_score = None, 0.0
    for i in range(len(tokens) - 1):
        gov, warn = tokens[i], tokens[i + 1].rstrip(":;.")
        g, w = gov.lower().translate(_LOOKALIKE_LETTERS), warn.lower().translate(_LOOKALIKE_LETTERS)
        near_gov = Levenshtein.distance(g, "government") <= 2 or (len(g) >= 6 and g in "government")
        if not near_gov or Levenshtein.distance(w, "warning") > 2:
            continue
        score = fuzz.ratio("government warning", f"{g} {w}")
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
            m2 = re.match(r"\s*(\S{2,14})\s+(WARNING)\s*:", text)
            if m2 and not m2.group(1).isalpha():
                # "\Noee WARNING:": the first word is there but garbled, "WARNING" is in capitals
                return Status.REVIEW, ("The first word of the heading was not read clearly; 'WARNING' is in capitals. "
                                       "Please check the heading by eye.")
            if m2:   # a clean word that is not GOVERNMENT: "HEALTH WARNING:", "GOVT WARNING:"
                return Status.FAIL, (f"The heading reads '{m2.group(1)} WARNING:'. It must read 'GOVERNMENT WARNING:'.")
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
                  *, bold_hint: bool | None = None, views=None, th: Thresholds = THRESHOLDS) -> WarningResult:
    line_views: list[int] = [0] * len(lines)
    line_tops: list[float] = [float(i) for i in range(len(lines))]
    if words:
        by_line: dict[int, list[OCRWord]] = {}
        for w in words:
            if 0 <= w.line_index < len(lines):
                by_line.setdefault(w.line_index, []).append(w)
        for i, ws in by_line.items():
            line_views[i] = ws[0].view
            line_tops[i] = sum(w.top for w in ws) / len(ws)
    span = locate_warning(lines, line_views=line_views, line_tops=line_tops if words else None, th=th)
    if span is None:
        return WarningResult(present=False, wording=Status.FAIL, wording_note="No government warning statement was found on the label.",
                             heading_caps=Status.FAIL, heading_caps_note="Not found.",
                             heading_bold=Status.FAIL, heading_bold_note="Not found.", overall=Status.FAIL)
    found_text = "\n".join(span.texts) if span.texts else "\n".join(lines[i] for i in span.line_ids)
    wording, score, wording_note, diff = check_wording(found_text, th=th)
    if span.beside:
        # Words set aside as a neighbouring column are always quoted to the agent: they may instead be
        # words added to the statement, which the regulation does not allow.
        aside = (f"text printed beside the statement was left out ('{span.beside}'). "
                 "Please check it is not part of the warning.")
        if wording == Status.PASS:
            wording, wording_note = Status.REVIEW, "Wording matches, but " + aside
        else:
            wording_note += " Also, " + aside
    heading_text = lines[span.heading_line] if span.heading_line is not None else lines[span.line_ids[0]]
    caps, caps_note = check_heading_caps(heading_text)
    in_span = set(span.line_ids)

    if bold_hint is not None:
        if bold_hint:
            bold, ratio, bold_note = Status.PASS, None, "Heading reported as bold by the vision model."
        else:
            bold = Status.FAIL if th.bold_failure_is_fail else Status.REVIEW
            ratio, bold_note = None, "Heading reported as NOT bold by the vision model. Please check it by eye."
    elif words is not None and ink is not None and span.heading_line is not None:
        heading_words = [w for w in words if w.line_index == span.heading_line
                         and normalize_loose(w.text) in _HEADING_TOKENS]
        # Measure in the view the heading was read in (a sideways warning is measured turned upright).
        view = heading_words[0].view if heading_words else 0
        view_ink = views[view].ink if views and view < len(views) else ink
        body_words = [w for w in words if w.line_index in in_span and w.view == view
                      and w not in heading_words and len(normalize_loose(w.text)) >= 3]
        bold, ratio, bold_note = estimate_heading_bold(view_ink, heading_words, body_words, th=th)
    else:
        bold, ratio, bold_note = Status.REVIEW, None, "Could not measure the heading's weight. Please check it by eye."

    statuses = [wording, caps, bold]
    if Status.FAIL in statuses:
        overall = Status.FAIL
    elif Status.REVIEW in statuses:
        overall = Status.REVIEW
    else:
        overall = Status.PASS
    box = None
    if words is not None and ink is not None:
        span_words = [w for w in words if w.line_index in in_span]
        if span_words:
            h, w_ = ink.shape
            boxes = [upright_box(w, views or []) for w in span_words]
            left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
            right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
            box = [round(100 * left / w_, 2), round(100 * top / h, 2),
                   round(100 * (right - left) / w_, 2), round(100 * (bottom - top) / h, 2)]
    return WarningResult(present=True, found_text=found_text, wording=wording, wording_score=score,
                         wording_note=wording_note, diff=diff, heading_caps=caps, heading_caps_note=caps_note,
                         heading_bold=bold, heading_bold_note=bold_note, bold_ratio=ratio, overall=overall, box=box)
