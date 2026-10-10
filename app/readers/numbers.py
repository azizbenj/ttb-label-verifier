"""A second look at the figures: alcohol content and net contents lines re-read from their own crop.

A whole-page pass reads a figure as one small thing among everything else on the label, and now and
then drops a decimal point or a digit ("1.5 L" read as "15L" or "LSL", "57.7%" as "07.7%"). A wrong
number is the one error a compliance check cannot afford, so when the pipeline finds the alcohol
content or the net contents missing, different or doubtful, the lines that carry a figure are cut out
of the page, scaled to the line height Tesseract reads best and read again on their own in its
single-line mode. Each second read is appended to the OCR result as a line of its own, in a view that
maps its words back onto the crop's place on the page, so the matching rules (app/matching.py) see it
as one more reading of that place; they decide what a reading that corrects the first one is worth.
"""

from __future__ import annotations

import re
from time import perf_counter
from typing import Callable

from PIL import Image

from ..config import TESSERACT_EXTRA_TIMEOUT_S, THRESHOLDS, Thresholds
from ..normalize import alcohol_candidates, normalize_loose, volume_candidates
from .base import LabelReading, OCRResult, OCRWord, ReaderError, Reread, View, text_height, upright_box

KINDS = ("alcohol", "volume")

# A short run of digits mixed with the letters OCR trades them for ("L751" for "1.75 L"): a figure whose
# unit or decimal point was lost may hide in it. A run of digits alone (a zip code, a year) is not one.
_FIGURE_LIKE_RE = re.compile(r"(?<![A-Za-z0-9])(?=[0-9LlIOoSsB.,]{2,6}(?![A-Za-z0-9]))"
                             r"(?=[0-9.,]*[LlIOoSsB])[0-9LlIOoSsB.,]*[0-9][0-9LlIOoSsB.,]*")
_VOLUME_LIKE_RE = re.compile(_FIGURE_LIKE_RE.pattern + r"(?![0-9LlIOoSsB.,])(?!\s*%)")   # a percentage is not a volume
# Letters alone in the shape of a volume ("LSL", "LSOL" for "1.5 L", "SOML" for "50ML"): the page pass never makes
# a figure of these (app/normalize.py), the crop reads them with their digits. Not a state code
# before a ZIP ("IL 60607").
_LETTERS_LIKE_LITRES_RE = re.compile(r"(?<![A-Za-z0-9])[LlIOoSsB][LlIOoSsB.,]{0,3}(?:L|\s?[mM][lL])(?![A-Za-z0-9])(?!\s*\d{5}\b)")
# Words that mark a figure as an alcohol content or a volume, with OCR's usual slips ("ALG", "V0L").
_ALC_HINT_RE = re.compile(r"%|\bproof\b|\ba[l1i][cg]\b|\bv[o0g][l1i]\b|\babv\b", re.IGNORECASE)
_VOL_HINT_RE = re.compile(r"\b(?:m\s?l|cl|l|oz|fl|liters?|litres?|pints?|quarts?|gal|gallons?)\b", re.IGNORECASE)

# Tesseract's single-text-line mode. The size of the crop matters far more than the mode: measured over
# the batch set, the stress renderings and the real labels (every figure line read at line heights of
# 20 to 44 px, in modes 7 and 13), mode 7 at 20-25 px read the volume the page pass had missed on 338
# of 353 lines with 6 wrong figures, where the page pass itself managed 322 with 31; at 44 px, or in
# the raw-line mode 13, wrong figures multiplied ("750 mL" as "790 mL", "1.75 L" as "1L75L").
PSM = 7
MIN_SCALE, MAX_SCALE = 0.25, 4.0


def line_kinds(line: str) -> dict[str, bool]:
    """Which figures a line may carry: {"alcohol": True} when it holds an alcohol statement outright,
    False when it only looks as if it might (a figure-like token next to "%" or a unit word)."""
    out: dict[str, bool] = {}
    figure_like = bool(_FIGURE_LIKE_RE.search(line))
    if alcohol_candidates(line):
        out["alcohol"] = True
    elif figure_like and _ALC_HINT_RE.search(line):
        out["alcohol"] = False
    if volume_candidates(line):
        out["volume"] = True
    elif _VOLUME_LIKE_RE.search(line) or _LETTERS_LIKE_LITRES_RE.search(line):
        out["volume"] = False
    return out


def lines_to_reread(ocr: OCRResult, kinds: set[str], limit: int) -> list[tuple[int, tuple[str, ...]]]:
    """The lines worth a second read for the given figure kinds: those that carry such a figure
    outright first, then those that only look as if they might, at most ``limit`` in all. Lines
    already re-read, and the second reads themselves, are left out."""
    done = {r.line for r in ocr.rereads} | {r.new_line for r in ocr.rereads if r.new_line is not None}
    sure: list[tuple[int, tuple[str, ...]]] = []
    maybe: list[tuple[int, tuple[str, ...]]] = []
    for i, line in enumerate(ocr.lines):
        if i in done:
            continue
        lk = line_kinds(line)
        wanted = tuple(k for k in KINDS if k in kinds and k in lk)
        if not wanted:
            continue
        (sure if any(lk[k] for k in wanted) else maybe).append((i, wanted))
    return (sure + maybe)[:limit]


def line_crop(ocr: OCRResult, index: int, text_px: int, pad_ratio: float = 0.5) -> tuple[Image.Image, View] | None:
    """Cut one OCR line out of the preprocessed page with some paper around it, turned the way its
    view was read (so the text is level) and scaled so the line is about ``text_px`` tall. Returns
    the crop and the View that maps words read on it back onto the page."""
    words = [w for w in ocr.words if w.line_index == index]
    if not words or ocr.source is None:
        return None
    page: Image.Image = ocr.source[0]
    view = ocr.views[words[0].view] if 0 <= words[0].view < len(ocr.views) else None
    rot = view.rot if view is not None else 0
    boxes = [upright_box(w, ocr.views) for w in words]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    tallest = max(text_height(w, ocr.views) for w in words)
    pad = max(4, round(tallest * pad_ratio))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(page.width, right + pad), min(page.height, bottom + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = page.crop((x0, y0, x1, y1))
    if rot:
        crop = crop.rotate(rot, expand=True)
    scale = min(MAX_SCALE, max(MIN_SCALE, text_px / max(1.0, tallest)))
    big = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    from .tesseract import ink_mask   # here, not at the top: tesseract.py imports this module
    return big, View(rot=rot, inverted=False, ink=ink_mask(big), size=big.size, scale=scale, offset=(x0, y0))


def record(ocr: OCRResult, line: int, kinds: tuple[str, ...], mode: str,
           lines2: list[str], words2: list[OCRWord], view: View) -> Reread:
    """Keep one second read: as a new line in a view of its own when it says something new, and
    always as a Reread, so the matcher knows what a closer look at that line said."""
    text = " ".join(l for l in lines2 if l.strip())
    conf = sum(w.conf for w in words2) / len(words2) if words2 else 0.0
    key = normalize_loose(text)
    new_line = None
    if key and key not in {normalize_loose(l) for l in ocr.lines}:
        view_index = len(ocr.views)
        ocr.views.append(view)
        new_line = len(ocr.lines)
        ocr.lines.append(text)
        for w in words2:
            w.line_index, w.view = new_line, view_index
            ocr.words.append(w)
    r = Reread(line=line, kinds=kinds, text=text, mode=mode, conf=conf, new_line=new_line)
    ocr.rereads.append(r)
    return r


def reread_numbers(reading: LabelReading, kinds: set[str], run_pass: Callable, submit: Callable,
                   *, th: Thresholds = THRESHOLDS, timeout: float = TESSERACT_EXTRA_TIMEOUT_S) -> bool:
    """Re-read the figure lines of ``kinds`` ("alcohol", "volume") from their own crops.

    ``run_pass(image, psm, timeout)`` must return (lines, words, mean confidence) or raise ReaderError;
    ``submit`` runs it in the reader's pool so the crops are read side by side. Every pass is bounded
    by ``timeout`` and one that fails or is too slow is skipped. Returns True when anything was read.
    Cost: one single-line pass per crop (about 0.1 s each), at most ``number_reread_max_crops``."""
    ocr = reading.ocr
    kinds = {k for k in kinds if k in KINDS}
    if not kinds or ocr.source is None:
        return False
    t0 = perf_counter()
    jobs = []
    for index, wanted in lines_to_reread(ocr, kinds, th.number_reread_max_crops):
        made = line_crop(ocr, index, th.number_reread_text_px)
        if made is None:
            continue
        crop, view = made
        jobs.append((index, wanted, view, f"psm {PSM} x{view.scale:.2f}", submit(run_pass, crop, PSM, timeout)))
    did = False
    for index, wanted, view, mode, fut in jobs:
        try:
            lines2, words2, _ = fut.result()
        except ReaderError:
            continue   # an optional pass that fails or takes too long is skipped, never fatal
        record(ocr, index, wanted, mode, lines2, words2, view)
        did = True
    if did:
        ocr.text = "\n".join(ocr.lines)
        ocr.ms += (perf_counter() - t0) * 1000
        if "figure crops" not in ocr.engine:
            ocr.engine += " + figure crops"
    return did
