"""Rule-based field extraction over raw OCR output (used by OCR-only readers such as Tesseract)."""

from __future__ import annotations

import statistics

from ..matching import compare_alcohol, compare_country, compare_volume, locate_and_compare, not_found
from ..models import Application, FieldResult, Verdict
from ..normalize import normalize_loose, normalize_strict
from .base import OCRResult, OCRWord, text_height, upright_box


def line_heights(ocr: OCRResult) -> dict[int, float]:
    """Median height of the real words (2+ characters, with letters) on each OCR line, as printed on
    the page (a line read again from an enlarged crop is scaled back)."""
    by_line: dict[int, list[float]] = {}
    for w in ocr.words:
        if len(w.text) >= 2 and any(ch.isalpha() for ch in w.text) and 0 <= w.line_index < len(ocr.lines):
            by_line.setdefault(w.line_index, []).append(text_height(w, ocr.views))
    return {i: float(statistics.median(hs)) for i, hs in by_line.items()}


def line_boxes(ocr: OCRResult) -> dict[int, tuple[float, float, float, float]]:
    """Each OCR line's box (left, top, width, height) on the upright image, whatever view it was read in."""
    by_line: dict[int, list[tuple[int, int, int, int]]] = {}
    for w in ocr.words:
        if 0 <= w.line_index < len(ocr.lines):
            by_line.setdefault(w.line_index, []).append(upright_box(w, ocr.views))
    out = {}
    for i, bs in by_line.items():
        left, top = min(b[0] for b in bs), min(b[1] for b in bs)
        out[i] = (float(left), float(top), float(max(b[0] + b[2] for b in bs) - left),
                  float(max(b[1] + b[3] for b in bs) - top))
    return out


def prominent_line_index(ocr: OCRResult, heights: dict[int, float] | None = None) -> int | None:
    """Index of the line printed in the largest type: on a label, almost always the brand name."""
    heights = line_heights(ocr) if heights is None else heights
    return max(heights, key=heights.__getitem__) if heights else None


def prominent_line(ocr: OCRResult) -> str | None:
    idx = prominent_line_index(ocr)
    return ocr.lines[idx] if idx is not None else None


def figure_text(ocr: OCRResult, kind: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """What the matcher of one figure ("alcohol" or "volume") sees: the label text without the second
    reads made for the other figure, and for each line read again for this figure, the line as the
    page pass read it with every closer read of it. A second read made for the net contents never
    reaches the alcohol content, so a clean MATCH on one figure is never changed by a crop of the other."""
    other = {r.new_line for r in ocr.rereads if r.new_line is not None and kind not in r.kinds}
    text = "\n".join(l for i, l in enumerate(ocr.lines) if i not in other)
    by_line: dict[int, list[str]] = {}
    for r in ocr.rereads:
        if kind in r.kinds:
            by_line.setdefault(r.line, []).append(r.text)
    return text, [(ocr.lines[i], texts) for i, texts in by_line.items()]


def extract_and_compare(app: Application, ocr: OCRResult) -> list[FieldResult]:
    lines = ocr.lines
    # The second reads of figure lines are readings of a number, not of the text around it: they are
    # kept out of the boxes the text fields compare readings of the same place with, and out of the
    # line heights that pick the brand line and judge small print (a crop of a garbled line can come
    # back as a few tall letters).
    reread_lines = {r.new_line for r in ocr.rereads if r.new_line is not None}
    heights = {i: h for i, h in line_heights(ocr).items() if i not in reread_lines}
    boxes = {i: b for i, b in line_boxes(ocr).items() if i not in reread_lines}
    confs: dict[int, list[tuple[str, float]]] = {}
    for w in ocr.words:
        confs.setdefault(w.line_index, []).append((w.text, w.conf))
    brand_line = prominent_line_index(ocr, heights)
    alcohol_text, alcohol_rereads = figure_text(ocr, "alcohol")
    volume_text, volume_rereads = figure_text(ocr, "volume")
    return [
        locate_and_compare("brand_name", app.brand_name, lines, preferred_line=brand_line, line_heights=heights,
                           line_boxes=boxes, line_words=confs,
                           fallback_found=lines[brand_line] if brand_line is not None else None),
        locate_and_compare("class_type", app.class_type, lines, line_boxes=boxes, line_words=confs),
        compare_alcohol(app.alcohol_content, alcohol_text, rereads=alcohol_rereads),
        compare_volume(app.net_contents, volume_text, rereads=volume_rereads),
        locate_and_compare("bottler_name_address", app.bottler_name_address, lines, line_boxes=boxes, line_words=confs),
        compare_country(app.country_of_origin, lines),
    ]


def compare_from_fields(app: Application, fields: dict[str, str | None]) -> list[FieldResult]:
    """When a reader already returned structured fields (vision model), compare them like the OCR path.

    The application value is located inside the statement the model returned, so "Distilled and
    Bottled by Old Tom Distillery, ..." holds "Old Tom Distillery, ..." exactly as it does on the OCR
    path, and a value inside a longer phrase gets the same NEAR MATCH.
    """
    def text_field(key: str, expected: str) -> FieldResult:
        found = (fields.get(key) or "").strip()
        if expected.strip() and not found:
            return not_found(key, expected)
        return locate_and_compare(key, expected, found.splitlines(), fallback_found=normalize_strict(found))

    return [
        text_field("brand_name", app.brand_name),
        text_field("class_type", app.class_type),
        compare_alcohol(app.alcohol_content, fields.get("alcohol_content") or "", statement=True),
        compare_volume(app.net_contents, fields.get("net_contents") or ""),
        text_field("bottler_name_address", app.bottler_name_address),
        compare_country(app.country_of_origin, (fields.get("country_of_origin") or "").splitlines()),
    ]


# --- evidence boxes: where on the label each value was read ----------------------------------------
def word_box(words: list[OCRWord], width: int, height: int, views=None) -> list[float] | None:
    """Union of word boxes as [left, top, width, height] in percent of the upright image."""
    if not words or width <= 0 or height <= 0:
        return None
    boxes = [upright_box(w, views or []) for w in words]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return [round(100 * left / width, 2), round(100 * top / height, 2),
            round(100 * (right - left) / width, 2), round(100 * (bottom - top) / height, 2)]


def words_for_text(ocr: OCRResult, text: str, line_range: tuple[int, int] | None = None) -> list[OCRWord]:
    """The OCR words that spell ``text``: a contiguous run of words on the given lines (or on any
    line, or across two consecutive lines) whose normalized tokens equal the text's tokens.
    Falls back to every word on the given lines when the tokens cannot be aligned."""
    want = [t for t in normalize_loose(text).split() if t]
    if not want:
        return []
    ranges: list[tuple[int, int]]
    if line_range is not None:
        ranges = [line_range]
    else:
        n = len(ocr.lines)
        ranges = [(i, i) for i in range(n)] + [(i, i + 1) for i in range(n - 1)]
    for start, end in ranges:
        cand = [w for w in ocr.words if start <= w.line_index <= end]
        toks = [normalize_loose(w.text).split() for w in cand]
        flat = [(t, i) for i, ts in enumerate(toks) for t in ts]
        seq = [t for t, _ in flat]
        for a in range(len(seq) - len(want) + 1):
            if seq[a:a + len(want)] == want:
                first, last = flat[a][1], flat[a + len(want) - 1][1]
                return cand[first:last + 1]
    if line_range is not None:
        return [w for w in ocr.words if line_range[0] <= w.line_index <= line_range[1]]
    return []


def attach_boxes(ocr: OCRResult, fields: list[FieldResult]) -> None:
    """Fill FieldResult.box for every value that was read somewhere on the label."""
    if ocr.ink is None or not ocr.words:
        return
    height, width = ocr.ink.shape
    for f in fields:
        if not f.found or f.verdict in (Verdict.NOT_FOUND, Verdict.SKIPPED):
            continue
        rng = (f.lines[0], f.lines[1]) if f.lines else None
        f.box = word_box(words_for_text(ocr, f.found, rng), width, height, ocr.views)
