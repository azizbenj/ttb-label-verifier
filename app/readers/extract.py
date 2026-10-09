"""Rule-based field extraction over raw OCR output (used by OCR-only readers such as Tesseract)."""

from __future__ import annotations

import statistics

from ..matching import compare_alcohol, compare_country, compare_volume, locate_and_compare
from ..models import Application, FieldResult
from .base import OCRResult


def line_heights(ocr: OCRResult) -> dict[int, float]:
    """Median height of the real words (2+ characters, with letters) on each OCR line."""
    by_line: dict[int, list[int]] = {}
    for w in ocr.words:
        if len(w.text) >= 2 and any(ch.isalpha() for ch in w.text) and 0 <= w.line_index < len(ocr.lines):
            by_line.setdefault(w.line_index, []).append(w.height)
    return {i: float(statistics.median(hs)) for i, hs in by_line.items()}


def prominent_line_index(ocr: OCRResult, heights: dict[int, float] | None = None) -> int | None:
    """Index of the line printed in the largest type: on a label, almost always the brand name."""
    heights = line_heights(ocr) if heights is None else heights
    return max(heights, key=heights.__getitem__) if heights else None


def prominent_line(ocr: OCRResult) -> str | None:
    idx = prominent_line_index(ocr)
    return ocr.lines[idx] if idx is not None else None


def extract_and_compare(app: Application, ocr: OCRResult) -> list[FieldResult]:
    lines, text = ocr.lines, ocr.text
    heights = line_heights(ocr)
    brand_line = prominent_line_index(ocr, heights)
    return [
        locate_and_compare("brand_name", app.brand_name, lines, preferred_line=brand_line, line_heights=heights,
                           fallback_found=lines[brand_line] if brand_line is not None else None),
        locate_and_compare("class_type", app.class_type, lines),
        compare_alcohol(app.alcohol_content, text),
        compare_volume(app.net_contents, text),
        locate_and_compare("bottler_name_address", app.bottler_name_address, lines),
        compare_country(app.country_of_origin, lines),
    ]


def compare_from_fields(app: Application, fields: dict[str, str | None]) -> list[FieldResult]:
    """When a reader already returned structured fields (vision model), compare them directly."""
    from ..matching import compare_text, not_found, skipped

    def text_field(key: str, expected: str) -> FieldResult:
        if not expected.strip():
            return skipped(key)
        found = fields.get(key)
        return compare_text(key, expected, found) if found else not_found(key, expected)

    country_lines = [fields.get("country_of_origin") or ""]
    return [
        text_field("brand_name", app.brand_name),
        text_field("class_type", app.class_type),
        compare_alcohol(app.alcohol_content, fields.get("alcohol_content") or ""),
        compare_volume(app.net_contents, fields.get("net_contents") or ""),
        text_field("bottler_name_address", app.bottler_name_address),
        compare_country(app.country_of_origin, country_lines),
    ]
