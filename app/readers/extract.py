"""Rule-based field extraction over raw OCR output (used by OCR-only readers such as Tesseract)."""

from __future__ import annotations

import statistics

from ..matching import compare_alcohol, compare_country, compare_volume, locate_and_compare
from ..models import Application, FieldResult
from .base import OCRResult


def prominent_line_index(ocr: OCRResult) -> int | None:
    """Index of the line printed in the largest type: on a label, almost always the brand name."""
    by_line: dict[int, list[int]] = {}
    for w in ocr.words:
        if len(w.text) >= 2 and any(ch.isalpha() for ch in w.text):
            by_line.setdefault(w.line_index, []).append(w.height)
    if not by_line:
        return None
    best = max(by_line, key=lambda i: statistics.median(by_line[i]))
    return best if best < len(ocr.lines) else None


def prominent_line(ocr: OCRResult) -> str | None:
    idx = prominent_line_index(ocr)
    return ocr.lines[idx] if idx is not None else None


def extract_and_compare(app: Application, ocr: OCRResult) -> list[FieldResult]:
    lines, text = ocr.lines, ocr.text
    brand_line = prominent_line_index(ocr)
    return [
        locate_and_compare("brand_name", app.brand_name, lines, fallback_found=prominent_line(ocr),
                           preferred_line=brand_line),
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
