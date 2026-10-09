"""One label, end to end: read -> extract -> compare -> warning checks -> overall verdict."""

from __future__ import annotations

from time import perf_counter

from PIL import Image

from .config import FIELD_BY_KEY
from .models import Application, FieldResult, Status, Timings, VerificationResult, Verdict, WarningResult
from .readers.base import LabelReader
from .readers.extract import attach_boxes, compare_from_fields, extract_and_compare
from .warning import check_warning


def overall_status(fields: list[FieldResult], warning: WarningResult) -> Status:
    verdicts = {f.verdict for f in fields}
    if Verdict.MISMATCH in verdicts or Verdict.NOT_FOUND in verdicts or warning.overall == Status.FAIL:
        return Status.FAIL
    if Verdict.NEAR_MATCH in verdicts or warning.overall == Status.REVIEW:
        return Status.REVIEW
    return Status.PASS


def summarize(status: Status, fields: list[FieldResult], warning: WarningResult) -> str:
    problems = [FIELD_BY_KEY[f.key].label.lower() for f in fields if f.verdict in (Verdict.MISMATCH, Verdict.NOT_FOUND)]
    reviews = [FIELD_BY_KEY[f.key].label.lower() for f in fields if f.verdict == Verdict.NEAR_MATCH]
    if warning.overall == Status.FAIL:
        problems.append("government warning")
    elif warning.overall == Status.REVIEW:
        reviews.append("government warning")
    checked = sum(1 for f in fields if f.verdict != Verdict.SKIPPED)
    if status == Status.PASS:
        return f"All {checked} fields match the application and the government warning is correct."
    parts = []
    if problems:
        parts.append(f"{len(problems)} problem{'s' if len(problems) > 1 else ''}: {', '.join(problems)}")
    if reviews:
        parts.append(f"Needs a look: {', '.join(reviews)}")
    return ". ".join(parts) + "."


def verify(app: Application, image: Image.Image, reader: LabelReader, image_name: str = "") -> VerificationResult:
    t0 = perf_counter()
    reading = reader.read(image)
    t_read = perf_counter()
    ocr = reading.ocr
    if reading.fields is not None:
        fields = compare_from_fields(app, reading.fields)
    else:
        fields = extract_and_compare(app, ocr)
        attach_boxes(ocr, fields)
    hint = reading.warning_hint
    warning = check_warning(ocr.lines, ocr.words, ocr.ink, bold_hint=hint.heading_bold if hint else None)
    status = overall_status(fields, warning)
    t_end = perf_counter()
    return VerificationResult(
        overall=status, summary=summarize(status, fields, warning), fields=fields, warning=warning,
        timings=Timings(read_ms=round((t_read - t0) * 1000, 1), match_ms=round((t_end - t_read) * 1000, 1),
                        total_ms=round((t_end - t0) * 1000, 1)),
        reader=ocr.engine or reader.name, ocr_text=ocr.text, image_name=image_name, application_id=app.application_id,
        read_confidence=round(ocr.mean_conf, 1) if ocr.mean_conf is not None else None,
        image_aspect=round(ocr.ink.shape[0] / ocr.ink.shape[1], 4) if ocr.ink is not None and ocr.ink.shape[1] else
        (round(image.height / image.width, 4) if image.width else None),
    )
