"""One label, end to end: read -> extract -> compare -> warning checks -> overall verdict."""

from __future__ import annotations

import threading
from time import perf_counter

from PIL import Image

from .config import EXPECTED_PASS_COST_S, FIELD_BY_KEY, LABEL_TIME_BUDGET_S, THRESHOLDS
from .models import Application, FieldResult, Status, Timings, VerificationResult, Verdict, WarningResult
from .readers.base import LabelReader
from .matching import wants_second_read
from .readers.extract import attach_boxes, compare_from_fields, extract_and_compare
from .warning import check_warning


class PassCost:
    """Running average of what an optional pass costs in this process, seeded from config so the
    first labels are budgeted sensibly before anything was measured. A pass that returned at once
    (nothing to do, or switched off) is not a sample."""

    MIN_SAMPLE_S = 0.05

    def __init__(self, seed: float):
        self.total, self.n = seed, 1
        self._lock = threading.Lock()

    def add(self, seconds: float) -> None:
        if seconds < self.MIN_SAMPLE_S:
            return
        with self._lock:
            self.total += seconds
            self.n += 1

    @property
    def expected(self) -> float:
        with self._lock:
            return self.total / self.n


PASS_COSTS = {name: PassCost(seed) for name, seed in EXPECTED_PASS_COST_S.items()}
SKIPPED_NOTE = {"extend": "turned and contrast passes skipped for time",
                "escalate": "RapidOCR escalation skipped for time"}


def was_read(f: FieldResult) -> bool:
    """Something on the label was read for this field, whether it agrees or not. The brand's fallback
    (the largest line, shown when nothing resembles the brand) does not count: it is not a reading of it."""
    if f.verdict in (Verdict.NOT_FOUND, Verdict.SKIPPED):
        return False
    return not (f.key == "brand_name" and f.verdict == Verdict.MISMATCH and f.lines is None)


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


def verify(app: Application, image: Image.Image, reader: LabelReader, image_name: str = "", *,
           budget_s: float = LABEL_TIME_BUDGET_S, clock=perf_counter) -> VerificationResult:
    """``budget_s``: the per-label time budget the optional passes must fit in (0 = none); ``clock`` is
    the time source, replaceable in tests."""
    t0 = perf_counter()
    started = clock()
    reading = reader.read(image)
    t_read = perf_counter()
    ocr = reading.ocr
    hint = reading.warning_hint
    if reading.fields is not None:
        fields = compare_from_fields(app, reading.fields)
        warning = check_warning(ocr.lines, ocr.words, ocr.ink, bold_hint=hint.heading_bold if hint else None)
    else:
        fields = extract_and_compare(app, ocr)
        warning = check_warning(ocr.lines, ocr.words, ocr.ink, views=ocr.views)
        # A figure read wrong is the one error this check cannot afford. When the alcohol content or the
        # net contents is missing, different or doubtful, the lines that carry a figure are cut out,
        # scaled to a size Tesseract reads well and read again on their own before anything else is
        # tried (app/readers/numbers.py).
        kinds = {k for k in (wants_second_read(f) for f in fields) if k}
        reread = getattr(reader, "reread_numbers", None)
        if kinds and reread is not None and reread(reading, kinds):
            t_read = perf_counter()
            fields = extract_and_compare(app, ocr)
        # Something missing or different: before concluding, read the label again turned sideways and
        # with local contrast (warnings and bottler lines on cans are often printed at 90 degrees), and
        # if that still leaves something, once more with the second engine (RapidOCR reads the display
        # typefaces and light-on-photo text Tesseract cannot). Each step runs only while needed, and
        # only when the time spent so far plus the step's expected cost fits the label's time budget:
        # a skipped step keeps the verdict as it stands and is named in the result's reader string.
        for name in ("extend", "escalate"):
            step = getattr(reader, name, None)
            unresolved = any(f.verdict in (Verdict.NOT_FOUND, Verdict.MISMATCH) for f in fields) or \
                warning.overall != Status.PASS
            if step is None or not unresolved:
                continue
            cost = PASS_COSTS[name]
            if budget_s and clock() - started + cost.expected > budget_s:
                ocr.engine += f" ({SKIPPED_NOTE[name]})"
                continue
            t_step = clock()
            added = step(reading)
            cost.add(clock() - t_step)
            if added:
                t_read = perf_counter()
                fields = extract_and_compare(app, ocr)
                warning = check_warning(ocr.lines, ocr.words, ocr.ink, views=ocr.views)
        attach_boxes(ocr, fields)
    words_read, unreadable = None, False
    if reading.fields is None:
        clear = [w for w in ocr.words if w.conf >= THRESHOLDS.unreadable_word_conf
                 and sum(ch.isalpha() for ch in w.text) >= 3]
        words_read = len(clear)
        # "Unreadable" withholds the verdict, so it is only for an image where nothing was read at all: a
        # neck label reading "VODKA / 40% ALC/VOL / 1 LITER" against 45% and 750 mL has few words but
        # two clear mismatches and must FAIL.
        read_any = any(was_read(f) for f in fields)
        unreadable = words_read < THRESHOLDS.unreadable_min_words and not read_any and not warning.present
    status = overall_status(fields, warning)
    t_end = perf_counter()
    return VerificationResult(
        overall=status, summary=summarize(status, fields, warning), fields=fields, warning=warning,
        timings=Timings(read_ms=round((t_read - t0) * 1000, 1), match_ms=round((t_end - t_read) * 1000, 1),
                        total_ms=round((t_end - t0) * 1000, 1)),
        reader=ocr.engine or reader.name, ocr_text=ocr.text, image_name=image_name, application_id=app.application_id,
        read_confidence=round(ocr.mean_conf, 1) if ocr.mean_conf is not None else None,
        skew_deg=round(ocr.skew, 2), unreadable=unreadable, words_read=words_read,
        image_aspect=round(ocr.ink.shape[0] / ocr.ink.shape[1], 4) if ocr.ink is not None and ocr.ink.shape[1] else
        (round(image.height / image.width, 4) if image.width else None),
    )
