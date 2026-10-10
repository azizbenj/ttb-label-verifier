"""One label, end to end: read -> extract -> compare -> warning checks -> overall verdict."""

from __future__ import annotations

import threading
from time import perf_counter

from PIL import Image

from .config import EXPECTED_PASS_COST_S, FIELD_BY_KEY, LABEL_TIME_BUDGET_S, THRESHOLDS, Thresholds
from .matching import NOTE_APPLICATION_UNREADABLE
from .models import Application, FieldResult, Status, Timings, VerificationResult, Verdict, WarningResult
from .readers.base import LabelReader
from .matching import wants_second_read
from .readers.extract import (attach_boxes, compare_from_fields, extract_and_compare, field_signals, label_signals,
                              warning_conf)
from .readers.rapid import last_read_was_cold as rapid_last_read_was_cold
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


POOR_IMAGE_NOTE = ("Much of this label could not be read clearly. If possible, ask for a sharper image or the "
                   "artwork file.")
_FIGURES = ("alcohol_content", "net_contents")


def label_read_well(label_sig: dict, th: Thresholds = THRESHOLDS) -> bool:
    """The label as a whole was read well enough that something not found on it is probably not there."""
    return label_sig.get("clear_share", 0.0) >= th.clear_label_share and \
        label_sig.get("low_share", 1.0) <= th.clear_label_low_share


def judge_clarity(f: FieldResult, label_sig: dict | None, th: Thresholds = THRESHOLDS) -> tuple[bool | None, str]:
    """Whether a MISMATCH or NOT FOUND rests on clear evidence, and if not, why (for the agent).

    None when there is nothing to judge (any other verdict, or a reader that reports no word
    confidences: the problem then counts as clear, as before). The rules, each from measurement
    (README "Matching rules", "Evidence strength"):
    - A figure that was read and differs (alcohol content, net contents MISMATCH) is always clear: on the
      real labels OCR confidence did not tell a misread figure from a wrong one (planted wrong figures were
      read at confidences from 0 to 100, like the 7 misread ones), so a confidence rule would have sent
      wrong numbers to the REVIEW pile for every misread it caught.
    - A text field that was read and differs is clear when every word was read with at least
      ``clear_min_conf``, on average ``clear_mean_conf``, upright, and the text is a plausible reading.
    - Something not found (or a brand where nothing resembled it) is clear only on a label read well.
    - An application value that cannot be read as a figure is the application's problem: clear."""
    if not is_problem(f) or f.signals is None or not label_sig or not label_sig.get("words"):
        return None, ""
    s = f.signals
    if f.key in _FIGURES and f.verdict == Verdict.MISMATCH or NOTE_APPLICATION_UNREADABLE in f.note or \
            not f.expected.strip():
        return True, ""
    if f.verdict == Verdict.NOT_FOUND or s.get("fallback"):
        if label_read_well(label_sig, th):
            return True, ""
        return False, ("Much of this label was not read clearly, so this may be printed but too small, too "
                       "faint or set sideways to read. Check the label.")
    conf, low = s.get("mean_conf"), s.get("min_conf")
    if conf is None:   # the words could not be traced: judge by the label as a whole
        return (True, "") if label_read_well(label_sig, th) else \
            (False, "This text was not read clearly, so this may be a reading error. Check the label.")
    alnum = sum(ch.isalnum() for ch in f.found or "")
    if s.get("plausible", 0.0) < th.clear_plausible or alnum < th.clear_min_chars:
        return False, "What was read here is not plausible text, so this is probably a reading error. Check the label."
    if not s.get("upright", True):
        return False, "This text was only read sideways, so this may be a reading error. Check the label."
    if low < th.clear_min_conf or conf < th.clear_mean_conf:
        return False, (f"Read with low confidence ({round(conf)}% on average, {round(low)}% for the least sure word), "
                       "so this may be a reading error. Check the label.")
    return True, ""


def judge_warning(w: WarningResult, label_sig: dict | None, th: Thresholds = THRESHOLDS) -> tuple[bool | None, str]:
    """Whether a failed government warning rests on clear evidence. Only the final status of
    check_warning is judged here; its own checks are untouched. "No warning found" on a label that was
    not read well, or on which words of the statement were read but could not be assembled, is unclear;
    a wording or capitals failure is unclear when the statement's own words were read with low confidence."""
    if w.overall != Status.FAIL or not label_sig or not label_sig.get("words"):
        return None, ""
    if not w.present:
        if label_read_well(label_sig, th) and label_sig.get("warning_words", 0) < th.clear_warning_words:
            return True, ""
        return False, ("No statement could be read, but much of this label was not read clearly: it may be printed "
                       "sideways or too small to read. Check the label.")
    conf = label_sig.get("warning_conf")
    if conf is not None and conf < th.clear_warning_conf:
        return False, (f"The statement was read with low confidence ({round(conf)}% on average), so these differences "
                       "may be reading errors. Check the label.")
    return True, ""


def poor_image(fields: list[FieldResult], warning: WarningResult, th: Thresholds = THRESHOLDS) -> bool:
    """Most of what the label must carry could not be read clearly: a sharper image settles more than a look."""
    unclear = sum(1 for f in fields if FIELD_BY_KEY[f.key].required and is_unclear(f))
    unclear += warning.overall == Status.FAIL and warning.clear is False
    return unclear >= th.poor_image_unclear


def is_problem(f: FieldResult) -> bool:
    return f.verdict in (Verdict.MISMATCH, Verdict.NOT_FOUND)


def is_unclear(f: FieldResult) -> bool:
    """A MISMATCH or NOT FOUND whose evidence may be a misreading (``clear`` is False). A problem whose
    clarity could not be judged (None: a reader without word confidences) counts as clear."""
    return is_problem(f) and f.clear is False


def overall_status(fields: list[FieldResult], warning: WarningResult) -> Status:
    """FAIL only on a problem that was read clearly: a clear MISMATCH or NOT FOUND, or a warning check
    that failed on a clearly read statement. Problems whose evidence is unclear (low OCR confidence,
    sideways or implausible text, a poorly read label) ask for a look instead: the field keeps its
    verdict, the label goes to REVIEW."""
    if any(is_problem(f) and f.clear is not False for f in fields) or \
            (warning.overall == Status.FAIL and warning.clear is not False):
        return Status.FAIL
    if any(is_problem(f) or f.verdict == Verdict.NEAR_MATCH for f in fields) or warning.overall != Status.PASS:
        return Status.REVIEW
    return Status.PASS


def summarize(status: Status, fields: list[FieldResult], warning: WarningResult, poor_image: bool = False) -> str:
    problems = [FIELD_BY_KEY[f.key].label.lower() for f in fields if is_problem(f) and not is_unclear(f)]
    unclear = [FIELD_BY_KEY[f.key].label.lower() for f in fields if is_unclear(f)]
    reviews = [FIELD_BY_KEY[f.key].label.lower() for f in fields if f.verdict == Verdict.NEAR_MATCH]
    if warning.overall == Status.FAIL:
        (unclear if warning.clear is False else problems).append("government warning")
    elif warning.overall == Status.REVIEW:
        reviews.append("government warning")
    checked = sum(1 for f in fields if f.verdict != Verdict.SKIPPED)
    if status == Status.PASS:
        return f"All {checked} fields match the application and the government warning is correct."
    parts = []
    if poor_image:
        parts.append(POOR_IMAGE_NOTE.rstrip("."))
    if problems:
        parts.append(f"{len(problems)} problem{'s' if len(problems) > 1 else ''}: {', '.join(problems)}")
    if unclear:
        parts.append(f"Could not be read clearly: {', '.join(unclear)}")
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
    label_sig = None
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
            if not (name == "escalate" and rapid_last_read_was_cold()):   # a cold start is not what a read costs
                cost.add(clock() - t_step)
            if added:
                t_read = perf_counter()
                fields = extract_and_compare(app, ocr)
                warning = check_warning(ocr.lines, ocr.words, ocr.ink, views=ocr.views)
        attach_boxes(ocr, fields)
        label_sig = label_signals(ocr, THRESHOLDS.unreadable_word_conf)
        label_sig["warning_conf"] = warning_conf(ocr, warning)
        for f in fields:
            f.signals = field_signals(ocr, f)
            f.read_conf = f.signals.get("mean_conf")
            f.clear, f.clear_note = judge_clarity(f, label_sig)
        warning.clear, warning.clear_note = judge_warning(warning, label_sig)
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
    poor = not unreadable and status != Status.PASS and poor_image(fields, warning)
    t_end = perf_counter()
    return VerificationResult(
        overall=status, summary=summarize(status, fields, warning, poor), fields=fields, warning=warning,
        poor_image=poor,
        timings=Timings(read_ms=round((t_read - t0) * 1000, 1), match_ms=round((t_end - t_read) * 1000, 1),
                        total_ms=round((t_end - t0) * 1000, 1)),
        reader=ocr.engine or reader.name, ocr_text=ocr.text, image_name=image_name, application_id=app.application_id,
        read_confidence=round(ocr.mean_conf, 1) if ocr.mean_conf is not None else None,
        skew_deg=round(ocr.skew, 2), unreadable=unreadable, words_read=words_read, signals=label_sig,
        image_aspect=round(ocr.ink.shape[0] / ocr.ink.shape[1], 4) if ocr.ink is not None and ocr.ink.shape[1] else
        (round(image.height / image.width, 4) if image.width else None),
    )
