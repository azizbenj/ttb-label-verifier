"""Batch mode: a CSV of applications plus their label images, processed in a thread pool.

Jobs live in memory for the life of the process (this is a prototype; see README).
"""

from __future__ import annotations

import csv
import io
import re
import logging
import threading
import uuid
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import (BATCH_JOBS_KEPT, BATCH_JOB_TTL_S, BATCH_WORKERS, FIELDS, IMAGE_EXTENSIONS, MAX_BATCH_IMAGES,
                     MAX_BATCH_UPLOAD_BYTES, MAX_IMAGE_BYTES, MAX_ZIP_MEMBERS, MAX_ZIP_UNCOMPRESSED)
from .decision_log import log_decision
from .decisions import prompts_for
from .images import MAX_LABEL_PARTS, ImageError, flatten, open_image, stitch
from .models import Application, Status, Verdict, VerificationResult
from .normalize import parse_alcohol, parse_net_contents
from .pipeline import verify
from .readers.base import LabelReader, ReaderError

log = logging.getLogger("labelcheck")
ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data" / "batch"

COLUMN_ALIASES = {
    "image": {"image", "image_filename", "image_file", "filename", "file", "label_image", "image_name", "label"},
    "application_id": {"application_id", "app_id", "id", "ttb_id", "cola_id", "application"},
    "brand_name": {"brand_name", "brand"},
    "class_type": {"class_type", "class", "type", "class_and_type", "classtype", "class_or_type", "class_type_designation"},
    "alcohol_content": {"alcohol_content", "alcohol", "abv", "alc", "alcohol_by_volume"},
    "net_contents": {"net_contents", "net_content", "volume", "contents", "size"},
    "bottler_name_address": {"bottler_name_address", "bottler", "bottler_name_and_address", "name_and_address",
                             "producer", "bottler_address"},
    "country_of_origin": {"country_of_origin", "country", "origin"},
}
REQUIRED_COLUMNS = ("image", "brand_name", "class_type", "alcohol_content", "net_contents")


class BatchError(Exception):
    """A problem the user can fix, phrased for them. ``columns``: for a CSV with missing headers, one
    (expected, header in the file or None, "rename" | "add") per required column that was not found."""

    def __init__(self, message: str, *, title: str | None = None, kind: str = "",
                 columns: list[tuple[str, str | None, str]] | None = None, found: list[str] | None = None):
        super().__init__(message)
        self.title, self.kind, self.columns, self.found = title, kind, columns, found


_TOO_MANY = "That's more than one batch can take."


PREVIEW_MAX_SIDE = 900                 # px, the copy of each label kept for the detail panel
PREVIEW_BUDGET_BYTES = 120 * 1024 * 1024   # across all jobs in memory; beyond it, no previews are kept
_preview_bytes = 0                     # previews held by the jobs in memory (released when a job is pruned)
_PREVIEW_LOCK = threading.Lock()
_ORDER = {"REVIEW": 0, "FAIL": 1, "ERROR": 2, "PASS": 3, "PENDING": 4}


@dataclass
class BatchItem:
    row: int
    application_id: str
    image_name: str
    application: Application | None = None
    result: VerificationResult | None = None
    error: str | None = None
    preview: bytes | None = None       # downscaled JPEG for the detail panel, when the budget allows
    needs: str = ""                    # what the agent has to do with this label, in a few words
    focus_box: list[float] | None = None   # region to outline in the detail panel: the first problem or review
    # The agent's answer to each question the label raises (prompt key -> pass, fail or skip). One answer
    # settles one question, never the label: see ``decision``.
    answers: dict[str, str] = field(default_factory=dict)

    def question_keys(self) -> list[str]:
        return [p.key for p in prompts_for(self.result)] if self.result else []

    def open_questions(self) -> list[str]:
        return [k for k in self.question_keys() if k not in self.answers]

    @property
    def decision(self) -> str:
        """The label's decision, from the answers: ``fail`` when any question was answered no, ``pass`` when
        every one was answered yes, ``skip`` when all are answered and some were skipped, ``partial`` while
        questions are still open, blank before the first answer."""
        if not self.answers:
            return ""
        keys = self.question_keys() or list(self.answers)
        values = [self.answers.get(k, "") for k in keys]
        if "fail" in values:
            return "fail"
        if "" in values:
            return "partial"
        return "pass" if all(v == "pass" for v in values) else "skip"

    def answers_text(self) -> str:
        """One question per entry, for the export: 'brand_name: yes; net_contents: open'."""
        words = {"pass": "yes", "fail": "no", "skip": "skipped"}
        return "; ".join(f"{k}: {words.get(self.answers.get(k, ''), 'open')}" for k in self.question_keys())

    @property
    def status(self) -> str:
        if self.error:
            return Status.ERROR.value
        return self.result.overall.value if self.result else "PENDING"

    @property
    def brand(self) -> str:
        return self.application.brand_name if self.application else ""

    @property
    def order(self) -> int:
        return _ORDER.get(self.status, 9)


def needs_phrase(result: VerificationResult) -> tuple[str, list[float] | None]:
    """Short instruction for the triage panel, plus the region to outline on the label."""
    reviews = [f for f in result.fields if f.verdict == Verdict.NEAR_MATCH]
    problems = [f for f in result.fields if f.verdict in (Verdict.MISMATCH, Verdict.NOT_FOUND)]
    w = result.warning
    if result.overall == Status.PASS:
        return "Nothing", None
    if result.overall == Status.FAIL:
        # Check before sending anything back: on real artwork a FAIL is often a misread the agent can clear.
        if problems:
            return f"Check the {problems[0].label.lower()}", problems[0].box or w.box
        return "Check the government warning", w.box
    if reviews:
        return f"Confirm the {reviews[0].label.lower()}", reviews[0].box
    if w.heading_bold == Status.REVIEW:
        return "A look at the heading weight", w.box
    if w.heading_caps == Status.REVIEW:
        return "A look at the heading", w.box
    return "A look at the warning statement", w.box


def make_preview(image, skew: float = 0.0) -> bytes | None:
    """A small JPEG of the label for the detail panel, within the memory budget of the jobs in memory."""
    global _preview_bytes
    if _preview_bytes >= PREVIEW_BUDGET_BYTES:
        return None
    try:
        from PIL import Image
        img = flatten(image)   # upright, transparency on white (a plain RGB conversion paints it black)
        if skew:
            img = img.convert("RGB").rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80, optimize=True)
    except Exception:  # a preview is a convenience, never a reason to fail the label
        return None
    data = buf.getvalue()
    with _PREVIEW_LOCK:
        _preview_bytes += len(data)
    return data


DECISIONS = ("pass", "fail", "skip", "clear")
EXPORT_INCLUDE = ("fields", "warning", "decisions", "ocr", "timings")
EXPORT_DEFAULT = {"fields", "warning", "decisions", "timings"}


@dataclass
class BatchJob:
    id: str
    source: str
    items: list[BatchItem]
    issues: list[str] = field(default_factory=list)
    started: float = field(default_factory=perf_counter)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    done: int = 0
    finished_ms: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def is_done(self) -> bool:
        return self.done >= self.total

    def counts(self) -> dict[str, int]:
        c = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "ERROR": 0}
        for it in self.items:
            if it.status in c:
                c[it.status] += 1
        return c

    @property
    def elapsed_ms(self) -> float:
        return self.finished_ms if self.finished_ms is not None else (perf_counter() - self.started) * 1000

    @property
    def processed(self) -> int:
        """Labels actually read (rows with a file problem count as done but took no time)."""
        return sum(1 for it in self.items if it.result is not None)

    @property
    def per_label_ms(self) -> float | None:
        """Wall-clock time per label so far, across the workers."""
        return self.elapsed_ms / self.processed if self.processed else None

    @property
    def eta_ms(self) -> float | None:
        """Estimate from the measured rate; None until 10 labels are done."""
        if self.is_done or self.processed < 10 or self.per_label_ms is None:
            return None
        return (self.total - self.done) * self.per_label_ms

    def mean_read_ms(self) -> float | None:
        xs = [it.result.timings.read_ms for it in self.items if it.result]
        return sum(xs) / len(xs) if xs else None

    # --- the review queues: the labels that need a look, or the fails to check, in triage order ------
    def review_queue(self, pile: str = "review") -> list[int]:
        status = Status.FAIL.value if pile == "fail" else Status.REVIEW.value
        return [i for i, it in self.ordered_items() if it.status == status]

    @property
    def decided_count(self) -> int:
        """Labels whose every question has an answer (a label with questions still open is not decided)."""
        return sum(1 for it in self.items if it.decision in ("pass", "fail", "skip"))

    @property
    def cleared_count(self) -> int:
        """Labels flagged by the tool that the agent cleared: every question answered yes."""
        return sum(1 for it in self.items if it.status != Status.PASS.value and it.decision == "pass")

    def decide(self, index: int, value: str, key: str = "") -> BatchItem:
        """Record the agent's answer to one of the label's questions (``key``); without a key the answer
        applies to every question. ``clear`` undoes it. Never changes a verdict. With ``DECISION_LOG`` set,
        the answer (and an undo) also goes to the decision log with what the tool concluded about the label."""
        if value not in DECISIONS:
            raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
        with self.lock:
            item = self.items[index]
            keys = item.question_keys()
            if key and key not in keys:
                raise ValueError(f"this label has no question '{key}'")
            targets = [key] if key else (keys or [""])
            for k in targets:
                if value == "clear":
                    item.answers.pop(k, None)
                else:
                    item.answers[k] = value
        log_decision(self, item, index, value, key)
        return item

    @property
    def slug(self) -> str:
        """The source name as a file-name part: 'Sample batch' -> 'sample-batch'."""
        s = re.sub(r"[^a-z0-9]+", "-", Path(self.source).stem.lower()).strip("-")
        return s or "batch"


    def ordered_items(self) -> list[tuple[int, "BatchItem"]]:
        """(index, item) in triage order: REVIEW, FAIL, ERROR, PASS, then still pending."""
        return sorted(enumerate(self.items), key=lambda p: (p[1].order, p[0]))

    @property
    def started_at_local(self) -> str:
        return self.created_at.astimezone().strftime("%H:%M:%S")

    @property
    def finished_at_local(self) -> str:
        if self.finished_ms is None:
            return ""
        from datetime import timedelta
        return (self.created_at + timedelta(milliseconds=self.finished_ms)).astimezone().strftime("%H:%M:%S")

    def image_count(self) -> int:
        return sum(1 for it in self.items if it.result is not None or (it.error and "No image named" not in it.error))


JOBS: dict[str, BatchJob] = {}
_JOBS_LOCK = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=BATCH_WORKERS, thread_name_prefix="batch")


def _prune_jobs() -> None:
    """Forget finished jobs past their retention so a long-running server does not grow without bound.
    Their previews go back to the budget: it was counted for the life of the process, so after 120 MB
    of previews no later batch ever showed a label image again."""
    global _preview_bytes
    finished = sorted((j for j in JOBS.values() if j.is_done), key=lambda j: j.created_at)
    cutoff = datetime.now(timezone.utc).timestamp() - BATCH_JOB_TTL_S
    for i, job in enumerate(finished):
        if job.created_at.timestamp() < cutoff or i < len(finished) - BATCH_JOBS_KEPT:
            JOBS.pop(job.id, None)
            freed = sum(len(it.preview) for it in job.items if it.preview)
            with _PREVIEW_LOCK:
                _preview_bytes = max(0, _preview_bytes - freed)


# --- CSV ------------------------------------------------------------------------------------------
def _norm_header(h: str) -> str:
    return (h or "").strip().lower().replace("﻿", "").replace("-", "_").replace(" ", "_").replace("/", "_")


def _missing_columns(headers: list[str], mapping: dict[str, str], missing: list[str]) -> BatchError:
    """Name each missing column next to the header in the file that most likely meant it."""
    from rapidfuzz import fuzz
    unused = [h for h in headers if h and h not in mapping.values()]
    rows: list[tuple[str, str | None, str]] = []
    for key in missing:
        best, best_score = None, 0.0
        for h in unused:
            n = _norm_header(h)
            score = max(fuzz.ratio(n, a) for a in COLUMN_ALIASES[key] | {key})
            if score > best_score:
                best, best_score = h, score
        if best is not None and best_score >= 60:
            rows.append((key, best, "rename"))
            unused.remove(best)
        else:
            rows.append((key, None, "add"))
    n = len(missing)
    found = [h for h in headers if h]
    return BatchError(
        f"The CSV is missing {n} required column{'s' if n != 1 else ''}: {', '.join(missing)}. Found: "
        f"{', '.join(found) or 'no headers'}. Rename the headers, or download the template and paste your rows in.",
        title=f"The CSV is missing {n} required column{'s' if n != 1 else ''}.", kind="csv_columns",
        columns=rows, found=found)


def parse_applications_csv(data: bytes) -> tuple[list[dict], list[str]]:
    """Return (rows, issues). Each row has canonical keys; blank optional values are ''."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    # Excel writes ";" (or tab) separated files in many locales: take the separator the header uses most.
    header = text.lstrip("\ufeff").split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count) if any(d in header for d in ",;\t") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise BatchError("The CSV file is empty.")
    mapping: dict[str, str] = {}
    for raw in reader.fieldnames:
        n = _norm_header(raw)
        for key, aliases in COLUMN_ALIASES.items():
            if n in aliases and key not in mapping:
                mapping[key] = raw
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        raise _missing_columns(list(reader.fieldnames), mapping, missing)
    rows, issues = [], []
    for raw_row in reader:
        row = {key: (raw_row.get(col) or "").strip() for key, col in mapping.items()}
        for key in COLUMN_ALIASES:
            row.setdefault(key, "")
        if not any(row.values()):
            continue
        row["_row"] = reader.line_num  # the file's line number, also when a quoted cell spans lines
        rows.append(row)
    if not rows:
        raise BatchError("The CSV has a header but no application rows.")
    if len(rows) > MAX_BATCH_IMAGES:
        raise BatchError(f"The CSV has {len(rows)} rows; the limit is {MAX_BATCH_IMAGES} per batch. Split it in two "
                         "and run them one after the other; results export separately.", title=_TOO_MANY)
    return rows, issues


def template_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image", "application_id", "brand_name", "class_type", "alcohol_content", "net_contents",
                "bottler_name_address", "country_of_origin"])
    w.writerow(["old_tom_clean.png", "APP-0001", "OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey",
                "45% Alc./Vol.", "750 mL", "Old Tom Distillery, Bardstown, Kentucky 40004", ""])
    w.writerow(["glen_morar.png", "APP-0002", "GLEN MORAR", "Single Malt Scotch Whisky", "86 Proof", "700 mL",
                "Highland Imports, Inc., New York, NY 10001", "Scotland"])
    return buf.getvalue()


# --- images ----------------------------------------------------------------------------------------
def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def collect_images(uploads: list[tuple[str, bytes]]) -> tuple[dict[str, bytes | None], list[str]]:
    """Accept loose image files and/or zip archives. Keys are lower-cased base names.

    Two different files with the same name (``lot1/label.png`` and ``lot2/label.png`` in a zip) map
    to None: rows naming that file must not be checked against whichever one happened to come last.
    """
    images: dict[str, bytes | None] = {}
    sources: dict[str, list[str]] = {}
    issues: list[str] = []
    total = 0

    def add(key: str, data: bytes, source: str) -> None:
        nonlocal total
        sources.setdefault(key, []).append(source)
        if key in images and images[key] != data:
            images[key] = None
            return
        if key not in images:
            total += len(data)
            if total > MAX_BATCH_UPLOAD_BYTES:
                raise BatchError(f"The images add up to more than {MAX_BATCH_UPLOAD_BYTES // (1024 * 1024)} MB. "
                                 "Split the folder in two and run them one after the other; results export "
                                 "separately.", title=_TOO_MANY)
            images[key] = data

    for name, data in uploads:
        base = Path(name).name
        if base.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir() and not Path(m.filename).name.startswith(".")]
                    if len(members) > MAX_ZIP_MEMBERS:
                        raise BatchError(f"'{base}' holds {len(members)} files; the limit is {MAX_ZIP_MEMBERS}. "
                                         "Split it in two and run them one after the other.", title=_TOO_MANY)
                    if sum(m.file_size for m in members) > MAX_ZIP_UNCOMPRESSED:
                        raise BatchError(f"'{base}' is too large when unpacked. Split it in two and run them one "
                                         "after the other.", title=_TOO_MANY)
                    for m in members:
                        mb = Path(m.filename).name
                        if not _is_image_name(mb):
                            if not mb.lower().endswith(".csv"):
                                issues.append(f"Skipped '{m.filename}' inside {base}: not an image.")
                            continue
                        if m.file_size > MAX_IMAGE_BYTES:
                            issues.append(f"Skipped '{mb}': larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
                            continue
                        try:
                            member = zf.read(m)
                        except (RuntimeError, NotImplementedError, zlib.error):  # encrypted or unsupported
                            issues.append(f"Skipped '{m.filename}' inside {base}: it is encrypted or compressed "
                                          "in a way this tool cannot open.")
                            continue
                        add(mb.lower(), member, f"{base}/{m.filename}")
            except zipfile.BadZipFile:
                raise BatchError(f"'{base}' could not be unpacked. Re-zip the images (no password) and try again.",
                                 title="We couldn't open the zip.") from None
        elif _is_image_name(base):
            if len(data) > MAX_IMAGE_BYTES:
                issues.append(f"Skipped '{base}': larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
                continue
            add(base.lower(), data, base)
        else:
            issues.append(f"Skipped '{base}': not an image (use PNG, JPG, TIFF, BMP or WEBP).")
    for key, data in images.items():
        if data is None:
            issues.append(f"Several different files are named '{key}' ({', '.join(sources[key])}); "
                          "rows using that name were not checked. Please rename them.")
    if len(images) > MAX_BATCH_IMAGES:
        raise BatchError(f"You added {len(images)} images; the limit is {MAX_BATCH_IMAGES} per batch. Split the folder "
                         "in two and run them one after the other; results export separately.", title=_TOO_MANY)
    return images, issues


_AMBIGUOUS = object()


def _lookup_image(images: dict[str, bytes | None], name: str) -> bytes | object | None:
    """The uploaded file for a CSV name, None if there is none, _AMBIGUOUS if several files could be meant."""
    key = Path(name).name.lower()
    if key in images:
        return images[key] if images[key] is not None else _AMBIGUOUS
    stem = Path(key).stem
    matches = [k for k in images if Path(k).stem == stem]  # allow the CSV to omit the extension
    if len(matches) > 1 or (matches and images[matches[0]] is None):
        return _AMBIGUOUS
    return images[matches[0]] if matches else None


# --- jobs ------------------------------------------------------------------------------------------
def _row_to_application(row: dict) -> tuple[Application | None, str | None]:
    missing = [f.label.lower() for f in FIELDS if f.required and not row.get(f.key)]
    if missing:
        return None, f"Row {row['_row']}: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} blank."
    # Same checks as the single-label form: an unreadable application value is a data problem to fix,
    # not a label FAIL.
    if parse_alcohol(row["alcohol_content"]) is None:
        return None, (f"Row {row['_row']}: alcohol content '{row['alcohol_content']}' should look like "
                      "'45% Alc./Vol.', '45%' or '90 Proof'.")
    if parse_net_contents(row["net_contents"]) is None:
        return None, (f"Row {row['_row']}: net contents '{row['net_contents']}' should look like "
                      "'750 mL', '1.75 L' or '12 fl oz'.")
    return Application(**{k: row.get(k, "") for k in COLUMN_ALIASES if k != "image"}), None


def build_items(rows: list[dict], images: dict[str, bytes | None]) -> tuple[list[BatchItem], list[tuple[BatchItem, bytes]], list[str]]:
    items, work, issues = [], [], []
    used: set[str] = set()
    referenced: set[str] = set()
    for row in rows:
        item = BatchItem(row=row["_row"], application_id=row.get("application_id") or f"row {row['_row']}",
                         image_name=row.get("image", ""))
        app, err = _row_to_application(row)
        item.application = app
        # Several images for one application: "front.png; back.png" (or separated by "|").
        names = [n.strip() for n in re.split(r"[;|]", item.image_name) if n.strip()] if item.image_name else []
        referenced.update(Path(n).name.lower() for n in names)   # a rejected row still names its image
        found = [_lookup_image(images, n) for n in names]
        data = None
        missing = [n for n, d in zip(names, found) if d is None]
        if names and not missing:
            data = _AMBIGUOUS if any(d is _AMBIGUOUS for d in found) else found
        if err:
            item.error = err
        elif not item.image_name:
            item.error = f"Row {row['_row']}: no image file name."
        elif len(names) > MAX_LABEL_PARTS:
            item.error = f"Row {row['_row']}: {len(names)} images listed; up to {MAX_LABEL_PARTS} per application."
        elif data is None:
            item.error = f"No image named '{missing[0] if missing else item.image_name}' was uploaded."
        elif data is _AMBIGUOUS:
            item.error = f"More than one uploaded file could be '{item.image_name}', so it was not checked."
        else:
            used.update(Path(n).name.lower() for n in names)
            work.append((item, data))
        items.append(item)
    for name, data in images.items():
        if data is None or name in used or any(Path(name).stem == Path(u).stem for u in used):
            continue
        if name in referenced or any(Path(name).stem == Path(u).stem for u in referenced):
            continue   # its row was rejected; that row's own message says why
        issues.append(f"Image '{name}' has no matching row in the CSV, so it was not checked.")
    return items, work, issues


def _process(job: BatchJob, item: BatchItem, data: list[bytes], reader: LabelReader) -> None:
    try:
        names = [n.strip() for n in re.split(r"[;|]", item.image_name) if n.strip()]
        image = stitch([open_image(d, n) for d, n in zip(data, names)])
        result = verify(item.application, image, reader, image_name=item.image_name)
        item.preview = make_preview(image, result.skew_deg)
        if result.unreadable:   # not a FAIL: nothing could be compared
            n = result.words_read or 0
            item.error = (f"Unreadable image: only {n} word{'s' if n != 1 else ''} could be read clearly, so nothing "
                          "was compared. Use the artwork file or a flat, straight-on scan.")
            item.needs = "Send a readable image"
        else:
            item.result = result
            item.needs, item.focus_box = needs_phrase(result)
    except (ImageError, ReaderError) as e:
        item.error = str(e)
    except Exception as e:  # keep the batch going; surface the reason
        log.exception("batch item %s failed", item.image_name)
        item.error = f"Could not check this label ({type(e).__name__}: {e})."
    finally:
        with job.lock:
            job.done += 1
            if job.done >= job.total:
                job.finished_ms = (perf_counter() - job.started) * 1000


def start_job(rows: list[dict], images: dict[str, bytes], reader: LabelReader, source: str,
              issues: list[str] | None = None) -> BatchJob:
    items, work, more_issues = build_items(rows, images)
    job = BatchJob(id=uuid.uuid4().hex[:10], source=source, items=items, issues=(issues or []) + more_issues)
    job.done = sum(1 for it in items if it.error)  # rows that cannot run count as finished
    if job.done >= job.total:
        job.finished_ms = 0.0
    with _JOBS_LOCK:
        _prune_jobs()
        JOBS[job.id] = job
    for item, data in work:
        _executor.submit(_process, job, item, data, reader)
    return job


def start_sample_job(reader: LabelReader) -> BatchJob:
    csv_path = BATCH_DIR / "applications.csv"
    if not csv_path.exists():
        raise BatchError("The sample batch is not available on this server.")
    rows, issues = parse_applications_csv(csv_path.read_bytes())
    images = {p.name.lower(): p.read_bytes() for p in BATCH_DIR.iterdir() if _is_image_name(p.name)}
    return start_job(rows, images, reader, source="sample batch", issues=issues)


def sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(BATCH_DIR.iterdir()):
            if _is_image_name(p.name) or p.name.endswith(".csv"):
                zf.write(p, p.name)
    return buf.getvalue()


# --- export ---------------------------------------------------------------------------------------
def _cell(value):
    """Neutralise spreadsheet formulas: label text and CSV values are untrusted ("=HYPERLINK(...)")."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def export_items(job: BatchJob, ids: set[str] | None = None, statuses: set[str] | None = None) -> list[BatchItem]:
    """The rows an export covers: the whole job, ``ids`` (application ids), or ``statuses``. The status
    ``CLEARED`` adds the labels the agent cleared (every question answered yes), so "ready to approve"
    is ``PASS,CLEARED``."""
    def wanted(it: BatchItem) -> bool:
        return it.status in statuses or ("CLEARED" in statuses and it.decision == "pass")
    return [it for it in job.items
            if (ids is None or it.application_id in ids) and (statuses is None or wanted(it))]


def export_name(job: BatchJob, ids: set[str] | None, statuses: set[str] | None, ext: str = "csv") -> str:
    scope = "-".join(sorted(s.lower() for s in statuses)) if statuses else ("selected" if ids is not None else "all")
    return f"label-check_{job.slug}_{scope}_{job.created_at.astimezone():%Y-%m-%d}.{ext}"


def export_csv(job: BatchJob, ids: set[str] | None = None, statuses: set[str] | None = None,
               include: set[str] | None = None) -> str:
    """CSV of the job; ``ids`` (application ids) or ``statuses`` (PASS/REVIEW/FAIL/ERROR) narrow it;
    ``include`` picks the column groups (fields, warning, decisions, ocr, timings)."""
    inc = EXPORT_DEFAULT if include is None else {i for i in include if i in EXPORT_INCLUDE}
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["application_id", "image", "overall", "needs", "summary"]
    if "fields" in inc:
        for f in FIELDS:
            header += [f"{f.key}_verdict", f"{f.key}_expected", f"{f.key}_found", f"{f.key}_note"]
    if "warning" in inc:
        header += ["warning_present", "warning_wording", "warning_heading_caps", "warning_heading_bold", "warning_notes",
                   "warning_diff"]
    if "decisions" in inc:
        header += ["decision", "answers"]
    if "ocr" in inc:
        header += ["label_text"]
    if "timings" in inc:
        header += ["read_ms", "total_ms"]
    header += ["reader", "error"]
    w.writerow(header)
    for it in export_items(job, ids, statuses):
        r = it.result
        row: list = [it.application_id, it.image_name, it.status, it.needs, r.summary if r else (it.error or "")]
        if "fields" in inc:
            by_key = {f.key: f for f in r.fields} if r else {}
            for f in FIELDS:
                fr = by_key.get(f.key)
                row += [fr.verdict.value, fr.expected, fr.found or "", fr.note] if fr else ["", "", "", ""]
        if "warning" in inc:
            if r:
                wn = r.warning
                notes = " | ".join(n for n in (wn.wording_note, wn.heading_caps_note, wn.heading_bold_note) if n)
                diff = "; ".join(f"required '{d.expected}' / label '{d.found}'" for d in (wn.diff or []))
                row += ["yes" if wn.present else "no", wn.wording.value, wn.heading_caps.value, wn.heading_bold.value,
                        notes, diff]
            else:
                row += ["", "", "", "", "", ""]
        if "decisions" in inc:
            row += [it.decision, it.answers_text()]
        if "ocr" in inc:
            row += [r.ocr_text if r else ""]
        if "timings" in inc:
            row += [r.timings.read_ms, r.timings.total_ms] if r else ["", ""]
        row += [r.reader if r else "", "" if r else (it.error or "")]
        w.writerow([_cell(v) for v in row])
    return buf.getvalue()
