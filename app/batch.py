"""Batch mode: a CSV of applications plus their label images, processed in a thread pool.

Jobs live in memory for the life of the process (this is a prototype; see README).
"""

from __future__ import annotations

import csv
import io
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from PIL import Image, UnidentifiedImageError

from .config import (BATCH_WORKERS, FIELDS, IMAGE_EXTENSIONS, MAX_BATCH_IMAGES, MAX_IMAGE_BYTES, MAX_ZIP_MEMBERS,
                     MAX_ZIP_UNCOMPRESSED)
from .models import Application, Status, VerificationResult
from .pipeline import verify
from .readers.base import LabelReader

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
    """A problem the user can fix, phrased for them."""


@dataclass
class BatchItem:
    row: int
    application_id: str
    image_name: str
    application: Application | None = None
    result: VerificationResult | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return Status.ERROR.value
        return self.result.overall.value if self.result else "PENDING"


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


JOBS: dict[str, BatchJob] = {}
_executor = ThreadPoolExecutor(max_workers=BATCH_WORKERS, thread_name_prefix="batch")


# --- CSV ------------------------------------------------------------------------------------------
def _norm_header(h: str) -> str:
    return (h or "").strip().lower().replace("﻿", "").replace("-", "_").replace(" ", "_").replace("/", "_")


def parse_applications_csv(data: bytes) -> tuple[list[dict], list[str]]:
    """Return (rows, issues). Each row has canonical keys; blank optional values are ''."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
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
        labels = {f.key: f.label for f in FIELDS}
        labels["image"] = "image"
        raise BatchError("The CSV is missing the column(s): " + ", ".join(labels.get(m, m) for m in missing)
                         + ". Download the template to see the expected columns.")
    rows, issues = [], []
    for i, raw_row in enumerate(reader, start=2):
        row = {key: (raw_row.get(col) or "").strip() for key, col in mapping.items()}
        for key in COLUMN_ALIASES:
            row.setdefault(key, "")
        if not any(row.values()):
            continue
        row["_row"] = i
        rows.append(row)
    if not rows:
        raise BatchError("The CSV has a header but no application rows.")
    if len(rows) > MAX_BATCH_IMAGES:
        raise BatchError(f"That CSV has {len(rows)} rows. Please split batches at {MAX_BATCH_IMAGES} labels.")
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


def collect_images(uploads: list[tuple[str, bytes]]) -> tuple[dict[str, bytes], list[str]]:
    """Accept loose image files and/or zip archives. Keys are lower-cased base names."""
    images: dict[str, bytes] = {}
    issues: list[str] = []
    for name, data in uploads:
        base = Path(name).name
        if base.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir() and not Path(m.filename).name.startswith(".")]
                    if len(members) > MAX_ZIP_MEMBERS:
                        raise BatchError(f"'{base}' holds {len(members)} files; the limit is {MAX_ZIP_MEMBERS}.")
                    if sum(m.file_size for m in members) > MAX_ZIP_UNCOMPRESSED:
                        raise BatchError(f"'{base}' is too large when unpacked.")
                    for m in members:
                        mb = Path(m.filename).name
                        if not _is_image_name(mb):
                            if not mb.lower().endswith(".csv"):
                                issues.append(f"Skipped '{m.filename}' inside {base}: not an image.")
                            continue
                        if m.file_size > MAX_IMAGE_BYTES:
                            issues.append(f"Skipped '{mb}': larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
                            continue
                        images[mb.lower()] = zf.read(m)
            except zipfile.BadZipFile:
                raise BatchError(f"'{base}' is not a valid zip file.") from None
        elif _is_image_name(base):
            if len(data) > MAX_IMAGE_BYTES:
                issues.append(f"Skipped '{base}': larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
                continue
            images[base.lower()] = data
        else:
            issues.append(f"Skipped '{base}': not an image (use PNG, JPG, TIFF, BMP or WEBP).")
    if len(images) > MAX_BATCH_IMAGES:
        raise BatchError(f"{len(images)} images were uploaded. Please split batches at {MAX_BATCH_IMAGES} labels.")
    return images, issues


def _lookup_image(images: dict[str, bytes], name: str) -> bytes | None:
    key = Path(name).name.lower()
    if key in images:
        return images[key]
    stem = Path(key).stem
    for k in images:  # allow the CSV to omit the extension
        if Path(k).stem == stem:
            return images[k]
    return None


# --- jobs ------------------------------------------------------------------------------------------
def _row_to_application(row: dict) -> tuple[Application | None, str | None]:
    missing = [f.label.lower() for f in FIELDS if f.required and not row.get(f.key)]
    if missing:
        return None, f"Row {row['_row']}: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} blank."
    return Application(**{k: row.get(k, "") for k in COLUMN_ALIASES if k != "image"}), None


def build_items(rows: list[dict], images: dict[str, bytes]) -> tuple[list[BatchItem], list[tuple[BatchItem, bytes]], list[str]]:
    items, work, issues = [], [], []
    used: set[str] = set()
    for row in rows:
        item = BatchItem(row=row["_row"], application_id=row.get("application_id") or f"row {row['_row']}",
                         image_name=row.get("image", ""))
        app, err = _row_to_application(row)
        item.application = app
        data = _lookup_image(images, item.image_name) if item.image_name else None
        if err:
            item.error = err
        elif not item.image_name:
            item.error = f"Row {row['_row']}: no image file name."
        elif data is None:
            item.error = f"No image named '{item.image_name}' was uploaded."
        else:
            used.add(Path(item.image_name).name.lower())
            work.append((item, data))
        items.append(item)
    for name in images:
        if name not in used and not any(Path(name).stem == Path(u).stem for u in used):
            issues.append(f"Image '{name}' has no matching row in the CSV, so it was not checked.")
    return items, work, issues


def _process(job: BatchJob, item: BatchItem, data: bytes, reader: LabelReader) -> None:
    try:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise BatchError(f"'{item.image_name}' could not be read as an image.") from None
        item.result = verify(item.application, image, reader, image_name=item.image_name)
    except BatchError as e:
        item.error = str(e)
    except Exception as e:  # keep the batch going; surface the reason
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
def export_csv(job: BatchJob) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["application_id", "image", "overall", "summary"]
    for f in FIELDS:
        header += [f"{f.key}_verdict", f"{f.key}_expected", f"{f.key}_found", f"{f.key}_note"]
    header += ["warning_present", "warning_wording", "warning_heading_caps", "warning_heading_bold", "warning_notes",
               "read_ms", "total_ms", "reader", "error"]
    w.writerow(header)
    for it in job.items:
        r = it.result
        row = [it.application_id, it.image_name, it.status, r.summary if r else (it.error or "")]
        if r:
            by_key = {f.key: f for f in r.fields}
            for f in FIELDS:
                fr = by_key.get(f.key)
                row += [fr.verdict.value, fr.expected, fr.found or "", fr.note] if fr else ["", "", "", ""]
            wn = r.warning
            notes = " | ".join(n for n in (wn.wording_note, wn.heading_caps_note, wn.heading_bold_note) if n)
            row += ["yes" if wn.present else "no", wn.wording.value, wn.heading_caps.value, wn.heading_bold.value,
                    notes, r.timings.read_ms, r.timings.total_ms, r.reader, ""]
        else:
            row += ["", "", "", ""] * len(FIELDS) + ["", "", "", "", "", "", "", "", it.error or ""]
        w.writerow(row)
    return buf.getvalue()
