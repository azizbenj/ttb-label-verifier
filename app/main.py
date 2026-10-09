"""FastAPI service: one screen (single label + batch), HTML partials for the UI, JSON for the API."""

from __future__ import annotations

import base64
import csv
import io
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from PIL import Image, ImageOps

from . import batch as batchmod
from .evidence import bold_meter_percent, evidence_for, pin_labels
from .config import (CLAUDE_MODEL, CLOUD_READER_AVAILABLE, FIELDS, MANDATED_WARNING, MAX_BATCH_UPLOAD_BYTES,
                     MAX_IMAGE_BYTES, OCR_ENGINE, THRESHOLDS)
from .images import MAX_LABEL_PARTS, ImageError, open_image, stitch
from .models import Application, Status, VerificationResult, Verdict
from .normalize import parse_alcohol, parse_net_contents
from .pipeline import verify
from .readers.base import LabelReader, ReaderError
from .readers.tesseract import TesseractReader, parse_psm, tesseract_version

log = logging.getLogger("labelcheck")
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "data" / "samples"
_SAFE_NAME = re.compile(r"[a-z0-9_\-]+")  # used with fullmatch: "$" would also accept a trailing newline

app = FastAPI(title="TTB Label Check", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Verdict and status -> design token class (pass / review / fail / none). NOT FOUND shares FAIL colours.
_CHIP = {Verdict.MATCH: "pass", Verdict.NEAR_MATCH: "review", Verdict.MISMATCH: "fail", Verdict.NOT_FOUND: "fail",
         Verdict.SKIPPED: "none", Status.PASS: "pass", Status.REVIEW: "review", Status.FAIL: "fail", Status.ERROR: "fail"}
# Glyphs are fixed per verdict so colour is never the only signal.
_GLYPH = {Verdict.MATCH: "check", Verdict.NEAR_MATCH: "bang", Verdict.MISMATCH: "cross", Verdict.NOT_FOUND: "question",
          Verdict.SKIPPED: "dash", Status.PASS: "check", Status.REVIEW: "bang", Status.FAIL: "cross", Status.ERROR: "cross"}
_ICON_PATHS = {
    "check": ('0 0 16 16', '<path d="M3 8.5l3 3 7-7"/>', 2.5),
    "bang": ('0 0 16 16', '<path d="M8 3v6"/><path d="M8 12.5h.01"/>', 2.5),
    "cross": ('0 0 16 16', '<path d="M4 4l8 8M12 4l-8 8"/>', 2.5),
    "question": ('0 0 16 16', '<path d="M5.5 6.2a2.5 2.5 0 1 1 3.7 2.2C8.4 8.8 8 9.3 8 10"/><path d="M8 12.6h.01"/>', 2.2),
    "dash": ('0 0 16 16', '<path d="M4 8h8"/>', 2.5),
    "lock": ('0 0 24 24', '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>', 2.4),
    "mark": ('0 0 24 24', '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6"/><path d="M9 12h6"/><path d="M9.5 16.5l1.8 1.8 3.2-3.6"/>', 2.2),
    "help": ('0 0 24 24', '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.6 2.2c-.7.4-1.1.9-1.1 1.8"/><path d="M12 17h.01"/>', 2),
    "upload": ('0 0 24 24', '<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 20h16"/>', 2),
    "download": ('0 0 24 24', '<path d="M12 4v12"/><path d="M7 11l5 5 5-5"/><path d="M4 20h16"/>', 2),
    "info": ('0 0 24 24', '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>', 2),
    "alert": ('0 0 24 24', '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/>', 2),
    "print": ('0 0 24 24', '<path d="M6 9V3h12v6"/><rect x="4" y="9" width="16" height="8" rx="1.5"/><path d="M6 21h12v-6H6z"/>', 2),
    "copy": ('0 0 24 24', '<rect x="8" y="3" width="12" height="14" rx="2"/><path d="M4 7v12a2 2 0 0 0 2 2h10"/>', 2),
    "expand": ('0 0 24 24', '<path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/>', 2),
    "search": ('0 0 24 24', '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>', 2),
    "play": ('0 0 24 24', '<path d="M5 3l14 9-14 9z"/>', 2),
    "csv": ('0 0 24 24', '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M8 13h8M8 17h8"/>', 2),
    "images": ('0 0 24 24', '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15l5-5 4 4 3-3 6 6"/><circle cx="16" cy="9" r="1.5"/>', 2),
}


def icon(name: str, size: int | None = None) -> Markup:
    """Inline SVG glyph. Without ``size`` it is a 14 px chip glyph (class "i")."""
    viewbox, paths, width = _ICON_PATHS[name]
    attrs = f'width="{size}" height="{size}"' if size else 'class="i"'
    return Markup(f'<svg {attrs} viewBox="{viewbox}" fill="none" stroke="currentColor" stroke-width="{width}" '
                  f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>')


def chip(v, cls_extra: str = "", word: str | None = None) -> Markup:
    """A verdict or status chip: glyph + word, with the fixed vocabulary."""
    text = word if word is not None else (v.value if hasattr(v, "value") else str(v))
    return Markup(f'<span class="chip {_CHIP.get(v, "none")} {cls_extra}">{icon(_GLYPH.get(v, "dash"))}{text}</span>')
_SHORT = {Verdict.MATCH: "Match", Verdict.NEAR_MATCH: "Review", Verdict.MISMATCH: "Mismatch",
          Verdict.NOT_FOUND: "Not found", Verdict.SKIPPED: "—", Status.PASS: "Pass", Status.REVIEW: "Review",
          Status.FAIL: "Fail", Status.ERROR: "Error"}
_HEADLINE = {Status.PASS: "Label matches the application", Status.REVIEW: "Needs a quick look",
             Status.FAIL: "Problems found", Status.ERROR: "Could not check"}
templates.env.filters["chip"] = lambda v: _CHIP.get(v, "none")
templates.env.filters["glyph"] = lambda v: _GLYPH.get(v, "dash")
templates.env.filters["short"] = lambda v: _SHORT.get(v, str(v))
templates.env.filters["headline"] = lambda v: _HEADLINE.get(v, "")
templates.env.filters["seconds"] = lambda ms: f"{(ms or 0) / 1000:.1f} s"
templates.env.globals.update(icon=icon, chip=chip, evidence_for=evidence_for, pin_labels=pin_labels,
                             bold_meter_percent=bold_meter_percent, thresholds=THRESHOLDS)


class UserError(Exception):
    """A message we can show the agent as-is."""


# --- readers --------------------------------------------------------------------------------------
_readers: dict[str, LabelReader] = {}


def get_reader(name: str | None, psm: str | None = None) -> LabelReader:
    name = (name or OCR_ENGINE).strip().lower()
    if psm and name != "claude":  # benchmarking knob, JSON API only: "4" or "4+11"
        try:
            first, extra = parse_psm(psm)
        except ValueError:
            raise UserError("psm must look like '4' or '4+11'.") from None
        if not all(0 <= m <= 13 for m in (first, extra if extra is not None else first)):
            raise UserError("psm values must be Tesseract page-segmentation modes 0-13.")
        return TesseractReader(psm=first, extra_psm=extra)
    if name == "claude":
        if not CLOUD_READER_AVAILABLE:
            raise UserError("The cloud reader is not enabled on this server. Please choose local OCR.")
        if "claude" not in _readers:
            from .readers.claude_vision import ClaudeVisionReader
            _readers["claude"] = ClaudeVisionReader()
        return _readers["claude"]
    if "tesseract" not in _readers:
        _readers["tesseract"] = TesseractReader()
    return _readers["tesseract"]


def reader_info() -> str:
    parts = [f"Local OCR · Tesseract {tesseract_version() or 'not installed'}"]
    if CLOUD_READER_AVAILABLE:
        parts.append(f"Cloud reader · {CLAUDE_MODEL}")
    return " · ".join(parts)


# --- samples --------------------------------------------------------------------------------------
_SAMPLE_TITLES = {
    "old_tom_clean": "Old Tom Distillery (the README sample, should pass)",
    "stones_throw_clean": "Stone's Throw Cellars (wine, should pass)",
    "river_bend_clean": "River Bend Brewing (beer, should pass)",
    "glen_morar_import_clean": "Glen Morar (import with country of origin, should pass)",
    "wrong_abv": "Wrong alcohol content",
    "missing_warning": "Government warning missing",
    "warning_not_caps": "Warning heading not in capitals",
    "warning_not_bold": "Warning heading not bold",
    "brand_case": "Brand name capitalization differs",
    "wrong_net_contents": "Wrong net contents",
    "wrong_brand": "Different brand name on the label",
    "warning_text_altered": "One word of the warning changed",
    "warning_truncated": "Warning statement cut short",
    "missing_net_contents": "Net contents missing from the label",
    "wrong_country": "Wrong country of origin",
}


# Gallery tiles: (short title, one-line kind, group). Clean samples are the four tiles; the rest sit
# behind "labels with a planted problem", grouped by what they demonstrate.
_SAMPLE_TILES = {
    "old_tom_clean": ("Old Tom Distillery", "Bourbon · should pass", "clean"),
    "stones_throw_clean": ("Stone's Throw Cellars", "Wine · should pass", "clean"),
    "river_bend_clean": ("River Bend Brewing", "Beer · should pass", "clean"),
    "glen_morar_import_clean": ("Glen Morar", "Import · country of origin", "clean"),
    "wrong_abv": ("Wrong alcohol content", "Old Tom · 40% printed, 45% filed", "fields"),
    "brand_case": ("Brand capitalization differs", "Stone's Throw · needs a look", "fields"),
    "wrong_net_contents": ("Wrong net contents", "Old Tom · 1 L printed, 750 mL filed", "fields"),
    "wrong_brand": ("Different brand on the label", "River Bend printed, Copper Kettle filed", "fields"),
    "missing_net_contents": ("Net contents missing", "River Bend · nothing printed", "fields"),
    "wrong_country": ("Wrong country of origin", "Glen Morar · Ireland printed", "fields"),
    "missing_warning": ("Warning missing", "River Bend · no statement", "warning"),
    "warning_not_caps": ("Heading not in capitals", "Stone's Throw · 'Government Warning:'", "warning"),
    "warning_not_bold": ("Heading not bold", "Old Tom · regular weight", "warning"),
    "warning_text_altered": ("One word changed", "Glen Morar · 'can' for 'may'", "warning"),
    "warning_truncated": ("Statement cut short", "Stone's Throw · second sentence missing", "warning"),
}


def load_samples() -> list[dict]:
    path = SAMPLES_DIR / "samples.csv"
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row["image"]).stem
            short, kind, group = _SAMPLE_TILES.get(name, (name.replace("_", " "), "", "fields"))
            out.append({"name": name, "title": _SAMPLE_TITLES.get(name, name.replace("_", " ")),
                        "expected": row.get("expected_overall", ""), "short": short, "kind": kind, "group": group,
                        "clean": group == "clean",
                        **{k: row.get(k, "") for k in ("brand_name", "class_type", "alcohol_content", "net_contents",
                                                       "bottler_name_address", "country_of_origin",
                                                       "application_id")}})
    return out


SAMPLES = load_samples()


# --- helpers ---------------------------------------------------------------------------------------
async def read_image(uploads: list[UploadFile] | UploadFile | None, sample: str) -> tuple[Image.Image, str]:
    """The label image, or several (front, back, neck) stacked into one so the label is read whole."""
    if sample:
        if not _SAFE_NAME.fullmatch(sample) or not (SAMPLES_DIR / f"{sample}.png").exists():
            raise UserError("That sample does not exist.")
        return Image.open(SAMPLES_DIR / f"{sample}.png"), f"{sample}.png"
    if isinstance(uploads, UploadFile) or uploads is None:
        uploads = [uploads] if uploads is not None else []
    uploads = [u for u in uploads if u is not None and u.filename]
    if not uploads:
        raise UserError("Please choose a label image (PNG, JPG, TIFF or WEBP), or pick a sample.")
    if len(uploads) > MAX_LABEL_PARTS:
        raise UserError(f"That is {len(uploads)} images for one label; up to {MAX_LABEL_PARTS} are accepted "
                        "(front, back, neck...). Use the batch tab for several applications.")
    images = []
    for upload in uploads:
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise UserError(f"'{upload.filename}' is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB. "
                            "Please use a smaller file.")
        try:
            images.append(open_image(data, upload.filename))
        except ImageError as e:
            raise UserError(str(e)) from None
    return stitch(images), " + ".join(u.filename for u in uploads)


def build_application(**values: str) -> Application:
    app_data = Application(**values)
    missing = [f.label for f in FIELDS if f.required and not getattr(app_data, f.key)]
    if missing:
        raise UserError("Please fill in: " + ", ".join(missing) + ".")
    if parse_alcohol(app_data.alcohol_content) is None:
        raise UserError("Alcohol content should look like '45% Alc./Vol.', '45%' or '90 Proof'.")
    if parse_net_contents(app_data.net_contents) is None:
        raise UserError("Net contents should look like '750 mL', '1.75 L' or '12 fl oz'.")
    return app_data


def error_response(request: Request, message: str, status: int = 400, title: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/error.html", {"message": message, "title": title},
                                      status_code=status)


PREVIEW_MAX_SIDE = 1000


def preview_data_url(image: Image.Image, skew: float = 0.0) -> str:
    """A downscaled JPEG of the label, inlined so the result can show it without storing the upload.
    Straightened by the same angle as the image the reader used, so the highlights line up."""
    img = ImageOps.exif_transpose(image)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if skew:
        img = img.convert("RGB").rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def result_response(request: Request, result: VerificationResult, application: Application | None,
                    image_url: str | None, *, compact: bool = False, full_page: bool = False) -> HTMLResponse:
    ctx = {"result": result, "application": application, "image_url": image_url, "compact": compact,
           "reader_info": reader_info(), "cloud": CLOUD_READER_AVAILABLE, "default_reader": OCR_ENGINE,
           "thresholds": THRESHOLDS, "warning_text": MANDATED_WARNING}
    return templates.TemplateResponse(request, "result_page.html" if full_page else "partials/result.html", ctx)


# Refuse oversized uploads from their Content-Length before the body is read (and spooled to disk).
_UPLOAD_LIMITS = {"/verify": MAX_IMAGE_BYTES + 1024 * 1024, "/api/verify": MAX_IMAGE_BYTES + 1024 * 1024,
                  "/batch": MAX_BATCH_UPLOAD_BYTES + 1024 * 1024}


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    limit = _UPLOAD_LIMITS.get(request.url.path) if request.method == "POST" else None
    length = request.headers.get("content-length", "")
    if limit and length.isdigit() and int(length) > limit:
        mb = (MAX_BATCH_UPLOAD_BYTES if request.url.path == "/batch" else MAX_IMAGE_BYTES) // (1024 * 1024)
        message = f"That upload is larger than {mb} MB. Please send smaller files" + (
            " or split the batch." if request.url.path == "/batch" else ".")
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=413)
        return error_response(request, message, 413)
    return await call_next(request)


# --- pages -----------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "fields": FIELDS, "samples": SAMPLES, "cloud": CLOUD_READER_AVAILABLE, "default_reader": OCR_ENGINE,
        "thresholds": THRESHOLDS, "warning_text": MANDATED_WARNING, "reader_info": reader_info(),
        "batch_sample": (batchmod.BATCH_DIR / "applications.csv").exists(),
    })


@app.get("/healthz")
def healthz():
    """503 when the default reader cannot run, so a health check does not route traffic to a broken container."""
    v = tesseract_version()
    ok = CLOUD_READER_AVAILABLE if OCR_ENGINE == "claude" else v is not None
    return JSONResponse({"status": "ok" if ok else "degraded", "tesseract": v, "cloud_reader": CLOUD_READER_AVAILABLE,
                         "default_reader": OCR_ENGINE, "batch_jobs": len(batchmod.JOBS),
                         "deployment": os.getenv("RAILWAY_DEPLOYMENT_ID")}, status_code=200 if ok else 503)


@app.get("/samples/{name}.png")
def sample_image(name: str):
    if not _SAFE_NAME.fullmatch(name) or not (SAMPLES_DIR / f"{name}.png").exists():
        return Response(status_code=404)
    return FileResponse(SAMPLES_DIR / f"{name}.png", media_type="image/png")


# --- single label ---------------------------------------------------------------------------------
async def _verify_from_form(brand_name, class_type, alcohol_content, net_contents, bottler_name_address,
                            country_of_origin, application_id, sample, reader, image, psm: str | None = None):
    app_data = build_application(brand_name=brand_name, class_type=class_type, alcohol_content=alcohol_content,
                                 net_contents=net_contents, bottler_name_address=bottler_name_address,
                                 country_of_origin=country_of_origin, application_id=application_id)
    img, name = await read_image(image, sample.strip())
    # OCR takes about a second of CPU: run it off the event loop so other agents, batch polling and
    # /healthz are not queued behind it.
    result = await run_in_threadpool(verify, app_data, img, get_reader(reader, psm), name)
    image_url = (f"/samples/{sample.strip()}.png" if sample.strip() and not result.skew_deg
                 else await run_in_threadpool(preview_data_url, img, result.skew_deg))
    return result, app_data, image_url


@app.post("/verify", response_class=HTMLResponse)
async def verify_html(request: Request, brand_name: str = Form(""), class_type: str = Form(""),
                      alcohol_content: str = Form(""), net_contents: str = Form(""),
                      bottler_name_address: str = Form(""), country_of_origin: str = Form(""),
                      application_id: str = Form(""), sample: str = Form(""), reader: str = Form(""),
                      image: list[UploadFile] = File([])):
    try:
        result, app_data, image_url = await _verify_from_form(brand_name, class_type, alcohol_content, net_contents,
                                                              bottler_name_address, country_of_origin, application_id,
                                                              sample, reader, image)
    except UserError as e:
        return error_response(request, str(e))
    except ReaderError as e:
        log.warning("reader failed: %s", e)
        return error_response(request, str(e), 502, title="We couldn't read this label.")
    except Exception:
        log.exception("verify failed")
        return error_response(request, "Something went wrong while reading this label. Please try another image.", 500)
    # The page's script asks for the partial; a plain form post (no JavaScript) gets a whole page.
    full_page = request.headers.get("x-partial") != "1"
    return result_response(request, result, app_data, image_url, full_page=full_page)


@app.post("/api/verify")
async def verify_json(brand_name: str = Form(""), class_type: str = Form(""), alcohol_content: str = Form(""),
                      net_contents: str = Form(""), bottler_name_address: str = Form(""),
                      country_of_origin: str = Form(""), application_id: str = Form(""), sample: str = Form(""),
                      reader: str = Form(""), image: list[UploadFile] = File([]), psm: str | None = Form(None)):
    try:
        result, _, _ = await _verify_from_form(brand_name, class_type, alcohol_content, net_contents,
                                               bottler_name_address, country_of_origin, application_id, sample,
                                               reader, image, psm)
    except UserError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except ReaderError as e:
        log.warning("reader failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception:
        log.exception("verify failed")
        return JSONResponse({"error": "Something went wrong while reading this label."}, status_code=500)
    return JSONResponse(result.model_dump(mode="json"))


# --- batch -----------------------------------------------------------------------------------------
def _batch_status_response(request: Request, job: batchmod.BatchJob) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/batch_status.html",
                                      {"job": job, "fields": FIELDS, "counts": job.counts()})


@app.post("/batch", response_class=HTMLResponse)
async def batch_start(request: Request, csv_file: UploadFile | None = File(None),
                      files: list[UploadFile] = File([]), reader: str = Form(""), sample: str = Form("")):
    try:
        rd = get_reader(reader)
        if sample:
            job = await run_in_threadpool(batchmod.start_sample_job, rd)
        else:
            if csv_file is None or not csv_file.filename:
                raise UserError("Please choose the CSV of application data.")
            rows, issues = batchmod.parse_applications_csv(await csv_file.read())
            uploads = [(f.filename, await f.read()) for f in files if f.filename]
            if not uploads:
                raise UserError("Please add the label images (select the files, or a zip of them).")
            images, more = await run_in_threadpool(batchmod.collect_images, uploads)
            if not images:
                raise UserError("None of the uploaded files were images. Use PNG, JPG, TIFF or WEBP, or a zip of them.")
            job = batchmod.start_job(rows, images, rd, source=csv_file.filename, issues=issues + more)
    except (UserError, batchmod.BatchError) as e:
        return error_response(request, str(e))
    except Exception:
        log.exception("batch start failed")
        return error_response(request, "Something went wrong starting this batch. Please check the files and try again.", 500)
    return _batch_status_response(request, job)


@app.get("/batch/template.csv")
def batch_template():
    return Response(batchmod.template_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=applications_template.csv"})


@app.get("/batch/sample.zip")
def batch_sample_zip():
    if not batchmod.BATCH_DIR.exists():
        return Response(status_code=404)
    return Response(batchmod.sample_zip(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=sample_batch.zip"})


@app.get("/batch/{job_id}", response_class=HTMLResponse)
def batch_status(request: Request, job_id: str):
    job = batchmod.JOBS.get(job_id)
    if job is None:
        return error_response(request, "That batch is no longer available (the server may have restarted). Please run it again.", 404)
    return _batch_status_response(request, job)


@app.get("/batch/{job_id}/item/{index}", response_class=HTMLResponse)
def batch_item(request: Request, job_id: str, index: int):
    job = batchmod.JOBS.get(job_id)
    if job is None or not (0 <= index < len(job.items)):
        return error_response(request, "That result is not available.", 404)
    item = job.items[index]
    if item.result is None:
        return error_response(request, item.error or "Still processing.")
    image_url = f"/batch/{job.id}/image/{index}" if getattr(item, "preview", None) else None
    return result_response(request, item.result, item.application, image_url, compact=True)


@app.get("/batch/{job_id}/image/{index}")
def batch_image(job_id: str, index: int):
    job = batchmod.JOBS.get(job_id)
    if job is None or not (0 <= index < len(job.items)) or not getattr(job.items[index], "preview", None):
        return Response(status_code=404)
    return Response(job.items[index].preview, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/batch/{job_id}/export.csv")
def batch_export(request: Request, job_id: str, ids: str = "", status: str = ""):
    """The whole batch, or ``?ids=APP-1,APP-2`` (selected rows), or ``?status=FAIL`` (one verdict)."""
    job = batchmod.JOBS.get(job_id)
    if job is None:
        return Response("Batch not found", status_code=404)
    id_set = {i.strip() for i in ids.split(",") if i.strip()} or None
    status_set = {s.strip().upper() for s in status.split(",") if s.strip()} or None
    suffix = f"-{status.lower()}" if status_set else ("-selected" if id_set else "")
    return Response(batchmod.export_csv(job, id_set, status_set), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=label-check{suffix}-{job.id}.csv"})


@app.get("/api/batch/{job_id}")
def batch_json(job_id: str):
    job = batchmod.JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": job.id, "done": job.done, "total": job.total, "finished": job.is_done, "counts": job.counts(),
            "elapsed_ms": round(job.elapsed_ms, 1), "issues": job.issues,
            "items": [{"row": it.row, "application_id": it.application_id, "image": it.image_name,
                       "status": it.status, "error": it.error,
                       "result": it.result.model_dump(mode="json") if it.result else None} for it in job.items]}
