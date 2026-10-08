"""FastAPI service: one screen (single label + batch), HTML partials for the UI, JSON for the API."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from . import batch as batchmod
from .config import (CLAUDE_MODEL, CLOUD_READER_AVAILABLE, FIELDS, MANDATED_WARNING, MAX_IMAGE_BYTES, OCR_ENGINE,
                     THRESHOLDS)
from .models import Application, Status, Verdict
from .normalize import parse_alcohol, parse_net_contents
from .pipeline import verify
from .readers.base import LabelReader
from .readers.tesseract import TesseractReader, parse_psm, tesseract_version

log = logging.getLogger("labelcheck")
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "data" / "samples"
_SAFE_NAME = re.compile(r"^[a-z0-9_\-]+$")

app = FastAPI(title="TTB Label Check", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

_CHIP = {Verdict.MATCH: "ok", Verdict.NEAR_MATCH: "warn", Verdict.MISMATCH: "bad", Verdict.NOT_FOUND: "missing",
         Verdict.SKIPPED: "skip", Status.PASS: "ok", Status.REVIEW: "warn", Status.FAIL: "bad", Status.ERROR: "bad"}
_SHORT = {Verdict.MATCH: "Match", Verdict.NEAR_MATCH: "Review", Verdict.MISMATCH: "Mismatch",
          Verdict.NOT_FOUND: "Not found", Verdict.SKIPPED: "—", Status.PASS: "Pass", Status.REVIEW: "Review",
          Status.FAIL: "Fail", Status.ERROR: "Error"}
_HEADLINE = {Status.PASS: "Label matches the application", Status.REVIEW: "Needs a quick look",
             Status.FAIL: "Problems found", Status.ERROR: "Could not check"}
templates.env.filters["chip"] = lambda v: _CHIP.get(v, "skip")
templates.env.filters["short"] = lambda v: _SHORT.get(v, str(v))
templates.env.filters["headline"] = lambda v: _HEADLINE.get(v, "")
templates.env.filters["seconds"] = lambda ms: f"{(ms or 0) / 1000:.1f} s"


class UserError(Exception):
    """A message we can show the agent as-is."""


# --- readers --------------------------------------------------------------------------------------
_readers: dict[str, LabelReader] = {}


def get_reader(name: str | None, psm: str | None = None) -> LabelReader:
    name = (name or OCR_ENGINE).strip().lower()
    if psm and name != "claude":  # benchmarking knob, JSON API only: "4" or "4+11"
        first, extra = parse_psm(psm)
        return TesseractReader(psm=first, extra_psm=extra)
    if name == "claude":
        if not CLOUD_READER_AVAILABLE:
            raise UserError("The cloud reader is not enabled on this server. Using local OCR is the default.")
        if "claude" not in _readers:
            from .readers.claude_vision import ClaudeVisionReader
            _readers["claude"] = ClaudeVisionReader()
        return _readers["claude"]
    if "tesseract" not in _readers:
        _readers["tesseract"] = TesseractReader()
    return _readers["tesseract"]


def reader_info() -> str:
    parts = [f"Local OCR: Tesseract {tesseract_version() or 'not installed'}"]
    if CLOUD_READER_AVAILABLE:
        parts.append(f"Cloud reader: {CLAUDE_MODEL}")
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


def load_samples() -> list[dict]:
    path = SAMPLES_DIR / "samples.csv"
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row["image"]).stem
            out.append({"name": name, "title": _SAMPLE_TITLES.get(name, name.replace("_", " ")),
                        "expected": row.get("expected_overall", ""),
                        **{k: row.get(k, "") for k in ("brand_name", "class_type", "alcohol_content", "net_contents",
                                                       "bottler_name_address", "country_of_origin",
                                                       "application_id")}})
    return out


SAMPLES = load_samples()


# --- helpers ---------------------------------------------------------------------------------------
async def read_image(upload: UploadFile | None, sample: str) -> tuple[Image.Image, str]:
    if sample:
        if not _SAFE_NAME.match(sample) or not (SAMPLES_DIR / f"{sample}.png").exists():
            raise UserError("That sample does not exist.")
        return Image.open(SAMPLES_DIR / f"{sample}.png"), f"{sample}.png"
    if upload is None or not upload.filename:
        raise UserError("Please choose a label image (PNG, JPG, TIFF or WEBP), or pick a sample.")
    data = await upload.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise UserError(f"That image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB. Please use a smaller file.")
    if not data:
        raise UserError("The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise UserError(f"'{upload.filename}' could not be read as an image. Please upload a PNG or JPG of the label.") from None
    return image, upload.filename


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


def error_response(request: Request, message: str, status: int = 400) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/error.html", {"message": message}, status_code=status)


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
    v = tesseract_version()
    import os
    return {"status": "ok" if v else "degraded", "tesseract": v, "cloud_reader": CLOUD_READER_AVAILABLE,
            "default_reader": OCR_ENGINE, "deployment": os.getenv("RAILWAY_DEPLOYMENT_ID")}


@app.get("/samples/{name}.png")
def sample_image(name: str):
    if not _SAFE_NAME.match(name) or not (SAMPLES_DIR / f"{name}.png").exists():
        return Response(status_code=404)
    return FileResponse(SAMPLES_DIR / f"{name}.png", media_type="image/png")


# --- single label ---------------------------------------------------------------------------------
async def _verify_from_form(brand_name, class_type, alcohol_content, net_contents, bottler_name_address,
                            country_of_origin, application_id, sample, reader, image, psm: str | None = None):
    app_data = build_application(brand_name=brand_name, class_type=class_type, alcohol_content=alcohol_content,
                                 net_contents=net_contents, bottler_name_address=bottler_name_address,
                                 country_of_origin=country_of_origin, application_id=application_id)
    img, name = await read_image(image, sample.strip())
    return verify(app_data, img, get_reader(reader, psm), image_name=name)


@app.post("/verify", response_class=HTMLResponse)
async def verify_html(request: Request, brand_name: str = Form(""), class_type: str = Form(""),
                      alcohol_content: str = Form(""), net_contents: str = Form(""),
                      bottler_name_address: str = Form(""), country_of_origin: str = Form(""),
                      application_id: str = Form(""), sample: str = Form(""), reader: str = Form(""),
                      image: UploadFile | None = File(None)):
    try:
        result = await _verify_from_form(brand_name, class_type, alcohol_content, net_contents, bottler_name_address,
                                         country_of_origin, application_id, sample, reader, image)
    except UserError as e:
        return error_response(request, str(e))
    except Exception:
        log.exception("verify failed")
        return error_response(request, "Something went wrong while reading this label. Please try another image.", 500)
    return templates.TemplateResponse(request, "partials/result.html", {"result": result})


@app.post("/api/verify")
async def verify_json(brand_name: str = Form(""), class_type: str = Form(""), alcohol_content: str = Form(""),
                      net_contents: str = Form(""), bottler_name_address: str = Form(""),
                      country_of_origin: str = Form(""), application_id: str = Form(""), sample: str = Form(""),
                      reader: str = Form(""), image: UploadFile | None = File(None), psm: str | None = Form(None)):
    try:
        result = await _verify_from_form(brand_name, class_type, alcohol_content, net_contents, bottler_name_address,
                                         country_of_origin, application_id, sample, reader, image, psm)
    except UserError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(result.model_dump(mode="json"))


# --- batch -----------------------------------------------------------------------------------------
def _batch_status_response(request: Request, job: batchmod.BatchJob) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/batch_status.html", {"job": job, "fields": FIELDS})


@app.post("/batch", response_class=HTMLResponse)
async def batch_start(request: Request, csv_file: UploadFile | None = File(None),
                      files: list[UploadFile] = File([]), reader: str = Form(""), sample: str = Form("")):
    try:
        rd = get_reader(reader)
        if sample:
            job = batchmod.start_sample_job(rd)
        else:
            if csv_file is None or not csv_file.filename:
                raise UserError("Please choose the CSV of application data.")
            rows, issues = batchmod.parse_applications_csv(await csv_file.read())
            uploads = [(f.filename, await f.read()) for f in files if f.filename]
            if not uploads:
                raise UserError("Please add the label images (select the files, or a zip of them).")
            images, more = batchmod.collect_images(uploads)
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
    return templates.TemplateResponse(request, "partials/result.html", {"result": item.result, "compact": True})


@app.get("/batch/{job_id}/export.csv")
def batch_export(request: Request, job_id: str):
    job = batchmod.JOBS.get(job_id)
    if job is None:
        return Response("Batch not found", status_code=404)
    return Response(batchmod.export_csv(job), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=label_check_{job.id}.csv"})


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
