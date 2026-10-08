"""Batch and single-label HTTP flows with a fake reader, so they run without Tesseract."""

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.readers.base import LabelReading, OCRResult

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
OLD_TOM_LINES = [
    "OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)", "750 mL",
    "Distilled and Bottled by Old Tom Distillery, Bardstown, Kentucky 40004",
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic",
    "beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic",
    "beverages impairs your ability to drive a car or operate machinery, and may cause health problems.",
]


class FakeReader:
    """Always 'reads' the Old Tom label; the warning bold check is unmeasurable (no boxes) -> REVIEW."""

    name = "fake"

    def read(self, image):
        return LabelReading(ocr=OCRResult(text="\n".join(OLD_TOM_LINES), lines=OLD_TOM_LINES, engine="fake"))


@pytest.fixture(autouse=True)
def fake_reader(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", FakeReader())
    yield


client = TestClient(main.app)
OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}


def _wait_done(job_id):
    for _ in range(100):
        r = client.get(f"/batch/{job_id}")
        if 'data-status="done"' in r.text:
            return r
        time.sleep(0.05)
    raise AssertionError("batch did not finish")


def test_single_verify_renders_result_card():
    r = client.post("/verify", data={**OLD_TOM, "sample": "old_tom_clean"})
    assert r.status_code == 200
    assert "Needs a quick look" in r.text and "Checked in" in r.text and "Government warning" in r.text


def test_sample_batch_runs_and_renders_table_and_export():
    r = client.post("/batch", data={"sample": "1"})
    assert r.status_code == 200, r.text
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    r = _wait_done(job_id)
    assert "250 labels checked" in r.text and "Download results (CSV)" in r.text
    csv_out = client.get(f"/batch/{job_id}/export.csv").text
    assert csv_out.count("\n") == 251
    detail = client.get(f"/batch/{job_id}/item/0").text
    assert "Checked in" in detail


def test_uploaded_batch_reports_missing_and_stray_files():
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 "old_tom_clean.png,A1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "missing.png,A3,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "old_tom_clean.png,A4,,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n").encode()
    files = [("csv_file", ("apps.csv", csv_bytes, "text/csv")),
             ("files", ("old_tom_clean.png", (SAMPLES / "old_tom_clean.png").read_bytes(), "image/png")),
             ("files", ("extra.png", (SAMPLES / "wrong_abv.png").read_bytes(), "image/png")),
             ("files", ("readme.txt", b"not an image", "text/plain"))]
    r = client.post("/batch", files=files)
    assert r.status_code == 200, r.text
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    r = _wait_done(job_id)
    assert "No image named" in r.text and "missing.png" in r.text
    assert "Row 4: brand name is blank" in r.text
    assert "extra.png" in r.text and "readme.txt" in r.text
    j = client.get(f"/api/batch/{job_id}").json()
    by_id = {it["application_id"]: it for it in j["items"]}
    assert by_id["A1"]["status"] == "REVIEW" and by_id["A3"]["status"] == "ERROR" and by_id["A4"]["status"] == "ERROR"
