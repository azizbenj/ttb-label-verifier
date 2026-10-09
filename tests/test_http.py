"""HTTP behaviour that needs no OCR: concurrency, error mapping, upload limits, health."""

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.readers.base import LabelReading, OCRResult, ReaderError

client = TestClient(main.app)
OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL", "sample": "old_tom_clean"}
LINES = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL"]


class SlowReader:
    name = "slow"

    def read(self, image):
        time.sleep(0.5)  # stands in for a second of Tesseract CPU time
        return LabelReading(ocr=OCRResult(text="\n".join(LINES), lines=LINES, engine="slow"))


class BrokenReader:
    name = "broken"

    def __init__(self, error):
        self.error = error

    def read(self, image):
        raise self.error


def test_single_checks_do_not_block_each_other(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", SlowReader())

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            t0 = time.perf_counter()
            responses = await asyncio.gather(*(c.post("/api/verify", data=OLD_TOM) for _ in range(3)))
            return time.perf_counter() - t0, responses

    elapsed, responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 1.2, f"three 0.5 s checks took {elapsed:.2f} s: they ran one after another"


@pytest.mark.parametrize("error, status", [(ReaderError("The cloud reader is busy right now."), 502),
                                           (KeyError("boom"), 500)])
def test_reader_failures_are_reported_not_raised(monkeypatch, error, status):
    monkeypatch.setitem(main._readers, "tesseract", BrokenReader(error))
    r = client.post("/api/verify", data=OLD_TOM)
    assert r.status_code == status and "error" in r.json()
    r = client.post("/verify", data=OLD_TOM)
    assert r.status_code == status and "We couldn" in r.text
    if status == 502:
        assert "busy" in r.text


def test_bad_psm_is_a_400_with_a_message():
    for psm in ("abc", "4+99"):
        r = client.post("/api/verify", data={**OLD_TOM, "psm": psm})
        assert r.status_code == 400 and "psm" in r.json()["error"]


def test_oversized_upload_is_refused_before_reading_it():
    r = client.post("/verify", data=OLD_TOM, headers={"content-length": str(500 * 1024 * 1024)})
    assert r.status_code == 413 and "larger than" in r.text


def test_sample_name_must_match_exactly():
    assert main._SAFE_NAME.fullmatch("old_tom_clean\n") is None
    r = client.post("/api/verify", data={**OLD_TOM, "sample": "../samples"})
    assert r.status_code == 400 and "does not exist" in r.json()["error"]
    assert client.get("/samples/..%2Fsamples.png").status_code == 404


def test_missing_tesseract_binary_is_a_readable_error(monkeypatch):
    from app.readers import tesseract as T

    def missing(*a, **k):
        raise T.pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(T.pytesseract, "image_to_data", missing)
    monkeypatch.setitem(main._readers, "tesseract", T.TesseractReader())
    r = client.post("/api/verify", data=OLD_TOM)
    assert r.status_code == 502 and "not installed" in r.json()["error"]


def test_healthz_is_503_when_the_default_reader_cannot_run(monkeypatch):
    monkeypatch.setattr(main, "tesseract_version", lambda: None)
    r = client.get("/healthz")
    assert r.status_code == 503 and r.json()["status"] == "degraded"
