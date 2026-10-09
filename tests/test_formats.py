"""Every file type the upload zones advertise is accepted and read the same: PNG, JPG, TIFF, WEBP, BMP, a zip."""

import io
import re
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

from .conftest import requires_tesseract

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "old_tom_clean.png"
client = TestClient(app)
OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}


def encoded(fmt: str) -> bytes:
    img = Image.open(SAMPLE).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 92} if fmt in ("JPEG", "WEBP") else {}))
    return buf.getvalue()


@requires_tesseract
@pytest.mark.parametrize("fmt,ext,mime", [("JPEG", "jpg", "image/jpeg"), ("TIFF", "tif", "image/tiff"),
                                          ("WEBP", "webp", "image/webp"), ("BMP", "bmp", "image/bmp")])
def test_single_label_accepts_every_advertised_format(fmt, ext, mime):
    r = client.post("/api/verify", data=OLD_TOM, files={"image": (f"label.{ext}", encoded(fmt), mime)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall"] == "PASS", body["summary"]
    assert all(f["verdict"] in ("MATCH", "SKIPPED") for f in body["fields"])


def test_wrong_extension_is_judged_by_content_not_name():
    r = client.post("/api/verify", data=OLD_TOM, files={"image": ("label.png", b"%PDF-1.4 not an image", "image/png")})
    assert r.status_code == 400 and "image" in r.json()["error"].lower()


@requires_tesseract
def test_batch_accepts_a_zip_of_mixed_formats():
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("lot1/a.jpg", encoded("JPEG"))
        zf.writestr("lot1/b.webp", encoded("WEBP"))
        zf.writestr("notes.txt", b"ignore me")
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 "a.jpg,Z1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "b.webp,Z2,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n").encode()
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv")),
                                     ("files", ("labels.zip", zbuf.getvalue(), "application/zip"))],
                    headers={"X-Partial": "1"})
    assert r.status_code == 200, r.text
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    for _ in range(300):
        j = client.get(f"/api/batch/{job_id}").json()
        if j["finished"]:
            break
        time.sleep(0.2)
    assert j["finished"] and {it["status"] for it in j["items"]} == {"PASS"}
    assert any("notes.txt" in i for i in j["issues"])
