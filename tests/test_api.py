from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

from .conftest import TIMING_BUDGET_MS, requires_tesseract

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
client = TestClient(app)

OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL",
           "bottler_name_address": "Old Tom Distillery, Bardstown, Kentucky 40004"}


def test_index_renders():
    r = client.get("/")
    assert r.status_code == 200 and "Check one label" in r.text and "Old Tom" in r.text


def test_healthz():
    from app.readers.tesseract import tesseract_version
    r = client.get("/healthz")
    assert r.status_code == (200 if tesseract_version() else 503) and "tesseract" in r.json()


def test_missing_fields_is_friendly():
    r = client.post("/verify", data={"brand_name": "X"}, files={"image": ("a.png", b"", "image/png")})
    assert r.status_code == 400 and "Please fill in" in r.text


def test_bad_alcohol_value_is_friendly():
    r = client.post("/verify", data={**OLD_TOM, "alcohol_content": "forty five"}, files={"image": ("a.png", b"x", "image/png")})
    assert r.status_code == 400 and "Alcohol content should look like" in r.text


def test_not_an_image_is_friendly():
    r = client.post("/verify", data=OLD_TOM, files={"image": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "could not be read as an image" in r.text


def test_decompression_bomb_is_friendly():
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, format="PNG")  # 48 KB file, 400 megapixels decoded
    r = client.post("/verify", data=OLD_TOM, files={"image": ("bomb.png", buf.getvalue(), "image/png")})
    assert r.status_code == 400 and ("too large" in r.text or "megapixels" in r.text)


def test_no_image_is_friendly():
    r = client.post("/verify", data=OLD_TOM)
    assert r.status_code == 400 and "Please add a label image" in r.text
    assert 'class="drop err"' in r.text and "The label image is missing" in r.text   # the drop zone is the error
    assert 'value="OLD TOM DISTILLERY"' in r.text                                     # what was typed is kept


def test_template_csv():
    r = client.get("/batch/template.csv")
    assert r.status_code == 200 and r.text.startswith("image,application_id,brand_name")


def test_batch_missing_columns_is_friendly():
    csv_bytes = b"image,brand\nfoo.png,X\n"
    r = client.post("/batch", files={"csv_file": ("apps.csv", csv_bytes, "text/csv"),
                                     "files": ("foo.png", b"x", "image/png")})
    assert r.status_code == 400 and "missing 3 required columns" in r.text
    assert 'class="cols-fix"' in r.text and "class_type" in r.text and "add</span>" in r.text


def test_unknown_batch_is_friendly():
    r = client.get("/batch/nope")
    assert r.status_code == 404 and "no longer available" in r.text


@requires_tesseract
def test_verify_sample_old_tom_html_and_json():
    r = client.post("/verify", data={**OLD_TOM, "sample": "old_tom_clean"})
    assert r.status_code == 200 and "Label matches the application" in r.text and 'class="timing"' in r.text
    r = client.post("/api/verify", data={**OLD_TOM, "sample": "old_tom_clean"})
    body = r.json()
    assert body["overall"] == "PASS" and body["timings"]["total_ms"] < TIMING_BUDGET_MS


@requires_tesseract
def test_batch_end_to_end_with_export():
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 "old_tom_clean.png,A1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "wrong_abv.png,A2,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "missing.png,A3,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n").encode()
    files = [("csv_file", ("apps.csv", csv_bytes, "text/csv"))]
    for name in ("old_tom_clean.png", "wrong_abv.png"):
        files.append(("files", (name, (SAMPLES / name).read_bytes(), "image/png")))
    files.append(("files", ("readme.txt", b"not an image", "text/plain")))
    r = client.post("/batch", files=files)
    assert r.status_code == 200
    import re
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    import time
    for _ in range(600):  # up to two minutes: shared CI runners can be slow
        r = client.get(f"/batch/{job_id}")
        if 'data-status="done"' in r.text:
            break
        time.sleep(0.2)
    assert 'data-status="done"' in r.text
    assert "No image named" in r.text and "missing.png" in r.text and "readme.txt" in r.text
    j = client.get(f"/api/batch/{job_id}").json()
    by_id = {it["application_id"]: it for it in j["items"]}
    assert by_id["A1"]["status"] == "PASS" and by_id["A2"]["status"] == "FAIL" and by_id["A3"]["status"] == "ERROR"
    csv_out = client.get(f"/batch/{job_id}/export.csv").text
    assert csv_out.count("\n") == 4 and "alcohol_content_verdict" in csv_out
    detail = client.get(f"/batch/{job_id}/item/1").text
    assert "Problems found" in detail
