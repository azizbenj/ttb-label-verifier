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


def _wait_done(job_id, timeout_s: float = 60.0):
    """Generous on purpose: shared CI runners can be many times slower than a laptop."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/batch/{job_id}")
        if 'data-status="done"' in r.text:
            return r
        time.sleep(0.05)
    raise AssertionError("batch did not finish")


def test_single_verify_renders_result_card():
    r = client.post("/verify", data={**OLD_TOM, "sample": "old_tom_clean"})
    assert r.status_code == 200
    assert "Needs a quick look" in r.text and 'class="timing"' in r.text and "Government warning" in r.text


def test_sample_batch_runs_and_renders_table_and_export():
    r = client.post("/batch", data={"sample": "1"})
    assert r.status_code == 200, r.text
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    r = _wait_done(job_id)
    assert "250 labels checked" in r.text and "Export…" in r.text
    csv_out = client.get(f"/batch/{job_id}/export.csv").text
    assert csv_out.count("\n") == 251
    detail = client.get(f"/batch/{job_id}/item/0").text
    assert 'class="timing"' in detail


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


# --- intake and export details -------------------------------------------------------------------
def _start(csv_bytes, files):
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv"))] + files)
    assert r.status_code == 200, r.text
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    _wait_done(job_id)
    return job_id, client.get(f"/api/batch/{job_id}").json()


def test_unreadable_application_values_are_row_errors_not_label_failures():
    png = (SAMPLES / "old_tom_clean.png").read_bytes()
    csv_bytes = (b"image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 b"old_tom_clean.png,A1,OLD TOM DISTILLERY,Bourbon,forty five,750 mL\n"
                 b"old_tom_clean.png,A2,OLD TOM DISTILLERY,Bourbon,45%,a bottle\n")
    _, j = _start(csv_bytes, [("files", ("old_tom_clean.png", png, "image/png"))])
    by_id = {it["application_id"]: it for it in j["items"]}
    assert by_id["A1"]["status"] == "ERROR" and "alcohol content 'forty five'" in by_id["A1"]["error"]
    assert by_id["A2"]["status"] == "ERROR" and "net contents 'a bottle'" in by_id["A2"]["error"]


def test_same_file_name_in_two_zip_folders_is_never_guessed():
    import io
    import zipfile
    png = (SAMPLES / "old_tom_clean.png").read_bytes()
    other = (SAMPLES / "wrong_abv.png").read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("lot1/label.png", png)
        z.writestr("lot2/label.png", other)
        z.writestr("lot2/copy/label_b.png", png)
    csv_bytes = (b"image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 b"label.png,A1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45%,750 mL\n"
                 b"label_b,A2,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45%,750 mL\n")
    _, j = _start(csv_bytes, [("files", ("labels.zip", buf.getvalue(), "application/zip"))])
    by_id = {it["application_id"]: it for it in j["items"]}
    assert by_id["A1"]["status"] == "ERROR" and "More than one" in by_id["A1"]["error"]
    assert by_id["A2"]["status"] == "REVIEW"  # extension omitted in the CSV, one candidate: fine
    assert any("Several different files are named 'label.png'" in i for i in j["issues"])


def test_encrypted_zip_member_is_skipped_with_a_message():
    import io
    import zipfile

    from app import batch
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("label.png", (SAMPLES / "old_tom_clean.png").read_bytes())
    data = bytearray(buf.getvalue())
    for sig, off in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):  # set the "encrypted" flag
        data[data.find(sig) + off] |= 1
    images, issues = batch.collect_images([("enc.zip", bytes(data))])
    assert images == {} and "encrypted" in issues[0]


def test_semicolon_csv_and_multiline_cells():
    from app import batch
    rows, _ = batch.parse_applications_csv(
        "image;brand_name;class_type;alcohol_content;net_contents\na.png;X;Y;45%;750 mL\n".encode())
    assert rows[0]["brand_name"] == "X"
    rows, _ = batch.parse_applications_csv(
        b'image,brand_name,class_type,alcohol_content,net_contents,bottler_name_address\n'
        b'a.png,X,Y,45%,750 mL,"Old Tom Distillery\nBardstown, KY"\nb.png,,Y,45%,750 mL,\n')
    assert [r["_row"] for r in rows] == [3, 4]  # file line numbers, so "Row 4" points at the right line


def test_batch_upload_total_is_capped(monkeypatch):
    import pytest

    from app import batch
    monkeypatch.setattr(batch, "MAX_BATCH_UPLOAD_BYTES", 100)
    with pytest.raises(batch.BatchError, match="add up to"):
        batch.collect_images([("a.png", b"x" * 60), ("b.png", b"y" * 60)])


def test_export_neutralises_spreadsheet_formulas():
    from app import batch
    from app.models import Application
    item = batch.BatchItem(row=2, application_id="=HYPERLINK(\"http://x\",\"open\")", image_name="@a.png",
                           application=Application(brand_name="X", class_type="Y", alcohol_content="45%",
                                                   net_contents="750 mL"), error="-1 problem")
    job = batch.BatchJob(id="t", source="t", items=[item], done=1)
    out = batch.export_csv(job).splitlines()[1]
    assert out.startswith("\"'=HYPERLINK") and ",'@a.png," in out and "'-1 problem" in out


def test_finished_jobs_are_pruned(monkeypatch):
    from app import batch
    monkeypatch.setattr(batch, "JOBS", {})
    monkeypatch.setattr(batch, "BATCH_JOBS_KEPT", 3)
    for i in range(6):
        job = batch.BatchJob(id=f"j{i}", source="t", items=[], done=0)
        with batch._JOBS_LOCK:
            batch._prune_jobs()
            batch.JOBS[job.id] = job
    assert sorted(batch.JOBS) == ["j2", "j3", "j4", "j5"]  # three kept plus the one just added
