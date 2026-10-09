"""Error states (Errors board): inline, specific, nothing typed is lost; an unreadable image is not a FAIL."""

import csv
import io
import re
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app import batch, main
from app.models import Application
from app.pipeline import verify
from app.readers.base import LabelReading, OCRResult, OCRWord

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
client = TestClient(main.app)
OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}
PARTIAL = {"X-Partial": "1"}


class BlankReader:
    """An image with almost nothing on it: two confident words that match nothing."""

    name = "blank"

    def read(self, image):
        words = [OCRWord(text="blurry", left=10, top=10, width=60, height=20, conf=91, line_index=0),
                 OCRWord(text="glare", left=80, top=10, width=50, height=20, conf=88, line_index=0)]
        return LabelReading(ocr=OCRResult(text="blurry glare", lines=["blurry glare"], words=words, engine="blank"))


class NoisyReader:
    """Plenty of confident text, none of it the application's: a wrong label, which must FAIL normally."""

    name = "noisy"

    def read(self, image):
        lines = ["SUNNY ORCHARD CIDER COMPANY", "Hard Apple Cider made from fresh pressed apples",
                 "Brewed and canned by Sunny Orchard, Portland Oregon"]
        words, idx = [], 0
        for i, line in enumerate(lines):
            for t in line.split():
                words.append(OCRWord(text=t, left=10 + 40 * idx, top=30 * i, width=36, height=20, conf=93, line_index=i))
                idx += 1
        return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, words=words, engine="noisy"))


def test_missing_fields_without_js_keep_values_and_mark_each_input():
    r = client.post("/verify", data={"brand_name": "OLD TOM DISTILLERY", "alcohol_content": "45%"})
    assert r.status_code == 400
    t = r.text
    assert 'value="OLD TOM DISTILLERY"' in t and 'value="45%"' in t
    assert re.search(r'id="f-class_type"[^>]*aria-invalid="true"', t)
    assert "Please fill in the class / type, e.g. Kentucky Straight Bourbon Whiskey." in t
    assert '<a href="#f-class_type" data-focus="f-class_type">Class / type</a>' in t
    assert 'id="f-brand_name"' in t and not re.search(r'id="f-brand_name"[^>]*aria-invalid', t)
    assert '<div id="result" class="result-slot">' in t       # shown where the result would be


def test_missing_fields_for_the_script_carry_the_fields_to_mark():
    r = client.post("/verify", data={"brand_name": "X"}, headers=PARTIAL)
    assert r.status_code == 400 and 'role="alert"' in r.text and 'data-kind="fields"' in r.text
    assert "f-net_contents" in r.text and "Please fill in the net contents, e.g. 750 mL." in r.text


def test_every_unreadable_number_is_named_at_once():
    r = client.post("/verify", data={**OLD_TOM, "alcohol_content": "strong", "net_contents": "big"}, headers=PARTIAL)
    assert r.status_code == 400
    assert "Alcohol content</a> should look like" in r.text and "Net contents</a> should look like" in r.text


def test_a_pdf_turns_the_drop_zone_into_the_error():
    r = client.post("/verify", data=OLD_TOM, files={"image": ("application-form.pdf", b"%PDF-1.4 x", "application/pdf")},
                    headers=PARTIAL)
    assert r.status_code == 400 and 'data-kind="image_unreadable"' in r.text
    assert "That isn&#39;t an image we can read" in r.text and "application-form.pdf" in r.text


def test_reader_failure_offers_a_retry(monkeypatch):
    from app.readers.base import ReaderError

    class Broken:
        name = "broken"

        def read(self, image):
            raise ReaderError("Reading the label took longer than 30 s and was stopped.")
    monkeypatch.setitem(main._readers, "tesseract", Broken())
    r = client.post("/verify", data={**OLD_TOM, "sample": "old_tom_clean"}, headers=PARTIAL)
    assert r.status_code == 502 and 'data-act="retry"' in r.text and "Try again" in r.text


def test_unreadable_image_is_not_a_fail():
    r = verify(Application(**OLD_TOM), Image.new("RGB", (600, 800), "white"), BlankReader())
    assert r.unreadable and r.words_read == 2
    assert r.overall.value == "FAIL"      # an API client that ignores the flag never takes it as approved


def test_a_wrong_but_readable_label_still_fails_normally():
    r = verify(Application(**OLD_TOM), Image.new("RGB", (600, 800), "white"), NoisyReader())
    assert not r.unreadable and r.overall.value == "FAIL"


def test_unreadable_card_has_no_verdict(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", BlankReader())
    r = client.post("/verify", data={**OLD_TOM, "sample": "old_tom_clean"}, headers=PARTIAL)
    assert r.status_code == 200
    assert "We couldn't read this label" in r.text and "This is not a FAIL." in r.text
    assert "Problems found" not in r.text and "Mandatory label information" not in r.text
    j = client.post("/api/verify", data={**OLD_TOM, "sample": "old_tom_clean"}).json()
    assert j["unreadable"] is True and j["words_read"] == 2


def _wait(job_id, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/batch/{job_id}")
        if 'data-status="done"' in r.text:
            return r
        time.sleep(0.05)
    raise AssertionError("batch did not finish")


def test_unreadable_image_in_a_batch_is_an_error_row_not_a_fail(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", BlankReader())
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 "old_tom_clean.png,A1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n").encode()
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv")),
                                     ("files", ("old_tom_clean.png", (SAMPLES / "old_tom_clean.png").read_bytes(), "image/png"))],
                    headers=PARTIAL)
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    t = _wait(job_id).text
    assert 'data-status="ERROR"' in t and "Unreadable image" in t
    assert 'data-filter="ERROR"' in t and "1 could not be checked" in t and "0 have problems" in t
    rows = list(csv.reader(io.StringIO(client.get(f"/batch/{job_id}/export.csv").text)))
    assert rows[1][2] == "ERROR" and "Unreadable image" in rows[1][4]


def test_csv_header_near_miss_is_named():
    csv_bytes = b"image,brand_nam,class_type,alcohol_content,net_contents\nfoo.png,X,Y,45%,750 mL\n"
    r = client.post("/batch", files={"csv_file": ("apps.csv", csv_bytes, "text/csv"),
                                     "files": ("foo.png", b"x", "image/png")}, headers=PARTIAL)
    assert r.status_code == 400 and "missing 1 required column." in r.text
    assert re.search(r"brand_name</td><td class=\"mono\">brand_nam</td>.*rename", r.text, re.S)


def test_too_many_images_says_how_to_split(monkeypatch):
    monkeypatch.setattr(batch, "MAX_BATCH_IMAGES", 2)
    try:
        batch.collect_images([(f"l{i}.png", bytes([i]) * 10) for i in range(3)])
    except batch.BatchError as e:
        assert e.title == "That's more than one batch can take."
        assert "You added 3 images; the limit is 2 per batch" in str(e)
    else:
        raise AssertionError("expected BatchError")


class SparseWrongReader:
    """A neck label read perfectly: three words, all of them disagreeing with the application."""

    name = "sparse"

    def read(self, image):
        lines = ["VODKA", "40% ALC/VOL", "1 LITER"]
        words = [OCRWord(text=t, left=10 + 60 * j, top=50 * i, width=50, height=30, conf=92, line_index=i)
                 for i, line in enumerate(lines) for j, t in enumerate(line.split())]
        return LabelReading(ocr=OCRResult(text="\n".join(lines), lines=lines, words=words, engine="sparse"))


def test_few_words_that_were_read_and_disagree_are_a_fail_not_unreadable():
    app = Application(brand_name="NORTH STAR", class_type="Gin", alcohol_content="45%", net_contents="750 mL")
    r = verify(app, Image.new("RGB", (300, 300), "white"), SparseWrongReader())
    assert not r.unreadable and r.overall.value == "FAIL"
    by_key = {f.key: f.verdict.value for f in r.fields}
    assert by_key["alcohol_content"] == "MISMATCH" and by_key["net_contents"] == "MISMATCH"
