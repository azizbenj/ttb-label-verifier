"""Without JavaScript: every form posts, every page is a whole page, a running batch reloads itself."""

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.readers.base import LabelReading, OCRResult

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
LINES = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL"]


class SlowFake:
    name = "fake"

    def read(self, image):
        time.sleep(0.3)
        return LabelReading(ocr=OCRResult(text="\n".join(LINES), lines=LINES, engine="fake"))


@pytest.fixture(autouse=True)
def fake_reader(monkeypatch):
    monkeypatch.setitem(main._readers, "tesseract", SlowFake())


client = TestClient(main.app)   # no X-Partial header: a browser without the page's script
CSV = (b"image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
       b"old_tom_clean.png,A1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45%,750 mL\n")


def test_the_index_works_without_a_script():
    page = client.get("/").text
    assert 'classList.add("js")' in page                      # panels and tabs only switch with a script
    css = client.get("/static/style.css").text
    assert "html:not(.js) .panel { display: block; }" in css   # without it, both panels show
    assert 'name="sample" value="1"' in page                   # the sample batch is a plain submit button


def test_a_batch_posted_without_a_script_gets_its_own_page_that_reloads_until_done():
    r = client.post("/batch", files=[("csv_file", ("a.csv", CSV, "text/csv")),
                                     ("files", ("old_tom_clean.png", (SAMPLES / "old_tom_clean.png").read_bytes(), "image/png"))],
                    follow_redirects=False)
    assert r.status_code == 303 and re.fullmatch(r"/batch/[a-f0-9]+", r.headers["location"])
    page = client.get(r.headers["location"]).text
    assert page.lstrip().lower().startswith("<!doctype html")
    assert 'http-equiv="refresh"' in page and 'data-status="running"' in page
    deadline = time.monotonic() + 60
    while 'data-status="done"' not in page and time.monotonic() < deadline:
        time.sleep(0.1)
        page = client.get(r.headers["location"]).text
    assert 'http-equiv="refresh"' not in page
    link = re.search(r'<a class="rowlink" href="(/batch/[a-f0-9]+/item/\d+)"', page).group(1)
    item = client.get(link).text
    assert item.lstrip().lower().startswith("<!doctype html") and "Back to the results table" in item


def test_batch_errors_without_a_script_are_whole_pages_on_the_batch_tab():
    r = client.post("/batch", files=[("csv_file", ("a.csv", b"brand\nX\n", "text/csv")),
                                     ("files", ("x.png", b"x", "image/png"))])
    assert r.status_code == 400 and r.text.lstrip().lower().startswith("<!doctype html")
    assert 'data-open-tab="batch"' in r.text and "missing" in r.text
    gone = client.get("/batch/nope")
    assert gone.status_code == 404 and gone.text.lstrip().lower().startswith("<!doctype html")
