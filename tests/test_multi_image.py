"""An application's front, back and neck labels can be checked together."""

import io
import re
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.images import stitch
from app.main import app

from .conftest import requires_tesseract

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "old_tom_clean.png"
client = TestClient(app)
OLD_TOM = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL"}


def halves() -> tuple[bytes, bytes]:
    img = Image.open(SAMPLE).convert("RGB")
    top, bottom = img.crop((0, 0, img.width, 1000)), img.crop((0, 1000, img.width, img.height))
    out = []
    for part in (top, bottom):
        buf = io.BytesIO()
        part.save(buf, "PNG")
        out.append(buf.getvalue())
    return out[0], out[1]


def test_stitch_stacks_centred_with_a_gap():
    a, b = Image.new("RGB", (100, 50), "red"), Image.new("RGB", (60, 30), "blue")
    s = stitch([a, b])
    assert s.size == (100, 50 + 40 + 30)
    assert s.getpixel((50, 10)) == (255, 0, 0) and s.getpixel((50, 100)) == (0, 0, 255) and s.getpixel((5, 100)) == (255, 255, 255)


def test_too_many_images_for_one_label_is_a_friendly_error():
    files = [("image", (f"p{i}.png", b"x", "image/png")) for i in range(7)]
    r = client.post("/api/verify", data=OLD_TOM, files=files)
    assert r.status_code == 400 and "up to 6" in r.json()["error"]


@requires_tesseract
def test_front_and_back_read_together_find_the_warning():
    front, back = halves()
    r = client.post("/api/verify", data=OLD_TOM, files=[("image", ("front.png", front, "image/png")),
                                                         ("image", ("back.png", back, "image/png"))])
    body = r.json()
    assert r.status_code == 200 and body["warning"]["present"] and body["image_name"] == "front.png + back.png"
    assert body["overall"] in ("PASS", "REVIEW")
    only_front = client.post("/api/verify", data=OLD_TOM, files={"image": ("front.png", front, "image/png")}).json()
    assert not only_front["warning"]["present"]


@requires_tesseract
def test_batch_row_lists_several_images():
    front, back = halves()
    csv_bytes = ("image,application_id,brand_name,class_type,alcohol_content,net_contents\n"
                 "front.png; back.png,M1,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n"
                 "front.png; neck.png,M2,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL\n").encode()
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv")),
                                     ("files", ("front.png", front, "image/png")),
                                     ("files", ("back.png", back, "image/png"))], headers={"X-Partial": "1"})
    job_id = re.search(r'data-job="([a-f0-9]+)"', r.text).group(1)
    for _ in range(300):
        j = client.get(f"/api/batch/{job_id}").json()
        if j["finished"]:
            break
        time.sleep(0.2)
    by_id = {it["application_id"]: it for it in j["items"]}
    assert by_id["M1"]["status"] in ("PASS", "REVIEW") and by_id["M1"]["result"]["warning"]["present"]
    assert by_id["M2"]["status"] == "ERROR" and "neck.png" in by_id["M2"]["error"]
