"""Evidence boxes: each located value points at the words it was read from."""

import numpy as np

from app.models import Application, Verdict
from app.pipeline import verify
from app.readers.base import LabelReading, OCRResult, OCRWord
from app.readers.extract import attach_boxes, word_box, words_for_text

WARNING = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages "
           "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs "
           "your ability to drive a car or operate machinery, and may cause health problems.")


def make_ocr():
    lines = ["OLD TOM DISTILLERY", "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)", "750 mL",
             "Distilled and Bottled by Old Tom Distillery, Bardstown, Kentucky 40004"] + [WARNING]
    words, y = [], 100
    for li, line in enumerate(lines):
        x = 100
        for tok in line.split():
            words.append(OCRWord(text=tok, left=x, top=y, width=60, height=40, conf=95, line_index=li))
            x += 70
        y += 100
    return OCRResult(text="\n".join(lines), lines=lines, words=words, ink=np.zeros((1000, 2000), dtype=bool),
                     mean_conf=96.5, engine="fake")


class FakeReader:
    name = "fake"

    def read(self, image):
        return LabelReading(ocr=make_ocr())


def test_words_for_text_aligns_tokens_inside_a_line():
    ocr = make_ocr()
    ws = words_for_text(ocr, "Old Tom Distillery", (4, 4))
    assert [w.text for w in ws] == ["Old", "Tom", "Distillery,"]


def test_word_box_is_percent_of_image():
    ocr = make_ocr()
    assert word_box(ocr.words[:3], 2000, 1000) == [5.0, 10.0, 10.0, 4.0]


def test_pipeline_attaches_boxes_and_confidence():
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="45% Alc./Vol.", net_contents="750 mL",
                      bottler_name_address="Old Tom Distillery, Bardstown, Kentucky 40004")
    from PIL import Image
    r = verify(app, Image.new("RGB", (10, 10)), FakeReader())
    by_key = {f.key: f for f in r.fields}
    assert by_key["brand_name"].box == [5.0, 10.0, 10.0, 4.0]          # first line, three words
    assert by_key["alcohol_content"].box[1] == 30.0                     # third line
    assert by_key["net_contents"].box == [5.0, 40.0, 6.5, 4.0]
    assert by_key["bottler_name_address"].box[1] == 50.0
    assert by_key["country_of_origin"].box is None and by_key["country_of_origin"].verdict == Verdict.SKIPPED
    assert r.warning.box is not None and r.warning.box[1] == 60.0
    assert r.read_confidence == 96.5


def test_attach_boxes_is_a_no_op_without_image_data():
    ocr = make_ocr()
    ocr.ink = None
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="x", alcohol_content="45%", net_contents="750 mL")
    from app.readers.extract import extract_and_compare
    fields = extract_and_compare(app, ocr)
    attach_boxes(ocr, fields)
    assert all(f.box is None for f in fields)


def test_boxes_read_in_a_turned_view_map_back_onto_the_upright_image():
    """Draw a block at a known place, turn the image as the reader does, find the block where Tesseract
    would report it in that view, and map it back: it must land where it was drawn."""
    from PIL import Image, ImageDraw

    from app.readers.base import View, upright_box
    W, H = 400, 300
    drawn = (50, 200, 120, 30)   # left, top, width, height on the upright image
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).rectangle([drawn[0], drawn[1], drawn[0] + drawn[2] - 1, drawn[1] + drawn[3] - 1], fill=0)
    views = [View(rot=0, inverted=False, ink=np.zeros((H, W), bool), size=(W, H))]
    for rot in (90, 270):
        turned = img.rotate(rot, expand=True)        # what TesseractReader.extend reads
        ys, xs = np.nonzero(np.asarray(turned) < 128)
        box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        views.append(View(rot=rot, inverted=False, ink=np.zeros(turned.size[::-1], bool), size=turned.size))
        word = OCRWord(text="x", left=box[0], top=box[1], width=box[2], height=box[3], conf=90, line_index=0,
                       view=len(views) - 1)
        assert upright_box(word, views) == drawn, (rot, box, upright_box(word, views))
