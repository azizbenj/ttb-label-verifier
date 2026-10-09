"""Image intake and the Tesseract reader's bookkeeping (most tests need no Tesseract binary)."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.images import ImageError, flatten, open_image
from app.readers import tesseract as T
from app.readers.base import ReaderError

from .conftest import requires_tesseract

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _transparent_label() -> Image.Image:
    """The Old Tom sample with its paper made transparent, as design tools export artwork."""
    rgb = np.asarray(Image.open(SAMPLES / "old_tom_clean.png").convert("RGB"))
    paper = rgb.mean(axis=2) > 128
    rgba = np.dstack([rgb, np.where(paper, 0, 255).astype(np.uint8)])
    rgba[paper, :3] = 0  # transparent pixels typically store black
    return Image.fromarray(rgba, "RGBA")


# --- flattening any image mode to plain pixels -------------------------------------------------
def test_transparent_background_becomes_white_not_black():
    flat = np.asarray(flatten(_transparent_label()).convert("L"))
    assert flat.mean() > 200  # mostly paper, as on the original label
    _, ink = T.preprocess(_transparent_label())
    assert 0.01 < ink.mean() < 0.3


def test_sixteen_bit_scan_keeps_its_contrast():
    a = np.full((200, 800), 60000, dtype=np.uint16)
    a[80:120, 100:700] = 2000
    img = Image.open(io.BytesIO(_png(Image.fromarray(a))))
    assert img.mode.startswith("I")
    _, ink = T.preprocess(img)
    assert 0.05 < ink.mean() < 0.5  # Pillow's own conversion clips this to a blank white page


def test_palette_transparency_and_cmyk_are_flattened():
    p = Image.new("P", (100, 100), 0)
    p.putpalette([0, 0, 0] + [255, 255, 255] * 255)
    p.info["transparency"] = 0
    assert np.asarray(flatten(p).convert("L")).mean() == 255
    assert flatten(Image.new("CMYK", (100, 100), (0, 0, 0, 255))).mode == "RGB"


# --- refusing what cannot be read safely ---------------------------------------------------------
def test_decompression_bomb_is_refused_with_a_message():
    bomb = _png(Image.new("1", (20000, 20000)))  # 48 KB on disk, 400 megapixels decoded
    with pytest.raises(ImageError, match="megapixels|too large"):
        open_image(bomb, "bomb.png")


def test_unreadable_tiny_and_truncated_images_are_refused():
    with pytest.raises(ImageError, match="could not be read"):
        open_image(b"not an image", "notes.png")
    with pytest.raises(ImageError, match="too small"):
        open_image(_png(Image.new("RGB", (20, 400), "white")), "tiny.png")
    whole = (SAMPLES / "old_tom_clean.png").read_bytes()
    with pytest.raises(ImageError, match="could not be read"):
        open_image(whole[: len(whole) // 2], "cut.png")


def test_ocr_scale_never_explodes_narrow_images():
    for w, h in [(40, 4000), (10, 3000), (1600, 30000)]:
        s = T.ocr_scale(w, h)
        assert w * h * s * s <= T.TESSERACT_MAX_PIXELS * 1.001 and max(w, h) * s <= T.TESSERACT_MAX_SIDE + 1
    assert T.ocr_scale(1200, 1600) == pytest.approx(1600 / 1200)  # ordinary labels unchanged


# --- two-pass merge ------------------------------------------------------------------------------
def _data(lines: list[str]) -> dict:
    d = {k: [] for k in ("text", "conf", "block_num", "par_num", "line_num", "left", "top", "width", "height")}
    for li, line in enumerate(lines):
        for wi, w in enumerate(line.split()):
            for k, v in (("text", w), ("conf", 90), ("block_num", 1), ("par_num", 1), ("line_num", li),
                         ("left", wi * 100), ("top", li * 50), ("width", 90), ("height", 20)):
                d[k].append(v)
    return d


def test_second_pass_words_keep_their_own_line(monkeypatch):
    # The second pass has more lines than the first: new line indices (2, 3) collide with later
    # second-pass indices, which used to move words to the wrong line and duplicate them.
    passes = iter([_data(["A1 A1", "B2 B2"]), _data(["NEW0 NEW0", "B2 B2", "NEW2 NEW2", "NEW3 NEW3"])])
    monkeypatch.setattr(T.pytesseract, "image_to_data", lambda *a, **k: next(passes))
    ocr = T.TesseractReader(psm=4, extra_psm=11).read(Image.new("L", (1600, 400), 255)).ocr
    assert ocr.lines == ["A1 A1", "B2 B2", "NEW0 NEW0", "NEW2 NEW2", "NEW3 NEW3"]
    assert len(ocr.words) == len({id(w) for w in ocr.words}) == 10
    assert all(w.text in ocr.lines[w.line_index].split() for w in ocr.words)


def test_tesseract_timeout_is_a_reader_error(monkeypatch):
    def slow(*a, **k):
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr(T.pytesseract, "image_to_data", slow)
    with pytest.raises(ReaderError, match="longer than"):
        T.TesseractReader().read(Image.new("L", (800, 400), 255))


@requires_tesseract
def test_transparent_label_reads_like_the_original():
    from app.models import Application, Status
    from app.pipeline import verify
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="45% Alc./Vol.", net_contents="750 mL")
    result = verify(app, _transparent_label(), T.TesseractReader())
    assert result.overall == Status.PASS, (result.summary, result.ocr_text[:200])


def test_deskew_estimates_a_known_tilt():
    from PIL import Image as _I
    from app.readers.tesseract import estimate_skew
    img = _I.open(Path(__file__).resolve().parents[1] / "data" / "samples" / "old_tom_clean.png").convert("L")
    assert abs(estimate_skew(img)) < 0.3
    tilted = img.rotate(-3, resample=_I.BICUBIC, expand=True, fillcolor=255)
    assert abs(estimate_skew(tilted) - 3) <= 0.3


def test_previews_follow_the_straightened_image():
    from PIL import Image as _I
    from app.batch import make_preview
    from app.main import preview_data_url
    img = _I.new("RGB", (400, 600), "white")
    assert make_preview(img, 3.0)
    assert preview_data_url(img, 3.0).startswith("data:image/jpeg;base64,")
