import os

import pytest

from app.readers.tesseract import tesseract_version

requires_tesseract = pytest.mark.skipif(tesseract_version() is None, reason="tesseract binary not installed")

# The 5 s requirement is measured on the deployment (scripts/bench.py --url ...). Shared CI runners can be
# many times slower on a cold start, so the budget the tests assert is configurable there.
TIMING_BUDGET_MS = float(os.getenv("LABELCHECK_TIMING_BUDGET_MS", "5000"))


@pytest.fixture(scope="session", autouse=True)
def warm_up_tesseract():
    """Load the OCR model once so the first timed test does not pay the cold-start cost."""
    if tesseract_version() is not None:
        from PIL import Image, ImageDraw
        from app.readers.tesseract import TesseractReader
        img = Image.new("RGB", (600, 200), "white")
        ImageDraw.Draw(img).text((20, 80), "WARM UP 750 mL", fill="black")
        TesseractReader().read(img)
    yield
