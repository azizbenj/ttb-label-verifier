import pytest

from app.readers.tesseract import tesseract_version

requires_tesseract = pytest.mark.skipif(tesseract_version() is None, reason="tesseract binary not installed")
