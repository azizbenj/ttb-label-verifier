"""End to end: generated sample labels through the real Tesseract pipeline."""

import csv
from pathlib import Path

import pytest
from PIL import Image

from app.models import Application, Status, Verdict
from app.pipeline import verify
from app.readers.tesseract import TesseractReader

from .conftest import requires_tesseract

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def load_rows():
    with open(SAMPLES / "samples.csv", newline="") as f:
        return list(csv.DictReader(f))


def run(row):
    app = Application(**{k: row[k] for k in ("brand_name", "class_type", "alcohol_content", "net_contents",
                                             "bottler_name_address", "country_of_origin")})
    return verify(app, Image.open(SAMPLES / row["image"]), TesseractReader(), image_name=row["image"])


@requires_tesseract
def test_old_tom_passes_fast():
    row = next(r for r in load_rows() if r["image"] == "old_tom_clean.png")
    result = run(row)
    assert {f.verdict for f in result.fields if f.verdict != Verdict.SKIPPED} == {Verdict.MATCH}, result.fields
    assert result.warning.overall == Status.PASS, result.warning
    assert result.overall == Status.PASS
    assert result.timings.total_ms < 5000, result.timings


@requires_tesseract
@pytest.mark.parametrize("row", load_rows(), ids=lambda r: Path(r["image"]).stem)
def test_sample_matches_expected_outcome(row):
    result = run(row)
    assert result.overall.value == row["expected_overall"], (result.summary, [(f.key, f.verdict, f.found) for f in result.fields], result.warning)
    issue = row["expected_issue"]
    if issue == "warning":
        assert result.warning.overall in (Status.FAIL, Status.REVIEW)
    elif issue:
        fr = next(f for f in result.fields if f.key == issue)
        assert fr.verdict in (Verdict.MISMATCH, Verdict.NOT_FOUND, Verdict.NEAR_MATCH), fr
    assert result.timings.total_ms < 5000, result.timings
