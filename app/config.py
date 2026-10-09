"""Single source of truth for every threshold, regulated text and field setting.

Everything an agent or reviewer might want to tune lives here and is explained
in the README ("Matching rules and thresholds").
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- Regulated text (27 CFR 16.21) ---------------------------------------------------------
WARNING_HEADING = "GOVERNMENT WARNING:"
MANDATED_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate "
    "machinery, and may cause health problems."
)


# --- Fields we verify -------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    required: bool
    kind: str  # "text" | "alcohol" | "volume" | "country"
    hint: str = ""


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("brand_name", "Brand name", True, "text", "e.g. OLD TOM DISTILLERY"),
    FieldSpec("class_type", "Class / type", True, "text", "e.g. Kentucky Straight Bourbon Whiskey"),
    FieldSpec("alcohol_content", "Alcohol content", True, "alcohol", "e.g. 45% Alc./Vol. or 90 Proof"),
    FieldSpec("net_contents", "Net contents", True, "volume", "e.g. 750 mL"),
    FieldSpec("bottler_name_address", "Bottler name & address", False, "text",
              "e.g. Bottled by Old Tom Distillery, Bardstown, KY"),
    FieldSpec("country_of_origin", "Country of origin", False, "country",
              "Imports only, e.g. Product of Scotland"),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}


# --- Matching thresholds ----------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    # Text fields: rapidfuzz similarity (0-100) after normalization.
    near_match: int = 88          # >= this and not identical  -> NEAR MATCH (agent decides)
    find_floor: int = 60          # best window on the label scores below this -> NOT FOUND
    locate_max_lines: int = 3     # a value may wrap over up to this many OCR lines (addresses, long class names)
    # Fields whose value must be the whole phrase on the label: "Rum" inside "SPICED RUM" is a NEAR MATCH.
    whole_phrase_fields: tuple[str, ...] = ("brand_name", "class_type")
    # A brand found only in text smaller than this fraction of the label's largest text (e.g. inside
    # "Bottled by ...") is a NEAR MATCH: the brand on the label may be a different one.
    brand_small_print_ratio: float = 0.5
    # Numeric fields.
    abv_tolerance: float = 0.05   # percentage points of alcohol by volume
    proof_tolerance: float = 0.5  # proof degrees: a label's proof may be rounded to a whole number (46.3% -> 93)
    volume_tolerance_ml: float = 0.5
    # Government warning.
    warning_locate: int = 75      # similarity needed to recognise the "GOVERNMENT WARNING" line
    warning_near: int = 97        # wording similarity >= this (but not exact) -> NEEDS REVIEW, else FAIL
    bold_ratio: float = 1.30      # heading stroke width / body stroke width -> "likely bold" (measured: bold 1.51-2.04, regular 1.03-1.14)
    bold_min_text_px: int = 14    # below this text height the stroke measurement is unreliable
    bold_failure_is_fail: bool = False  # the bold check is a heuristic, so by default it only asks for review
    # Fields where a capitalization/punctuation-only difference is flagged for human review
    # instead of being accepted silently ("STONE'S THROW" vs "Stone's Throw").
    case_review_fields: tuple[str, ...] = ("brand_name",)


THRESHOLDS = Thresholds()


# --- OCR / runtime settings (environment overrides) --------------------------------------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").lower()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")          # empty -> use the one on PATH
_psm_spec = os.getenv("TESSERACT_PSM", "4+11").split("+")  # "4+11": block mode, then sparse mode merged (see README)
TESSERACT_PSM = int(_psm_spec[0])
TESSERACT_PSM_EXTRA = int(_psm_spec[1]) if len(_psm_spec) > 1 else None
TESSERACT_MIN_WIDTH = 1600                              # upscale smaller images before OCR
TESSERACT_MAX_WIDTH = 2400                              # downscale huge images (speed)
TESSERACT_MAX_PIXELS = 10_000_000                       # never hand Tesseract more than this (a 40 x 4000 strip
TESSERACT_MAX_SIDE = 10_000                             # would otherwise be upscaled to 1600 x 160000)
TESSERACT_TIMEOUT_S = 30                                # per pass; a pathological image must not hold a worker
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLOUD_READER_AVAILABLE = bool(ANTHROPIC_API_KEY)

BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", str(min(4, os.cpu_count() or 2))))
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000                           # decoded size limit (a 20 MB PNG can decode to gigabytes)
MIN_IMAGE_SIDE = 50                                     # smaller images cannot be read
MAX_BATCH_IMAGES = 500
MAX_ZIP_MEMBERS = 1000
MAX_ZIP_UNCOMPRESSED = 1024 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
