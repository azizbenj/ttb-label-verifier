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
    # Another reading of the same place (a second OCR pass or view) at least this similar to the
    # application value but not equal to it turns a MATCH into a NEAR MATCH: the readings disagree.
    conflict_floor: int = 70
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
    # Unreadable image: fewer than this many words (3+ letters) read with at least this confidence, and
    # none of the application's values found. Shown as "We couldn't read this label", never as a FAIL.
    # (A mean-confidence floor alone is not used: a real beer can averages 41% and still matches 4 of 5 fields.)
    unreadable_min_words: int = 8
    unreadable_word_conf: int = 70
    # Second read of the figures (app/readers/numbers.py). When the alcohol content or the net contents is
    # missing, different or doubtful, up to this many lines carrying a figure are cut out of the page,
    # scaled so the line is this many pixels tall and read again on their own. 25 px was measured over
    # the batch set, the stress renderings and the real labels: Tesseract reads a figure best when the
    # line is 20-25 px tall; at the 40-60 px the page pass works with, "1.5 L" comes back "L5L" and
    # "750 mL" "790 mL" (enlarging the crop, as one might expect to help, makes it worse). A first
    # reading is set aside as a misread only when every closer read of that line agrees with the
    # application and the first reading's digits are within this many edits of it ("15" for "1.5",
    # "790" for "750", "077" for "577"); a first reading further away stays a NEAR MATCH naming both.
    number_reread_max_crops: int = 4
    number_reread_text_px: int = 25
    number_reread_max_edits: int = 1


THRESHOLDS = Thresholds()


# --- OCR / runtime settings (environment overrides) --------------------------------------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").lower()
_LOCAL_TESSERACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tesseract", "bin", "tesseract")
# TESSERACT_CMD env var, else a project-local install (.tesseract/, e.g. a conda env: see README), else PATH.
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or (_LOCAL_TESSERACT if os.access(_LOCAL_TESSERACT, os.X_OK) else "")
_psm_spec = os.getenv("TESSERACT_PSM", "4+11").split("+")  # "4+11": block mode, then sparse mode merged (see README)
TESSERACT_PSM = int(_psm_spec[0])
TESSERACT_PSM_EXTRA = int(_psm_spec[1]) if len(_psm_spec) > 1 else None
TESSERACT_MIN_WIDTH = 1600                              # upscale smaller images before OCR
TESSERACT_MAX_WIDTH = 2400                              # downscale huge images (speed)
TESSERACT_MAX_PIXELS = 10_000_000                       # never hand Tesseract more than this (a 40 x 4000 strip
TESSERACT_MAX_SIDE = 10_000                             # would otherwise be upscaled to 1600 x 160000)
TESSERACT_TIMEOUT_S = 30                                # per pass; a pathological image must not hold a worker
TESSERACT_EXTRA_TIMEOUT_S = 8                           # the optional turned/contrast passes: skipped if slower
DESKEW_MAX_DEG = 6.0                                    # straighten scans tilted by up to this much
DESKEW_MIN_DEG = 0.8                                    # smaller tilts are left alone: Tesseract copes, resampling costs
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_TIMEOUT_S = float(os.getenv("CLAUDE_TIMEOUT_S", "30"))  # per attempt; the SDK default is 10 minutes
CLAUDE_MAX_RETRIES = 1
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLOUD_READER_AVAILABLE = bool(ANTHROPIC_API_KEY)

BATCH_JOBS_KEPT = 50                                    # finished batch jobs kept in memory
BATCH_JOB_TTL_S = 24 * 3600                             # ... and for at most this long
BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", str(min(4, os.cpu_count() or 2))))
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000                           # decoded size limit (a 20 MB PNG can decode to gigabytes)
MIN_IMAGE_SIDE = 50                                     # smaller images cannot be read
MAX_BATCH_IMAGES = 500
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024           # all images of one batch together
MAX_ZIP_MEMBERS = 1000
MAX_ZIP_UNCOMPRESSED = 1024 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
