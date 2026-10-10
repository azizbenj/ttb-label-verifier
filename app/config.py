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
    # Government warning layout (word boxes). Two OCR lines whose boxes overlap by this fraction of the
    # smaller one's height and width are two readings of one printed line: only one is taken.
    same_place_overlap: float = 0.5
    # A word is outside the statement's column when its box ends before the column's left edge or starts
    # after its right edge, allowing this fraction of the text height; the edge counts as a neighbouring
    # column only when words lie beyond it on at least column_min_lines of the statement's rows (a single
    # line sticking out may be words added to the statement, and is judged as wording).
    column_tolerance: float = 0.25
    column_min_lines: int = 2
    # A run of lines outside the statement whose text matches a sentence of it at least this well
    # (partial similarity) is that sentence printed a second time; a complete second statement is not reported.
    duplicate_sentence: int = 90
    # Meaning-changing differences in the warning's wording (app/warning.py, README "Government warning
    # check"). A difference fails the wording, instead of asking for a look, when the required word is one
    # whose loss or replacement changes what the statement says and the label's word is a real word read
    # confidently, not an OCR garble. The real-word test is these lists: a word the label prints in place
    # of a required word counts only when it is listed for that word below. A word that becomes the
    # required one once OCR's usual confusions are undone (rn/m, cl/d, 0/o, 1/l/i, 5/s, 8/b: "wornen",
    # "rnay", "n0t"), or a word split or run together ("no t"), is a garble whatever the lists say.
    # Each word of the label that makes the change must have been read with at least this confidence
    # (0-100); a word the reader gave no box for is not trusted, and without word boxes at all (the cloud
    # reader) no difference fails this way: it asks for a look like any other.
    meaning_conf: int = 70
    # A missing "not" or modal counts only when the words on both sides of the gap were read with
    # meaning_conf, on the same line, and no further apart than this many times the text height: a gap
    # wide enough to hold a word is a word OCR dropped, not a word the label left out.
    meaning_gap: float = 1.2
    # Words are judged one by one only when the statement as read is at least this similar to the required
    # text (rapidfuzz, 0-100): below it, what was read is mostly something else and fails on similarity.
    meaning_min_score: int = 90
    # Required words whose absence changes the statement ("not"; the modals "should" and "may").
    meaning_missing: tuple[str, ...] = ("not", "should", "may")
    # Words that, inserted where the required text has none, change what it says (negations, quantifiers,
    # hedges). An inserted word counts only between two words of the statement on its own line: a word at
    # a line's end may belong to a neighbouring column ("FOR SALE ONLY IN OHIO").
    meaning_inserted: tuple[str, ...] = ("not", "no", "never", "only", "sometimes", "rarely", "seldom",
                                         "occasionally", "always", "usually", "often", "hardly", "cannot",
                                         "possibly", "barely")
    # Required word -> the real words that, printed in its place, change the statement. Plural and
    # singular of the same noun ("problem", "defect") and a verb's other form ("impair") are left out on
    # purpose: they break the fixed text (a look is asked for) but do not change what it warns about;
    # "woman" is in, because the regulation's "women" names a group, not a person.
    meaning_substitutes: tuple[tuple[str, str], ...] = (
        ("not", "now also always only ever even then too just still often rather soon"),
        ("should", "can could may might must will would shall need do does cannot"),
        ("may", "can could might must will would shall should need do does cannot"),
        ("women", "men man woman people persons adults children kids mothers girls ladies everyone anyone "
                  "nobody wives"),
        ("pregnancy", "childhood infancy adolescence nursing breastfeeding lactation meals work holidays"),
        ("birth", "brain heart liver mental physical"),
        ("defects", "weight"),
        ("impairs", "improves improve enhances enhance increases increase boosts boost helps aids sharpens "
                    "benefits affects affect reduces reduce"),
        ("drive", "ride walk fly steer park own buy"),
        ("car", "vehicle truck boat bike bicycle plane horse bus"),
        ("operate", "use run own fix repair build"),
        ("machinery", "machines equipment tools vehicles computers"),
        ("cause", "prevent prevents cure cures avoid reduce solve treat ease"),
        ("health", "wealth heart money legal family financial"),
        ("problems", "benefits improvements"),
        ("surgeon", "attorney president secretary doctor physician governor"),
        ("drink", "eat consume buy sell serve enjoy use taste touch try smoke"),
    )


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
# --- Per-label time budget ---------------------------------------------------------------------
# The optional passes (turned and contrast views, then the RapidOCR escalation) run only while the
# time already spent on the label plus the pass's expected cost (a running average per process,
# seeded below) fits inside this budget; a pass that does not fit is skipped and the result's reader
# string says so. On a shared vCPU every pass costs two to three times what it costs on a laptop,
# and a hard label already reached 4.6 s there. 0 disables the budget.
LABEL_TIME_BUDGET_S = float(os.getenv("LABEL_TIME_BUDGET_S", "5"))
EXPECTED_PASS_COST_S = {"extend": 1.2, "escalate": 1.5}   # seeds until the process has measured its own

# --- RapidOCR, the second local reader (PP-OCRv4 on ONNX Runtime; app/readers/rapid.py) ---------
# Escalation: when Tesseract's passes leave a field missing or the warning short of PASS, read the
# label once more with RapidOCR and let the matching see its lines. Off: RAPID_ESCALATION=0.
RAPID_ESCALATION = os.getenv("RAPID_ESCALATION", "1").strip().lower() not in ("0", "false", "no", "off")
RAPID_INPUT = os.getenv("RAPID_INPUT", "gray")        # "gray": Tesseract's preprocessed image; "color": the scaled original
RAPID_TIMEOUT_S = 10                                    # a read past this is abandoned (the escalation is then skipped)
RAPID_MIN_LINE_CONF = 60                                # escalation lines below this confidence (0-100) are not appended
# Confidences of the two engines are not comparable (Tesseract: 80-96 for good words; RapidOCR: a
# 0-1 line score, usually 0.9-1.0). When two readings of one place disagree, the matcher keeps the
# disagreement only if the disagreeing reading was at least as confident as the agreeing one. A
# RapidOCR line at or above this score is treated as certain (100) for that comparison, so it wins
# against any Tesseract reading of the same place; below it, its score competes as read.
RAPID_TRUST_CONF = 90
RAPID_THREADS = int(os.getenv("RAPID_THREADS", "4"))   # ONNX Runtime threads per read (0 = the runtime's default)
# Reads in flight at once. Each worker process holds its own copy of the models: about 500-600 MB
# resident (measured), so 2 workers need about 1.2 GB; raise it on a container with memory to spare.
RAPID_WORKERS = int(os.getenv("RAPID_WORKERS", "2"))
# Run RapidOCR in worker processes (on by default): a crash in its native code then fails only that read
# instead of the whole server. Off (RAPID_ISOLATE=0): one engine shared by threads in the server process.
RAPID_ISOLATE = os.getenv("RAPID_ISOLATE", "1").strip().lower() not in ("0", "false", "no", "off")
RAPID_START_TIMEOUT_S = 60                              # a worker's first read also loads the models

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_TIMEOUT_S = float(os.getenv("CLAUDE_TIMEOUT_S", "30"))  # per attempt; the SDK default is 10 minutes
CLAUDE_MAX_RETRIES = 1
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLOUD_READER_AVAILABLE = bool(ANTHROPIC_API_KEY)

BATCH_JOBS_KEPT = 50                                    # finished batch jobs kept in memory
BATCH_JOB_TTL_S = 24 * 3600                             # ... and for at most this long
BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", str(min(4, os.cpu_count() or 2))))
# Review decisions: with DECISION_LOG set, every pass / fail / skip answer (and undo) from the review queue
# is appended to that file as one JSON line with what the tool had concluded about the label, never the
# image; scripts/decisions_report.py turns it into pass rates per rule. Off when unset. README: "Learning
# from decisions".
DECISION_LOG = os.getenv("DECISION_LOG", "")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000                           # decoded size limit (a 20 MB PNG can decode to gigabytes)
MIN_IMAGE_SIDE = 50                                     # smaller images cannot be read
MAX_BATCH_IMAGES = 500
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024           # all images of one batch together
MAX_ZIP_MEMBERS = 1000
MAX_ZIP_UNCOMPRESSED = 1024 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
