"""Text and unit normalization used before any comparison.

Two levels of text normalization:
  * normalize_strict  - unicode/whitespace cleanup only, case and punctuation kept.
  * normalize_loose   - also lower-cases, unifies "Alc./Vol." style phrases and drops punctuation.

Plus parsers that turn free-text alcohol content and net contents into numbers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "…": "...",
})

# Phrases that all mean "alcohol by volume". Order matters: longest first.
_ABV_PATTERNS = [
    r"alcohol\s+by\s+volume",
    r"alc(?:ohol)?\.?\s*/\s*vol(?:ume)?\.?",
    r"alc(?:ohol)?\.?\s+by\s+vol(?:ume)?\.?",
    r"alc\.?\s*vol\.?",
    r"\babv\b",
    r"\balc\.?(?=\s|$)",
]
_ABV_RE = re.compile("|".join(_ABV_PATTERNS), re.IGNORECASE)

# Punctuation that is NOT between two digits (keeps 45.5 and 1,000 intact).
_PUNCT_RE = re.compile(r"(?<!\d)[^\w\s%](?!\d)|(?<=\d)[^\w\s%.,](?!\d)|(?<!\d)[^\w\s%](?=\d)")
_WS_RE = re.compile(r"\s+")


def unify_unicode(text: str) -> str:
    """NFKC-normalize and map typographic quotes/dashes/spaces to ASCII."""
    return unicodedata.normalize("NFKC", text or "").translate(_QUOTE_MAP)


def fold_accents(text: str) -> str:
    """Drop diacritics ("Rosé" -> "Rose"): OCR and applications are inconsistent about them."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_strict(text: str) -> str:
    """Case and punctuation preserved; only unicode and whitespace are cleaned."""
    return collapse_ws(unify_unicode(text))


def unify_abv_phrases(text: str) -> str:
    return _ABV_RE.sub(" abv ", text)


def normalize_loose(text: str) -> str:
    """Lower-case, unify ABV phrases, drop punctuation (except inside numbers), collapse spaces."""
    t = fold_accents(unify_unicode(text)).lower()
    t = unify_abv_phrases(t)
    t = t.replace("_", " ")
    t = _PUNCT_RE.sub(" ", t)
    return collapse_ws(t)


def case_insensitive_equal(a: str, b: str) -> bool:
    return normalize_loose(a) == normalize_loose(b)


# --- OCR digit repair -------------------------------------------------------------------------
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})
# A run of digit-like characters not glued to the end of a word ("75O", "4O%", "l0 ml"; not "OLD").
_NUMERIC_RUN_RE = re.compile(r"(?<![A-Za-z])[0-9OolI|][0-9OolI|.,]*")


def _repair_run(m: re.Match) -> str:
    run = m.group(0)
    return run.translate(_DIGIT_FIX) if any(ch.isdigit() for ch in run) else run


# "1" read as "L" right before a decimal part or a volume unit: "L.75 L" -> "1.75 L", "LL" -> "1 L".
_L_BEFORE_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9])L(?=[.,]\d)")
_L_BEFORE_UNIT_RE = re.compile(r"(?<![A-Za-z0-9])L(?=\s?(?:L|l|ml|mL|ML|liters?|litres?|LITERS?|LITRES?)\b)")
_L_BEFORE_DIGITS_UNIT_RE = re.compile(r"(?<![A-Za-z0-9])L(?=\d+(?:[.,]\d+)?\s?(?:L|l|ml|mL|ML|liters?|litres?|LITERS?|LITRES?)\b)")


def fix_ocr_digits(token: str) -> str:
    """Repair common OCR confusions inside a token that is clearly a number ("75O" -> "750")."""
    token = _L_BEFORE_DECIMAL_RE.sub("1", token)
    token = _L_BEFORE_UNIT_RE.sub("1", token)
    token = _L_BEFORE_DIGITS_UNIT_RE.sub("1", token)
    return _NUMERIC_RUN_RE.sub(_repair_run, token)


def fix_ocr_digits_in_text(text: str) -> str:
    return fix_ocr_digits(text)


# --- Alcohol content --------------------------------------------------------------------------
@dataclass(frozen=True)
class AlcoholValue:
    abv: float | None            # percent alcohol by volume
    proof: float | None
    abv_from_proof: bool         # True when the label only stated proof
    text: str                    # the phrase we parsed it from

    def describe(self) -> str:
        if self.abv is None:
            return ""
        s = f"{self.abv:g}% ABV"
        if self.proof is not None:
            s += f" ({self.proof:g} proof)"
        return s


_PERCENT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{1,2})?)\s*%")
_MAX_PLAUSIBLE_ABV = 96.0  # anything above this is not an alcohol content ("100% agave")
_PROOF_RE = re.compile(r"(\d{1,3}(?:[.,]\d)?)\s*(?:°\s*)?proof\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*$")
_ALC_CONTEXT_RE = re.compile(r"alc|abv|vol", re.IGNORECASE)


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def parse_alcohol(text: str) -> AlcoholValue | None:
    """Pull alcohol by volume (and proof) out of free text.

    Accepts "45% Alc./Vol.", "ALC. 45% BY VOL.", "45% ABV", "90 Proof", "45".
    Proof is converted to ABV (proof / 2) when no percentage is stated.
    """
    if not text:
        return None
    t = fix_ocr_digits_in_text(unify_unicode(text))
    percents = [m for m in _PERCENT_RE.finditer(t) if _to_float(m.group(1)) <= _MAX_PLAUSIBLE_ABV]
    proofs = list(_PROOF_RE.finditer(t))

    percent_match = None
    if percents:
        # Prefer a percentage that sits next to an "alc"/"abv"/"vol" word.
        for m in percents:
            window = t[max(0, m.start() - 30): m.end() + 30]
            if _ALC_CONTEXT_RE.search(window):
                percent_match = m
                break
        if percent_match is None:
            percent_match = percents[0]

    proof_val = _to_float(proofs[0].group(1)) if proofs else None

    if percent_match is not None:
        abv = _to_float(percent_match.group(1))
        start = percent_match.start()
        end = max(percent_match.end(), proofs[0].end()) if proofs else percent_match.end()
        return AlcoholValue(abv=abv, proof=proof_val, abv_from_proof=False, text=collapse_ws(t[start:end]))
    if proof_val is not None:
        return AlcoholValue(abv=proof_val / 2.0, proof=proof_val, abv_from_proof=True,
                            text=collapse_ws(proofs[0].group(0)))
    bare = _BARE_NUMBER_RE.match(t)
    if bare:
        return AlcoholValue(abv=_to_float(bare.group(1)), proof=None, abv_from_proof=False, text=t.strip())
    return None


# --- Net contents -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeValue:
    ml: float
    unit: str        # canonical unit as written on the source ("mL", "L", "fl oz", "cL")
    text: str        # the phrase we parsed it from
    digits: str = ""  # the digits as printed, without the decimal separator ("1.75" -> "175")

    def describe(self) -> str:
        return f"{self.ml:g} mL"


_UNIT_TO_ML = {
    "ml": 1.0, "milliliter": 1.0, "millilitre": 1.0,
    "cl": 10.0, "centiliter": 10.0, "centilitre": 10.0,
    "l": 1000.0, "liter": 1000.0, "litre": 1000.0,
    "floz": 29.5735, "fluidounce": 29.5735, "oz": 29.5735, "ounce": 29.5735,
}
_CANON_UNIT = {
    "ml": "mL", "milliliter": "mL", "millilitre": "mL",
    "cl": "cL", "centiliter": "cL", "centilitre": "cL",
    "l": "L", "liter": "L", "litre": "L",
    "floz": "fl oz", "fluidounce": "fl oz", "oz": "fl oz", "ounce": "fl oz",
}
_VOLUME_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(m\s?l|milliliters?|millilitres?|c\s?l|centiliters?|centilitres?|"
    r"fl\.?\s*oz\.?|fluid\s*ounces?|oz\.?|ounces?|liters?|litres?|l)\b\.?",
    re.IGNORECASE,
)


def _canon_unit_key(raw: str) -> str:
    u = re.sub(r"[\s.]", "", raw.lower())
    if u.endswith("s") and u not in ("oz",):
        u = u[:-1]
    return u


def parse_net_contents(text: str) -> VolumeValue | None:
    """Pull a volume out of free text and express it in millilitres.

    Accepts "750 mL", "750ml", "1.75 L", "12 FL. OZ.", "70 cl". When several volumes
    appear (e.g. "12 FL OZ (355 mL)") the metric one wins because it is exact.
    """
    if not text:
        return None
    t = fix_ocr_digits_in_text(unify_unicode(text))
    candidates = []
    for m in _VOLUME_RE.finditer(t):
        key = _canon_unit_key(m.group(2))
        if key not in _UNIT_TO_ML:
            continue
        value = _to_float(m.group(1))
        candidates.append(VolumeValue(ml=round(value * _UNIT_TO_ML[key], 2), unit=_CANON_UNIT[key],
                                      text=collapse_ws(m.group(0)), digits=re.sub(r"[.,]", "", m.group(1))))
    if not candidates:
        return None
    metric = [c for c in candidates if c.unit in ("mL", "L", "cL")]
    return metric[0] if metric else candidates[0]
