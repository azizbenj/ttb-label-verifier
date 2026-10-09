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
    r"\bby\s+vol(?:ume)?\.?",
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


_MARKS_RE = re.compile(r"\s*[\u00ae\u2122\u00a9\u2120]")   # (R) TM (C) SM marks: not part of a name


def normalize_strict(text: str) -> str:
    """Case and punctuation preserved; only unicode, whitespace and trademark marks are cleaned."""
    return collapse_ws(_MARKS_RE.sub("", unify_unicode(text)))


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
# "90 Proof" or, as some labels print it, "PROOF 102".
_PROOF_RE = re.compile(r"(\d{1,3}(?:[.,]\d)?)\s*(?:°\s*)?proof\b|\bproof\s*:?\s*(\d{2,3}(?:[.,]\d)?)\b",
                       re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*$")
# Words that mark a percentage as alcohol by volume, allowing OCR's usual confusions ("ALG/VGL",
# "AIC/V0L"). Not "alcoholic" (the warning statement says "alcoholic beverages") and not "volcanic":
# those must not turn a blend percentage into an ABV.
_ALC_CONTEXT_RE = re.compile(r"\b(?:a[l1i][cg](?!oholic)|abv|v[o0g][l1i](?!can))", re.IGNORECASE)


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def _context_starts(t: str) -> list[int]:
    # Searched on the whole text, not on a slice: a slice can end inside "volc|anic" and defeat the lookahead.
    return [m.start() for m in _ALC_CONTEXT_RE.finditer(t)]


def _near_context(starts: list[int], m: re.Match, reach: int = 30) -> bool:
    return any(m.start() - reach <= s < m.end() + reach for s in starts)


def _proof_value(m: re.Match) -> float:
    return _to_float(m.group(1) or m.group(2))


def parse_alcohol(text: str) -> AlcoholValue | None:
    """Pull alcohol by volume (and proof) out of free text.

    Accepts "45% Alc./Vol.", "ALC. 45% BY VOL.", "45% ABV", "90 Proof", "45".
    Proof is converted to ABV (proof / 2) when no percentage is stated.
    """
    if not text:
        return None
    t = fix_ocr_digits(unify_unicode(text))
    percents = [m for m in _PERCENT_RE.finditer(t) if _to_float(m.group(1)) <= _MAX_PLAUSIBLE_ABV]
    proofs = list(_PROOF_RE.finditer(t))

    percent_match = None
    if percents:
        # Prefer a percentage that sits next to an "alc"/"abv"/"vol" word.
        starts = _context_starts(t)
        for m in percents:
            if _near_context(starts, m):
                percent_match = m
                break
        if percent_match is None:
            percent_match = percents[0]

    proof_val = _proof_value(proofs[0]) if proofs else None

    if percent_match is not None:
        abv = _to_float(percent_match.group(1))
        start = percent_match.start()
        end = percent_match.end()
        # The proof belongs to this statement only when it is printed right next to the percentage.
        near = [m for m in proofs if 0 <= m.start() - percent_match.end() <= 25]
        if near:
            proof_val = _proof_value(near[0])
            end = near[0].end()
        elif not (len(proofs) == 1 and len(percents) == 1):
            proof_val = None
        # Show the whole statement ("45% Alc./Vol.", "ALC. 14.5% BY VOL.") rather than the bare number.
        left = t[max(0, start - 14):start]
        m_left = None
        for m in _ABV_RE.finditer(left):
            m_left = m
        if m_left and not left[m_left.end():].strip():
            start = max(0, start - 14) + m_left.start()
        right = t[end:end + 25]
        m_right = _ABV_RE.match(right.lstrip())
        if m_right:
            end += (len(right) - len(right.lstrip())) + m_right.end()
        if t[end:end + 1] == ")" and "(" in t[start:end]:
            end += 1
        return AlcoholValue(abv=abv, proof=proof_val, abv_from_proof=False, text=collapse_ws(t[start:end]))
    if proof_val is not None:
        return AlcoholValue(abv=proof_val / 2.0, proof=proof_val, abv_from_proof=True,
                            text=collapse_ws(proofs[0].group(0)))
    bare = _BARE_NUMBER_RE.match(t)
    if bare:
        return AlcoholValue(abv=_to_float(bare.group(1)), proof=None, abv_from_proof=False, text=t.strip())
    return None


def alcohol_candidates(text: str) -> list[AlcoholValue]:
    """Every alcohol statement read on the label: each percentage next to an alc/abv/vol word, and
    each proof figure with no percentage beside it. Several OCR passes can read the same statement
    differently, and labels often state it twice, so the matcher looks at all of them."""
    if not text:
        return []
    t = fix_ocr_digits(unify_unicode(text))
    out: list[AlcoholValue] = []
    used: set[int] = set()
    proofs = list(_PROOF_RE.finditer(t))
    starts = _context_starts(t)
    for m in _PERCENT_RE.finditer(t):
        value = _to_float(m.group(1))
        if value > _MAX_PLAUSIBLE_ABV:
            continue
        if not _near_context(starts, m):
            continue
        proof = None
        for pm in proofs:
            if 0 <= pm.start() - m.end() <= 25 or 0 <= m.start() - pm.end() <= 6:
                proof = _proof_value(pm)
                used.add(pm.start())
                break
        piece = parse_alcohol(t[max(0, m.start() - 14): m.end() + 40])
        shown = piece.text if piece is not None and piece.abv == value else collapse_ws(m.group(0))
        out.append(AlcoholValue(abv=value, proof=proof, abv_from_proof=False, text=shown))
    for pm in proofs:
        p = _proof_value(pm)
        if pm.start() not in used and 20 <= p <= 192:
            out.append(AlcoholValue(abv=p / 2.0, proof=p, abv_from_proof=True, text=collapse_ws(pm.group(0))))
    return out


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
    "pt": 473.176, "pint": 473.176, "qt": 946.353, "quart": 946.353, "gal": 3785.41, "gallon": 3785.41,
}
_CANON_UNIT = {
    "ml": "mL", "milliliter": "mL", "millilitre": "mL",
    "cl": "cL", "centiliter": "cL", "centilitre": "cL",
    "l": "L", "liter": "L", "litre": "L",
    "floz": "fl oz", "fluidounce": "fl oz", "oz": "fl oz", "ounce": "fl oz",
    "pt": "pt", "pint": "pt", "qt": "qt", "quart": "qt", "gal": "gal", "gallon": "gal",
}
_METRIC_UNITS = ("mL", "L", "cL")
# The number must not be glued to a digit look-alike ("7S0 mL" is not "0 mL"; the repair below reads it).
_VOLUME_RE = re.compile(
    r"(?<![0-9.,LlIOoSsB])(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)\s*(?:u\.?\s?s\.?\s+)?"
    r"(m\s?l|milliliters?|millilitres?|c\s?l|centiliters?|centilitres?|"
    r"fl\.?\s*oz\.?|fluid\s*ounces?|oz\.?|ounces?|liters?|litres?|l|"
    r"pints?|pt\.?|quarts?|qt\.?|gallons?|gal\.?)\b\.?",
    re.IGNORECASE,
)
_THOUSANDS_RE = re.compile(r"^[1-9]\d{0,2}(?:,\d{3})+$")   # "1,000" or "1,750" (but "0,750" is a decimal comma)


def _volume_number(s: str) -> float:
    return float(s.replace(",", "")) if _THOUSANDS_RE.match(s) else _to_float(s)


def _canon_unit_key(raw: str) -> str:
    u = re.sub(r"[\s.]", "", raw.lower())
    if u.endswith("s") and u not in ("oz",):
        u = u[:-1]
    return u


# A volume whose digits were read as look-alike letters ("LSL" for "1.5 L", "7S0 mL"): only
# considered when no ordinary volume is present, and only right in front of a volume unit.
_MANGLED_VOLUME_RE = re.compile(
    r"(?<![A-Za-z0-9])([0-9LlIOoSsB][0-9LlIOoSsB.,]{0,5})\s?(L|ml|mL|ML|liters?|litres?|LITERS?|LITRES?)\b", )
_LOOKALIKE_DIGITS = str.maketrans({"L": "1", "l": "1", "I": "1", "O": "0", "o": "0", "S": "5", "s": "5", "B": "8"})


_ADDRESS_CONTEXT_RE = re.compile(r",\s*$")
_ZIP_AFTER_RE = re.compile(r"^\s*\d{5}\b")


def _repair_mangled_volumes(t: str) -> str:
    def fix(m: re.Match) -> str:
        number, unit = m.group(1), m.group(2)
        glued = m.group(0)[len(number):len(number) + 1] != " "
        if not any(ch.isalpha() for ch in number):
            return m.group(0)                       # an ordinary number: nothing to repair
        if not any(ch.isdigit() for ch in number):
            if not (glued and len(number) <= 3):
                return m.group(0)                   # a real word ("BOLS L"), not a mangled number
            # "..., IL 60607": a state code in an address, not a volume
            if _ADDRESS_CONTEXT_RE.search(t[:m.start()]) or _ZIP_AFTER_RE.match(t[m.end():]):
                return m.group(0)
        return number.translate(_LOOKALIKE_DIGITS) + " " + unit
    return _MANGLED_VOLUME_RE.sub(fix, t)


_WORD_NUMBER_RE = re.compile(
    r"\b(one|two|three|four|five|half(?:\s+a)?)\s+(?=(?:u\.?s\.?\s+)?(?:pints?|quarts?|gallons?|liters?|litres?)\b)",
    re.IGNORECASE)
_WORD_NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "half": "0.5", "half a": "0.5"}


def _volume_candidates(t: str) -> list[VolumeValue]:
    # "ONE PINT", "HALF A GALLON"
    t = _WORD_NUMBER_RE.sub(lambda m: _WORD_NUMBERS[" ".join(m.group(1).lower().split())] + " ", t)
    found: list[tuple[re.Match, VolumeValue]] = []
    for m in _VOLUME_RE.finditer(t):
        key = _canon_unit_key(m.group(2))
        if key not in _UNIT_TO_ML:
            continue
        value = _volume_number(m.group(1))
        found.append((m, VolumeValue(ml=round(value * _UNIT_TO_ML[key], 2), unit=_CANON_UNIT[key],
                                     text=collapse_ws(m.group(0)), digits=re.sub(r"[.,]", "", m.group(1)))))
    # US compound statements: "1 PINT 6 FL. OZ." is 22 fl oz, not 6.
    out: list[VolumeValue] = []
    i = 0
    while i < len(found):
        m, v = found[i]
        if v.unit in ("pt", "qt", "gal") and i + 1 < len(found):
            m2, v2 = found[i + 1]
            if v2.unit == "fl oz" and not t[m.end():m2.start()].strip(" ,&"):
                out.append(VolumeValue(ml=round(v.ml + v2.ml, 2), unit="fl oz",
                                       text=collapse_ws(t[m.start():m2.end()]), digits=v.digits + v2.digits))
                i += 2
                continue
        out.append(v)
        i += 1
    return out


def volume_candidates(text: str) -> list[VolumeValue]:
    """Every net contents statement read on the label (compound US statements combined)."""
    if not text:
        return []
    t = fix_ocr_digits(unify_unicode(text))
    return _volume_candidates(t) or _volume_candidates(_repair_mangled_volumes(t))


def parse_net_contents(text: str) -> VolumeValue | None:
    """Pull a volume out of free text and express it in millilitres.

    Accepts "750 mL", "750ml", "1.75 L", "1,000 mL", "12 FL. OZ.", "70 cl", "1 PINT 6 FL. OZ.". When
    several volumes appear (e.g. "12 FL OZ (355 mL)") the metric one wins because it is exact.
    Digits read as look-alike letters are repaired only when no ordinary volume is present, so a
    word such as "SOIL" is never turned into "501 L" in front of the real statement.
    """
    if not text:
        return None
    t = fix_ocr_digits(unify_unicode(text))
    candidates = _volume_candidates(t) or _volume_candidates(_repair_mangled_volumes(t))
    if not candidates:
        return None
    metric = [c for c in candidates if c.unit in _METRIC_UNITS]
    return metric[0] if metric else candidates[0]
