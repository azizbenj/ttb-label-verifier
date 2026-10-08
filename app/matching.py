"""Compare what the application says with what we read on the label.

Text fields are located on the label with an application-guided fuzzy search
(we know what we are looking for, so we look for the best-matching span of words
instead of trying to parse an arbitrary label blindly). Alcohol content and net
contents are compared as numbers after unit conversion.

All thresholds come from app.config.THRESHOLDS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .config import FIELD_BY_KEY, THRESHOLDS, FieldSpec, Thresholds
from .models import FieldResult, Verdict
from .normalize import normalize_loose, normalize_strict, parse_alcohol, parse_net_contents

_EDGE_PUNCT = " ,.;:-|'\"()[]"


@dataclass(frozen=True)
class Located:
    text: str        # the span of label text that best matches, original case kept
    score: int       # similarity (0-100) between that span and the expected value
    line_start: int  # first OCR line index used
    line_end: int    # last OCR line index used (inclusive)


def locate_text(expected: str, lines: list[str], *, max_window: int = 3,
                floor: int | None = None, th: Thresholds = THRESHOLDS) -> Located | None:
    """Find the span of consecutive label words that best matches ``expected``.

    Windows of up to ``max_window`` consecutive OCR lines are joined so that values
    wrapped over several lines (addresses, long class/type names) are still found.
    Returns None when nothing on the label scores at least ``floor``.
    """
    floor = th.find_floor if floor is None else floor
    exp_loose = normalize_loose(expected)
    if not exp_loose or not lines:
        return None
    n_exp = max(1, len(normalize_strict(expected).split()))
    min_len, max_len = max(1, n_exp - 3), n_exp + 3

    raw_tokens: list[list[str]] = [normalize_strict(line).split() for line in lines]
    loose_tokens: list[list[str]] = [[normalize_loose(t) for t in toks] for toks in raw_tokens]

    best: Located | None = None
    for i in range(len(lines)):
        raw: list[str] = []
        loose: list[str] = []
        for w in range(max_window):
            j = i + w
            if j >= len(lines):
                break
            last_line_start = len(raw)
            raw.extend(raw_tokens[j])
            loose.extend(loose_tokens[j])
            if not raw_tokens[j]:
                continue
            for a in range(len(raw)):
                for span_len in range(min_len, max_len + 1):
                    b = a + span_len
                    if b > len(raw):
                        break
                    if b <= last_line_start:   # already scored inside a smaller window
                        continue
                    cand = " ".join(t for t in loose[a:b] if t)
                    if not cand:
                        continue
                    score = int(round(fuzz.ratio(exp_loose, cand)))
                    if best is None or score > best.score:
                        text = " ".join(raw[a:b]).strip(_EDGE_PUNCT)
                        best = Located(text=text, score=score, line_start=i, line_end=j)
    if best is None or best.score < floor:
        return None
    return best


def _spec(key: str) -> FieldSpec:
    return FIELD_BY_KEY[key]


def skipped(key: str, expected: str = "") -> FieldResult:
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None,
                       verdict=Verdict.SKIPPED, score=0, note="Not provided on the application, so not checked.")


def not_found(key: str, expected: str, fallback_found: str | None = None, note: str = "") -> FieldResult:
    if fallback_found:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=fallback_found,
                           verdict=Verdict.MISMATCH, score=0,
                           note=note or "Nothing on the label resembles the application value.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None,
                       verdict=Verdict.NOT_FOUND, score=0, note=note or "We could not find this on the label.")


def compare_text(key: str, expected: str, found: str | None, *, th: Thresholds = THRESHOLDS) -> FieldResult:
    """Classify a located text span against the application value."""
    label = _spec(key).label
    if found is None:
        return not_found(key, expected)
    if normalize_strict(expected) == normalize_strict(found):
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Exact match.")
    exp_loose, found_loose = normalize_loose(expected), normalize_loose(found)
    if exp_loose == found_loose:
        if key in th.case_review_fields:
            return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.NEAR_MATCH,
                               score=100, note="Same words, but capitalization or punctuation differs. Please confirm.")
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Matches (capitalization and punctuation ignored).")
    score = int(round(max(fuzz.ratio(exp_loose, found_loose), fuzz.token_sort_ratio(exp_loose, found_loose) - 2)))
    if score >= th.near_match:
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.NEAR_MATCH,
                           score=score, note=f"Very similar ({score}% alike) but not identical. Please confirm.")
    return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MISMATCH,
                       score=score, note=f"Does not match the application ({score}% alike).")


def locate_and_compare(key: str, expected: str, lines: list[str], *, fallback_found: str | None = None,
                       th: Thresholds = THRESHOLDS) -> FieldResult:
    spec = _spec(key)
    if not expected.strip():
        return skipped(key) if not spec.required else not_found(key, expected, note="Required on the application.")
    loc = locate_text(expected, lines, th=th)
    if loc is None:
        return not_found(key, expected, fallback_found=fallback_found)
    return compare_text(key, expected, loc.text, th=th)


def compare_alcohol(expected: str, label_text: str, *, th: Thresholds = THRESHOLDS) -> FieldResult:
    key = "alcohol_content"
    exp = parse_alcohol(expected)
    if exp is None or exp.abv is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note="The application value could not be read as an alcohol content "
                                "(try '45% Alc./Vol.' or '90 Proof').")
    got = parse_alcohol(label_text)
    if got is None or got.abv is None:
        return not_found(key, expected, note="No alcohol content (e.g. '45% Alc./Vol.' or '90 Proof') was found on the label.")
    notes = []
    if got.abv_from_proof:
        notes.append(f"Label states {got.proof:g} proof, which is {got.abv:g}% ABV.")
    if got.proof is not None and not got.abv_from_proof and abs(got.proof / 2.0 - got.abv) > th.abv_tolerance:
        notes.append(f"The label's proof ({got.proof:g}) does not agree with its percentage ({got.abv:g}%).")
    if abs(got.abv - exp.abv) <= th.abv_tolerance:
        verdict, score = Verdict.MATCH, 100
        notes.insert(0, f"{exp.abv:g}% ABV on both.")
    else:
        verdict, score = Verdict.MISMATCH, 0
        notes.insert(0, f"Label says {got.abv:g}% ABV, application says {exp.abv:g}% ABV.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=verdict,
                       score=score, note=" ".join(notes))


def compare_volume(expected: str, label_text: str, *, th: Thresholds = THRESHOLDS) -> FieldResult:
    key = "net_contents"
    exp = parse_net_contents(expected)
    if exp is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note="The application value could not be read as a volume (try '750 mL' or '1.75 L').")
    got = parse_net_contents(label_text)
    if got is None:
        return not_found(key, expected, note="No net contents (e.g. '750 mL') was found on the label.")
    if abs(got.ml - exp.ml) <= th.volume_tolerance_ml:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.MATCH,
                           score=100, note=f"{exp.describe()} on both.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.MISMATCH,
                       score=0, note=f"Label says {got.describe()}, application says {exp.describe()}.")


_ORIGIN_RE = re.compile(
    r"\b(product of|produce of|produced in|made in|imported from|distilled in|bottled in)\s+"
    r"([A-Za-z][A-Za-z .'-]{2,40}?)(?=[,.;:)\n]|\s{2,}|$)",
    re.IGNORECASE,
)


def compare_country(expected: str, lines: list[str], *, th: Thresholds = THRESHOLDS) -> FieldResult:
    """Country of origin (imports only). Blank on the application means domestic: skipped."""
    key = "country_of_origin"
    if not expected.strip():
        return skipped(key)
    country = expected.strip()
    for prefix in ("product of", "produce of", "made in", "imported from"):
        if country.lower().startswith(prefix):
            country = country[len(prefix):].strip(" :,-")
            break
    # 1. The country name itself, anywhere on the label (high bar: short strings fuzz easily).
    loc = locate_text(country, lines, floor=max(th.near_match, 85), th=th)
    if loc is not None:
        # Show the whole origin statement when there is one ("Product of Scotland").
        text = "\n".join(lines[max(0, loc.line_start - 1): loc.line_end + 1])
        m = _ORIGIN_RE.search(text)
        found = m.group(0).strip() if m and normalize_loose(country) in normalize_loose(m.group(2)) else loc.text
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Country of origin found on the label.")
    # 2. A different origin statement is present: that is a mismatch, and we show it.
    m = _ORIGIN_RE.search("\n".join(lines))
    if m:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=m.group(0).strip(),
                           verdict=Verdict.MISMATCH, score=0,
                           note=f"Label says '{m.group(0).strip()}', application says '{country}'.")
    return not_found(key, expected, note="No country of origin statement was found on the label.")
