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
from rapidfuzz.distance import Levenshtein

from .normalize import (alcohol_candidates, normalize_loose, normalize_strict, parse_alcohol, parse_net_contents,
                        volume_candidates)

_EDGE_PUNCT = " ,.;:-|'\"()[]"
_SEPARATORS = set(",;:|/·•-()[].")  # punctuation that ends a phrase on a label line ("RUM · AGED 4 YEARS")


@dataclass(frozen=True)
class Located:
    text: str          # the span of label text that best matches, original case kept
    score: int         # similarity (0-100) between that span and the expected value
    line_start: int    # OCR line holding the span's first word
    line_end: int      # OCR line holding the span's last word (inclusive)
    joined: bool = False  # the span runs straight into other words on its line ("SPICED RUM" for "Rum")


def _is_word(token: str) -> bool:
    return sum(ch.isalpha() for ch in token) >= 2


def _joined(before: list[str], span: list[str], after: list[str]) -> bool:
    """Is the span part of a longer phrase on its line, with no punctuation in between?"""
    left = bool(before) and _is_word(before[-1]) and before[-1][-1] not in _SEPARATORS and span[0][0] not in _SEPARATORS
    right = bool(after) and _is_word(after[0]) and span[-1][-1] not in _SEPARATORS and after[0][0] not in _SEPARATORS
    return left or right


def locate_text(expected: str, lines: list[str], *, max_window: int | None = None,
                floor: int | None = None, preferred_line: int | None = None,
                th: Thresholds = THRESHOLDS) -> Located | None:
    """Find the span of consecutive label words that best matches ``expected``.

    Spans may run over up to ``max_window`` consecutive OCR lines so that values wrapped over
    several lines (addresses, long class/type names) are still found. Returns None when nothing
    on the label scores at least ``floor``. Ties between equally good spans go to the one on
    ``preferred_line`` (the most prominent line, for the brand name), then to the one that also
    matches case.
    """
    floor = th.find_floor if floor is None else floor
    max_window = th.locate_max_lines if max_window is None else max_window
    exp_loose = normalize_loose(expected)
    if not exp_loose or not lines:
        return None
    n_exp = max(1, len(normalize_strict(expected).split()))
    min_len, max_len = max(1, n_exp - 3), n_exp + 3

    raw_tokens: list[list[str]] = [normalize_strict(line).split() for line in lines]
    loose_tokens: list[list[str]] = [[normalize_loose(t) for t in toks] for toks in raw_tokens]
    exp_strict = normalize_strict(expected).strip(_EDGE_PUNCT)

    best: Located | None = None
    best_rank: tuple = ()
    for i, first_line in enumerate(raw_tokens):
        # Spans start on line i and end on line j; spans starting later are found when i gets there.
        raw: list[str] = []
        loose: list[str] = []
        for j in range(i, min(len(lines), i + max_window)):
            ends_before = len(raw)
            raw.extend(raw_tokens[j])
            loose.extend(loose_tokens[j])
            if not raw_tokens[j]:
                continue
            for a in range(len(first_line)):
                for span_len in range(min_len, max_len + 1):
                    b = a + span_len
                    if b > len(raw):
                        break
                    if b <= ends_before:   # ends on an earlier line: scored with a shorter window
                        continue
                    cand = " ".join(t for t in loose[a:b] if t)
                    if not cand:
                        continue
                    score = int(round(fuzz.ratio(exp_loose, cand)))
                    if best is not None and score < best.score:
                        continue
                    text = " ".join(raw[a:b]).strip(_EDGE_PUNCT)
                    rank = (score, preferred_line is not None and i == j == preferred_line, text == exp_strict)
                    if best is None or rank > best_rank:
                        after = raw_tokens[j][b - ends_before:]
                        best = Located(text=text, score=score, line_start=i, line_end=j,
                                       joined=_joined(first_line[:a], raw[a:b], after))
                        best_rank = rank
    if best is None or best.score < th.near_match:
        spaced = _locate_letter_spaced(exp_loose, raw_tokens, loose_tokens, max_window)
        if spaced is not None and (best is None or spaced.score > best.score):
            best = spaced
    if best is None or best.score < floor:
        return None
    return best


def _despaced(s: str) -> str:
    return s.replace(" ", "")


def _letter_spaced(s: str) -> bool:
    """Mostly single characters: "S O U T H  C O A S T"."""
    toks = s.split()
    return len(toks) >= 4 and sum(1 for t in toks if len(t) == 1) >= 0.6 * len(toks)


def _locate_letter_spaced(exp_loose: str, raw_tokens: list[list[str]], loose_tokens: list[list[str]],
                          max_window: int) -> Located | None:
    """Brands set with wide letter spacing come back as single letters. Compare without spaces, on
    lines made mostly of single characters."""
    exp_ds = _despaced(exp_loose)
    if len(exp_ds) < 4:
        return None
    best: Located | None = None
    for i in range(len(raw_tokens)):
        raw: list[str] = []
        loose: list[str] = []
        for j in range(i, min(len(raw_tokens), i + max_window)):
            raw += raw_tokens[j]
            loose += [t for t in loose_tokens[j] if t]
            if not _letter_spaced(" ".join(loose)):
                continue
            joined = "".join(loose)
            if len(joined) > 1.6 * len(exp_ds):
                break
            score = int(round(fuzz.partial_ratio(exp_ds, joined)))
            if best is None or score > best.score:
                best = Located(text=" ".join(raw).strip(_EDGE_PUNCT), score=score, line_start=i, line_end=j)
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
    if normalize_strict(expected).strip(_EDGE_PUNCT) == normalize_strict(found).strip(_EDGE_PUNCT):
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Exact match.")
    exp_loose, found_loose = normalize_loose(expected), normalize_loose(found)
    if exp_loose != found_loose and _letter_spaced(found_loose) and _despaced(exp_loose) == _despaced(found_loose):
        # Letter-spaced on the label ("S O U T H  C O A S T  W I N E R Y"): the same letters in order.
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Matches (the label spaces the letters out).")
    if exp_loose == found_loose:
        if key in th.case_review_fields:
            return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.NEAR_MATCH,
                               score=100, note="Same words, but capitalization or punctuation differs. Please confirm.")
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MATCH,
                           score=100, note="Matches (capitalization and punctuation ignored).")
    score = int(round(max(fuzz.ratio(exp_loose, found_loose), fuzz.token_sort_ratio(exp_loose, found_loose) - 2)))
    if score < th.near_match and len(exp_loose) >= 5 and Levenshtein.distance(exp_loose, found_loose) == 1:
        # One letter apart ("CON PAZ" / "CON FAZ"): as likely a misread as a different name. Never a
        # silent match, but not a hard failure either: the agent looks at the crop.
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.NEAR_MATCH,
                           score=score, note="One letter differs. This may be a reading error or a different name. "
                                             "Please confirm.")
    if score >= th.near_match:
        return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.NEAR_MATCH,
                           score=score, note=f"Very similar ({score}% alike) but not identical. Please confirm.")
    return FieldResult(key=key, label=label, expected=expected, found=found, verdict=Verdict.MISMATCH,
                       score=score, note=f"Does not match the application ({score}% alike).")


def locate_and_compare(key: str, expected: str, lines: list[str], *, fallback_found: str | None = None,
                       preferred_line: int | None = None, line_heights: dict[int, float] | None = None,
                       th: Thresholds = THRESHOLDS) -> FieldResult:
    """Locate ``expected`` on the label and classify it.

    A MATCH found inside a longer phrase ("Rum" in "SPICED RUM", "OLD TOM" in "OLD TOM DISTILLERY")
    or, for the brand, only in small print ("Bottled by River Bend Brewing Co.") is downgraded to a
    NEAR MATCH: the words are on the label, but the label may not say what the application says.
    ``line_heights`` (median word height per OCR line) enables the small-print check.
    """
    spec = _spec(key)
    if not expected.strip():
        return skipped(key) if not spec.required else not_found(key, expected, note="Required on the application.")
    loc = locate_text(expected, lines, preferred_line=preferred_line, th=th)
    if loc is None:
        return not_found(key, expected, fallback_found=fallback_found)
    result = compare_text(key, expected, loc.text, th=th)
    result.lines = [loc.line_start, loc.line_end]
    if result.verdict != Verdict.MATCH or key not in th.whole_phrase_fields:
        return result
    context = normalize_strict(" ".join(lines[loc.line_start: loc.line_end + 1]))
    if loc.joined:
        return result.model_copy(update={
            "verdict": Verdict.NEAR_MATCH, "found": context,
            "note": f"The application value is only part of what the label says ('{context}'). Please confirm."})
    if key == "brand_name" and line_heights:
        span_h = max((line_heights.get(i, 0.0) for i in range(loc.line_start, loc.line_end + 1)), default=0.0)
        if span_h and span_h < th.brand_small_print_ratio * max(line_heights.values()):
            biggest = f" The largest text on the label reads '{fallback_found}'." if fallback_found else ""
            return result.model_copy(update={
                "verdict": Verdict.NEAR_MATCH, "found": context,
                "note": f"Found only in small print ('{context}'), not as the brand on the label.{biggest} Please confirm."})
    return result


def compare_alcohol(expected: str, label_text: str, *, statement: bool = False,
                    th: Thresholds = THRESHOLDS) -> FieldResult:
    """Alcohol content compared as numbers, over every alcohol statement read on the label.

    ``statement``: ``label_text`` is already the alcohol statement (a vision model's field), so a bare
    "45%" in it counts. On OCR text a percentage only counts when "Alc./Vol.", "ABV" or a proof figure
    marks it as alcohol: "13.5% Petit Verdot" on a wine label whose real statement was not read must
    never match an application of 13.5%.
    """
    key = "alcohol_content"
    exp = parse_alcohol(expected)
    if exp is None or exp.abv is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note="The application value could not be read as an alcohol content "
                                "(try '45% Alc./Vol.' or '90 Proof').")
    cands = alcohol_candidates(label_text)
    if not cands:
        got = parse_alcohol(label_text)
        if got is not None and got.abv is not None and not statement:
            if abs(got.abv - exp.abv) <= th.abv_tolerance:
                return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text,
                                   verdict=Verdict.NEAR_MATCH, score=60,
                                   note=f"{got.text} is on the label, but not marked as alcohol by volume ('Alc./Vol.', "
                                        "'ABV' or a proof figure), so it may be something else, such as a blend "
                                        "percentage. Please confirm the alcohol statement.")
            return not_found(key, expected, note="No alcohol content (e.g. '45% Alc./Vol.' or '90 Proof') was found on "
                                                 f"the label. The only percentage read, '{got.text}', is not marked as "
                                                 "alcohol by volume.")
        cands = [got] if got is not None and got.abv is not None else []
    if not cands:
        return not_found(key, expected, note="No alcohol content (e.g. '45% Alc./Vol.' or '90 Proof') was found on the label.")
    matching = [c for c in cands if abs(c.abv - exp.abv) <= th.abv_tolerance]
    got = matching[0] if matching else cands[0]
    notes = []
    if got.abv_from_proof:
        notes.append(f"Label states {got.proof:g} proof, which is {got.abv:g}% ABV.")
    inconsistent = got.proof is not None and not got.abv_from_proof and abs(got.proof - 2 * got.abv) > th.proof_tolerance
    if inconsistent:
        notes.append(f"The label's proof ({got.proof:g}) does not agree with its percentage ({got.abv:g}%).")
    others = sorted({round(c.abv, 2) for c in cands if abs(c.abv - exp.abv) > th.abv_tolerance})
    if matching and others:
        return FieldResult(key=key, label=_spec(key).label, expected=expected,
                           found="; ".join(dict.fromkeys(c.text for c in cands)), verdict=Verdict.NEAR_MATCH, score=80,
                           note=f"{exp.abv:g}% ABV is on the label, but it also reads "
                                f"{', '.join(f'{v:g}%' for v in others)}. This may be a reading error or a second "
                                "statement. Please confirm.")
    if matching:
        # A label contradicting itself is never a silent match, even when its percentage agrees.
        verdict, score = (Verdict.NEAR_MATCH, 90) if inconsistent else (Verdict.MATCH, 100)
        notes.insert(0, f"{exp.abv:g}% ABV on both.")
        if inconsistent:
            notes.append("Please confirm.")
    else:
        verdict, score = Verdict.MISMATCH, 0
        notes.insert(0, f"Label says {got.abv:g}% ABV, application says {exp.abv:g}% ABV.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=verdict,
                       score=score, note=" ".join(notes))


# Standard container sizes (27 CFR 5.203 spirits, 4.72 wine, common malt beverage packages), in mL.
_STANDARD_ML = {50, 100, 180, 187, 200, 250, 300, 331, 350, 355, 365, 375, 473, 475, 500, 568, 570, 600, 620, 650,
                700, 710, 720, 750, 900, 945, 946, 1000, 1500, 1750, 1800, 2000, 2250, 3000, 3750}


def _one_digit_apart(a: int, b: int) -> bool:
    sa, sb = str(a), str(b)
    return len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) == 1


def compare_volume(expected: str, label_text: str, *, th: Thresholds = THRESHOLDS) -> FieldResult:
    """Net contents compared in millilitres, over every volume statement read on the label."""
    key = "net_contents"
    exp = parse_net_contents(expected)
    if exp is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note="The application value could not be read as a volume (try '750 mL' or '1.75 L').")
    cands = volume_candidates(label_text)
    if not cands:
        return not_found(key, expected, note="No net contents (e.g. '750 mL') was found on the label.")

    def same(a: float, b: float) -> bool:   # US customary figures on labels are rounded ("750 mL / 25.4 OZ")
        return abs(a - b) <= max(th.volume_tolerance_ml, 0.005 * b)

    matching = [c for c in cands if same(c.ml, exp.ml)]
    if matching:
        shown = next((c for c in matching if c.unit in ("mL", "L", "cL")), matching[0])
        # A reading with the same digits and no decimal point ("15L" beside "1.5L") is the same
        # statement with the point lost by one pass, not a second statement.
        lost_point = {(m.unit, m.digits) for m in matching}
        others = [c for c in cands if abs(c.ml - exp.ml) > 0.02 * exp.ml
                  and not ((c.unit, c.digits) in lost_point and "." not in c.text)]
        if others:
            return FieldResult(key=key, label=_spec(key).label, expected=expected,
                               found="; ".join(dict.fromkeys(c.text for c in cands)), verdict=Verdict.NEAR_MATCH,
                               score=80, note=f"{exp.describe()} is on the label, but it also reads "
                                              f"{others[0].describe()}. This may be a reading error or a second "
                                              "statement. Please confirm.")
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=shown.text, verdict=Verdict.MATCH,
                           score=100, note=f"{exp.describe()} on both.")
    got = next((c for c in cands if c.unit in ("mL", "L", "cL")), cands[0])
    if got.unit == exp.unit and got.digits == exp.digits and "." not in got.text:
        # Same digits and unit, but the label read has no decimal point ("L5L" for "1.5 L"): the
        # point was probably lost by OCR. Never silently accept it; ask the agent to look.
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.NEAR_MATCH,
                           score=90, note=f"Reads like {expected} but the decimal point was not read clearly. Please confirm.")
    if round(exp.ml) in _STANDARD_ML and round(got.ml) not in _STANDARD_ML and _one_digit_apart(round(got.ml), round(exp.ml)):
        # "760 mL" is not a size anyone fills; "750 mL" is, and it is one digit away: most likely a misread.
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.NEAR_MATCH,
                           score=85, note=f"Reads {got.describe()}, which is not a standard size; the application's "
                                          f"{exp.describe()} is one digit away. Probably a reading error. Please confirm.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.MISMATCH,
                       score=0, note=f"Label says {got.describe()}, application says {exp.describe()}.")


# "Product of Scotland", "Imported from Mexico", "Distilled in Ireland": the country runs to the next
# delimiter, or stops before the words that usually follow it ("Product of France Imported by ...").
_ORIGIN_PREFIXES = ("product of", "produce of", "produced in", "made in", "imported from", "distilled in", "bottled in")
_ORIGIN_RE = re.compile(
    r"\b(" + "|".join(p.replace(" ", r"\s+") for p in _ORIGIN_PREFIXES) + r")\s+"
    r"([^\W\d_](?:[^\W\d_]|[ .'\-]){1,40}?)"
    r"(?=\s*(?:[,;:)\n]|(?<!st)(?<!ste)\.|$)|\s+(?:imported|bottled|distilled|produced|brewed|vinted|packed|by|for)\b|\s*\d)",
    re.IGNORECASE,
)
_NOT_A_COUNTRY = {"bond"}  # "Bottled in Bond" is a US designation, not an origin


def _country_key(name: str) -> str:
    key = re.sub(r"\bst(e?)\b", r"saint\1", normalize_loose(name))  # "St. Lucia" = "Saint Lucia"
    return key[4:] if key.startswith("the ") else key


def origin_statements(lines: list[str]) -> list[tuple[str, str]]:
    """Every origin statement on the label as (whole statement, country as printed)."""
    out = []
    for m in _ORIGIN_RE.finditer("\n".join(normalize_strict(l) for l in lines)):
        country = m.group(2).strip(" .'-")
        if country and _country_key(country) not in _NOT_A_COUNTRY:
            out.append((m.group(0).strip(" .'-"), country))
    return out


def compare_country(expected: str, lines: list[str], *, th: Thresholds = THRESHOLDS) -> FieldResult:
    """Country of origin (imports only). Blank on the application means domestic: skipped.

    The label's origin statements ("Product of X", "Imported from X") decide. Only an identical country
    name is a MATCH: country names are short and many differ by a few letters (Austria / Australia,
    Niger / Nigeria) or contain one another (Guinea / Equatorial Guinea), so anything else is either a
    NEAR MATCH for the agent to confirm or a MISMATCH.
    """
    key = "country_of_origin"
    label = _spec(key).label
    if not expected.strip():
        return skipped(key)
    country = expected.strip()
    for prefix in _ORIGIN_PREFIXES:
        if country.lower().startswith(prefix):
            country = country[len(prefix):].strip(" :,-")
            break
    want = _country_key(country)

    statements = origin_statements(lines)
    if statements:
        same = [s for s in statements if _country_key(s[1]) == want]
        others = [s for s in statements if _country_key(s[1]) != want]
        if same and not others:
            return FieldResult(key=key, label=label, expected=expected, found=same[0][0], verdict=Verdict.MATCH,
                               score=100, note="Country of origin matches the label's origin statement.")
        if same:
            return FieldResult(key=key, label=label, expected=expected, found="; ".join(s[0] for s in statements),
                               verdict=Verdict.NEAR_MATCH, score=100,
                               note="The label names more than one country. Please confirm the country of origin.")
        best = max(others, key=lambda s: fuzz.ratio(want, _country_key(s[1])))
        score = int(round(fuzz.ratio(want, _country_key(best[1]))))
        if score >= th.near_match:
            return FieldResult(key=key, label=label, expected=expected, found=best[0], verdict=Verdict.NEAR_MATCH,
                               score=score, note=f"Label says '{best[1]}', application says '{country}'. These are close "
                                                 "but not the same. Please confirm (a misprint, a reading error, or a "
                                                 "different country).")
        return FieldResult(key=key, label=label, expected=expected, found=best[0], verdict=Verdict.MISMATCH,
                           score=score, note=f"Label says '{best[1]}', application says '{country}'.")

    # No origin statement: accept the country name printed on a line of its own; anything weaker is
    # shown to the agent rather than accepted.
    for line in lines:
        if _country_key(line) == want:
            return FieldResult(key=key, label=label, expected=expected, found=normalize_strict(line), verdict=Verdict.MATCH,
                               score=100, note="Country name printed on the label (no 'Product of' statement).")
    loc = locate_text(country, lines, floor=th.near_match, th=th)
    if loc is not None:
        context = normalize_strict(" ".join(lines[loc.line_start: loc.line_end + 1]))
        return FieldResult(key=key, label=label, expected=expected, found=context, verdict=Verdict.NEAR_MATCH,
                           score=loc.score, note=f"'{loc.text}' appears on the label, but not in a country of origin "
                                                 "statement ('Product of ...'). Please confirm.")
    return not_found(key, expected, note="No country of origin statement was found on the label.")
