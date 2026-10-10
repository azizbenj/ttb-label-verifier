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


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over the smaller box's area: 1.0 when one box sits inside the other."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return ix * iy / smaller if smaller > 0 else 0.0


_OCR_CONFUSIONS = (("rn", "m"), ("cl", "d"), ("vv", "w"), ("0", "o"), ("1", "l"), ("|", "l"), ("i", "l"),
                   ("5", "s"), ("8", "b"))


def _ocr_fold(s: str) -> str:
    """Collapse the letter shapes OCR confuses, so 'Bam' and 'Barn' fold to the same text."""
    s = _despaced(normalize_loose(s))
    for a, b in _OCR_CONFUSIONS:
        s = s.replace(a, b)
    return s


def _span_conf(line_words: dict[int, list[tuple[str, float]]], first: int, last: int, text: str) -> float | None:
    """The lowest OCR confidence among the words that spell ``text`` on lines first..last."""
    toks = set(normalize_loose(text).split())
    confs = [c for i in range(first, last + 1) for t, c in line_words.get(i, []) if normalize_loose(t) in toks]
    return min(confs) if confs else None


def conflicting_reading(expected: str, lines: list[str], loc: Located,
                        line_boxes: dict[int, tuple[float, float, float, float]],
                        line_words: dict[int, list[tuple[str, float]]] | None = None,
                        th: Thresholds = THRESHOLDS) -> str | None:
    """Another reading of the same place on the label that says something else.

    The label is read several times (two page-segmentation passes, and turned or contrast views when
    something is missing), and the matcher takes the reading that agrees best with the application.
    When another reading of the same region disagrees, the agreeing one may itself be the misread
    ("BARK BREW" printed, one pass reading "BARN BREW"), so the match must not be silent. A
    disagreeing reading that OCR was less sure of than the agreeing one ("CQ." at 65 beside "CO." at
    77 in a display face) is noise; one it was at least as sure of counts."""
    span = [line_boxes[i] for i in range(loc.line_start, loc.line_end + 1) if i in line_boxes]
    if not span:
        return None
    left, top = min(b[0] for b in span), min(b[1] for b in span)
    box = (left, top, max(b[0] + b[2] for b in span) - left, max(b[1] + b[3] for b in span) - top)
    want = _ocr_fold(expected)
    for i, other in line_boxes.items():
        if loc.line_start <= i <= loc.line_end or _overlap(box, other) < 0.6:
            continue
        alt = locate_text(expected, [lines[i]], max_window=1, floor=th.conflict_floor, th=th)
        if alt is None:
            continue
        got = _ocr_fold(alt.text)
        # Not evidence of a different spelling: the usual letter confusions ("Bam" for "Barn") and a
        # reading cut short ("IRISH WHISKE"). A different letter ("BARK" for "BARN") is.
        if not got or got == want or got in want:
            continue
        if line_words:
            mine = _span_conf(line_words, loc.line_start, loc.line_end, loc.text)
            theirs = _span_conf(line_words, i, i, alt.text)
            if mine is not None and theirs is not None and theirs < mine:
                continue
        return alt.text
    return None


def locate_and_compare(key: str, expected: str, lines: list[str], *, fallback_found: str | None = None,
                       preferred_line: int | None = None, line_heights: dict[int, float] | None = None,
                       line_boxes: dict[int, tuple[float, float, float, float]] | None = None,
                       line_words: dict[int, list[tuple[str, float]]] | None = None,
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
    if result.verdict == Verdict.MATCH and line_boxes:
        alt = conflicting_reading(expected, lines, loc, line_boxes, line_words, th)
        if alt is not None:
            return result.model_copy(update={
                "verdict": Verdict.NEAR_MATCH, "score": 90,
                "note": f"Read as '{loc.text}', but another reading of the same place says '{alt}'. "
                        "One of them is a reading error. Please confirm."})
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


# --- second reads of a figure line (app/readers/numbers.py) ----------------------------------------
# Fragments of the notes below that mean "this figure rests on a doubtful reading"; the pipeline asks for
# a second read of the figure's own crop when it sees one of them.
NOTE_ALSO_READS = "but it also reads"
NOTE_LOST_POINT = "the decimal point was not read clearly"
NOTE_PROBABLE_MISREAD = "Probably a reading error"
NOTE_PROOF_DISAGREES = "does not agree with its percentage"
NOTE_APPLICATION_UNREADABLE = "The application value could not be read"
NOTE_CONFIRMED = "A closer read of that line"
NOTE_SECOND_READ_SAME = "A second read of that line says the same."
NOTE_READ_ON_CROP = "read on a closer look at the line"


def wants_second_read(f: FieldResult) -> str | None:
    """The figure kind ("alcohol" or "volume") a second read of its own crop could settle for this field,
    or None. Missing or different figures, and NEAR MATCHes that rest on a doubtful reading (readings
    disagree, a decimal point was lost, a probable misread, proof and percentage disagree) qualify; a
    percentage that is simply not marked as alcohol does not, since the crop would not mark it either."""
    if f.key not in ("alcohol_content", "net_contents") or NOTE_APPLICATION_UNREADABLE in f.note:
        return None
    kind = "alcohol" if f.key == "alcohol_content" else "volume"
    if f.verdict in (Verdict.NOT_FOUND, Verdict.MISMATCH):
        return kind
    if f.verdict == Verdict.NEAR_MATCH and any(n in f.note for n in
                                               (NOTE_ALSO_READS, NOTE_LOST_POINT, NOTE_PROBABLE_MISREAD, NOTE_PROOF_DISAGREES)):
        return kind
    return None


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _supersede(cands: list, first: list, closer: list, *, agrees, harmless, figure, th: Thresholds) -> list[tuple[str, str]]:
    """Set aside the page's readings of one line that a closer read of the same line corrected.

    Every closer read must agree with the application (a closer read that disagrees, other than the
    same figure with its decimal point lost, keeps every reading on the table), and a page reading is
    set aside only when its digits are within ``number_reread_max_edits`` of the closer read's: a
    dropped point ("15" for "1.5"), one digit ("790" for "750", "077" for "577"). Returns the pairs
    (first reading, closer reading) that were set aside; ``cands`` is edited in place."""
    good = [c for c in closer if agrees(c)]
    if not good or any(not agrees(c) and not harmless(c) for c in closer):
        return []
    out = []
    for c in first:
        if agrees(c) or Levenshtein.distance(figure(c), figure(good[0])) > th.number_reread_max_edits:
            continue
        same = next((x for x in cands if x == c), None) or \
            next((x for x in cands if figure(x) == figure(c) and not agrees(x)), None)
        if same is not None:
            cands.remove(same)
        # A page reading that only the line on its own yields ("LSL" repaired to "15 L" when the whole
        # text has an ordinary volume elsewhere) is not among ``cands``; it was still corrected.
        out.append((c.text, good[0].text))
    return out


def _second_read_note(got, rereads: list[tuple[str, list[str]]], candidates, figure) -> str:
    """When every closer read of the line the shown reading came from says the same figure, say so:
    the agent then knows the reading was checked, not just taken from the page."""
    for first, seconds in rereads:
        if not any(figure(c) == figure(got) for c in candidates(first)):
            continue
        read = [c for s in seconds for c in candidates(s)]
        if read and all(figure(c) == figure(got) for c in read):
            return " " + NOTE_SECOND_READ_SAME
    return ""


def _from_second_read(got, rereads: list[tuple[str, list[str]]], candidates, figure) -> bool:
    """The shown reading came only from a closer read, not from the page pass."""
    in_page = any(figure(c) == figure(got) for first, _ in rereads for c in candidates(first))
    in_crop = any(figure(c) == figure(got) for _, seconds in rereads for s in seconds for c in candidates(s))
    return in_crop and not in_page


def _abv_figure(c) -> str:
    """The digits of an alcohol statement as printed: percentage, then proof when it stated one."""
    return _digits(f"{c.abv:g}") + ("" if c.proof is None or c.abv_from_proof else _digits(f"{c.proof:g}"))


def compare_alcohol(expected: str, label_text: str, *, statement: bool = False,
                    rereads: list[tuple[str, list[str]]] | None = None, th: Thresholds = THRESHOLDS) -> FieldResult:
    """Alcohol content compared as numbers, over every alcohol statement read on the label.

    ``statement``: ``label_text`` is already the alcohol statement (a vision model's field), so a bare
    "45%" in it counts. On OCR text a percentage only counts when "Alc./Vol.", "ABV" or a proof figure
    marks it as alcohol: "13.5% Petit Verdot" on a wine label whose real statement was not read must
    never match an application of 13.5%.
    ``rereads``: for each line read again from its own crop, the page's reading and the closer reads
    (see ``_supersede`` for what a closer read may correct).
    """
    key = "alcohol_content"
    exp = parse_alcohol(expected)
    if exp is None or exp.abv is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note=NOTE_APPLICATION_UNREADABLE + " as an alcohol content (try '45% Alc./Vol.' or '90 Proof').")
    cands = alcohol_candidates(label_text)
    rereads = rereads or []

    def agrees(c) -> bool:
        return abs(c.abv - exp.abv) <= th.abv_tolerance and not (
            c.proof is not None and not c.abv_from_proof and abs(c.proof - 2 * c.abv) > th.proof_tolerance)

    confirmed: list[tuple[str, str]] = []
    for first, seconds in rereads:
        confirmed += _supersede(cands, alcohol_candidates(first), [c for s in seconds for c in alcohol_candidates(s)],
                                agrees=agrees, harmless=lambda c: False, figure=_abv_figure, th=th)
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
    if confirmed:
        notes.append(f"{NOTE_CONFIRMED} confirms it: the page was first read as '{confirmed[0][0]}'.")
    elif _from_second_read(got, rereads, alcohol_candidates, _abv_figure):
        notes.append(f"The statement was {NOTE_READ_ON_CROP}.")
    notes.append(_second_read_note(got, rereads, alcohol_candidates, _abv_figure).strip())
    notes = [n for n in notes if n]
    if got.abv_from_proof:
        notes.append(f"Label states {got.proof:g} proof, which is {got.abv:g}% ABV.")
    inconsistent = got.proof is not None and not got.abv_from_proof and abs(got.proof - 2 * got.abv) > th.proof_tolerance
    if inconsistent:
        notes.append(f"The label's proof ({got.proof:g}) {NOTE_PROOF_DISAGREES} ({got.abv:g}%).")
    others = sorted({round(c.abv, 2) for c in cands if abs(c.abv - exp.abv) > th.abv_tolerance})
    if matching and others:
        return FieldResult(key=key, label=_spec(key).label, expected=expected,
                           found="; ".join(dict.fromkeys(c.text for c in cands)), verdict=Verdict.NEAR_MATCH, score=80,
                           note=f"{exp.abv:g}% ABV is on the label, {NOTE_ALSO_READS} "
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


_METRIC = ("mL", "L", "cL")
# Standard container sizes (27 CFR 5.203 spirits, 4.72 wine, common malt beverage packages), in mL.
_STANDARD_ML = {50, 100, 180, 187, 200, 250, 300, 331, 350, 355, 365, 375, 473, 475, 500, 568, 570, 600, 620, 650,
                700, 710, 720, 750, 900, 945, 946, 1000, 1500, 1750, 1800, 2000, 2250, 3000, 3750}


def _one_digit_apart(a: int, b: int) -> bool:
    sa, sb = str(a), str(b)
    return len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) == 1


def _vol_figure(c) -> str:
    return c.digits


def compare_volume(expected: str, label_text: str, *, rereads: list[tuple[str, list[str]]] | None = None,
                   th: Thresholds = THRESHOLDS) -> FieldResult:
    """Net contents compared in millilitres, over every volume statement read on the label.
    ``rereads``: for each line read again from its own crop, the page's reading and the closer reads
    (see ``_supersede`` for what a closer read may correct)."""
    key = "net_contents"
    exp = parse_net_contents(expected)
    if exp is None:
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=None, verdict=Verdict.NOT_FOUND,
                           note=NOTE_APPLICATION_UNREADABLE + " as a volume (try '750 mL' or '1.75 L').")
    cands = volume_candidates(label_text)
    rereads = rereads or []

    def same(c, e) -> bool:
        # A US customary figure beside a metric one is rounded ("750 mL / 25.4 FL OZ" is 751.2 mL): 0.5% is
        # allowed between the two systems. Within one system the figures must agree (753 mL is not 750 mL).
        if (c.unit in _METRIC) != (e.unit in _METRIC):
            return abs(c.ml - e.ml) <= max(th.volume_tolerance_ml, 0.005 * e.ml)
        return abs(c.ml - e.ml) <= th.volume_tolerance_ml

    def lost_point_of_expected(c) -> bool:
        # The expected figure with its decimal point lost ("15L" for "1.5 L"): the same statement, misread.
        return c.unit == exp.unit and c.digits == exp.digits and "." not in c.text

    confirmed: list[tuple[str, str]] = []
    for first, seconds in rereads:
        confirmed += _supersede(cands, volume_candidates(first), [c for s in seconds for c in volume_candidates(s)],
                                agrees=lambda c: same(c, exp), harmless=lost_point_of_expected, figure=_vol_figure, th=th)
    if not cands:
        return not_found(key, expected, note="No net contents (e.g. '750 mL') was found on the label.")

    matching = [c for c in cands if same(c, exp)]
    if matching:
        shown = next((c for c in matching if c.unit in _METRIC), matching[0])
        # A reading with the same digits and no decimal point ("15L" beside "1.5L") is the same
        # statement with the point lost by one pass, not a second statement. Any other reading that
        # disagrees (a "760 mL" beside the "750 mL") is shown to the agent.
        lost_point = {(m.unit, m.digits) for m in matching}
        others = [c for c in cands if not same(c, exp)
                  and not ((c.unit, c.digits) in lost_point and "." not in c.text)]
        if others:
            return FieldResult(key=key, label=_spec(key).label, expected=expected,
                               found="; ".join(dict.fromkeys(c.text for c in cands)), verdict=Verdict.NEAR_MATCH,
                               score=80, note=f"{exp.describe()} is on the label, {NOTE_ALSO_READS} "
                                              f"{others[0].describe()}. This may be a reading error or a second "
                                              "statement. Please confirm.")
        note = f"{exp.describe()} on both."
        if confirmed:
            note += f" {NOTE_CONFIRMED} confirms it: the page was first read as '{confirmed[0][0]}'."
        elif _from_second_read(shown, rereads, volume_candidates, _vol_figure):
            note += f" The statement was {NOTE_READ_ON_CROP}."
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=shown.text, verdict=Verdict.MATCH,
                           score=100, note=note)
    got = next((c for c in cands if c.unit in _METRIC), cands[0])
    checked = _second_read_note(got, rereads, volume_candidates, _vol_figure)
    if lost_point_of_expected(got):
        # Same digits and unit, but the label read has no decimal point ("L5L" for "1.5 L"): the
        # point was probably lost by OCR. Never silently accept it; ask the agent to look.
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.NEAR_MATCH,
                           score=90, note=f"Reads like {expected} but {NOTE_LOST_POINT}.{checked} Please confirm.")
    if round(exp.ml) in _STANDARD_ML and round(got.ml) not in _STANDARD_ML and _one_digit_apart(round(got.ml), round(exp.ml)):
        # "760 mL" is not a size anyone fills; "750 mL" is, and it is one digit away: most likely a misread.
        return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.NEAR_MATCH,
                           score=85, note=f"Reads {got.describe()}, which is not a standard size; the application's "
                                          f"{exp.describe()} is one digit away. {NOTE_PROBABLE_MISREAD}.{checked} Please confirm.")
    return FieldResult(key=key, label=_spec(key).label, expected=expected, found=got.text, verdict=Verdict.MISMATCH,
                       score=0, note=f"Label says {got.describe()}, application says {exp.describe()}.{checked}")


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
