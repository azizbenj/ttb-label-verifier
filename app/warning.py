"""Government warning statement checks (27 CFR 16.21 / 16.22).

Four things are checked, each reported separately so an agent can see exactly what
is wrong:
  1. present      - a warning statement exists on the label
  2. wording      - the words match the mandated text exactly (punctuation ignored,
                    because OCR drops commas and periods unreliably), each sentence
                    once and in order
  3. heading caps - "GOVERNMENT WARNING:" is in capital letters, as printed
  4. heading bold - the heading is printed bolder than the body text. This is a
                    heuristic (stroke-width comparison on the image); see README.

Locating the statement uses the word boxes when the reader provides them: two OCR lines
at the same place are two readings of one printed line (never both taken), and words
whose boxes sit beyond the statement's own column on several lines belong to a
neighbouring column (set aside and quoted, not judged as wording).
"""

from __future__ import annotations

import difflib
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .config import MANDATED_WARNING, THRESHOLDS, Thresholds
from .models import DiffItem, Status, WarningResult
from .normalize import normalize_loose, normalize_strict
from .readers.base import OCRWord, upright_box

_MANDATED_LOOSE = normalize_loose(MANDATED_WARNING)
_MANDATED_WORDS = _MANDATED_LOOSE.split()
_MANDATED_VOCAB = set(_MANDATED_WORDS)
_DESCENDER_CHARS = set("gjpqy,;()[]{}")
_HEIGHT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt0123456789")

# The two sentences of the statement, each with the phrase that identifies it.
_SENTENCES: tuple[tuple[int, str, str], ...] = (
    (1, normalize_loose(MANDATED_WARNING.split("(1)")[1].split("(2)")[0]), "according to the surgeon general"),
    (2, normalize_loose(MANDATED_WARNING.split("(2)")[1]), "consumption of alcoholic beverages"),
)


# --- locating the statement -------------------------------------------------------------------
@dataclass(frozen=True)
class WarningSpan:
    line_ids: tuple[int, ...]         # lines that make up the statement, in reading order
    heading_line: int | None          # line holding "GOVERNMENT WARNING", if recognised
    texts: tuple[str, ...] = ()       # those lines as read, minus text that sits beside the statement
    beside: str = ""                  # words set aside because their boxes put them in a neighbouring column
    trimmed: str = ""                 # words left out on wording alone (no boxes to place them): asks for a look
    heading_lines: tuple[int, ...] = ()   # the heading's line(s): two when it is split "GOVERNMENT" / "WARNING:"
    kept_words: tuple[OCRWord, ...] = ()  # the word boxes that make up ``texts`` (empty without boxes)
    conflicts: tuple[str, ...] = ()       # a discarded reading of the same line disagreed on a word
    second_copies: tuple[tuple[int, str], ...] = ()   # (sentence number, text) printed again outside the statement

    @property
    def start(self) -> int:
        return self.line_ids[0]

    @property
    def end(self) -> int:
        return self.line_ids[-1]


def _contains_score(query: str, line: str) -> float:
    """How well ``query`` appears inside ``line``. Short OCR fragments ("a", "—") must not score high."""
    if len(line) < 0.8 * len(query):
        return fuzz.ratio(query, line)
    return fuzz.partial_ratio(query, line)


_MAX_SKIPS = 4        # rows in a row that may be passed over (another column, "For Sale Only In Ohio")
_MAX_LINES_AHEAD = 80  # safety bound on OCR lines walked; a frame holds every reading of every row


def _statement_share(line: str) -> float:
    """Share of a line's real words (3+ letters) that occur in the mandated statement."""
    toks = [t for t in line.split() if sum(ch.isalpha() for ch in t) >= 3]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _MANDATED_VOCAB) / len(toks)


def _loose_tokens(word: OCRWord) -> list[str]:
    return normalize_loose(word.text).split()


def _is_real(word: OCRWord) -> bool:
    """A word, as opposed to a stray mark ("~", "|", "4") that OCR read beside the text."""
    return sum(ch.isalpha() for ch in word.text) >= 2


# -- word-box geometry, in the coordinates of the line's frame (view) --
@dataclass
class _Line:
    idx: int
    words: list[OCRWord]      # reading order
    top: float                # median top and bottom of the words: a stray tall box must not stretch the line
    bottom: float
    left: int
    right: int
    height: float


def _line_geometry(idx: int, words: list[OCRWord]) -> _Line:
    return _Line(idx, words, float(statistics.median(w.top for w in words)),
                 float(statistics.median(w.bottom for w in words)), min(w.left for w in words),
                 max(w.right for w in words), float(statistics.median(w.height for w in words)))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _same_place(a: _Line, b: _Line, th: Thresholds) -> bool:
    """Two OCR lines at the same place are two readings of one printed line (the block pass and the
    sparse pass both read it, or one read across two columns and the other read each column)."""
    ha, hb = a.bottom - a.top, b.bottom - b.top
    wa, wb = a.right - a.left, b.right - b.left
    if min(ha, hb) <= 0 or min(wa, wb) <= 0:
        return False
    return (_overlap(a.top, a.bottom, b.top, b.bottom) >= th.same_place_overlap * min(ha, hb)
            and _overlap(a.left, a.right, b.left, b.right) >= th.same_place_overlap * min(wa, wb))


def _assemble(order: list[int], loose: list[str], start_pos: int,
              same_place: Callable[[int, int], bool] | None = None,
              conf: Callable[[int], float] | None = None) -> tuple[list[int], float]:
    """Grow the statement from ``order[start_pos]``, taking each following line that is made mostly of
    the statement's own words and brings the text closer to it, and passing over rows that do not
    ("For Sale Only In Ohio" printed between two lines of a sideways warning). A line at the same place
    as one already taken is another reading of it: the better reading is kept, never both (when the two
    read equally well, the one OCR was surer of)."""
    ids = [order[start_pos]]

    def score(ids_: list[int]) -> float:
        return fuzz.ratio(_MANDATED_LOOSE, " ".join(loose[i] for i in ids_))

    best = score(ids)
    skips = 0
    for pos in range(start_pos + 1, min(len(order), start_pos + _MAX_LINES_AHEAD)):
        j = order[pos]
        if not loose[j]:
            continue
        trial: list[int] | None = None
        dup: list[int] = []
        r = -1.0
        if _statement_share(loose[j]) >= 0.5:
            dup = [k for k in ids if same_place is not None and same_place(k, j)]
            if dup:
                at = ids.index(dup[0])
                trial = [k for k in ids if k not in dup]
                trial.insert(at, j)
            else:
                trial = ids + [j]
            r = score(trial)
        surer = bool(dup) and conf is not None and r >= best - 1e-9 and conf(j) > max(conf(k) for k in dup)
        if trial is not None and (r > best + 1e-9 or surer):
            ids, best, skips = trial, r, 0
        elif not dup:          # another reading of a row already taken is not a row passed over
            skips += 1
            if skips > _MAX_SKIPS:
                break
    return ids, best


def _trim_beside(texts: list[str]) -> tuple[list[str], str]:
    """Drop words at the start or end of a line that belong to a neighbouring column when there are no
    word boxes to place them (a keg collar's "ATTENTION-READ BEFORE TAPPING" printed beside the warning).
    Returns the trimmed lines and the real words that were left out (short OCR noise such as "7" or "—c"
    is dropped silently). Wording alone cannot tell a neighbouring column from words added to the
    statement, so the caller asks for a look whenever anything was left out this way."""
    texts = list(texts)
    dropped: list[str] = []

    def score(ts):
        return fuzz.ratio(_MANDATED_LOOSE, normalize_loose(" ".join(ts)))

    for i, line in enumerate(texts):
        toks = line.split()
        if len(toks) < 2:
            continue
        best, best_cut = score(texts), (0, len(toks))
        for a in range(0, min(len(toks), 8)):
            for b in range(len(toks), max(a, len(toks) - 12), -1):
                if (a, b) == (0, len(toks)) or b <= a:
                    continue
                trial = texts[:i] + [" ".join(toks[a:b])] + texts[i + 1:]
                r = score(trial)
                if r > best + 0.5:
                    best, best_cut = r, (a, b)
        a, b = best_cut
        if (a, b) != (0, len(toks)):
            cut = toks[:a] + toks[b:]
            dropped += [t for t in cut if sum(ch.isalpha() for ch in t) >= 3]
            texts[i] = " ".join(toks[a:b])
    return texts, " ".join(dropped)


_LOOKALIKE_LETTERS = str.maketrans({"0": "o", "1": "i", "|": "i", "l": "i"})  # neither heading word has an "l"


def _near_word(token: str, target: str) -> bool:
    return Levenshtein.distance(token.translate(_LOOKALIKE_LETTERS), target) <= 2


def _heading_starts(order: list[int], loose: list[str], th: Thresholds) -> dict[int, tuple[float, tuple[int, ...]]]:
    """Positions in ``order`` where the statement may start: a line that reads "GOVERNMENT WARNING", or a
    line ending in "GOVERNMENT" with the next line starting "WARNING" (the heading split over two lines).
    Maps position -> (score, the heading's line ids)."""
    heads: dict[int, tuple[float, tuple[int, ...]]] = {}
    for pos, i in enumerate(order):
        sc = _contains_score("government warning", loose[i])
        if sc >= th.warning_locate:
            heads[pos] = (sc, (i,))
    for pos in range(len(order) - 1):
        if pos in heads:
            continue
        a, b = order[pos], order[pos + 1]
        ta, tb = loose[a].split(), loose[b].split()
        if ta and tb and _near_word(ta[-1], "government") and _near_word(tb[0], "warning"):
            heads[pos] = (fuzz.ratio("government warning", f"{ta[-1]} {tb[0]}"), (a, b))
    return heads


def _matched_words(ids: list[int], geo: dict[int, _Line]) -> list[OCRWord]:
    """The word boxes of the span whose tokens align with the mandated text, in order."""
    tokens: list[str] = []
    owners: list[OCRWord] = []
    for i in ids:
        for w in geo[i].words:
            for t in _loose_tokens(w):
                tokens.append(t)
                owners.append(w)
    sm = difflib.SequenceMatcher(a=_MANDATED_WORDS, b=tokens, autojunk=False)
    matched: list[OCRWord] = []
    seen: set[int] = set()
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for j in range(j1, j2):
            if id(owners[j]) not in seen:
                seen.add(id(owners[j]))
                matched.append(owners[j])
    return matched


@dataclass
class _Columns:
    """Each line of the statement's rows cut to the statement's own column."""
    kept: dict[int, list[OCRWord]]
    beside: dict[int, list[OCRWord]]


def _split_columns(order: list[int], ids: list[int], geo: dict[int, _Line], th: Thresholds) -> _Columns | None:
    """Use the word boxes to separate the statement from a neighbouring column.

    The words of the provisional statement that align with the mandated text define the column's left and
    right extent across all its lines. A word whose box lies entirely beyond that extent is outside the
    column; when such words appear beyond the same edge on at least ``column_min_lines`` of the
    statement's rows, that edge borders another column and all words beyond it are set aside. A single
    line with words sticking out is left alone (they may be words added to the statement). Stray marks
    with no letters ("~", "|", "4") beyond the extent are dropped silently. The gap between the columns
    plays no part: on a real keg collar it is a normal word gap.
    """
    matched = _matched_words(ids, geo)
    if len(matched) < 8:
        return None
    x_left, x_right = min(w.left for w in matched), max(w.right for w in matched)
    tol = th.column_tolerance * float(statistics.median(w.height for w in matched))
    y0, y1 = min(geo[i].top for i in ids), max(geo[i].bottom for i in ids)
    sides: dict[int, list[tuple[OCRWord, str]]] = {}
    lines_left: set[int] = set()
    lines_right: set[int] = set()
    for i in order:
        g = geo.get(i)
        if g is None or g.bottom < y0 or g.top > y1:
            continue
        row = []
        for w in g.words:
            if w.right <= x_left + tol:
                side = "left"
            elif w.left >= x_right - tol:
                side = "right"
            else:
                side = "in"
            row.append((w, side))
            if side == "left" and _is_real(w):
                lines_left.add(i)
            elif side == "right" and _is_real(w):
                lines_right.add(i)
        sides[i] = row
    column = {"left": len(lines_left) >= th.column_min_lines, "right": len(lines_right) >= th.column_min_lines}
    out = _Columns({}, {})
    for i, row in sides.items():
        kept, beside = [], []
        for w, side in row:
            if side == "in":
                kept.append(w)
            elif column[side]:
                beside.append(w)
            elif any(ch.isalpha() for ch in w.text):
                kept.append(w)          # one line sticking out: left in place, judged as wording
            # else: a stray mark beyond the column ("~", "|", "4"), dropped
        out.kept[i] = kept
        out.beside[i] = [w for w in beside if _is_real(w)]
    return out


def _conflicts(ids: list[int], order: list[int], geo: dict[int, _Line], matched: list[OCRWord],
               th: Thresholds) -> list[str]:
    """Words on which a discarded reading of a statement line disagrees with a word of the reading kept
    that aligns with the mandated text, when OCR was at least as sure of the discarded word. A misread
    that happens to agree with the mandated text must not hide a misprint that the other reading saw."""
    out: list[str] = []
    taken = set(ids)
    aligned = {id(w) for w in matched}
    for j in order:
        if j in taken or j not in geo:
            continue
        for k in ids:
            if k not in geo or not _same_place(geo[k], geo[j], th):
                continue
            for d in geo[j].words:
                if sum(ch.isalpha() for ch in d.text) < 3:
                    continue
                dt = set(_loose_tokens(d))
                if not dt or dt <= _MANDATED_VOCAB:
                    continue
                for w in geo[k].words:
                    if id(w) not in aligned or _overlap(d.left, d.right, w.left, w.right) < 0.5 * min(d.width, w.width):
                        continue
                    if dt <= set(_loose_tokens(w)) or d.conf < w.conf:
                        continue
                    out.append(f"'{d.text}' where the statement reads '{w.text}'")
    return out


def _second_copies(ids: list[int], order: list[int], loose: list[str], lines: list[str], geo: dict[int, _Line],
                   th: Thresholds) -> list[tuple[int, str]]:
    """A sentence of the statement printed again outside it (sentence (1) twice, say). Lines at the same
    place as a statement line are other readings of it and do not count; a complete second statement
    is not a wording problem either."""
    taken = set(ids)
    rest = [j for j in order if j not in taken and loose[j]
            and not any(k in geo and j in geo and _same_place(geo[k], geo[j], th) for k in ids)
            and _statement_share(loose[j]) >= 0.5 and len(loose[j].split()) >= 3]
    if not rest:
        return []
    text = " ".join(loose[j] for j in rest)
    if fuzz.ratio(_MANDATED_LOOSE, text) >= th.duplicate_sentence:
        return []
    quoted = " ".join(lines[j] for j in rest)
    found = []
    for number, sentence, _anchor in _SENTENCES:
        if len(text) >= 0.5 * len(sentence) and fuzz.partial_ratio(sentence, text) >= th.duplicate_sentence:
            found.append((number, quoted))
    return found


@dataclass
class _Candidate:
    ratio: float
    span: WarningSpan


def _assemble_span(order: list[int], loose: list[str], pos: int, heads: dict[int, tuple[float, tuple[int, ...]]],
                   lines: list[str], geo: dict[int, _Line], th: Thresholds) -> _Candidate:
    same = (lambda a, b: a in geo and b in geo and _same_place(geo[a], geo[b], th)) if geo else None
    kept_words: dict[int, list[OCRWord]] = {i: g.words for i, g in geo.items()}
    beside: dict[int, list[OCRWord]] = {}

    def conf(i: int) -> float:
        ws = kept_words.get(i) or []
        return sum(w.conf for w in ws) / len(ws) if ws else 0.0

    ids, ratio = _assemble(order, loose, pos, same, conf if geo else None)
    loose_now = loose
    if geo and all(i in geo for i in ids):
        cols = _split_columns(order, ids, geo, th)
        if cols is not None and cols.kept.get(ids[0]):
            loose_now = list(loose)
            for i, ws in cols.kept.items():
                loose_now[i] = " ".join(t for w in ws for t in _loose_tokens(w))
            kept_words.update(cols.kept)
            beside = cols.beside
            ids, ratio = _assemble(order, loose_now, pos, same, conf)
    # The heading is whatever line now opens the statement (a cleaner reading may have replaced the start).
    by_first = {tup[0]: tup for _sc, tup in heads.values()}
    heading_lines = tuple(i for i in by_first.get(ids[0], ()) if i in ids)
    if geo:
        texts = tuple(" ".join(w.text for w in kept_words.get(i, [])) or lines[i] for i in ids)
        span_words = [w for i in ids for w in kept_words.get(i, [])]
        # Words set aside from every reading of the statement's rows, quoted once each.
        seen: set[tuple[str, int, int]] = set()
        beside_words: list[OCRWord] = []
        for k in ids:
            for j in order:
                if j == k or (j in geo and k in geo and _same_place(geo[k], geo[j], th)):
                    for w in beside.get(j, []):
                        step = max(8, int(geo[j].height)) if j in geo else 8
                        key = (normalize_loose(w.text), w.left // step, w.top // step)
                        if key not in seen:
                            seen.add(key)
                            beside_words.append(w)
        beside_words.sort(key=lambda w: (geo[w.line_index].top if w.line_index in geo else 0, w.left))
        matched = _matched_words(ids, geo)
        conflicts = tuple(_conflicts(ids, order, geo, matched, th))
    else:
        texts = tuple(lines[i] for i in ids)
        span_words, beside_words, conflicts = [], [], ()
    trimmed_texts, trimmed = _trim_beside(list(texts))
    span = WarningSpan(
        line_ids=tuple(ids), heading_line=heading_lines[0] if heading_lines else None, texts=tuple(trimmed_texts),
        beside=" ".join(w.text for w in beside_words), trimmed=trimmed, heading_lines=heading_lines,
        kept_words=tuple(span_words), conflicts=conflicts,
        second_copies=tuple(_second_copies(ids, order, loose_now, lines, geo, th)))
    return _Candidate(ratio, span)


def locate_warning(lines: list[str], *, line_views: list[int] | None = None, line_tops: list[float] | None = None,
                   line_words: dict[int, list[OCRWord]] | None = None, th: Thresholds = THRESHOLDS) -> WarningSpan | None:
    """Find the statement. Each frame the reader produced (upright, turned each way) is tried on its own,
    its lines ordered top to bottom, and the reading that comes closest to the mandated text wins.
    ``line_views`` groups lines that share a coordinate frame; ``line_words`` gives each line's word boxes
    in that frame, which lets two readings of one printed line and a neighbouring column be told apart."""
    if not lines:
        return None
    geo = {i: _line_geometry(i, ws) for i, ws in (line_words or {}).items() if ws and 0 <= i < len(lines)}
    loose = [" ".join(t for w in geo[i].words for t in _loose_tokens(w)) if i in geo else normalize_loose(l)
             for i, l in enumerate(lines)]
    views = line_views or [0] * len(lines)
    tops = line_tops or [float(i) for i in range(len(lines))]
    tops = [geo[i].top if i in geo else tops[i] for i in range(len(lines))]
    best: _Candidate | None = None
    for view in sorted(set(views)):
        order = sorted((i for i in range(len(lines)) if views[i] == view and loose[i]), key=lambda i: (tops[i], i))
        heads = _heading_starts(order, loose, th)
        starts = [pos for pos, _ in sorted(heads.items(), key=lambda h: -h[1][0])[:3]]
        if not starts:   # heading unreadable or missing: start from the body of the statement
            starts = [pos for pos, i in enumerate(order)
                      if _contains_score("according to the surgeon general", loose[i]) >= th.warning_locate][:2]
        for pos in starts:
            cand = _assemble_span(order, loose, pos, heads, lines, geo, th)
            if best is None or cand.ratio > best.ratio:
                best = cand
    return best.span if best else None


# --- wording ----------------------------------------------------------------------------------
_MANDATED_RAW_WORDS = MANDATED_WARNING.split()   # same count as _MANDATED_WORDS (no punctuation-only tokens)


def join_hyphenated(texts: list[str]) -> list[str]:
    """Rejoin a word of the statement that the label hyphenates over a line break ("SUR-" / "GEON").
    Only a join that spells a word of the mandated text is made; anything else stays as read."""
    out = list(texts)
    for i in range(len(out) - 1):
        a, b = out[i].split(), out[i + 1].split()
        if not a or not b or not a[-1].endswith("-") or len(a[-1]) < 2:
            continue
        head, tail = a[-1][:-1], b[0].lstrip("[({|\"'")
        h, t = normalize_loose(head).split(), normalize_loose(tail).split()
        if len(h) != 1 or len(t) != 1 or h[0] + t[0] not in _MANDATED_VOCAB:
            continue
        out[i] = " ".join(a[:-1] + [head + tail])
        out[i + 1] = " ".join(b[1:])
    return out


def word_diff(found_text: str, context: int = 5) -> list[DiffItem]:
    """Word-level differences from the mandated text, each with a few words of context on both sides."""
    raw_found = [t for t in found_text.split() if normalize_loose(t)]
    got = normalize_loose(found_text).split()
    show_found = raw_found if len(raw_found) == len(got) else got
    show_exp = _MANDATED_RAW_WORDS if len(_MANDATED_RAW_WORDS) == len(_MANDATED_WORDS) else _MANDATED_WORDS
    sm = difflib.SequenceMatcher(a=_MANDATED_WORDS, b=got, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append(DiffItem(
            expected=" ".join(show_exp[i1:i2]), found=" ".join(show_found[j1:j2]),
            expected_before=" ".join(show_exp[max(0, i1 - context):i1]), expected_after=" ".join(show_exp[i2:i2 + context]),
            found_before=" ".join(show_found[max(0, j1 - context):j1]), found_after=" ".join(show_found[j2:j2 + context])))
    return out


def sentence_structure(found_text: str) -> list[str]:
    """Problems with the statement's two sentences: one printed twice, or (2) printed before (1).
    A missing sentence is already a wording failure (the text is far from the mandated one)."""
    got = normalize_loose(found_text)
    problems = []
    positions = {}
    for number, _sentence, anchor in _SENTENCES:
        n = got.count(anchor)
        if n > 1:
            problems.append(f"sentence ({number}) appears {n} times")
        if n:
            positions[number] = got.index(anchor)
    if len(positions) == 2 and positions[2] < positions[1]:
        problems.append("sentence (2) comes before sentence (1)")
    return problems


def check_wording(found_text: str, *, th: Thresholds = THRESHOLDS) -> tuple[Status, int, str, list[DiffItem]]:
    got = normalize_loose(found_text)
    if got == _MANDATED_LOOSE:
        return Status.PASS, 100, "Wording matches the required statement word for word.", []
    score = int(round(fuzz.ratio(_MANDATED_LOOSE, got)))
    diff = word_diff(found_text)
    structure = sentence_structure(found_text)
    n = len(diff)
    if structure:
        return (Status.FAIL, score,
                f"The statement must carry each sentence once, in order: {'; '.join(structure)}.", diff)
    if score >= th.warning_near:
        return (Status.REVIEW, score,
                f"{n} difference{'s' if n != 1 else ''} from the required wording. "
                "This may be a misprint on the label or a reading error. Please check the label.", diff)
    return Status.FAIL, score, "The wording does not match the required statement.", diff


# --- heading capitalization -------------------------------------------------------------------
_HEADING_RE = re.compile(r"(government)\s+(warning)(\s*:)?", re.IGNORECASE)


def _misread_heading(text: str) -> tuple[str, str, bool] | None:
    """The two words that read most like 'GOVERNMENT WARNING' when OCR garbled a letter or two ("WARNlNG",
    "G0VERNMENT"), or cut at the edge of the image ("ERNMENT WARMING"). Each word must be a near spelling
    or a long fragment of its target: "GOVT WARNING" is an abbreviation printed on the label, not a
    misread, and must fail."""
    tokens = text.split()
    best, best_score = None, 0.0
    for i in range(len(tokens) - 1):
        gov, warn = tokens[i], tokens[i + 1].rstrip(":;.")
        g, w = gov.lower().translate(_LOOKALIKE_LETTERS), warn.lower().translate(_LOOKALIKE_LETTERS)
        near_gov = Levenshtein.distance(g, "government") <= 2 or (len(g) >= 6 and g in "government")
        if not near_gov or Levenshtein.distance(w, "warning") > 2:
            continue
        score = fuzz.ratio("government warning", f"{g} {w}")
        if score > best_score:
            colon = tokens[i + 1].endswith(":") or (i + 2 < len(tokens) and tokens[i + 2].startswith(":"))
            best, best_score = (gov, warn, colon), score
    return best if best_score >= 80 else None


def check_heading_caps(heading_line_text: str | None) -> tuple[Status, str]:
    if not heading_line_text:
        return Status.FAIL, "The words 'GOVERNMENT WARNING' were not found at the start of the statement."
    text = normalize_strict(heading_line_text)
    m = _HEADING_RE.search(text)
    if m:
        gov, warn, colon = m.group(1), m.group(2), bool(m.group(3))
    else:
        misread = _misread_heading(text)
        if misread is None:
            m2 = re.match(r"\s*(\S{2,14})\s+(WARNING)\s*:", text)
            if m2 and not m2.group(1).isalpha():
                # "\Noee WARNING:": the first word is there but garbled, "WARNING" is in capitals
                return Status.REVIEW, ("The first word of the heading was not read clearly; 'WARNING' is in capitals. "
                                       "Please check the heading by eye.")
            if m2:   # a clean word that is not GOVERNMENT: "HEALTH WARNING:", "GOVT WARNING:"
                return Status.FAIL, (f"The heading reads '{m2.group(1)} WARNING:'. It must read 'GOVERNMENT WARNING:'.")
            return Status.FAIL, "The words 'GOVERNMENT WARNING' were not found at the start of the statement."
        gov, warn, colon = misread
        lower = {ch for ch in gov + warn if ch.islower()}
        if lower <= {"l", "i", "o"}:  # capitals with a letter or two misread by OCR
            return Status.REVIEW, (f"The heading reads '{gov} {warn}': it looks like capitals, but a letter was not read "
                                   "clearly. Please check it by eye.")
    if not (gov.isupper() and warn.isupper()):
        return Status.FAIL, f"The heading is printed as '{gov} {warn}'. It must be all capitals: 'GOVERNMENT WARNING:'."
    if not colon:
        return Status.REVIEW, "Heading is in capitals, but the colon after 'WARNING' was not detected."
    return Status.PASS, "Heading is in capitals."


# --- heading boldness (heuristic) --------------------------------------------------------------
def _ink_area_and_boundary(ink: np.ndarray, word: OCRWord) -> tuple[int, int] | None:
    """Ink pixel count and boundary pixel count inside a word box (boundary = ink with a background 4-neighbour).

    The ink mask is global, so on a light-on-dark panel of an otherwise light label it marks the panel,
    not the letters. The ring of pixels just outside the word box is background either way: when it is
    mostly "ink", the polarity is flipped for this word.
    """
    h, w = ink.shape
    x0, y0 = max(0, word.left - 1), max(0, word.top - 1)
    x1, y1 = min(w, word.right + 1), min(h, word.bottom + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    rx0, ry0, rx1, ry1 = max(0, x0 - 3), max(0, y0 - 3), min(w, x1 + 3), min(h, y1 + 3)
    ring = ink[ry0:ry1, rx0:rx1].copy()
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    ring_px = (ry1 - ry0) * (rx1 - rx0) - (y1 - y0) * (x1 - x0)
    crop = ink[y0:y1, x0:x1]
    if ring_px and ring.sum() > 0.5 * ring_px:
        crop = ~crop
    crop = np.pad(crop, 1)
    inner = crop[1:-1, 1:-1]
    bg_neighbour = (~crop[:-2, 1:-1]) | (~crop[2:, 1:-1]) | (~crop[1:-1, :-2]) | (~crop[1:-1, 2:])
    area = int(inner.sum())
    boundary = int((inner & bg_neighbour).sum())
    if area < 20 or boundary == 0:
        return None
    return area, boundary


def stroke_width(ink: np.ndarray, words: list[OCRWord]) -> float | None:
    """Mean stroke width over a set of words: 2 x ink area / ink perimeter.

    For a stroke of width w and length L the area is w*L and the perimeter about 2L, so the
    estimate is orientation independent (diagonals and curves do not inflate it, unlike run
    lengths) and insensitive to letter case. Aggregating over all words makes it stable.
    """
    area = boundary = 0
    for w in words:
        ab = _ink_area_and_boundary(ink, w)
        if ab:
            area += ab[0]
            boundary += ab[1]
    if boundary == 0:
        return None
    return 2.0 * area / boundary


def _reference_height(words: list[OCRWord]) -> float | None:
    """Median height of words whose letters reach cap height but have no descenders."""
    hs = [w.height for w in words
          if w.text and not (set(w.text) & _DESCENDER_CHARS) and (set(w.text) & _HEIGHT_CHARS)]
    if not hs:
        hs = [w.height for w in words if w.text]
    return float(statistics.median(hs)) if hs else None


def estimate_heading_bold(ink: np.ndarray, heading_words: list[OCRWord], body_words: list[OCRWord],
                          *, th: Thresholds = THRESHOLDS) -> tuple[Status, float | None, str]:
    """Compare stroke width of the heading with the body text of the statement.

    Returns (status, ratio, note). ratio > 1 means the heading strokes are thicker than the
    body's after normalizing for text size (cap height of words without descenders). This is a
    heuristic: it assumes the body is set in regular weight at a similar size, which is how the
    statement is printed in practice.
    """
    head_sw = stroke_width(ink, heading_words)
    body_sw = stroke_width(ink, body_words)
    head_h, body_h = _reference_height(heading_words), _reference_height(body_words)
    if not head_sw or not body_sw or len(body_words) < 3 or not head_h or not body_h:
        return Status.REVIEW, None, "Could not measure the heading's weight. Please check it by eye."
    if min(head_h, body_h) < th.bold_min_text_px:
        return Status.REVIEW, None, "The statement is printed too small to measure its weight. Please check it by eye."
    ratio = round((head_sw / head_h) / (body_sw / body_h), 2)
    if ratio >= th.bold_ratio:
        return Status.PASS, ratio, f"Heading looks bold (strokes {ratio:.2f}x thicker than the text)."
    status = Status.FAIL if th.bold_failure_is_fail else Status.REVIEW
    return status, ratio, (f"Heading does not look bolder than the rest of the statement (strokes {ratio:.2f}x). "
                           "Please check by eye: the heading must be bold and the rest of the statement must not be.")


# --- putting it together -----------------------------------------------------------------------
_HEADING_TOKENS = ("government", "warning")


def heading_words(words: list[OCRWord], heading_text: str) -> list[OCRWord]:
    """The word boxes of "GOVERNMENT" and "WARNING:" among the words of the heading line(s): read cleanly,
    glued to a neighbour ("WARNING:(1)"), or misread the way the capitals check tolerates ("ERNMENT WARMING")."""
    found = [w for w in words if (_loose_tokens(w) or [""])[0] in _HEADING_TOKENS]
    if len(found) >= 2:
        return found
    misread = _misread_heading(normalize_strict(heading_text))
    if misread:
        for target in misread[:2]:
            for w in words:
                if normalize_strict(w.text) == target or normalize_strict(w.text).rstrip(":;.") == target:
                    if not any(w is f for f in found):
                        found.append(w)
                    break
    return found


def check_warning(lines: list[str], words: list[OCRWord] | None = None, ink: np.ndarray | None = None,
                  *, bold_hint: bool | None = None, views=None, th: Thresholds = THRESHOLDS) -> WarningResult:
    # Lines are grouped by frame: the views that share a coordinate system (the upright read and the
    # contrast read of the same image) are searched together, so a heading read in one and a body read
    # in the other still make one statement; each turned view is a frame of its own. A second engine's
    # read (RapidOCR) is a frame of its own too: it often runs words together ("ALCOHOLICBEVERAGES"),
    # and mixed into Tesseract's frame its line could be taken instead of Tesseract's reading of the same
    # place (measured: the warning got worse on 6 of 20 real labels); alone, it wins only when its own
    # statement reads closer to the required text.
    frame_of_view: dict[int, int] = {}
    if views:
        keys: dict[tuple, int] = {}
        for v_idx, v in enumerate(views):
            frame_of_view[v_idx] = keys.setdefault((v.rot, tuple(v.size), getattr(v, "engine", "")), len(keys))
    line_views: list[int] = [0] * len(lines)
    line_tops: list[float] = [float(i) for i in range(len(lines))]
    by_line: dict[int, list[OCRWord]] = {}
    if words:
        for w in words:
            if 0 <= w.line_index < len(lines):
                by_line.setdefault(w.line_index, []).append(w)
        for i, ws in by_line.items():
            line_views[i] = frame_of_view.get(ws[0].view, ws[0].view)
            line_tops[i] = sum(w.top for w in ws) / len(ws)
    # Tesseract's reading of the statement is judged when it found one; a RapidOCR read only when it did
    # not. RapidOCR finds statements Tesseract misses, but it runs words together in small print
    # ("GOVERNMENTWARNING:(1)ACCORDING"), which the word-for-word and capitals checks would count against
    # the label: measured on the real labels, letting its reading compete turned a REVIEW into a FAIL.
    rapid_frames = {frame_of_view[v_idx] for v_idx, v in enumerate(views or []) if getattr(v, "engine", "") == "rapid"}
    rapid_lines = {i for i in by_line if line_views[i] in rapid_frames}
    span = None
    if rapid_lines and len(rapid_lines) < len(by_line):
        span = locate_warning(["" if i in rapid_lines else l for i, l in enumerate(lines)], line_views=line_views,
                              line_tops=line_tops if words else None,
                              line_words={i: ws for i, ws in by_line.items() if i not in rapid_lines}, th=th)
    if span is None:
        span = locate_warning(lines, line_views=line_views, line_tops=line_tops if words else None,
                              line_words=by_line or None, th=th)
    if span is None:
        return WarningResult(present=False, wording=Status.FAIL, wording_note="No government warning statement was found on the label.",
                             heading_caps=Status.FAIL, heading_caps_note="Not found.",
                             heading_bold=Status.FAIL, heading_bold_note="Not found.", overall=Status.FAIL)
    found_lines = join_hyphenated(list(span.texts) if span.texts else [lines[i] for i in span.line_ids])
    found_text = "\n".join(found_lines)
    wording, score, wording_note, diff = check_wording(found_text, th=th)
    if span.second_copies:
        # Sentence (1) printed on the front and again on the back, or (2) left out of one copy: the agent
        # must see that the statement is not printed once, whole.
        copies = "; ".join(f"sentence ({n}) is printed again elsewhere ('{t[:90]}')" for n, t in span.second_copies)
        diff = diff + [DiffItem(expected="", found=t[:90]) for _n, t in span.second_copies]
        if wording == Status.PASS:
            wording, wording_note = Status.REVIEW, f"Wording matches, but {copies}. Please check the label."
        else:
            wording_note += f" Also, {copies}."
    if span.trimmed:
        # Words left out on wording alone are always quoted to the agent: they may instead be words
        # added to the statement, which the regulation does not allow.
        aside = (f"text printed beside the statement was left out ('{span.trimmed}'). "
                 "Please check it is not part of the warning.")
        if wording == Status.PASS:
            wording, wording_note = Status.REVIEW, "Wording matches, but " + aside
        else:
            wording_note += " Also, " + aside
    if span.conflicts:
        disagree = "; ".join(span.conflicts[:3])
        if wording == Status.PASS:
            wording, wording_note = Status.REVIEW, f"Wording matches, but another reading of the same line said {disagree}. Please check the label."
        else:
            wording_note += f" Another reading of the same line said {disagree}."
    if span.beside:
        # A neighbouring column, told apart by its word boxes: not judged as wording, but shown.
        quoted = span.beside if len(span.beside) <= 240 else span.beside[:240].rsplit(" ", 1)[0] + " ..."
        wording_note += f" Text printed beside the statement was left out ('{quoted}')."
    text_of = dict(zip(span.line_ids, span.texts)) if span.texts else {}
    head_ids = span.heading_lines or (span.line_ids[0],)
    heading_text = " ".join(text_of.get(i, lines[i]) for i in head_ids)
    caps, caps_note = check_heading_caps(heading_text)
    in_span = set(span.line_ids)

    if bold_hint is not None:
        if bold_hint:
            bold, ratio, bold_note = Status.PASS, None, "Heading reported as bold by the vision model."
        else:
            bold = Status.FAIL if th.bold_failure_is_fail else Status.REVIEW
            ratio, bold_note = None, "Heading reported as NOT bold by the vision model. Please check it by eye."
    elif words is not None and ink is not None and span.heading_lines:
        span_words = list(span.kept_words) or [w for w in words if w.line_index in in_span]
        head_ws = heading_words([w for w in span_words if w.line_index in head_ids], heading_text)
        # Measure in the ink of the view that read the statement (a sideways warning is measured turned
        # upright); a heading read in the other view of the same frame has valid boxes there too.
        view = Counter(w.view for w in span_words).most_common(1)[0][0]
        view_ink = views[view].ink if views and view < len(views) else ink
        body_words = [w for w in span_words if not any(w is h for h in head_ws) and len(normalize_loose(w.text)) >= 3]
        bold, ratio, bold_note = estimate_heading_bold(view_ink, head_ws, body_words, th=th)
    else:
        bold, ratio, bold_note = Status.REVIEW, None, "Could not measure the heading's weight. Please check it by eye."

    statuses = [wording, caps, bold]
    if Status.FAIL in statuses:
        overall = Status.FAIL
    elif Status.REVIEW in statuses:
        overall = Status.REVIEW
    else:
        overall = Status.PASS
    box = None
    if words is not None and ink is not None:
        span_words = [w for w in words if w.line_index in in_span]
        if span_words:
            h, w_ = ink.shape
            boxes = [upright_box(w, views or []) for w in span_words]
            left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
            right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
            box = [round(100 * left / w_, 2), round(100 * top / h, 2),
                   round(100 * (right - left) / w_, 2), round(100 * (bottom - top) / h, 2)]
    return WarningResult(present=True, found_text=found_text, wording=wording, wording_score=score,
                         wording_note=wording_note, diff=diff, heading_caps=caps, heading_caps_note=caps_note,
                         heading_bold=bold, heading_bold_note=bold_note, bold_ratio=ratio, overall=overall, box=box)
