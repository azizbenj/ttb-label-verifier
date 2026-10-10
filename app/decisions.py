"""The questions a label that needs a look asks the agent (Review-Queue and Copy boards).

A prompt names what to look at, shows the two readings side by side, asks a yes/no question and
labels the answers. "Yes" always means the label is fine (a reading error on our side), "No" that the
label is wrong. Every field or warning check that is not a clean match asks one, a MISMATCH included,
so an agent can record a misread instead of sending a good label back. Answers never change a verdict;
they travel with the batch into the export."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .config import FIELD_BY_KEY, THRESHOLDS
from .matching import NOTE_AMBIGUOUS_MISREAD
from .models import Status, VerificationResult, Verdict
from .warning import reported_meaning_changes


@dataclass(frozen=True)
class Prompt:
    key: str                 # field key, or warning_wording / warning_bold / warning_caps
    what: str                # "Brand name", "Government warning · wording"
    verdict: str             # chip text: NEAR MATCH, 1 DIFFERENCE, COULD NOT TELL
    question: str
    yes: str                 # the label is fine
    no: str                  # the label is wrong
    why: str
    left_head: str = "On the application"
    right_head: str = "On the label"
    left: tuple[str, str, str] = ("", "", "")     # (before, highlighted, after)
    right: tuple[str, str, str] = ("", "", "")
    box: list[float] | None = None
    # Whether "Yes" is the answer the evidence leans to, shown as the suggested button. False when the tool
    # read something different, so the application's value is never presented as the likely truth.
    lean: bool = True


def _quote(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def _split_phrases(required: str, found: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """(before, differing words, after) of the required and the label's phrase: "should not drink" and
    "should drink" give ("should ", "not", " drink") and ("should ", "", " drink")."""
    a, b = required.split(), found.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return ("", required, ""), ("", found, "")
    _tag, i1, i2, j1, j2 = ops[0]

    def parts(ws: list[str], s: int, e: int) -> tuple[str, str, str]:
        before, mid, after = " ".join(ws[:s]), " ".join(ws[s:e]), " ".join(ws[e:])
        return (before + " " if before else "", mid, (" " + after) if after else "")

    return parts(a, i1, i2), parts(b, j1, j2)


def prompts_for(result: VerificationResult) -> list[Prompt]:
    """Every question this result raises, in the order of the result table; the review queue asks each."""
    out: list[Prompt] = []
    for f in result.fields:
        label = FIELD_BY_KEY[f.key].label if f.key in FIELD_BY_KEY else f.label
        exp, found = f.expected or "", f.found or ""
        if f.verdict == Verdict.MISMATCH:
            out.append(Prompt(f.key, label, "MISMATCH", f"Does the label say {_quote(exp)}?",
                              f"Yes, it says {_quote(exp)}: reading error", "No, the label differs", f.note or "",
                              left=("", exp, ""), right=("", found, ""), box=f.box, lean=False))
            continue
        if f.verdict == Verdict.NOT_FOUND:
            out.append(Prompt(f.key, label, "NOT FOUND", f"Is {_quote(exp)} printed on the label?",
                              "Yes, it is printed: reading error", "No, it is missing", f.note or "",
                              left=("", exp, ""), right=("", found or "(nothing read)", ""), box=f.box, lean=False))
            continue
        if f.verdict != Verdict.NEAR_MATCH:
            continue
        note = (f.note or "").lower()
        lean = True
        if f.key == "brand_name" and ("capitalization" in note or "punctuation" in note):
            q, yes, no = "Is this the same brand name?", "Yes, same brand", "No, different"
        elif "another reading of the same place" in note:
            q, yes, no = f"Does the label say {_quote(exp)}?", f"Yes, {_quote(exp)}: reading error", "No, it says something else"
        elif f.key == "alcohol_content" and "does not agree with its percentage" in note:
            q, yes, no = ("Does the label's proof agree with its percentage (proof = 2 × ABV)?", "Yes: reading error",
                          "No, the label contradicts itself")
        elif f.key == "alcohol_content" and "not marked as alcohol" in note:
            q, yes, no = (f"Is {_quote(found)} the label's alcohol content statement?", "Yes, that is the statement",
                          "No, the statement is missing or different")
        elif f.key == "net_contents" and "decimal" in note:
            q, yes, no = f"Does the label say {exp}?", f"Yes, {exp}", "No, something else"
        elif f.key == "net_contents" and NOTE_AMBIGUOUS_MISREAD in note:
            # a misread that is one digit from more than one standard size: the evidence leans nowhere
            q, yes, no, lean = f"Does the label say {exp}?", f"Yes, it says {exp}", "No, it says something else", False
        elif f.key == "net_contents" and ("reading error" in note or "also reads" in note):
            q, yes, no = f"Does the label say {exp}?", f"Yes, {exp}: reading error", "No, the label differs"
        elif f.key == "alcohol_content":
            q, yes, no = f"Does the label say {exp}?", f"Yes, {exp}: reading error", "No, the label differs"
        else:
            q, yes, no = f"Is this the same {label.lower()}?", "Yes, the same", "No, different"
        out.append(Prompt(f.key, label, "NEAR MATCH", q, yes, no, f.note or "",
                          left=("", exp, ""), right=("", found, ""), box=f.box, lean=lean))
    w = result.warning
    if not w.present:
        out.append(Prompt("warning_present", "Government warning", "NOT FOUND",
                          "Is the government warning printed on the label?", "Yes, it is printed: reading error",
                          "No, it is missing", w.wording_note or "", "Required", "On the label",
                          ("", "GOVERNMENT WARNING: (1) According to the Surgeon General…", ""),
                          ("", "(nothing read)", ""), w.box, lean=False))
    meaning = reported_meaning_changes(w.wording_note) if w.present and w.wording == Status.FAIL else []
    if meaning:
        # A difference that changes what the warning says ("should drink" for "should not drink"): the
        # question names the word at stake so a "Yes" is a deliberate look at the label, never a reflex.
        found, required = meaning[0]
        (rb, rw, ra), (fb, fw, fa) = _split_phrases(required, found)
        if rw and fw:
            q = f"Does the label print {_quote(rw)} in {_quote(required)}, not {_quote(fw)}?"
        elif rw:
            q = f"Does the label print {_quote(rw)} in {_quote(required)}?"
        else:
            q = f"Does the label print {_quote(required)}, without {_quote(fw)}?"
        why = (f"{w.wording_note} The warning must be printed word for word (27 CFR 16.21); this difference "
               "fails the label unless the label itself is correct and the tool misread it.")
        out.append(Prompt("warning_wording", "Government warning · wording", "CHANGES MEANING",
                          q, f"Yes, it prints {_quote(required)}: reading error",
                          f"No, it says {_quote(found)}: misprint", why, "Required text says", "Label says",
                          (rb, rw or "", ra), (fb, fw or "(nothing)", fa), w.box, lean=False))
    elif w.present and w.wording in (Status.REVIEW, Status.FAIL) and w.diff:
        d = w.diff[0]
        n = len(w.diff)
        exp, found = d.expected or "(nothing)", d.found or "(nothing)"
        # "Yes" always means the label is fine (the first button, key Y), whatever the wording result:
        # a FAIL used to ask "Does the label say <the misprint>?" with "No, ...: reading error" on Y.
        if d.expected and d.found:
            q = f"Does the label say {_quote(exp)}?"
            yes, no = f"Yes, it says {_quote(exp)}: reading error", f"No, it says {_quote(found)}: misprint"
        elif d.expected:   # words missing from the label as read
            q = f"Is {_quote(exp)} printed on the label?"
            yes, no = "Yes, it is printed: reading error", "No, it is missing: misprint"
        else:              # words on the label that the required text does not have
            q = f"Is {_quote(found)} only a reading error (not printed in the statement)?"
            yes, no = "Yes, reading error", "No, the label adds it: misprint"
        why = (f"Similarity {w.wording_score} / 100; review from {THRESHOLDS.warning_near}, fail below. "
               f"Look at the crop: does the label print {_quote(found)}?")
        out.append(Prompt("warning_wording", "Government warning · wording", f"{n} DIFFERENCE{'S' if n != 1 else ''}",
                          q, yes, no, why, "Required text says", "Label says",
                          (d.expected_before or "", exp, d.expected_after or ""),
                          (d.found_before or "", found, d.found_after or ""), w.box, lean=False))
    elif w.present and w.wording in (Status.REVIEW, Status.FAIL):   # no word diff (text beside it was left out)
        out.append(Prompt("warning_wording", "Government warning · wording", "NEEDS A LOOK",
                          "Is the statement printed complete and correct?", "Yes, it is correct", "No, it is wrong",
                          w.wording_note or "", "Required", "On the label", ("", "the statement word for word", ""),
                          ("", w.found_text or "", ""), w.box, lean=False))
    if w.present and w.heading_bold in (Status.REVIEW, Status.FAIL):
        measured = (f"Heading strokes {w.bold_ratio:.2f}× thicker (regular text measures 1.03–1.14×)"
                    if w.bold_ratio is not None else "The heading's weight could not be measured")
        out.append(Prompt("warning_bold", "Government warning · heading bold", "COULD NOT TELL",
                          'Is "GOVERNMENT WARNING:" printed in bold?', "Yes, it is bold", "No, it is not bold",
                          "A heuristic from stroke width; it only asks for a look, it never fails a label on its own.",
                          "Required", "Measured on the label",
                          ("Heading strokes at least ", f"{THRESHOLDS.bold_ratio:.2f}×", " thicker than the text"),
                          ("", measured, ""), w.box, lean=w.heading_bold == Status.REVIEW))
    if w.present and w.heading_caps in (Status.REVIEW, Status.FAIL):
        failed = w.heading_caps == Status.FAIL
        out.append(Prompt("warning_caps", "Government warning · heading", "FAIL" if failed else "NEEDS A LOOK",
                          'Is the heading printed as "GOVERNMENT WARNING:" in capitals?',
                          "Yes, in capitals: reading error" if failed else "Yes, in capitals", "No, it is not",
                          w.heading_caps_note or "", "Required", "As read",
                          ("", "GOVERNMENT WARNING:", ""), ("", w.found_text.split("\n")[0] if w.found_text else "", ""),
                          w.box, lean=not failed))
    return out


def prompt_map(result: VerificationResult) -> dict[str, Prompt]:
    return {p.key: p for p in prompts_for(result)}
