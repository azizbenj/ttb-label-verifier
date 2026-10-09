"""The one question a label that needs a look asks the agent (Review-Queue and Copy boards).

A prompt names what to look at, shows the two readings side by side, asks a yes/no question and
labels the answers. "Yes" always means the label is fine (a reading error on our side), "No" that the
label is wrong. Decisions never change a verdict; they travel with the batch into the export."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FIELD_BY_KEY, THRESHOLDS
from .models import Status, VerificationResult, Verdict


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


def _quote(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def prompts_for(result: VerificationResult) -> list[Prompt]:
    """Every question this result raises, the most pressing first (the review queue asks the first)."""
    out: list[Prompt] = []
    for f in result.fields:
        if f.verdict != Verdict.NEAR_MATCH:
            continue
        label = FIELD_BY_KEY[f.key].label if f.key in FIELD_BY_KEY else f.label
        note = (f.note or "").lower()
        exp, found = f.expected or "", f.found or ""
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
        elif f.key == "net_contents" and ("reading error" in note or "also reads" in note):
            q, yes, no = f"Does the label say {exp}?", f"Yes, {exp}: reading error", "No, the label differs"
        elif f.key == "alcohol_content":
            q, yes, no = f"Does the label say {exp}?", f"Yes, {exp}: reading error", "No, the label differs"
        else:
            q, yes, no = f"Is this the same {label.lower()}?", "Yes, the same", "No, different"
        out.append(Prompt(f.key, label, "NEAR MATCH", q, yes, no, f.note or "",
                          left=("", exp, ""), right=("", found, ""), box=f.box))
    w = result.warning
    if w.present and w.wording in (Status.REVIEW, Status.FAIL) and w.diff:
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
                          (d.found_before or "", found, d.found_after or ""), w.box))
    elif w.present and w.wording == Status.REVIEW:   # text beside the statement was left out
        out.append(Prompt("warning_wording", "Government warning · wording", "NEEDS A LOOK",
                          "Is the statement printed complete and correct?", "Yes, it is correct", "No, it is wrong",
                          w.wording_note or "", "Required", "On the label", ("", "the statement word for word", ""),
                          ("", w.found_text or "", ""), w.box))
    if w.present and w.heading_bold == Status.REVIEW:
        measured = (f"Heading strokes {w.bold_ratio:.2f}× thicker (regular text measures 1.03–1.14×)"
                    if w.bold_ratio is not None else "The heading's weight could not be measured")
        out.append(Prompt("warning_bold", "Government warning · heading bold", "COULD NOT TELL",
                          'Is "GOVERNMENT WARNING:" printed in bold?', "Yes, it is bold", "No, it is not bold",
                          "A heuristic from stroke width; it only asks for a look, it never fails a label on its own.",
                          "Required", "Measured on the label",
                          ("Heading strokes at least ", f"{THRESHOLDS.bold_ratio:.2f}×", " thicker than the text"),
                          ("", measured, ""), w.box))
    if w.present and w.heading_caps == Status.REVIEW:
        out.append(Prompt("warning_caps", "Government warning · heading", "NEEDS A LOOK",
                          'Is the heading printed as "GOVERNMENT WARNING:" in capitals?', "Yes, in capitals",
                          "No, it is not", w.heading_caps_note or "", "Required", "As read",
                          ("", "GOVERNMENT WARNING:", ""), ("", w.found_text.split("\n")[0] if w.found_text else "", ""),
                          w.box))
    return out


def prompt_map(result: VerificationResult) -> dict[str, Prompt]:
    return {p.key: p for p in prompts_for(result)}
