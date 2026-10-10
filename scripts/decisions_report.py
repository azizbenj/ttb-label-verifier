#!/usr/bin/env python
"""What the agents' review decisions say about the rules: which raise false alarms, which catch real problems.

    python scripts/decisions_report.py decisions.jsonl
    python scripts/decisions_report.py decisions.jsonl --csv calibration.csv

The log is the JSON-lines file the server appends to when DECISION_LOG is set (app/decision_log.py). The
last answer per job, application and question counts and an undone answer (``clear``) drops out. Per
question key the report prints how many labels raised that question, how many answers were about it,
the pass / fail / skip answers, and the pass rate: pass / (pass + fail),
the share of that rule's flags the agents found to be false alarms. Then the ten most frequent notes
behind pass answers (the rules that cost agents the most time) and behind fail answers (the rules that
catch real problems).

--csv writes a calibration file in the layout of scripts/real_labels.csv, one row per label answered pass
or fail (a skip carries no information): the application values from the log, ``<field>=review`` in
``expect`` when the agent answered pass on a NEAR MATCH question (a legitimate near match, which
scripts/real_labels.py then expects to come back NEAR MATCH), nothing when they answered fail, the
application id as ``ttbid`` and the decision with the question in ``notes``. Standard library only, so it
runs on a copy of the log anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# The columns of scripts/real_labels.csv, in its order.
CSV_COLUMNS = ("ttbid", "kind", "brand_name", "class_type", "alcohol_content", "net_contents",
               "bottler_name_address", "country_of_origin", "expect", "notes")
FIELD_KEYS = CSV_COLUMNS[2:8]
WARNING_KEYS = ("warning_wording", "warning_bold", "warning_caps")
NO_PROMPT = "(no question)"      # a decision on a label that raised no question
ANSWERS = ("pass", "fail", "skip")


def read_log(path) -> list[dict]:
    """Every record in the file, in order. A line that is not a JSON object is reported and skipped."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{n}: not a JSON record, skipped ({e})", file=sys.stderr)
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def latest_decisions(records: list[dict]) -> list[dict]:
    """The answer that stands for each question: the last one per job, application and question (``key``;
    blank in logs from before answers were kept per question, when one answer stood for the label). A
    question whose last record is an undo (``clear``) drops out. Ordered by when the standing answer was made."""
    standing: dict[tuple, dict] = {}
    for rec in records:
        key = (rec.get("job"), rec.get("application_id"), rec.get("key") or "")
        standing.pop(key, None)
        if rec.get("decision") != "clear":
            standing[key] = rec
    return list(standing.values())


def asked_prompt(rec: dict) -> dict | None:
    """The question the answer is about (marked in the log), else the first one raised."""
    prompts = rec.get("prompts") or []
    for p in prompts:
        if p.get("asked"):
            return p
    return prompts[0] if prompts else None


def note_for(rec: dict) -> str:
    """The rule text behind the answer: the field's note for a field question, the question itself for a
    warning check (there it names the difference: 'Does the label say "should"?')."""
    p = asked_prompt(rec)
    if p is None:
        return ""
    for f in rec.get("fields") or []:
        if f.get("key") == p.get("key"):
            return f.get("note") or p.get("question") or ""
    return p.get("question") or ""


def summarise(decided: list[dict]) -> list[dict]:
    """Per question key: raised (labels that raised it), asked (the queue asked it first), the pass / fail /
    skip answers to it, and pass_rate = pass / (pass + fail), None without an answer. Field questions first
    in the application's order, then the warning checks, then anything else."""
    rows: dict[str, Counter] = {}
    seen: set[tuple] = set()   # a label answered question by question appears once per answer: count it once
    for rec in decided:
        label = (rec.get("job"), rec.get("application_id"))
        for p in rec.get("prompts") or []:
            c = rows.setdefault(p.get("key") or "?", Counter())
            if label not in seen:
                c["raised"] += 1
        seen.add(label)
        p = asked_prompt(rec)
        c = rows.setdefault((p.get("key") or "?") if p else NO_PROMPT, Counter())
        c["asked"] += 1
        if rec.get("decision") in ANSWERS:
            c[rec["decision"]] += 1
    order = (*FIELD_KEYS, *WARNING_KEYS)
    keys = [k for k in order if k in rows] + sorted(k for k in rows if k not in order and k != NO_PROMPT)
    if NO_PROMPT in rows:
        keys.append(NO_PROMPT)
    out = []
    for k in keys:
        c = rows[k]
        answered = c["pass"] + c["fail"]
        out.append({"key": k, "raised": c["raised"], "asked": c["asked"], "pass": c["pass"], "fail": c["fail"],
                    "skip": c["skip"], "pass_rate": c["pass"] / answered if answered else None})
    return out


def top_notes(decided: list[dict], decision: str, n: int = 10) -> list[tuple[str, str, int]]:
    """The ``n`` most frequent (question key, note, count) behind answers of ``decision``."""
    c: Counter = Counter()
    for rec in decided:
        if rec.get("decision") != decision:
            continue
        p = asked_prompt(rec)
        c[((p.get("key") or "?") if p else NO_PROMPT, note_for(rec))] += 1
    return [(key, note, count) for (key, note), count in c.most_common(n)]


def calibration_rows(decided: list[dict]) -> list[dict]:
    """One scripts/real_labels.csv row per label decided pass or fail. A label is answered question by
    question; its decision is the one logged with its latest answer (``label_decision``; logs from before
    answers were kept per question carry only ``decision``, which stood for the label)."""
    labels: dict[tuple, list[dict]] = {}
    for rec in decided:
        labels.setdefault((rec.get("job"), rec.get("application_id")), []).append(rec)
    rows = []
    for recs in labels.values():
        last = max(recs, key=lambda r: r.get("ts") or "")
        d = last.get("label_decision", last.get("decision"))
        if d not in ("pass", "fail"):
            continue
        values = {f.get("key"): f.get("expected") or "" for f in last.get("fields") or []}
        expects, questions = [], []
        for rec in recs:
            p = asked_prompt(rec)
            if rec.get("decision") == "pass" and p and p.get("key") in FIELD_KEYS and p.get("verdict") == "NEAR MATCH":
                expects.append(f"{p['key']}=review")      # the agent confirmed it: a legitimate NEAR MATCH
            if p and p.get("question"):
                questions.append(f"{rec.get('decision')}: {p['question']}")
        row = {"ttbid": last.get("application_id") or "",
               "kind": "import" if values.get("country_of_origin") else ""}
        row.update({k: values.get(k, "") for k in FIELD_KEYS})
        row["expect"] = ";".join(dict.fromkeys(expects))
        row["notes"] = " | ".join(questions) if questions else d
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def print_report(path, records: list[dict], decided: list[dict], out=sys.stdout) -> None:
    print(f"{path}: {len(records)} records, {len(decided)} answers that stand "
          "(the last answer per question counts; an undone answer drops out)", file=out)
    print(file=out)
    print(f"{'question':24} {'raised':>6} {'asked':>6} {'pass':>5} {'fail':>5} {'skip':>5} {'pass rate':>9}", file=out)
    for r in summarise(decided):
        rate = f"{r['pass_rate'] * 100:.0f}%" if r["pass_rate"] is not None else "-"
        print(f"{r['key']:24} {r['raised']:6} {r['asked']:6} {r['pass']:5} {r['fail']:5} {r['skip']:5} {rate:>9}",
              file=out)
    print("\nraised: labels that raised the question; asked: answers about this question;\npass rate: pass / (pass + fail), the share of this rule's flags the agents found to be false "
          "alarms.", file=out)
    for decision, title in (("pass", 'Notes behind "pass" answers (false alarms: the rules that cost agents the most time)'),
                            ("fail", 'Notes behind "fail" answers (the rules that catch real problems)')):
        print(f"\n{title}", file=out)
        notes = top_notes(decided, decision)
        if not notes:
            print("  none", file=out)
        for key, note, count in notes:
            print(f"  {count:4}  {key:20}  {note}", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise a DECISION_LOG file: pass rates per rule, the notes behind "
                                             "pass and fail answers, and optionally a calibration CSV.")
    ap.add_argument("log", type=Path, help="the JSON-lines file DECISION_LOG points at")
    ap.add_argument("--csv", type=Path, metavar="OUT",
                    help="write a calibration CSV in the layout of scripts/real_labels.csv")
    a = ap.parse_args(argv)
    records = read_log(a.log)
    decided = latest_decisions(records)
    print_report(a.log, records, decided)
    if a.csv:
        rows = calibration_rows(decided)
        write_csv(rows, a.csv)
        print(f"\nwrote {len(rows)} rows to {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
