#!/usr/bin/env python
"""Run the verifier on real approved labels from the public registry and score it field by field.

Fetch the images first (they are not in the repository):
    python scripts/fetch_registry_labels.py $(cut -d, -f1 scripts/real_labels.csv | tail -n +2)
    python scripts/real_labels.py --stitch          # combine each record's images into data/real/<id>.jpg
    python scripts/real_labels.py [--verbose] [--defects]

scripts/real_labels.csv holds what each label actually says, so every field should come back MATCH
unless the "expect" column says otherwise (review = a legitimate NEAR MATCH, absent = really not on
the label). --defects also checks each label against a wrong alcohol content and a wrong net contents,
which must never come back MATCH.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.images import open_image  # noqa: E402
from app.models import Application  # noqa: E402
from app.normalize import parse_alcohol, parse_net_contents  # noqa: E402
from app.pipeline import verify  # noqa: E402
from app.readers.tesseract import TesseractReader  # noqa: E402

REAL = ROOT / "data" / "real"
FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin")


def stitch() -> None:
    """Combine each record's images (front, back, neck) into one file, the way the app stacks several
    uploads for one application (same function, same pixel cap), so the scorer sees what the app would."""
    import warnings
    from app.images import stitch as stack
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    Image.MAX_IMAGE_PIXELS = None
    recs = json.loads((REAL / "raw" / "records.json").read_text())
    n = 0
    for t, r in recs.items():
        ims = [ImageOps.exif_transpose(Image.open(REAL / "raw" / im["file"])).convert("RGB") for im in r["images"]]
        if not ims:
            continue
        stack(ims).save(REAL / f"{t}.jpg", quality=92)
        n += 1
    print(f"stitched {n} records into {REAL}")


def wrong_abv(text: str) -> str | None:
    v = parse_alcohol(text)
    if v is None:   # a statement the parser does not read ("57.5 ALC BY VOL"): nothing to plant
        return None
    return f"{v.abv + (5 if v.abv > 20 else 1.5):g}% Alc./Vol."


def wrong_volume(text: str) -> str:
    v = parse_net_contents(text)
    if v is None:
        return "1 L"
    return "1 L" if abs(v.ml - 1000) > 1 else "750 mL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--defects", action="store_true")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--reader", default="tesseract", choices=["tesseract", "rapid", "rapid+tesseract"],
                    help="the primary reader (RapidOCR escalation of tesseract follows RAPID_ESCALATION)")
    ap.add_argument("--csv", default=str(ROOT / "scripts" / "real_labels.csv"),
                    help="ground truth to score (default: the 20 labels transcribed during development); several: a,b")
    ap.add_argument("--json", help="write every field's verdict, found text and note here, for diffs between builds")
    ap.add_argument("--only", help="comma-separated kinds to score (spirits,wine,beer,import)")
    a = ap.parse_args()
    from app.readers.rapid import warm_up as rapid_warm_up
    rapid_warm_up()   # start the RapidOCR workers first, as the server does at start-up
    if a.stitch:
        stitch()
        return
    rows = []
    for path in a.csv.split(","):
        rows += list(csv.DictReader(open(path, newline="")))
    rows = [r for r in rows if (REAL / f"{r['ttbid']}.jpg").exists()]
    skipped = [r["ttbid"] for r in rows if not all(r.get(k, "").strip() for k in FIELDS[:4])]
    rows = [r for r in rows if r["ttbid"] not in skipped]
    if a.only:
        rows = [r for r in rows if r["kind"] in a.only.split(",")]
    if skipped:
        print(f"{len(skipped)} rows skipped: a required application value is blank in the ground truth")
    if not rows:
        sys.exit("No images in data/real/. Run fetch_registry_labels.py and --stitch first.")
    if a.reader.startswith("rapid"):
        from app.readers.rapid import RapidOCRReader
        reader = RapidOCRReader(escalate_with_tesseract=a.reader == "rapid+tesseract")
    else:
        reader = TesseractReader()

    def run(row, override=None):
        values = {k: row[k] for k in FIELDS}
        if override:
            values.update(override)
        for attempt in range(3):   # a file in a synced folder can take a moment to arrive
            try:
                data = (REAL / f"{row['ttbid']}.jpg").read_bytes()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
        img = open_image(data, f"{row['ttbid']}.jpg")
        return verify(Application(**values), img, reader)

    def run_safely(row):
        try:
            return run(row)
        except Exception as e:   # one bad file must not lose the whole run
            print(f"{row['ttbid']}: ERROR {type(e).__name__}: {e}")
            return None

    with ThreadPoolExecutor(a.j) as pool:
        results = list(pool.map(run_safely, rows))
    errors = [row["ttbid"] for row, r in zip(rows, results) if r is None]
    rows = [row for row, r in zip(rows, results) if r is not None]
    results = [r for r in results if r is not None]

    tally = {"right": 0, "conservative": 0, "false_alarm": 0, "accepted": 0}
    by_kind: dict[str, dict] = {}
    warn = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    overall = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    times = []
    dump = []
    print(f"{'record':16} {'kind':8} {'overall':7} {'time':>6}  fields that did not come back as expected")
    for row, r in zip(rows, results):
        expect = dict(e.split("=") for e in row["expect"].split(";") if e)
        times.append(r.timings.total_ms)
        warn[r.warning.overall.value] += 1
        kt = by_kind.setdefault(row["kind"], {"labels": 0, "right": 0, "conservative": 0, "false_alarm": 0,
                                              "accepted": 0, "warn_pass": 0})
        kt["labels"] += 1
        kt["warn_pass"] += r.warning.overall.value == "PASS"
        problems = []
        dump += dump_rows(row, r, "clean", expect)
        for f in r.fields:
            want = expect.get(f.key, "skip" if not row[f.key] else "match")
            got = f.verdict.value
            if want == "skip":
                continue
            if (want == "match" and got == "MATCH") or (want == "review" and got == "NEAR MATCH") or \
               (want == "absent" and got == "NOT FOUND"):
                tally["right"] += 1; kt["right"] += 1
            elif want == "review" and got == "MATCH":
                # The ground truth says this needs a look (the brand only in the bottler statement, the class
                # inside a longer phrase): a MATCH is the silent acceptance the rules exist to prevent.
                tally["accepted"] += 1; kt["accepted"] += 1
                problems.append(f"{f.key}: MATCH where a look was expected ({f.found!r})")
            elif got == "NEAR MATCH":
                tally["conservative"] += 1; kt["conservative"] += 1
                problems.append(f"{f.key}: NEAR MATCH ({f.found!r})")
            else:
                tally["false_alarm"] += 1; kt["false_alarm"] += 1
                problems.append(f"{f.key}: {got} ({f.found!r})")
        overall[r.overall.value] += 1
        w = r.warning
        wtxt = f"warning {w.overall.value}" + ("" if w.overall.value == "PASS" else
                                                f" (present={w.present} wording={w.wording.value} caps={w.heading_caps.value} bold={w.heading_bold.value})")
        print(f"{row['ttbid']:16} {row['kind']:8} {r.overall.value:7} {r.timings.total_ms / 1000:5.1f}s  "
              f"{'; '.join(problems) or 'all fields as expected'} · {wtxt}")
        if a.verbose:
            print("      " + row["notes"])
    n = sum(tally.values())
    print(f"\nfields: {tally['right']}/{n} as expected, {tally['conservative']} flagged for review, "
          f"{tally['false_alarm']} false alarms (MISMATCH or NOT FOUND for text that is on the label), "
          f"{tally['accepted']} accepted without the look the ground truth expects")
    print(f"government warning: {warn['PASS']} pass, {warn['REVIEW']} review, {warn['FAIL']} fail (all {len(rows)} labels carry the warning)")
    print(f"overall on the correct labels: {overall['PASS']} PASS, {overall['REVIEW']} REVIEW, {overall['FAIL']} FAIL")
    print(f"timing: median {statistics.median(times) / 1000:.2f} s, max {max(times) / 1000:.2f} s")
    if errors:
        print(f"{len(errors)} label(s) could not be run: {', '.join(errors)}")
    if len(by_kind) > 1:
        for k, t in sorted(by_kind.items()):
            fields = t["right"] + t["conservative"] + t["false_alarm"] + t["accepted"]
            print(f"  {k:8} {t['labels']:3} labels · fields {t['right']}/{fields} as expected, {t['conservative']} review, "
                  f"{t['false_alarm']} false alarms, {t['accepted']} accepted · warning pass {t['warn_pass']}/{t['labels']}")
    if a.defects:
        missed = caught = passed = 0
        planted_overall = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        def planted(row):
            out = []
            for key, val in (("alcohol_content", wrong_abv(row["alcohol_content"])), ("net_contents", wrong_volume(row["net_contents"]))):
                if val is None:
                    continue
                try:
                    out.append((key, val, run(row, {key: val})))
                except Exception as e:
                    print(f"{row['ttbid']}: ERROR on the planted {key}: {e}")
            return out
        with ThreadPoolExecutor(a.j) as pool:
            planted_results = list(pool.map(planted, rows))
        for row, outs in zip(rows, planted_results):
            expect = dict(e.split("=") for e in row["expect"].split(";") if e)
            for key, val, r in outs:
                dump.extend(dump_rows(row, r, f"planted_{key}", expect, planted=key))
                planted_overall[r.overall.value] += 1
                v = next(f for f in r.fields if f.key == key).verdict.value
                if v == "MATCH":
                    missed += 1
                    print(f"  MISSED: {row['ttbid']} {key} filed as {val!r} came back MATCH")
                else:
                    caught += 1
                if r.overall.value == "PASS":   # the outcome a compliance check must never produce
                    passed += 1
                    print(f"  PASSED: {row['ttbid']} with a wrong {key} ({val!r}) came back PASS")
        print(f"planted defects on real labels: {caught} caught, {missed} reported as MATCH, {passed} labels PASS; "
              f"overall {planted_overall['FAIL']} FAIL, {planted_overall['REVIEW']} REVIEW, {planted_overall['PASS']} PASS")
    if a.json:
        Path(a.json).write_text(json.dumps(dump, indent=1))
        print(f"wrote {len(dump)} rows to {a.json}")
    if a.defects and (missed or passed):
        sys.exit(1)   # a planted defect taken for a MATCH, or a planted label that PASSed


def dump_rows(row: dict, r, run: str, expect: dict, planted: str | None = None) -> list[dict]:
    """One row per field, one for the warning and one for the overall verdict of one run of one label.
    ``want`` is the ground truth: match / review / absent / skip, or "defect" for the field a planted run
    filed wrong. ``signals`` are the measurements the clarity of a MISMATCH or NOT FOUND is decided from."""
    base = {"ttbid": row["ttbid"], "kind": row["kind"], "run": run}
    out = []
    for f in r.fields:
        want = "defect" if f.key == planted else expect.get(f.key, "skip" if not row[f.key] else "match")
        out.append({**base, "field": f.key, "want": want, "verdict": f.verdict.value, "expected": f.expected,
                    "found": f.found, "note": f.note, "clear": getattr(f, "clear", None),
                    "signals": getattr(f, "signals", None)})
    w = r.warning
    out.append({**base, "field": "warning", "want": "pass", "verdict": w.overall.value, "expected": "",
                "found": w.found_text, "present": w.present, "wording": w.wording.value,
                "caps": w.heading_caps.value, "bold": w.heading_bold.value,
                "clear": getattr(w, "clear", None),
                "note": " | ".join(n for n in (w.wording_note, w.heading_caps_note, w.heading_bold_note) if n)})
    out.append({**base, "field": "overall", "want": "", "verdict": r.overall.value, "expected": "", "found": "",
                "note": f"{r.timings.total_ms:.0f} ms", "ms": r.timings.total_ms, "summary": r.summary,
                "read_confidence": r.read_confidence, "signals": getattr(r, "signals", None),
                "reader": r.reader})
    return out


if __name__ == "__main__":
    main()
