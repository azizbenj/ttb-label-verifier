#!/usr/bin/env python
"""Build the calibration ground truth from the registry's filed values and per-label transcriptions.

    python scripts/build_calibration.py            # writes scripts/calibration_labels.csv

Inputs (all under the gitignored data/real/):
  raw/registry.csv          what the applicant filed (brand, class/type description, origin), from
                            fetch_registry_labels.py --search
  transcripts/<ttbid>.json  what the label prints, read by a person or a vision model:
      {"brand_name": "...", "class_type": "...", "alcohol_content": "...", "net_contents": "...",
       "bottler_name_address": "...", "country_of_origin": "...",
       "legible": true, "notes": "...", "expect": {"brand_name": "review"}}
      A field is the statement exactly as printed (case, punctuation); "" when the label does not print it.
      "expect" (optional) follows scripts/real_labels.csv: review = the tool should ask for a look even with
      this exact text (the brand only in the bottler line, the class inside a longer phrase); absent = the
      field is not on the label at all.

Output: scripts/calibration_labels.csv in the exact layout of scripts/real_labels.csv, so
`python scripts/real_labels.py --csv scripts/calibration_labels.csv --defects` scores it. Rows whose
transcription is missing, marked illegible, or lacks a required value are listed and left out.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "real" / "raw"
TRANSCRIPTS = ROOT / "data" / "real" / "transcripts"
OUT = ROOT / "scripts" / "calibration_labels.csv"
FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin")
REQUIRED = FIELDS[:4]


def main() -> None:
    registry = {r["ttbid"]: r for r in csv.DictReader((RAW / "registry.csv").open(newline=""))}
    rows, left_out = [], []
    for ttbid, reg in sorted(registry.items()):
        path = TRANSCRIPTS / f"{ttbid}.json"
        if not path.exists():
            left_out.append((ttbid, "no transcript"))
            continue
        t = json.loads(path.read_text())
        if not t.get("legible", True):
            left_out.append((ttbid, "marked illegible: " + t.get("notes", "")))
            continue
        missing = [k for k in REQUIRED if not (t.get(k) or "").strip()]
        if missing:
            left_out.append((ttbid, "not printed or not read: " + ", ".join(missing)))
            continue
        kind = "import" if reg["imported"] == "yes" else reg["kind"]
        expect = ";".join(f"{k}={v}" for k, v in (t.get("expect") or {}).items())
        notes = " · ".join(x for x in (t.get("notes", ""), f"filed brand {reg['brand_name']!r}, class {reg['class_desc']!r}, "
                                                        f"origin {reg['origin_desc']!r}") if x)
        rows.append({"ttbid": ttbid, "kind": kind, **{k: (t.get(k) or "").strip() for k in FIELDS},
                     "expect": expect, "notes": notes})
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ttbid", "kind"] + list(FIELDS) + ["expect", "notes"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["kind"] for r in rows)
    print(f"{len(rows)} labels written to {OUT.relative_to(ROOT)}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    if left_out:
        print(f"{len(left_out)} left out:")
        for ttbid, why in left_out:
            print(f"  {ttbid}: {why}")


if __name__ == "__main__":
    main()
