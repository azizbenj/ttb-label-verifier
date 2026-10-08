#!/usr/bin/env python
"""Run the sample set (or the batch set) through the verifier and report accuracy and timings.

    python scripts/bench.py                       # local pipeline, data/samples
    python scripts/bench.py --set batch           # local pipeline, data/batch
    python scripts/bench.py --url https://host    # remote /api/verify (samples only, by name)
    python scripts/bench.py --psm 11              # try another Tesseract page-segmentation mode
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin")


def rows_for(which: str):
    d = ROOT / "data" / which
    with open(d / ("samples.csv" if which == "samples" else "applications.csv"), newline="") as f:
        return d, list(csv.DictReader(f))


def run_remote(url: str, rows, image_dir, psm=None):
    import httpx
    out = []
    with httpx.Client(timeout=60) as c:
        for r in rows:
            data = {k: r[k] for k in FIELDS}
            data["sample"] = Path(r["image"]).stem
            if psm is not None:
                data["psm"] = str(psm)
            resp = c.post(f"{url.rstrip('/')}/api/verify", data=data)
            out.append(resp.json())
    return out


def run_local(rows, image_dir, psm):
    from PIL import Image
    from app.models import Application
    from app.pipeline import verify
    from app.readers.tesseract import TesseractReader
    reader = TesseractReader(psm=psm)
    out = []
    for r in rows:
        app = Application(**{k: r[k] for k in FIELDS})
        out.append(verify(app, Image.open(image_dir / r["image"]), reader).model_dump(mode="json"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="samples", choices=["samples", "batch"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--psm", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    image_dir, rows = rows_for(a.set)
    if a.psm is not None:
        import os
        os.environ["TESSERACT_PSM"] = str(a.psm)
    results = run_remote(a.url, rows, image_dir, a.psm) if a.url else run_local(rows, image_dir, a.psm or 6)
    ok, times = 0, []
    for r, res in zip(rows, results):
        if "error" in res:
            print(f"{r['image']:28} ERROR {res['error']}")
            continue
        times.append(res["timings"]["total_ms"])
        hit = res["overall"] == r["expected_overall"]
        ok += hit
        if not hit or a.verbose:
            flag = "ok " if hit else "BAD"
            fields = ", ".join(f"{f['key']}={f['verdict']}" for f in res["fields"] if f["verdict"] not in ("MATCH", "SKIPPED"))
            w = res["warning"]
            print(f"{flag} {r['image']:28} got {res['overall']:6} want {r['expected_overall']:6} "
                  f"{res['timings']['total_ms']:7.0f} ms | {fields} | warning={w['overall']} "
                  f"(wording={w['wording']} caps={w['heading_caps']} bold={w['heading_bold']} ratio={w['bold_ratio']})")
    n = len(times)
    print(f"\n{ok}/{len(rows)} labels got the expected overall verdict")
    if times:
        print(f"timing ms: median {statistics.median(times):.0f}  p95 {sorted(times)[int(0.95 * (n - 1))]:.0f}  max {max(times):.0f}")


if __name__ == "__main__":
    main()
