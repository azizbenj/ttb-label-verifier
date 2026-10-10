#!/usr/bin/env python
"""Run the sample set (or the batch set) through the verifier and report accuracy and timings.

    python scripts/bench.py                       # local pipeline, data/samples
    python scripts/bench.py --set batch           # local pipeline, data/batch
    python scripts/bench.py --set batch -j 4      # local, four labels at a time (accuracy runs; timings inflate)
    python scripts/bench.py --url https://host    # remote /api/verify (samples by name, batch images uploaded)
    python scripts/bench.py --psm 11              # try another Tesseract page-segmentation mode
    python scripts/bench.py --reader rapid        # RapidOCR as the primary reader (or rapid+tesseract)
    python scripts/bench.py --fail-under 1.0      # exit 1 below this accuracy or on any missed defect (CI)

A "missed defect" is a label the generator planted a problem in (expected REVIEW or FAIL) that came back
PASS: the outcome a compliance tool must never produce, so it is counted and gated separately.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
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
            if psm is not None:
                data["psm"] = str(psm)
            path = image_dir / r["image"]
            for attempt in range(3):  # transient network errors should not void a long run
                try:
                    if image_dir.name == "samples":
                        data["sample"] = path.stem
                        resp = c.post(f"{url.rstrip('/')}/api/verify", data=data)
                    else:
                        resp = c.post(f"{url.rstrip('/')}/api/verify", data=data,
                                      files={"image": (path.name, path.read_bytes(), "image/png")})
                    out.append(resp.json())
                    break
                except (httpx.HTTPError, ValueError) as e:
                    if attempt == 2:
                        out.append({"error": f"request failed: {e}"})
    return out


def make_reader(name: str, psm=None):
    """tesseract (the default; RapidOCR escalation follows RAPID_ESCALATION), rapid (RapidOCR alone) or
    rapid+tesseract (RapidOCR with Tesseract as its escalation)."""
    from app.readers.tesseract import TesseractReader, parse_psm
    if name.startswith("rapid"):
        from app.readers.rapid import RapidOCRReader
        return RapidOCRReader(escalate_with_tesseract=name == "rapid+tesseract")
    first, extra = parse_psm(psm)
    return TesseractReader(psm=first, extra_psm=extra)


def run_local(rows, image_dir, psm, workers: int = 1, reader_name: str = "tesseract"):
    from PIL import Image
    from app.models import Application
    from app.pipeline import verify
    reader = make_reader(reader_name, psm)

    def one(r):
        app = Application(**{k: r[k] for k in FIELDS})
        return verify(app, Image.open(image_dir / r["image"]), reader).model_dump(mode="json")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(one, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="samples", choices=["samples", "batch"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--psm", default=None, help='e.g. "4", "11" or "4+11" (two merged passes)')
    ap.add_argument("-j", "--workers", type=int, default=1, help="local runs: labels processed in parallel")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit 1 if accuracy is below this fraction or any planted defect came back PASS")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--reader", default="tesseract", choices=["tesseract", "rapid", "rapid+tesseract"],
                    help="local runs: the primary reader (RapidOCR escalation of tesseract follows RAPID_ESCALATION)")
    a = ap.parse_args()
    if not a.url:
        from app.readers.rapid import warm_up as rapid_warm_up
        rapid_warm_up()   # start the RapidOCR workers first, as the server does at start-up
    image_dir, rows = rows_for(a.set)
    results = run_remote(a.url, rows, image_dir, a.psm) if a.url else run_local(rows, image_dir, a.psm, a.workers, a.reader)
    ok, missed, errors, times = 0, 0, 0, []
    for r, res in zip(rows, results):
        if "error" in res:
            errors += 1
            print(f"{r['image']:28} ERROR {res['error']}")
            continue
        times.append(res["timings"]["total_ms"])
        hit = res["overall"] == r["expected_overall"]
        ok += hit
        missed += r["expected_overall"] != "PASS" and res["overall"] == "PASS"
        if not hit or a.verbose:
            flag = "ok " if hit else "BAD"
            fields = ", ".join(f"{f['key']}={f['verdict']}" for f in res["fields"] if f["verdict"] not in ("MATCH", "SKIPPED"))
            w = res["warning"]
            print(f"{flag} {r['image']:28} got {res['overall']:6} want {r['expected_overall']:6} "
                  f"{res['timings']['total_ms']:7.0f} ms | {fields} | warning={w['overall']} "
                  f"(wording={w['wording']} caps={w['heading_caps']} bold={w['heading_bold']} ratio={w['bold_ratio']})")
    n = len(times)
    print(f"\n{ok}/{len(rows)} labels got the expected overall verdict; "
          f"{missed} planted defect(s) reported as PASS; {errors} error(s)")
    if times:
        print(f"timing ms: median {statistics.median(times):.0f}  p95 {sorted(times)[int(0.95 * (n - 1))]:.0f}  max {max(times):.0f}"
              + (f"  (measured {a.workers} at a time)" if a.workers > 1 and not a.url else ""))
    if a.fail_under is not None and (ok < a.fail_under * len(rows) or missed or errors):
        sys.exit(1)


if __name__ == "__main__":
    main()
