#!/usr/bin/env python
"""Download label images and filed details for a list of public registry records (TTB IDs).

For local testing only: label artwork belongs to the brand owners, so the files go to data/real/
(gitignored) and are never committed. Uses the system curl (macOS trust store) with a cookie jar,
because the image endpoint serves the attachments of the record last opened in the session.

    python scripts/fetch_registry_labels.py 26203001000533 26189001000502 ...
    python scripts/fetch_registry_labels.py --search 09/15/2026 09/29/2026 --sample 180 [--seed 1] [--dry-run]

--search lists the records completed in a date range (the registry's own "save results to file" export,
at most 1000 per search), keeps one record per permit holder, picks a sample balanced over wine / beer /
spirits and domestic / imported, and fetches those. The filed values (brand, class/type, origin) are kept
in data/real/raw/registry.csv next to records.json: they are the application side of the ground truth.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real" / "raw"
BASE = "https://ttbonline.gov/colasonline"
CURL = "/usr/bin/curl"
UA = "Mozilla/5.0 (Macintosh) LabelCheck-test"
JAR = "/tmp/ttb_registry_jar.txt"


def get(url: str, out: Path | None = None) -> bytes:
    cmd = [CURL, "-sS", "--max-time", "60", "-c", JAR, "-b", JAR, "-A", UA, url]
    if out:
        cmd += ["-o", str(out)]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def field(text: str, label: str) -> str:
    m = re.search(re.escape(label) + r"\s*(?:\([^)]*\))?\s*\n+(.+?)\n", text)
    return m.group(1).strip() if m else ""


def text_of(page: str) -> str:
    page = re.sub(r"(?is)<(script|style).*?</\1>", "", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|td|th|li|h\d)>", "\n", page)
    page = re.sub(r"<[^>]+>", " ", page)
    lines = [re.sub(r"[ \t\xa0]+", " ", html.unescape(l)).strip() for l in page.splitlines()]
    return "\n".join(l for l in lines if l)


def fetch(ttbid: str) -> dict:
    detail = text_of(get(f"{BASE}/viewColaDetails.do?action=publicDisplaySearchBasic&ttbid={ttbid}").decode("utf-8", "replace"))
    form_html = get(f"{BASE}/viewColaDetails.do?action=publicFormDisplay&ttbid={ttbid}").decode("utf-8", "replace")
    form = text_of(form_html)
    rec = {
        "ttbid": ttbid,
        "class_type": (re.search(r"Class/Type Code:\s*(.+)", detail) or [None, ""])[1].strip(),
        "origin": (re.search(r"Origin Code:\s*(.+)", detail) or [None, ""])[1].strip(),
        "brand_name": (re.search(r"Brand Name:\s*(.+)", detail) or [None, ""])[1].strip(),
        "fanciful_name": (re.search(r"Fanciful Name:\s*(.+)", detail) or [None, ""])[1].strip(),
        "imported": bool(re.search(r"Imported", form) and re.search(r"x\s*Imported|checked[^>]*Imported", form_html, re.I)),
        "applicant": "",
        "images": [],
    }
    m = re.search(r"8\. NAME AND ADDRESS OF APPLICANT.*?\(Required\)\s*\n(.*?)\n4\. SERIAL NUMBER", form, re.S)
    if m:
        rec["applicant"] = " / ".join(l for l in m.group(1).splitlines() if l.strip())
    imgs = re.findall(r'<img[^>]+src="(/colasonline/publicViewAttachment\.do\?[^"]+)"[^>]*alt="([^"]*)"', form_html)
    imgs += [(s, a) for a, s in re.findall(r'<img[^>]+alt="([^"]*)"[^>]*src="(/colasonline/publicViewAttachment\.do\?[^"]+)"', form_html)]
    seen = set()
    for i, (src, alt) in enumerate(imgs, start=1):
        if src in seen:
            continue
        seen.add(src)
        src = html.unescape(src)
        name = re.search(r"filename=([^&]+)", src).group(1)
        url = BASE.rsplit("/colasonline", 1)[0] + src.split("filename=")[0] + "filename=" + quote(name) + src[src.index("&", src.index("filename=")):]
        ext = Path(name).suffix.lower() or ".jpg"
        out = OUT / f"{ttbid}_{len(rec['images']) + 1}{ext}"
        get(url, out)
        rec["images"].append({"file": out.name, "alt": alt, "bytes": out.stat().st_size})
        time.sleep(0.5)  # be gentle with a public service
    return rec


US_PLACES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA",
    "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE",
    "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS",
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
    "PUERTO RICO", "U.S. VIRGIN ISLANDS", "VIRGIN ISLANDS", "GUAM", "AMERICAN", "UNITED STATES",
}
_KINDS = (   # spirits first: a single malt Scotch is not a malt beverage
    ("spirits", re.compile(r"\b(WHISK|BOURBON|VODKA|GIN|RUM|TEQUILA|MEZCAL|BRANDY|COGNAC|LIQUEUR|CORDIAL|SPIRIT|"
                           r"DISTILLED|ABSINTHE|SCHNAPPS|GRAPPA|AQUAVIT|SOJU|BAIJIU|NEUTRAL)", re.I)),
    ("wine", re.compile(r"\b(WINE|CHAMPAGNE|SPARKLING|VERMOUTH|PORT|SHERRY|GRAPE)\b", re.I)),
    ("beer", re.compile(r"\b(BEER|ALE|LAGER|MALT|STOUT|PORTER|PILS|IPA|CIDER|MEAD|SAKE)\b", re.I)),
)


def kind_of(class_desc: str) -> str:
    for kind, rx in _KINDS:
        if rx.search(class_desc or ""):
            return kind
    return "other"


def search(date_from: str, date_to: str) -> list[dict]:
    """The registry's results export for a completed-date range: one dict per record."""
    import csv
    import io
    form = [
        ("searchCriteria.dateCompletedFrom", date_from), ("searchCriteria.dateCompletedTo", date_to),
        ("searchCriteria.productOrFancifulName", ""), ("searchCriteria.productNameSearchType", "E"),
        ("searchCriteria.classTypeFrom", ""), ("searchCriteria.classTypeTo", ""), ("searchCriteria.originCode", ""),
    ]
    cmd = [CURL, "-sS", "--max-time", "90", "-c", JAR, "-b", JAR, "-A", UA, "-X", "POST",
           f"{BASE}/publicSearchColasBasicProcess.do?action=search", "-o", "/dev/null"]
    for k, v in form:
        cmd += ["--data-urlencode", f"{k}={v}"]
    subprocess.run(cmd, check=True, capture_output=True)
    time.sleep(1)
    data = get(f"{BASE}/publicSaveSearchResultsToFile.do?path=/publicSearchColasBasicProcess").decode("utf-8", "replace")
    rows = []
    for r in csv.DictReader(io.StringIO(data)):
        ttbid = (r.get("TTB ID") or "").strip().strip("'")
        if not ttbid.isdigit():
            continue
        origin_desc = (r.get("Origin Desc") or "").strip()
        rows.append({
            "ttbid": ttbid, "permit": (r.get("Permit No.") or "").strip(), "date": (r.get("Completed Date") or "").strip(),
            "fanciful_name": (r.get("Fanciful Name") or "").strip(), "brand_name": (r.get("Brand Name") or "").strip(),
            "origin_code": (r.get("Origin") or "").strip(), "origin_desc": origin_desc,
            "class_code": (r.get("Class/Type") or "").strip(), "class_desc": (r.get("Class/Type Desc") or "").strip(),
        })
    for r in rows:
        r["kind"] = kind_of(r["class_desc"])
        r["imported"] = r["origin_desc"].upper() not in US_PLACES and r["origin_desc"] != ""
    return rows


def pick_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """One record per permit holder, then as even a spread as possible over kind x domestic/imported."""
    import random
    rnd = random.Random(seed)
    rows = [r for r in rows if r["kind"] != "other" and r["brand_name"]]
    rnd.shuffle(rows)
    seen_permit, unique = set(), []
    for r in rows:
        if r["permit"] in seen_permit:
            continue
        seen_permit.add(r["permit"])
        unique.append(r)
    strata: dict[tuple[str, bool], list[dict]] = {}
    for r in unique:
        strata.setdefault((r["kind"], r["imported"]), []).append(r)
    picked: list[dict] = []
    keys = sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def main(ids: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "records.json"
    recs = json.loads(index.read_text()) if index.exists() else {}
    for t in ids:
        if t in recs and all((OUT / im["file"]).exists() for im in recs[t]["images"]):
            print(f"{t}: cached")
            continue
        try:
            recs[t] = fetch(t)
            r = recs[t]
            print(f"{t}: {r['brand_name']} / {r['fanciful_name']} · {r['class_type']} · {r['origin']} · "
                  f"{len(r['images'])} image(s), {sum(i['bytes'] for i in r['images']) // 1024} KB")
        except subprocess.CalledProcessError as e:
            print(f"{t}: failed ({e.stderr.decode()[:120]})")
        index.write_text(json.dumps(recs, indent=2))
        time.sleep(1)


def main_search(date_from: str, date_to: str, n: int, seed: int, dry_run: bool) -> None:
    import csv
    rows = search(date_from, date_to)
    picked = pick_sample(rows, n, seed)
    from collections import Counter
    c = Counter((r["kind"], "import" if r["imported"] else "domestic") for r in picked)
    print(f"{len(rows)} records listed, {len(picked)} picked: " + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(c.items())))
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registry.csv"
    existing = {}
    if reg.exists():
        existing = {r["ttbid"]: r for r in csv.DictReader(reg.open())}
    for r in picked:
        existing[r["ttbid"]] = {k: ("yes" if v is True else "no" if v is False else v) for k, v in r.items()}
    with reg.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ttbid", "permit", "date", "fanciful_name", "brand_name", "origin_code",
                                          "origin_desc", "class_code", "class_desc", "kind", "imported"])
        w.writeheader()
        for r in existing.values():
            w.writerow(r)
    if dry_run:
        for r in picked[:12]:
            print(f"  {r['ttbid']} {r['kind']:8} {'import' if r['imported'] else 'domestic':9} {r['brand_name'][:28]:28} {r['class_desc'][:30]} ({r['origin_desc']})")
        return
    main([r["ttbid"] for r in picked])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--search", nargs=2, metavar=("FROM", "TO"), help="completed-date range, MM/DD/YYYY")
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="list the sample, download nothing")
    a = ap.parse_args()
    if a.search:
        main_search(a.search[0], a.search[1], a.sample, a.seed, a.dry_run)
    else:
        main(a.ids)
