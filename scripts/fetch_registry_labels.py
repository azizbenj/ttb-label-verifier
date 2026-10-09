#!/usr/bin/env python
"""Download label images and filed details for a list of public registry records (TTB IDs).

For local testing only: label artwork belongs to the brand owners, so the files go to data/real/
(gitignored) and are never committed. Uses the system curl (macOS trust store) with a cookie jar,
because the image endpoint serves the attachments of the record last opened in the session.

    python scripts/fetch_registry_labels.py 26203001000533 26189001000502 ...
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


if __name__ == "__main__":
    main(sys.argv[1:])
