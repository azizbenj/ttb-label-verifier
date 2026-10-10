#!/usr/bin/env python
"""Stress the reader beyond the bundled DejaVu labels: other typefaces and degraded images.

Renders the 15 sample specs (scripts/generate_labels.py) under each condition and reports, per
condition, how many labels got the expected verdict, how many planted defects were reported as PASS
(the dangerous error), and how many clean labels were flagged. Uses fonts installed on this machine
at runtime only: nothing is written to the repository.

    python scripts/stress_test.py            # all conditions
    python scripts/stress_test.py --verbose  # also list each miss
"""

from __future__ import annotations

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_labels as gen  # noqa: E402
from app.models import Application  # noqa: E402
from app.pipeline import verify  # noqa: E402
from app.readers.tesseract import TesseractReader  # noqa: E402

SUP = Path("/System/Library/Fonts/Supplemental")
SYS = Path("/System/Library/Fonts")
FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin")


def _face(path: Path, want_bold: bool) -> tuple[str, int] | None:
    """(file, index) of the regular or bold face inside a .ttf/.ttc."""
    if not path.exists():
        return None
    best = None
    for idx in range(12):
        try:
            f = ImageFont.truetype(str(path), 20, index=idx)
        except OSError:
            break
        style = (f.getname()[1] or "").lower()
        is_bold = "bold" in style and "semi" not in style and "italic" not in style and "oblique" not in style
        is_reg = style in ("regular", "roman", "book", "medium", "plain", "normal", "")
        if want_bold and is_bold:
            return str(path), idx
        if not want_bold and is_reg:
            return str(path), idx
        if best is None and "italic" not in style:
            best = (str(path), idx)
    return best


FAMILIES = {   # whole label in another typeface
    "Georgia": (SUP / "Georgia.ttf", SUP / "Georgia Bold.ttf"),
    "Times": (SYS / "Times.ttc", SYS / "Times.ttc"),
    "Baskerville": (SUP / "Baskerville.ttc", SUP / "Baskerville.ttc"),
    "Didot": (SUP / "Didot.ttc", SUP / "Didot.ttc"),
    "Futura": (SUP / "Futura.ttc", SUP / "Futura.ttc"),
    "Helvetica Neue": (SYS / "HelveticaNeue.ttc", SYS / "HelveticaNeue.ttc"),
    "Optima": (SYS / "Optima.ttc", SYS / "Optima.ttc"),
    "Gill Sans": (SUP / "GillSans.ttc", SUP / "GillSans.ttc"),
    "Rockwell": (SUP / "Rockwell.ttc", SUP / "Rockwell.ttc"),
    "Avenir Next Condensed": (SYS / "Avenir Next Condensed.ttc", SYS / "Avenir Next Condensed.ttc"),
}
DISPLAY = {    # brand line only, in a display face, the rest in DejaVu
    "Copperplate brand": SUP / "Copperplate.ttc",
    "Chalkduster brand": SUP / "Chalkduster.ttf",
    "Papyrus brand": SUP / "Papyrus.ttc",
    "Trattatello brand": SUP / "Trattatello.ttf",
    "Herculanum brand": SUP / "Herculanum.ttf",
    "Impact brand": SUP / "Impact.ttf",
}


def jpeg(q):
    def f(img):
        buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=q); return Image.open(io.BytesIO(buf.getvalue()))
    return f


def scale(factor):
    return lambda img: img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)


def rotate(deg):
    return lambda img: img.convert("RGB").rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def noise(amount):
    def f(img):
        import numpy as np
        a = np.asarray(img.convert("L"), dtype=np.int16)
        rng = np.random.default_rng(7)
        a = np.clip(a + rng.normal(0, amount, a.shape), 0, 255).astype("uint8")
        return Image.fromarray(a)
    return f


DEGRADE = {
    "JPEG quality 35": jpeg(35),
    "Half resolution (600 px wide)": scale(0.5),
    "Third resolution (400 px wide)": scale(1 / 3),
    "Rotated 1.5°": rotate(1.5),
    "Rotated 4°": rotate(4),
    "Blur radius 1.2": lambda img: img.filter(ImageFilter.GaussianBlur(1.2)),
    "Sensor noise": noise(28),
}


def render_with(spec, regular=None, bold=None, brand_face=None):
    """Render a spec with substitute fonts by patching the generator's font helpers."""
    orig_font, orig_fit = gen.font, gen.fit_font
    reg_face = _face(Path(regular), False) if regular else None
    bold_face = _face(Path(bold), True) if bold else None
    disp = (str(brand_face), 0) if brand_face else None

    def font(family, is_bold, size):
        face = bold_face if is_bold else reg_face
        if face:
            return ImageFont.truetype(face[0], size, index=face[1])
        return orig_font(family, is_bold, size)

    def fit_font(draw, text, family, is_bold, size, max_w, min_size=36):
        if disp:
            f = ImageFont.truetype(disp[0], size, index=disp[1])
            while draw.textlength(text, font=f) > max_w and size > min_size:
                size -= 4
                f = ImageFont.truetype(disp[0], size, index=disp[1])
            return f
        return orig_fit(draw, text, family, is_bold, size, max_w, min_size)

    gen.font, gen.fit_font = font, fit_font
    try:
        return gen.render(spec)
    finally:
        gen.font, gen.fit_font = orig_font, orig_fit


def run_one(reader, name, spec, img):
    app = Application(**{k: spec.app.get(k, "") for k in FIELDS})
    r = verify(app, img, reader)
    return name, spec, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--reader", default="tesseract", choices=["tesseract", "rapid", "rapid+tesseract"],
                    help="the primary reader (RapidOCR escalation of tesseract follows RAPID_ESCALATION)")
    a = ap.parse_args()
    from app.readers.rapid import warm_up as rapid_warm_up
    rapid_warm_up()   # start the RapidOCR workers first, as the server does at start-up
    specs = gen.sample_specs()
    conditions = [("Baseline (DejaVu)", None, None, None, None)]
    conditions += [(f"Typeface: {n}", str(r), str(b), None, None) for n, (r, b) in FAMILIES.items() if r.exists()]
    conditions += [(f"Display: {n}", None, None, str(p), None) for n, p in DISPLAY.items() if p.exists()]
    conditions += [(f"Image: {n}", None, None, None, fn) for n, fn in DEGRADE.items()]
    if a.reader.startswith("rapid"):
        from app.readers.rapid import RapidOCRReader
        reader = RapidOCRReader(escalate_with_tesseract=a.reader == "rapid+tesseract")
    else:
        reader = TesseractReader()
    print(f"{'condition':38} {'ok':>6} {'missed defects':>15} {'clean flagged':>14}  median")
    for label, reg, bold, disp, degrade in conditions:
        jobs = []
        for name, spec in specs:
            img = render_with(spec, reg, bold, disp)
            if degrade:
                img = degrade(img)
            jobs.append((name, spec, img))
        with ThreadPoolExecutor(a.j) as pool:
            results = list(pool.map(lambda j: run_one(reader, *j), jobs))
        ok = missed = flagged = 0
        times, misses = [], []
        for name, spec, r in results:
            times.append(r.timings.total_ms)
            got, want = r.overall.value, spec.expected_overall
            ok += got == want
            if want != "PASS" and got == "PASS":
                missed += 1
            if want == "PASS" and got != "PASS":
                flagged += 1
            if got != want:
                bad = [f"{f.key}={f.verdict.value}" for f in r.fields if f.verdict.value not in ("MATCH", "SKIPPED")]
                misses.append(f"    {name:26} want {want:6} got {got:6} {', '.join(bad)} warning={r.warning.overall.value}")
        times.sort()
        print(f"{label:38} {ok:>3}/{len(results):<2} {missed:>15} {flagged:>14}  {times[len(times) // 2] / 1000:.2f} s")
        if a.verbose and misses:
            print("\n".join(misses))


if __name__ == "__main__":
    main()
