"""Evidence shown next to each verdict: number conversions, normalized forms, and the crop of the label
where the value was read. Pure presentation helpers used by the result template."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import THRESHOLDS
from .models import FieldResult, VerificationResult, Verdict
from .normalize import normalize_loose, parse_alcohol, parse_net_contents

CROP_WIDTH = 300          # px, the crop box in an evidence strip
CROP_MIN_H, CROP_MAX_H = 36, 150


@dataclass
class Crop:
    box_style: str   # style for the clipping container
    img_style: str   # style for the positioned image


@dataclass
class Evidence:
    rows: list[tuple[str, str, str]] = field(default_factory=list)   # (label, value, css class)
    text: str = ""
    crop: Crop | None = None


def crop_for(box: list[float] | None, aspect: float | None, width: int = CROP_WIDTH) -> Crop | None:
    """Position the full label image inside a small window so that ``box`` (percentages) fills it."""
    if not box or not aspect:
        return None
    left, top, w, h = box
    if w <= 0 or h <= 0:
        return None
    pad_x = max(w * 0.08, 1.5)                      # a little context around the words
    pad_y = max(h * 0.35, 0.8)
    vis_w = min(100.0, w + 2 * pad_x)
    img_w = width * 100.0 / vis_w                   # the image is scaled so the region spans the window
    img_h = img_w * aspect
    x0 = max(0.0, left - pad_x)
    y0 = max(0.0, top - pad_y)
    vis_h = (h + 2 * pad_y) / 100.0 * img_h
    box_h = int(min(CROP_MAX_H, max(CROP_MIN_H, vis_h)))
    return Crop(box_style=f"width:{width}px;height:{box_h}px",
                img_style=f"width:{img_w:.0f}px;left:{-x0 / 100.0 * img_w:.0f}px;top:{-y0 / 100.0 * img_h:.0f}px")


def _fmt_abv(text: str) -> str:
    v = parse_alcohol(text)
    if v is None or v.abv is None:
        return text
    proof = v.proof if v.proof is not None else v.abv * 2
    return f"{v.abv:g}% = {proof:g} proof"


def _fmt_vol(text: str) -> str:
    v = parse_net_contents(text)
    if v is None:
        return text
    return f"{v.ml:g} mL" if v.unit == "mL" else f"{v.ml:g} mL ({v.text})"


def evidence_for(f: FieldResult, result: VerificationResult) -> Evidence:
    ev = Evidence(crop=crop_for(f.box, result.image_aspect))
    bad = "bad" if f.verdict in (Verdict.MISMATCH, Verdict.NOT_FOUND) else ""
    conf = f"{result.read_confidence:.0f}%" if result.read_confidence is not None else "n/a"
    if f.key == "alcohol_content":
        ev.rows = [("application", _fmt_abv(f.expected), ""), ("label", _fmt_abv(f.found) if f.found else "not found", bad)]
        a, b = parse_alcohol(f.expected), parse_alcohol(f.found or "")
        if a and b and a.abv is not None and b.abv is not None:
            ev.rows.append(("difference", f"{abs(a.abv - b.abv):.1f} points · allowed 0.05", ""))
        ev.rows.append(("read confidence", conf, ""))
        if f.verdict == Verdict.MISMATCH and b and b.proof is not None and not b.abv_from_proof \
                and abs(b.proof - 2 * b.abv) <= THRESHOLDS.proof_tolerance:
            ev.text = ("Both numbers were read from the same line; proof and percent agree with each other, "
                       "so this is not a misread. The label does not match the application.")
    elif f.key == "net_contents":
        ev.rows = [("application", _fmt_vol(f.expected), ""), ("label", _fmt_vol(f.found) if f.found else "not found", bad)]
        a, b = parse_net_contents(f.expected), parse_net_contents(f.found or "")
        if a and b:
            ev.rows.append(("difference", f"{abs(a.ml - b.ml):g} mL · allowed 0.5", ""))
        ev.rows.append(("read confidence", conf, ""))
    else:
        ev.rows = [("application", f.expected or "—", ""), ("label", f.found or "not found", bad)]
        if f.found:
            same = normalize_loose(f.expected) == normalize_loose(f.found)
            ev.rows.append(("after normalizing", "identical · 100 / 100" if same else f"{f.score} / 100", ""))
        ev.rows.append(("read confidence", conf, ""))
    return ev


def pin_labels(result: VerificationResult) -> dict[str, str]:
    """Pin number per field key: 1, 2, 3... for fields that were read; SKIPPED and NOT FOUND get none."""
    out, n = {}, 0
    for f in result.fields:
        if f.verdict in (Verdict.SKIPPED, Verdict.NOT_FOUND) or not f.box:
            out[f.key] = ""
        else:
            n += 1
            out[f.key] = str(n)
    return out


def bold_meter_percent(ratio: float | None, threshold: float, lo: float = 0.8, hi: float = 2.5) -> int | None:
    if ratio is None:
        return None
    return int(round(100 * (min(hi, max(lo, ratio)) - lo) / (hi - lo)))
