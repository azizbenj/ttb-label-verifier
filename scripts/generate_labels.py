#!/usr/bin/env python
"""Generate synthetic alcohol label images with Pillow, plus the matching application CSVs.

    python scripts/generate_labels.py samples              # 15 labels, one per failure type -> data/samples/
    python scripts/generate_labels.py batch --count 250    # batch demo set -> data/batch/

Every image gets a CSV row holding the *application* values (what the applicant filed) and the
outcome the verifier is expected to produce, so the same files drive the tests and the UI demo.
Labels are rendered flat and clean on purpose: photos, angles and glare are out of scope.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import MANDATED_WARNING  # noqa: E402

FONT_DIR = ROOT / "data" / "fonts"
FAMILIES = {"sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
            "serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf")}
W, H = 1200, 1600
PALETTES = [((248, 243, 230), (32, 26, 20)), ((255, 255, 255), (20, 20, 20)),
            ((240, 236, 222), (60, 30, 20)), ((250, 250, 245), (25, 35, 60))]
CSV_COLUMNS = ["image", "application_id", "brand_name", "class_type", "alcohol_content", "net_contents",
               "bottler_name_address", "country_of_origin", "variant", "expected_overall", "expected_issue"]


@dataclass
class Spec:
    """What gets printed on the label (plus how), and what the application says."""

    brand: str
    class_type: str
    alcohol_text: str
    volume_text: str | None
    bottler: str
    origin: str | None = None
    warning_text: str | None = MANDATED_WARNING
    heading_caps: bool = True
    heading_bold: bool = True
    template: str = "classic"
    family: str = "serif"
    palette: int = 0
    blur: float = 0.0
    app: dict = field(default_factory=dict)
    variant: str = "clean"
    expected_overall: str = "PASS"
    expected_issue: str = ""


# --- drawing helpers ----------------------------------------------------------------------------
def font(family: str, bold: bool, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FAMILIES[family][1 if bold else 0]), size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def fit_font(draw, text, family, bold, size, max_w, min_size=36):
    fnt = font(family, bold, size)
    while draw.textlength(text, font=fnt) > max_w and size > min_size:
        size -= 4
        fnt = font(family, bold, size)
    return fnt


def draw_lines(draw, lines, fnt, y, fill, align="center", x=0, max_w=W, spacing=1.25) -> int:
    lh = int(fnt.size * spacing)
    for line in lines:
        tw = draw.textlength(line, font=fnt)
        lx = x if align == "left" else (W - tw) / 2 if align == "center" else x + max_w - tw
        draw.text((lx, y), line, font=fnt, fill=fill)
        y += lh
    return y


def draw_warning(draw, spec: Spec, x: int, y: int, max_w: int, size: int, fill, frame: bool,
                 boxes: list | None = None) -> int:
    """Flow the statement word by word; heading words in bold (or not), body words regular.

    When ``boxes`` is given, (text, (l, t, r, b), is_heading) is appended for every word drawn.
    """
    if not spec.warning_text:
        return y
    reg, bold = font(spec.family, False, size), font(spec.family, True, size)
    words = spec.warning_text.split()
    heading = words[:2]  # "GOVERNMENT" "WARNING:"
    if not spec.heading_caps:
        heading = [w.capitalize() for w in heading]
    tokens = [(w, bold if spec.heading_bold else reg) for w in heading] + [(w, reg) for w in words[2:]]
    pad = 28 if frame else 0
    lh = int(size * 1.4)
    cx, cy = x + pad, y + pad
    top = y
    for i, (text, fnt) in enumerate(tokens):
        tw = draw.textlength(text + " ", font=fnt)
        if cx + tw > x + max_w - pad and cx > x + pad:
            cx, cy = x + pad, cy + lh
        draw.text((cx, cy), text, font=fnt, fill=fill)
        if boxes is not None:
            boxes.append((text, draw.textbbox((cx, cy), text, font=fnt), i < 2))
        cx += tw
    bottom = cy + lh + pad
    if frame:
        draw.rectangle([x, top, x + max_w, bottom], outline=fill, width=3)
    return bottom


# --- templates -----------------------------------------------------------------------------------
def render(spec: Spec, boxes: list | None = None) -> Image.Image:
    bg, ink = PALETTES[spec.palette % len(PALETTES)]
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fam = spec.family
    if spec.template == "classic":
        d.rectangle([40, 40, W - 40, H - 40], outline=ink, width=4)
        d.rectangle([56, 56, W - 56, H - 56], outline=ink, width=2)
        y = 150
        bf = fit_font(d, spec.brand, fam, True, 96, W - 200)
        y = draw_lines(d, wrap(d, spec.brand, bf, W - 200), bf, y, ink) + 20
        d.line([250, y, W - 250, y], fill=ink, width=3)
        y += 40
        cf = font(fam, False, 52)
        y = draw_lines(d, wrap(d, spec.class_type, cf, W - 240), cf, y, ink) + 30
        d.line([250, y, W - 250, y], fill=ink, width=3)
        y += 50
        af = font(fam, False, 44)
        y = draw_lines(d, [spec.alcohol_text], af, y, ink) + 10
        if spec.volume_text:
            y = draw_lines(d, [spec.volume_text], af, y, ink) + 10
        if spec.origin:
            y = draw_lines(d, [spec.origin], font(fam, False, 38), y + 10, ink) + 10
        bfnt = font(fam, False, 30)
        y = draw_lines(d, wrap(d, spec.bottler, bfnt, W - 260), bfnt, y + 30, ink)
        draw_warning(d, spec, 110, 1150, W - 220, 28, ink, frame=True, boxes=boxes)
    elif spec.template == "modern":
        x = 90
        y = 140
        bf = fit_font(d, spec.brand, fam, True, 108, W - 180)
        y = draw_lines(d, wrap(d, spec.brand, bf, W - 180), bf, y, ink, align="left", x=x) + 16
        d.rectangle([x, y, x + 320, y + 14], fill=ink)
        y += 60
        cf = font(fam, False, 50)
        y = draw_lines(d, wrap(d, spec.class_type, cf, W - 180), cf, y, ink, align="left", x=x) + 60
        af = font(fam, False, 42)
        info = [spec.alcohol_text] + ([spec.volume_text] if spec.volume_text else []) + ([spec.origin] if spec.origin else [])
        y = draw_lines(d, info, af, y, ink, align="left", x=x, spacing=1.5) + 40
        bfnt = font(fam, False, 30)
        y = draw_lines(d, wrap(d, spec.bottler, bfnt, W - 180), bfnt, y, ink, align="left", x=x)
        d.line([x, 1130, W - x, 1130], fill=ink, width=2)
        draw_warning(d, spec, x, 1160, W - 2 * x, 28, ink, frame=False, boxes=boxes)
    else:  # compact
        d.rectangle([30, 30, W - 30, H - 30], outline=ink, width=6)
        y = 120
        bf = fit_font(d, spec.brand, fam, True, 88, W - 160)
        y = draw_lines(d, wrap(d, spec.brand, bf, W - 160), bf, y, ink) + 10
        cf = font(fam, False, 46)
        y = draw_lines(d, wrap(d, spec.class_type, cf, W - 200), cf, y, ink) + 50
        d.line([120, y, W - 120, y], fill=ink, width=2)
        y += 50
        af = font(fam, False, 40)
        d.text((140, y), spec.alcohol_text, font=af, fill=ink)
        if spec.volume_text:
            tw = d.textlength(spec.volume_text, font=af)
            d.text((W - 140 - tw, y), spec.volume_text, font=af, fill=ink)
        y += 90
        if spec.origin:
            y = draw_lines(d, [spec.origin], font(fam, False, 36), y, ink) + 20
        bfnt = font(fam, False, 30)
        y = draw_lines(d, wrap(d, spec.bottler, bfnt, W - 260), bfnt, y + 20, ink)
        draw_warning(d, spec, 100, 1170, W - 200, 28, ink, frame=True, boxes=boxes)
    if spec.blur:
        img = img.filter(ImageFilter.GaussianBlur(spec.blur))
    return img


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("P", palette=Image.ADAPTIVE, colors=32).save(path, optimize=True)


# --- catalog ---------------------------------------------------------------------------------------
def base_specs() -> dict[str, Spec]:
    """Four realistic base labels spanning spirits, wine, beer and an import."""
    old_tom = Spec(
        brand="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_text="45% Alc./Vol. (90 Proof)", volume_text="750 mL",
        bottler="Distilled and Bottled by Old Tom Distillery, Bardstown, Kentucky 40004",
        template="classic", family="serif", palette=0,
        app=dict(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                 alcohol_content="45% Alc./Vol. (90 Proof)", net_contents="750 mL",
                 bottler_name_address="Old Tom Distillery, Bardstown, Kentucky 40004", country_of_origin=""))
    stones = Spec(
        brand="STONE'S THROW CELLARS", class_type="Napa Valley Cabernet Sauvignon",
        alcohol_text="ALC. 14.5% BY VOL.", volume_text="750 mL",
        bottler="Produced and Bottled by Stone's Throw Cellars, St. Helena, CA 94574",
        template="modern", family="sans", palette=1,
        app=dict(brand_name="STONE'S THROW CELLARS", class_type="Napa Valley Cabernet Sauvignon",
                 alcohol_content="14.5% Alc./Vol.", net_contents="750 mL",
                 bottler_name_address="Stone's Throw Cellars, St. Helena, CA 94574", country_of_origin=""))
    river = Spec(
        brand="RIVER BEND BREWING CO.", class_type="India Pale Ale",
        alcohol_text="6.8% ALC./VOL.", volume_text="12 FL. OZ.",
        bottler="Brewed and Bottled by River Bend Brewing Co., Portland, OR 97209",
        template="compact", family="sans", palette=2,
        app=dict(brand_name="RIVER BEND BREWING CO.", class_type="India Pale Ale",
                 alcohol_content="6.8% ABV", net_contents="12 fl oz",
                 bottler_name_address="River Bend Brewing Co., Portland, OR 97209", country_of_origin=""))
    glen = Spec(
        brand="GLEN MORAR", class_type="Single Malt Scotch Whisky",
        alcohol_text="43% Alc./Vol. (86 Proof)", volume_text="700 mL", origin="Product of Scotland",
        bottler="Imported by Highland Imports, Inc., New York, NY 10001",
        template="classic", family="sans", palette=3,
        app=dict(brand_name="GLEN MORAR", class_type="Single Malt Scotch Whisky",
                 alcohol_content="86 Proof", net_contents="700 mL",
                 bottler_name_address="Highland Imports, Inc., New York, NY 10001", country_of_origin="Scotland"))
    return {"old_tom": old_tom, "stones": stones, "river": river, "glen": glen}


def sample_specs() -> list[tuple[str, Spec]]:
    b = base_specs()
    from copy import deepcopy as dc
    out: list[tuple[str, Spec]] = []

    def add(name, base, **changes):
        s = dc(b[base])
        for k, v in changes.items():
            setattr(s, k, v)
        out.append((name, s))

    add("old_tom_clean", "old_tom")
    add("stones_throw_clean", "stones")
    add("river_bend_clean", "river")
    add("glen_morar_import_clean", "glen")
    add("wrong_abv", "old_tom", alcohol_text="40% Alc./Vol. (80 Proof)", variant="wrong_abv",
        expected_overall="FAIL", expected_issue="alcohol_content")
    add("missing_warning", "river", warning_text=None, variant="missing_warning",
        expected_overall="FAIL", expected_issue="warning")
    add("warning_not_caps", "stones", heading_caps=False, variant="warning_not_caps",
        expected_overall="FAIL", expected_issue="warning")
    add("warning_not_bold", "old_tom", heading_bold=False, variant="warning_not_bold",
        expected_overall="REVIEW", expected_issue="warning")
    add("brand_case", "stones", brand="Stone's Throw Cellars", variant="brand_case",
        expected_overall="REVIEW", expected_issue="brand_name")
    add("wrong_net_contents", "old_tom", volume_text="1 L", variant="wrong_net_contents",
        expected_overall="FAIL", expected_issue="net_contents")
    add("wrong_brand", "river", app={**b["river"].app, "brand_name": "COPPER KETTLE BREWING CO."},
        variant="wrong_brand", expected_overall="FAIL", expected_issue="brand_name")
    # "can cause" for "may cause" changes what the warning says: a FAIL, not a look (README, wording check).
    add("warning_text_altered", "glen", warning_text=MANDATED_WARNING.replace("may cause", "can cause"),
        variant="warning_text_altered", expected_overall="FAIL", expected_issue="warning")
    add("warning_truncated", "stones", warning_text=MANDATED_WARNING.split(" (2)")[0],
        variant="warning_truncated", expected_overall="FAIL", expected_issue="warning")
    add("missing_net_contents", "river", volume_text=None, variant="missing_net_contents",
        expected_overall="FAIL", expected_issue="net_contents")
    add("wrong_country", "glen", origin="Product of Ireland", variant="wrong_country",
        expected_overall="FAIL", expected_issue="country_of_origin")
    return out


# --- batch generation ---------------------------------------------------------------------------
ADJ = ["Old", "Copper", "Silver", "Iron", "Golden", "Wild", "Blue", "Red", "Stone", "High", "Black", "White",
       "Hollow", "Crooked", "Northern", "Prairie", "Harbor", "Cedar", "Maple", "Granite"]
NOUN = ["Tom", "Kettle", "Creek", "Ridge", "Fox", "Oak", "River", "Mountain", "Barrel", "Anchor", "Crow",
        "Falcon", "Lantern", "Compass", "Valley", "Meadow", "Bridge", "Mill", "Harvest", "Timber"]
CITIES = [("Bardstown", "KY", "40004"), ("Louisville", "KY", "40202"), ("Portland", "OR", "97209"),
          ("St. Helena", "CA", "94574"), ("Austin", "TX", "78701"), ("Denver", "CO", "80202"),
          ("Asheville", "NC", "28801"), ("Brooklyn", "NY", "11201"), ("Milwaukee", "WI", "53202"),
          ("Sonoma", "CA", "95476"), ("Nashville", "TN", "37203"), ("Burlington", "VT", "05401")]
SPIRITS = [("Kentucky Straight Bourbon Whiskey", "Distillery"), ("Straight Rye Whiskey", "Distillery"),
           ("Vodka", "Distilling Co."), ("London Dry Gin", "Distillers"), ("Silver Rum", "Rum Co."),
           ("American Single Malt Whiskey", "Distillery")]
WINES = [("Cabernet Sauvignon", "Cellars"), ("Chardonnay", "Vineyards"), ("Pinot Noir", "Winery"),
         ("Red Table Wine", "Cellars"), ("Rosé Wine", "Vineyards")]
BEERS = [("India Pale Ale", "Brewing Co."), ("Lager", "Brewery"), ("Oatmeal Stout", "Brewing Co."),
         ("Pilsner", "Brewery"), ("Hefeweizen", "Brewing")]
IMPORTS = [("Single Malt Scotch Whisky", "Scotland", "Distillers"), ("Tequila Blanco", "Mexico", "Tequila"),
           ("Irish Whiskey", "Ireland", "Distillery"), ("Cognac", "France", "Maison"),
           ("Prosecco", "Italy", "Vini"), ("Rioja Red Wine", "Spain", "Bodegas")]
IMPORTERS = ["Highland Imports, Inc., New York, NY 10001", "Atlantic Wine & Spirits, Miami, FL 33130",
             "Pacific Crest Importers, Seattle, WA 98101", "Great Lakes Beverage Imports, Chicago, IL 60607"]
BATCH_VARIANTS = (["clean"] * 78 + ["wrong_abv"] * 4 + ["missing_warning"] * 3 + ["warning_not_caps"] * 3 +
                  ["warning_not_bold"] * 2 + ["brand_case"] * 3 + ["wrong_net_contents"] * 3 +
                  ["wrong_brand"] * 1 + ["warning_text_altered"] * 2 + ["missing_net_contents"] * 1)


def random_spec(rng: random.Random) -> Spec:
    category = rng.choices(["spirits", "wine", "beer", "import"], weights=[40, 25, 20, 15])[0]
    adj, noun = rng.choice(ADJ), rng.choice(NOUN)
    city = rng.choice(CITIES)
    origin, country = None, ""
    if category == "spirits":
        class_type, suffix = rng.choice(SPIRITS)
        abv = rng.choice([40, 40, 43, 45, 46, 47.5, 50])
        alcohol_text = rng.choice([f"{abv:g}% Alc./Vol. ({abv * 2:g} Proof)", f"{abv:g}% ALC./VOL. ({abv * 2:g} PROOF)",
                                   f"ALC. {abv:g}% BY VOL. ({abv * 2:g} PROOF)"])
        app_alc = rng.choice([f"{abv:g}% Alc./Vol.", f"{abv * 2:g} Proof", f"{abv:g}%"])
        vol = rng.choice(["750 mL", "750 mL", "1 L", "1.75 L", "375 mL", "50 mL"])
        verb = rng.choice(["Distilled and Bottled by", "Bottled by", "Produced and Bottled by"])
    elif category == "wine":
        class_type, suffix = rng.choice(WINES)
        abv = rng.choice([11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15])
        alcohol_text = rng.choice([f"ALC. {abv:g}% BY VOL.", f"{abv:g}% Alc./Vol.", f"Alcohol {abv:g}% by Volume"])
        app_alc = rng.choice([f"{abv:g}% Alc./Vol.", f"{abv:g}% ABV", f"{abv:g}"])
        vol = rng.choice(["750 mL", "750 mL", "1.5 L", "375 mL"])
        verb = rng.choice(["Produced and Bottled by", "Vinted and Bottled by", "Cellared and Bottled by"])
    elif category == "beer":
        class_type, suffix = rng.choice(BEERS)
        abv = rng.choice([4.2, 4.5, 5, 5.2, 5.5, 6, 6.5, 6.8, 7.2, 8, 9])
        alcohol_text = rng.choice([f"{abv:g}% ALC./VOL.", f"{abv:g}% Alc. by Vol.", f"ALC. {abv:g}% BY VOL."])
        app_alc = rng.choice([f"{abv:g}% ABV", f"{abv:g}% Alc./Vol."])
        vol = rng.choice(["12 FL. OZ.", "12 FL OZ (355 mL)", "16 FL. OZ.", "750 mL", "22 FL. OZ."])
        verb = rng.choice(["Brewed and Bottled by", "Brewed and Canned by", "Brewed by"])
    else:
        class_type, country, suffix = rng.choice(IMPORTS)
        abv = rng.choice([40, 40, 43, 46]) if "Wine" not in class_type and class_type != "Prosecco" else rng.choice([11, 12.5, 13.5])
        alcohol_text = f"{abv:g}% Alc./Vol." + (f" ({abv * 2:g} Proof)" if abv >= 20 else "")
        app_alc = f"{abv:g}% Alc./Vol."
        vol = rng.choice(["750 mL", "700 mL", "1 L"])
        origin = f"Product of {country}"
        verb = "Imported by"
    brand = f"{adj} {noun} {suffix}".upper() if rng.random() < 0.7 else f"{adj} {noun} {suffix}"
    if category == "import":
        bottler_core = rng.choice(IMPORTERS)
    else:
        bottler_core = f"{brand.title() if brand.isupper() else brand}, {city[0]}, {city[1]} {city[2]}"
    app_vol = vol
    if vol == "12 FL OZ (355 mL)":
        app_vol = rng.choice(["355 mL", "12 fl oz"])
    elif vol.endswith("FL. OZ."):
        app_vol = rng.choice([vol, vol.replace("FL. OZ.", "fl oz")])
    spec = Spec(brand=brand, class_type=class_type, alcohol_text=alcohol_text, volume_text=vol,
                bottler=f"{verb} {bottler_core}", origin=origin,
                template=rng.choice(["classic", "modern", "compact"]), family=rng.choice(["serif", "sans"]),
                palette=rng.randrange(len(PALETTES)), blur=rng.choice([0, 0, 0, 0, 0.5]),
                app=dict(brand_name=brand, class_type=class_type, alcohol_content=app_alc, net_contents=app_vol,
                         bottler_name_address=bottler_core if rng.random() < 0.85 else "", country_of_origin=country))
    return spec


def apply_variant(spec: Spec, variant: str, rng: random.Random) -> Spec:
    spec.variant = variant
    if variant == "clean":
        return spec
    spec.expected_overall = "FAIL"
    if variant == "wrong_abv":
        spec.expected_issue = "alcohol_content"
        import re
        m = re.search(r"(\d+(?:\.\d+)?)%", spec.alcohol_text)
        abv = float(m.group(1))
        new = abv + rng.choice([-5, 5, 2.5, -1]) if abv > 20 else abv + rng.choice([1, -1, 0.5])
        spec.alcohol_text = spec.alcohol_text.replace(f"{abv:g}%", f"{new:g}%").replace(f"{abv * 2:g} P", f"{new * 2:g} P")
    elif variant == "missing_warning":
        spec.expected_issue, spec.warning_text = "warning", None
    elif variant == "warning_not_caps":
        spec.expected_issue, spec.heading_caps = "warning", False
    elif variant == "warning_not_bold":
        spec.expected_issue, spec.heading_bold, spec.expected_overall = "warning", False, "REVIEW"
    elif variant == "brand_case":
        spec.expected_issue, spec.expected_overall = "brand_name", "REVIEW"
        spec.brand = spec.brand.title() if spec.brand.isupper() else spec.brand.upper()
    elif variant == "wrong_net_contents":
        spec.expected_issue = "net_contents"
        spec.volume_text = {"750 mL": "1 L", "1 L": "750 mL", "1.75 L": "1 L", "375 mL": "750 mL", "50 mL": "100 mL",
                            "1.5 L": "750 mL", "700 mL": "750 mL"}.get(spec.volume_text, "750 mL")
    elif variant == "wrong_brand":
        spec.expected_issue = "brand_name"
        spec.app["brand_name"] = f"{rng.choice(ADJ)} {rng.choice(NOUN)} {spec.brand.split()[-1]}".upper()
    elif variant == "warning_text_altered":
        old = rng.choice(["may cause", "should not", "impairs"])
        # A modal replaced changes the warning's meaning (FAIL); "impair" for "impairs" breaks the fixed text
        # without changing what it says (a look).
        spec.expected_issue, spec.expected_overall = "warning", "REVIEW" if old == "impairs" else "FAIL"
        spec.warning_text = MANDATED_WARNING.replace(old, {"may cause": "can cause", "should not": "must not",
                                                           "impairs": "impair"}[old])
    elif variant == "missing_net_contents":
        spec.expected_issue, spec.volume_text = "net_contents", None
    return spec


# --- writing ---------------------------------------------------------------------------------------
def write_set(specs: list[tuple[str, Spec]], out_dir: Path, csv_name: str, id_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, spec) in enumerate(specs, start=1):
        filename = f"{name}.png"
        save(render(spec), out_dir / filename)
        rows.append({"image": filename, "application_id": f"{id_prefix}-{i:04d}", **spec.app,
                     "variant": spec.variant, "expected_overall": spec.expected_overall,
                     "expected_issue": spec.expected_issue})
    with open(out_dir / csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    total = sum((out_dir / r["image"]).stat().st_size for r in rows)
    print(f"wrote {len(rows)} labels to {out_dir} ({total / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["samples", "batch"])
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.what == "samples":
        write_set(sample_specs(), args.out or ROOT / "data" / "samples", "samples.csv", "SAMPLE")
    else:
        rng = random.Random(args.seed)
        specs = []
        for i in range(args.count):
            spec = apply_variant(random_spec(rng), rng.choice(BATCH_VARIANTS), rng)
            specs.append((f"label_{i + 1:04d}", spec))
        write_set(specs, args.out or ROOT / "data" / "batch", "applications.csv", "APP")


if __name__ == "__main__":
    main()
