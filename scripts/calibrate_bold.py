#!/usr/bin/env python
"""Calibrate the warning-heading bold heuristic on rendered labels with exact word boxes.

Renders the batch specs (same seed as data/batch), runs the reader's preprocessing (no Tesseract needed) and
reports the heading/body stroke-width ratio for bold and regular headings, by font family and blur.
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_labels import BATCH_VARIANTS, apply_variant, random_spec, render  # noqa: E402
from app.readers.base import OCRWord  # noqa: E402
from app.readers.tesseract import preprocess  # noqa: E402
from app.warning import estimate_heading_bold  # noqa: E402


def main(count: int = 250, seed: int = 42) -> None:
    rng = random.Random(seed)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for i in range(count):
        spec = apply_variant(random_spec(rng), rng.choice(BATCH_VARIANTS), rng)
        if not spec.warning_text:
            continue
        boxes: list = []
        img = render(spec, boxes)
        scaled, ink = preprocess(img)
        f = scaled.width / img.width
        words = [OCRWord(text=t, left=round(l * f), top=round(tp * f), width=round((r - l) * f), height=round((b - tp) * f),
                         conf=99, line_index=0 if head else 1) for t, (l, tp, r, b), head in boxes]
        heading = [w for w in words if w.line_index == 0]
        body = [w for w in words if w.line_index == 1 and len(w.text) >= 3]
        status, ratio, note = estimate_heading_bold(ink, heading, body)
        groups[(spec.heading_bold, spec.family, spec.blur)].append(ratio or 0.0)
    print(f"{'bold':5} {'family':6} {'blur':4} {'n':>3} {'min':>5} {'med':>5} {'max':>5}")
    for key in sorted(groups):
        vals = groups[key]
        print(f"{str(key[0]):5} {key[1]:6} {key[2]:<4} {len(vals):3d} {min(vals):5.2f} {statistics.median(vals):5.2f} {max(vals):5.2f}")


if __name__ == "__main__":
    main()
