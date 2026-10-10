"""The reader interface: anything that turns a label image into text (and optionally fields).

Two implementations ship:
  * TesseractReader (local, default)  - returns OCR text with word boxes; fields are then
    extracted by rules in app/readers/extract.py.
  * ClaudeVisionReader (cloud, opt-in) - returns a transcription plus structured fields
    and a judgment on the warning heading, so no rule-based extraction is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


class ReaderError(RuntimeError):
    """The reader could not read this label (timeout, cloud error, refusal); the message is for the agent."""


@dataclass
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line_index: int  # index into OCRResult.lines
    view: int = 0    # index into OCRResult.views: the coordinates are those of that view's image
    engine: str = ""  # which engine read it ("" = Tesseract, "rapid" = RapidOCR): their confidences differ in meaning

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class View:
    """One image the OCR read: the label as is, turned 90 degrees either way (sideways text), inverted
    (light text on a dark panel), or a crop of one line scaled up for a second read of a figure
    (app/readers/numbers.py). Word boxes are in the coordinates of their view."""

    rot: int                      # 0, 90 (turned counter-clockwise) or 270 (turned clockwise)
    inverted: bool
    ink: np.ndarray               # True = ink, in this view's coordinates
    size: tuple[int, int]         # (width, height) of this view
    engine: str = ""              # "rapid" for a RapidOCR read: searched as a frame of its own by the warning check
    scale: float = 1.0            # a crop was enlarged by this factor before it was read
    offset: tuple[int, int] = (0, 0)   # where the crop's top-left corner sits on the upright image


def upright_box(word: "OCRWord", views: list[View]) -> tuple[int, int, int, int]:
    """(left, top, width, height) of a word in the upright view's coordinates."""
    if not views or word.view == 0 or word.view >= len(views):
        return word.left, word.top, word.width, word.height
    v = views[word.view]
    left, top, width, height = word.left, word.top, word.width, word.height
    if v.scale != 1.0:
        left, top = round(left / v.scale), round(top / v.scale)
        width, height = round(width / v.scale), round(height / v.scale)
    # The region of the upright image this view shows, as the view's size before the turn.
    region_w, region_h = round(v.size[0] / v.scale), round(v.size[1] / v.scale)
    if v.rot == 90:      # image turned counter-clockwise: x' = y, y' = W - x  (W = the region's upright width)
        left, top, width, height = region_h - top - height, left, height, width
    elif v.rot == 270:   # image turned clockwise: x' = H - y, y' = x  (H = the region's upright height)
        left, top, width, height = top, region_w - left - width, height, width
    return left + v.offset[0], top + v.offset[1], width, height


def text_height(word: "OCRWord", views: list[View]) -> float:
    """The height of a word's letters on the page: a word read from an enlarged crop is scaled back."""
    if views and 0 < word.view < len(views):
        return word.height / views[word.view].scale
    return float(word.height)


@dataclass
class Reread:
    """A second read of one line's own crop, made for a figure (app/readers/numbers.py)."""

    line: int                   # the OCR line that was cropped
    kinds: tuple[str, ...]      # the figures it was read for: "alcohol" and/or "volume"
    text: str                   # what the second read says
    mode: str                   # how it was read (Tesseract page-segmentation mode and scale)
    conf: float                 # mean word confidence of the second read
    new_line: int | None = None  # index in OCRResult.lines, or None when the second read repeats a line


@dataclass
class OCRResult:
    text: str                       # full text, one OCR line per line
    lines: list[str]                # same, split
    words: list[OCRWord] = field(default_factory=list)
    engine: str = ""
    ms: float = 0.0
    ink: np.ndarray | None = None   # binarized upright image (True = ink), if available
    mean_conf: float | None = None
    views: list[View] = field(default_factory=list)   # views[0] is the upright image
    extended: bool = False          # the extra rotated / inverted passes have run
    escalated: bool = False         # the second engine (RapidOCR) has read the label
    source: object = None           # the preprocessed upright image, kept so extra passes need not redo it
    skew: float = 0.0               # degrees the image was turned to straighten it (boxes refer to the straightened image)
    rereads: list[Reread] = field(default_factory=list)   # second reads of figure lines, each a line of its own


@dataclass
class WarningHint:
    """What a vision model reported about the warning statement (cloud reader only)."""

    text: str | None = None
    heading_bold: bool | None = None


@dataclass
class LabelReading:
    ocr: OCRResult
    fields: dict[str, str | None] | None = None   # key -> text as printed on the label, or None if absent
    warning_hint: WarningHint | None = None


class LabelReader(Protocol):
    name: str

    def read(self, image: Image.Image) -> LabelReading: ...
