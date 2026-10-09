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

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class View:
    """One image the OCR read: the label as is, turned 90 degrees either way (sideways text), or
    inverted (light text on a dark panel). Word boxes are in the coordinates of their view."""

    rot: int                      # 0, 90 (turned counter-clockwise) or 270 (turned clockwise)
    inverted: bool
    ink: np.ndarray               # True = ink, in this view's coordinates
    size: tuple[int, int]         # (width, height) of this view


def upright_box(word: "OCRWord", views: list[View]) -> tuple[int, int, int, int]:
    """(left, top, width, height) of a word in the upright view's coordinates."""
    if not views or word.view == 0 or word.view >= len(views):
        return word.left, word.top, word.width, word.height
    v, base = views[word.view], views[0]
    w0, h0 = base.size
    if v.rot == 90:      # image turned counter-clockwise: x' = y, y' = W - x
        return w0 - word.top - word.height, word.left, word.height, word.width
    if v.rot == 270:     # image turned clockwise: x' = H - y, y' = x
        return word.top, h0 - word.left - word.width, word.height, word.width
    return word.left, word.top, word.width, word.height


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
    source: object = None           # the preprocessed upright image, kept so extra passes need not redo it


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
