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

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class OCRResult:
    text: str                       # full text, one OCR line per line
    lines: list[str]                # same, split
    words: list[OCRWord] = field(default_factory=list)
    engine: str = ""
    ms: float = 0.0
    ink: np.ndarray | None = None   # binarized image the boxes refer to (True = ink), if available
    mean_conf: float | None = None


@dataclass
class WarningHint:
    """What a vision model reported about the warning statement (cloud reader only)."""

    text: str | None = None
    heading_caps: bool | None = None
    heading_bold: bool | None = None


@dataclass
class LabelReading:
    ocr: OCRResult
    fields: dict[str, str | None] | None = None   # key -> text as printed on the label, or None if absent
    warning_hint: WarningHint | None = None


class LabelReader(Protocol):
    name: str

    def read(self, image: Image.Image) -> LabelReading: ...
