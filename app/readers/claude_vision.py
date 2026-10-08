"""Cloud reader: a Claude vision model transcribes the label and returns the fields directly.

Opt-in (needs ANTHROPIC_API_KEY). More robust than Tesseract on decorative fonts and it can
judge the warning heading's weight itself, but it needs outbound internet access, which the
agency network blocks. See README "OCR choice and trade-offs".
"""

from __future__ import annotations

import base64
import io
from time import perf_counter

import anthropic
from PIL import Image
from pydantic import BaseModel, Field

from ..config import CLAUDE_MODEL
from .base import LabelReading, OCRResult, WarningHint

_MAX_SIDE = 1568  # Claude's vision sweet spot; larger images are downscaled anyway

SYSTEM_PROMPT = (
    "You read alcohol beverage labels for a government compliance check. Transcribe text exactly as "
    "printed: keep capitalization, punctuation and spelling, even when they look wrong. Never infer or "
    "complete a value that is not printed. Use null for anything absent."
)


class LabelExtraction(BaseModel):
    transcription: str = Field(description="All text on the label, one printed line per line, in reading order, exactly as printed.")
    brand_name: str | None = Field(description="The brand name exactly as printed, or null.")
    class_type: str | None = Field(description="The class/type designation (e.g. 'Kentucky Straight Bourbon Whiskey') exactly as printed, or null.")
    alcohol_content: str | None = Field(description="The alcohol content statement exactly as printed, e.g. '45% Alc./Vol. (90 Proof)', or null.")
    net_contents: str | None = Field(description="The net contents statement exactly as printed, e.g. '750 mL', or null.")
    bottler_name_address: str | None = Field(description="The name-and-address statement (e.g. 'Bottled by ...') exactly as printed, or null.")
    country_of_origin: str | None = Field(description="The country of origin statement exactly as printed (e.g. 'Product of Scotland'), or null.")
    government_warning_text: str | None = Field(description="The complete government warning statement exactly as printed, or null if there is none.")
    warning_heading_is_bold: bool | None = Field(description="True if the words 'GOVERNMENT WARNING' are printed in heavier (bold) type than the rest of the statement, false if not, null if there is no warning.")


def _encode(image: Image.Image) -> str:
    img = image.convert("RGB")
    scale = _MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class ClaudeVisionReader:
    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()

    def read(self, image: Image.Image) -> LabelReading:
        t0 = perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},  # transcription needs little deliberation; keeps latency low
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode(image)}},
                    {"type": "text", "text": "Read this label and fill in every field."},
                ],
            }],
            output_format=LabelExtraction,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The cloud reader declined to read this image.")
        parsed: LabelExtraction = response.parsed_output
        lines = [l.strip() for l in (parsed.transcription or "").splitlines() if l.strip()]
        if parsed.government_warning_text and "government warning" not in parsed.transcription.lower():
            lines += [l.strip() for l in parsed.government_warning_text.splitlines() if l.strip()]
        ocr = OCRResult(text="\n".join(lines), lines=lines, words=[], engine=f"claude ({self.model})",
                        ms=(perf_counter() - t0) * 1000)
        fields = {k: getattr(parsed, k) for k in ("brand_name", "class_type", "alcohol_content", "net_contents",
                                                   "bottler_name_address", "country_of_origin")}
        hint = WarningHint(text=parsed.government_warning_text, heading_bold=parsed.warning_heading_is_bold)
        return LabelReading(ocr=ocr, fields=fields, warning_hint=hint)
