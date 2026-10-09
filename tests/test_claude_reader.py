"""The cloud reader's mapping from the model's structured answer to a LabelReading (SDK call mocked)."""

import base64
import io
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest
from PIL import Image

from app.models import Application, Status, Verdict
from app.pipeline import verify
from app.readers import claude_vision
from app.readers.base import ReaderError
from app.readers.claude_vision import ClaudeVisionReader, LabelExtraction
from app.readers.extract import compare_from_fields


class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn", error=None):
        self.parsed, self.stop_reason, self.error, self.calls = parsed, stop_reason, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


def make_reader(monkeypatch, parsed, stop_reason="end_turn", model="test-model", error=None):
    fake = FakeMessages(parsed, stop_reason, error)
    client_kwargs = {}

    def client(**kwargs):
        client_kwargs.update(kwargs)
        return SimpleNamespace(messages=fake, beta=SimpleNamespace(messages=fake))
    monkeypatch.setattr(claude_vision.anthropic, "Anthropic", client)
    reader = ClaudeVisionReader(model=model)
    fake.client_kwargs = client_kwargs
    return reader, fake


EMPTY = dict(transcription="", brand_name=None, class_type=None, alcohol_content=None, net_contents=None,
             bottler_name_address=None, country_of_origin=None, government_warning_text=None, warning_heading_is_bold=None)


WARNING = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages "
           "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs "
           "your ability to drive a car or operate machinery, and may cause health problems.")


def test_reader_maps_fields_and_warning_hint(monkeypatch):
    parsed = LabelExtraction(
        transcription="OLD TOM DISTILLERY\nKentucky Straight Bourbon Whiskey\n45% Alc./Vol. (90 Proof)\n750 mL\n" + WARNING,
        brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol. (90 Proof)", net_contents="750 mL", bottler_name_address=None,
        country_of_origin=None, government_warning_text=WARNING, warning_heading_is_bold=True)
    reader, fake = make_reader(monkeypatch, parsed)
    reading = reader.read(Image.new("RGB", (1200, 1600), "white"))
    assert reading.fields["brand_name"] == "OLD TOM DISTILLERY" and reading.fields["bottler_name_address"] is None
    assert reading.warning_hint.heading_bold is True and reading.ocr.lines[0] == "OLD TOM DISTILLERY"
    call = fake.calls[0]
    assert call["model"] == "test-model" and call["output_format"] is LabelExtraction
    assert call["messages"][0]["content"][0]["type"] == "image"

    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="90 Proof", net_contents="750 mL")
    result = verify(app, Image.new("RGB", (10, 10), "white"), reader)
    assert result.overall == Status.PASS and result.warning.heading_bold == Status.PASS
    assert {f.verdict for f in result.fields if f.verdict != Verdict.SKIPPED} == {Verdict.MATCH}


def test_reader_appends_warning_when_missing_from_transcription(monkeypatch):
    parsed = LabelExtraction(transcription="OLD TOM DISTILLERY", brand_name="OLD TOM DISTILLERY", class_type=None,
                             alcohol_content=None, net_contents=None, bottler_name_address=None, country_of_origin=None,
                             government_warning_text=WARNING, warning_heading_is_bold=False)
    reader, _ = make_reader(monkeypatch, parsed)
    reading = reader.read(Image.new("RGB", (10, 10), "white"))
    assert any(l.startswith("GOVERNMENT WARNING") for l in reading.ocr.lines)


def test_refusal_and_incomplete_answers_are_reader_errors(monkeypatch):
    reader, _ = make_reader(monkeypatch, LabelExtraction(**EMPTY), stop_reason="refusal")
    with pytest.raises(ReaderError, match="declined"):
        reader.read(Image.new("RGB", (10, 10), "white"))
    reader, _ = make_reader(monkeypatch, None, stop_reason="max_tokens")
    with pytest.raises(ReaderError, match="complete reading"):
        reader.read(Image.new("RGB", (10, 10), "white"))


def _sdk_error(cls, **attrs):
    e = cls.__new__(cls)  # the SDK constructors need a live HTTP request/response
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


@pytest.mark.parametrize("error, message", [
    (_sdk_error(anthropic.APITimeoutError), "did not answer"),
    (_sdk_error(anthropic.APIConnectionError), "could not be reached"),
    (_sdk_error(anthropic.AuthenticationError, status_code=401), "API key"),
    (_sdk_error(anthropic.RateLimitError, status_code=429), "busy"),
    (_sdk_error(anthropic.InternalServerError, status_code=500), r"error \(500\)"),
])
def test_api_failures_become_readable_reader_errors(monkeypatch, error, message):
    reader, _ = make_reader(monkeypatch, None, error=error)
    with pytest.raises(ReaderError, match=message):
        reader.read(Image.new("RGB", (10, 10), "white"))


def test_client_is_bounded_and_default_model_uses_refusal_fallbacks(monkeypatch):
    parsed = LabelExtraction(**{**EMPTY, "transcription": "OLD TOM DISTILLERY"})
    reader, fake = make_reader(monkeypatch, parsed, model="claude-opus-5-5")
    reader.read(Image.new("RGB", (10, 10), "white"))
    assert fake.client_kwargs["timeout"] <= 60 and fake.client_kwargs["max_retries"] <= 1
    call = fake.calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"} and call["output_format"] is LabelExtraction


def test_transparent_image_is_sent_on_white():
    rgba = Image.new("RGBA", (200, 100), (0, 0, 0, 0))  # transparent pixels storing black
    jpeg = Image.open(io.BytesIO(base64.b64decode(claude_vision._encode(rgba))))
    assert np.asarray(jpeg.convert("L")).mean() > 250


def test_cloud_fields_are_compared_like_ocr_lines():
    app = Application(brand_name="OLD TOM", class_type="Kentucky Straight Bourbon Whiskey", alcohol_content="45%",
                      net_contents="750 mL", bottler_name_address="Old Tom Distillery, Bardstown, Kentucky 40004")
    fields = {"brand_name": "OLD TOM DISTILLERY", "class_type": "Kentucky Straight Bourbon Whiskey",
              "alcohol_content": "45% Alc./Vol.", "net_contents": "750 mL", "country_of_origin": None,
              "bottler_name_address": "Distilled and Bottled by Old Tom Distillery, Bardstown, Kentucky 40004"}
    by_key = {r.key: r for r in compare_from_fields(app, fields)}
    assert by_key["bottler_name_address"].verdict == Verdict.MATCH  # the "Bottled by" prefix is expected
    assert by_key["brand_name"].verdict == Verdict.NEAR_MATCH       # "OLD TOM" is only part of the brand
    assert by_key["country_of_origin"].verdict == Verdict.SKIPPED
