from app.normalize import (
    fix_ocr_digits,
    normalize_loose,
    normalize_strict,
    parse_alcohol,
    parse_net_contents,
)


def test_strict_keeps_case_but_fixes_quotes_and_spaces():
    assert normalize_strict("STONE’S   THROW ") == "STONE'S THROW"


def test_loose_lowercases_and_drops_punctuation():
    assert normalize_loose("Stone's Throw!") == "stone s throw"
    assert normalize_loose("STONE’S THROW") == normalize_loose("Stone's Throw")


def test_loose_keeps_decimal_numbers():
    assert normalize_loose("45.5% alc") == "45.5% abv"


def test_abv_phrases_unify():
    for phrase in ("45% Alc./Vol.", "45% ABV", "45% Alcohol by Volume", "45% alc by vol", "45% ALC/VOL"):
        assert normalize_loose(phrase) == "45% abv", phrase


def test_fix_ocr_digits_only_touches_numeric_tokens():
    assert fix_ocr_digits("75O") == "750"
    assert fix_ocr_digits("l0") == "10"
    assert fix_ocr_digits("OLD") == "OLD"
    assert fix_ocr_digits("Oil") == "Oil"


def test_parse_alcohol_percent_and_proof():
    v = parse_alcohol("45% Alc./Vol. (90 Proof)")
    assert v.abv == 45.0 and v.proof == 90.0 and not v.abv_from_proof


def test_parse_alcohol_proof_only_converts():
    v = parse_alcohol("90 PROOF")
    assert v.abv == 45.0 and v.abv_from_proof


def test_parse_alcohol_variants():
    assert parse_alcohol("ALC. 40% BY VOL.").abv == 40.0
    assert parse_alcohol("13.5% ABV").abv == 13.5
    assert parse_alcohol("Alcohol 12,5% by volume").abv == 12.5
    assert parse_alcohol("45").abv == 45.0
    assert parse_alcohol("4O% alc/vol").abv == 40.0  # OCR letter O


def test_parse_alcohol_prefers_percent_near_alc_word():
    v = parse_alcohol("Aged 100% in oak. 45% Alc./Vol.")
    assert v.abv == 45.0


def test_parse_alcohol_none():
    assert parse_alcohol("Kentucky Straight Bourbon") is None
    assert parse_alcohol("") is None


def test_parse_net_contents_ml_variants():
    for s in ("750 mL", "750ml", "750 ML", "750 m l", "75O mL", "750 milliliters"):
        assert parse_net_contents(s).ml == 750.0, s


def test_parse_net_contents_other_units():
    assert parse_net_contents("1.75 L").ml == 1750.0
    assert parse_net_contents("1 LITER").ml == 1000.0
    assert parse_net_contents("70 cl").ml == 700.0
    assert abs(parse_net_contents("12 FL. OZ.").ml - 354.88) < 0.01


def test_parse_net_contents_prefers_metric_when_both():
    v = parse_net_contents("12 FL OZ (355 mL)")
    assert v.ml == 355.0 and v.unit == "mL"


def test_parse_net_contents_none():
    assert parse_net_contents("OLD TOM DISTILLERY") is None


def test_accents_are_folded_in_loose_normalization():
    assert normalize_loose("Rosé Wine") == normalize_loose("Rose Wine")


def test_one_read_as_L_before_decimal_or_litre_unit():
    assert parse_net_contents("L.75L").ml == 1750.0
    assert parse_net_contents("LL").ml == 1000.0
    assert parse_net_contents("L.5 L").ml == 1500.0
    assert parse_net_contents("LITERS") is None  # a plain word is not a volume


def test_parse_alcohol_text_shows_whole_statement():
    assert parse_alcohol("Fine spirit. 45% Alc./Vol. (90 Proof) 750 mL").text == "45% Alc./Vol. (90 Proof)"
    assert parse_alcohol("ALC. 14.5% BY VOL. 750 mL").text == "ALC. 14.5% BY VOL."
    assert parse_alcohol("7.2% ALC./VOL. 12 FL. OZ.").text == "7.2% ALC./VOL."


def test_mangled_volume_letters_are_repaired_only_as_a_fallback():
    assert parse_net_contents("ALC. 15% BY VOL. LSL").ml == 15000.0  # "15 L": the matcher turns it into a near match
    assert parse_net_contents("7S0 mL").ml == 750.0
    assert parse_net_contents("750 mL").ml == 750.0
    assert parse_net_contents("MILLERS LITERS CLUB") is None
    assert parse_net_contents("BOLS L") is None  # a word in front of a unit is not a number


def test_state_codes_in_addresses_are_not_volumes():
    assert parse_net_contents("Imported by Great Lakes Beverage Imports, Chicago, IL 60607") is None
    assert parse_net_contents("Bottled in Springfield, IL") is None
    assert parse_net_contents("50% ALC./VOL. (100 PROOF) LL").ml == 1000.0


def test_thousands_separator_in_volumes():
    assert parse_net_contents("1,000 mL").ml == 1000.0
    assert parse_net_contents("1,750 mL").ml == 1750.0
    assert parse_net_contents("1,5 L").ml == 1500.0       # decimal comma
    assert parse_net_contents("0,750 L").ml == 750.0      # decimal comma, not thousands
    assert parse_net_contents("1.2.3 L") is None          # garbage never crashes the parser


def test_look_alike_repair_never_overrides_a_real_volume():
    # "SOIL" used to become "501 L" and win because it comes first on the label.
    assert parse_net_contents("GROWN IN VOLCANIC SOIL\n750 mL").ml == 750.0
    assert parse_net_contents("OLIVE OIL CASK FINISH 750 mL").ml == 750.0


def test_us_customary_units_and_compound_statements():
    assert abs(parse_net_contents("1 PINT 6 FL. OZ.").ml - parse_net_contents("22 fl oz").ml) < 0.01
    assert abs(parse_net_contents("1 PINT").ml - 473.18) < 0.01
    assert abs(parse_net_contents("1 QT. 8 FL OZ").ml - parse_net_contents("40 fl oz").ml) < 0.01
    assert abs(parse_net_contents("1 GALLON").ml - 3785.41) < 0.01
