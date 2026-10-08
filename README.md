# TTB Label Check

A prototype that verifies an alcohol beverage label image against the application data filed with
TTB, built for the Treasury take-home. An agent types (or uploads a CSV of) the application values,
adds the label image(s), and gets a per-field verdict in a few seconds, plus a strict check of the
government health warning.

- **Live demo:** https://ttb-label-verifier-production-28e6.up.railway.app
- **Source:** https://github.com/azizbenj/ttb-label-verifier

| What you give it | What you get back |
|---|---|
| Brand name, class/type, alcohol content, net contents, bottler name & address, country of origin (imports) | One row per field: the application value, the text found on the label, and a verdict: **MATCH**, **NEAR MATCH** (agent decides), **MISMATCH** or **NOT FOUND**, with a one-line reason |
| The label image (PNG/JPG/TIFF/WEBP), or a CSV + 200-300 images for a batch | Government warning: present / exact wording (with a word diff) / heading in capitals / heading bold |
| | Timing for every label, and for batches a filterable table and a CSV export |

## Quick start (local)

Prerequisites: Python 3.12 and the Tesseract binary (`brew install tesseract`, `apt-get install tesseract-ocr tesseract-ocr-eng`, or `conda install -c conda-forge tesseract`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
pytest -q
```

If Tesseract is not on your PATH, point the app at it with `TESSERACT_CMD=/path/to/tesseract`.

## Run with Docker

```bash
docker build -t ttb-label-check .
docker run --rm -p 8000:8000 ttb-label-check
```

The image is `python:3.12-slim` plus the Debian `tesseract-ocr` package (about 30 MB). It makes no
network calls at runtime, so it can run inside the agency network.

## Deploy (Railway)

```bash
railway login
railway init -n ttb-label-verifier
railway up --detach
railway domain
```

Railway builds the Dockerfile on its side. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OCR_ENGINE` | `tesseract` | Default reader (`tesseract` or `claude`) |
| `ANTHROPIC_API_KEY` | unset | When set, the UI shows a "Reader" toggle and the cloud reader can be selected per request |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used by the cloud reader |
| `BATCH_WORKERS` | `min(4, CPUs)` | Parallel OCR workers for batch jobs |
| `TESSERACT_PSM` | `4+11` | Tesseract page-segmentation mode(s); two modes joined with `+` are merged (see "OCR choice") |
| `TESSERACT_CMD` | unset | Path to the tesseract binary if it is not on PATH |

## How it works

```
label image ──> LabelReader ──> text + word boxes ──> locate each application value ──> verdict per field
                 (Tesseract,                           on the label (fuzzy search)
                  or Claude)                       └─> government warning checks ──────> PASS / REVIEW / FAIL
application ─────────────────────────────────────────┘
```

1. **Read.** The image is converted to grayscale, scaled so the width is at least 1600 px, contrast-stretched and
   handed to Tesseract 5 (LSTM engine) twice: once in "single column" mode (PSM 4) and once in "sparse text" mode
   (PSM 11). No single mode reads every label (block modes drop oversized brand lines, sparse mode occasionally
   misses short centred lines), so lines the first pass did not produce are appended from the second. The two passes
   cost about 1 s together on Railway's shared CPU. We keep the word boxes and a binarized copy of the image for the
   bold check. OCR digit confusions are repaired in context (`75O` → `750`, `L.75 L` → `1.75 L`).
2. **Find each field.** We know what we are looking for, so instead of parsing an arbitrary label we search for the
   span of consecutive label words (across up to three lines) that best matches each application value. This is far
   more robust than blind extraction: "Bottled by OLD TOM DISTILLERY, Bardstown" still yields "OLD TOM DISTILLERY".
   Alcohol content and net contents are picked up with patterns (`45% Alc./Vol.`, `90 Proof`, `750 mL`, `12 FL OZ`)
   and compared as numbers. If nothing resembles the brand name, the most prominent line on the label is shown as the
   mismatch so the agent sees what the label actually says.
3. **Compare** after normalization (next section) and classify with the thresholds in `app/config.py`.
4. **Government warning:** locate the statement, diff it word by word against 27 CFR 16.21, check the heading's
   capitalization from the OCR text, and estimate its boldness from the image.

Code map:

| Path | Role |
|---|---|
| `app/config.py` | Every threshold, the mandated warning text, field definitions, runtime settings |
| `app/normalize.py` | Unicode/case/punctuation normalization, ABV-phrase unification, proof→ABV, volume→mL, OCR digit repair |
| `app/matching.py` | Guided fuzzy location of a value on the label, verdict rules, numeric comparison, country of origin |
| `app/warning.py` | Warning statement location, wording diff, heading caps, stroke-width bold heuristic |
| `app/readers/` | `base.py` (interface), `tesseract.py` (local OCR), `claude_vision.py` (cloud), `extract.py` (rules over OCR) |
| `app/pipeline.py` | One label end to end with timings and the overall verdict |
| `app/batch.py` | CSV parsing, image/zip intake, thread-pool jobs, CSV export |
| `app/main.py` + `templates/` + `static/` | FastAPI routes, server-rendered UI (no build step, no CDN) |
| `scripts/generate_labels.py` | Synthetic label generator (Pillow) |
| `data/samples/`, `data/batch/` | 15 sample labels (one per failure type) and a 250-label batch, each with its application CSV |
| `tests/` | Unit tests for every module plus end-to-end and API tests |

## OCR choice and trade-offs

The local default is **Tesseract 5**. The reader sits behind a small interface (`app/readers/base.py`), and a
**Claude Vision** reader can be enabled with an environment variable for demos.

| | Tesseract 5 (default) | PaddleOCR | Claude Vision (opt-in) |
|---|---|---|---|
| Runs inside a locked-down network | Yes, no downloads at runtime | Yes, but models must be baked into the image (it downloads them on first run) | No, needs outbound HTTPS |
| Speed per label on CPU | ~0.5-1.5 s on a 1200×1600 label | ~1.5-4 s | ~3-8 s (network + model) |
| Deployment weight | +30 MB apt package | +1 GB (PaddlePaddle + models) | SDK only |
| Clean rendered labels (artwork, scans) | Very good | Very good | Excellent |
| Decorative/script fonts, curved text | Weak | Better | Best |
| Photos at an angle, glare | Poor (out of scope) | Fair | Good |
| Word boxes (needed for the bold heuristic) | Yes | Line boxes only | Not needed: the model judges boldness itself |
| Data leaves the agency | No | No | Yes |

Why Tesseract: the 5-second budget and the "cloud APIs may be blocked" constraint dominate. Tesseract is the
only option that is both fast on a small container and trivially deployable in an air-gapped network, and the
application-guided matching compensates for most of its OCR noise. The main cost is weak reading of highly
stylized brand typography; that is where the cloud reader shines, and why it is one flag away.

Page-segmentation mode: the generated labels were benchmarked with PSM 3, 4, 6, 11 and merged pairs; see
"Measured results". `TESSERACT_PSM=4+11` is the default; a single mode (`TESSERACT_PSM=4`) halves the read time.

## Matching rules and thresholds

Normalization applied before any text comparison (`app/normalize.py`):

- Unicode NFKC; curly quotes and apostrophes → straight; dashes → `-`; non-breaking spaces → spaces
- Whitespace collapsed; case folded; punctuation dropped except inside numbers (`45.5`, `1,000`)
- `Alc./Vol.`, `Alc. by Vol.`, `Alcohol by Volume`, `ABV` → one token
- Proof → ABV (`90 Proof` = 45%); `mL`/`ml`/`ML`/`milliliters`, `L`/`liter(s)`, `cl`, `fl oz`/`oz` → millilitres
- OCR digit repair inside numbers only: `75O mL` → `750 mL`, `l0` → `10`

Verdicts (`app/matching.py`), all thresholds in `app/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| strict equality | — | Identical after unicode/whitespace cleanup → **MATCH** |
| loose equality | — | Identical after full normalization → **MATCH**, except for fields in `case_review_fields` (`brand_name`) where a capitalization/punctuation-only difference is a **NEAR MATCH** ("STONE'S THROW" vs "Stone's Throw": trivial, but an agent should confirm) |
| `near_match` | 88 | rapidfuzz similarity (0-100) at or above this → **NEAR MATCH**, below → **MISMATCH** |
| `find_floor` | 60 | Best-matching span on the label scores below this → **NOT FOUND** |
| `abv_tolerance` | 0.05 pp | Alcohol content compared as numbers; any larger difference is a **MISMATCH** |
| `volume_tolerance_ml` | 0.5 mL | Net contents compared in millilitres (so `12 FL OZ` = `355 mL`). Same digits and unit but no decimal point on the label (`15 L` read for `1.5 L`) → NEAR MATCH, never a silent match |
| `warning_locate` | 75 | Similarity needed to recognise the "GOVERNMENT WARNING" line |
| `warning_near` | 97 | Warning wording at or above this (but not exact) → NEEDS REVIEW with a diff; below → FAIL |
| `bold_ratio` | 1.45 | Heading stroke width ÷ body stroke width at or above this → "looks bold" (measured: bold headings 1.5-2.4, regular 0.95-1.38) |
| `bold_min_text_px` | 14 | Below this text height the stroke measurement is not attempted |
| `bold_failure_is_fail` | false | A "does not look bold" result asks for review instead of failing the label |

Blank optional fields on the application (bottler, country) are **SKIPPED**, not failed. Country of origin is
matched against "Product of X" style statements; a different country on the label is a MISMATCH that shows what
the label says.

Overall label status: **FAIL** if any field is MISMATCH/NOT FOUND or the warning fails; **REVIEW** if any field is
NEAR MATCH or a warning check needs a look; otherwise **PASS**.

## Government warning check

Required text (27 CFR 16.21):

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during
> pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to
> drive a car or operate machinery, and may cause health problems.

Four separate results are shown so the agent sees exactly what is wrong:

1. **Present**: the statement (or its body) was found on the label.
2. **Wording**: compared word for word after normalization. Punctuation is ignored because OCR drops commas and
   periods unreliably. Any difference is listed as "required text says / label says". A difference with similarity
   ≥ 97 is flagged for **review** (it may be a misprint or an OCR error; the diff lets the agent decide in a second);
   anything larger (missing sentence, paraphrase) **fails**.
3. **Heading in capitals**: the OCR text of the heading must read `GOVERNMENT WARNING:`; title case fails, a missing
   colon asks for review.
4. **Heading bold (heuristic)**: from the word boxes, we crop the heading words and the body words of the statement,
   measure the median length of ink runs (horizontal and vertical, long runs excluded), i.e. the stroke width, and
   normalize by text height. If the heading's strokes are at least `bold_ratio` (1.45×) thicker than the body's it
   "looks bold". Measured on the bundled fonts the ratio is 1.5-2.1 for bold and 0.95-1.06 for regular weight, so
   there is margin, but it is still a heuristic that assumes the body is set in regular weight at a similar size,
   which is how the statement is printed in practice. Doubtful results ask for a look rather than failing the label.
   With the cloud reader the model answers the bold question directly.

## Batch mode

Upload a CSV and the label images (multi-select, or a zip, or both). Images are matched to rows by file name
(case-insensitive; the extension may be omitted in the CSV). Columns are matched case-insensitively and common
aliases are accepted (`brand`, `abv`, `volume`, `bottler`, `country`...):

| Column | Required | Example |
|---|---|---|
| `image` | yes | `label_0001.png` |
| `brand_name` | yes | `OLD TOM DISTILLERY` |
| `class_type` | yes | `Kentucky Straight Bourbon Whiskey` |
| `alcohol_content` | yes | `45% Alc./Vol.`, `90 Proof`, `45` |
| `net_contents` | yes | `750 mL`, `1.75 L`, `12 fl oz` |
| `bottler_name_address` | no | `Old Tom Distillery, Bardstown, Kentucky 40004` |
| `country_of_origin` | no (imports) | `Scotland` |
| `application_id` | no | `APP-0001` |

A template is downloadable from the UI. Rows with blank required values, rows without an image, and images without
a row are reported in plain language and the rest of the batch still runs. Jobs run in a thread pool
(`BATCH_WORKERS`) and the page polls every second; results have Pass/Review/Fail filters, a search box, click-to-expand
details, and a CSV export with every field's verdict, found text, note, the four warning results and timings.

Limits: 500 labels per batch, 20 MB per image, zip-bomb guards. Jobs live in memory for the life of the process.

## Test data

`scripts/generate_labels.py` renders flat, clean labels with Pillow (DejaVu Sans/Serif, three layouts, four palettes).

```bash
python scripts/generate_labels.py samples               # data/samples: 15 labels + samples.csv
python scripts/generate_labels.py batch --count 250     # data/batch: 250 labels + applications.csv
```

Sample set: four clean bases (the README's OLD TOM DISTILLERY bourbon, a wine, a beer, an imported Scotch with
"Product of Scotland") and one variant per failure type: wrong ABV, missing warning, warning heading not in capitals,
warning heading not bold, brand capitalization differs, wrong net contents, different brand, one word of the warning
changed, warning cut short, net contents missing, wrong country. Every row carries the application values plus the
expected outcome, so the same files drive the end-to-end tests and the "Try a sample" menu.

Batch set: 250 random labels across spirits, wine, beer and imports; about 80% clean, the rest spread over the failure
types; `expected_overall` is in the CSV so accuracy can be measured.

## Tests

```bash
pytest -q
```

- `test_normalize.py`: quotes, case, ABV phrases, proof→ABV, mL/L/cl/fl oz, OCR digit repair
- `test_matching.py`: every verdict boundary, the brand-name case rule, numeric tolerances, multi-line addresses, country of origin
- `test_warning.py`: exact/altered/truncated wording with diffs, heading capitalization, the bold heuristic on rendered bold vs regular text in both font families and two sizes
- `test_e2e.py`: every sample label through the real Tesseract pipeline must produce its expected outcome in under 5 s
- `test_api.py`: the HTTP surface, friendly errors (missing fields, bad values, non-image files), a small batch with a missing image and a stray file, CSV export
- `test_batch_flow.py`: the single and batch HTTP flows with a fake reader, so templates and job handling are covered without Tesseract

OCR-dependent tests skip automatically when the Tesseract binary is absent. `.github/workflows/ci.yml` runs the whole
suite on Ubuntu with Tesseract installed and benchmarks the sample set. `scripts/bench.py` reports accuracy and timing
for the sample or batch set, locally or against a deployed URL (`--url`).

## Measured results

All numbers from `scripts/bench.py` against the Railway deployment (shared vCPU, one container), so they include
HTTP overhead; local runs on a laptop are faster. "Expected verdict" means the overall PASS / REVIEW / FAIL the
generator recorded for that label.

**Sample set (15 labels, one per failure type), reader `4+11`:** 15/15 expected verdicts, median 1.1 s per label,
p95 1.3 s, max 1.3 s. The README's OLD TOM DISTILLERY label passes every field and the warning in about 1 s.

**Batch set (250 labels: 202 clean, 48 with a planted defect):**

| Tesseract mode | Expected verdict | Median | p95 |
|---|---|---|---|
| `6` (uniform block) | 12/15 on the samples: drops oversized brand lines | 0.44 s | 0.46 s |
| `3` (auto) | 214/250 | 0.49 s | 0.71 s |
| `11` (sparse) | 230/250 | 0.57 s | 0.65 s |
| `4` (single column) | 229/250 | 0.57 s | 0.66 s |
| `4+11` merged (default) | **241/250 (96%)** | 1.09 s | 1.23 s |
| `3+11` merged | 241/250 | 1.07 s | 1.24 s |

Of the 9 misses with the default reader, 7 are conservative: the tool asked for a look (REVIEW) on a label the
generator marked clean, because Tesseract dropped the decimal point in a litre volume ("1.5 L" read as "15L") or
misread a capital in the brand line and matched the title-case bottler mention instead. One is a genuine OCR error
("750 mL" read as "790 mL", reported as a MISMATCH an agent would resolve from the image). One was a regular-weight
warning heading that measured 1.38× and passed the earlier 1.25 bold threshold; the threshold is now 1.45 (bold
headings measure 1.5-2.4, regular ones 0.95-1.38 on this data). No clean label was failed for a wrong reason and,
more importantly, no planted defect was reported as a PASS.

Throughput: the 250-label sample batch completes in about 90 s through the UI with four workers, i.e. 0.35 s per
label of wall-clock time.

## Assumptions

- Labels arrive as flat artwork files or straight-on scans, the way they are attached to applications.
- The application data is trusted; the label is what is being verified.
- "Exact wording" of the warning means the words; punctuation and line breaks are not judged.
- The regulation requires only the heading to be bold and in capitals; the body's weight is not checked.
- A capitalization-only difference matters for the brand name (flagged for review) but not for the other fields.
- English-language labels.

## Known limitations

- Photos taken at an angle, with glare, shadows or poor lighting are out of scope; expect NOT FOUND results and a
  warning-statement failure on such images. Deskewing and perspective correction would be the first thing to add.
- Tesseract struggles with decorative, script or outlined brand typography and with text on busy backgrounds. The
  guided matching tolerates a fair amount of noise, but a brand set in a script face may come back NOT FOUND
  (the cloud reader handles these).
- The bold heuristic is calibrated on synthetic labels and assumes the statement body is in regular weight. A
  real-world calibration set would be needed before trusting it unattended, which is why it only asks for review.
- Batch jobs are kept in memory and disappear on restart; for production they would go to a queue and a database.
- No authentication: the prototype assumes it runs on an internal network.
- Vertical or rotated text is not read.

## Security notes

Uploads are validated (image type, 20 MB limit, zip member and size limits), processed in memory and never written
to disk. Sample file names are whitelisted. The app makes no outbound calls unless the cloud reader is enabled.
