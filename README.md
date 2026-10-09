# Label Check

A prototype that verifies an alcohol beverage label image against the data in its label approval
application, built for a take-home exercise. It is an independent prototype, not affiliated with or
endorsed by any government agency. An agent types (or uploads a CSV of) the application values, adds
the label image(s), and gets a per-field verdict in a few seconds, plus a strict check of the
government health warning.

- **Live demo:** https://ttb-label-verifier-production-28e6.up.railway.app
- **Source:** https://github.com/azizbenj/ttb-label-verifier

| What you give it | What you get back |
|---|---|
| Brand name, class/type, alcohol content, net contents, bottler name & address, country of origin (imports) | One row per field: the application value, the text found on the label, and a verdict: **MATCH**, **NEAR MATCH** (agent decides), **MISMATCH** or **NOT FOUND**, with a one-line reason |
| The label image (PNG/JPG/TIFF/WEBP), or several (front, back, neck) for one application, or a CSV + 200-300 images for a batch | Government warning: present / exact wording (with a word diff) / heading in capitals / heading bold |
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
docker build -t label-check .
docker run --rm -p 8000:8000 label-check
```

The image is `python:3.12-slim` plus the Debian `tesseract-ocr` package (about 30 MB). It makes no
network calls at runtime, so it can run inside the agency network.

## Deploy (Railway)

```bash
railway login
railway init -n ttb-label-verifier
railway domain
./scripts/deploy.sh          # railway up, then waits until /healthz reports the new deployment id
```

Railway builds the Dockerfile on its side. The container runs as an unprivileged user, has a Docker
`HEALTHCHECK`, and must run as **one instance with one process**: batch jobs live in that process's memory.
`/healthz` returns 503 when the default reader cannot run (for example no Tesseract binary), so point the
platform's health check at it. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OCR_ENGINE` | `tesseract` | Default reader (`tesseract` or `claude`) |
| `ANTHROPIC_API_KEY` | unset | When set, the UI shows a "Reader" toggle and the cloud reader can be selected per request |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used by the cloud reader |
| `CLAUDE_TIMEOUT_S` | `30` | Per-attempt timeout of a cloud read (one retry) |
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

1. **Read.** The image is flattened (EXIF rotation applied, transparent backgrounds put on white, 16-bit scans
   rescaled), converted to grayscale, scaled so the width is at least 1600 px (never above 10 megapixels), contrast-stretched and
   handed to Tesseract 5 (LSTM engine) twice: once in "single column" mode (PSM 4) and once in "sparse text" mode
   (PSM 11). No single mode reads every label (block modes drop oversized brand lines, sparse mode occasionally
   misses short centred lines), so lines the first pass did not produce are appended from the second. The two passes
   cost about 1 s together on Railway's shared CPU. We keep the word boxes and a binarized copy of the image for the
   bold check. OCR digit confusions are repaired in context (`75O` → `750`, `L.75 L` → `1.75 L`).
   Before reading, a scan tilted by 0.8° to 6° is straightened (the angle at which the rows of ink are most
   uneven); smaller tilts are left alone because Tesseract copes with them and resampling costs detail.
   When the first read leaves a field missing or different, or the warning is not a clean pass, the label is
   read again three more ways, in parallel: turned 90° each way (warnings and bottler lines printed sideways on
   cans and wine labels) and as a color-aware local-contrast image (light or colored text on colored panels).
   Only lines that look like real text are kept from these passes, and a pass that fails or takes longer than
   8 s is skipped rather than failing the label.
   Several images for one application (front, back, neck) are stacked into one before reading, because the
   warning is usually on the back.
2. **Find each field.** We know what we are looking for, so instead of parsing an arbitrary label we search for the
   span of consecutive label words (across up to three lines) that best matches each application value. This is far
   more robust than blind extraction: "Bottled by OLD TOM DISTILLERY, Bardstown" still yields "OLD TOM DISTILLERY".
   The other side of that coin is guarded: for the brand and the class/type, a value that is only part of a longer
   phrase ("Rum" on a label reading "SPICED RUM") or a brand found only in small print (inside "Bottled by ...") is
   a NEAR MATCH, never a silent MATCH.
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
| `app/images.py` | Safe image opening (size, pixel and decompression-bomb limits) and flattening of any image mode to plain pixels |
| `app/normalize.py` | Unicode/case/punctuation normalization, ABV-phrase unification, proof→ABV, volume→mL, OCR digit repair |
| `app/matching.py` | Guided fuzzy location of a value on the label, verdict rules, numeric comparison, country of origin |
| `app/warning.py` | Warning statement location, wording diff, heading caps, stroke-width bold heuristic |
| `app/readers/` | `base.py` (interface), `tesseract.py` (local OCR), `claude_vision.py` (cloud), `extract.py` (rules over OCR) |
| `app/pipeline.py` | One label end to end with timings and the overall verdict |
| `app/evidence.py` | What the result page shows as evidence: crops of the label, unit conversions, misread explanations |
| `app/decisions.py` | The one question a label that needs a look asks the agent (review queue and single result) |
| `app/batch.py` | CSV parsing, image/zip intake, thread-pool jobs, CSV export |
| `app/main.py` + `templates/` + `static/` | FastAPI routes, server-rendered UI (no build step, no CDN, no external assets); `review.html` is the review queue, `report.html` the printable reports |
| `design/` | The UI design spec: 23 boards, tokens, the implementation order |
| `scripts/generate_labels.py` | Synthetic label generator (Pillow) |
| `scripts/bench.py`, `scripts/stress_test.py` | Accuracy and timing on the synthetic sets; the same labels in other typefaces and degraded images |
| `scripts/fetch_registry_labels.py`, `scripts/real_labels.py`, `scripts/real_labels.csv` | Real approved labels: download, ground truth, field-level scoring |
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
- Proof → ABV (`90 Proof` = 45%); `mL`/`ml`/`ML`/`milliliters`, `L`/`liter(s)`, `cl`, `fl oz`/`oz`, pints, quarts
  and gallons → millilitres, including compound statements (`1 PINT 6 FL. OZ.` = 22 fl oz); `1,000 mL` is a
  thousands separator, `1,5 L` a decimal comma
- OCR digit repair inside numbers only: `75O mL` → `750 mL`, `l0` → `10`; volumes whose digits were all read as
  look-alike letters (`LSL` for `1.5 L`) are repaired only when the label has no ordinary volume

Verdicts (`app/matching.py`), all thresholds in `app/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| strict equality | — | Identical after unicode/whitespace cleanup → **MATCH** |
| loose equality | — | Identical after full normalization → **MATCH**, except for fields in `case_review_fields` (`brand_name`) where a capitalization/punctuation-only difference is a **NEAR MATCH** ("STONE'S THROW" vs "Stone's Throw": trivial, but an agent should confirm) |
| `near_match` | 88 | rapidfuzz similarity (0-100) at or above this → **NEAR MATCH**, below → **MISMATCH** |
| `find_floor` | 60 | Best-matching span on the label scores below this → **NOT FOUND** |
| `locate_max_lines` | 3 | A value may wrap over up to this many OCR lines (addresses, long class names) |
| `whole_phrase_fields` | brand, class/type | A MATCH inside a longer phrase on its line ("Rum" in "SPICED RUM", no punctuation between) → **NEAR MATCH** showing the whole line |
| `brand_small_print_ratio` | 0.5 | A brand found only in text under half the height of the label's largest line → **NEAR MATCH** |
| `abv_tolerance` | 0.05 pp | Alcohol content compared as numbers; any larger difference is a **MISMATCH** |
| `proof_tolerance` | 0.5 proof | A label whose proof disagrees with its own percentage by more than this ("45% (80 Proof)") → **NEAR MATCH** |
| `volume_tolerance_ml` | 0.5 mL | Net contents compared in millilitres (so `12 FL OZ` = `355 mL`). Same digits and unit but no decimal point on the label (`15 L` read for `1.5 L`) → NEAR MATCH, never a silent match |
| `warning_locate` | 75 | Similarity needed to recognise the "GOVERNMENT WARNING" line |
| `warning_near` | 97 | Warning wording at or above this (but not exact) → NEEDS REVIEW with a diff; below → FAIL |
| `bold_ratio` | 1.30 | Heading stroke width ÷ body stroke width at or above this → "looks bold" (measured: bold headings 1.51-2.04, regular 1.03-1.14) |
| `bold_min_text_px` | 14 | Below this text height the stroke measurement is not attempted |
| `bold_failure_is_fail` | false | A "does not look bold" result asks for review instead of failing the label |

Real labels taught a few more rules, each covered by tests:

- **Every reading is considered.** A label often states the alcohol content or the volume more than once
  (`750 mL` on the front, `75 cl` on the back), and OCR may produce two readings of one statement. If any
  reading agrees with the application and another one disagrees, the result is a NEAR MATCH that lists both.
  A reading that only lost its decimal point is treated as the same statement, and US and metric figures within
  0.5% of each other (`16 FL OZ` and `473 mL`) agree.
- **A likely misread is not a mismatch.** A volume that is not a standard size and is one digit away from the
  application's (`760 mL` for `750 mL`) is a NEAR MATCH explained as a probable reading error.
- **Letter-spaced brands** (`B A R N  B R E W`) are matched with the spaces closed up; a single-letter
  difference in the brand is a NEAR MATCH, and trademark signs (® ™ ©) are ignored.
- **`U.S.` before a unit and words for numbers** (`ONE PINT`) are understood.

Blank optional fields on the application (bottler, country) are **SKIPPED**, not failed. Country of origin is
decided by the label's origin statements ("Product of X", "Imported from X", "Distilled in X"): only an identical
country name is a MATCH; a close spelling (an OCR slip, or Austria vs Australia) is a NEAR MATCH; a different
country, including one that merely contains the expected name ("Equatorial Guinea" for "Guinea"), is a MISMATCH
that shows what the label says. Without an origin statement, only the country name on a line of its own matches.

Overall label status: **FAIL** if any field is MISMATCH/NOT FOUND or the warning fails; **REVIEW** if any field is
NEAR MATCH or a warning check needs a look; otherwise **PASS**.

## Government warning check

Required text (27 CFR 16.21):

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during
> pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to
> drive a car or operate machinery, and may cause health problems.

Four separate results are shown so the agent sees exactly what is wrong:

1. **Present**: the statement (or its body) was found on the label. Each reading of the label (upright, turned,
   contrast) is searched on its own, its lines in top-to-bottom order; the statement is grown line by line while
   that brings it closer to the required text, passing over up to four lines that belong to something else (a
   neighbouring column, "For sale only in Ohio"). Words at the start or end of a line that belong to text printed
   beside the statement are left out, and the wording result then asks for a look, quoting them.
2. **Wording**: compared word for word after normalization. Punctuation is ignored because OCR drops commas and
   periods unreliably. Any difference is listed as "required text says / label says". A difference with similarity
   ≥ 97 is flagged for **review** (it may be a misprint or an OCR error; the diff lets the agent decide in a second);
   anything larger (missing sentence, paraphrase) **fails**.
3. **Heading in capitals**: the OCR text of the heading must read `GOVERNMENT WARNING:`; title case fails, a missing
   colon asks for review, and capitals with a letter OCR could not read cleanly (`WARNlNG`) ask for review.
4. **Heading bold (heuristic)**: from the word boxes, we take the heading words and the body words of the statement
   and estimate each group's mean stroke width as 2 × ink area ÷ ink perimeter on the binarized image (for a stroke
   of width w and length L the area is wL and the perimeter about 2L, so the estimate does not depend on stroke
   orientation or letter case), normalized by cap height. If the heading's strokes are at least `bold_ratio`
   (1.30×) thicker than the body's it "looks bold". Ink polarity is decided per word, so a light-on-dark warning
   panel is measured correctly. Calibrated with `scripts/calibrate_bold.py`, which renders the 243 warning
   statements of the batch set twice, with a bold and a regular heading (both font families, with and without
   blur): bold headings measure 1.51-2.04, regular ones 1.03-1.14. It is still a heuristic that assumes the body
   is set in regular weight at a similar size; 27 CFR 16.22(a)(2) also forbids a bold body, and an all-bold
   statement measures about 1.0, so it asks for a look. Doubtful results ask for a look rather than failing the
   label. With the cloud reader the model answers the bold question directly.

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

Several images for one application go in one cell, separated by `;` or `|` (`front.png; back.png`, up to six);
they are stacked and read as one label. Comma-, semicolon- and tab-separated files are accepted. A template is downloadable from the UI. Rows with blank
required values or an alcohol content / net contents that cannot be read, rows without an image, images without a
row, and file names shared by several different images (`lot1/label.png` and `lot2/label.png`: never guessed) are
reported in plain language and the rest of the batch still runs. Jobs run in a thread pool (`BATCH_WORKERS`) and
the page polls every second; results have Pass/Review/Fail filters, a search box, click-to-expand details, and a
CSV export with every field's verdict, found text, note, the four warning results and timings (cells that would
start a spreadsheet formula are prefixed with an apostrophe).

Limits: 500 labels and 1 GB of images per batch, 20 MB and 40 megapixels per image, zip-bomb guards. Jobs live in
memory; finished ones are kept for 24 hours (at most the 50 most recent) and disappear on restart. A finished
batch has a link of its own (`/batch/<id>`) that opens the page on the batch tab.

### Reviewing and exporting

- **Triage first.** The results table lists REVIEW rows first, then FAIL, ERROR and PASS; filters (`1` `2` `3`
  `4` `0`), search, sort, `J`/`K` to move, `X` to select, `Enter` for the full comparison. A row whose image
  could not be read is an ERROR (grey), never a FAIL: nothing is wrong with the label.
- **Review queue** (`R`, or "Review the N"): one label that needs a look at a time, with the one thing to look at
  outlined on the label and a single question: "Is this the same brand name?", "Does the label say 'should'?",
  "Is GOVERNMENT WARNING: printed in bold?". `Y` means the label is fine (a reading error on our side), `N` that
  the label is wrong, `S` skips; a decision moves on to the next. Decisions never change a verdict; they are
  kept with the batch and go into the export as a `decision` column. The same prompts appear under a NEAR MATCH
  row or a warning check in the single result and in the full comparison, and print with the report. The
  queue works without JavaScript: the answers are forms.
- **Export dialog**: which labels (all, shown now with the current filter and search, or the selected rows),
  what to include (per-field verdicts and reasons, the warning checks and diff, decisions, all text read,
  timings), and the format: CSV (UTF-8 with a BOM so Excel keeps accents) or printable reports, one page per
  label for the case file with a reviewer decision block. Quick exports: passes only, fails only. The file is
  named after the batch, the scope and the date.
- **Print**: "Print report" on any result prints a Letter page: the application, the verdict, the label with
  its numbered regions, the table, the warning checks, and a reviewer decision block (approve / return /
  second look, signature, notes).

### Error states

Errors are inline, where the result would have been, and say what happened in one line and what to do in the
next. Missing fields are checked in the browser before anything is sent, each input is marked and the
summary links to it; without JavaScript the server returns the same page with the typed values kept. The drop
zone itself becomes the error for a missing, unreadable or oversized image, with the file name quoted. A CSV
with the wrong headers gets a table of expected header vs the near-miss header in the file. An image that
opens but yields almost no text ("fewer than 8 confident words and nothing matched") gets a grey, verdict-free
card with what usually fixes it, never a FAIL. The page works at 1280, 768 and 390 px without horizontal
scrolling; on a phone the batch table becomes a list.

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

Batch set: 250 random labels across spirits, wine, beer and imports: 202 clean and 48 with a planted defect (wrong
ABV 11, heading not in capitals 10, missing warning 7, brand capitalization 6, wrong net contents 5, missing net
contents 4, altered warning 3, heading not bold 2). Wrong brand and wrong country appear only in the sample set.
`expected_overall` is in the CSV so accuracy can be measured.

### Real approved labels

Synthetic labels only prove the tool reads its own generator. To test it on what agents actually see,
`scripts/real_labels.py` runs it on 20 approved labels taken from the public registry of approved label
applications: 6 domestic spirits, 4 imports, 5 wines and 5 beers, chosen to be hard (light text over photos,
sideways warnings, curved display type, a handwritten keg collar, colored panels). The images belong to the
brand owners, so they are not in the repository; `scripts/fetch_registry_labels.py` downloads them into the
gitignored `data/real/`. `scripts/real_labels.csv` records what each label really says, so every field should come
back MATCH unless its `expect` column says otherwise (4 fields are legitimate NEAR MATCHes, 3 are not on the
label). `--defects` also checks every label against a wrong alcohol content and a wrong net contents, which must
never come back MATCH.

```bash
python scripts/fetch_registry_labels.py $(cut -d, -f1 scripts/real_labels.csv | tail -n +2)
python scripts/real_labels.py --stitch       # each record's images into one file
python scripts/real_labels.py --defects
```

`scripts/stress_test.py` renders the 15 sample labels again in eleven other typefaces, six display faces for the
brand, and degraded images (JPEG quality 35, a half and a third of the resolution, tilted 1.5° and 4°, blur,
sensor noise). It only uses fonts already installed on the machine and writes nothing to the repository.

## Tests

```bash
pytest -q
```

- `test_normalize.py`: quotes, case, ABV phrases, proof→ABV, mL/L/cl/fl oz, OCR digit repair
- `test_matching.py`: every verdict boundary, the brand-name case rule, numeric tolerances, multi-line addresses, country of origin
- `test_warning.py`: exact/altered/truncated wording with diffs, heading capitalization, the bold heuristic on rendered bold vs regular text in both font families and two sizes
- `test_e2e.py`: every sample label through the real Tesseract pipeline must produce its expected outcome in under 5 s
- `test_api.py`: the HTTP surface, friendly errors (missing fields, bad values, non-image files, decompression bombs), a small batch with a missing image and a stray file, CSV export
- `test_batch_flow.py`: the single and batch HTTP flows with a fake reader, so templates and job handling are covered without Tesseract; CSV dialects, unreadable rows, same-named images, encrypted zips, export escaping, job retention
- `test_http.py`: concurrent checks do not queue behind each other, reader failures map to readable 502/500 errors, upload size guard, `/healthz` 503
- `test_reader.py`: transparent / 16-bit / palette / CMYK images, size and pixel limits, the two-pass merge bookkeeping, Tesseract timeouts, tilt estimation, straightened previews
- `test_boxes.py`: highlight boxes for each field, including text read from a turned view mapped back onto the upright image
- `test_formats.py`: JPG, TIFF, WEBP and BMP uploads and zips of them, single and batch
- `test_multi_image.py`: several images per application, in the form and in batch rows
- `test_errors.py`: every error state (missing fields with and without JavaScript, bad or oversized image, CSV headers, batch limits, reader failure with retry) and the unreadable-image card and ERROR row
- `test_decisions.py`: review prompts, decisions kept with a batch, the review queue (with and without JavaScript), export scopes and column groups, printable reports, the shared batch link
- `test_claude_reader.py`: the cloud reader's mapping, error handling, refusal fallbacks and client limits, with the SDK mocked

OCR-dependent tests skip automatically when the Tesseract binary is absent. `.github/workflows/ci.yml` runs the whole
suite on Ubuntu with Tesseract installed, then benchmarks both sets and fails the build if a sample misses its
expected verdict, if batch accuracy drops below 95%, or if any label with a planted defect comes back PASS.
`scripts/bench.py` reports accuracy, missed defects and timing for the sample or batch set, locally (`-j N` to run
labels in parallel) or against a deployed URL (`--url`).

## Measured results

All numbers from `scripts/bench.py` against the Railway deployment (shared vCPU, one container), so they include
HTTP overhead; local runs on a laptop are faster. "Expected verdict" means the overall PASS / REVIEW / FAIL the
generator recorded for that label.

**Sample set (15 labels, one per failure type), reader `4+11`:** 15/15 expected verdicts, median 0.8 s per label,
p95 1.0 s. The README's OLD TOM DISTILLERY label passes every field and the warning in under a second.

**Batch set (250 labels: 202 clean, 48 with a planted defect):**

| Tesseract mode | Expected verdict | Median | p95 |
|---|---|---|---|
| `6` (uniform block) | 12/15 on the samples: drops oversized brand lines | 0.44 s | 0.46 s |
| `3` (auto) | 214/250 | 0.49 s | 0.71 s |
| `11` (sparse) | 230/250 | 0.57 s | 0.65 s |
| `4` (single column) | 229/250 | 0.57 s | 0.66 s |
| `4+11` merged (default) | **242/250 (97%)** | 0.81 s | 0.91 s |
| `3+11` merged | 241/250 | 1.07 s | 1.24 s |

(The single-mode and `3+11` rows were measured one build earlier, before the final bold-heuristic and digit-repair
changes; the default row is the final build. Timings vary with Railway's load: the same default reader measured
1.09 s median on a busier run.)

Of the 8 misses with the default reader, 7 are conservative: the tool asked for a look (REVIEW) on a label the
generator marked clean, because Tesseract dropped the decimal point in a litre volume ("1.5 L" read as "15L") or
misread a capital in the brand line and matched the title-case bottler mention instead. The remaining one is a clean
label failed by a genuine OCR error ("750 mL" read as "790 mL", reported as a MISMATCH that shows the misread text,
so an agent resolves it from the image in seconds). More importantly, no planted defect was reported as a PASS: every wrong
ABV, wrong or missing volume, missing or altered warning and non-capital or non-bold heading was caught, and every
brand-capitalization case was flagged for review (the wrong-brand and wrong-country cases are in the sample set,
where all 15 labels get their expected verdict).

Throughput: the 250-label sample batch completes in about 90 s through the UI with four workers, i.e. 0.35 s per
label of wall-clock time.

**After the code review (October 2026), measured locally** (Tesseract 5.5.3, 11-core Mac, `scripts/bench.py`, one
label at a time; the Railway figures for the current build are in the next table):

| Set | Expected verdict | Planted defects reported as PASS | Median | p95 |
|---|---|---|---|---|
| Samples, before the review | 15/15 | 0 | 1.30 s | 1.35 s |
| Samples, after | 15/15 | 0 | 0.67 s | 0.75 s |
| Batch, before the review | 243/250 | 0 | 1.24 s | 1.33 s |
| Batch, after | 243/250 (same seven labels) | 0 | 0.68 s | 0.74 s |

The review's fixes (country of origin, whole-phrase brand and class/type, label proof consistency, volume parsing,
warning heading, image intake) change no verdict on these two sets: their labels never exercise the cases that were
wrong, which is why each fix comes with its own tests. The speed-up comes from running Tesseract with one OpenMP
thread per process (`OMP_THREAD_LIMIT=1`); with four batch workers the 250-label batch takes 44 s instead of 157 s
on the same machine.

**On Railway after the latest deploy (9 October 2026, `scripts/bench.py --url ...`, one label at a time, so the
figures include HTTP overhead and the extra turned and contrast passes where they run):**

| Set | Expected verdict | Planted defects reported as PASS | Median | p95 | Max |
|---|---|---|---|---|---|
| Samples (15) | 15/15 | 0 | 1.00 s | 1.13 s | 1.13 s |
| Batch (250) | 242/250 | 0 | 0.45 s | 1.08 s | 1.27 s |

All eight batch misses are conservative: a clean label asked for a look (a litre volume read without its decimal
point, or a brand matched in title case). Compared with the 0.81 s median measured on Railway before the code
review, the per-label time halved with `OMP_THREAD_LIMIT=1` (one OpenMP thread per Tesseract process), so that
setting stays; the slowest label, 1.27 s, is one that needed the extra passes.

**Real approved labels (October 2026, local, `scripts/real_labels.py --defects`).** The 20 labels carry 104 filled
application fields:

| | Before the real-label work | Now |
|---|---|---|
| Fields with the expected verdict | 54/104 | 59/104 |
| Flagged for review (NEAR MATCH where MATCH was expected) | not measured | 29 |
| False alarms: MISMATCH or NOT FOUND for text that is on the label | 32 | 16 |
| Government warning: pass / review / fail (all 20 carry it) | 13 fail | 5 / 11 / 4 |
| Planted wrong alcohol content or net contents reported as MATCH | 0 of 40 | 0 of 40 |
| Time per label: median / max | | 2.4 s / 4.3 s |

Read this honestly: no real label passes untouched. 8 come back REVIEW and 12 FAIL, so on real artwork the tool is a
fast first pass that points the agent at what to look at, not an unattended approver. What it does not do is let a
wrong value through: all 40 planted defects were caught. The remaining false alarms come from brand names in
display or curved typefaces, light text over photographs, a handwritten keg collar, and single-digit misreads
(`57.7%` read as `07.7%`); each shows the misread text and its place on the label, so the agent settles it from the
image. Real labels take longer than the synthetic ones (median 2.4 s against under 1 s) because most of them need the
extra turned and contrast passes; the slowest was 4.3 s, inside the 5-second budget on a laptop. Re-measure on
Railway before relying on that there.

**Stress test (`scripts/stress_test.py`, the 15 sample labels per condition, local).**

| Condition | Expected verdict | Planted defect reported as PASS |
|---|---|---|
| DejaVu (baseline), Georgia, Times, Baskerville, Helvetica Neue, Optima, Gill Sans, Avenir Next Condensed | 15/15 each | 0 |
| Futura / Rockwell | 14/15 / 13/15 | 0 |
| Didot (hairline serifs) | 8/15 | 0 |
| Brand in Papyrus, Trattatello or Impact | 15/15 each | 0 |
| Brand in Chalkduster / Herculanum | 12/15 each | 0 |
| Brand in Copperplate | 13/15 | 1, see below |
| JPEG quality 35, half or a third of the resolution | 14/15 each | 0 |
| Tilted 1.5° / 4° (10/15 and 8/15 before straightening was added) | 15/15 / 14/15 | 0 |
| Blur radius 1.2 | 15/15 | 0 |
| Sensor noise (failed by time-out before the optional passes got their own limit) | 14/15 | 0 |

The one defect counted as missed is the brand-capitalization sample ("Stone's Throw Cellars" against "STONE'S
THROW CELLARS"). Copperplate has no lowercase letters: it draws them as small capitals, so the rendered label really
does read in capitals and the planted difference disappears. Every other miss is either a clean label flagged for
a look or a defect caught with a different severity (FAIL where REVIEW was expected, or the reverse).

## Assumptions

- Labels arrive as flat artwork files or straight-on scans, the way they are attached to applications. Scans
  tilted by up to 6° are straightened.
- The application data is trusted; the label is what is being verified.
- "Exact wording" of the warning means the words; punctuation and line breaks are not judged.
- The regulation (27 CFR 16.22(a)(2)) requires the heading in capitals and bold and the rest of the statement not
  bold. The bold check compares the two, so an all-bold statement asks for review; a heading heavier than an
  already-bold body is not detected. Type size (16.22(b)) is not checked: the image does not carry a physical scale.
- A capitalization-only difference matters for the brand name (flagged for review) but not for the other fields.
- English-language labels.

## Known limitations

- Photos taken at an angle, with glare, shadows or poor lighting are out of scope; expect NOT FOUND results and a
  warning-statement failure on such images. Tilt up to 6° is corrected; perspective correction would be next.
- On real approved labels the tool asks for a look on every label (see "Measured results"): brand names in display
  or curved typefaces, light text over photographs, handwriting and single-digit misreads produce false alarms.
- Tesseract struggles with decorative, script or outlined brand typography and with text on busy backgrounds. The
  guided matching tolerates a fair amount of noise, but a brand set in a script face may come back NOT FOUND
  (the cloud reader handles these).
- The bold heuristic is calibrated on synthetic labels and assumes the statement body is in regular weight. A
  real-world calibration set would be needed before trusting it unattended, which is why it only asks for review.
- Batch jobs are kept in memory and disappear on restart; for production they would go to a queue and a database.
- No authentication: the prototype assumes it runs on an internal network.
- Text at 90° is read only when the first, upright read leaves something missing; text at other angles (curved
  around a seal, set diagonally) is not read.
- A "GOVERNMENT WARNING" heading split across two lines is not recognised, so the capitals check fails (a false
  FAIL, never a false PASS).
- The brand's small-print rule assumes the brand is printed larger than the bottler statement. A label whose
  brand is deliberately small next to a large fanciful name gets a NEAR MATCH for the agent to confirm.
- Uploads are size-checked from their Content-Length; a chunked upload without one is only limited per file after
  it arrives. A reverse proxy limit is the production answer.

## Security notes

Uploads are validated (image type, 20 MB and 40 megapixel limits, decompression-bomb and zip member/size limits)
and processed in memory; the app itself never writes them to disk, but the web framework spools any upload over
1 MB to a temporary file in `/tmp` for the duration of the request. Sample file names are whitelisted, the CSV
export neutralises spreadsheet formulas, and the container runs as an unprivileged user. The app makes no outbound
calls unless the cloud reader is enabled.
