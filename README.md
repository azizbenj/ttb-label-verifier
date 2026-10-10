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
| `DECISION_LOG` | unset | JSON-lines file that records every review decision with what the tool had concluded about the label, never the image (see "Learning from decisions") |
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
| `app/decision_log.py` | The decision log: one JSON line per review answer and undo when `DECISION_LOG` is set (never the image) |
| `app/batch.py` | CSV parsing, image/zip intake, thread-pool jobs, CSV export |
| `app/main.py` + `templates/` + `static/` | FastAPI routes, server-rendered UI (no build step, no CDN, no external assets); `review.html` is the review queue, `report.html` the printable reports |
| `design/` | The UI design spec: 23 boards, tokens, the implementation order |
| `scripts/generate_labels.py` | Synthetic label generator (Pillow) |
| `scripts/bench.py`, `scripts/stress_test.py` | Accuracy and timing on the synthetic sets; the same labels in other typefaces and degraded images |
| `scripts/fetch_registry_labels.py`, `scripts/real_labels.py`, `scripts/real_labels.csv` | Real approved labels: download, ground truth, field-level scoring |
| `scripts/decisions_report.py` | Pass rates per rule and a calibration CSV from the decision log |
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
| `conflict_floor` | 70 | Another reading of the same place on the label (a second OCR pass or view) at least this similar to the value but saying something else turns a MATCH on the brand, class/type or bottler into a **NEAR MATCH** quoting both readings |
| `abv_tolerance` | 0.05 pp | Alcohol content compared as numbers; any larger difference is a **MISMATCH** |
| `proof_tolerance` | 0.5 proof | A label whose proof disagrees with its own percentage by more than this ("45% (80 Proof)") → **NEAR MATCH** |
| `volume_tolerance_ml` | 0.5 mL | Net contents compared in millilitres (so `12 FL OZ` = `355 mL`). Between a US figure and a metric one 0.5% is allowed (`750 mL / 25.4 FL OZ`); within one system the figures must agree (`753 mL` is not `750 mL`). Same digits and unit but no decimal point on the label (`15 L` read for `1.5 L`) → NEAR MATCH, never a silent match |
| `warning_locate` | 75 | Similarity needed to recognise the "GOVERNMENT WARNING" line |
| `warning_near` | 97 | Warning wording at or above this (but not exact) → NEEDS REVIEW with a diff; below → FAIL |
| `bold_ratio` | 1.30 | Heading stroke width ÷ body stroke width at or above this → "looks bold" (measured: bold headings 1.51-2.04, regular 1.03-1.14) |
| `bold_min_text_px` | 14 | Below this text height the stroke measurement is not attempted |
| `bold_failure_is_fail` | false | A "does not look bold" result asks for review instead of failing the label |
| `same_place_overlap` | 0.5 | Two OCR lines whose boxes overlap by this fraction of the smaller one's height and width are two readings of one printed line of the warning: only one is taken |
| `column_tolerance` / `column_min_lines` | 0.25 / 2 | A word whose box ends before the warning's left edge or starts after its right edge (allowing this fraction of the text height) is outside its column; the edge counts as a neighbouring column, and such words are set aside and quoted, only when they occur on at least this many of the statement's rows |
| `duplicate_sentence` | 90 | Lines outside the warning that match one of its sentences at least this well (partial similarity) are that sentence printed again → NEEDS REVIEW, quoting them; a complete second statement is not reported |
| `unreadable_min_words` / `unreadable_word_conf` | 8 / 70 | Fewer clear words than this, nothing read for any field (not even a disagreeing value) and no warning → "We couldn't read this label", no verdict |

Real labels taught a few more rules, each covered by tests:

- **Every reading is considered.** A label often states the alcohol content or the volume more than once
  (`750 mL` on the front, `75 cl` on the back), and OCR may produce two readings of one statement. If any
  reading agrees with the application and another one disagrees, the result is a NEAR MATCH that lists both.
  A reading that only lost its decimal point is treated as the same statement, and US and metric figures within
  0.5% of each other (`16 FL OZ` and `473 mL`) agree. The same holds for the brand, class/type and bottler: a
  MATCH found in one reading while another reading of the same place says something else ("BARK BREW" printed,
  one pass reading "BARN BREW") is a NEAR MATCH. Readings that differ only by OCR's usual letter confusions
  (`rn`/`m`, `0`/`O`, `5`/`S`...) or are cut short do not count as disagreeing.
- **A percentage is only an alcohol content when the label says so.** It must sit next to `Alc./Vol.`, `ABV`,
  `alcohol by volume` or a proof figure (OCR slips such as `ALG/VGL` included; "alcoholic" in the warning and
  "volcanic" are not). A matching figure without such a word ("Blend: 13.5% Petit Verdot") is a NEAR MATCH,
  never a MATCH.
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

1. **Present**: the statement (or its body) was found on the label. The OCR lines are grouped by *frame*: the
   upright read and the contrast read of the same image share one frame (their boxes are in the same coordinates,
   so a heading read in one and a body read in the other still make one statement); each turned read is a frame of
   its own. Within a frame the lines are ordered top to bottom and the statement is grown line by line while that
   brings it closer to the required text, passing over up to four rows that belong to something else ("For sale
   only in Ohio"). The word boxes then decide what belongs to it:
   - *Two readings of one printed line.* The block pass and the sparse pass often read the same row differently
     (one across two columns, the other each column; one with a stray mark). Two lines whose boxes overlap by half
     their height and width are two readings of one row: the one closer to the required text is kept, never both
     (equal readings: the one OCR was surer of). A duplicate read can no longer add a sentence fragment twice.
   - *A neighbouring column.* The words of the provisional statement that align with the required text define the
     statement's own left and right extent across all its lines. A word whose box lies entirely beyond that extent
     is outside the column; when words lie beyond the same edge on at least two of the statement's rows, that
     edge borders another column (a keg collar's "ATTENTION-READ BEFORE TAPPING", a "12 FL OZ" to the left) and
     all words beyond it are set aside and quoted in the note without affecting the verdict. The gap between the
     columns plays no part (on a real keg collar it is a normal word gap); only the extent does. A single line with
     words sticking out is left alone and judged as wording: they may be words added to the statement. Stray
     marks with no letters beyond the extent ("~", "|", a "4" at the edge of a can) are dropped silently.
   - *No boxes* (the cloud reader, or a test): words at the start or end of a line that belong beside the
     statement are left out on wording alone, and the wording result then asks for a look, quoting them, because
     their position cannot tell a neighbouring column from words added to the statement.
   - A heading split over two lines ("GOVERNMENT" / "WARNING:") is one heading: both lines are the heading for the
     capitals check and both words are measured for the bold check.
2. **Wording**: compared word for word after normalization. Punctuation is ignored because OCR drops commas and
   periods unreliably, and a word the label hyphenates over a line break ("SUR-" / "GEON") is rejoined when the
   join spells a word of the required text. Any difference is listed as "required text says / label says". A
   difference with similarity ≥ 97 is flagged for **review** (it may be a misprint or an OCR error; the diff lets
   the agent decide in a second); anything larger (missing sentence, paraphrase) **fails**. The statement must
   carry each sentence once, in order: sentence (1) printed twice inside the statement, or (2) before (1), fails;
   a sentence printed again elsewhere on the label (sentence (1) on the front and again on the back, without (2))
   asks for a look and quotes the second copy, while a complete second statement is not a problem. When a
   discarded reading of a statement line disagrees on a word that is not in the required text and OCR was at
   least as sure of it, the wording asks for a look and names both readings: a misread that happens to agree
   with the required text must not hide a misprint the other reading saw.
3. **Heading in capitals**: the OCR text of the heading must read `GOVERNMENT WARNING:`; title case fails, a missing
   colon asks for review, and capitals with a letter OCR could not read cleanly (`WARNlNG`, or `ERNMENT` cut at the
   image edge) ask for review. A different word that was read clearly (`HEALTH WARNING:`, `GOVT WARNING:`) fails.
4. **Heading bold (heuristic)**: from the word boxes, we take the heading words (read cleanly, glued to a
   neighbour such as `WARNING:(1)`, or misread the way the capitals check tolerates, `ERNMENT WARMING:`) and the
   body words of the statement, measured in the ink of the view that read the statement, and estimate each
   group's mean stroke width as 2 × ink area ÷ ink perimeter on the binarized image (for a stroke
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

#### Learning from decisions

Set `DECISION_LOG=/path/decisions.jsonl` and every answer from the review queue (and every undo) is appended
to that file as one JSON object per line, with what the tool had concluded about the label: the job, the
batch's source name, the application id and image name, the decision (`pass`: the label is fine and our flag
was a false alarm, `fail`: the label is wrong, `skip`, `clear`: undone), the overall verdict, every question
the label raised (key, what, verdict, question, the two readings) with the one the queue asked first marked,
each field's verdict / expected / found / note, the four warning statuses and the wording score, the read
confidence, the words read, the reader and the timings. Never the image, its preview or the text read from
it. Writes are append-only behind a lock (the server's workers share it; keep one process per log file), and
a write that fails (a missing directory, a full disk) is a warning in the server log, never an error for the
agent. Off when the variable is unset.

`python scripts/decisions_report.py decisions.jsonl` keeps the last decision per job and application (an
undone one drops out) and prints, per question (`brand_name`, `net_contents`, `warning_wording`,
`warning_bold`...), how many labels raised it, how many times the queue asked it first, the pass / fail / skip
answers and the pass rate: pass ÷ (pass + fail), the share of that rule's flags the agents found to be false
alarms. A rule with a high pass rate costs agents time for nothing; one with a low pass rate catches real
problems. Then the ten most frequent notes behind `pass` answers and the ten behind `fail`. `--csv
calibration.csv` writes one row per label answered pass or fail in the layout of `scripts/real_labels.csv`
(the application id as `ttbid`, the application values, `<field>=review` in `expect` when the agent confirmed
a NEAR MATCH, nothing after a `fail`, the decision and the question in `notes`), ready for
`scripts/real_labels.py` once the images are in `data/real/`. The script needs only the standard library, so
it runs on a copy of the log anywhere.

### Error states

Errors are inline, where the result would have been, and say what happened in one line and what to do in the
next. Missing fields are checked in the browser before anything is sent, each input is marked and the
summary links to it; without JavaScript the server returns the same page with the typed values kept. The drop
zone itself becomes the error for a missing, unreadable or oversized image, with the file name quoted. A CSV
with the wrong headers gets a table of expected header vs the near-miss header in the file. An image that
opens but yields almost no text ("fewer than 8 confident words and nothing read for any field") gets a grey,
verdict-free card with what usually fixes it, never a FAIL; a label where a value was read and disagrees is a
normal FAIL however few words it has. The page works at 1280, 768 and 390 px without horizontal scrolling; on a
phone the batch table becomes a list.

Every screen also works without JavaScript: both tabs show one under the other, the forms post and come back as
whole pages, a running batch's page (`/batch/<id>`) reloads itself every 3 seconds until it is done, each row
links to its full comparison, and the review queue's answers are forms.

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
- `test_decisions.py`: review prompts (Yes always means the label is fine), decisions kept with a batch, the review queue (with and without JavaScript), export scopes and column groups, printable reports, the shared batch link
- `test_decision_log.py`: the decision log (one compact line per answer and undo with the documented keys, never the image, concurrent writes, an unwritable path never fails the request) and the report script (the last decision wins, pass rates per question, the notes behind the answers, the calibration CSV in the `real_labels.csv` layout)
- `test_nojs.py`: the batch flow as a browser without JavaScript runs it: post, redirect, self-reloading progress page, row links, whole-page errors
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

**On Railway (`scripts/bench.py --url ...`, one label at a time, so the figures include HTTP overhead and the
extra turned and contrast passes where they run).** Two deploys measured a day apart:

| Build | Set | Expected verdict | Planted defects reported as PASS | Median | p95 | Max |
|---|---|---|---|---|---|---|
| `4c88e85` (before the second review), 8 Oct 2026 | Samples (15) | 15/15 | 0 | 1.00 s | 1.13 s | 1.13 s |
| | Batch (250) | 242/250 | 0 | 0.45 s | 1.08 s | 1.27 s |
| `5e3e150` (after the second review), 9 Oct 2026 | Samples (15) | 15/15 | 0 | 2.5 s | 3.2 s | 3.6 s |
| | Batch (250) | 242/250 | 0 | 1.02 s | 2.8 s | 4.6 s |

The verdicts are identical between the two builds (the same eight conservative misses: a clean label asked for a
look because a litre volume was read without its decimal point, or the brand was matched in title case). The
times are not: on 9 October every label took about twice as long, including the four clean samples, whose two
Tesseract passes took 1.0 s instead of 0.45 s and which run none of the review's new logic. The review's changes
add no OCR pass (the extra passes run on the same 10 of 15 samples as before) and measured faster locally, so
the difference is the shared vCPU on that day; the README's earlier note that the same build measured 0.81 s and
1.09 s on two runs is the same effect. What the slow day does show: a label that needs the extra passes costs
2.4-3.5 s there, and the slowest batch label reached 4.6 s, close to the 5-second budget. If that recurs, the
options are a container with a guaranteed CPU, or fewer extra passes (`TESSERACT_PARALLEL_PASSES`, and which
views `extend` tries), which is a product trade-off rather than a bug. Compared with the 0.81 s median measured
on Railway before the first code review, `OMP_THREAD_LIMIT=1` (one OpenMP thread per Tesseract process) halved
the per-label time on the quiet day, so that setting stays.

**Real approved labels (October 2026, local, `scripts/real_labels.py --defects`).** The 20 labels carry 104 filled
application fields:

| | Before the real-label work | Now |
|---|---|---|
| Fields with the expected verdict | 54/104 | 57/104 |
| Flagged for review (NEAR MATCH where MATCH was expected) | not measured | 29 |
| Accepted without the look the ground truth expects (MATCH where NEAR MATCH was expected) | not measured | 2 |
| False alarms: MISMATCH or NOT FOUND for text that is on the label | 32 | 16 |
| Government warning: pass / review / fail (all 20 carry it) | 13 fail | 5 / 11 / 4, then 8 / 9 / 3 with the layout rules below |
| Planted wrong alcohol content or net contents reported as MATCH | 0 of 40 | 0 of 40 |
| Time per label: median / max | | 2.3 s / 4.3 s |

(Until the second code review the scorer counted a MATCH where the ground truth expects a look as "expected",
which read 59/104. The two such fields are the class "GIN" printed under "THE SPIRIT OF TENNESSEE" and the country
"MEXICO" read from "HECHO EN MEXICO"; both are defensible readings, so they are reported rather than changed.)

Read this honestly: no real label passes untouched. 8 come back REVIEW and 12 FAIL, so on real artwork the tool is a
fast first pass that points the agent at what to look at, not an unattended approver. What it does not do is let a
wrong value through: all 40 planted defects were caught. The remaining false alarms come from brand names in
display or curved typefaces, light text over photographs, a handwritten keg collar, and single-digit misreads
(`57.7%` read as `07.7%`); each shows the misread text and its place on the label, so the agent settles it from the
image. Real labels take longer than the synthetic ones (median 2.4 s against under 1 s) because most of them need the
extra turned and contrast passes; the slowest was 4.3 s, inside the 5-second budget on a laptop. Re-measure on
Railway before relying on that there.

**Government warning, layout rules (October 2026).** Before the word-box rules in "Government warning check" the
warning came back 5 pass / 11 review / 4 fail on the 20 real labels, every one of which carries a correct warning;
after them 8 / 9 / 3, with no change to the field verdicts (the old and the new warning code were scored back to
back through the same scorer and images and gave the same 56/104, 27 flagged, 19 false alarms, 2 accepted; the
scorer's own inputs had moved since the table above), the sample set (15/15), the batch set (243/250, the same
seven misses, 0 planted defects passed), the stress table or the 40 planted real-label defects. The warning check
can only reach the fields through the pipeline's decision to run the extra OCR passes, and on these labels that
decision changes for one label whose field verdicts are the same either way. What changed, label by label:

- Three wine and spirits labels whose two OCR passes had read the same row differently (one with a stray mark from
  the artwork, one clean) went from REVIEW to PASS: the duplicate readings are no longer both assembled, so the
  wording is exact (a condensed wine warning, a rosé, a letter-spaced whiskey where the wording is now exact but the
  type is too small to measure the bold heading, so it stays REVIEW for that reason).
- A beer can whose last line started with a stray "4" outside the statement's left edge: REVIEW to PASS.
- A gin label hyphenating "SUR-GEON" and "BEVER-AGES" over line breaks: wording REVIEW to PASS (8 px type, so the
  bold check still asks for a look).
- The handwritten keg collar: FAIL to REVIEW. The neighbouring column ("ATTENTION-READ BEFORE TAPPING ... THIS KEG
  MAY RUPTURE ...") is now set aside by its boxes and quoted; what remains is a genuine reading difference on the
  first line ("(1)" lost, or "TO" read as "10", depending on which of two readings is kept), which the agent must
  look at.
- A tequila label went from 3 reading differences to 1 (same REVIEW). The rest are unchanged: two labels whose
  warning the OCR did not read at all (light text over a photo, a slanted panel), one whose 6 px body is unreadable,
  three REVIEWs for "to" read as "10"/"T0" or a one-word last line that the turned-view reader drops, one cropped at
  the image edge, and three honest bold measurements (1.05, 1.26 and 1.27 against the 1.30 threshold; one of them
  really is set in the same weight as its body).

**After the second code review (October 2026, local, Tesseract 5.5.3):** samples 15/15, median 0.87 s; batch
243/250 with the same seven misses, 0 planted defects reported as PASS; real labels as in the table above; stress
test identical to the table below. None of the round-two fixes changed a verdict on these sets: the cases they
close (a blend percentage taken for the ABV, a misread that agrees with the application, a clearly read wrong
heading word, an unreadable card hiding a mismatch, 753 mL against 750 mL) do not occur in them, which is why each
fix has its own tests.

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
- A warning statement printed in two side-by-side panels (sentence (1) left, sentence (2) right) interleaves in
  top-to-bottom order and fails as wording; one above the other, or wrapped around a seal, assembles correctly.
- A one-word last line of a sideways warning ("PROBLEMS.") is dropped by the turned-view reader, which keeps only
  lines of two or more words, so that label asks for a look over a missing word that is on the label.
- The brand's small-print rule assumes the brand is printed larger than the bottler statement. A label whose
  brand is deliberately small next to a large fanciful name gets a NEAR MATCH for the agent to confirm.
- Uploads are size-checked from their Content-Length; a chunked upload without one is only limited per file after
  it arrives. A reverse proxy limit is the production answer.
- A letter-spaced brand is compared with its spaces closed up, so "B A R N O N E" matches both "BARN ONE" and
  "BAR NONE": OCR does not keep the wider gap between words.
- Two readings of the same place that disagree are only flagged when OCR was at least as confident of the one
  that disagrees; a label misprint that OCR also reads with less confidence than a misread of it would pass.
- A warning statement printed in two halves (the first sentence on the front, the second on the back) is
  assembled into one and can pass; each sentence must appear once and in order, and a sentence printed again
  elsewhere is reported, but the regulation's requirement that the statement be printed as one unit is not judged.
- Words beyond the statement's column on two or more of its rows are taken for a neighbouring column and set
  aside (quoted, not judged). A label that deliberately added words to the statement on two lines, each sticking
  out past every other line, would be read the same way; the quote in the note is the agent's guard.
- Single-label checks and batch jobs share one pool of four Tesseract processes for the second pass and the
  optional passes; measured locally, a single check took 0.6-0.7 s during a 250-label batch against 0.4-0.7 s
  idle, but a smaller container will queue more.

## Security notes

Uploads are validated (image type, 20 MB and 40 megapixel limits, decompression-bomb and zip member/size limits)
and processed in memory; the app itself never writes them to disk, but the web framework spools any upload over
1 MB to a temporary file in `/tmp` for the duration of the request. Sample file names are whitelisted, the CSV
export neutralises spreadsheet formulas, and the container runs as an unprivileged user. The app makes no outbound
calls unless the cloud reader is enabled.
