# Label Check (ttb-label-verifier)

FastAPI + Jinja prototype that verifies an alcohol label image against its label approval application: OCR with Tesseract, per-field verdicts, government-warning checks, batch mode with CSV export. Details and thresholds are in `README.md` and `app/config.py`.

## Run and test

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
pytest -q
```

Tesseract must be on PATH (or set `TESSERACT_CMD`). Templates: `app/templates/` (`base.html`, `index.html`, `review.html` (review queue), `report.html` (printable reports), `partials/result.html`, `partials/batch_status.html` (table, detail, export dialog), `partials/error.html`, `partials/decide.html`). Styles: `app/static/style.css` (tokens, components, phone layouts, print). Behaviour: `app/static/app.js`. No build step, no framework, no external assets. Static links carry a content stamp (`?v=`), so a browser never runs a stale script after a redeploy. Every screen must also work without JavaScript: an inline script in `base.html` adds a `js` class to `<html>`; without it `style.css` shows both panels and hides `.js-only` controls, and routes answer requests without the `X-Partial` header with whole pages (`main.is_partial`).

Real-label testing: `scripts/real_labels.py` scores 20 approved labels from the public registry (images in the gitignored `data/real/`, fetched with `scripts/fetch_registry_labels.py`); `scripts/stress_test.py` renders the samples in other typefaces and degraded images. Numbers are in the README under "Measured results".

Decision log: with `DECISION_LOG=<file>` every review answer and undo is appended as one JSON line (`app/decision_log.py`, never the image); `scripts/decisions_report.py LOG [--csv OUT]` turns it into pass rates per rule and a calibration CSV in the `scripts/real_labels.csv` layout (README, "Learning from decisions").

## UI redesign spec

Before any change under `app/templates/` or `app/static/`, read `design/README.md`. It points to the 23 design boards in `design/boards/`, the tokens in `design/tokens.css`, the implementation order, and the interactive canvas (https://claude.ai/artifact/QRoVDnra5hD8R6jGh6JAk3), which a session can also read with the Artifact tool. Match the boards' markup, copy and tokens rather than inventing new ones.

## Naming constraint

This is an unaffiliated prototype. Never add "Department of the Treasury", "TTB", "Alcohol and Tobacco Tax and Trade Bureau", a seal, or agency-style attribution to the UI, titles, or footer (31 U.S.C. § 333). The product name is "Label Check". Regulation citations (27 CFR parts 4, 5, 7 and 16) are fine; "government warning" is the regulation's own term and stays. The owner also asked that the acronym "COLA" never appear in the UI or docs (the design boards predate this): say "label approval application" instead.
