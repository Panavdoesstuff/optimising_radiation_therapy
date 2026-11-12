# IMRT plan optimisation on the CORT prostate case

Easy (LP) vs hard (dose-volume constraints, MILP) formulations. See BUILDMAP.md for the brief.

## Install
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Data
CORT dataset (GigaDB 100110) is read-only and not committed. Default location is `~/Downloads/{PROSTATE,TG119,codeSnippetsCortDataset}`;
override with `CORT_DATA_ROOT=/path`. Paths are set in `src/config.py`.

## Run order (from repo root)
    .venv/bin/python scripts/00_inspect_data.py     # confirms beam counts / grid sizes
    .venv/bin/python scripts/01_sanity_checks.py    # reproduces published numbers -> results/sanity_checks.csv
    .venv/bin/python -m pytest tests                # fast tests on TG119
(further scripts added as phases complete)
