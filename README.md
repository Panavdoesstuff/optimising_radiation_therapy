# IMRT plan optimisation on the CORT prostate case

Course project (Optimisation). Two formulations on the same real data:

- **Easy (LP):** fixed 5 equispaced beams; minimise weighted mean dose to rectum, bladder and
  body subject to target coverage and a dose cap.
- **Hard (MILP), headline: beam angle selection.** Choose at most K of 20 candidate beams
  (binary per beam). Real machines deliver few beams, so this is the more faithful model.
- **Hard (secondary): dose-volume constraint** on the rectum (binary per voxel). The exact MILP
  could not be solved at this size; the result is a feasible plan from a labelled
  approximation, bracketed by a valid lower bound.

See `results/SUMMARY.md` for the findings and `BUILDMAP.md` for the original brief.

## Install
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

`highspy` is optional. If installed, it is used for the MILPs because it accepts a warm start;
otherwise `scipy.optimize.milp` (same HiGHS solver) is used through the same interface
(`src/solvers.py`).

## Data
The CORT dataset (GigaDB 100110, Craft et al. 2014) is read-only and never committed (~2 GB).
Default location: `~/Downloads/{PROSTATE,TG119}`. Override with `CORT_DATA_ROOT=/path`.
All paths, prescriptions and weights are in `src/config.py`.

## Run order (from the repo root, no arguments)

| Script | What it does | Time |
|---|---|---|
| `scripts/00_inspect_data.py` | Lists files, confirms beam counts, beamlet totals, grid sizes | <1 min |
| `scripts/01_sanity_checks.py` | Reproduces the paper's "ones" doses and LPs → `results/sanity_checks.csv` | ~1 min |
| `scripts/02_run_easy.py` | Easy LP, verified → `results/easy_solution.npz`, `results.json["easy"]` | ~1 min |
| `scripts/03_run_hard.py` | DVC plan + beam selection sweep K=3..9 (600 s limit each), verified. Resumable | ~1.5 h |
| `scripts/04_analysis.py` | Figures and tables from the saved solutions | ~2 min |
| `.venv/bin/python -m pytest tests` | TG119 loader tests + synthetic beam-selection tests | <10 s |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `sanity_checks.csv` | Published vs computed numbers (Checks A and B) |
| `results.json` | Parameters, solver status, verification (constraint violations, recomputed objective, key variables) for every plan |
| `bao_sweep.json` | Beam selection per K: plan, chosen angles, best bound, gap, verification |
| `fig1_dvh.png` | Dose-volume histograms, easy vs hard |
| `fig2_dose_maps.png` | Axial dose through the PTV centre, plus hard − easy difference maps |
| `fig3_fluence.png` | Beamlet intensities per beam |
| `fig4_price_of_realism.png` | Objective per plan, split into rectum / bladder / body terms |
| `fig5_sweep_K.png` | Objective vs number of beams K, with proven bounds |
| `model_summary.csv` | Size, solve time, status and gap of every model |
| `dose_table.csv` | Per-structure min / mean / max, D95 (targets), V30–V70 (organs) |
| `logs/` | Raw logs of the exact DVC MILP attempts and feasibility scans |

## Layout
    src/config.py      paths, beam sets, prescriptions, weights
    src/io_cort.py     .mat loaders (scipy, with h5py fallback for v7.3), grid parser
    src/problem.py     stacked dose matrix, structure sets (raw / disjoint), cache
    src/easy_lp.py     easy LP and the shared model builder
    src/hard_bao.py    beam angle selection MILP, heuristic, bounds
    src/hard_dvc.py    dose-volume MILP, fix-and-resolve, CVaR, row generation
    src/solvers.py     MIP backend (scipy.milp or highspy)
    src/verify.py      independent feasibility / objective checks
    src/plots.py       figures
