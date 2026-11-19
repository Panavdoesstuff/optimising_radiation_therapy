# IMRT plan optimisation on the CORT prostate case

Course project (Optimisation). Choose beamlet intensities `x` so the tumour gets its prescribed dose
(`d = D x`) while the rectum, bladder and the rest of the body get as little as possible.

- **Easy problem (LP):** 5 fixed, equispaced beams. Minimise weighted mean dose to the rectum,
  bladder and body, subject to target coverage (68 / 56 Gy) and a 74.8 Gy cap on the targets, the
  organs and **all normal tissue** (the body cap is enforced by row generation).
- **Hard problem (integer): beam angle selection.** Choose at most K beams from a candidate set
  (one binary per beam). Solved **exactly by enumeration** on a 10-beam grid (every 36°): every
  K-subset is solved as an exact LP and verified, so the optimum is certified with zero gap.
  A big-M MILP was tried first; its relaxation collapses to the all-beam LP, so it never closed the
  gap (kept as evidence: `results/logs/`, `results/bao_sweep.json`).
- **Second round (`results/improve/`):** pruned enumeration certifies the best 3 and 4 beams on a finer
  18° grid; a fine-angle (2°) local search finds uncertified plans at **−20.8%** (5 beams) and
  **−28.2%** (6 beams) against the easy plan; the rectal limit is pushed to 12.75% on the good beams.
- **Realism on fixed beams:** a rectal dose-volume limit (≤17% of the rectum above 50 Gy) by
  fix-and-resolve, bracketed by the LP, plus a provable floor on how far that limit can go.

Findings and numbers: `results/SUMMARY.md` (first-round version: `results/SUMMARY_v1.md`).

## Install
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

`highspy` is used for warm-started LPs and the MILP warm start; without it the code falls back to
`scipy.optimize` (same HiGHS engine), more slowly.

## Data
CORT dataset (GigaDB 100110, Craft et al. 2014); read-only, never committed (~2 GB).
Default location `~/Downloads/{PROSTATE,TG119}`; override with `CORT_DATA_ROOT=/path`.
Paths, prescriptions, weights and candidate grids are in `src/config.py`.

## Run order (from the repo root, no arguments)

| Script | What it does | Time |
|---|---|---|
| `scripts/00_inspect_data.py` | Confirms beam counts, beamlet totals, grid sizes | <1 min |
| `scripts/01_sanity_checks.py` | Reproduces the paper's "ones" doses and LPs → `sanity_checks.csv` | ~1 min |
| `scripts/02_run_easy.py` | Easy LP with the body cap, verified → `easy_solution.npz` | ~2 min |
| `scripts/03_run_hard.py` | Default: re-verifies all saved plans (`05_verify_all.py`). `RERUN=1`: rectum DVC plan + capped beam-selection MILP for K=3..5 | 5 min / ~5 h |
| `scripts/03b_enumerate.py` | Exact beam selection: all 792 subsets for K=3..6 on the 36° grid, 4 workers, resumable | ~1.5 h |
| `scripts/03c_dvc_floor.py` | Provable floor on the rectal V50 + limit sweep | ~2 h (budgeted) |
| `scripts/03d_swap_final.py` | Swap search on the 18° grid (uncertified) + final plan (best 5 + cap + rectum limit) | ~30 min |
| `scripts/06a_*.py`, `06b_*.py`, `06c_*.py` | Second round: 18° grid certification and swap search (06a), rectal-limit search and voxel floor (06b), fine-angle search (06c) | hours each |
| `scripts/07a_enumerate18.py`, `07b_*.py`, `07c_*.py` | K=4 certificate on the 18° grid (07a), fine-angle round 2 (07b), rectal limit on the good beams (07c) | hours each |
| `scripts/08_final_figures.py` | Re-verifies the headline plans, makes Fig 7–10 and `final_comparison.csv` | ~3 min |
| `scripts/04_analysis.py` | Figures and tables from saved solutions | ~5 min |
| `scripts/05_verify_all.py` | Re-verifies every saved plan on the full grid → `verification_summary.csv` | ~5 min |
| `.venv/bin/python -m pytest tests` | Loader tests (TG119) + synthetic tests (row generation, MILP relaxation, voxel floor) | <10 s |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `SUMMARY.md` | Findings, numbers, limitations, changes from the brief |
| `sanity_checks.csv` | Published vs computed numbers |
| `results.json` | Parameters, solver status and verification for the main plans |
| `enum/K{3..6}.jsonl`, `enum/best_K*.npz` | Every enumerated subset (objective, verification) and the optimum per K |
| `bao_sweep.json` | Capped MILP sweep K=3..5 (bounds, gaps) |
| `swap_final.json` | Swap search and final plan |
| `dvc_floor.json` | Proven-forced rectal voxels and the limit sweep |
| `fig1_dvh.png` … `fig6_rectum_distance.png` | DVHs, dose maps, fluence, price of realism, price of a beam budget, rectum vs PTV distance |
| `fig7_beam_budget_final.png` … `fig10_organ_means.png`, `final_comparison.csv` | Final round: certified vs best-found beam budget, DVHs by structure, rectal-limit trade-off, organ mean doses, metrics table |
| `improve/` | Second-round runs, one folder per search, each with a `REPORT.md` (index: `improve/README.md`) |
| `model_summary.csv`, `dose_table.csv`, `before_after_body_cap.csv`, `verification_summary.csv` | Tables |
| `no_body_cap/` | The earlier run without the normal-tissue cap (superseded; kept for the comparison) |
| `logs/` | Raw logs, including the failed exact MILP attempts |

## Layout
    src/config.py      paths, beam sets, prescriptions, weights, candidate grids
    src/io_cort.py     .mat loaders (scipy, with h5py fallback for v7.3), grid parser
    src/problem.py     stacked dose matrix, structure sets (raw / disjoint), cache
    src/easy_lp.py     LP model builder, body-cap row generation
    src/hard_bao.py    beam selection: reduced matrices, exact LP with all-row generation, MILP, heuristics
    src/hard_dvc.py    dose-volume limit: fix-and-resolve, provable voxel dose floor
    src/bao_search.py  pruned enumeration of beam subsets (cutoff on the reduced LP's bound)
    src/fine_search.py fine-angle (2°) coordinate descent
    src/dvc_improve.py rectal-limit search: elastic swap on a persistent LP
    src/solvers.py     MIP backend (scipy.milp or highspy)
    src/verify.py      independent full-grid feasibility / objective checks
    src/plots.py       figures
