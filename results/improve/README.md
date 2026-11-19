# results/improve: second optimisation round

Each folder has its own `REPORT.md` (method, tables, caveats). All plans were re-verified on the full
grid with `src/verify.py`. "Certified" means every subset was resolved; everything else is best found.

| Folder | Scripts / module | What it holds |
|---|---|---|
| `bao18/` | `06a_*.py`, `src/bao_search.py` | 18° grid (20 candidates): K=3 certified at 60.789; multi-start swap search for K=4…6 |
| `dvc/` | `06b_*.py`, `src/dvc_improve.py` | rectal V50 limit on equispaced beams: 17% → 15.76%; proven floor 0.85% → 1.22%. `old/` has superseded attempts; the `0.15` row of `sweep_stA.json` is invalid |
| `fine/` | `06c_*.py`, `src/fine_search.py` | first fine-angle (2°) descent: K=5 → 41.632 |
| `cert18/` | `07a_enumerate18.py` | K=4 certified on the 18° grid at 49.263 (4,845 subsets, 0 unresolved); K=5 sample only |
| `fine2/` | `07b_*.py` | second fine-angle round: K=6 → 37.736, K=3 → 60.181; K=4, 5 confirmed local optima |
| `dvc2/` | `07c_*.py` | rectal limit sweep on the certified and best-found 5-beam sets down to 12.75% |

Final figures and the comparison table are made by `scripts/08_final_figures.py` into `results/`.
