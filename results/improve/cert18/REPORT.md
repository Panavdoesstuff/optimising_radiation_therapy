# Certification on the 18-degree grid (20 candidates), K=4 and K=5

Script: `scripts/07a_enumerate18.py` (pruned enumeration, same row-generation LP as `src/bao_search.py`; every reduced LP is a
relaxation, so a reduced-LP optimum above the cutoff is a valid lower bound and prunes the subset). Logs/records in this folder.

## K=4: CERTIFIED
- Cutoff 49.262601 (best on-grid plan 54,126,180,270 + 1e-6; the off-grid 47.717 plan was NOT used).
- All C(20,4) = 4845 subsets processed (4845 distinct records in `K4.jsonl`): **4844 pruned, 1 solved, 0 infeasible, 0 unresolved**
  (so no retry was needed).
- The one solved subset is 54,126,180,270, objective **49.262593** (full-grid `verify.verify`, run inside the worker: verified=True,
  max relative violation 1.8e-11, max body dose 74.8 Gy). Plan saved in `best_K4.npz`.
- Every other subset has LP value >= its recorded lb; smallest lb among pruned: 49.26386 ([54,90,162,198]), 49.26816, 49.27132.
  Nothing solved below the cutoff, so **the certified K=4 optimum on the 18-degree grid is 49.2626 at 54,126,180,270**
  (the earlier best was optimal). The nearest rival is only 0.0013 above it.
- Run time: 171 min wall on 4 workers (41,017 CPU-s; machine load average 11-17 from other jobs inflated it).

## K=5: NOT certified, NOT run (estimate too large)
- Random sample of 300 of 15,504 subsets (seed 1), cutoff 44.203201: 300 pruned, 0 solved, 0 infeasible, 0 unresolved.
  Prune rate 100% in the sample; 9.22 CPU-s per subset, 11.6 min wall on 4 workers (2.3 s wall/subset, load ~14).
- Extrapolation: 15,504 x 2.3 s = about 10 h wall (about 40 CPU-hours); at the best observed K=4 rate (about 40 subsets/min)
  still more than 6 h. Over the 3 h threshold, so the full K=5 enumeration was not started. K=5 best known (44.2032 at
  90,126,180,198,234) remains "best found" only. The sample is not a certificate.

## K=6: not attempted (K=5 not done).
