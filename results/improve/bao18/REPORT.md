# Beam selection on the 18-degree grid (20 candidates): report

Objective: weighted mean dose (minimise), 74.8 Gy body cap. Lower bound for every K: all-20-beam LP = 32.67.
Every plan below was re-verified on the full grid with `src/verify.py` (all checks pass).

## Results

| K | Old best | New best | Beams (deg) | Certified? | Gap to 32.67 | Max rel. violation |
|---|---|---|---|---|---|---|
| 3 | 60.789 | **60.7891** | 90, 180, 270 | **YES** (all 1140 subsets) | 46.3% | 2.6e-12 |
| 4 | 52.931 | **49.2626** | 54, 126, 180, 270 | no (best found) | 33.7% | 1.8e-11 |
| 5 | 44.203 | **44.2032** (unchanged) | 90, 126, 180, 198, 234 | no (best found) | 26.1% | 5.8e-12 |
| 6 | none (36-degree certified: 44.066) | **42.0274** | 18, 72, 108, 144, 180, 270 | no (best found; run cut by the time budget) | 22.3% | 7.4e-11 |

Gap = (objective - 32.67) / objective. Max body dose 74.8 Gy in all four plans. Plans: `best_K3_enum.npz`,
`best_K4_search.npz`, `best_K5_search.npz`, `best_K6_search.npz` (x and angles).

## 1. K=3 certificate (`scripts/06a_enumerate18.py`, `K3.jsonl`)
All C(20,3)=1140 subsets were processed: 1042 pruned, 98 infeasible, 0 solved-with-cutoff, **0 unresolved**.
- Infeasible: the reduced LP (a relaxation) is infeasible, so the subset is infeasible.
- Pruned: a reduced LP (row generation, a relaxation) already had optimum > cutoff 60.789001 (the old best).
  Every pruned subset therefore has true LP value >= its recorded `lb`; the smallest lb is 60.78907 ([90,180,270]).
- That subset was then solved exactly with no cutoff (`scripts/06a_k3_final.py`, `K3_final.json`): 60.789070,
  verified. So the certified K=3 optimum on the 18-degree grid is **60.7891 at 90, 180, 270**, which confirms the old
  plan; it is zero gap among all subsets (the old value was already optimal).
- The cutoff was just below the true value (60.789001 vs 60.789070), which is why no subset ended "solved"; this does not affect validity.
- Caveat: the pruning uses the same row-generation code as 03b (copied into `src/bao_search.py` with a cutoff); the
  first run was interrupted at 777/1140 (process died, apparently killed externally) and resumed from the jsonl.

## 2. K=4,5,6 search (`scripts/06a_search18.py`, `search_cache.jsonl`, `search_log.txt`)
Steepest descent where a move swaps any one beam for ANY unused candidate (K*(20-K) neighbours per round, evaluated
in parallel), multi-start. Starts: top-10 36-degree enumeration plans, earlier best plans, 3 best greedy adds
from the best (K-1) plan, 3 best greedy drops from the certified 36-degree (K+1) plan. Neighbours whose
relaxation exceeds the current objective are pruned (valid; stored with their lb). All evaluated subsets cached (1,800+ entries;
exact LP values for ~hundreds, rest are pruned lower bounds). Exact K=4 optimum is NOT known.
- K=4: all 14 starts, plus an all-two-beam-swap polish (720 neighbours) of the best: no improvement. Starts ended in two local optima
  (49.263 and 49.648). Old 52.931 -> 49.263 (-6.9%).
- K=5: 5 starts (the 45.45 earlier plan, the 44.203 earlier plan, 3 greedy adds); 44.203 is a local optimum under all single swaps and
  no start beat it. The remaining starts (top-10, drops) and the 2-swap polish were NOT run (time). No improvement.
- K=6: only one start (certified 36-degree best, 44.066); one descent was in progress when stopped. Steps: ->42.107 (swap 252->270)
  and the cache also contains 42.027 (18,72,108,144,180,270), an evaluated neighbour of the 42.107 plan from the round in progress when the run was stopped (so the descent had not finished; the reported best is the cache minimum, re-solved and verified by `06a_finalize.py`). Not a local optimum
  proven; more starts would likely improve it.

## Run times
K=3 enumeration about 1.0 CPU-hour of LP time, about 20 min wall on 4 workers (pruning makes most subsets 2-10 s).
K=4 search 82 min wall; K=5 about 1.5 h wall (two attempts, machine load average 11-27 from other jobs); K=6 25 min.

## Problems / honesty notes
- Only K=3 is certified. K=4,5,6 are best found; the third optional item (pruned K=4 enumeration) was not attempted.
  Note the same cutoff pruning would make a K=4 enumeration (4845 subsets) plausible: K=3 took only about 20 min wall.
- The K=5 and K=6 searches were cut short; scripts support `MAXSTART` and `NOPOLISH` environment variables for that.
- Per-subset verification: only the reported best plans were verified on the full grid; other subsets are checked by row generation
  against all target/organ rows and the full-grid body cap (as 03b, which also verified every subset).
- The enumerations ran under heavy machine load from other jobs, so timings are inflated.
- New files: `src/bao_search.py`, `scripts/06a_enumerate18.py`, `06a_k3_final.py`, `06a_search18.py`, `06a_finalize.py`, this folder.
