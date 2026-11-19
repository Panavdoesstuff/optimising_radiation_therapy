# Fine-angle refinement of K-beam plans (2-degree angles)

Heuristic, NOT certified. No valid lower bound is known for the full 180-angle problem, so no gap is claimed.

## Method
- Each candidate beam set is an exact LP (all-row generation, `hard_bao.rowgen_active`, body cap 74.8 Gy
  converged on the full grid), seeded with the saved body-row pool. A fresh reduced matrix is built for
  exactly the K beams per evaluation (~0.06 s per beam load), so no large union matrix is needed
  (about 0.7 GB per worker). Code: `src/fine_search.py`, `scripts/06c_fine_search.py`, `scripts/06c_finalize.py`.
- Coordinate descent, best improvement per sweep: every beam shifted by -14,-10,-8,-6,-4,-2,+2,...,+10,+14 deg
  (all candidate sets of a sweep solved in parallel, 3 workers), take the best, repeat until no move helps.
  After a local optimum, perturbation restarts (two beams moved at once) while time lasts.
- Every evaluation is cached in `evals.jsonl` (616 lines; 324 distinct sets for K=5, 173 K=4, 73 K=6, 46 K=3).
- Final plans were re-solved and verified on the full grid with `verify.py` on the beams' full dose matrices;
  x and angles in `best_K{K}.npz`, details in `final_5_4_6_3.json`.

## Results
| K | old best | new best | beams (deg) | verified | max rel. violation | distinct LPs | status |
|---|---|---|---|---|---|---|---|
| 5 | 44.203 | **41.632** (-5.8%) | 90,126,172,190,240 | yes | 2.9e-11 | 324 | local optimum for shifts up to +-14 deg; 1 perturbation restart (ended worse, 42.44, cut by time budget) |
| 4 | 52.931 | **47.717** (-9.9%) | 62,122,174,270 | yes | 2.9e-12 | 173 | NOT converged (time-limited; moves were still improving) |
| 6 | 44.066 (certified 36-deg) | **42.005** (-4.7%) | 72,108,144,180,216,266 | yes | 8.1e-12 | 73 | one sweep only (216->266 gave the gain); not converged |
| 3 | 60.789 | **60.335** (-0.7%) | 86,182,270 | yes | 4.1e-13 | 46 | one sweep done, second sweep killed at the time limit |

Note: the 6-beam plan (42.005) is worse than the 5-beam plan (41.632) only because K=6 was barely searched;
a 6-beam plan can contain the 5-beam plan plus any sixth beam, so the K=6 optimum is at most 41.63.

Best K=5 plan, full-grid dose (verified): rectum mean 22.13 Gy, rectum V50 13.76% (the 17% limit is met without
any extra constraint), bladder mean 23.58 Gy, body mean 8.91 Gy. Objective 1*22.135 + 0.6*23.584 + 0.6*8.911 = 41.632.
Other plans (rectum mean / V50 / bladder / body): K4 27.27/16.9%/25.03/9.05; K6 21.14/12.1%/25.97/8.80; K3 36.89/29.0%/30.56/8.51.

## K=5 search path (main descent, start = 18-deg swap result)
1. (90,126,180,198,234) 44.2032
2. 198 -> 192: (90,126,180,192,234) 43.6008
3. 180 -> 172: (90,126,172,192,234) 42.1215
4. 234 -> 240: (90,126,172,192,240) 41.6707
5. 192 -> 190: (90,126,172,190,240) 41.6322, local optimum (all 60 neighbours worse)
Restart 1 (160 and 246 perturbed) descended to (90,126,174,190,246) 42.4393 before the time budget stopped it.
Both K=4 and K=5 improved most by moving the beams near 180 to ~172-174 and the beam at 198 to ~190, i.e. the
18-deg grid forced them off the rectum-sparing angles.

## Time and problems
- K=5 descent + restart 1: 5620 s (~1.6 h); K=4 (two runs, second resumed from the cache) ~1.4 h; K=6 ~0.9 h (one LP is ~60 s under load);
  K=3 ~0.15 h. Total wall clock about 3.7 h. Machine load average was 17-24 on 10 cores (other agents), so LPs took 25-60 s each.
- Because of the budget, only K=5 reached a true local optimum; K=4, 6 and 3 are time-limited and can likely be improved
  further (K=4 was still gaining ~0.2-2.7 per sweep). `search_K4.json` is from the second run; `search_K4_run1.json` is the first.
  The "local optimum" line in K3/K4/K6 logs is printed when the time budget ends, not only on true convergence.
- An accidental duplicate launch at the start (two copies writing the same cache) was killed and restarted; `K5_aborted_dup.log` is its log. The cache
  remains valid (each line is an independent exact LP result; duplicates are harmless).
- Beam-start perturbations from far points need ~3 sweeps of ~13 min, so few restarts were possible.
