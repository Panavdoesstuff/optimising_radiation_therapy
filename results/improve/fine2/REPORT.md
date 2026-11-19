# Fine-angle search, round 2 (2-degree angles). Heuristic, NOT certified.

Same exact-LP evaluator as round 1 (`src/fine_search.py`); old cache `../fine/evals.jsonl` reused (nothing re-solved),
new LPs appended to `evals.jsonl` here (678 new LPs: K6 399, K5 160, K4 37, K3 82). Code: `scripts/07b_search.py`, `scripts/07b_finalize.py`.
Neighbourhood: every single beam shifted by -14,-10,-8,-6,-4,-2,+2,4,6,8,10,14 deg, best improvement per sweep.
"Converged" = a full sweep of all neighbours found no improvement (local optimum for single-beam shifts up to +-14 deg only).

## Results
| K | old best | new best | beams (deg) | converged | new LPs (distinct total for K) | verified max rel. violation |
|---|---|---|---|---|---|---|
| 6 | 42.005 | **37.736** (-10.2%) | 84,130,172,192,238,288 | yes | 399 (472) | 1.3e-11 |
| 5 | 41.632 | 41.632 (unchanged) | 90,126,172,190,240 | yes (round 1); no better optimum found | 160 (484) | 2.0e-11 |
| 4 | 47.717 | 47.717 (unchanged) | 62,122,174,270 | yes (full sweep, no improving shift) | 37 (210) | 5.1e-12 |
| 3 | 60.335 | **60.181** (-0.3%) | 86,186,272 | yes | 82 (128) | 3.2e-10 |

All four plans re-solved and verified on the full grid (`verify.py`, all constraints incl. 74.8 Gy body cap): pass.
Saved as `best_K{3,4,5,6}.npz` (x, angles); details in `final_6_5_4_3.json`.

| K | objective | rectum mean (Gy) | rectum V50 | bladder mean (Gy) | body mean (Gy) | PTV_68 D95 (Gy) |
|---|---|---|---|---|---|---|
| 6 | 37.736 | 20.16 | 12.7% | 20.61 | 8.69 | 68.54 |
| 5 | 41.632 | 22.13 | 13.8% | 23.58 | 8.91 | 68.55 |
| 4 | 47.717 | 27.27 | 16.9% | 25.03 | 9.05 | 68.73 |
| 3 | 60.181 | 36.76 | 28.0% | 30.48 | 8.55 | 68.52 |
(PTV_68 D95 computed as the 5th percentile of voxel doses in PTV_68; K=3 violates the 17% rectal V50 guideline but it is not a constraint in this model.)

## Search paths
- K6: scan of a 6th beam added to the K5 best (candidates every 20 deg): best 280 (38.560), 300 (38.738); e.g. 100 gave 39.01, 80 39.83, 60 40.03.
  Descent from (90,126,172,190,240,280): 38.560 -> 288 (38.254) -> 90->84 (37.962) -> 126->130 (37.789) -> 240->238 (37.753) -> 190->192 (37.736).
  Last sweep (no improving move) confirmed convergence. Only the best scan start was descended; the second scan start (…,300) and the old K6 plan were not (time), so other K6 basins are unexplored.
- K4: the full sweep from (62,122,174,270) found no improving shift: it was already a local optimum, so round 1's "not converged" note was pessimistic.
- K3: (86,182,270) 60.335 -> 272 (60.224) -> 182->186 (60.181), converged.
- K5 restart from a different start (18-deg grid 2nd best (126,180,198,234,270), 43.63 in cache... ): descended to (126,172,194,236,270) = 42.146 when the time budget ended, NOT converged
  (still improving; path 44.28 -> 43.63 -> 42.37 -> 42.15). Not better than 41.632. The other three starts were not run.

## Time and problems
- Wall clock ~3.8 h (20:11 to ~00:00), 3 workers. Machine was heavily loaded in the first half (load average 12-24): ~1-2 LPs/min, a K=6 sweep took ~50 min;
  later ~4 LPs/min. Initial K6 job was budgeted 1.7 h and one more K6 job (0.7 h) was chained afterwards to confirm convergence.
- Budget checks happen only between sweeps, so jobs overran their nominal budgets slightly; no sweep was ever cut mid-way (so convergence flags are valid).
- Caveat: K=6 (37.736) is a local optimum of a single basin; K=5 and K=6 values are best-found only. Gain from the sixth beam is large (-4.1 vs K=5), which suggests the K=5 plan may also have better basins (e.g. including a beam near 280-290 deg, as the K=6 plan does), not explored.
