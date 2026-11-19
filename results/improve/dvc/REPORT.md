# Rectal V50 improvement (equispaced beams 0,72,144,216,288; body cap 74.8 Gy)

Everything here is a feasible point (upper bound) except the floor, which is a proof.
All plans: `plan_<tag>_a<alpha*1000>.npz` (x). Re-verified on the full grid (coverage, all caps incl. the
675,120-voxel body cap, recomputed objective, V50 recomputed from d = D x on all 1,643 rectal voxels) by
`scripts/06b_verify_table.py` -> `verified_table.json`: all rows below PASS.

## Methods (src/dvc_improve.py, scripts/06b_dvc_sweep.py, scripts/06b_floor.py)
- One persistent highspy LP (warm start) with an excess variable e_i per rectal voxel (D_i x - e_i <= 50);
  each voxel is "hard" (e_i = 0), "hinge" (cost lam*w_i) or "free". Body cap by row generation.
- (a) Plain hinge sum (convex surrogate): min plan cost + lam*sum e. Result: the hinge sum cannot drop below
  ~2,600 Gy*voxel and leaves 273-275 voxels hot even for huge lam (cost 54.4 at lam 0.03, 59.6 at lam 0.3).
  It does NOT sparsify, so it is a poor start for fix-and-resolve. (b) Reweighted L1 (w=1/(e+0.5)) lowered the hot
  count by only ~1 voxel/iteration (275 -> 262 in 4 LPs); abandoned.
- What worked: elastic swap search. Free set F (K = floor(alpha*1643) voxels, started from the K hottest of the
  previous alpha's plan, i.e. a continuation in alpha); non-F voxels get a hinge penalty; violators swap into F,
  replacing the coolest F voxels; stop when no violator; then re-solve as a hard LP (non-F capped at 50, F free) and
  take dual-guided swaps (c; these did not change anything when all K free voxels were hot). Stepsize in alpha has
  to be small (<= ~1 point), big jumps stall at 12-15 violators.
- LPs are slow here (1-9 min each under shared CPU), which limited how far the search could go.

## Trade-off (equispaced beams); easy LP = 52.5344, V50 = 18.7% (307 voxels)
| alpha | objective | price vs 52.534 | verified V50 (voxels) | method | time |
|---|---|---|---|---|---|
| 17.0% | 52.6829 | +0.28% | 16.98% (279) | elastic swap from easy plan (same value as old fix-and-resolve) | 6 min |
| 16.5% | 53.0616 | +1.00% | 16.49% (271) | elastic swap from the 17% plan | 9 min |
| 16.0% | 54.1129 | +3.00% | 15.95% (262) | elastic swap (lam 1) from the 17% plan, 3 LPs + polish | 21 min |
| 15.8% | 54.3753 | +3.50% | 15.76% (259) | elastic swap from the 16% plan (worse than the 16% plan, which also satisfies 15.8%) | 20 min |
| 15.6% .. 10% | no plan | - | - | search stalled at 1-13 violators (15.6%: 3 violators cycling; 15.5%: ~3 violators; 15%: 12-13 violators) | - |

Other verified plan: 16.0% with lam 5, objective 55.0089 (`plan_stB_a160.npz`), dominated by 54.1129.
Not a proof of infeasibility for anything below 15.8%.

Honest reading: this is only a modest improvement over the old result: the best feasible V50 is now
15.8% (259 voxels) instead of 17%, at a price of +3.5% (the curve is steep: ~+1.4 objective units per point
below 17%). The steepness is partly real (voxels next to the PTV are forced hot) and partly the heuristic.
The search did not reach 15%, 12%, 10%.

## Failures
- Plain hinge LP and reweighted hinge: no useful sparsification (above).
- Swap search at alpha <= 15.6%: cycles / stalls; the "no plan" entries are heuristic failures, not proofs.
- Early runs (logs in `old/`) used large lam (5), slow (one LP took 22 min) and cycled; a tabu variant diverged.
- A bug in the first stepwise run (missing failure branch) wrote mislabeled copies of the 16% plan for 15% and 14%;
  those files were deleted and the `sweep_stA.json` rows marked verified_ok=False for 0.15 are invalid. The same bug
  in `sweep_stC` produced no kept plans. The bug is fixed in the script (later runs stD/stE are clean).
- Goal 3 (best-5 beams sweep) was not run (time).

## Proven lower bound (scripts/06b_floor.py, floor_new.json)
Single-voxel dose minimisation over the relaxation (same method as `dvc_floor.json`) on the 208 voxels that were
"not attempted": 6 solves (+1 voxel ruled out by a witness) in the budget (7-20 min each) -> **6 more voxels forced above 50 Gy**
(1 ruled out). Total forced: 14 + 6 = **20 voxels, V50 >= 20/1643 = 1.22%** (was 0.85%). 201 voxels
remain unattempted (each solve is too slow; witnesses did not prune any of them).

## New bracket on the equispaced beams
**V50 in [1.22% (proven), 15.76% (best found, objective 54.375)].** At 17% the cheapest known plan costs 52.683.
