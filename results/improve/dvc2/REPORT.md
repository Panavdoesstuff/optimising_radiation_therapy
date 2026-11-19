# Rectal V50 price-of-realism on the good beam sets (heuristic upper bounds)

Limit: at most alpha of the 1,643 rectal voxels above 50 Gy, body cap 74.8 Gy active. Method: the previous agent's
elastic-swap + polish fix-and-resolve (`src/dvc_improve.py`), driven by `scripts/07c_dvc_good_beams.py`
(adaptive step: start 0.5 point, halve on failure; each alpha warm-started from the previous plan;
lam = 1 for the main runs, 0.3 for the retries B2/B3/A2). Table/figure: `scripts/07c_report.py`.
Every plan below was re-verified on the full grid with `src/verify.py` (coverage, all caps incl. the full body cap,
recomputed objective, V50 recomputed from d = D x on all rectal voxels) -> `verified_table.json`; plans saved as
`plan_<A|B>_a<alpha*1000>.npz` (12.75% is `a128`). A failed run saves no plan (the save is gated on verification and V50 <= allowed).
Price % is against the unconstrained plan of the same beam set. Times are per-alpha wall clock, shared CPU.
Build of the fine-angle problem is fast (cached, ~1 s; the unconstrained LP re-solve takes 1.5-4 min).

## Plan B: beams 90,126,172,190,240 (unconstrained 41.6322, V50 13.76% = 226 voxels)
| alpha | objective | price | verified V50 (voxels) | time |
|---|---|---|---|---|
| 13.50% | 41.6438 | +0.03% | 13.45% (221) | 2 min |
| 13.00% | 42.0495 | +1.00% | 12.96% (213) | 5 min |
| 12.75% | 42.6117 | +2.35% | 12.72% (209) | 7 min |
| 12.70%, 12.60%, 12.50% (x2) | no plan found | - | - | 20-39 min each |

**Lowest alpha reached: 12.75% (209 voxels), price +2.35%.** The 12.6% attempt ended with only 2 violators; 12.5% stalled at 4-6.

## Plan A: beams 108,144,180,216,252 (unconstrained 47.2050, V50 14.18% = 233 voxels)
| alpha | objective | price | verified V50 (voxels) | time |
|---|---|---|---|---|
| 14.00% | 47.2064 | +0.00% | 14.00% (230) | 2 min |
| 13.50% | 47.5899 | +0.82% | 13.45% (221) | 4 min |
| 13.00% | 47.9908 | +1.66% | 12.96% (213) | 4 min |
| 12.75% | 48.8644 | +3.52% | 12.72% (209) | 10 min |
| 12.70%, 12.50% (x2) | no plan found | - | - | 27-47 min each |

**Lowest alpha reached: 12.75%, price +3.52%.**

## Figure
`fig_dvc_tradeoff.png`: objective vs alpha for A, B and the equispaced curve (from `../dvc/verified_table.json`,
dominated lam-5 point excluded). Dotted lines are the unconstrained optima. Plan B dominates A at every alpha, and both
are far below the equispaced curve (52.7 at 17%).

## Caveats / problems
- All numbers are feasible plans (upper bounds on the true optimum at that alpha). "No plan found" is a heuristic stall
  (cycling at 2-6 violators), not a proof of infeasibility; the true limit is lower than 12.75% for both sets.
  The proven floor for the equispaced beams (>= 1.22%) is not established for these beam sets.
- Both curves stop at the same alpha (12.75%, 209 voxels) and the price rises steeply at the end, so the heuristic is
  probably the limit, not geometry.
- Objectives at 14.0% (A) and 13.5% (B) are within 0.03% of unconstrained: the unconstrained optimum is only slightly
  above those alphas. The saved unconstrained plan and a fresh LP re-solve agreed (41.6322, 226 voxels).
- Retry runs B2/B3/A2 (lam 0.3, finer steps) are separate logs/jsons (`sweep_B2/B3/A2.*`) and only produced failures;
  the 12.5% failure appears twice (different step/start). No mislabeled copies: plan files exist only for verified successes.
