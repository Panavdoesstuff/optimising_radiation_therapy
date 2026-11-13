# Results summary (ARCHIVED, without the normal-tissue dose cap)

> **Superseded.** This is the earlier run in which dose was capped only on the targets, rectum and
> bladder. Its plans reach up to 189 Gy in normal tissue. The current results, with a 74.8 Gy cap on
> all body tissue, are in `results/SUMMARY.md`; `results/before_after_body_cap.csv` compares the two.
> File paths below refer to this folder (`results/no_body_cap/`), except `results/logs/`.

Group: Panav RG, Karthik Annadurai. Every number here comes from `results/results.json`,
`results/bao_sweep.json` or the CSV tables, and every plan was re-verified on the full dose grid.

## In one paragraph

We plan radiation for a real prostate patient: pick beamlet intensities `x` so the tumour gets
its prescribed dose (dose is linear, `d = D x`) while the rectum, bladder and the rest of the body
get as little as possible. The **easy problem** is a linear program on 5 fixed, evenly spaced
beams. It solves to proven optimality in 19 s (objective 51.82). The **hard problem** adds a
real-world restriction: the machine may use at most K beams, and the optimiser picks which of
20 candidate directions they are (one binary per beam, so it's a MILP). Choosing the 5 beams
instead of fixing them gives a plan **18.3% better** (42.33). The relaxed bound shows no 5-beam
plan can score below 32.55, so the true optimum lies in [32.55, 42.33]. Two honest caveats
dominate the story. (1) The MILP never got past its root node in 600 s, so the plans come
from heuristics and the gaps are 9 to 32%. (2) Neither formulation caps dose in ordinary
tissue, and the hard plan uses that freedom: its maximum body dose is 189 Gy, against 125 Gy
for the easy plan.

## 1. Data and verification (rubric: verification)

- Data: CORT dataset (Craft et al. 2014). Prostate case: 180 beams, 25,404 beamlets, grid
  184×184×90, 3 mm voxels. All expected facts confirmed (`scripts/00_inspect_data.py`).
- **Check A**: set every beamlet to 1 and compare min/mean/max dose per structure with the paper:
  all 12 values match to ≤3×10⁻⁵ relative error. Loader and voxel indexing are correct.
- **Check B**: re-solve the paper's two LPs. Objectives match to 4 significant figures
  (prostate 0.58063 vs 0.5806; TG119 0.35092 vs 0.351). See `sanity_checks.csv`.
- **Every plan** is checked by `src/verify.py`, independently of the solver: recompute
  `d = D x`, check every constraint, recompute the objective. Largest constraint violation
  over all 9 plans: 2.4×10⁻¹² (relative). Recomputed objectives agree with the solver to
  within 1.5×10⁻¹⁵.
  The beam selection plans are re-verified on the *full* dose matrices of the chosen beams.

## 2. Easy problem: LP

```
minimise    1·mean_Rectum(d) + 0.6·mean_Bladder(d) + 0.6·mean_BODY(d)
subject to  d_i ≥ 68 Gy (PTV 68),  d_i ≥ 56 Gy (PTV 56),
            d_i ≤ 74.8 Gy on targets, rectum and bladder,   0 ≤ x_j ≤ x_max
```
Beams 0°, 72°, 144°, 216°, 288°; 721 beamlets; 41,857 constraint rows; 6.9 M nonzeros.

| Result | Value |
|---|---|
| Status | Optimal (HiGHS), first rung of the feasibility ladder (no relaxation needed) |
| Objective | **51.824** Gy; dual objective equal to within 9×10⁻¹³ |
| Mean dose: rectum / bladder / body | 27.58 / 31.00 / 9.40 Gy |
| Key variables | 375 of 721 beamlets on; none at the upper bound; max intensity 7,605 |

## 3. Hard problem (headline): beam angle selection, MILP

```
same objective and dose constraints as the easy LP, over 20 candidate beams (every 18°), plus
x_j ≤ x_max · y_b   (beamlet j belongs to beam b),    Σ_b y_b ≤ K,    y_b ∈ {0, 1}
```
2,854 variables (20 binary), 44,692 constraints, 27 M nonzeros. Limit 600 s per K.

**Why this is more realistic:** a real treatment uses a handful of beams (each extra beam costs
delivery time). The easy LP fixes them arbitrarily; the hard model chooses them.

| K | Best plan (verified) | Beams chosen (°) | Proven lower bound | Gap | Plan came from |
|---|---|---|---|---|---|
| 3 | 48.995 | 90, 180, 270 | 33.375 | 31.9% | heuristic |
| 4 | 45.933 | 54, 90, 180, 270 | 32.790 | 28.6% | heuristic |
| **5** | **42.326** | 54, 162, 198, 270, 306 | 32.546 | 23.1% | drop-one |
| 6 | 39.615 | 54, 162, 198, 252, 270, 306 | 32.461 | 18.1% | drop-one |
| 7 | 37.259 | + 288 | 32.455 | 12.9% | drop-one |
| 8 | 35.706 | + 180 | 32.455 | 9.1% | drop-one |
| 9 | 35.558 | 54, 90, 162, 180, 198, 252, 270, 288, 306 | 32.455 | 8.7% | heuristic |
| all 20 | 32.455 (LP) | — | — | — | LP optimum |

**What changed between the easy plan and the best 5-beam plan (`dose_table.csv`, Fig 1–3):**

| | Easy (5 fixed) | Hard (best 5) | Change |
|---|---|---|---|
| Objective | 51.82 | 42.33 | −9.50 (−18.3%) |
| Rectum mean / V50 | 27.6 Gy / 18.4% | 24.8 Gy / 15.0% | better |
| Bladder mean / V30 | 31.0 Gy / 50.8% | 20.5 Gy / 28.9% | much better |
| Body mean | 9.40 Gy | 8.75 Gy | slightly better |
| **Body maximum** | **124.6 Gy** | **188.6 Gy** | **much worse** |
| Body voxels > 74.8 Gy | 5,164 (0.8%) | 13,533 (2.0%) | worse |
| Target coverage (D95 of PTV 68) | 68.58 Gy | 68.56 Gy | same (both meet 68) |

Most of the gain is in the bladder. The chosen set avoids beams entering through the bladder
and puts the most intensity on the lateral 270° beam (Fig 3). As a side effect, the best
5-beam plan meets the brief's original bladder limit (≤30% above 30 Gy), which the easy plan
violated by 21 points.

**How good is the MILP result?** All seven MILPs stopped at the 600 s limit with **0 nodes
explored**. Each node LP is a 27 M-nonzero solve (about 2 min), and the `x ≤ x_max·y` linking
gives a weak relaxation: fractional `y` lets all 20 beams stay partly open. So the root bound
is barely above the all-beam LP for every K. What the MILP gave us:
- a **valid lower bound** for every K (the dashed line in Fig 5);
- at K = 6 and 7, **better plans than the heuristic** found by HiGHS's root heuristics (40.63
  vs 41.74; 38.56 vs 39.27). Those set off the drop-one post-pass, which improved K = 5…8
  further (for example K = 5: 44.12 → 42.33).

The upper curve in Fig 5 is therefore "best verified plan found", not "proven optimum".

## 4. Hard problem (secondary): dose-volume constraint, MILP

Clinical protocols limit organs by volume ("at most α of the rectum above t Gy"). Exact
model: one binary per rectal voxel, `d_i ≤ t + (74.8 − t)·z_i`, `Σ z_i ≤ ⌊α·1643⌋`.

- The brief's starting point (rectum ≤50% above 50 Gy) is already met by the easy plan (18.4%),
  so it costs nothing. It had to be tightened.
- At ≤14% above 50 Gy, three exact MILP attempts found **no feasible plan** and never explored a
  node: full organ, scipy (904 s); every 3rd voxel with a warm start, highspy (651 s);
  row generation with 5,618 rows (900 s). Logs are in `results/logs/`. We cannot tell whether
  14% is infeasible or just too hard for HiGHS.
- Fix-and-resolve (free the hottest voxels, cap the rest, re-solve the LP) found feasible plans
  only down to **17%**. Tighter limits (16%, 15%, 13%, 11% at 50 Gy; 25% and 20% at 40 Gy;
  30% and 25% at 30 Gy) were infeasible for this heuristic. The CVaR approximation was
  infeasible or numerically unstable (HiGHS status 15) everywhere.

| Final constraint: rectum ≤17% above 50 Gy (279 of 1,643 voxels) | |
|---|---|
| Easy plan | violates it: 302 voxels (18.4%) above 50 Gy |
| Hard plan (fix-and-resolve) | 279 voxels (17.0%), feasible on the full organ |
| Objective | 52.217 vs 51.824: **+0.39 (+0.76%)** |
| True optimum bracket | [51.824, 52.217]: the easy LP is a relaxation, so it is a valid lower bound |

The price is small, but so is the tightening. The rectum gains slightly (mean 27.6 → 27.1 Gy)
and the bladder pays (V30 50.8% → 52.8%). This result is weak, and we present it as such.

## 5. Price of realism (Fig 4, Fig 5)

| Comparison | Absolute | Relative |
|---|---|---|
| Choosing 5 beams vs fixing them (easy → best 5) | −9.50 | **−18.3%** (choice helps) |
| Limiting to 5 beams vs allowing all 20 (all-20 LP → best 5) | between +0.09 and +9.87 | between +0.3% and +30% |
| Adding the rectum dose-volume limit to the easy LP | between 0 and +0.39 | between 0% and +0.76% |

The second row is the real price of realism for beam selection. The bracket is wide because
the MILP did not close the gap. Fig 5 shows the trade-off: each extra beam helps, with
diminishing returns after about 8.

## 6. Limitations to state in the report

1. **Normal-tissue hot spots.** Dose is capped only on the targets, rectum and bladder. The
   mean-dose term barely penalises a few very hot voxels, so all plans have them and fewer beams
   make them worse (max 189 Gy for the best 5-beam plan). A clinical plan would add a cap on
   all of `BODY`. That is the obvious next model, and it would change every number here.
2. **The MILPs are not solved to optimality.** Beam selection gaps are 9–32%. The
   dose-volume MILP found no feasible point at all at 14%.
3. Five beams and one patient. No robustness to setup errors or organ motion is modelled.

## 7. Changes from the brief (with reasons)

| Item | Brief | Used | Why |
|---|---|---|---|
| `x_max` | 50 × 68 = 3,400 | 13,600 | 3,400 binds (78 beamlets at the bound, objective +4.7%); the unbounded optimum peaks at 7,605 |
| Hard problem | dose-volume MILP; beam selection as stretch | beam selection is the headline, dose-volume is secondary | the dose-volume MILP never produced a feasible point (Section 4) |
| Rectum constraint | ≤50% above 50 Gy | ≤17% above 50 Gy | 50% is already met by the easy plan; 17% is the tightest limit for which we found a feasible plan |
| Bladder constraint | ≤30% above 30 Gy (subsampled) | not imposed | the rectum-only MILP already failed, so adding bladder binaries was pointless |
| Beam candidates | every 20° or 10° | every 18° (20 beams) | the only nearby spacing that contains the easy plan's beams, so the easy plan is a feasible K=5 point |
| Fig 5 sweep | objective vs α | objective vs K | α could not be swept: no feasible plan below 17% |
| Solver | scipy HiGHS | scipy HiGHS, plus highspy when installed | only highspy accepts a warm start; same HiGHS engine |
| Fix-and-resolve | approximation | approximation, labelled as such | as specified |

## 8. Suggested 10-minute talk

1. Problem and data (1 min): beams, beamlets, `d = Dx`, one CT slice (Fig 2, top left).
2. We trust the data (1 min): Check A/B table matching the paper.
3. Easy LP (1.5 min): formulation, optimal 51.82, duality gap ~10⁻¹², where the dose goes.
4. Hard model (2 min): binaries per beam, why it is realistic, why it is hard (0 nodes in 600 s).
5. Results (2.5 min): Fig 5 (trade-off curve and gap), Fig 1 (DVH), Fig 4 (price of realism).
6. Honest caveats (1.5 min): body hot spots (Fig 2 bottom), unsolved gaps, dose-volume attempt.
7. Next steps (0.5 min): body dose cap; stronger formulation or decomposition for the MILP.

## Files

Figures: `fig1_dvh.png`, `fig2_dose_maps.png`, `fig3_fluence.png`, `fig4_price_of_realism.png`,
`fig5_sweep_K.png`. Tables: `model_summary.csv`, `dose_table.csv`, `sanity_checks.csv`.
Solutions: `easy_solution.npz`, `hard_dvc_solution.npz`, `bao_K{3..9}_solution.npz`.
