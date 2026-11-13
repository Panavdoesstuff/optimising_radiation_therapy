# Results summary: IMRT plan optimisation on the CORT prostate case

Group: Panav RG, Karthik Annadurai. Every number below comes from files in `results/`, and every
plan was re-verified against freshly loaded full-grid dose matrices (`scripts/05_verify_all.py`:
11/11 saved plans pass; `verification_summary.csv`). The earlier run without the normal-tissue cap
is archived in `results/no_body_cap/`.

## The story in three claims

1. **Realism on fixed beams is cheap.** Capping dose in *all* normal tissue costs **+1.37%**
   and removes hot spots of up to 125 Gy (now exactly 74.8 Gy everywhere). Adding a clinical
   rectal limit (≤17% of the rectum above 50 Gy) costs a further **+0.28%**.
2. **A beam budget has a real, certified price.** Choosing *which* 5 beams to use (out of 10
   candidates, every 36°) is a 0/1 problem. Solving all 792 subsets for K = 3…6 exactly gives the
   **certified optimum** with zero gap. The best 5 beams beat the 5 equispaced beams by **10.1%**,
   entirely by sparing the rectum. With those beams the rectal limit is already met, so it
   costs nothing.
3. **Method finding: the textbook MILP does not work here, and we can show why.** The big-M model
   (`x_j ≤ x_max·y_b`, `Σ y_b ≤ K`) has a relaxation that is essentially the all-beam LP, so its
   bound never moved. Exploiting the structure (few binaries) by enumeration solved it exactly.

## 1. Data and verification

- Data: CORT prostate case (Craft et al. 2014): 180 beams, 25,404 beamlets, 184×184×90 grid,
  3 mm voxels. All facts confirmed (`scripts/00_inspect_data.py`).
- **Check A** (all beamlets = 1): all 12 published min/mean/max doses match to ≤3×10⁻⁵.
  **Check B**: both published LPs match to 4 significant figures (`sanity_checks.csv`).
- **`verify.py`** recomputes `d = D x` independently of the solver and checks every
  constraint, including the 74.8 Gy cap on all 675,120 body voxels outside the targets. It also
  recomputes the objective.
- **Results:** all 792 enumerated subsets verified (largest relative violation 1.5×10⁻⁷, tolerance
  10⁻⁶); the 11 headline plans have violations ≤1.2×10⁻⁹. There are 18 tests (TG119 loader tests plus
  synthetic tests for row generation, the MILP relaxation and the voxel floor).

## 2. Easy problem: LP on 5 equispaced beams (0°, 72°, 144°, 216°, 288°)

```
minimise    1·mean_Rectum(d) + 0.6·mean_Bladder(d) + 0.6·mean_BODY(d)
subject to  d_i ≥ 68 Gy (PTV 68),  d_i ≥ 56 Gy (PTV 56)
            d_i ≤ 74.8 Gy on the targets, rectum, bladder AND every other body voxel
            0 ≤ x_j ≤ x_max
```

The body cap covers 675,120 voxels, so it is enforced by **row generation**: solve, recompute
the dose on the full grid, add the violated voxels (plus those within 0.5 Gy of the cap), and
repeat. Every reduced LP is a relaxation, so a solution with no violation is optimal for the
full problem. Here it converged in 3 rounds using **5,785 of 675,120** body voxels.

| | Without body cap | With body cap |
|---|---|---|
| Objective | 51.824 | **52.534** (+1.37%) |
| Max dose in normal tissue | 124.6 Gy | **74.8 Gy** |
| Rectum / bladder / body mean | 27.58 / 31.00 / 9.40 Gy | 28.23 / 31.29 / 9.22 Gy |
| PTV 68 D95 | 68.58 Gy | 68.61 Gy |
| Certificate | duality gap 9×10⁻¹³ | duality gap 1×10⁻¹¹ (final reduced LP; its dual extends to the full LP) |

## 3. Hard problem: beam angle selection (integer)

```
same objective and dose constraints, over a candidate set of beams, plus
x_j ≤ x_max · y_b   (beamlet j in beam b),     Σ_b y_b ≤ K,     y_b ∈ {0,1}
```

**Why it is more realistic:** a real treatment uses a handful of beams, and each extra beam costs
delivery time. The easy LP fixes them arbitrarily; the hard model chooses them.

### 3a. Exact solution by enumeration (headline)

With 10 candidates (every 36°, a grid that contains the easy plan's beams), each K-subset is an
ordinary LP. We solved **all 792 of them** for K = 3…6:
- each with row generation over all target, organ and body rows (converged against every row and
  the full-grid body cap);
- each verified on its beams' full dose matrices.

The minimum per K is the certified optimum of "choose at most K of these 10 beams" (smaller subsets
are dominated by supersets). Six 3-beam subsets are infeasible, which is proven: their reduced LP
is already infeasible. No subset is unresolved. Run time was 1.5 h on 4 cores
(`scripts/03b_enumerate.py`).

| K | Certified optimum | Beams (°) | vs easy (52.53) | vs no budget (all 10 beams: 40.48) |
|---|---|---|---|---|
| 3 | 63.164 | 72, 180, 288 | +20.2% | +56.0% |
| 4 | 55.298 | 108, 144, 180, 252 | +5.3% | +36.6% |
| **5** | **47.205** | **108, 144, 180, 216, 252** | **−10.1%** | +16.6% |
| 6 | 44.066 | 72, 108, 144, 180, 216, 252 | −16.1% | +8.9% |

**What changes between the easy plan and the certified best 5** (`dose_table.csv`, Fig 1–3):

| | Easy (equispaced) | Certified best 5 | Change |
|---|---|---|---|
| Objective | 52.53 | 47.21 | −5.33 (−10.1%) |
| Rectum mean / V30 / V50 | 28.2 Gy / 41.6% / 18.7% | **22.9 Gy / 30.6% / 14.2%** | much better |
| Bladder mean | 31.29 Gy | 31.30 Gy | unchanged |
| Body mean / max | 9.22 / 74.8 Gy | 9.25 / 74.8 Gy | unchanged |
| PTV 68 D95 | 68.61 Gy | 68.45 Gy | both meet 68 |

The whole gain is rectal sparing. The chosen beams lie in the 108°–252° half of the circle
(0° is anterior in this dataset), so we report the effect and do not claim a mechanism.

### 3b. The 20-candidate problem (18° grid): bounds only

| K | Best plan found | How | Proven lower bound | Gap |
|---|---|---|---|---|
| 3 | 60.789 | MILP sweep + exact LP | 32.67 | 46% |
| 4 | 52.931 | MILP sweep + exact LP | 32.67 | 38% |
| 5 | **44.203** (90, 126, 180, 198, 234) | swap search from the certified optimum | 32.67 | 26% |

The finer grid gives about 6% more at K=5, but it is **not certified**. The lower bound is the
all-20-beam LP (32.67), which no MILP improved.

### 3c. Why the big-M MILP failed (keep this: it is a result)

- **What happened:** capped, on every 3rd row, the MILP ran 600–1,800 s per K, explored 0, 0 and 87
  nodes for K = 3, 4, 5, and its bound never rose above the all-beam LP.
- **The reason:** in the LP relaxation `y_b` can be fractional. Take the all-20-beam optimum and
  set `y_b = max_j x_j / x_max`. Every one of the 20 beams stays available for a total budget of
  only **Σ y_b = 5.34**, because no beam comes near `x_max`. (In the 10-beam optimum every
  `y_b ≤ 0.40`, Σ = 3.32.) So for K ≥ 6 the relaxation *is* the all-beam LP, and for K ≤ 5 it is
  barely above it.
- **What doesn't fix it:** tighter per-beamlet bounds (checked: median 12,570 vs 13,600) and coarser
  rows. Only branching on almost every binary would, and that is what enumeration does directly.
- Logs and the capped sweep are in `results/logs/` and `bao_sweep.json`.

## 4. Rectal dose-volume limit (fixed beams)

The clinical form is "at most α of the rectum above t Gy". The exact model needs a binary per
rectal voxel (1,643 of them). It never produced a feasible plan (see `no_body_cap/SUMMARY.md` and
`results/logs/`). We used fix-and-resolve instead (free the hottest voxels, cap the rest, re-solve),
labelled as an approximation.

| Rectum ≤17% above 50 Gy (279 of 1,643 voxels) | Equispaced beams | Certified best 5 beams |
|---|---|---|
| Without the limit | 307 voxels above (18.7%): violates | 233 above (14.2%): already satisfied |
| With the limit | **52.683**, feasible | 47.205 (unchanged) |
| Price of the limit | **+0.28%**, certified within [52.534, 52.683] | **0** |

**How low could the limit go on the equispaced beams?** Here is exactly what we know
(`dvc_floor.json`):
- **Proven floor: V50 ≥ 0.85%.** For each rectal voxel we minimised its dose alone over a
  relaxation of the plan constraints. 14 voxels can never go below 50 Gy, in any plan.
- **Best found: 17%.** Fix-and-resolve found no plan at 16%, 15%, …, 10%. That is a heuristic
  failure, not a proof of infeasibility.
- So the true limit lies in **[0.85%, 17%]** on these beams. A per-voxel floor is weak by
  nature: each voxel can be cooled alone, but not all at once. Of the 307 candidate voxels, 85
  were shown coolable and 208 were not attempted within the 90-minute budget.
- Fig 6 gives the geometry: 17% of the rectum lies within 6.7 mm of the PTV, and in the
  equispaced-plus-limit plan every voxel above 50 Gy is within 10.8 mm of it (median 4.2 mm). The certified beam set reaches 14.2% without being asked
  to, so 17% is **not** a geometric limit, only the limit of equispaced beams plus our heuristic.

## 5. Price of realism (Fig 4, Fig 5, `before_after_body_cap.csv`)

| Added realism | Price |
|---|---|
| Normal-tissue cap (easy LP) | **+1.37%**; max body dose 124.6 → 74.8 Gy |
| Rectal limit on equispaced beams | **+0.28%** (certified bracket) |
| Rectal limit on the certified best beams | **0** |
| Beam budget K = 5 (vs all 10 candidates) | **+16.6%**, certified |
| Choosing 5 beams instead of fixing them | **−10.1%**, certified (a gain, not a cost) |

The cap matters most when beams are few. The best plans found for 20 candidates rise by +24%
(K=3), +15% (K=4) and +7.4% (K=5) once the cap is imposed. Without it, those plans had reached
129–189 Gy in normal tissue.

## 6. Limitations to state in the report

1. **The certificate is for the 36° grid.** The 18° grid gives a better but uncertified plan
   (44.20, gap 26%).
2. **The rectal limit on equispaced beams is bracketed between 0.85% and 17%, not solved.**
3. **Mean-dose objective with fixed weights (1, 0.6, 0.6).** Different weights would choose
   different beams. The 10.1% gain is all rectum, so it depends on the rectum's weight.
4. **Clinical realism not modelled:** fluence smoothness and deliverability (the fluence maps in
   Fig 3 are patchy), robustness to setup error and organ motion, and more than one patient.
5. The 5-beam base case follows the paper. Clinical IMRT usually uses 5–9 beams or arcs.

## 7. Changes from the brief (with reasons)

| Item | Brief | Used | Why |
|---|---|---|---|
| Normal-tissue dose | not capped | 74.8 Gy cap on all body voxels (row generation) | uncapped plans reached 125–189 Gy; requested after the first run |
| `x_max` | 50 × 68 = 3,400 | 13,600 | 3,400 binds (78 beamlets at the bound, objective +4.7%) |
| Hard problem | dose-volume MILP; beam selection as stretch | beam selection is the headline, solved by enumeration | the dose-volume MILP never produced a feasible point; the big-M beam MILP's bound never moved (§3c) |
| Candidate beams | every 20° or 10° | every 36° (certified), every 18° (bounds) | both contain the easy plan's beams, so the easy plan is a feasible point |
| Rectum limit | ≤50% above 50 Gy | ≤17% above 50 Gy | 50% was already met; 17% is the tightest found on the equispaced beams |
| Bladder limit | ≤30% above 30 Gy | not imposed | the rectum-only exact MILP already failed |
| Fig 5 | objective vs α | objective vs K (certified) | α could not be swept below 17% (§4) |
| Solver | HiGHS via scipy | HiGHS via scipy and highspy (warm starts) | same engine; highspy re-solves after adding rows |

## 8. Suggested 10-minute talk

1. Problem and data (1 min): beams, beamlets, `d = Dx`, one CT slice (Fig 2).
2. We trust our numbers (1 min): the paper reproduced, and every plan checked on 3 million voxels.
3. Easy LP plus realism on fixed beams (2 min): row generation, the cap's +1.4%, the rectal limit's +0.3%.
4. Hard model (2 min): one binary per beam; why the big-M MILP cannot work (two lines, §3c).
5. Exact answer (2.5 min): 792 LPs, Fig 5 (certified curve), Fig 1 and Fig 4 (where the 10% comes from).
6. Limits (1 min): 36° vs 18° grid, the [0.85%, 17%] bracket, the objective weights.
7. Next steps (0.5 min): finer-grid certification, fluence smoothness, more patients.

## Files

Figures: `fig1_dvh.png` (DVHs), `fig2_dose_maps.png`, `fig3_fluence.png`, `fig4_price_of_realism.png`,
`fig5_sweep_K.png` (price of a beam budget), `fig6_rectum_distance.png`.
Tables: `model_summary.csv`, `dose_table.csv`, `before_after_body_cap.csv`, `verification_summary.csv`,
`sanity_checks.csv`. Data: `enum/`, `bao_sweep.json`, `swap_final.json`, `dvc_floor.json`, `results.json`.
