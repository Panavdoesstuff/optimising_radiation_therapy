# Results summary: IMRT plan optimisation on the CORT prostate case

Group: Panav RG, Karthik Annadurai. Every number below comes from a file in `results/`, and every
plan quoted was re-verified against freshly loaded full-grid dose matrices (`src/verify.py`: coverage,
every dose cap including all 675,120 body voxels, recomputed objective). The first summary is kept
as `SUMMARY_v1.md`; this one adds the second optimisation round (`results/improve/`).

## The story in four claims

1. **Realism on fixed beams is cheap.** A dose cap on *all* normal tissue costs **+1.37%** and removes
   hot spots of up to 125 Gy. A rectal limit (≤17% of the rectum above 50 Gy) costs a further **+0.28%**.
2. **Choosing the beams is where the gain is.** Against the easy plan (5 equispaced beams, 52.534):
   - **Certified** (all subsets solved, zero gap), 36° grid, 5 beams: **47.205 (−10.1%)**.
   - **Best found**, beams at any 2° angle, 5 beams: **41.632 (−20.8%)**.
   - **Best found**, 6 beams: **37.736 (−28.2%)**.
   The best-found plans are not certified (§3c).
3. **We can prove optimality on a finer grid for few beams.** On the 18° grid (20 candidate angles),
   the best 3 beams are certified at **60.789** and the best 4 beams at **49.263**, by pruned
   enumeration (§3b).
4. **The textbook MILP does not work here, and we can show why** (§3d). Its relaxation is essentially
   the all-beam LP, so its bound never moved. Enumeration solves the same problem exactly.

## 1. Data and verification

- Data: CORT prostate case (Craft et al. 2014): 180 beams, 25,404 beamlets, 184×184×90 grid, 3 mm voxels.
- **Check A** (all beamlets = 1): all 12 published min/mean/max doses match to ≤3×10⁻⁵.
  **Check B**: both published LPs match to 4 significant figures (`sanity_checks.csv`).
- **`verify.py`** recomputes `d = D x` independently of the solver, checks every constraint on the
  full grid and recomputes the objective. Every plan below passes (largest relative violation 3.2×10⁻¹⁰,
  tolerance 10⁻⁶; `final_comparison.csv`, `verification_summary.csv`, `improve/*/REPORT.md`).
- 18 tests (TG119 loader tests plus synthetic tests for row generation, the MILP relaxation and the
  voxel floor). `scripts/08_final_figures.py` re-verifies the four headline plans before plotting.

## 2. Easy problem: LP on 5 equispaced beams (0°, 72°, 144°, 216°, 288°)

```
minimise    1·mean_Rectum(d) + 0.6·mean_Bladder(d) + 0.6·mean_BODY(d)
subject to  d_i ≥ 68 Gy (PTV 68),  d_i ≥ 56 Gy (PTV 56)
            d_i ≤ 74.8 Gy on the targets, rectum, bladder AND every other body voxel
            0 ≤ x_j ≤ 13,600
```

The body cap is 675,120 rows, so it is enforced by **row generation**: solve, recompute the dose on the
full grid, add the violated voxels, repeat. Each reduced LP is a relaxation, so a solution that violates
nothing is optimal for the full problem. It converged in 3 rounds using **5,785 of 675,120** body rows.

| | Without body cap | With body cap |
|---|---|---|
| Objective | 51.824 | **52.534** (+1.37%) |
| Max dose in normal tissue | 124.6 Gy | **74.8 Gy** |
| Rectum / bladder / body mean | 27.58 / 31.00 / 9.40 Gy | 28.23 / 31.29 / 9.22 Gy |
| Certificate | duality gap 9×10⁻¹³ | duality gap 1×10⁻¹¹ |

## 3. Hard problem: beam angle selection

```
same objective and constraints, over a candidate set of beams, plus
x_j ≤ x_max · y_b   (beamlet j in beam b),     Σ_b y_b ≤ K,     y_b ∈ {0,1}
```

A real treatment uses a handful of beams, each costing delivery time. The easy LP fixes them arbitrarily;
the hard model chooses them. It is a mixed-integer program, and for a *given* set of beams it is just an LP.

### 3a. Results at a glance (objective, lower is better; easy plan = 52.534)

| K beams | Certified, 36° grid (10 angles) | Certified, 18° grid (20 angles) | Best found, 2° angles | vs easy |
|---|---|---|---|---|
| 3 | 63.164 | **60.789** (90, 180, 270) | 60.181 (86, 186, 272) | +14.6% |
| 4 | 55.298 | **49.263** (54, 126, 180, 270) | 47.717 (62, 122, 174, 270) | −9.2% |
| 5 | **47.205** (108, 144, 180, 216, 252) | not certified (best 44.203) | **41.632** (90, 126, 172, 190, 240) | **−20.8%** |
| 6 | 44.066 (72, 108, 144, 180, 216, 252) | not certified (best 42.027) | **37.736** (84, 130, 172, 192, 238, 288) | **−28.2%** |

The columns are different problems: more candidate angles can only lower the optimum. Bold marks a
proof (certified columns) or the headline plan (best found).

### 3b. What is certified, and how

**36° grid (10 candidates), K = 3…6.** All 792 K-subsets are ordinary LPs, each solved by row
generation over every target, organ and body row, and each verified on the full grid. The minimum per K
is the certified optimum (smaller subsets are dominated by supersets). Six 3-beam subsets are infeasible,
which is proven (their reduced LP is already infeasible). No subset is unresolved. 1.5 h on 4 cores.

**18° grid (20 candidates), K = 3 and K = 4: pruned enumeration.** C(20,3) = 1,140 and C(20,4) = 4,845
subsets. Each reduced LP is a relaxation, so if its value already exceeds the incumbent plan's value
(the cutoff), the subset cannot win, and its bound is recorded.

| K | Subsets | Solved | Pruned (bound > cutoff) | Infeasible | Unresolved | Optimum | Time |
|---|---|---|---|---|---|---|---|
| 3 | 1,140 | 0 new | 1,042 | 98 | **0** | 60.789 | ~20 min, 4 workers |
| 4 | 4,845 | 1 | 4,844 | 0 | **0** | 49.263 | 171 min, 4 workers |

For K = 4 the smallest recorded bound among pruned subsets is 49.2639, only 0.0013 above the optimum, so
the proof is tight. The old best-found plans for both K were already optimal. The certificate holds only
because no subset is unresolved (`results/improve/bao18/`, `results/improve/cert18/`).

**K = 5 on the 18° grid was not certified.** 15,504 subsets; a 300-subset sample had a 100% prune rate
but extrapolates to about 10 h, so we stopped (the sample is not a certificate).

### 3c. Best-found plans at 2° angles (not certified)

Starting from the best grid plans, we ran a coordinate descent: shift one beam by ±2…±10° (±14° in the
first round), solve the exact LP for each candidate, keep improvements, repeat
(`results/improve/fine/`, `results/improve/fine2/`; 324 + 160 LPs for K = 5).

- **K = 5** reached **41.632**, a local optimum for single-beam shifts up to ±14°. Search path:
  44.203 → 43.601 → 42.122 → 41.671 → 41.632. A restart from a different start ended worse (42.15, not converged).
- **K = 6** improved from 42.005 to **37.736** once a sixth beam near 280–288° was tried. It was descended
  from a single start, so other basins are unexplored.
- **K = 4** (47.717) and **K = 3** (60.181) converged: a full sweep found no improving shift.
- **No bound exists for the 180-angle problem.** The all-20-beam LP (32.67) bounds the 18° grid only.
  We make no gap claim for 2° plans.

Dose metrics of the main plans (`final_comparison.csv`, Fig 8, Fig 10):

| | Easy (equispaced) | Certified best 5 | Best found 5 | Best found 6 |
|---|---|---|---|---|
| Objective | 52.534 | 47.205 | 41.632 | 37.736 |
| vs easy | | −10.1% | −20.8% | −28.2% |
| Rectum mean / V50 | 28.2 Gy / 18.7% | 22.9 Gy / 14.2% | 22.1 Gy / 13.8% | 20.2 Gy / 12.7% |
| Bladder mean | 31.3 Gy | 31.3 Gy | 23.6 Gy | 20.6 Gy |
| Body mean / max | 9.22 / 74.8 Gy | 9.25 / 74.8 Gy | 8.91 / 74.8 Gy | 8.69 / 74.8 Gy |
| PTV 68 D95 | 68.61 Gy | 68.45 Gy | 68.55 Gy | 68.54 Gy |

The certified 36° plan spares only the rectum. The 2° plans spare the bladder as well (31.3 → 23.6 Gy),
which the 36° grid could not reach. Tumour coverage is met in every plan (D95 ≥ 68 Gy).
The chosen beams cluster on the 84°–288° side (0° is anterior in this dataset); we report the effect
and do not claim a mechanism.

### 3d. Why the big-M MILP failed (a result, not a gap)

- **What happened:** capped, on every 3rd row, the MILP ran 600–1,800 s per K, explored 0, 0 and 87
  nodes for K = 3, 4, 5, and its bound never rose above the all-beam LP (32.67; gaps 26–46%).
- **The reason:** in the relaxation `y_b` can be fractional. Take the all-20-beam optimum and set
  `y_b = max_j x_j / x_max`: all 20 beams stay available for a total budget of only **Σ y_b = 5.34**,
  because no beam comes near `x_max`. For K ≥ 6 the relaxation is the all-beam LP; for K ≤ 5 it is
  barely above it.
- **What doesn't fix it:** tighter per-beamlet bounds, coarser rows. Branching on almost every binary
  would, and that is what enumeration does directly. Logs: `results/logs/`, `bao_sweep.json`.

## 4. Rectal dose-volume limit

The clinical form is "at most α of the rectum above t Gy". The exact model needs a binary per rectal
voxel (1,643). It never produced a feasible plan, so we used heuristics built on LPs (fix-and-resolve,
then an elastic swap search), labelled as **upper bounds** (feasible plans, not optima).
Each reported V50 is recomputed from `d = D x` on all 1,643 rectal voxels.

| Beam set | Unconstrained V50 | Tightest limit reached | Objective there | Price |
|---|---|---|---|---|
| Equispaced 5 | 18.7% | **15.76%** | 54.375 | +3.5% |
| Certified best 5 (36°) | 14.18% | **12.75%** | 48.864 | +3.52% |
| Best found 5 (2°) | 13.76% | **12.75%** | 42.612 | +2.35% |

(At the original 17% limit, equispaced beams cost +0.28%: 52.683.) Fig 9 shows the full curves.

- **Beam choice dominates.** Both good beam sets beat the equispaced curve at every limit; even their
  tighter 12.75% limit is cheaper than the equispaced plan at 17%.
- **Proven floor: V50 ≥ 1.22%** (20 rectal voxels provably above 50 Gy in every feasible plan; each found
  by minimising that voxel's dose over a relaxation). 201 voxels were not attempted.
- **Below 12.75% the search stalled** (2–6 voxels left over) on both beam sets. That is a heuristic
  failure, not a proof of infeasibility. The true limit lies between 1.22% and 12.75%.

## 5. Price of realism (Fig 4, Fig 5, Fig 7, Fig 9, `before_after_body_cap.csv`)

| Added realism | Price |
|---|---|
| Normal-tissue cap (easy LP) | **+1.37%**; max body dose 124.6 → 74.8 Gy |
| Rectal limit ≤17% on equispaced beams | **+0.28%** |
| Rectal limit ≤12.75% on best-found 5 beams | **+2.35%** |
| Beam budget K=5 vs all 10 candidates (36° grid) | **+16.6%**, certified |
| Choosing 5 beams instead of fixing them | **−10.1%** certified, **−20.8%** best found (a gain, not a cost) |
| Choosing 6 beams, best found | **−28.2%** |

The cap matters most when beams are few: the earlier uncapped best plans reached 129–189 Gy in normal
tissue and rose 7–24% once the cap was imposed.

## 6. Limitations to state in the report

1. **Certificates cover the grids we enumerated.** 36° for K = 3…6; 18° for K = 3 and 4 only.
   Everything at 2° angles, and K = 5, 6 on the 18° grid, is best found.
2. **The 2° searches are local.** K = 5 and 4 converged for single-beam shifts; K = 6 was descended from
   one start; nothing was tried that moves two beams at once beyond one failed restart.
3. **The rectal limit is bracketed, not solved:** [1.22%, 12.75%] on the good beams.
4. **Mean-dose objective with fixed weights (1, 0.6, 0.6).** Different weights would choose different
   beams. Part of the gain is rectum and bladder sparing, so it depends on those weights.
5. **Clinical realism not modelled:** fluence smoothness and deliverability, robustness to setup error
   and organ motion, and more than one patient. The numbers belong to this one patient; the method does not.
6. The 5-beam base case follows the paper. Clinical IMRT usually uses 5–9 beams or arcs.

## 7. Changes from the brief (with reasons)

| Item | Brief | Used | Why |
|---|---|---|---|
| Normal-tissue dose | not capped | 74.8 Gy cap on all body voxels (row generation) | uncapped plans reached 125–189 Gy |
| `x_max` | 50 × 68 = 3,400 | 13,600 | 3,400 binds (78 beamlets at the bound, objective +4.7%) |
| Hard problem | dose-volume MILP | beam selection is the headline, solved by enumeration | the dose-volume MILP never produced a feasible point; the big-M beam MILP's bound never moved |
| Candidate beams | every 20° or 10° | 36° and 18° grids, then free 2° angles | both grids contain the easy plan's beams, so the easy plan is a feasible point |
| Rectum limit | ≤50% above 50 Gy | ≤17% originally, then down to 12.75% | 50% was already met; tighter limits found by the later sweeps |
| Bladder limit | ≤30% above 30 Gy | not imposed | the rectum-only exact MILP already failed |
| Solver | HiGHS via scipy | HiGHS via scipy and highspy | same engine; highspy re-solves after adding rows |

## 8. Suggested 10-minute talk

1. Problem and data (1 min): beams, beamlets, `d = Dx`, one CT slice (Fig 2).
2. We trust our numbers (1 min): the paper reproduced, every plan checked on 3 million voxels.
3. Easy LP plus realism on fixed beams (2 min): row generation, the cap's +1.4%, the rectal limit.
4. Hard model (1.5 min): one binary per beam; why the big-M MILP cannot work (§3d, two lines).
5. Exact answer (2 min): enumeration, pruning, Fig 7: certified vs best found.
6. Where the gain comes from (1.5 min): Fig 8 and Fig 10, the −10% / −21% / −28% table.
7. Limits and next steps (1 min): §6; a 5-beam proof on the 18° grid (~10 h), more patients, weights.

## Files

Figures: `fig1_dvh.png`, `fig2_dose_maps.png`, `fig3_fluence.png`, `fig4_price_of_realism.png`,
`fig5_sweep_K.png`, `fig6_rectum_distance.png` (first round); `fig7_beam_budget_final.png`,
`fig8_dvh_final.png`, `fig9_dvc_tradeoff_final.png`, `fig10_organ_means.png` (final round).
Tables: `final_comparison.csv`, `model_summary.csv`, `dose_table.csv`, `before_after_body_cap.csv`,
`verification_summary.csv`, `sanity_checks.csv`. Data: `enum/`, `improve/` (see `improve/README.md`),
`bao_sweep.json`, `swap_final.json`, `dvc_floor.json`, `results.json`.
