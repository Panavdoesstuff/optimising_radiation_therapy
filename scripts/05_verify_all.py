"""Re-verify every saved solution on the full dose grid (no optimisation). Runs in a few minutes.

For each plan: rebuild the full dose matrix of its beams from the data files, recompute d = D x,
check every constraint (targets, organ caps, body cap on all body voxels, dose-volume limits where
they apply) and recompute the objective. Writes results/verification_summary.csv.
"""
import csv, json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
spec = easy_lp.Spec()
sets = problem.build("prostate", config.PROSTATE_5).sets("disjoint")
DVC = config.DVC_FINAL


def plans():
    yield "Easy LP (5 equispaced beams)", R / "easy_solution.npz", (), None
    yield "Rectum DVC plan (easy beams)", R / "hard_dvc_solution.npz", DVC, None
    for K in sorted(config.ENUM_K_VALUES):
        yield f"Certified best K={K} of 10 (enumeration)", R / "enum" / f"best_K{K}.npz", (), ("enum", K)
    for K in config.BAO_K_VALUES:
        yield f"Best found K={K} of 20 (MILP sweep)", R / f"bao_K{K}_solution.npz", (), None
    yield "Best found K=5 of 20 (swap search)", R / "swap_K5_solution.npz", (), None
    yield "Final: certified best 5 + rectum limit", R / "final_plan_solution.npz", DVC, None


def reported(tag):
    if tag and tag[0] == "enum":
        recs = [json.loads(l) for l in open(R / "enum" / f"K{tag[1]}.jsonl")]
        return min(r["objective"] for r in recs if r["outcome"] == "solved")
    return None


rows, t0 = [], time.time()
for name, path, dvcs, tag in plans():
    if not path.exists():
        continue
    z = np.load(path)
    angles = [int(a) for a in z["angles"]]
    P = problem.build("prostate", angles, use_cache=False) if angles != config.PROSTATE_5 else \
        problem.build("prostate", config.PROSTATE_5)
    v = verify.verify(P, sets, spec, z["x"], solver_objective=reported(tag), dvcs=dvcs)
    worst = max(c.get("relative", 0.0) for c in v["checks"].values())
    rows.append(dict(plan=name, beams=" ".join(map(str, angles)), objective=round(v["objective_recomputed"], 4),
                     max_rel_violation=f"{worst:.1e}", max_body_dose=round(v["dose_stats"]["BODY"]["max"], 2),
                     dvc_ok=all(d["ok"] for d in v["dvc"].values()) if v["dvc"] else "",
                     all_checks_pass=v["all_ok"]))
    print(f"{'PASS' if v['all_ok'] else 'FAIL'}  {name:45s} obj {v['objective_recomputed']:.4f}  "
          f"max viol {worst:.1e}  body max {v['dose_stats']['BODY']['max']:.2f}", flush=True)
with open(R / "verification_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{sum(r['all_checks_pass'] for r in rows)}/{len(rows)} plans pass ({time.time()-t0:.0f}s)")
if not all(r["all_checks_pass"] for r in rows):
    sys.exit(1)
