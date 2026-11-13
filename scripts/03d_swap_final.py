"""After the enumeration (03b):

1. Swap local search on the 20-candidate grid (18 degrees) for K=5, starting from the certified
   36-degree optimum and from the best earlier 20-candidate plan. Moves: shift one beam by +-18
   degrees; first improvement; exact LP (all-row generation) per move. Result: an UPPER bound for
   the 20-candidate problem, NOT certified (lower bound: the all-20-beam LP).
2. Final plan: the certified best 5 beams + body cap + rectum dose-volume limit, by fix-and-resolve
   (approximation; bracketed below by the same beams without the limit).
Writes results/swap_final.json and the plans' solutions; every plan verified on the full grid.
"""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, hard_bao, hard_dvc, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
spec = easy_lp.Spec()
sets = problem.build("prostate", config.PROSTATE_5).sets("disjoint")
out = {}

# ------------------------------------------------------------------ 1. swap search (20 candidates)
red = hard_bao.build_reduced(config.BAO_CANDIDATES, sets)
red.beam_cache = {}
pool = np.load(R / "bao_body_pool.npy")
extra = R / "enum" / "pool_added.npy"
if extra.exists():
    pool = np.union1d(pool, np.load(extra))
hard_bao.add_rows(red, pool)
cache = {}


def evaluate(angles):
    key = tuple(sorted(angles))
    if key not in cache:
        r = hard_bao.rowgen_active(red, spec, sets, [red.angles.index(a) for a in key])
        cache[key] = r if (r["success"] and r["converged"]) else None
        print(f"    {list(key)}: {None if cache[key] is None else round(cache[key]['objective'], 4)} "
              f"({r['time']:.0f}s)", flush=True)
    return cache[key]


best10 = np.load(R / "enum" / "best_K5.npz")
starts = {"certified 36-degree optimum": [int(a) for a in best10["angles"]]}
prev = R / "bao_K5_solution.npz"
if prev.exists():
    starts["earlier best 20-candidate plan (MILP)"] = [int(a) for a in np.load(prev)["angles"]]
swap = {}
for name, start in starts.items():
    print(f"== swap search from {name}: {start}", flush=True)
    cur = sorted(start); cur_r = evaluate(cur); path = [(cur, cur_r["objective"])]
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(cur):
            for step in (-18, 18):
                b = (a + step) % 360
                if b in cur:
                    continue
                trial = sorted(cur[:i] + [b] + cur[i + 1:])
                r = evaluate(trial)
                if r is not None and r["objective"] < cur_r["objective"] - 1e-7:
                    cur, cur_r, improved = trial, r, True
                    path.append((cur, cur_r["objective"]))
                    print(f"  improved -> {cur} {cur_r['objective']:.4f}", flush=True)
                    break
            if improved:
                break
    swap[name] = dict(start=start, final=cur, objective=cur_r["objective"], path=path)

best_name = min(swap, key=lambda k: swap[k]["objective"])
bs = swap[best_name]
r = cache[tuple(bs["final"])]
P_sel = problem.build("prostate", bs["final"], use_cache=False)
x_sel = np.concatenate([r["x"][red.beam_of_column == red.angles.index(a)] for a in bs["final"]])
v = verify.verify(P_sel, sets, spec, x_sel, solver_objective=r["objective"])
verify.print_report("swap-search best K=5 (20 candidates)", v)
assert v["all_ok"]
np.savez(R / "swap_K5_solution.npz", x=x_sel, angles=np.array(bs["final"]))
lb_all = json.loads((R / "bao_sweep.json").read_text())["all_beams_lp"]["objective"]
out["swap_search"] = dict(runs=swap, best_from=best_name, best_angles=bs["final"],
                          objective=v["objective_recomputed"], lower_bound_all20=lb_all,
                          gap=(v["objective_recomputed"] - lb_all) / v["objective_recomputed"],
                          n_lp_evaluations=len(cache), certified=False, verification=v)

# ------------------------------------------------------------------ 2. final combined plan
angles = [int(a) for a in best10["angles"]]
print(f"== final plan: certified best 5 beams {angles} + body cap + rectum DVC {config.DVC_FINAL}", flush=True)
P = problem.build("prostate", angles, use_cache=False)
base = easy_lp.solve_lp_rowgen(P, spec, sets)                 # same beams, no DVC: the lower bracket
assert base["success"] and base["converged"]
x0 = base["x"][:P.n_beamlets]
d0 = P.dose(x0)
organ, t, alpha = config.DVC_FINAL[0]
S = sets[organ]
pre = dict(frac_over=float((d0[S] > t * (1 + verify.REL_TOL)).mean()), allowed=int(np.floor(alpha * len(S))),
           n_over=int((d0[S] > t * (1 + verify.REL_TOL)).sum()))
print(f"   without the limit: objective {base['objective']:.4f}, {organ} above {t} Gy: {pre}", flush=True)
dvcs = [hard_dvc.DVC(o, tt, a) for o, tt, a in config.DVC_FINAL]
fr = hard_dvc.fix_and_resolve(P, spec, sets, dvcs, x0, n_iter=4, body0=base["body_rows"])
final = dict(angles=angles, without_limit=dict(objective=base["objective"], **pre))
if fr["best"] is None:
    print("   fix-and-resolve found NO feasible plan with the limit for these beams", flush=True)
    final.update(feasible_plan_found=False, history=fr["history"])
else:
    vf = verify.verify(P, sets, spec, fr["best"]["x"], solver_objective=fr["best"]["objective"],
                       dvcs=config.DVC_FINAL, extra=dict(method="fix-and-resolve", history=fr["history"]))
    verify.print_report("final plan (best 5 + cap + DVC)", vf)
    assert vf["all_ok"]
    np.savez(R / "final_plan_solution.npz", x=fr["best"]["x"], angles=np.array(angles))
    final.update(feasible_plan_found=True, objective=vf["objective_recomputed"],
                 bracket=[base["objective"], vf["objective_recomputed"]],
                 price_of_limit_pct=100 * (vf["objective_recomputed"] - base["objective"]) / base["objective"],
                 verification=vf)
out["final_plan"] = final
(R / "swap_final.json").write_text(json.dumps(out, indent=2, default=verify._enc))
print("done")
