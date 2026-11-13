"""Dose-volume constraint on the fixed easy beams: how low can the rectal V50 go?

1. Provable floor: for every rectal voxel the easy plan puts above t, minimise that voxel's dose
   alone subject to all plan constraints (body cap on the generated rows: a relaxation, so valid).
   A voxel whose minimum is above t is above t in EVERY feasible plan. (A voxel the easy plan keeps
   at or below t can never be forced above t, so only those voxels need checking.)
2. Limit sweep: fix-and-resolve at tighter limits. A plan found is feasible (upper bound on the
   cost of that limit); "none found" is NOT a proof of infeasibility.
Writes results/dvc_floor.json.
"""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, hard_dvc, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
spec = easy_lp.Spec()
P = problem.build("prostate", config.PROSTATE_5)
sets = P.sets("disjoint")
z = np.load(R / "easy_solution.npz")
x_easy, body_rows = z["x"], z["body_rows"]
organ, t, alpha_final = config.DVC_FINAL[0]
S = sets[organ]
d = P.dose(x_easy)
cand = S[d[S] > t * (1 + verify.REL_TOL)]
cand = cand[np.argsort(d[cand])]                 # coolest first: their solves prune the most
witnesses = [np.load(R / "hard_dvc_solution.npz")["x"]] if (R / "hard_dvc_solution.npz").exists() else []
print(f"{organ}: {len(S)} voxels; easy plan has {len(cand)} above {t} Gy -> checking each", flush=True)

t0 = time.time()
fl = hard_dvc.voxel_dose_floor(P, spec, sets, cand, t, body_rows=body_rows, witnesses=witnesses,
                              time_budget=90 * 60)
status, floor = fl["status"], fl["bound"]
forced = cand[status == "forced"]
print(f"floor done in {time.time()-t0:.0f}s: {len(forced)} voxels forced above {t} Gy "
      f"(V{t:g} >= {len(forced)/len(S):.2%}); {fl['n_solves']} solves; unresolved "
      f"{int((status == 'unresolved').sum())}", flush=True)

sweep = []
for a in (0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10):
    if np.floor(a * len(S)) < len(forced):
        sweep.append(dict(alpha=a, result="provably infeasible", reason=f"{len(forced)} voxels are forced above {t} Gy"))
        print(f"  alpha {a:.2f}: provably infeasible", flush=True)
        continue
    fr = hard_dvc.fix_and_resolve(P, spec, sets, [hard_dvc.DVC(organ, t, a)], x_easy, n_iter=3,
                                  body0=body_rows, verbose=False)
    if fr["best"] is None:
        sweep.append(dict(alpha=a, result="no plan found (heuristic; not a proof)"))
        print(f"  alpha {a:.2f}: no plan found by fix-and-resolve", flush=True)
    else:
        sweep.append(dict(alpha=a, result="feasible", objective=fr["best"]["objective"]))
        print(f"  alpha {a:.2f}: feasible, objective {fr['best']['objective']:.4f}", flush=True)

out = dict(organ=organ, t=t, n_voxels=int(len(S)), easy_n_above=int(len(cand)),
           n_forced=int(len(forced)), V_floor=len(forced) / len(S), n_solves=fl["n_solves"],
           n_unresolved=int((status == "unresolved").sum()),
           n_not_attempted=int((status == "not attempted").sum()),
           n_ruled_out=int((status == "not forced").sum()),
           forced_voxels=forced.tolist(), limit_sweep=sweep, time=time.time() - t0,
           note="floor uses the body cap on the easy plan's generated rows only (a relaxation), so it is valid")
(R / "dvc_floor.json").write_text(json.dumps(out, indent=2))
np.savez(R / "dvc_floor_values.npz", voxels=cand, status=status.astype(str), bound=floor)
print("done")
