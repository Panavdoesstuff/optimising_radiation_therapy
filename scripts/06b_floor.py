"""Finish the 'not attempted' voxels of results/dvc_floor.json (provable forced-above-50Gy voxels).
Witness plans (any plan satisfying the relaxation) rule voxels out; only single-voxel solves prove 'forced'.
Writes results/improve/dvc/floor_new.json."""
import json, sys, time, warnings, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, hard_dvc
warnings.simplefilter("ignore")
R = config.RESULTS_DIR; OUT = R / "improve" / "dvc"
budget = float(sys.argv[1]) * 60
spec = easy_lp.Spec(); P = problem.build("prostate", config.PROSTATE_5); sets = P.sets("disjoint"); S = sets["Rectum"]
z = np.load(R / "dvc_floor_values.npz", allow_pickle=True)
vox, st = z["voxels"], z["status"].astype(str)
old = json.loads((R / "dvc_floor.json").read_text())
todo = vox[st == "not attempted"]
print(len(todo), "not attempted;", len(old["forced_voxels"]), "forced before", flush=True)
body_rows = np.load(R / "easy_solution.npz")["body_rows"]
wit = [np.load(R / "easy_solution.npz")["x"], np.load(R / "hard_dvc_solution.npz")["x"]]
for f in sorted(glob.glob(str(OUT / "plan_*.npz"))): wit.append(np.load(f)["x"])
print(len(wit), "witnesses", flush=True)
t0 = time.time()
fl = hard_dvc.voxel_dose_floor(P, spec, sets, todo, 50.0, body_rows=body_rows, witnesses=wit, time_budget=budget, witness_rounds=0, n_groups_max=0)
s = fl["status"]
new_forced = todo[s == "forced"]
forced = np.concatenate([np.asarray(old["forced_voxels"]), new_forced])
out = dict(n_attempted_set=len(todo), new_forced=new_forced.tolist(), n_forced_total=int(len(forced)),
           V_floor=len(forced) / len(S), counts={k: int((s == k).sum()) for k in set(s)},
           n_solves=fl["n_solves"], time=time.time() - t0, n_witnesses=len(wit),
           forced_voxels=forced.tolist())
(OUT / "floor_new.json").write_text(json.dumps(out, indent=2))
print({k: v for k, v in out.items() if "voxels" not in k and k != "new_forced"}, flush=True)
