"""Phase 3: easy LP on the prostate (5 beams, disjoint sets, Gy), with the normal-tissue
(BODY) dose cap enforced by row generation."""
import sys, warnings
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, verify

warnings.simplefilter("ignore", DeprecationWarning)
P = problem.build("prostate", config.PROSTATE_5)
sets = P.sets("disjoint")
print("structure sizes (disjoint):", {k: len(v) for k, v in sets.items()})

res = easy_lp.solve_easy(P)
assert res["success"], res["message"]
m = res["model"]
x = res["x"][:P.n_beamlets]
spec = m.spec
print(f"level used: {res['level']} ({res['level_desc']}); obj {res['objective']:.6f}; "
      f"dual {res['dual_objective']:.6f}; gap {res['duality_gap']:.2e}; {res['solve_time']:.1f}s")

v = verify.verify(P, sets, spec, x, solver_objective=res["objective"],
                  extra=dict(status=res["message"], dual_objective=res["dual_objective"],
                             duality_gap=res["duality_gap"], solve_time=res["solve_time"],
                             iterations=res["iterations"], ladder_level=res["level"],
                             ladder_desc=res["level_desc"], ladder_log=res["ladder_log"],
                             rowgen=res["rowgen"], body_rows_generated=int(len(res["body_rows"])),
                             note="duality gap of the final reduced LP; its dual extends (zeros on the "
                                  "missing body rows) to the full LP, so it certifies full optimality"))
verify.print_report("easy LP", v)
assert v["all_ok"]

config.RESULTS_DIR.mkdir(exist_ok=True)
np.savez(config.RESULTS_DIR / "easy_solution.npz", x=x, angles=np.array(P.angles),
         body_rows=np.asarray(res["body_rows"], np.int64))
verify.update_results("easy", dict(
    params=asdict(spec), d_cap=spec.d_cap, angles=P.angles, set_mode="disjoint",
    set_sizes={k: len(s) for k, s in sets.items()},
    model=dict(n_vars=m.n_vars, n_constraints=m.A_ub.shape[0], n_binaries=0, nnz=int(m.A_ub.nnz),
               body_rows=int(len(res["body_rows"])), body_voxels_total=int(len(sets["BODY"]))),
    verification=v))
print("saved results/easy_solution.npz and results/results.json")
