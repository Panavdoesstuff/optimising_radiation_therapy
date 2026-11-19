"""Fine-angle local search for K-beam plans (angles on the 2-degree dataset grid).
Each evaluation builds a Reduced over exactly the K beams (cheap: ~0.06 s per beam load) and solves
the exact LP with all-row generation (hard_bao.rowgen_active)."""
import json, time, warnings
import numpy as np
import scipy.sparse as sp

_W = {}


def init_worker(pool_rows):
    warnings.simplefilter("ignore")
    from . import config, problem, easy_lp
    P5 = problem.build("prostate", config.PROSTATE_5)
    _W.update(sets=P5.sets("disjoint"), spec=easy_lp.Spec(), grid=P5.grid, pool=np.asarray(pool_rows, np.int64))


def evaluate(angles, verify_full=False):
    from . import problem, hard_bao, verify
    sets, spec = _W["sets"], _W["spec"]
    angles = sorted(int(a) for a in angles)
    t0 = time.time()
    red = hard_bao.build_reduced(angles, sets, use_cache=False)
    hard_bao.add_rows(red, _W["pool"])
    n0 = len(red.body)
    r = hard_bao.rowgen_active(red, spec, sets, list(range(len(angles))))
    out = dict(angles=angles, success=bool(r["success"]), converged=bool(r.get("converged")))
    x = None
    if r["success"] and r["converged"]:
        out["objective"] = float(r["objective"]); x = r["x"]
        if verify_full:
            mats = [hard_bao.beam_D(red, a) for a in angles]
            boc = np.concatenate([np.full(m.shape[1], i, np.int32) for i, m in enumerate(mats)])
            P = problem.Problem("prostate", angles, sp.hstack(mats, format="csr"), boc, _W["grid"])
            v = verify.verify(P, sets, spec, x, solver_objective=r["objective"])
            out.update(verified=bool(v["all_ok"]), objective_verified=v["objective_recomputed"],
                       max_rel_violation=max(c.get("relative", 0.0) for c in v["checks"].values()),
                       dose_stats=v["dose_stats"])
            out["P"] = P
    else:
        out["message"] = r["message"]
    out["time"] = time.time() - t0
    new = red.rows[red.body[n0:]]
    if len(new):
        _W["pool"] = np.union1d(_W["pool"], new)
    return out, x
