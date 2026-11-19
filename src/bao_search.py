"""Helpers for beam selection on the 18-degree grid: worker evaluation of one beam subset with an
optional cutoff (early stop when a relaxation already exceeds it), and full-grid verification."""
import time, warnings
import numpy as np
import scipy.sparse as sp
from . import config, problem, easy_lp, hard_bao, verify
from .hard_bao import _IncrementalLP, _cost, _dose_rows, full_dose, add_rows

CANDS = list(range(0, 360, 18))
_W = {}


def init_worker(pool_rows):
    warnings.simplefilter("ignore")
    P5 = problem.build("prostate", config.PROSTATE_5)
    sets = P5.sets("disjoint")
    red = hard_bao.build_reduced(CANDS, sets)
    red.beam_cache = {}
    add_rows(red, pool_rows)
    _W.update(sets=sets, red=red, spec=easy_lp.Spec(), grid=P5.grid)


def load_pool(R=None):
    R = R or config.RESULTS_DIR
    pool = np.load(R / "bao_body_pool.npy")
    extra = R / "enum" / "pool_added.npy"
    if extra.exists():
        pool = np.union1d(pool, np.load(extra))
    return pool


def rowgen_cutoff(red, spec, sets, beams, cutoff=None, seed_stride=10, margin=0.5, max_iter=40):
    """Same algorithm as hard_bao.rowgen_active, plus: if a reduced LP (a relaxation) has optimal
    value > cutoff, stop and return pruned=True with that value as a valid lower bound."""
    from .easy_lp import body_violations
    tol = 1e-6
    N = sets["BODY"]
    cols = np.flatnonzero(np.isin(red.beam_of_column, beams))
    c = _cost(red, spec)[cols]
    s = int(seed_stride)
    sel = dict(T68=red.loc["T68"][::s], T56=red.loc["T56"][::s], cap=np.arange(len(red.rows))[::s])
    t0 = time.time(); nit = 0
    lp = _IncrementalLP(red, spec, cols, c)
    added = dict(T68=np.zeros(0, np.int64), T56=np.zeros(0, np.int64), cap=np.zeros(0, np.int64))
    for it in range(1, max_iter + 1):
        nit = it
        new = {k: np.setdiff1d(sel[k], added[k]) for k in sel}
        lp.add(new)
        added = {k: np.union1d(added[k], new[k]) for k in sel}
        r = lp.solve()
        if not r.success:
            return dict(success=False, message=r.message, converged=False, rounds=nit, time=time.time() - t0)
        if cutoff is not None and r.fun > cutoff:
            return dict(success=True, converged=False, pruned=True, lb=float(r.fun), message="pruned",
                        rounds=nit, time=time.time() - t0)
        d_red = red.D[:, cols] @ r.x
        T68, T56 = red.loc["T68"], red.loc["T56"]
        allc = np.arange(len(red.rows))
        v68 = T68[d_red[T68] < spec.rx68 * (1 - tol)]
        v56 = T56[d_red[T56] < spec.rx56 * (1 - tol)]
        vcap = allc[d_red > spec.d_cap * (1 + tol)]
        n_viol = len(v68) + len(v56) + len(vcap)
        if n_viol == 0:
            x = np.zeros(red.n_beamlets); x[cols] = r.x
            d = full_dose(red, x)
            over, rel = body_violations(d, N, spec)
            if spec.body_cap and len(over):
                nb = add_rows(red, N[d[N] > spec.d_cap - margin])
                if nb == 0:
                    pos = np.flatnonzero(np.isin(red.rows, over))
                    sel["cap"] = np.union1d(sel["cap"], pos)
                else:
                    sel["cap"] = np.union1d(sel["cap"], np.arange(len(red.rows) - nb, len(red.rows)))
            else:
                return dict(success=True, converged=True, message=r.message, x=x, objective=float(r.fun),
                            rounds=nit, time=time.time() - t0, max_body_dose=float(d[N].max()))
        else:
            sel["T68"] = np.union1d(sel["T68"], T68[d_red[T68] < spec.rx68 + margin])
            sel["T56"] = np.union1d(sel["T56"], T56[d_red[T56] < spec.rx56 + margin])
            sel["cap"] = np.union1d(sel["cap"], allc[d_red > spec.d_cap - margin])
    return dict(success=True, converged=False, message="max_iter", rounds=nit, time=time.time() - t0)


def verify_plan(angles, x_red_full, red=None, sets=None, spec=None, obj=None):
    """Independent full-grid verification (as scripts/03b does). x_red_full is over all candidate columns."""
    red = red or _W["red"]; sets = sets or _W["sets"]; spec = spec or _W["spec"]
    angs = sorted(angles)
    beams = [red.angles.index(a) for a in angs]
    mats = [hard_bao.beam_D(red, a) for a in angs]
    boc = np.concatenate([np.full(m.shape[1], i, np.int32) for i, m in enumerate(mats)])
    P = problem.Problem("prostate", angs, sp.hstack(mats, format="csr"), boc, _W["grid"])
    x = np.concatenate([x_red_full[red.beam_of_column == b] for b in beams])
    v = verify.verify(P, sets, spec, x, solver_objective=obj)
    return v, x


def solve_subset(args):
    """args = (angles, cutoff, verify_below). Returns (record, x_selected or None, new_body_rows)."""
    angles, cutoff, verify_below = args
    red, sets, spec = _W["red"], _W["sets"], _W["spec"]
    t0 = time.time(); n0 = len(red.body)
    beams = [red.angles.index(a) for a in angles]
    r = rowgen_cutoff(red, spec, sets, beams, cutoff)
    rec = dict(angles=list(angles), K=len(angles), rounds=r.get("rounds"), status=r["message"])
    x_sel = None
    if r.get("pruned"):
        rec.update(outcome="pruned", lb=r["lb"])
    elif r["success"] and r["converged"]:
        rec.update(outcome="solved", objective=r["objective"], max_body_dose=r["max_body_dose"])
        if verify_below is not None and r["objective"] < verify_below:
            v, x_sel = verify_plan(angles, r["x"], obj=r["objective"])
            rec.update(objective=v["objective_recomputed"], verified=bool(v["all_ok"]),
                       max_rel_violation=max(c.get("relative", 0.0) for c in v["checks"].values()))
    elif not r["success"] and "infeasible" in r["message"].lower():
        rec["outcome"] = "infeasible"
    else:
        rec["outcome"] = "unresolved"
    rec["time"] = time.time() - t0
    return rec, x_sel, red.rows[red.body[n0:]].tolist()
