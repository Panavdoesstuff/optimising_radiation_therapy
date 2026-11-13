"""Hard problem (secondary): dose-volume constraints on fixed beams.

A dose-volume constraint (DVC) is (organ, t, alpha): at most floor(alpha*|S|) voxels of S may
receive more than t Gy. The exact model needs one binary per organ voxel; it never produced a
feasible plan here (logs in results/logs/). That code (exact big-M MILP, CVaR approximation, row
generation) was removed after commit 6fb0d07, where it can still be found.

What remains:
- fix_and_resolve: a feasible plan for the DVC (an upper bound), labelled as an approximation.
- voxel_dose_floor: a PROVABLE lower bound on how many voxels must exceed t in every feasible plan.
"""
import time
from dataclasses import dataclass
import numpy as np

from . import easy_lp


@dataclass(frozen=True)
class DVC:
    organ: str      # key into the structure sets ("Rectum", "Bladder")
    t: float        # Gy
    alpha: float    # allowed volume fraction above t


def fix_and_resolve(P, spec, sets, dvcs, x0, n_iter=4, body0=None, verbose=True):
    """APPROXIMATION (upper bound): free the floor(alpha|S|) hottest voxels, cap the rest at t,
    solve the LP (body cap by row generation), repeat from the new plan. Every iterate is
    feasible for the exact full-organ problem, so its objective is an upper bound."""
    x = np.asarray(x0, float)
    history, best = [], None
    rows = body0
    for it in range(n_iter):
        d = P.dose(x)
        extra = []
        for dv in dvcs:
            S = sets[dv.organ]
            K = int(np.floor(dv.alpha * len(S)))
            fixed = S[np.argsort(-d[S])[K:]]           # everything but the K hottest
            extra.append((fixed, dv.t))
        r = easy_lp.solve_lp_rowgen(P, spec, sets, extra_ub=extra, body0=rows, verbose=verbose)
        rec = dict(iter=it + 1, success=bool(r["success"] and r.get("converged")), message=r["message"],
                   time=sum(h["time"] for h in r["rowgen"]), rowgen_rounds=len(r["rowgen"]),
                   n_rows=int(r["model"].A_ub.shape[0]))
        if r["success"]:
            rec["objective"] = float(r["objective"])
        history.append(rec)
        if verbose:
            print(f"  fix-and-resolve iter {it+1}: {r['message'].split('(')[0].strip()} "
                  f"obj={rec.get('objective')} ({rec['time']:.0f}s)")
        if not rec["success"]:
            break
        rows = r["body_rows"]
        x = r["x"][:P.n_beamlets]
        best = dict(x=x, objective=float(r["objective"]), body_rows=rows)
        if it > 0 and abs(history[-2].get("objective", np.inf) - r["objective"]) < 1e-7 * abs(r["objective"]):
            break
    return dict(best=best, history=history)


def voxel_dose_floor(P, spec, sets, voxels, t, body_rows=None, witnesses=(), seed_stride=10,
                     margin=0.5, max_rounds=30, witness_rounds=10, group_size=12, n_groups_max=6,
                     time_budget=None, verbose=True):
    """Which voxels are PROVABLY above t in every plan satisfying the target coverage and dose caps?

    For voxel i, min d_i over a relaxation of the plan constraints (body cap only on `body_rows`,
    and only a generated subset of the model rows) is a valid lower bound on its dose in every
    feasible plan. If it exceeds t, voxel i is forced above t.

    Pruning: any plan that satisfies the relaxation and gives voxel j a dose <= t shows that j's
    relaxation minimum is <= t, so j cannot be proven forced and needs no solve of its own. Witness
    plans: the given `witnesses` (fully feasible plans), then up to `witness_rounds` plans that
    minimise the SUMMED dose of all still-open voxels, then one plan per group of `group_size` open
    voxels minimising their summed dose (cools one region hard), then every converged single-voxel
    solve. Only single-voxel solves can mark a voxel forced, so pruning never changes the answer. One highspy model is reused (objective changes, rows only added).

    Returns dict(status per voxel: 'forced' / 'not forced' / 'unresolved', bound per voxel).
    """
    import highspy
    m = easy_lp.build_model(P, spec, sets, extra_cap=body_rows if spec.body_cap else None)
    assert m.n_s == 0, "floor assumes no target slack"
    A_all = m.A_ub.tocsr(); b_all = m.b_ub
    n = m.n_vars
    D_v = P.D[voxels]                                    # candidate rows, for pruning
    tol = 1e-6 * t
    status = np.array(["open"] * len(voxels), dtype=object)
    bound = np.full(len(voxels), np.nan)
    wmin = np.full(len(voxels), np.inf)          # lowest dose seen for each voxel in any witness plan
    for x in witnesses:
        dv = D_v @ np.asarray(x)[:n]
        wmin = np.minimum(wmin, dv)
        status[(status == "open") & (dv <= t + tol)] = "not forced"
    if verbose:
        print(f"    {int((status == 'not forced').sum())} of {len(voxels)} ruled out by the witness plans", flush=True)
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.addVars(n, m.lb, m.ub)
    in_model = np.zeros(A_all.shape[0], bool)

    def add(rows):
        rows = rows[~in_model[rows]]
        if len(rows):
            A = A_all[rows]
            h.addRows(len(rows), np.full(len(rows), -highspy.kHighsInf), b_all[rows], A.nnz,
                      A.indptr[:-1].astype(np.int32), A.indices.astype(np.int32), A.data)
            in_model[rows] = True

    add(np.arange(0, A_all.shape[0], seed_stride))
    idx = np.arange(n, dtype=np.int32)

    def solve(cost):
        """Solve to convergence over all model rows. Returns (objective at last round, x, converged)."""
        h.changeColsCost(n, idx, np.asarray(cost, float).ravel())
        obj, x = np.nan, None
        for _ in range(max_rounds):
            h.run()
            if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return obj, x, False
            x = np.array(h.getSolution().col_value)
            obj = h.getInfo().objective_function_value
            slack = b_all - A_all @ x
            if not (slack < -1e-6 * np.maximum(np.abs(b_all), 1.0)).any():
                return obj, x, True
            add(np.flatnonzero(slack < margin))
        return obj, x, False

    t0, n_solved = time.time(), 0
    for w in range(witness_rounds):
        open_ = np.flatnonzero(status == "open")
        if len(open_) == 0:
            break
        _, x, ok = solve(np.asarray(D_v[open_].sum(axis=0)).ravel())
        n_solved += 1
        pruned = 0
        if ok:
            dv = D_v @ x; wmin = np.minimum(wmin, dv)
            hit = (status == "open") & (dv <= t + tol)
            pruned = int(hit.sum()); status[hit] = "not forced"
        if verbose:
            print(f"    witness round {w+1}: ruled out {pruned}; open {int((status == 'open').sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if pruned < 3:
            break
    groups = [g for g in np.array_split(np.flatnonzero(status == "open"),
                                        max(1, int(np.ceil((status == "open").sum() / group_size))))]
    for gi, g in enumerate(groups[:n_groups_max]):
        g = g[status[g] == "open"]
        if len(g) == 0:
            continue
        _, x, ok = solve(np.asarray(D_v[g].sum(axis=0)).ravel())
        n_solved += 1
        pruned = 0
        if ok:
            dv = D_v @ x; wmin = np.minimum(wmin, dv)
            hit = (status == "open") & (dv <= t + tol)
            pruned = int(hit.sum()); status[hit] = "not forced"
        if verbose:
            print(f"    group {gi+1}/{min(len(groups), n_groups_max)}: ruled out {pruned}; open {int((status == 'open').sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    t_single = time.time()
    while (status == "open").any():
        if time_budget is not None and time.time() - t_single > time_budget:
            status[status == "open"] = "not attempted"
            break
        op = np.flatnonzero(status == "open")
        k = int(op[np.argmax(wmin[op])])          # most likely forced first
        bound[k], x, converged = solve(np.asarray(D_v[k].todense()).ravel())
        n_solved += 1
        if bound[k] > t + tol:
            status[k] = "forced"          # the relaxation minimum is a valid bound, converged or not
        elif converged:
            status[k] = "not forced"
        else:
            status[k] = "unresolved"
        if converged:                     # x satisfies the relaxation: a witness for other voxels
            dv = D_v @ x; wmin = np.minimum(wmin, dv)
            status[(status == "open") & (dv <= t + tol)] = "not forced"
        if verbose:
            print(f"    solve {n_solved}: voxel {k} bound {bound[k]:.2f} -> {status[k]}; "
                  f"open {int((status == 'open').sum())}, forced {int((status == 'forced').sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return dict(status=status, bound=bound, n_solves=n_solved)
