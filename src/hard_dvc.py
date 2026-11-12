"""Hard problem: dose-volume constraints (MILP) plus two labelled approximations.

A dose-volume constraint (DVC) is (organ, t, alpha): at most floor(alpha*|S|) voxels of S may
receive more than t Gy.
"""
import time
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

from . import easy_lp


@dataclass(frozen=True)
class DVC:
    organ: str      # key into the structure sets ("Rectum", "Bladder")
    t: float        # Gy
    alpha: float    # allowed volume fraction above t
    stride: int = 1  # MILP only: constrain every stride-th voxel of the organ


def _pad(A, n_cols_total):
    """Pad a CSR block with zero columns on the right up to n_cols_total columns."""
    if A.shape[1] == n_cols_total:
        return A.tocsr()
    return sp.hstack([A, sp.csr_matrix((A.shape[0], n_cols_total - A.shape[1]))], format="csr")


def _D_rows(P, S, m):
    return _pad(P.D[S], m.n_x + m.n_s)


# ---------------------------------------------------------------- MILP
def build_milp(P, m: easy_lp.Model, dvcs):
    """Extend the easy model with z_i binaries. Returns (c, A, lo, hi, lb, ub, integrality, info)."""
    n0 = m.n_vars
    blocks = [m.A_ub]; rhs = [m.b_ub]
    n_z_total = sum(len(m.idx[_key(d.organ)][::d.stride]) for d in dvcs)
    col = n0
    zinfo = []
    extra_rows = []
    for d in dvcs:
        S_full = m.idx[_key(d.organ)]
        S = S_full[::d.stride]
        nz = len(S)
        M = m.spec.d_cap - d.t
        assert M > 0
        # d_i - M z_i <= t
        Ad = sp.hstack([_D_rows(P, S, m), sp.csr_matrix((nz, col - n0)),
                        -M * sp.identity(nz, format="csr"),
                        sp.csr_matrix((nz, n_z_total - (col - n0) - nz))], format="csr")
        blocks.append(Ad); rhs.append(d.t * np.ones(nz))
        # sum z <= floor(alpha*|S|)
        k = int(np.floor(d.alpha * nz + 1e-9))
        row = sp.csr_matrix((np.ones(nz), (np.zeros(nz, int), np.arange(col, col + nz))),
                            shape=(1, n0 + n_z_total))
        extra_rows.append((row, k))
        zinfo.append(dict(organ=d.organ, t=d.t, alpha=d.alpha, stride=d.stride, n_z=nz,
                          n_organ=len(S_full), allowed_in_subsample=k, z_cols=(col, col + nz), voxels=S))
        col += nz
    A = sp.vstack([_pad(b, n0 + n_z_total) for b in blocks] + [r for r, _ in extra_rows], format="csr")
    hi = np.concatenate(rhs + [np.array([k], float) for _, k in extra_rows])
    lo = np.full(len(hi), -np.inf)
    c = np.concatenate([m.c, np.zeros(n_z_total)])
    lb = np.concatenate([m.lb, np.zeros(n_z_total)])
    ub = np.concatenate([m.ub, np.ones(n_z_total)])
    integrality = np.concatenate([np.zeros(n0), np.ones(n_z_total)])
    return dict(c=c, A=A, lo=lo, hi=hi, lb=lb, ub=ub, integrality=integrality, zinfo=zinfo,
                n_z=n_z_total, n0=n0)


def solve_milp(P, m, dvcs, time_limit=600.0, mip_rel_gap=1e-4, disp=False):
    mm = build_milp(P, m, dvcs)
    t0 = time.time()
    r = milp(mm["c"], integrality=mm["integrality"], bounds=Bounds(mm["lb"], mm["ub"]),
             constraints=LinearConstraint(mm["A"], mm["lo"], mm["hi"]),
             options=dict(time_limit=time_limit, mip_rel_gap=mip_rel_gap, disp=disp))
    out = dict(status=int(r.status), message=r.message, solve_time=time.time() - t0,
               success=r.x is not None, n_vars=len(mm["c"]), n_constraints=mm["A"].shape[0],
               n_binaries=mm["n_z"], nnz=int(mm["A"].nnz), zinfo=[{k: v for k, v in z.items() if k != "voxels"} for z in mm["zinfo"]],
               mip_gap=getattr(r, "mip_gap", None), best_bound=getattr(r, "mip_dual_bound", None),
               node_count=getattr(r, "mip_node_count", None))
    if r.x is not None:
        out.update(x=r.x[:m.n_x], objective=float(r.fun))
    return out


def _key(organ):
    return {"Rectum": "R", "Bladder": "B"}[organ]


# ---------------------------------------------------------------- approximations
def _solve_with_rows(m, A_extra, b_extra, extra_cols=0, c_extra=None, lb_extra=None, ub_extra=None):
    n0 = m.n_vars
    n = n0 + extra_cols
    A = sp.vstack([_pad(m.A_ub, n)] + [_pad(a, n) for a in A_extra], format="csr")
    b = np.concatenate([m.b_ub] + list(b_extra))
    c = np.concatenate([m.c, np.zeros(extra_cols) if c_extra is None else c_extra])
    lb = np.concatenate([m.lb, np.zeros(extra_cols) if lb_extra is None else lb_extra])
    ub = np.concatenate([m.ub, np.full(extra_cols, np.inf) if ub_extra is None else ub_extra])
    bounds = list(zip(lb, [None if np.isinf(u) else u for u in ub]))
    t0 = time.time()
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return r, time.time() - t0, A.shape


def fix_and_resolve(P, m, dvcs, x0, n_iter=4, verbose=True):
    """APPROXIMATION (upper bound): free the floor(alpha|S|) hottest voxels, cap the rest at t,
    solve the LP, repeat. Every iterate is feasible for the exact (full-organ) MILP."""
    x = np.asarray(x0, float)
    history = []
    best = None
    for it in range(n_iter):
        d = P.dose(x)
        rows, rhs = [], []
        for dv in dvcs:
            S = m.idx[_key(dv.organ)]
            K = int(np.floor(dv.alpha * len(S)))
            order = np.argsort(-d[S])
            fixed = S[order[K:]]                  # everything but the K hottest
            rows.append(_D_rows(P, fixed, m)); rhs.append(dv.t * np.ones(len(fixed)))
        r, tt, shape = _solve_with_rows(m, rows, rhs)
        rec = dict(iter=it + 1, success=bool(r.success), message=r.message, time=tt)
        if r.success:
            rec["objective"] = float(r.fun)
        history.append(rec)
        if verbose:
            print(f"  fix-and-resolve iter {it+1}: {r.message} obj={rec.get('objective')} ({tt:.0f}s)")
        if not r.success:
            break
        x = r.x[:m.n_x]
        best = dict(x=x, objective=float(r.fun), x_full=r.x)
        if it > 0 and abs(history[-2].get("objective", np.inf) - r.fun) < 1e-7 * abs(r.fun):
            break
    return dict(best=best, history=history)


def cvar_lp(P, m, dvcs, verbose=True):
    """APPROXIMATION (conservative inner approximation, upper bound): the mean of the hottest
    alpha-fraction of S is <= t (Rockafellar-Uryasev). CVaR_alpha <= t implies at most alpha|S|
    voxels exceed t."""
    n0 = m.n_vars
    blocks_A, blocks_b, c_extra_parts, lb_parts, ub_parts = [], [], [], [], []
    n_extra = 0
    layout = []
    for dv in dvcs:
        S = m.idx[_key(dv.organ)]; N = len(S)
        # vars: zeta (free), u_1..u_N >= 0 appended at columns n0 + n_extra
        col0 = n0 + n_extra
        layout.append((dv, col0))
        n_extra += 1 + N
    for dv, col0 in layout:
        S = m.idx[_key(dv.organ)]; N = len(S)
        tot = n0 + n_extra
        # d_i - zeta - u_i <= 0
        zeta_col = sp.csr_matrix((-np.ones(N), (np.arange(N), np.zeros(N, int))), shape=(N, 1))
        A1 = sp.hstack([_D_rows(P, S, m), sp.csr_matrix((N, col0 - n0)), zeta_col,
                        -sp.identity(N, format="csr"), sp.csr_matrix((N, tot - col0 - 1 - N))], format="csr")
        # zeta + sum(u)/(alpha N) <= t
        vals = np.concatenate([[1.0], np.full(N, 1.0 / (dv.alpha * N))])
        A2 = sp.csr_matrix((vals, (np.zeros(N + 1, int), np.arange(col0, col0 + N + 1))), shape=(1, tot))
        blocks_A += [A1, A2]; blocks_b += [np.zeros(N), np.array([dv.t])]
        lb_parts += [np.array([-np.inf]), np.zeros(N)]; ub_parts += [np.array([np.inf]), np.full(N, np.inf)]
    r, tt, shape = _solve_with_rows(m, blocks_A, blocks_b, extra_cols=n_extra,
                                    lb_extra=np.concatenate(lb_parts), ub_extra=np.concatenate(ub_parts))
    if verbose:
        print(f"  CVaR LP: {r.message} obj={r.fun if r.success else None} ({tt:.0f}s) shape={shape}")
    out = dict(success=bool(r.success), message=r.message, solve_time=tt, n_vars=n0 + n_extra, n_constraints=shape[0])
    if r.success:
        out.update(x=r.x[:m.n_x], objective=float(r.fun))
    return out


# ---------------------------------------------------------------- MILP by constraint generation
def _reduced_milp(P, spec, active, dvcs, organ_sets):
    """MILP over a subset of rows. Dropping rows (and dropping voxels from a DVC count while
    keeping the full-organ allowance K) only relaxes the problem, so its bound is valid for
    the full model.  active: dict T68, T56, cap -> voxel arrays; 'dvc' -> list of voxel arrays."""
    D = P.D
    n_x = P.n_beamlets
    n_z = sum(len(v) for v in active["dvc"])
    n = n_x + n_z
    blocks, hi = [], []

    def xrows(S, sign):
        return _pad(sign * D[S], n)

    blocks.append(xrows(active["T68"], -1)); hi.append(-spec.rx68 * np.ones(len(active["T68"])))
    blocks.append(xrows(active["T56"], -1)); hi.append(-spec.rx56 * np.ones(len(active["T56"])))
    blocks.append(xrows(active["cap"], 1)); hi.append(spec.d_cap * np.ones(len(active["cap"])))
    col = n_x
    count_rows, Ks = [], []
    for dv, V in zip(dvcs, active["dvc"]):
        k = len(V)
        M = spec.d_cap - dv.t
        Z = sp.csr_matrix((-M * np.ones(k), (np.arange(k), col + np.arange(k))), shape=(k, n))
        blocks.append(_pad(D[V], n) + Z); hi.append(dv.t * np.ones(k))
        K = int(np.floor(dv.alpha * len(organ_sets[dv.organ]) + 1e-9))    # full-organ allowance
        count_rows.append(sp.csr_matrix((np.ones(k), (np.zeros(k, int), col + np.arange(k))), shape=(1, n)))
        Ks.append(K)
        col += k
    A = sp.vstack(blocks + count_rows, format="csr")
    hi = np.concatenate(hi + [np.array(Ks, float)])
    c = np.zeros(n)
    c[:n_x] = spec.w_R * P.mean_dose_vector(organ_sets["Rectum"]) + \
        spec.w_B * P.mean_dose_vector(organ_sets["Bladder"]) + spec.w_N * P.mean_dose_vector(organ_sets["BODY"])
    lb = np.zeros(n); ub = np.concatenate([np.full(n_x, spec.x_max), np.ones(n_z)])
    integ = np.concatenate([np.zeros(n_x), np.ones(n_z)])
    return c, A, hi, lb, ub, integ, Ks


def solve_milp_cg(P, spec, dvcs, x_init, sets=None, margin=2.0, add_margin=0.5,
                  dvc_floor=20.0, time_limit=600.0, max_iter=8, mip_rel_gap=1e-3, verbose=True):
    """Exact MILP via row generation. Returns the final iterate plus the history.

    Lower bound: the best bound of each reduced MILP (a relaxation of the full MILP).
    Upper bound: objective of any iterate that is feasible for the full model.
    """
    assert not (spec.slack56 or spec.slack68), "row generation assumes no target slack"
    sets = sets if sets is not None else P.sets("disjoint")
    T68, T56 = sets["PTV_68"], sets["PTV_56"]
    cap_all = np.unique(np.concatenate([T68, T56, sets["Rectum"], sets["Bladder"]]))
    tol = 1e-6
    d0 = P.dose(x_init)
    active = dict(
        T68=T68[d0[T68] < spec.rx68 + margin],
        T56=T56[d0[T56] < spec.rx56 + margin],
        cap=cap_all[d0[cap_all] > spec.d_cap - margin],
        dvc=[sets[dv.organ][d0[sets[dv.organ]] >= dv.t - dvc_floor] for dv in dvcs])
    history, best_lb = [], -np.inf
    t_start = time.time()
    res = None
    for it in range(1, max_iter + 1):
        remaining = time_limit - (time.time() - t_start)
        if remaining <= 5:
            break
        c, A, hi, lb, ub, integ, Ks = _reduced_milp(P, spec, active, dvcs, sets)
        t0 = time.time()
        r = milp(c, integrality=integ, bounds=Bounds(lb, ub),
                 constraints=LinearConstraint(A, -np.inf, hi),
                 options=dict(time_limit=remaining, mip_rel_gap=mip_rel_gap))
        tt = time.time() - t0
        rec = dict(iter=it, rows=A.shape[0], cols=A.shape[1], binaries=int(integ.sum()), nnz=int(A.nnz),
                   status=int(r.status), message=r.message, time=tt,
                   objective=None if r.x is None else float(r.fun),
                   best_bound=getattr(r, "mip_dual_bound", None), mip_gap=getattr(r, "mip_gap", None),
                   nodes=getattr(r, "mip_node_count", None),
                   n_active=dict(T68=len(active["T68"]), T56=len(active["T56"]), cap=len(active["cap"]),
                                 dvc=[len(v) for v in active["dvc"]]))
        # A reduced MILP stopped at its time limit still provides a valid bound.
        if rec["best_bound"] is not None and np.isfinite(rec["best_bound"]):
            best_lb = max(best_lb, rec["best_bound"]) if r.status == 0 else max(best_lb, rec["best_bound"])
        if r.x is None:
            rec["note"] = "no incumbent"
            history.append(rec)
            if verbose:
                print(f"  CG iter {it}: {r.message} rows={A.shape[0]} bin={int(integ.sum())} ({tt:.0f}s)")
            break
        x = r.x[:P.n_beamlets]
        d = P.dose(x)
        # violations on the FULL model
        v68 = T68[d[T68] < spec.rx68 * (1 - tol)]
        v56 = T56[d[T56] < spec.rx56 * (1 - tol)]
        vcap = cap_all[d[cap_all] > spec.d_cap * (1 + tol)]
        vdvc, full_ok = [], True
        for dv, V in zip(dvcs, active["dvc"]):
            S = sets[dv.organ]
            over = S[d[S] > dv.t * (1 + tol)]
            n_over = len(over)
            K = int(np.floor(dv.alpha * len(S) + 1e-9))
            missing = np.setdiff1d(over, V)              # hot voxels not yet modelled
            vdvc.append(missing)
            full_ok &= n_over <= K
        rec.update(viol=dict(T68=len(v68), T56=len(v56), cap=len(vcap), dvc_missing=[len(v) for v in vdvc]),
                   full_feasible=bool(len(v68) == 0 and len(v56) == 0 and len(vcap) == 0 and full_ok))
        history.append(rec)
        if verbose:
            print(f"  CG iter {it}: {r.message.split('(')[0].strip()} obj={rec['objective']:.4f} "
                  f"bound={rec['best_bound']:.4f} gap={rec['mip_gap']} rows={A.shape[0]} "
                  f"bin={int(integ.sum())} nodes={rec['nodes']} ({tt:.0f}s) viol={rec['viol']}")
        res = dict(x=x, objective=rec["objective"], status=rec["status"], message=r.message,
                   full_feasible=rec["full_feasible"], best_bound_reduced=rec["best_bound"])
        if rec["full_feasible"] and not any(len(v) for v in vdvc):
            break
        # add violated rows and those within add_margin of binding
        active["T68"] = np.union1d(active["T68"], T68[d[T68] < spec.rx68 + add_margin])
        active["T56"] = np.union1d(active["T56"], T56[d[T56] < spec.rx56 + add_margin])
        active["cap"] = np.union1d(active["cap"], cap_all[d[cap_all] > spec.d_cap - add_margin])
        active["dvc"] = [np.union1d(V, S_new) for V, S_new in
                         zip(active["dvc"], [sets[dv.organ][d[sets[dv.organ]] > dv.t - add_margin] for dv in dvcs])]
    out = dict(history=history, lower_bound=best_lb if np.isfinite(best_lb) else None,
               total_time=time.time() - t_start)
    if res is not None:
        out.update(res)
        ub = res["objective"] if res["full_feasible"] else None
        out["upper_bound"] = ub
        out["gap_full"] = None if (ub is None or out["lower_bound"] is None) else (ub - out["lower_bound"]) / abs(ub)
    return out
