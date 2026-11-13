"""Easy problem: the LP.  Also provides the shared model builder used by the hard MILP.

Variables: x (beamlet intensities, n_x), then optional slacks s (target shortfall).
Dose d = D x is in Gy (D is Gy per MU).
"""
import time
from dataclasses import dataclass, field, replace, asdict
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from . import config


@dataclass
class Spec:
    w_R: float = config.WEIGHTS["Rectum"]
    w_B: float = config.WEIGHTS["Bladder"]
    w_N: float = config.WEIGHTS["BODY"]
    rx68: float = config.RX_T68
    rx56: float = config.RX_T56
    d_cap_factor: float = config.D_CAP_FACTOR
    x_max: float = config.X_MAX_EASY
    rho: float = config.RHO_SLACK
    slack56: bool = False
    slack68: bool = False
    body_cap: bool = config.BODY_CAP     # d_i <= d_cap on BODY voxels outside the targets

    @property
    def d_cap(self):
        return self.d_cap_factor * self.rx68


@dataclass
class Model:
    """min c@v  s.t.  A_ub v <= b_ub, lb <= v <= ub.  v = [x, s, (extra)]"""
    c: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    n_x: int
    n_s: int
    spec: Spec
    idx: dict            # voxel index sets used (T68, T56, R, B, N, cap)
    cost_x: dict         # per-term cost vectors on x: R, B, N (unweighted mean-dose vectors)
    row_groups: dict     # name -> slice of rows in A_ub
    integer: np.ndarray = None
    notes: dict = field(default_factory=dict)

    @property
    def n_vars(self):
        return len(self.c)


def build_model(P, spec: Spec, sets=None, extra_cap=None, extra_ub=()):
    """extra_cap: additional voxels (e.g. generated body rows) that get d_i <= d_cap.
    extra_ub: list of (voxel_idx, t) adding d_i <= t (used by fix-and-resolve)."""
    sets = sets if sets is not None else P.sets("disjoint")
    D = P.D
    T68, T56, R, B, N = (sets[k] for k in ("PTV_68", "PTV_56", "Rectum", "Bladder", "BODY"))
    n_x = P.n_beamlets
    cR, cB, cN = (P.mean_dose_vector(S) for S in (R, B, N))
    n_s68 = len(T68) if spec.slack68 else 0
    n_s56 = len(T56) if spec.slack56 else 0
    n_s = n_s68 + n_s56

    c = np.zeros(n_x + n_s)
    c[:n_x] = spec.w_R * cR + spec.w_B * cB + spec.w_N * cN
    if n_s68:
        c[n_x:n_x + n_s68] = spec.rho / len(T68)
    if n_s56:
        c[n_x + n_s68:] = spec.rho / len(T56)

    blocks, rhs, groups, r0 = [], [], {}, 0

    def add(name, A, b):
        nonlocal r0
        blocks.append(A.tocsr()); rhs.append(b)
        groups[name] = slice(r0, r0 + A.shape[0]); r0 += A.shape[0]

    def lower(name, S, rx, s_off, ns):
        """-D[S] x (- s) <= -rx   (i.e. d_i (+ s_i) >= rx)."""
        parts = [-D[S]]
        if ns:
            parts += [sp.csr_matrix((len(S), s_off)), -sp.identity(len(S), format="csr"),
                      sp.csr_matrix((len(S), n_s - s_off - ns))]
        else:
            parts += [sp.csr_matrix((len(S), n_s))]
        add(name, sp.hstack(parts, format="csr"), -rx * np.ones(len(S)))

    lower("T68_lo", T68, spec.rx68, 0, n_s68)
    lower("T56_lo", T56, spec.rx56, n_s68, n_s56)

    parts = [T68, T56, R, B] + ([np.asarray(extra_cap, np.int64)] if extra_cap is not None else [])
    cap_idx = np.unique(np.concatenate(parts))
    Acap = sp.hstack([D[cap_idx], sp.csr_matrix((len(cap_idx), n_s))], format="csr")
    add("cap", Acap, spec.d_cap * np.ones(len(cap_idx)))
    for k, (idx, t) in enumerate(extra_ub):
        add(f"ub{k}", sp.hstack([D[idx], sp.csr_matrix((len(idx), n_s))], format="csr"), t * np.ones(len(idx)))

    A_ub = sp.vstack(blocks, format="csr")
    lb = np.zeros(n_x + n_s)
    ub = np.concatenate([np.full(n_x, spec.x_max), np.full(n_s, np.inf)])
    return Model(c=c, A_ub=A_ub, b_ub=np.concatenate(rhs), lb=lb, ub=ub, n_x=n_x, n_s=n_s,
                 spec=spec, idx=dict(T68=T68, T56=T56, R=R, B=B, N=N, cap=cap_idx),
                 cost_x=dict(R=cR, B=cB, N=cN), row_groups=groups)


def solve_lp(m: Model):
    """HiGHS via linprog. Returns dict with primal/dual objective for the duality gap."""
    t0 = time.time()
    bounds = list(zip(m.lb, [None if np.isinf(u) else u for u in m.ub]))
    r = linprog(m.c, A_ub=m.A_ub, b_ub=m.b_ub, bounds=bounds,
                method="highs")
    out = dict(status=int(r.status), message=r.message, solve_time=time.time() - t0,
               success=bool(r.success))
    if r.success:
        y = r.ineqlin.marginals; lo = r.lower.marginals; up = r.upper.marginals
        ub_fin = np.where(np.isinf(m.ub), 0.0, m.ub)
        dual_obj = float(m.b_ub @ y + m.lb @ lo + ub_fin @ up)
        out.update(x=r.x, objective=float(r.fun), dual_objective=dual_obj,
                   duality_gap=float(r.fun - dual_obj), iterations=int(r.nit))
    return out


def body_violations(d, N, spec):
    """Body voxels above the cap (relative tolerance) and the max relative violation."""
    dN = d[N]
    over = N[dN > spec.d_cap * (1 + config.BODY_ROWGEN_TOL)]
    return over, max(float(dN.max()) - spec.d_cap, 0.0) / spec.d_cap


def solve_lp_rowgen(P, spec: Spec, sets=None, extra_ub=(), body0=None, max_iter=30, verbose=True):
    """Solve the LP with the body cap enforced by row generation.

    Each round solves with the current body rows, recomputes d = D x on the full grid, and adds
    every violated body voxel plus those within BODY_ROWGEN_MARGIN Gy of the cap. A reduced LP is
    a relaxation, so when its solution has no violation it is optimal for the full problem.
    """
    sets = sets if sets is not None else P.sets("disjoint")
    N = sets["BODY"]
    rows = np.asarray(body0 if body0 is not None else [], np.int64)
    hist = []
    for it in range(1, max_iter + 1):
        m = build_model(P, spec, sets, extra_cap=rows if spec.body_cap else None, extra_ub=extra_ub)
        r = solve_lp(m)
        rec = dict(iter=it, body_rows=int(len(rows)), rows=int(m.A_ub.shape[0]),
                   status=r["message"], time=r["solve_time"], objective=r.get("objective"))
        if not r["success"]:
            hist.append(rec)
            r.update(model=m, rowgen=hist, body_rows=rows, converged=False)
            return r
        d = P.dose(r["x"][:m.n_x])
        over, rel = body_violations(d, N, spec)
        rec.update(n_violated=int(len(over)), max_rel_violation=rel)
        hist.append(rec)
        if verbose:
            print(f"    rowgen {it}: obj {r['objective']:.4f}  body rows {len(rows)}  "
                  f"violated {len(over)}  max rel viol {rel:.1e}  ({r['solve_time']:.0f}s)")
        if not spec.body_cap or len(over) == 0:
            r.update(model=m, rowgen=hist, body_rows=rows, converged=True)
            return r
        rows = np.union1d(rows, N[d[N] > spec.d_cap - config.BODY_ROWGEN_MARGIN])
    r.update(model=m, rowgen=hist, body_rows=rows, converged=False)
    return r


def solve_easy(P, spec: Spec = None, sets=None, ladder=True, verbose=True):
    """Solve the easy LP, walking the feasibility ladder if the full model is infeasible."""
    spec = spec or Spec()
    rungs = [
        (1, "full model", spec),
        (2, f"raise d_cap to {config.D_CAP_LADDER}*68", replace(spec, d_cap_factor=config.D_CAP_LADDER)),
        (3, "slack on T56 lower bound", replace(spec, d_cap_factor=config.D_CAP_LADDER, slack56=True)),
        (4, "slack on T56 and T68", replace(spec, d_cap_factor=config.D_CAP_LADDER, slack56=True, slack68=True)),
    ]
    log = []
    for level, desc, sp_ in rungs:
        res = solve_lp_rowgen(P, sp_, sets, verbose=verbose)
        m = res["model"]
        log.append(dict(level=level, desc=desc, status=res["status"], message=res["message"]))
        if verbose:
            print(f"  ladder level {level} ({desc}): {res['message']}")
        if (res["success"] and res["converged"]) or not ladder:
            res.update(model=m, level=level, level_desc=desc, ladder_log=log)
            return res
    res.update(model=m, level=None, level_desc="infeasible at all levels", ladder_log=log)
    return res
