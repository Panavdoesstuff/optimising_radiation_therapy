"""Rectal V50 improvement: hinge-surrogate LPs, reweighting, fix-and-resolve from better rankings.
All plans are feasible points (upper bounds). Body cap by row generation, as in easy_lp."""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from . import config, easy_lp

T = 50.0
METHOD = "highs"


def solve_hinge(P, spec, sets, rows, hard=(), hinge=(), w=None, lam=1.0, max_iter=30):
    """min cost(x) + lam*sum_i w_i e_i  s.t. coverage, caps, d_i<=T for i in hard,
    d_i - e_i <= T, e>=0 for i in hinge. Body cap by row generation (rows = initial body rows).
    Returns dict(x, objective (true plan cost c@x), hinge_val, rows)."""
    N = sets["BODY"]
    hinge = np.asarray(hinge, np.int64)
    hard = np.asarray(hard, np.int64)
    w = np.ones(len(hinge)) if w is None else np.asarray(w, float)
    rows = np.asarray(rows, np.int64)
    nh = len(hinge)
    for it in range(max_iter):
        ex = [(hard, T)] if len(hard) else []
        m = easy_lp.build_model(P, spec, sets, extra_cap=rows, extra_ub=ex)
        nx = m.n_x
        A = m.A_ub
        b = m.b_ub
        c = m.c
        if nh:
            Ah = sp.hstack([P.D[hinge], -sp.identity(nh, format="csr")], format="csr")
            A = sp.hstack([A, sp.csr_matrix((A.shape[0], nh))], format="csr")
            A = sp.vstack([A, Ah], format="csr")
            b = np.concatenate([b, T * np.ones(nh)])
            c = np.concatenate([c, lam * w])
        ub = np.concatenate([m.ub, np.full(nh, np.inf)])
        lb = np.concatenate([m.lb, np.zeros(nh)])
        r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, [None if np.isinf(u) else u for u in ub])),
                    method=METHOD)
        if not r.success:
            return dict(success=False, message=r.message, rows=rows)
        x = r.x[:nx]
        d = P.dose(x)
        over = N[d[N] > spec.d_cap * (1 + config.BODY_ROWGEN_TOL)]
        if len(over) == 0:
            return dict(success=True, x=x, cost=float(m.c @ x), hinge=r.x[nx:], total=float(r.fun),
                        rows=rows, duals_ub=r.ineqlin.marginals, n_hard=len(hard))
        rows = np.union1d(rows, N[d[N] > spec.d_cap - config.BODY_ROWGEN_MARGIN])
    return dict(success=False, message="rowgen not converged", rows=rows)


def nabove(P, S, x):
    d = P.dose(x)
    return int((d[S] > T * (1 + 1e-6)).sum())


class Persistent:
    """One persistent highspy model: variables [x (n_x), e (one per rectal voxel)].
    Rows: base model rows (coverage, caps, generated body rows) + D_R x - e <= T.
    Per-voxel mode through the bounds/cost of e_i: hard cap (ub 0), hinge (cost lam*w_i), free (ub inf, cost 0).
    Body cap rows are added by row generation, never removed; warm starts between solves."""

    def __init__(self, P, spec, sets, rows):
        import highspy
        self.hs = highspy
        self.P, self.spec, self.sets = P, spec, sets
        self.R = sets["Rectum"]; self.N = sets["BODY"]
        m = easy_lp.build_model(P, spec, sets, extra_cap=np.asarray(rows, np.int64))
        self.m = m; self.nx = m.n_x; nR = len(self.R); self.nR = nR
        h = highspy.Highs(); h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        lb = np.concatenate([m.lb, np.zeros(nR)]); ub = np.concatenate([m.ub, np.full(nR, highspy.kHighsInf)])
        h.addVars(self.nx + nR, lb, ub)
        A = sp.hstack([m.A_ub, sp.csr_matrix((m.A_ub.shape[0], nR))], format="csr")
        Ah = sp.hstack([P.D[self.R], -sp.identity(nR, format="csr")], format="csr")
        A = sp.vstack([A, Ah], format="csr")
        b = np.concatenate([m.b_ub, T * np.ones(nR)])
        h.addRows(A.shape[0], np.full(A.shape[0], -highspy.kHighsInf), b, A.nnz,
                  A.indptr[:-1].astype(np.int32), A.indices.astype(np.int32), A.data)
        self.h = h
        self.cx = m.c.copy()
        self.in_rows = set(np.asarray(rows, np.int64).tolist()) | set(m.idx["cap"].tolist())
        self.idx_e = np.arange(self.nx, self.nx + nR, dtype=np.int32)
        self.idx_all = np.arange(self.nx + nR, dtype=np.int32)

    def solve(self, hard_mask=None, w=None, lam=0.0, max_iter=40, free_mask=None):
        """hard_mask (bool over rectal voxels): d<=T. free_mask: no constraint (cost 0).
        Others get hinge cost lam*w_i (w default 1). Returns dict like solve_hinge."""
        hs, h, nR, nx = self.hs, self.h, self.nR, self.nx
        hard = np.zeros(nR, bool) if hard_mask is None else hard_mask
        free = np.zeros(nR, bool) if free_mask is None else free_mask
        wv = np.ones(nR) if w is None else w
        cost = np.where(hard | free, 0.0, lam * wv)
        eub = np.where(hard, 0.0, hs.kHighsInf)
        h.changeColsCost(nx + nR, self.idx_all, np.concatenate([self.cx, cost]))
        h.changeColsBounds(nR, self.idx_e, np.zeros(nR), eub)
        N = self.N
        for it in range(max_iter):
            h.run()
            st = h.getModelStatus()
            if st != hs.HighsModelStatus.kOptimal:
                return dict(success=False, message=str(st))
            sol = np.array(h.getSolution().col_value)
            x = sol[:nx]
            d = self.P.dose(x)
            over = N[d[N] > self.spec.d_cap * (1 + config.BODY_ROWGEN_TOL)]
            if len(over) == 0:
                return dict(success=True, x=x, e=sol[nx:], cost=float(self.cx @ x), d=d)
            new = np.setdiff1d(N[d[N] > self.spec.d_cap - config.BODY_ROWGEN_MARGIN],
                               np.fromiter(self.in_rows, np.int64))
            A = self.P.D[new]
            A = sp.hstack([A, sp.csr_matrix((len(new), nR))], format="csr")
            h.addRows(len(new), np.full(len(new), -hs.kHighsInf), self.spec.d_cap * np.ones(len(new)), A.nnz,
                      A.indptr[:-1].astype(np.int32), A.indices.astype(np.int32), A.data)
            self.in_rows |= set(new.tolist())
        return dict(success=False, message="rowgen not converged")
