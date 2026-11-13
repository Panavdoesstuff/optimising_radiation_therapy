"""Hard problem: beam angle selection (MILP).

min  w_R mean_R(d) + w_B mean_B(d) + w_N mean_N(d)
s.t. same dose constraints as the easy LP,
     x_j <= x_max * y_b   (beamlet j belongs to beam b),
     sum_b y_b <= K,   y_b in {0, 1}.

Memory: only rows of voxels that carry a constraint (T68, T56, R, B) are stacked; the BODY,
rectum and bladder mean-dose cost vectors are accumulated beam by beam.
"""
import time
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from . import config, io_cort
from .solvers import solve_mip


@dataclass
class Reduced:
    angles: list
    D: sp.csr_matrix            # rows = `rows` (constraint voxels, original numbering), cols = beamlets
    rows: np.ndarray            # original voxel indices of the rows of D
    loc: dict                   # T68, T56, R, B, cap -> local row positions in D
    cost: dict                  # R, B, N -> unweighted mean-dose vectors (length n_beamlets)
    beam_of_column: np.ndarray

    @property
    def n_beamlets(self):
        return self.D.shape[1]


def _cache_path(angles):
    return config.CACHE_DIR / f"prostate_bao_{angles[0]}-{angles[-1]}_n{len(angles)}_step{angles[1]-angles[0]}.npz"


def build_reduced(angles, sets, use_cache=True):
    path = _cache_path(angles)
    T68, T56, R, B, N = (sets[k] for k in ("PTV_68", "PTV_56", "Rectum", "Bladder", "BODY"))
    rows = np.unique(np.concatenate([T68, T56, R, B]))
    pos = lambda S: np.searchsorted(rows, S)
    loc = dict(T68=pos(T68), T56=pos(T56), R=pos(R), B=pos(B), cap=np.arange(len(rows)))
    if use_cache and path.exists():
        z = np.load(path)
        D = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return Reduced(list(angles), D, rows, loc, dict(R=z["cR"], B=z["cB"], N=z["cN"]), z["boc"])
    mats, cR, cB, cN, boc = [], [], [], [], []
    for b, g in enumerate(angles):
        Db = io_cort.load_D("prostate", g)
        mats.append(Db[rows])
        cR.append(np.asarray(Db[R].mean(axis=0)).ravel())
        cB.append(np.asarray(Db[B].mean(axis=0)).ravel())
        cN.append(np.asarray(Db[N].mean(axis=0)).ravel())
        boc.append(np.full(Db.shape[1], b, dtype=np.int32))
    D = sp.hstack(mats, format="csr")
    cR, cB, cN, boc = map(np.concatenate, (cR, cB, cN, boc))
    if use_cache:
        config.CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, data=D.data, indices=D.indices, indptr=D.indptr, shape=np.array(D.shape),
                 cR=cR, cB=cB, cN=cN, boc=boc)
    return Reduced(list(angles), D, rows, loc, dict(R=cR, B=cB, N=cN), boc)


def _cost(red, spec):
    return spec.w_R * red.cost["R"] + spec.w_B * red.cost["B"] + spec.w_N * red.cost["N"]


def _dose_rows(red, spec, cols=None):
    """A_ub, b_ub for the dose constraints on columns `cols` (all if None)."""
    D = red.D if cols is None else red.D[:, cols]
    A = sp.vstack([-D[red.loc["T68"]], -D[red.loc["T56"]], D[red.loc["cap"]]], format="csr")
    b = np.concatenate([-spec.rx68 * np.ones(len(red.loc["T68"])),
                        -spec.rx56 * np.ones(len(red.loc["T56"])),
                        spec.d_cap * np.ones(len(red.loc["cap"]))])
    return A, b


def restricted_lp(red, spec, beams):
    """LP using only the beams in `beams` (positions in red.angles). Feasible point for any K>=len(beams)."""
    cols = np.flatnonzero(np.isin(red.beam_of_column, beams))
    A, b = _dose_rows(red, spec, cols)
    t0 = time.time()
    r = linprog(_cost(red, spec)[cols], A_ub=A, b_ub=b, bounds=(0, spec.x_max), method="highs")
    out = dict(success=bool(r.success), message=r.message, time=time.time() - t0, beams=sorted(int(b) for b in beams))
    if r.success:
        x = np.zeros(red.n_beamlets); x[cols] = r.x
        out.update(x=x, objective=float(r.fun))
    return out


def all_beams_lp(red, spec):
    """Lower bound for every K: all candidates allowed."""
    return restricted_lp(red, spec, list(range(len(red.angles))))


def heuristic(red, spec, K, x_ref, n_swaps=2, verbose=True):
    """Primal heuristic: top-K beams by intensity in a reference plan, then best single swaps."""
    nb = len(red.angles)
    w = np.bincount(red.beam_of_column, weights=x_ref, minlength=nb)
    cur = list(np.argsort(-w)[:K])
    best = restricted_lp(red, spec, cur)
    if verbose:
        print(f"    heuristic K={K}: top-K {sorted(red.angles[b] for b in cur)} -> {best.get('objective')}")
    for _ in range(n_swaps):
        if not best["success"]:
            break
        x = best["x"]
        wk = np.bincount(red.beam_of_column, weights=x, minlength=nb)
        base = list(cur)                                 # fixed for this round
        out_b = min(base, key=lambda b: wk[b])           # weakest chosen beam
        cand = [b for b in np.argsort(-w) if b not in base][:3]
        improved = False
        for b_in in cand:
            trial = [b for b in base if b != out_b] + [b_in]
            assert len(trial) == K
            r = restricted_lp(red, spec, trial)
            if r["success"] and r["objective"] < best["objective"] - 1e-9:
                best, cur, improved = r, trial, True
        if verbose:
            print(f"    swap: {sorted(red.angles[b] for b in cur)} -> {best['objective']:.4f}")
        if not improved:
            break
    return best


def drop_one(red, spec, beams):
    """Best plan obtained by removing one beam from `beams` (positions in red.angles)."""
    best = None
    for b in beams:
        r = restricted_lp(red, spec, [c for c in beams if c != b])
        if r["success"] and (best is None or r["objective"] < best["objective"]):
            best = r
    return best


def solve_bao(red, spec, K, time_limit=config.BAO_TIME_LIMIT, start=None, mip_rel_gap=1e-4, verbose=False):
    """Exact MILP. start: a feasible plan dict with 'x' and 'beams' (warm start, optional)."""
    nb = len(red.angles); n_x = red.n_beamlets
    A_d, b_d = _dose_rows(red, spec)
    # x_j - x_max y_b <= 0
    J = np.arange(n_x)
    link = sp.hstack([sp.identity(n_x, format="csr"),
                      sp.csr_matrix((-spec.x_max * np.ones(n_x), (J, red.beam_of_column)), shape=(n_x, nb))], format="csr")
    card = sp.csr_matrix((np.ones(nb), (np.zeros(nb, int), n_x + np.arange(nb))), shape=(1, n_x + nb))
    A = sp.vstack([sp.hstack([A_d, sp.csr_matrix((A_d.shape[0], nb))], format="csr"), link, card], format="csr")
    hi = np.concatenate([b_d, np.zeros(n_x), [K]])
    c = np.concatenate([_cost(red, spec), np.zeros(nb)])
    lb = np.zeros(n_x + nb); ub = np.concatenate([np.full(n_x, spec.x_max), np.ones(nb)])
    integ = np.concatenate([np.zeros(n_x), np.ones(nb)])
    v0 = None
    if start is not None and start.get("success"):
        y0 = np.zeros(nb); y0[start["beams"]] = 1
        v0 = np.concatenate([start["x"], y0])
    r = solve_mip(c, A, hi, lb, ub, integ, time_limit=time_limit, mip_rel_gap=mip_rel_gap, start=v0, disp=verbose)
    r.update(K=K, n_vars=len(c), n_constraints=A.shape[0], n_binaries=nb, nnz=int(A.nnz))
    if r.get("x") is not None:
        y = r["x"][n_x:]
        r["beams"] = [int(b) for b in np.flatnonzero(y > 0.5)]
        r["angles_selected"] = [red.angles[b] for b in r["beams"]]
        r["x"] = r["x"][:n_x]
    return r
