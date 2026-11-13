"""Body-cap row generation on a synthetic problem (no data needed)."""
import numpy as np
import pytest
import scipy.sparse as sp
from src import easy_lp, problem, verify


def _synthetic(n_x=8, n_body=200):
    """Each beamlet gives the targets 0.1 per unit and has one hot 'entrance' voxel (0.3) plus a
    cold background whose level differs by beamlet. Without a body cap the LP piles intensity on
    the beamlet with the coldest background, overdosing its entrance voxel."""
    rng = np.random.default_rng(0)
    n_t = 20
    T = np.full((2 * n_t, n_x), 0.1)
    bg = np.linspace(0.002, 0.02, n_x)
    Bd = rng.uniform(0.5, 1.5, size=(n_body, n_x)) * bg
    Bd[np.arange(n_x), np.arange(n_x)] = 0.3                 # entrance voxels
    D = sp.csr_matrix(np.vstack([T, Bd]))
    n_vox = D.shape[0]
    P = problem.Problem("synthetic", [0], D, np.zeros(n_x, np.int32), dict(shape=(n_vox, 1, 1)))
    idx = np.arange(n_vox)
    body = idx[2 * n_t:]
    sets = {"PTV_68": idx[:n_t], "PTV_56": idx[n_t:2 * n_t], "Rectum": body[n_x:n_x + 20],
            "Bladder": body[n_x + 20:n_x + 50], "BODY": body}
    return P, sets


def _binding_cap(P, sets):
    """A cap factor where the LP is feasible and the body cap actually binds."""
    for f in (1.5, 2.0, 2.5, 3.0):
        off = easy_lp.Spec(rx68=1.0, rx56=0.8, d_cap_factor=f, x_max=1e4, body_cap=False)
        r = easy_lp.solve_lp_rowgen(P, off, sets, verbose=False)
        if r["success"] and P.dose(r["x"][:P.n_beamlets])[sets["BODY"]].max() > off.d_cap * 1.01:
            on = easy_lp.Spec(rx68=1.0, rx56=0.8, d_cap_factor=f, x_max=1e4, body_cap=True)
            if easy_lp.solve_lp_rowgen(P, on, sets, verbose=False)["success"]:
                return off, on
    raise AssertionError("no binding, feasible cap found for the synthetic instance")


def test_rowgen_matches_full_cap():
    P, sets = _synthetic()
    _, spec = _binding_cap(P, sets)
    r = easy_lp.solve_lp_rowgen(P, spec, sets, verbose=False)
    assert r["success"] and r["converged"]
    # the same LP with every body voxel capped up front
    full = easy_lp.solve_lp(easy_lp.build_model(P, spec, sets, extra_cap=sets["BODY"]))
    assert r["objective"] == pytest.approx(full["objective"], rel=1e-7)
    assert len(r["body_rows"]) < len(sets["BODY"])     # row generation used fewer rows
    v = verify.verify(P, sets, spec, r["x"][:P.n_beamlets], solver_objective=r["objective"])
    assert v["all_ok"]
    assert any("BODY" in k for k in v["checks"])


def test_body_cap_costs_and_holds():
    P, sets = _synthetic()
    off, on = _binding_cap(P, sets)
    r_off = easy_lp.solve_lp_rowgen(P, off, sets, verbose=False)
    r_on = easy_lp.solve_lp_rowgen(P, on, sets, verbose=False)
    assert r_on["objective"] >= r_off["objective"] - 1e-9          # a cap can only cost
    assert P.dose(r_on["x"][:P.n_beamlets])[sets["BODY"]].max() <= on.d_cap * (1 + 1e-6)
    assert len(r_on["rowgen"]) >= 2                                 # it really iterated


def test_voxel_dose_floor_is_a_lower_bound():
    from src import hard_dvc
    P, sets = _synthetic()
    _, spec = _binding_cap(P, sets)
    r = easy_lp.solve_lp_rowgen(P, spec, sets, verbose=False)
    d = P.dose(r["x"][:P.n_beamlets])
    vox = sets["Rectum"]
    from scipy.optimize import linprog
    one = easy_lp.build_model(P, spec, sets, extra_cap=r["body_rows"])
    true_min = np.array([linprog(np.asarray(P.D[i].todense()).ravel(), A_ub=one.A_ub, b_ub=one.b_ub,
                                 bounds=list(zip(one.lb, one.ub)), method="highs").fun for i in vox])
    t = float(np.median(true_min))                 # some voxels forced above t, some not
    fl = hard_dvc.voxel_dose_floor(P, spec, sets, vox, t, body_rows=r["body_rows"], verbose=False)
    forced = fl["status"] == "forced"
    assert not (fl["status"] == "open").any() and not (fl["status"] == "unresolved").any()
    assert np.array_equal(forced, true_min > t + 1e-6 * t)     # pruning never changes the answer
    solved = np.isfinite(fl["bound"])
    assert np.all(fl["bound"][solved] <= true_min[solved] + 1e-7)   # bounds are lower bounds
