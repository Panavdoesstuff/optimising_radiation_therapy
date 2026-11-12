"""Synthetic (no data) checks on the beam-angle-selection MILP structure."""
import numpy as np
import scipy.sparse as sp
import pytest
from src import easy_lp, hard_bao


def _toy(nb=4, per=3, seed=0):
    rng = np.random.default_rng(seed)
    n_x = nb * per
    nT68, nT56, nR, nB = 6, 4, 3, 3
    rows = nT68 + nT56 + nR + nB
    D = sp.csr_matrix(rng.uniform(0.5, 1.5, size=(rows, n_x)) * 0.02)
    r = np.arange(rows)
    loc = dict(T68=r[:nT68], T56=r[nT68:nT68 + nT56], R=r[nT68 + nT56:nT68 + nT56 + nR],
               B=r[-nB:], cap=r)
    cost = {k: np.asarray(D[loc[k2]].mean(axis=0)).ravel() for k, k2 in (("R", "R"), ("B", "B"))}
    cost["N"] = np.asarray(D.mean(axis=0)).ravel()
    return hard_bao.Reduced(list(range(0, 360, 360 // nb)), D, r, loc, cost,
                            np.repeat(np.arange(nb), per).astype(np.int32))


@pytest.mark.parametrize("K", [1, 2, 3])
def test_bao_respects_K(K):
    red = _toy()
    spec = easy_lp.Spec(rx68=1.0, rx56=0.8, d_cap_factor=3.0, x_max=1e3)
    r = hard_bao.solve_bao(red, spec, K, time_limit=30)
    assert r["success"]
    assert len(r["beams"]) <= K
    unused = ~np.isin(red.beam_of_column, r["beams"])
    assert np.all(r["x"][unused] <= 1e-7)
    # objective of the MILP equals the LP restricted to the chosen beams (optimal x for those beams)
    lp = hard_bao.restricted_lp(red, spec, r["beams"])
    assert lp["objective"] == pytest.approx(r["objective"], rel=1e-5)


def test_bao_more_beams_never_worse():
    red = _toy()
    spec = easy_lp.Spec(rx68=1.0, rx56=0.8, d_cap_factor=3.0, x_max=1e3)
    objs = [hard_bao.solve_bao(red, spec, K, time_limit=30)["objective"] for K in (1, 2, 3, 4)]
    assert all(a >= b - 1e-9 for a, b in zip(objs, objs[1:]))
    assert objs[-1] == pytest.approx(hard_bao.all_beams_lp(red, spec)["objective"], rel=1e-6)


def test_heuristic_returns_K_beams():
    red = _toy(nb=6)
    spec = easy_lp.Spec(rx68=1.0, rx56=0.8, d_cap_factor=3.0, x_max=1e3)
    ref = hard_bao.all_beams_lp(red, spec)
    for K in (2, 3):
        h = hard_bao.heuristic(red, spec, K, ref["x"], verbose=False)
        assert len(h["beams"]) == K
