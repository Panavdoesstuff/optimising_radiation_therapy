import numpy as np
import pytest
import scipy.sparse as sp
from src import config, io_cort, problem

pytestmark = pytest.mark.skipif(not config.CASE_DIRS["tg119"].exists(), reason="TG119 data missing")


@pytest.fixture(scope="module")
def tg():
    return problem.build("tg119", config.TG119_ANGLES, use_cache=False)


def test_shapes(tg):
    nvox = np.prod(tg.grid["shape"])
    assert tg.D.shape == (nvox, 418)
    assert tg.D.dtype == np.float64 and sp.isspmatrix_csr(tg.D)
    assert len(tg.beam_of_column) == 418
    assert np.bincount(tg.beam_of_column).sum() == 418 and tg.beam_of_column.max() == 4


def test_voilist_ranges(tg):
    nvox = np.prod(tg.grid["shape"])
    for mode in ("raw", "disjoint"):
        for name, v in tg.sets(mode).items():
            assert v.dtype == np.int64 and len(v) > 0
            assert np.all(np.diff(v) > 0), name           # sorted, unique
            assert v.min() >= 0 and v.max() < nvox, name  # 0-based, in grid


def test_disjoint(tg):
    s = tg.sets("disjoint")
    assert len(np.intersect1d(s["Core"], s["OuterTarget"])) == 0
    assert len(np.intersect1d(s["BODY"], s["OuterTarget"])) == 0


def test_mean_dose_vector(tg):
    S = tg.sets("raw")["Core"]
    x = np.random.default_rng(0).random(tg.n_beamlets)
    assert tg.mean_dose_vector(S) @ x == pytest.approx(tg.dose(x)[S].mean())


def test_beaminfo_matches_D(tg):
    for b, g in enumerate(config.TG119_ANGLES):
        bi = io_cort.load_beaminfo("tg119", g)
        assert bi["num_beamlets"] == (tg.beam_of_column == b).sum()
        assert len(bi["x"]) == len(bi["y"]) == bi["num_beamlets"]


def test_grid():
    assert io_cort.load_grid("tg119")["shape"] == (167, 167, 129)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    a = problem.build("tg119", config.TG119_ANGLES)
    assert any(tmp_path.iterdir())
    b = problem.build("tg119", config.TG119_ANGLES)
    assert (a.D != b.D).nnz == 0
    assert np.array_equal(a.beam_of_column, b.beam_of_column)


def test_h5py_sparse_path(tmp_path):
    """v7.3 files store sparse as groups data/ir/jc; check our reader maps them."""
    h5py = pytest.importorskip("h5py")
    A = sp.random(20, 7, density=0.3, random_state=1, format="csc")
    with h5py.File(tmp_path / "x.mat", "w") as f:
        g = f.create_group("D")
        g.create_dataset("data", data=A.data)
        g.create_dataset("ir", data=A.indices)
        g.create_dataset("jc", data=A.indptr)
        g.attrs["MATLAB_sparse"] = 20
        f.create_dataset("v", data=np.arange(1, 6, dtype=float).reshape(1, 5))
    m = io_cort.load_mat(tmp_path / "x.mat")
    assert (m["D"] != A).nnz == 0
    assert m["v"].shape == (5, 1)
