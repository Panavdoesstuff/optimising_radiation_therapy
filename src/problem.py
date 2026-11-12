"""Build stacked dose matrices, structure index sets and cost vectors."""
import hashlib
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp

from . import config, io_cort

TARGETS = ("PTV_68", "PTV_56", "OuterTarget")


@dataclass
class Problem:
    case: str
    angles: list
    D: sp.csr_matrix            # (n_voxels_total x n_beamlets), full grid row numbering
    beam_of_column: np.ndarray  # beam index (position in `angles`) per column
    grid: dict
    couch: int = 0
    _sets: dict = field(default_factory=dict, repr=False)

    @property
    def n_voxels(self):
        return self.D.shape[0]

    @property
    def n_beamlets(self):
        return self.D.shape[1]

    def sets(self, mode="raw"):
        """Structure index sets (0-based, sorted). mode: 'raw' | 'disjoint'."""
        if mode not in self._sets:
            self._sets[mode] = _make_sets(self.case, mode)
        return self._sets[mode]

    def dose(self, x):
        return self.D @ np.asarray(x, dtype=float)

    def mean_dose_vector(self, S):
        """c_S with mean dose to S == c_S @ x (dense 1D)."""
        return np.asarray(self.D[S, :].mean(axis=0)).ravel()

    def volume_shape(self):
        return self.grid["shape"]

    def to_volume(self, values, idx=None):
        """Scatter per-voxel values to a 3D array indexed [ap, rl, si]  (see volume_axes)."""
        vol = np.zeros(self.n_voxels)
        if idx is None:
            vol[:] = values
        else:
            vol[idx] = values
        return reshape_volume(vol, self.grid)


def reshape_volume(vec, grid):
    """Flat voxel vector -> 3D array indexed [ap, rl, si].

    Voxel 1 is anterior/superior/patient-right; AP varies fastest, SI slowest,
    i.e. a Fortran-order reshape to (ny, nx, nz). Verified by plotting BODY
    (axial slice looks like a body, anterior at row 0, couch at the bottom).
    """
    nx, ny, nz = grid["shape"]
    return np.asarray(vec).reshape((ny, nx, nz), order="F")


def _make_sets(case, mode):
    names = config.STRUCTURES[case]
    raw = {n: io_cort.load_voilist(case, n) for n in names}
    if mode == "raw":
        return raw
    if mode != "disjoint":
        raise ValueError(mode)
    out = dict(raw)
    if case == "prostate":
        out["PTV_56"] = np.setdiff1d(raw["PTV_56"], raw["PTV_68"])
        tgt = np.union1d(raw["PTV_68"], raw["PTV_56"])
        for n in ("Rectum", "Bladder", "BODY"):
            out[n] = np.setdiff1d(raw[n], tgt)
    elif case == "tg119":
        for n in ("Core", "BODY"):
            out[n] = np.setdiff1d(raw[n], raw["OuterTarget"])
    return out


def _cache_path(case, angles, couch):
    key = "-".join(map(str, angles))
    if len(key) > 60:
        key = hashlib.md5(key.encode()).hexdigest()[:12] + f"_n{len(angles)}"
    return config.CACHE_DIR / f"{case}_c{couch}_{key}.npz"


def build(case, gantry_angles, couch=0, use_cache=True):
    """Horizontally stack per-beam D matrices (cached to cache/)."""
    angles = [int(a) for a in gantry_angles]
    path = _cache_path(case, angles, couch)
    grid = io_cort.load_grid(case)
    if use_cache and path.exists():
        z = np.load(path)
        D = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return Problem(case, angles, D, z["beam_of_column"], grid, couch)
    mats, boc = [], []
    for b, g in enumerate(angles):
        Db = io_cort.load_D(case, g, couch)
        mats.append(Db)
        boc.append(np.full(Db.shape[1], b, dtype=np.int32))
    D = sp.hstack(mats, format="csr")
    boc = np.concatenate(boc)
    if use_cache:
        config.CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, data=D.data, indices=D.indices, indptr=D.indptr,
                 shape=np.array(D.shape), beam_of_column=boc)
    return Problem(case, angles, D, boc, grid, couch)
