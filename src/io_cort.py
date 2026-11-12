"""Loaders for CORT dataset files. Handles MATLAB v5 (scipy) and v7.3 (h5py)."""
import re
import warnings
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from . import config


def _case_dir(case):
    return config.CASE_DIRS[case]


def _is_hdf5(path):
    """v7.3 .mat files are HDF5 (signature after a 512-byte MATLAB userblock)."""
    try:
        import h5py
    except ImportError:
        return False
    return h5py.is_hdf5(path)


def _h5_to_python(f, key):
    """Read variable `key` from an HDF5 .mat; sparse groups -> csc_matrix."""
    obj = f[key]
    if hasattr(obj, "keys") and {"data", "ir", "jc"} <= set(obj.keys()):
        data = np.asarray(obj["data"]).ravel()
        ir = np.asarray(obj["ir"]).ravel()
        jc = np.asarray(obj["jc"]).ravel()
        nrows = int(obj.attrs["MATLAB_sparse"]) if "MATLAB_sparse" in obj.attrs else int(ir.max()) + 1
        return sp.csc_matrix((data, ir, jc), shape=(nrows, len(jc) - 1))
    # HDF5 stores arrays transposed relative to MATLAB
    return np.asarray(obj).T


def load_mat(path, key=None):
    """Load a .mat file; returns dict (or the single value for `key`)."""
    path = str(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            m = sio.loadmat(path)
        m = {k: v for k, v in m.items() if not k.startswith("__")}
    except (NotImplementedError, ValueError) as e:
        if not _is_hdf5(path):
            raise
        import h5py
        with h5py.File(path, "r") as f:
            m = {k: _h5_to_python(f, k) for k in f.keys() if not k.startswith("#")}
    return m[key] if key is not None else m


def mat_path(case, gantry, couch, kind):
    return _case_dir(case) / f"Gantry{gantry}_Couch{couch}_{kind}.mat"


def load_D(case, gantry, couch=0):
    """Dose matrix (voxels x beamlets), Gy per MU, as float64 CSR."""
    D = load_mat(mat_path(case, gantry, couch, "D"), "D")
    return sp.csr_matrix(D, dtype=np.float64)


def load_voilist(case, name):
    """Sorted unique 0-based int64 voxel indices of a structure."""
    stem = config.STRUCTURES[case].get(name, name)
    v = load_mat(_case_dir(case) / f"{stem}_VOILIST.mat", "v")
    return np.unique(np.asarray(v).ravel().astype(np.int64) - 1)


def load_beaminfo(case, gantry, couch=0):
    m = load_mat(mat_path(case, gantry, couch, "BEAMINFO"))
    return dict(
        num_beamlets=int(np.ravel(m["numBeamlets"])[0]),
        gantry=float(np.ravel(m["gantryAngle"])[0]),
        couch=float(np.ravel(m["couchAngle"])[0]),
        x=np.ravel(m["x"]).astype(float),
        y=np.ravel(m["y"]).astype(float),
    )


def load_grid(case):
    """Parse CTVOXEL_INFO.txt -> dict(shape=(nx,ny,nz), voxel_size, start, isocenter)."""
    txt = (_case_dir(case) / "CTVOXEL_INFO.txt").read_text()
    def nums(line_key):
        for line in txt.splitlines():
            if line_key in line:
                return [float(t) for t in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line.split("=")[-1])]
        raise KeyError(line_key)
    nx = int(nums("[cerr x]")[0]); ny = int(nums("[cerr gui y]")[0]); nz = int(nums("[cerr gui z]")[0])
    return dict(
        shape=(nx, ny, nz),
        voxel_size=(nums("[horizontal] voxel size")[0], nums("[vertical] voxel size")[0],
                    nums("[sup/inf] voxel size")[0]),
        start=(nums("x position of dose voxel 1")[0], nums("y position of dose voxel 1")[0],
               nums("z position of dose voxel 1")[0]),
        isocenter=tuple(nums("isocenter")),
    )


def available_gantry_angles(case, couch=0):
    out = []
    for p in _case_dir(case).glob(f"Gantry*_Couch{couch}_D.mat"):
        out.append(int(re.match(r"Gantry(-?\d+)_", p.name).group(1)))
    return sorted(out)
