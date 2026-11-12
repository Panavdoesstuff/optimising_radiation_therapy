"""Phase 0: inspect the CORT data and confirm the expected facts."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, io_cort

warnings.simplefilter("ignore", DeprecationWarning)

for case, d in config.CASE_DIRS.items():
    print("=" * 70, f"\n{case}: {d}")
    files = sorted(p.name for p in d.iterdir())
    kinds = {}
    for f in files:
        k = "D" if f.endswith("_D.mat") else "BEAMINFO" if f.endswith("_BEAMINFO.mat") else \
            "VOILIST" if f.endswith("_VOILIST.mat") else "other"
        kinds.setdefault(k, []).append(f)
    for k, v in kinds.items():
        print(f"  {k}: {len(v)} files" + (f" -> {v}" if k in ("VOILIST", "other") else ""))
    print("-- CTVOXEL_INFO.txt\n" + (d / "CTVOXEL_INFO.txt").read_text())
    grid = io_cort.load_grid(case)
    print("parsed grid:", grid)

    angles = io_cort.available_gantry_angles(case)
    print(f"gantry angles ({len(angles)}): {angles[:5]}...{angles[-3:]}, step set={sorted(set(b-a for a,b in zip(angles, angles[1:])))}")
    total = 0
    for g in angles:
        total += io_cort.load_beaminfo(case, g)["num_beamlets"]
    print("total beamlets (BEAMINFO sum):", total)

    g0 = angles[0]
    for kind, fn in (("D", f"Gantry{g0}_Couch0_D.mat"), ("BEAMINFO", f"Gantry{g0}_Couch0_BEAMINFO.mat")):
        m = io_cort.load_mat(d / fn)
        print(f"-- {fn}:", {k: (getattr(v, 'shape', None), type(v).__name__) for k, v in m.items()})
    D = io_cort.load_D(case, g0)
    print(f"D: shape={D.shape}, nnz={D.nnz}, max={D.max():.4g}, dtype={D.dtype}")
    nvox = grid["shape"][0] * grid["shape"][1] * grid["shape"][2]
    print("grid voxels:", nvox, "| D rows == grid voxels:", D.shape[0] == nvox)
    for name in config.STRUCTURES[case]:
        v = io_cort.load_voilist(case, name)
        print(f"  {name}: n={len(v)}, min={v.min()}, max={v.max()} (<{nvox}: {v.max() < nvox})")

    exp = config.EXPECTED[case]
    print("EXPECTED vs ACTUAL:",
          f"beams {exp['n_beams']} vs {len(angles)};",
          f"beamlets {exp['n_beamlets']} vs {total};",
          f"grid {exp['grid']} vs {grid['shape']}")
