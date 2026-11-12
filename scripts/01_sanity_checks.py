"""Phase 2: reproduce published numbers (Craft et al. 2014) -> results/sanity_checks.csv."""
import csv, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp
from src import config, io_cort, problem

warnings.simplefilter("ignore", DeprecationWarning)
rows = []

# ---------------- Check A: ones solution ----------------
PUBLISHED_A = {
    ("tg119", "Core"): (0.026798, 0.050724, 0.053313),
    ("tg119", "OuterTarget"): (0.049379, 0.051067, 0.052702),
    ("prostate", "PTV_56"): (1.3015, 1.3631, 1.4089),
    ("prostate", "Bladder"): (0.66549, 1.2753, 1.3863),
}
for case in ("tg119", "prostate"):
    t0 = time.time()
    angles = io_cort.available_gantry_angles(case)
    sets = {n: io_cort.load_voilist(case, n) for n in config.STRUCTURES[case]}
    nvox = int(np.prod(io_cort.load_grid(case)["shape"]))
    d = np.zeros(nvox)
    for g in angles:                      # beam by beam: never stack 180 beams
        Db = io_cort.load_D(case, g)
        d += Db @ np.ones(Db.shape[1])
    print(f"[A] {case}: {len(angles)} beams in {time.time()-t0:.0f}s")
    for (c, name), pub in PUBLISHED_A.items():
        if c != case:
            continue
        got = (d[sets[name]].min(), d[sets[name]].mean(), d[sets[name]].max())
        for stat, p, v in zip(("min", "mean", "max"), pub, got):
            relerr = abs(v - p) / p
            sig = f"{p:.4g}"
            ok = f"{v:.4g}" == sig or relerr < 5e-4   # 4 significant figures
            rows.append(dict(check="A_ones", case=case, quantity=f"{name}_{stat}", published=p,
                             computed=v, rel_err=relerr, pass_=ok))
            print(f"   {name:12s} {stat:4s} published {p:<10} computed {v:.6g}  relerr {relerr:.1e} {'OK' if ok else 'FAIL'}")

# ---------------- Check B: paper LPs ----------------
def solve_lp(P, S, c, lo_rows, hi_rows, ub):
    """min c@x  s.t. D[S_i]x >= 1 (lo_rows), D[S_j]x <= 1.2 (hi_rows); bounds [0, ub]."""
    A, b = [], []
    for idx, lo in lo_rows:
        A.append(-P.D[idx]); b.append(-lo * np.ones(len(idx)))
    for idx, hi in hi_rows:
        A.append(P.D[idx]); b.append(hi * np.ones(len(idx)))
    A = sp.vstack(A, format="csr"); b = np.concatenate(b)
    t0 = time.time()
    r = linprog(c, A_ub=A, b_ub=b, bounds=(0, ub), method="highs")
    return r, time.time() - t0

# TG119
P = problem.build("tg119", config.TG119_ANGLES); S = P.sets("raw")
cC, cB = P.mean_dose_vector(S["Core"]), P.mean_dose_vector(S["BODY"])
r, t = solve_lp(P, S, cC + cB, [(S["OuterTarget"], 1.0)],
                [(S["OuterTarget"], 1.2), (S["Core"], 1.2)], None)
print(f"[B] TG119: {r.message} ({t:.1f}s)")
d = P.dose(r.x)
for q, pub, v in (("obj", 0.2489 + 0.1021, r.fun), ("mean_Core", 0.2489, d[S["Core"]].mean()),
                  ("mean_BODY", 0.1021, d[S["BODY"]].mean())):
    ok = abs(v - pub) / pub < 2e-3
    rows.append(dict(check="B_lp", case="tg119", quantity=q, published=pub, computed=v,
                     rel_err=abs(v - pub) / pub, pass_=ok))
    print(f"   {q:10s} published {pub:.4f} computed {v:.4f} {'OK' if ok else 'FAIL'}")

# Prostate 5 beams
P = problem.build("prostate", config.PROSTATE_5); S = P.sets("raw")
cR, cB, cN = (P.mean_dose_vector(S[n]) for n in ("Rectum", "Bladder", "BODY"))
r, t = solve_lp(P, S, cR + 0.6 * cB + 0.6 * cN, [(S["PTV_68"], 1.0)], [], 50.0)
print(f"[B] Prostate: {r.message} ({t:.1f}s)")
d = P.dose(r.x)
mR, mB, mN = d[S["Rectum"]].mean(), d[S["Bladder"]].mean(), d[S["BODY"]].mean()
pubR, pubB, pubN = 0.2842, 0.4035, 0.0905
for q, pub, v in (("obj", pubR + 0.6 * pubB + 0.6 * pubN, r.fun), ("mean_Rectum", pubR, mR),
                  ("mean_Bladder", pubB, mB), ("mean_BODY", pubN, mN)):
    ok = abs(v - pub) / pub < 2e-3
    rows.append(dict(check="B_lp", case="prostate", quantity=q, published=pub, computed=v,
                     rel_err=abs(v - pub) / pub, pass_=ok))
    print(f"   {q:12s} published {pub:.4f} computed {v:.4f} {'OK' if ok else 'FAIL'}")

config.RESULTS_DIR.mkdir(exist_ok=True)
with open(config.RESULTS_DIR / "sanity_checks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "case", "quantity", "published", "computed", "rel_err", "pass_"])
    w.writeheader(); w.writerows(rows)
print("ALL PASS" if all(r["pass_"] for r in rows) else "SOME CHECKS FAILED")
