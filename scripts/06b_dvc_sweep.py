"""Rectal V50 trade-off sweep by reweighted hinge + dual-guided fix-and-resolve (all feasible points).
usage: 06b_dvc_sweep.py <tag> <angles comma sep|equi> <alphas comma sep> <budget_minutes> [lam0]
Writes results/improve/dvc/sweep_<tag>.json (appended per alpha) and plan_<tag>_a<alpha>.npz."""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, dvc_improve as di, verify

warnings.simplefilter("ignore")
tag, beams, alphas, budget = sys.argv[1], sys.argv[2], [float(a) for a in sys.argv[3].split(",")], float(sys.argv[4]) * 60
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
MAXIT = int(sys.argv[6]) if len(sys.argv) > 6 else 8
PROT = 0
START = sys.argv[7] if len(sys.argv) > 7 else None
OUT = config.RESULTS_DIR / "improve" / "dvc"; OUT.mkdir(parents=True, exist_ok=True)
angles = config.PROSTATE_5 if beams == "equi" else [int(a) for a in beams.split(",")]
P = problem.build("prostate", angles)
sets = P.sets("disjoint"); spec = easy_lp.Spec(); R = sets["Rectum"]; nR = len(R)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# start rows: the equispaced easy plan's generated body rows are a valid start for any beams (more are generated)
rows0 = np.load(config.RESULTS_DIR / "easy_solution.npz")["body_rows"]
M = di.Persistent(P, spec, sets, rows0)
base = M.solve(lam=0.0)
log("easy LP: cost", base["cost"], "V50 n", int((base["d"][R] > 50 * (1 + 1e-6)).sum()))
easy_cost = base["cost"]
res_file = OUT / f"sweep_{tag}.json"
results = []

def count(d): return int((d[R] > 50 * (1 + 1e-6)).sum())

def polish(F_mask):
    """cap everything outside F_mask at 50, free F: true plan cost (LP)."""
    return M.solve(hard_mask=~F_mask, lam=0.0)

def save(alpha, r, method):
    x = r["x"]
    v = verify.verify(P, sets, spec, x, dvcs=[("Rectum", 50.0, alpha)])
    d = P.dose(x); n_over = int((d[R] > 50 * (1 + 1e-6)).sum())
    rec = dict(alpha=alpha, objective=v["objective_recomputed"], price_pct=100 * (v["objective_recomputed"] / easy_cost - 1),
               n_over=n_over, V50=n_over / nR, allowed=int(np.floor(alpha * nR)), verified_ok=bool(v["all_ok"]),
               method=method, time_s=time.time() - t0)
    np.savez(OUT / f"plan_{tag}_a{int(round(alpha*1000)):03d}.npz", x=x)
    return rec

# best known plan so far (warm start for the next, tighter alpha)
w = np.ones(nR); cur = base; best_for = {}
if START:
    xs = np.load(START)["x"]; cur = dict(x=xs, d=P.dose(xs), cost=float(M.cx @ xs)); log("start plan hot", count(cur["d"]))
for alpha in alphas:
    K = int(np.floor(alpha * nR)); tA = time.time()
    if time.time() - t0 > budget: log("budget exhausted"); break
    # --- stage A: elastic swap search. F = K free voxels (start: K hottest of the previous plan).
    # Non-F voxels get a hinge penalty; violators (e>0) swap into F, replacing the coolest F voxels.
    ok = False
    dprev = cur["d"][R]
    F = np.zeros(nR, bool); F[np.argsort(-dprev)[:K]] = True
    prot = np.zeros(nR, int)       # iterations of protection for voxels just swapped in (tabu)
    for it in range(MAXIT):
        r = M.solve(free_mask=F, lam=lam)
        if not r["success"]: log("elastic failed", r.get("message")); break
        e = np.where(F, 0.0, r["e"]); viol = np.flatnonzero(e > 1e-6)
        dd = r["d"][R]
        log(f"alpha {alpha:.3f} K={K} elastic {it}: cost {r['cost']:.4f} viol {len(viol)} sum_e {e.sum():.2f} hot {count(r['d'])}")
        if len(viol) == 0:
            ok = True; hot = r; break
        # swap in the violators (largest excess first), out the F voxels with the lowest dose
        nsw = len(viol)
        prot = np.maximum(prot - 1, 0)
        inF = np.flatnonzero(F & (prot == 0))
        out = inF[np.argsort(dd[inF])[:nsw]]
        F = F.copy(); F[out] = False; F[viol] = True; prot[viol] = PROT
    if not ok:
        rec = dict(alpha=alpha, result="no plan found by elastic swap search (not a proof)", time_s=time.time() - t0)
        results.append(rec); res_file.write_text(json.dumps(results, indent=2)); log("FAIL", rec); break
    # --- stage B: polish: free the hot voxels (+ fill K by closeness to 50), cap the rest, LP; then dual swaps
    def fillF(r):
        d = r["d"][R]; F = d > 50 * (1 + 1e-6)
        extra = K - int(F.sum())
        if extra > 0:
            cand = np.flatnonzero(~F); F[cand[np.argsort(-d[cand])[:extra]]] = True
        return F
    F = fillF(hot); bestr, bestmeth = None, ""
    for sw in range(3):
        pr = polish(F)
        if not pr["success"]:
            log("polish infeasible"); break
        c = count(pr["d"]); log(f"   polish/swap {sw}: cost {pr['cost']:.4f} hot {c} free {int(F.sum())}")
        if bestr is None or pr["cost"] < bestr["cost"] - 1e-9:
            bestr, bestmeth = pr, "reweighted hinge + polish" + (f" + {sw} dual swap(s)" if sw else "")
        else:
            break
        # dual-guided swap: reduced cost of e_i for hard voxels (how much freeing it would save)
        rc = np.array(M.h.getSolution().col_dual)[M.nx:]
        Fn = pr["d"][R] > 50 * (1 + 1e-6)
        hard_idx = np.flatnonzero(~Fn)
        extra = K - int(Fn.sum())
        top = hard_idx[np.argsort(-np.abs(rc[hard_idx]))[:max(extra, 0)]]
        Fn = Fn.copy(); Fn[top] = True
        if (Fn == F).all(): break
        F = Fn
    if bestr is None:
        rec = dict(alpha=alpha, result="polish infeasible", time_s=time.time() - t0)
    else:
        rec = save(alpha, bestr, bestmeth); rec["stageA_lam"] = lam
        cur = bestr
    results.append(rec); res_file.write_text(json.dumps(results, indent=2)); log("DONE", rec)
log("finished")
