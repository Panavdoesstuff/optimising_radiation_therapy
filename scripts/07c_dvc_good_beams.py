"""Rectal V50 price-of-realism sweep on a good beam set (elastic swap + polish, warm-started).
usage: 07c_dvc_good_beams.py <tag> <angles> <start_plan.npz> <first_alpha_pct> <budget_min> [lam] [step_pct]
Adaptive step: on failure retry with half the step down to 0.25 points. Writes results/improve/dvc2/."""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, dvc_improve as di, verify
warnings.simplefilter("ignore")
tag, angles = sys.argv[1], [int(a) for a in sys.argv[2].split(",")]
start, a0, budget = sys.argv[3], float(sys.argv[4]) / 100, float(sys.argv[5]) * 60
lam = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
step0 = float(sys.argv[7]) / 100 if len(sys.argv) > 7 else 0.01
MAXIT = 10
OUT = config.RESULTS_DIR / "improve" / "dvc2"; OUT.mkdir(parents=True, exist_ok=True)
P = problem.build("prostate", angles)
sets = P.sets("disjoint"); spec = easy_lp.Spec(); R = sets["Rectum"]; nR = len(R)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
rows0 = np.load(config.RESULTS_DIR / "easy_solution.npz")["body_rows"]
M = di.Persistent(P, spec, sets, rows0)
base = M.solve(lam=0.0)
cnt = lambda d: int((d[R] > 50 * (1 + 1e-6)).sum())
log("easy LP cost", base["cost"], "V50 n", cnt(base["d"]), cnt(base["d"]) / nR)
easy = base["cost"]
xs = np.load(start)["x"]; ds = P.dose(xs)
log("saved plan cost", float(M.cx @ xs), "V50 n", cnt(ds))
cur = dict(x=xs, d=ds, cost=float(M.cx @ xs))
if cur["cost"] > easy + 1e-6: cur = base   # LP is the optimum; prefer it
results = []
def save(alpha, x, method, tt):
    v = verify.verify(P, sets, spec, x, dvcs=[("Rectum", 50.0, alpha)])
    d = P.dose(x); n = cnt(d)
    rec = dict(alpha=alpha, objective=v["objective_recomputed"], price_pct=100 * (v["objective_recomputed"] / easy - 1),
               n_over=n, V50=n / nR, allowed=int(np.floor(alpha * nR)), verified_ok=bool(v["all_ok"]),
               failed_checks=[k for k, c in v["checks"].items() if not c["ok"]], method=method, time_s=tt)
    if rec["verified_ok"] and n <= rec["allowed"]:
        np.savez(OUT / f"plan_{tag}_a{int(round(alpha*1000)):03d}.npz", x=x)
        rec["file"] = f"plan_{tag}_a{int(round(alpha*1000)):03d}.npz"
    else:
        rec["verified_ok"] = False
    return rec

def attempt(alpha, cur):
    K = int(np.floor(alpha * nR))
    F = np.zeros(nR, bool); F[np.argsort(-cur["d"][R])[:K]] = True
    hot = None
    for it in range(MAXIT):
        if time.time() - t0 > budget: return None
        r = M.solve(free_mask=F, lam=lam)
        if not r["success"]: log("elastic failed", r.get("message")); return None
        e = np.where(F, 0.0, r["e"]); viol = np.flatnonzero(e > 1e-6); dd = r["d"][R]
        log(f" a={alpha:.4f} K={K} it{it}: cost {r['cost']:.4f} viol {len(viol)} hot {cnt(r['d'])}")
        if len(viol) == 0: hot = r; break
        inF = np.flatnonzero(F); out = inF[np.argsort(dd[inF])[:len(viol)]]
        F = F.copy(); F[out] = False; F[viol] = True
    if hot is None: return None
    def fillF(r):
        d = r["d"][R]; F = d > 50 * (1 + 1e-6); extra = K - int(F.sum())
        if extra > 0:
            cand = np.flatnonzero(~F); F[cand[np.argsort(-d[cand])[:extra]]] = True
        return F
    F = fillF(hot); best = None
    for sw in range(3):
        if time.time() - t0 > budget and best is not None: break
        pr = M.solve(hard_mask=~F, lam=0.0)
        if not pr["success"]: break
        log(f"   polish {sw}: cost {pr['cost']:.4f} hot {cnt(pr['d'])}")
        if best is None or pr["cost"] < best["cost"] - 1e-9: best = pr
        else: break
        rc = np.array(M.h.getSolution().col_dual)[M.nx:]
        Fn = pr["d"][R] > 50 * (1 + 1e-6); hard_idx = np.flatnonzero(~Fn)
        extra = K - int(Fn.sum())
        top = hard_idx[np.argsort(-np.abs(rc[hard_idx]))[:max(extra, 0)]]
        Fn = Fn.copy(); Fn[top] = True
        if (Fn == F).all(): break
        F = Fn
    return best

alpha = a0; step = step0
while time.time() - t0 < budget and alpha > 0.005:
    tA = time.time()
    r = attempt(alpha, cur)
    if r is None:
        rec = dict(alpha=alpha, result="no plan found (heuristic failure, not a proof)", step=step, time_s=time.time() - tA)
        results.append(rec); (OUT / f"sweep_{tag}.json").write_text(json.dumps(results, indent=2)); log("FAIL", rec)
        if time.time() - t0 > budget: break
        alpha = alpha + step; step /= 2     # back off, retry smaller step
        if step < 0.0025: break
        alpha -= step
        continue
    rec = save(alpha, r["x"], "elastic swap + polish", time.time() - tA)
    results.append(rec); (OUT / f"sweep_{tag}.json").write_text(json.dumps(results, indent=2)); log("DONE", rec)
    if rec["verified_ok"]: cur = r; alpha -= step
    else: log("verification failed; stop"); break
log("finished")
