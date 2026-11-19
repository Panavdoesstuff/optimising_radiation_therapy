"""Re-verify all saved dvc2 plans, make figure and REPORT tables. Run after sweeps."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config, problem, easy_lp, verify
OUT = config.RESULTS_DIR / "improve" / "dvc2"
beams = {"A": [108,144,180,216,252], "B": [90,126,172,190,240]}
ref = {"A": 47.2054, "B": 41.6322}
tabs = {}; md = []
for tag, ang in beams.items():
    P = problem.build("prostate", ang); sets = P.sets("disjoint"); spec = easy_lp.Spec(); R = sets["Rectum"]
    rows = []
    for suf in ["", "2", "3"]:
        f = OUT / f"sweep_{tag}{suf}.json"
        if f.exists(): rows += json.loads(f.read_text())
    rows.sort(key=lambda r: (-r["alpha"], "file" not in r))
    base = np.load({"A": config.RESULTS_DIR/"enum"/"best_K5.npz", "B": config.RESULTS_DIR/"improve"/"fine"/"best_K5.npz"}[tag])["x"]
    ob = verify.verify(P, sets, spec, base)["objective_recomputed"]; d0 = P.dose(base)
    n0 = int((d0[R] > 50*(1+1e-6)).sum())
    out = [dict(alpha=None, objective=ob, price=0.0, V50=n0/len(R), n=n0, ok=True, t=None)]
    md.append(f"\n### Plan {tag}: beams {ang}\nUnconstrained: objective {ob:.4f}, V50 {100*n0/len(R):.2f}% ({n0} voxels)\n\n| alpha | objective | price vs unconstrained | verified V50 (voxels) | verified | time |\n|---|---|---|---|---|---|")
    md.append(f"| none | {ob:.4f} | 0 | {100*n0/len(R):.2f}% ({n0}) | yes | - |")
    for r in rows:
        if "file" not in r:
            md.append(f"| {100*r['alpha']:.2f}% | no plan (heuristic failure) | - | - | - | {r['time_s']/60:.0f} min |"); continue
        x = np.load(OUT / r["file"])["x"]
        v = verify.verify(P, sets, spec, x, dvcs=[("Rectum", 50.0, r["alpha"])])
        n = int((P.dose(x)[R] > 50*(1+1e-6)).sum()); ok = v["all_ok"] and n <= int(np.floor(r["alpha"]*len(R)))
        o = v["objective_recomputed"]
        md.append(f"| {100*r['alpha']:.2f}% | {o:.4f} | {100*(o/ob-1):+.2f}% | {100*n/len(R):.2f}% ({n}) | {'PASS' if ok else 'FAIL'} | {r['time_s']/60:.0f} min |")
        if ok: out.append(dict(alpha=r["alpha"], objective=o, price=100*(o/ob-1), V50=n/len(R), n=n, ok=True, t=r["time_s"]))
    tabs[tag] = out
(OUT / "verified_table.json").write_text(json.dumps(tabs, indent=1))
(OUT / "tables.md").write_text("\n".join(md))
eq = [r for r in json.loads((config.RESULTS_DIR/"improve"/"dvc"/"verified_table.json").read_text()) if r["ok"] and "stB" not in r["file"]]
fig, ax = plt.subplots(figsize=(7, 4.5))
for tag, c, lab in [("B", "#0072B2", "Plan B: fine-angle 5 beams (90,126,172,190,240)"), ("A", "#D55E00", "Plan A: certified 36-degree 5 beams (108,...,252)")]:
    pts = [r for r in tabs[tag] if r["alpha"] is not None]
    ax.plot([100*r["alpha"] for r in pts], [r["objective"] for r in pts], "o-", color=c, label=lab)
    ax.axhline(tabs[tag][0]["objective"], color=c, ls=":", lw=1)
ee = sorted(eq, key=lambda r: -r["alpha"])
ax.plot([100*r["alpha"] for r in ee], [r["objective"] for r in ee], "s-", color="#555555", label="Equispaced 5 beams (0,72,...,288)")
ax.set_xlabel("Rectal limit alpha: allowed % of rectum above 50 Gy"); ax.set_ylabel("Plan objective (lower is better)")
ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("Price of the rectal V50 limit (heuristic upper bounds)")
fig.tight_layout(); fig.savefig(OUT / "fig_dvc_tradeoff.png", dpi=150)
print("\n".join(md))
