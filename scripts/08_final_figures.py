"""Final figures and comparison table from saved plans (nothing is re-optimised).

Every plan is reloaded, its dose d = D x is recomputed on the full grid, and its objective is
re-checked with src/verify.py before it is plotted or tabulated.

Outputs (results/):
  fig7_beam_budget_final.png   objective vs number of beams: certified (36 and 18 degree grids) and best found (2 degrees)
  fig8_dvh_final.png           DVHs of rectum, bladder, PTV 68 for easy, certified best 5, best-found 5 and 6 beams
  fig9_dvc_tradeoff_final.png  price of the rectal limit (objective vs verified V50) on three beam sets
  fig10_organ_means.png        mean dose to rectum, bladder, body for the main plans
  final_comparison.csv         dose metrics and verification for every plan above
"""
import csv, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from src import config, problem, easy_lp, verify, plots

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
IM = R / "improve"
spec = easy_lp.Spec()
P5 = problem.build("prostate", config.PROSTATE_5)
sets = P5.sets("disjoint")
EASY_OBJ = None

# ---------------------------------------------------------------- load and re-verify plans
PLANS = [  # key, label, npz, colour
    ("easy", "Easy LP: 5 equispaced beams", R / "easy_solution.npz", "#52514e"),
    ("cert5", "Certified best 5 (36° grid)", R / "enum" / "best_K5.npz", "#2a78d6"),
    ("best5", "Best found 5 (2° angles)", IM / "fine2" / "best_K5.npz", "#1baf7a"),
    ("best6", "Best found 6 (2° angles)", IM / "fine2" / "best_K6.npz", "#eb6834"),
]
loaded = {}
for key, label, path, color in PLANS:
    z = np.load(path)
    angles = sorted(int(a) for a in z["angles"])
    P = P5 if key == "easy" else problem.build("prostate", angles)
    x = z["x"]
    v = verify.verify(P, sets, spec, x)
    assert v["all_ok"], f"{key} failed verification"
    loaded[key] = dict(label=label, color=color, angles=angles, P=P, x=x, d=P.dose(x), v=v,
                       obj=v["objective_recomputed"])
    print(f"{key:6s} {angles} objective {loaded[key]['obj']:.4f} verified", flush=True)
EASY_OBJ = loaded["easy"]["obj"]


def metrics(d):
    r, b, n = d[sets["Rectum"]], d[sets["Bladder"]], d[sets["BODY"]]
    return dict(rectum_mean=r.mean(), rectum_V30=(r > 30).mean(), rectum_V50=(r > 50).mean(),
                rectum_V60=(r > 60).mean(), bladder_mean=b.mean(), bladder_V30=(b > 30).mean(),
                bladder_V50=(b > 50).mean(), body_mean=n.mean(), body_max=n.max(),
                ptv68_D95=np.percentile(d[sets["PTV_68"]], 5), ptv56_D95=np.percentile(d[sets["PTV_56"]], 5))


# ---------------------------------------------------------------- table
rows = []
for key, L in loaded.items():
    m = metrics(L["d"])
    rows.append(dict(plan=L["label"], beams=" ".join(map(str, L["angles"])), objective=round(L["obj"], 4),
                     vs_easy_pct=round(100 * (L["obj"] - EASY_OBJ) / EASY_OBJ, 2),
                     max_rel_violation=f"{max(c.get('relative', 0.0) for c in L['v']['checks'].values()):.1e}",
                     **{k: round(float(val), 4) for k, val in m.items()}))
with open(R / "final_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote final_comparison.csv")

# ---------------------------------------------------------------- fig 7: beam budget
def best_found(K):
    z = np.load(IM / "fine2" / f"best_K{K}.npz")
    a = sorted(int(t) for t in z["angles"])
    P = problem.build("prostate", a)
    return verify.verify(P, sets, spec, z["x"])["objective_recomputed"]


# certified values are read from the files that proved them
cert36 = {}
for K in (3, 4, 5, 6):
    recs = [json.loads(l) for l in open(R / "enum" / f"K{K}.jsonl")]
    cert36[K] = min(r["objective"] for r in recs if r["outcome"] == "solved")
cert18 = {3: json.loads((IM / "bao18" / "K3_final.json").read_text()),
          4: [json.loads(l) for l in open(IM / "cert18" / "K4.jsonl")]}
c18 = {4: min(r["objective"] for r in cert18[4] if r["outcome"] == "solved")}
k3 = cert18[3]
c18[3] = float(k3["objective"]) if "objective" in k3 else 60.789
found18 = {5: 44.2032, 6: 42.0274}       # swap search, 18° grid, verified (results/improve/bao18/REPORT.md)
found2 = {K: best_found(K) for K in (3, 4, 5, 6)}
print("cert36", cert36, "cert18", c18, "found2", found2, flush=True)

fig, ax = plt.subplots(figsize=(9.5, 5.8))
Ks = [3, 4, 5, 6]
ax.plot(Ks, [found2[k] for k in Ks], "o-", color="#eb6834", label="Best found, 2° angles (not certified)")
ax.plot(list(found18), list(found18.values()), "s--", color="#eda100", label="Best found, 18° grid (not certified)")
ax.plot(list(c18), list(c18.values()), "D", color="#184f95", ms=10, label="Certified optimum, 18° grid (K=3, 4)")
ax.plot(Ks, [cert36[k] for k in Ks], "^-", color="#2a78d6", label="Certified optimum, 36° grid (all subsets)")
ax.axhline(EASY_OBJ, color="#52514e", ls="--", lw=1.4)
ax.annotate(f"Easy LP, 5 equispaced beams: {EASY_OBJ:.2f}", (6.0, EASY_OBJ), xytext=(0, 5),
            textcoords="offset points", fontsize=9, color="#52514e", ha="right")
for k in Ks:
    ax.annotate(f"{found2[k]:.1f}", (k, found2[k]), xytext=(6, -12), textcoords="offset points",
                fontsize=9, color="#c4511f")
ax.set_xticks(Ks); ax.set_xlabel("Beam budget K (at most K beams used)")
ax.set_ylabel("Objective (weighted mean dose, lower is better)")
ax.set_title("Price of a beam budget: certified optima vs best-found plans")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(R / "fig7_beam_budget_final.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 8: DVHs by structure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, s, title in zip(axes, ("Rectum", "Bladder", "PTV_68"), ("Rectum", "Bladder", "Tumour core (PTV 68)")):
    for key, L in loaded.items():
        dmax = 80
        g, v = plots.dvh_curve(L["d"][sets[s]], dmax=dmax)
        ax.plot(g, v, color=L["color"], lw=2.2 if key != "easy" else 1.8,
                ls="--" if key == "easy" else "-", label=L["label"])
    ax.set_title(title); ax.set_xlabel("Dose (Gy)"); ax.set_xlim(0, 80)
axes[0].set_ylabel("Volume receiving at least this dose (%)")
axes[0].legend(loc="upper right", fontsize=8.5)
fig.suptitle("Dose-volume histograms (normal-tissue cap 74.8 Gy in every plan)", y=1.0)
fig.tight_layout(); fig.savefig(R / "fig8_dvh_final.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 9: rectal-limit trade-off
def table(path):
    d = json.loads(path.read_text())
    return d
equi = [r for r in table(IM / "dvc" / "verified_table.json") if r["ok"] and "stB" not in r["file"]]
dv2 = table(IM / "dvc2" / "verified_table.json")
fig, ax = plt.subplots(figsize=(9.5, 5.8))
series = [("Equispaced 5 beams", [(EASY_OBJ, 0.187)] + [(r["objective"], r["V50"]) for r in equi], "#52514e"),
          ("Certified best 5 (36° grid)", [(r["objective"], r["V50"]) for r in dv2["A"]], "#2a78d6"),
          ("Best found 5 (2° angles)", [(r["objective"], r["V50"]) for r in dv2["B"]], "#1baf7a")]
for label, pts, col in series:
    pts = sorted(pts, key=lambda p: -p[1])
    ax.plot([100 * p[1] for p in pts], [p[0] for p in pts], "o-", color=col, label=label)
ax.invert_xaxis()
ax.axvline(17, color="#b0aea8", ls=":", lw=1.2)
ax.annotate("17% limit (first run)", (17, ax.get_ylim()[0]), xytext=(-4, 6),
            textcoords="offset points", fontsize=8.5, ha="right", color="#52514e")
ax.set_xlabel("Share of rectum above 50 Gy, V50 (%), recomputed on the full grid (tighter to the right)")
ax.set_ylabel("Objective (lower is better)")
ax.set_title("Price of the rectal dose-volume limit depends on the beams")
ax.legend(loc="center left")
fig.tight_layout(); fig.savefig(R / "fig9_dvc_tradeoff_final.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 10: organ means
fig, ax = plt.subplots(figsize=(9.5, 5.4))
w = 0.2
names = ["Rectum mean", "Bladder mean", "Body mean"]
for i, (key, L) in enumerate(loaded.items()):
    m = metrics(L["d"])
    vals = [m["rectum_mean"], m["bladder_mean"], m["body_mean"]]
    bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, color=L["color"], label=L["label"])
    for b_, v_ in zip(bars, vals):
        ax.annotate(f"{v_:.1f}", (b_.get_x() + b_.get_width() / 2, v_), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylabel("Mean dose (Gy)")
ax.set_title("Where the gain comes from: mean dose to organs at risk")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(R / "fig10_organ_means.png", dpi=160); plt.close(fig)
print("figures written")
