"""Phase 6: figures and tables from saved solutions (no optimisation is re-run)."""
import csv, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, io_cort, hard_bao, plots, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
res = json.loads((R / "results.json").read_text())
sweep = json.loads((R / "bao_sweep.json").read_text())

# ---------------- plans
P5 = problem.build("prostate", config.PROSTATE_5)
sets = P5.sets("disjoint")
x_easy = np.load(R / "easy_solution.npz")["x"]
x_dvc = np.load(R / "hard_dvc_solution.npz")["x"]
z5 = np.load(R / "bao_K5_solution.npz")
bao_angles = [int(a) for a in z5["angles"]]
P_bao = problem.build("prostate", bao_angles, use_cache=False)
x_bao = z5["x"]
d_easy, d_dvc, d_bao = P5.dose(x_easy), P5.dose(x_dvc), P_bao.dose(x_bao)
lab_easy = "Easy LP: 5 equispaced beams"
lab_bao = f"Hard: best 5 of 20 beams ({', '.join(map(str, bao_angles))}°)"
lab_dvc = "Hard (secondary): + rectum ≤17% > 50 Gy"
STRUCTS = ["PTV_68", "PTV_56", "Rectum", "Bladder", "BODY"]

# 1. DVH
plots.fig_dvh(R / "fig1_dvh.png", [("easy", lab_easy, d_easy), ("bao", lab_bao, d_bao), ("dvc", lab_dvc, d_dvc)],
              sets, STRUCTS, constraints=config.DVC_FINAL,
              title="Dose-volume histograms: easy LP vs hard plans")

# 2. dose maps
grid = P5.grid
vol = lambda d: problem.reshape_volume(d, grid)
masks = {}
for n in ("PTV_68", "PTV_56", "Rectum", "Bladder"):
    m = np.zeros(P5.n_voxels, bool); m[sets[n]] = True; masks[n] = problem.reshape_volume(m, grid)
z = int(np.round(np.nonzero(masks["PTV_68"])[2].mean()))
body = np.zeros(P5.n_voxels, bool); body[sets["BODY"]] = True
bz = np.nonzero(problem.reshape_volume(body, grid)[:, :, z])
crop = ((max(bz[0].min() - 3, 0), bz[0].max() + 4), (max(bz[1].min() - 3, 0), bz[1].max() + 4))
V_e, V_b, V_d = vol(d_easy), vol(d_bao), vol(d_dvc)
vmax = max(V[:, :, z].max() for V in (V_e, V_b, V_d))
plots.fig_dose_maps(R / "fig2_dose_maps.png",
                    [("Easy LP", V_e), ("Hard: best 5 beams", V_b), ("Hard: + rectum DVC", V_d)],
                    masks, z, [("Best 5 beams − easy", V_b - V_e), ("Rectum DVC − easy", V_d - V_e)], crop, vmax)

# 3. fluence maps
def beams_of(P, x):
    out = []
    for b, g in enumerate(P.angles):
        out.append((g, io_cort.load_beaminfo("prostate", g), x[P.beam_of_column == b]))
    return out
vmax_f = max(x_easy.max(), x_bao.max())
plots.fig_fluence(R / "fig3_fluence.png", [("Easy LP", beams_of(P5, x_easy)),
                                           ("Hard: best 5", beams_of(P_bao, x_bao))], vmax_f)

# 4. price of realism
def terms(v):
    return v["objective_terms_weighted"]
red = hard_bao.build_reduced(config.BAO_CANDIDATES, sets)
x_all = np.load(R / "bao_all_beams_x.npy")
sp_ = res["easy"]["params"]
t_all = dict(Rectum=sp_["w_R"] * red.cost["R"] @ x_all, Bladder=sp_["w_B"] * red.cost["B"] @ x_all,
             BODY=sp_["w_N"] * red.cost["N"] @ x_all)
v_easy, v_dvc, v_bao = res["easy"]["verification"], res["hard_dvc"]["verification"], sweep["5"]["verification"]
plots.fig_price(R / "fig4_price_of_realism.png", [
    ("Easy LP\n5 equispaced beams", terms(v_easy), ""),
    ("Hard: rectum DVC\n(fix-and-resolve)", terms(v_dvc), "approximation, feasible"),
    ("Hard: best 5 of 20 beams\n(MILP + heuristics)", terms(v_bao), f"gap to proven bound {sweep['5']['gap']:.0%}"),
    ("All 20 beams (LP)\nno beam limit", t_all, "lower bound for any K"),
], ref_label="Easy LP\n5 equispaced beams")

# 5. sweep over K
Ks = sorted(int(k) for k in sweep if k.isdigit())
plots.fig_sweep(R / "fig5_sweep_K.png", Ks, [sweep[str(K)]["objective"] for K in Ks],
                [sweep[str(K)]["best_bound"] for K in Ks], sweep["all_beams_lp"]["objective"],
                (5, v_easy["objective_recomputed"]))

# 6. model summary table
rows = []
e = res["easy"]
rows.append(dict(model="Easy LP (5 equispaced beams)", kind="LP", n_vars=e["model"]["n_vars"],
                 n_constraints=e["model"]["n_constraints"], n_binaries=0, nnz=e["model"]["nnz"],
                 solve_time_s=round(e["verification"]["solver"]["solve_time"], 1), status="optimal",
                 gap=f"duality gap {e['verification']['solver']['duality_gap']:.1e}",
                 objective=round(e["verification"]["objective_recomputed"], 4), plan_source="LP optimum"))
for h in res["hard_dvc"]["verification"]["solver"]["history"]:
    rows.append(dict(model=f"Rectum DVC: fix-and-resolve LP, iter {h['iter']}", kind="LP (approximation)",
                     n_vars=e["model"]["n_vars"],
                     n_constraints=e["model"]["n_constraints"] + sum(
                         v["n_voxels"] - v["allowed"] for v in res["hard_dvc"]["verification"]["dvc"].values()),
                     n_binaries=0, nnz="", solve_time_s=round(h["time"], 1), status="optimal" if h["success"] else h["message"],
                     gap=f"bracket [{res['hard_dvc']['bracket']['lower']:.3f}, {res['hard_dvc']['bracket']['upper']:.3f}]",
                     objective=round(h.get("objective", float("nan")), 4), plan_source="fix-and-resolve (approximation)"))
for name, nv, nc, nb, nnz, t in [
        ("Rectum DVC exact MILP (≤14% > 50 Gy), scipy.milp, full organ", 2364, 43501, 1643, 7167210, 904.7),
        ("Rectum DVC exact MILP (≤14% > 50 Gy), highspy, every 3rd voxel, warm start", 1269, 42406, 548, 6982905, 651.2),
        ("Rectum DVC exact MILP (≤14% > 50 Gy), row generation, iter 1", 1390, 5618, 669, "", 900.2)]:
        # sizes copied from results/logs/dvc_milp_*.log (these runs predate 03_run_hard.py)
    rows.append(dict(model=name, kind="MILP", n_vars=nv, n_constraints=nc, n_binaries=nb, nnz=nnz,
                     solve_time_s=t, status="time limit, no incumbent", gap="inf (no feasible point)", objective="",
                     plan_source="none"))
for K in Ks:
    s = sweep[str(K)]; m = s["milp"]
    rows.append(dict(model=f"Beam selection MILP, K={K} of 20", kind="MILP", n_vars=m["n_vars"],
                     n_constraints=m["n_constraints"], n_binaries=m["n_binaries"], nnz=m["nnz"],
                     solve_time_s=round(m["time"], 1), status=m["status"],
                     gap=f"{s['gap']:.1%} (bound {s['best_bound']:.3f}; nodes {m['nodes']})",
                     objective=round(s["objective"], 4),
                     plan_source=s.get("source") or ("MILP incumbent" if m["improved_on_start"] else "heuristic warm start (MILP did not improve)")))
with open(R / "model_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 7. per-structure dose table
def organ_fracs(d, S):
    v = d[S]
    # same tolerance as verify.py: voxels at the threshold up to solver round-off (~1e-11 Gy) are not "above"
    return {f"V{t}": round(100 * float((v > t * (1 + verify.REL_TOL)).mean()), 1) for t in (30, 50, 60, 70)}
drows = []
for plan, d in (("easy", d_easy), ("hard_bao_K5", d_bao), ("hard_dvc", d_dvc)):
    for s in STRUCTS:
        v = d[sets[s]]
        row = dict(plan=plan, structure=s, n_voxels=len(v), min=round(float(v.min()), 2),
                   mean=round(float(v.mean()), 2), max=round(float(v.max()), 2))
        if s.startswith("PTV"):
            row["D95"] = round(float(np.percentile(v, 5)), 2)
        if s in ("Rectum", "Bladder"):
            row.update({k: f"{val}%" for k, val in organ_fracs(d, sets[s]).items()})
        drows.append(row)
keys = ["plan", "structure", "n_voxels", "min", "mean", "max", "D95", "V30", "V50", "V60", "V70"]
with open(R / "dose_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(drows)
print("wrote figures fig1..fig5, model_summary.csv, dose_table.csv")
