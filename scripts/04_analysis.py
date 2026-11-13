"""Phase 6: figures and tables from saved solutions (no optimisation is re-run).

Current results (with the normal-tissue dose cap) are in results/; the earlier run without the
cap is in results/no_body_cap/ and is used only for the before/after comparison.
"""
import csv, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import distance_transform_edt
from src import config, problem, io_cort, hard_bao, plots, verify, easy_lp
easy_lp_spec = easy_lp.Spec()

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
OLD = R / "no_body_cap"
res = json.loads((R / "results.json").read_text())
sweep = json.loads((R / "bao_sweep.json").read_text())
old_res = json.loads((OLD / "results.json").read_text())
old_sweep = json.loads((OLD / "bao_sweep.json").read_text())

# ---------------- plans
P5 = problem.build("prostate", config.PROSTATE_5)
sets = P5.sets("disjoint")
x_easy = np.load(R / "easy_solution.npz")["x"]
has_dvc = res["hard_dvc"].get("feasible_plan_found", False)
x_dvc = np.load(R / "hard_dvc_solution.npz")["x"] if has_dvc else None
z5 = np.load(R / "enum" / "best_K5.npz")                      # certified optimum, 36-degree grid
bao_angles = [int(a) for a in z5["angles"]]
P_bao = problem.build("prostate", bao_angles, use_cache=False)
x_bao = z5["x"]
swap_final = json.loads((R / "swap_final.json").read_text()) if (R / "swap_final.json").exists() else {}
has_final = swap_final.get("final_plan", {}).get("feasible_plan_found", False)
x_final = np.load(R / "final_plan_solution.npz")["x"] if has_final else None
d_easy, d_bao = P5.dose(x_easy), P_bao.dose(x_bao)
d_dvc = P5.dose(x_dvc) if has_dvc else None
d_final = P_bao.dose(x_final) if has_final else None
alpha_dvc = config.DVC_FINAL[0][2]
lab_easy = "Easy LP: 5 equispaced beams"
lab_bao = f"Hard: certified best 5 of 10 beams ({', '.join(map(str, bao_angles))}°)"
lab_dvc = f"Equispaced beams + rectum ≤{alpha_dvc:.0%} above 50 Gy"
STRUCTS = ["PTV_68", "PTV_56", "Rectum", "Bladder", "BODY"]
plans = [("easy", lab_easy, d_easy), ("bao", lab_bao, d_bao)] + ([("dvc", lab_dvc, d_dvc)] if has_dvc else [])

# enumeration (certified optimum per K on the 36-degree grid)
enum = {}
for K in sorted(config.ENUM_K_VALUES):
    recs = [json.loads(l) for l in open(R / "enum" / f"K{K}.jsonl")]
    sol = [r for r in recs if r["outcome"] == "solved"]
    b = min(sol, key=lambda r: r["objective"])
    enum[K] = dict(n=len(recs), solved=len(sol), infeasible=sum(r["outcome"] == "infeasible" for r in recs),
                   unresolved=sum(r["outcome"] == "unresolved" for r in recs), opt=b["objective"], angles=b["angles"],
                   lp_time=sum(r["time"] for r in recs), max_viol=max(r["max_rel_violation"] for r in sol))
verify.update_results("enumeration", dict(candidates=config.ENUM_CANDIDATES, per_K={str(k): v for k, v in enum.items()}))

# 1. DVH
plots.fig_dvh(R / "fig1_dvh.png", plans, sets, STRUCTS, constraints=config.DVC_FINAL if has_dvc else (),
              title="Dose-volume histograms: easy LP vs hard plans (normal-tissue cap 74.8 Gy)")

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
V_e, V_b = vol(d_easy), vol(d_bao)
tops = [("Easy LP", V_e), ("Hard: certified best 5 beams", V_b)]
diffs = [("Certified best 5 − easy", V_b - V_e)]
if has_dvc:
    V_d = vol(d_dvc); tops.append(("Equispaced + rectum limit", V_d)); diffs.append(("Equispaced + limit − easy", V_d - V_e))
vmax = max(V[:, :, z].max() for _, V in tops)
plots.fig_dose_maps(R / "fig2_dose_maps.png", tops, masks, z, diffs, crop, vmax)

# 3. fluence maps
def beams_of(P, x):
    return [(g, io_cort.load_beaminfo("prostate", g), x[P.beam_of_column == b]) for b, g in enumerate(P.angles)]
plots.fig_fluence(R / "fig3_fluence.png", [("Easy LP", beams_of(P5, x_easy)), ("Hard: best 5", beams_of(P_bao, x_bao))],
                  max(x_easy.max(), x_bao.max()))

# 4. price of realism
terms = lambda v: v["objective_terms_weighted"]
red = hard_bao.build_reduced(config.BAO_CANDIDATES, sets)
x_all = np.load(R / "bao_all_beams_x.npy")
sp_ = res["easy"]["params"]
t_all = dict(Rectum=sp_["w_R"] * red.cost["R"] @ x_all, Bladder=sp_["w_B"] * red.cost["B"] @ x_all,
             BODY=sp_["w_N"] * red.cost["N"] @ x_all)
v_easy = res["easy"]["verification"]
rows4 = [("Easy LP\n5 equispaced beams", terms(v_easy), "")]
if has_dvc:
    rows4.append(("Equispaced beams\n+ rectum limit", terms(res["hard_dvc"]["verification"]), "approximation"))
v_cert = verify.verify(P_bao, sets, easy_lp_spec, x_bao)
rows4.append(("Hard: best 5 of 10 beams\n(certified, enumeration)", terms(v_cert), "optimal, gap 0"))
if swap_final.get("swap_search"):
    rows4.append(("Best 5 of 20 beams found\n(swap search)", terms(swap_final["swap_search"]["verification"]),
                  f"not certified; gap {swap_final['swap_search']['gap']:.0%}"))
if has_final:
    rows4.append(("Final: certified 5 beams\n+ rectum limit", terms(swap_final["final_plan"]["verification"]),
                  "limit already met: costs 0"))
rows4.append(("All 20 beams (LP)\nno beam limit", t_all, "lower bound for any K"))
plots.fig_price(R / "fig4_price_of_realism.png", rows4, ref_label="Easy LP\n5 equispaced beams")

# 5. price of a beam budget: certified optimum (10 candidates) + uncertified 20-candidate plans
Ks = sorted(enum)
K20 = sorted(int(k) for k in sweep if k.isdigit() and sweep[k].get("objective") is not None)
ub20 = {K: sweep[str(K)]["objective"] for K in K20}
if swap_final.get("swap_search"):
    ub20[5] = min(ub20.get(5, np.inf), swap_final["swap_search"]["objective"])
plots.fig_sweep_certified(R / "fig5_sweep_K.png", Ks, [enum[K]["opt"] for K in Ks], "10 candidates (every 36°)",
                          K20, [ub20[K] for K in K20], [sweep[str(K)]["lower_bound"] for K in K20],
                          sweep["all_beams_lp"]["objective"], (5, v_easy["objective_recomputed"]))

# 6. rectum voxels by distance to the PTV
ptv = np.zeros(P5.n_voxels, bool); ptv[sets["PTV_68"]] = True; ptv[sets["PTV_56"]] = True
ptv3 = problem.reshape_volume(ptv, grid)                         # [ap, rl, si]
vs = grid["voxel_size"]                                           # cm; (x=rl, y=ap, z=si)
spacing_mm = (abs(vs[1]) * 10, abs(vs[0]) * 10, abs(vs[2]) * 10)  # matches [ap, rl, si]
dist3 = distance_transform_edt(~ptv3, sampling=spacing_mm)
rect3 = masks["Rectum"]
dist_r = dist3.ravel(order="F")[sets["Rectum"]]
dproj = np.where(rect3, dist3, np.inf).min(axis=1)               # min over rl -> [ap, si]
dproj = np.where(np.isfinite(dproj), dproj, np.nan).T             # -> [si, ap]
plots.fig_rectum_distance(R / "fig6_rectum_distance.png", dproj, ptv3.any(axis=1).T, dist_r,
                          [("Easy LP (equispaced beams)", d_easy[sets["Rectum"]])]
                          + ([(f"Equispaced beams + limit ≤{alpha_dvc:.0%}", d_dvc[sets["Rectum"]])] if has_dvc else [])
                          + [("Certified best 5 beams", d_bao[sets["Rectum"]])],
                          50.0, alpha_dvc,
                          title="Rectal dose vs distance to the PTV, by plan")
r_alpha = float(np.percentile(dist_r, 100 * alpha_dvc))
hot = d_dvc[sets["Rectum"]] > 50 * (1 + verify.REL_TOL) if has_dvc else d_easy[sets["Rectum"]] > 50
rect_dist_stats = dict(alpha=alpha_dvc, distance_at_alpha_mm=r_alpha,
                       frac_within_3mm=float((dist_r <= 3.01).mean()), frac_within_6mm=float((dist_r <= 6.01).mean()),
                       hot_voxels_median_distance_mm=float(np.median(dist_r[hot])),
                       hot_voxels_max_distance_mm=float(dist_r[hot].max()))
verify.update_results("rectum_distance", rect_dist_stats)

# 7. model summary table
rows = []
e = res["easy"]; es = e["verification"]["solver"]
rows.append(dict(model="Easy LP (5 equispaced beams), body cap by row generation", kind="LP",
                 n_vars=e["model"]["n_vars"], n_constraints=e["model"]["n_constraints"], n_binaries=0,
                 nnz=e["model"]["nnz"], body_rows=f"{e['model']['body_rows']} of {e['model']['body_voxels_total']}",
                 solve_time_s=round(sum(h["time"] for h in es["rowgen"]), 1), status="optimal",
                 gap=f"duality gap {es['duality_gap']:.1e}", lower_bound="", objective=round(e["verification"]["objective_recomputed"], 4),
                 plan_source=f"LP optimum ({len(es['rowgen'])} row-generation rounds)"))
if has_dvc:
    hd = res["hard_dvc"]
    for h in hd["verification"]["solver"]["history"]:
        rows.append(dict(model=f"Rectum DVC: fix-and-resolve LP, iter {h['iter']}", kind="LP (approximation)",
                         n_vars=e["model"]["n_vars"], n_constraints=h["n_rows"], n_binaries=0, nnz="", body_rows="",
                         solve_time_s=round(h["time"], 1), status="optimal" if h["success"] else h["message"],
                         gap=f"bracket [{hd['bracket']['lower']:.3f}, {hd['bracket']['upper']:.3f}]", lower_bound=round(hd["bracket"]["lower"], 4),
                         objective=round(h.get("objective", float("nan")), 4), plan_source="fix-and-resolve (approximation)"))
for K in sorted(int(k) for k in sweep if k.isdigit()):
    s = sweep[str(K)]; m = s["milp"]
    rows.append(dict(model=f"Beam selection MILP, K={K} of 20 (every {m['stride']}rd row)", kind="MILP",
                     n_vars=m["n_vars"], n_constraints=m["n_constraints"], n_binaries=m["n_binaries"], nnz=m["nnz"],
                     body_rows="subsampled pool", solve_time_s=round(m["time"], 1), status=m["status"],
                     gap="" if s.get("gap") is None else f"{s['gap']:.1%} (nodes {m['nodes']})",
                     lower_bound=round(s["lower_bound"], 4),
                     objective="" if s.get("objective") is None else round(s["objective"], 4),
                     plan_source=s.get("source", "none") + " → exact LP with body row generation"))
for K in sorted(enum):
    e_ = enum[K]
    rows.append(dict(model=f"Beam selection by enumeration, K={K} of 10 (every 36°)", kind=f"{e_['n']} exact LPs",
                     n_vars="~140 per beam", n_constraints="row generation (all rows)", n_binaries="—", nnz="",
                     body_rows="shared pool", solve_time_s=round(e_["lp_time"], 0),
                     status=f"{e_['solved']} solved, {e_['infeasible']} infeasible, {e_['unresolved']} unresolved",
                     gap="0 (certified)" if e_["unresolved"] == 0 else "certificate incomplete",
                     lower_bound=round(e_["opt"], 4), objective=round(e_["opt"], 4),
                     plan_source=f"enumeration; every subset verified on the full grid (max viol {e_['max_viol']:.1e})"))
if swap_final.get("swap_search"):
    ss = swap_final["swap_search"]
    rows.append(dict(model="Swap local search, K=5 of 20 (±18° moves)", kind=f"{ss['n_lp_evaluations']} exact LPs",
                     n_vars="", n_constraints="row generation (all rows)", n_binaries="—", nnz="", body_rows="shared pool",
                     solve_time_s="", status="local optimum", gap=f"{ss['gap']:.1%} (not certified)",
                     lower_bound=round(ss["lower_bound_all20"], 4), objective=round(ss["objective"], 4),
                     plan_source=f"from {ss['best_from']}"))
if swap_final.get("final_plan"):
    fp = swap_final["final_plan"]
    rows.append(dict(model="Final: certified best 5 beams + body cap + rectum limit", kind="LP (approximation)",
                     n_vars="", n_constraints="", n_binaries=0, nnz="", body_rows="row generation", solve_time_s="",
                     status="feasible" if fp.get("feasible_plan_found") else "no plan found",
                     gap="" if not fp.get("feasible_plan_found") else f"bracket [{fp['bracket'][0]:.3f}, {fp['bracket'][1]:.3f}]",
                     lower_bound=round(fp["without_limit"]["objective"], 4),
                     objective=round(fp["objective"], 4) if fp.get("feasible_plan_found") else "",
                     plan_source="fix-and-resolve on the certified beams"))
with open(R / "model_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 8. per-structure dose table
def organ_fracs(d, S):
    v = d[S]   # same tolerance as verify.py
    return {f"V{t}": round(100 * float((v > t * (1 + verify.REL_TOL)).mean()), 1) for t in (30, 50, 60, 70)}
drows = []
for plan, d in [("easy", d_easy), ("certified_best5", d_bao)] + ([("easy_beams_dvc", d_dvc)] if has_dvc else []) \
        + ([("final_best5_dvc", d_final)] if has_final else []):
    for s in STRUCTS:
        v = d[sets[s]]
        row = dict(plan=plan, structure=s, n_voxels=len(v), min=round(float(v.min()), 2),
                   mean=round(float(v.mean()), 2), max=round(float(v.max()), 2))
        if s.startswith("PTV"):
            row["D95"] = round(float(np.percentile(v, 5)), 2)
        if s in ("Rectum", "Bladder"):
            row.update({k: f"{val}%" for k, val in organ_fracs(d, sets[s]).items()})
        drows.append(row)
with open(R / "dose_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["plan", "structure", "n_voxels", "min", "mean", "max", "D95", "V30", "V50", "V60", "V70"])
    w.writeheader(); w.writerows(drows)

# 9. before/after table (no cap vs cap)
def body_max(v):
    return v["dose_stats"]["BODY"]["max"]
x_all_old = np.load(OLD / "bao_all_beams_x.npy")
d_all_old = hard_bao.full_dose(red, x_all_old)
ba = [dict(plan="Easy LP (5 equispaced beams)",
           objective_no_cap=old_res["easy"]["verification"]["objective_recomputed"], objective_cap=v_easy["objective_recomputed"],
           max_body_no_cap=body_max(old_res["easy"]["verification"]), max_body_cap=body_max(v_easy)),
      dict(plan="Rectum DVC plan (fix-and-resolve)",
           objective_no_cap=old_res["hard_dvc"]["verification"]["objective_recomputed"],
           objective_cap=res["hard_dvc"]["verification"]["objective_recomputed"] if has_dvc else None,
           max_body_no_cap=body_max(old_res["hard_dvc"]["verification"]),
           max_body_cap=body_max(res["hard_dvc"]["verification"]) if has_dvc else None)]
for K in config.BAO_K_VALUES:
    o, n = old_sweep.get(str(K), {}), sweep.get(str(K), {})
    ba.append(dict(plan=f"Best plan, K={K} beams",
                   objective_no_cap=o.get("objective"), objective_cap=n.get("objective"),
                   max_body_no_cap=body_max(o["verification"]) if o.get("verification") else None,
                   max_body_cap=body_max(n["verification"]) if n.get("verification") else None))
ba.append(dict(plan="All 20 beams (LP)", objective_no_cap=old_sweep["all_beams_lp"]["objective"],
               objective_cap=sweep["all_beams_lp"]["objective"], max_body_no_cap=float(d_all_old[sets["BODY"]].max()),
               max_body_cap=sweep["all_beams_lp"]["max_body_dose"]))
for r in ba:
    a, b = r["objective_no_cap"], r["objective_cap"]
    r["objective_change_pct"] = None if (a is None or b is None) else round(100 * (b - a) / a, 2)
    for k in ("objective_no_cap", "objective_cap", "max_body_no_cap", "max_body_cap"):
        r[k] = None if r[k] is None else round(r[k], 3 if "objective" in k else 1)
with open(R / "before_after_body_cap.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["plan", "objective_no_cap", "objective_cap", "objective_change_pct",
                                      "max_body_no_cap", "max_body_cap"])
    w.writeheader(); w.writerows(ba)
print("wrote fig1..fig6, model_summary.csv, dose_table.csv, before_after_body_cap.csv")
print("rectum distance:", rect_dist_stats)
