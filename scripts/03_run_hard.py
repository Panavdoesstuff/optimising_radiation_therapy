"""Phase 4/5: hard problems.

(a) Beam angle selection MILP (headline hard problem): sweep K, warm start from a heuristic,
    600 s limit per K, report incumbent, best bound and gap. Resumable: per-K results are
    stored in results/bao_sweep.json and skipped on rerun.
(b) Dose-volume constraint on the rectum (secondary): fix-and-resolve plan, bracketed by the
    easy LP (a relaxation). Labelled as an approximation.
Every plan is verified on the full dose grid with verify.py.
"""
import json, sys, time, warnings
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, hard_dvc, hard_bao, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
spec = easy_lp.Spec()
P5 = problem.build("prostate", config.PROSTATE_5)
sets = P5.sets("disjoint")
x_easy = np.load(R / "easy_solution.npz")["x"]
f_easy = json.loads((R / "results.json").read_text())["easy"]["verification"]["objective_recomputed"]


def verify_bao_plan(red, x, beams, solver_obj, extra):
    """Rebuild the FULL dose matrix for the selected beams only and verify there."""
    beams = sorted(beams)
    angles = [red.angles[b] for b in beams]
    P = problem.build("prostate", angles, use_cache=False)
    x_sel = np.concatenate([x[red.beam_of_column == b] for b in beams])
    v = verify.verify(P, sets, spec, x_sel, solver_objective=solver_obj, extra=extra)
    off = np.abs(x[~np.isin(red.beam_of_column, beams)]).max(initial=0.0)
    v["checks"]["x = 0 on unselected beams"] = dict(max_violation=float(off), relative=float(off), ok=off <= 1e-6)
    v["all_ok"] = v["all_ok"] and off <= 1e-6
    return v, P, x_sel


# ------------------------------------------------------------------ (b) DVC
print("== (b) dose-volume constraint (fix-and-resolve, approximation)")
dvcs = [hard_dvc.DVC(o, t, a) for o, t, a in config.DVC_FINAL]
m5 = easy_lp.build_model(P5, spec, sets)
d_easy = P5.dose(x_easy)
easy_viol = {o: dict(t=t, alpha=a, easy_frac_over=float((d_easy[sets[o]] > t).mean()),
                     easy_n_over=int((d_easy[sets[o]] > t).sum()),
                     allowed=int(np.floor(a * len(sets[o])))) for o, t, a in config.DVC_FINAL}
print("   easy plan vs DVC:", easy_viol)
fr = hard_dvc.fix_and_resolve(P5, m5, dvcs, x_easy, n_iter=4)
assert fr["best"] is not None, "fix-and-resolve found no feasible plan"
x_dvc = fr["best"]["x"]
v_dvc = verify.verify(P5, sets, spec, x_dvc, solver_objective=fr["best"]["objective"],
                      dvcs=config.DVC_FINAL, extra=dict(method="fix-and-resolve (approximation)", history=fr["history"]))
verify.print_report("hard DVC (fix-and-resolve)", v_dvc)
assert v_dvc["all_ok"]
np.savez(R / "hard_dvc_solution.npz", x=x_dvc, angles=np.array(P5.angles))
verify.update_results("hard_dvc", dict(
    dvcs=config.DVC_FINAL, easy_plan_violation=easy_viol,
    bracket=dict(lower=f_easy, lower_source="easy LP (relaxation of the DVC problem)",
                 upper=v_dvc["objective_recomputed"], upper_source="fix-and-resolve feasible plan",
                 rel_gap=(v_dvc["objective_recomputed"] - f_easy) / v_dvc["objective_recomputed"]),
    exact_milp_note=("Exact MILP (big-M, scipy/HiGHS and highspy, full and subsampled, row generation) "
                     "found no incumbent within 600-900 s at rectum <=14% above 50 Gy; root bound equalled the easy LP."),
    verification=v_dvc))

# ------------------------------------------------------------------ (a) BAO
print("== (a) beam angle selection")
red = hard_bao.build_reduced(config.BAO_CANDIDATES, sets)
sweep_path = R / "bao_sweep.json"
sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
if "all_beams_lp" not in sweep:
    ra = hard_bao.all_beams_lp(red, spec)
    np.save(R / "bao_all_beams_x.npy", ra["x"])
    sweep["all_beams_lp"] = dict(objective=ra["objective"], time=ra["time"], n_candidates=len(red.angles))
    sweep_path.write_text(json.dumps(sweep, indent=2))
x_all = np.load(R / "bao_all_beams_x.npy")
lb_all = sweep["all_beams_lp"]["objective"]
print(f"   all {len(red.angles)} candidates LP (lower bound for every K): {lb_all:.4f}")

for K in config.BAO_K_VALUES:
    if str(K) in sweep:
        print(f"   K={K}: done ({sweep[str(K)]['objective']:.4f}), skipping")
        continue
    t0 = time.time()
    h = hard_bao.heuristic(red, spec, K, x_all)
    start = h
    if K == 5:   # the easy plan's beams are candidates too
        eb = [config.BAO_CANDIDATES.index(a) for a in config.PROSTATE_5]
        e = hard_bao.restricted_lp(red, spec, eb)
        if e["success"] and e["objective"] < h["objective"]:
            start = e
    r = hard_bao.solve_bao(red, spec, K, time_limit=config.BAO_TIME_LIMIT, start=start)
    best = r if (r["success"] and r["objective"] <= start["objective"] + 1e-9) else dict(start, objective=start["objective"])
    beams = best["beams"]
    v, P_sel, x_sel = verify_bao_plan(red, best["x"], beams, best["objective"], extra=dict(
        backend=r.get("backend"), status=r["message"], best_bound=r.get("best_bound"), mip_gap=r.get("mip_gap"),
        node_count=r.get("node_count"), solve_time=r["solve_time"], heuristic_objective=h["objective"],
        start_objective=start["objective"], milp_improved_on_start=bool(r["success"] and r["objective"] < start["objective"] - 1e-7)))
    verify.print_report(f"BAO K={K}", v)
    assert v["all_ok"], f"verification failed for K={K}"
    lb = max(r["best_bound"] or -np.inf, lb_all)
    sweep[str(K)] = dict(K=K, objective=v["objective_recomputed"], angles=[red.angles[b] for b in beams],
                         best_bound=lb, gap=(v["objective_recomputed"] - lb) / v["objective_recomputed"],
                         milp=dict(status=r["message"], best_bound=r.get("best_bound"), mip_gap=r.get("mip_gap"),
                                   nodes=r.get("node_count"), time=r["solve_time"], n_vars=r["n_vars"],
                                   n_constraints=r["n_constraints"], n_binaries=r["n_binaries"], nnz=r["nnz"],
                                   improved_on_start=v["solver"]["milp_improved_on_start"]),
                         heuristic_objective=h["objective"], total_time=time.time() - t0, verification=v)
    np.savez(R / f"bao_K{K}_solution.npz", x=x_sel, angles=np.array(sorted(red.angles[b] for b in beams)))
    sweep_path.write_text(json.dumps(sweep, indent=2, default=verify._enc))
    print(f"   K={K}: obj {v['objective_recomputed']:.4f} beams {sweep[str(K)]['angles']} "
          f"bound {lb:.4f} gap {sweep[str(K)]['gap']:.1%} ({time.time()-t0:.0f}s)")

# Post-pass: the MILP found one-sided beam sets at larger K that the swap heuristic missed.
# Dropping one beam from the best (K+1)-plan gives a cheap extra candidate for K; keep it if better.
Ks_done = sorted(int(k) for k in sweep if k.isdigit())
for K in sorted(Ks_done, reverse=True)[1:]:
    if sweep[str(K)].get("drop_one_checked"):
        continue
    up = [red.angles.index(a) for a in sweep[str(K + 1)]["angles"]]
    cand = hard_bao.drop_one(red, spec, up)
    cur = sweep[str(K)]["objective"]
    print(f"   drop-one K={K}: best {cand['objective']:.4f} vs current {cur:.4f}")
    if cand["objective"] < cur - 1e-7:
        v, _, x_sel = verify_bao_plan(red, cand["x"], cand["beams"], cand["objective"],
                                      extra=dict(source="drop-one from K+1 plan"))
        verify.print_report(f"BAO K={K} (drop-one)", v)
        assert v["all_ok"]
        s = sweep[str(K)]
        s.update(objective=v["objective_recomputed"], angles=[red.angles[b] for b in cand["beams"]],
                 gap=(v["objective_recomputed"] - s["best_bound"]) / v["objective_recomputed"],
                 verification=v, source="drop-one from K+1 plan", replaced_objective=cur)
        np.savez(R / f"bao_K{K}_solution.npz", x=x_sel, angles=np.array(sorted(s["angles"])))
    sweep[str(K)]["drop_one_checked"] = True
    sweep_path.write_text(json.dumps(sweep, indent=2, default=verify._enc))

verify.update_results("hard_bao", dict(candidates=config.BAO_CANDIDATES, time_limit=config.BAO_TIME_LIMIT,
                                       sweep=sweep, params=asdict(spec)))
print("done")
