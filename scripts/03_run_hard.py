"""Phase 4/5: hard problems, with the normal-tissue (BODY) dose cap.

(a) Beam angle selection (headline). Per K:
    lower bound = best bound of the MILP built on every 3rd constraint row (targets, rectum,
                  bladder, generated body rows) with full-organ cost vectors - a relaxation;
                  also never below the all-20-beam LP.
    upper bound = exact LP (body cap by row generation) on the chosen beams, verified on the
                  full dose grid of those beams.
    Plus a drop-one post-pass as an extra, labelled source of upper bounds.
    Resumable: results/bao_sweep.json (per K) and results/bao_body_pool.npy (body-row pool).
(b) Rectum dose-volume constraint (secondary): fix-and-resolve plan with the body cap,
    bracketed by the easy LP (a relaxation). Labelled as an approximation.
"""
import json, os, runpy, sys, time, warnings
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Default: results exist -> only re-verify them (a few minutes). Set RERUN=1 to recompute (hours).
_R = Path(__file__).resolve().parents[1] / "results"
if not os.environ.get("RERUN") and (_R / "bao_sweep.json").exists() and (_R / "easy_solution.npz").exists():
    print("Saved results found: re-verifying them (set RERUN=1 to re-solve; takes several hours).")
    runpy.run_path(str(Path(__file__).with_name("05_verify_all.py")), run_name="__main__")
    sys.exit(0)
import numpy as np
from src import config, problem, easy_lp, hard_dvc, hard_bao, verify

warnings.simplefilter("ignore")
R = config.RESULTS_DIR
spec = easy_lp.Spec()
P5 = problem.build("prostate", config.PROSTATE_5)
sets = P5.sets("disjoint")
z_easy = np.load(R / "easy_solution.npz")
x_easy, easy_rows = z_easy["x"], z_easy["body_rows"]
f_easy = json.loads((R / "results.json").read_text())["easy"]["verification"]["objective_recomputed"]
print(f"body cap: {spec.body_cap} (d <= {spec.d_cap} Gy on BODY outside targets); easy LP = {f_easy:.4f}")


def verify_bao_plan(red, x, beams, solver_obj, extra):
    """Rebuild the FULL dose matrix for the selected beams only and verify there."""
    beams = sorted(beams)
    P = problem.build("prostate", [red.angles[b] for b in beams], use_cache=False)
    x_sel = np.concatenate([x[red.beam_of_column == b] for b in beams])
    v = verify.verify(P, sets, spec, x_sel, solver_objective=solver_obj, extra=extra)
    off = np.abs(x[~np.isin(red.beam_of_column, beams)]).max(initial=0.0)
    v["checks"]["x = 0 on unselected beams"] = dict(max_violation=float(off), relative=float(off), ok=off <= 1e-6)
    v["all_ok"] = v["all_ok"] and off <= 1e-6
    return v, x_sel


# ------------------------------------------------------------------ (b) DVC
print("== (b) rectum dose-volume constraint (fix-and-resolve, approximation)")
dvcs = [hard_dvc.DVC(o, t, a) for o, t, a in config.DVC_FINAL]
d_easy = P5.dose(x_easy)
easy_viol = {o: dict(t=t, alpha=a, easy_frac_over=float((d_easy[sets[o]] > t * (1 + verify.REL_TOL)).mean()),
                     easy_n_over=int((d_easy[sets[o]] > t * (1 + verify.REL_TOL)).sum()),
                     allowed=int(np.floor(a * len(sets[o])))) for o, t, a in config.DVC_FINAL}
print("   easy plan vs DVC:", easy_viol)
fr = hard_dvc.fix_and_resolve(P5, spec, sets, dvcs, x_easy, n_iter=4, body0=easy_rows)
if fr["best"] is None:
    print("   fix-and-resolve found NO feasible plan with the body cap")
    verify.update_results("hard_dvc", dict(dvcs=config.DVC_FINAL, easy_plan_violation=easy_viol,
                                           feasible_plan_found=False, history=fr["history"]))
else:
    x_dvc = fr["best"]["x"]
    v_dvc = verify.verify(P5, sets, spec, x_dvc, solver_objective=fr["best"]["objective"], dvcs=config.DVC_FINAL,
                          extra=dict(method="fix-and-resolve (approximation)", history=fr["history"]))
    verify.print_report("hard DVC (fix-and-resolve)", v_dvc)
    assert v_dvc["all_ok"]
    np.savez(R / "hard_dvc_solution.npz", x=x_dvc, angles=np.array(P5.angles))
    verify.update_results("hard_dvc", dict(
        dvcs=config.DVC_FINAL, easy_plan_violation=easy_viol, feasible_plan_found=True,
        bracket=dict(lower=f_easy, lower_source="easy LP (relaxation of the DVC problem)",
                     upper=v_dvc["objective_recomputed"], upper_source="fix-and-resolve feasible plan",
                     rel_gap=(v_dvc["objective_recomputed"] - f_easy) / v_dvc["objective_recomputed"]),
        verification=v_dvc))

# ------------------------------------------------------------------ (a) BAO
print("== (a) beam angle selection")
red = hard_bao.build_reduced(config.BAO_CANDIDATES, sets)
pool_path = R / "bao_body_pool.npy"
seed_rows = np.union1d(easy_rows, np.load(pool_path)) if pool_path.exists() else easy_rows
print(f"   seeding body-row pool with {hard_bao.add_rows(red, seed_rows)} voxels")
save_pool = lambda: np.save(pool_path, red.rows[red.body])

sweep_path = R / "bao_sweep.json"
sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
save_sweep = lambda: sweep_path.write_text(json.dumps(sweep, indent=2, default=verify._enc))
all_idx = list(range(len(red.angles)))
if "all_beams_lp" not in sweep:
    t0 = time.time()
    ra = hard_bao.rowgen_restricted(red, spec, sets, all_idx)
    assert ra["converged"], "all-beam LP did not converge"
    np.save(R / "bao_all_beams_x.npy", ra["x"])
    sweep["all_beams_lp"] = dict(objective=ra["objective"], time=time.time() - t0, n_candidates=len(red.angles),
                                 rowgen=ra["rowgen"], max_body_dose=ra["max_body_dose"])
    save_pool(); save_sweep()
x_all = np.load(R / "bao_all_beams_x.npy")
lb_all = sweep["all_beams_lp"]["objective"]
print(f"   all {len(red.angles)} candidates LP, body-capped (lower bound for every K): {lb_all:.4f}")


def exact_plan(beams, label):
    """Exact (row-generated) LP on a beam set; returns None if infeasible."""
    print(f"   exact LP on {label} beams {sorted(red.angles[b] for b in beams)}", flush=True)
    r = hard_bao.rowgen_restricted(red, spec, sets, list(beams))
    save_pool()
    return r if (r["success"] and r["converged"]) else None


for K in config.BAO_K_VALUES:
    if str(K) in sweep:
        print(f"   K={K}: done ({sweep[str(K)].get('objective')}), skipping")
        continue
    t0 = time.time()
    tl = config.BAO_TIME_LIMITS.get(K, config.BAO_TIME_LIMIT)
    h = hard_bao.heuristic(red, spec, K, x_all)
    start = h if h["success"] else None
    if K == 5:   # the easy plan's beams are candidates too
        e = hard_bao.restricted_lp(red, spec, [config.BAO_CANDIDATES.index(a) for a in config.PROSTATE_5])
        if e["success"] and (start is None or e["objective"] < start["objective"]):
            start = e
    print(f"   K={K}: MILP on every {config.BAO_MILP_STRIDE}rd row, {tl:.0f}s limit, "
          f"start {None if start is None else round(start['objective'], 4)}", flush=True)
    r = hard_bao.solve_bao(red, spec, K, time_limit=tl, start=start, stride=config.BAO_MILP_STRIDE)
    print(f"     MILP: {r['message']}  incumbent {r.get('objective')}  bound {r.get('best_bound')}  "
          f"nodes {r.get('node_count')}  rows {r['n_constraints']}  ({r['solve_time']:.0f}s)", flush=True)
    # candidate beam sets: the MILP's choice and the heuristic start
    cands = []
    if r["success"]:
        cands.append(("MILP", r["beams"]))
    if start is not None and (not cands or sorted(start["beams"]) != sorted(cands[0][1])):
        cands.append(("heuristic start", start["beams"]))
    best, best_src = None, None
    for src, beams in cands:
        up = exact_plan(beams, src)
        if up is not None and (best is None or up["objective"] < best["objective"]):
            best, best_src = up, src
    milp_bound = r.get("best_bound")
    milp_bound = milp_bound if (milp_bound is not None and np.isfinite(milp_bound)) else None
    lb = max(milp_bound if milp_bound is not None else -np.inf, lb_all)
    rec = dict(K=K, lower_bound=lb, best_bound=lb, lower_source="MILP bound (subsampled rows)"
               if milp_bound is not None and milp_bound >= lb_all else "all-20-beam LP",
               milp=dict(status=r["message"], best_bound=milp_bound, mip_gap=r.get("mip_gap"),
                         incumbent_subsampled=r.get("objective"), nodes=r.get("node_count"),
                         time=r["solve_time"], time_limit=tl, n_vars=r["n_vars"], n_constraints=r["n_constraints"],
                         n_binaries=r["n_binaries"], nnz=r["nnz"], stride=r["stride"],
                         angles=r.get("angles_selected")),
               heuristic_objective_pool_rows=h.get("objective"), total_time=None)
    if best is None:
        rec.update(objective=None, gap=None, note="no feasible plan found for this K")
        print(f"   K={K}: NO feasible plan (lower bound {lb:.4f})")
    else:
        v, x_sel = verify_bao_plan(red, best["x"], best["beams"], best["objective"],
                                   extra=dict(source=best_src, rowgen=best["rowgen"]))
        verify.print_report(f"BAO K={K}", v)
        assert v["all_ok"], f"verification failed for K={K}"
        rec.update(objective=v["objective_recomputed"], angles=[red.angles[b] for b in best["beams"]],
                   gap=(v["objective_recomputed"] - lb) / v["objective_recomputed"], source=best_src,
                   verification=v)
        np.savez(R / f"bao_K{K}_solution.npz", x=x_sel, angles=np.array(sorted(rec["angles"])))
        print(f"   K={K}: upper {rec['objective']:.4f} ({best_src}) beams {rec['angles']}  "
              f"lower {lb:.4f}  gap {rec['gap']:.1%}", flush=True)
    rec["total_time"] = time.time() - t0
    sweep[str(K)] = rec
    save_sweep()

# Post-pass: drop one beam from the best (K+1)-plan; keep the result if its exact LP is better.
Ks_done = sorted(int(k) for k in sweep if k.isdigit())
for K in sorted(Ks_done, reverse=True)[1:]:
    s, up = sweep[str(K)], sweep.get(str(K + 1), {})
    if s.get("drop_one_checked") or not up.get("angles"):
        continue
    cand = hard_bao.drop_one(red, spec, [red.angles.index(a) for a in up["angles"]])
    cur = s.get("objective")
    if cand is not None:
        ex = exact_plan(cand["beams"], f"drop-one K={K}")
        print(f"   drop-one K={K}: exact {None if ex is None else round(ex['objective'], 4)} vs current {cur}")
        if ex is not None and (cur is None or ex["objective"] < cur - 1e-7):
            v, x_sel = verify_bao_plan(red, ex["x"], ex["beams"], ex["objective"],
                                       extra=dict(source="drop-one from K+1 plan", rowgen=ex["rowgen"]))
            verify.print_report(f"BAO K={K} (drop-one)", v)
            assert v["all_ok"]
            s.update(objective=v["objective_recomputed"], angles=[red.angles[b] for b in ex["beams"]],
                     gap=(v["objective_recomputed"] - s["lower_bound"]) / v["objective_recomputed"],
                     verification=v, source="drop-one from K+1 plan", replaced_objective=cur)
            np.savez(R / f"bao_K{K}_solution.npz", x=x_sel, angles=np.array(sorted(s["angles"])))
    s["drop_one_checked"] = True
    save_sweep()

verify.update_results("hard_bao", dict(candidates=config.BAO_CANDIDATES, time_limits=dict(
    default=config.BAO_TIME_LIMIT, **{str(k): v for k, v in config.BAO_TIME_LIMITS.items()}),
    milp_stride=config.BAO_MILP_STRIDE, body_pool_size=int(len(red.body)), sweep=sweep, params=asdict(spec)))
print("done")
