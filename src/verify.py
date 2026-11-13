"""Independent feasibility / optimality checks. Uses only P.D, structure sets and the spec;
never the solver's model matrices."""
import json
import numpy as np
from . import config

REL_TOL = 1e-6


def _viol(name, amount, scale, out):
    amount = float(max(amount, 0.0))
    out[name] = dict(max_violation=amount, relative=amount / max(abs(scale), 1.0), ok=amount / max(abs(scale), 1.0) <= REL_TOL)


def dose_stats(d, S, rx=None):
    v = d[S]
    out = dict(n=int(len(S)), min=float(v.min()), mean=float(v.mean()), max=float(v.max()))
    if rx is not None:
        out["D95"] = float(np.percentile(v, 5))     # dose received by >=95% of the volume
        out["frac_below_rx"] = float((v < rx - 1e-6 * rx).mean())
    return out


def verify(P, sets, spec, x, solver_objective=None, dvcs=(), extra=None):
    """x: beamlet intensities (length n_beamlets). dvcs: iterable of (name, t, alpha)."""
    x = np.asarray(x, float)[:P.n_beamlets]
    d = P.dose(x)
    T68, T56, R, B, N = (sets[k] for k in ("PTV_68", "PTV_56", "Rectum", "Bladder", "BODY"))
    cap = np.unique(np.concatenate([T68, T56, R, B]))
    checks = {}
    _viol("x >= 0", (-x).max(), 1, checks)
    _viol("x <= x_max", (x - spec.x_max).max(), spec.x_max, checks)
    s68 = np.maximum(spec.rx68 - d[T68], 0); s56 = np.maximum(spec.rx56 - d[T56], 0)
    _viol("T68 d>=rx68" + (" (slack allowed)" if spec.slack68 else ""),
          0 if spec.slack68 else s68.max(), spec.rx68, checks)
    _viol("T56 d>=rx56" + (" (slack allowed)" if spec.slack56 else ""),
          0 if spec.slack56 else s56.max(), spec.rx56, checks)
    _viol("d<=d_cap on T68,T56,R,B", (d[cap] - spec.d_cap).max(), spec.d_cap, checks)
    if getattr(spec, "body_cap", False):
        # every BODY voxel outside the targets, on the full grid (not just generated rows)
        _viol("d<=d_cap on all BODY voxels outside targets", (d[N] - spec.d_cap).max(), spec.d_cap, checks)

    dv = {}
    for name, t, alpha in dvcs:
        S = sets[name]
        n_over = int((d[S] > t * (1 + REL_TOL)).sum()); allowed = int(np.floor(alpha * len(S)))
        dv[name] = dict(t=t, alpha=alpha, n_voxels=int(len(S)), n_over=n_over, allowed=allowed,
                        frac_over=n_over / len(S), ok=n_over <= allowed)
        checks[f"DVC {name}: <= {alpha:.0%} above {t} Gy"] = dict(ok=n_over <= allowed, n_over=n_over, allowed=allowed)

    terms = dict(Rectum=float(d[R].mean()), Bladder=float(d[B].mean()), BODY=float(d[N].mean()))
    weighted = dict(Rectum=spec.w_R * terms["Rectum"], Bladder=spec.w_B * terms["Bladder"], BODY=spec.w_N * terms["BODY"])
    obj = sum(weighted.values())
    pen = 0.0
    if spec.slack68: pen += spec.rho * s68.mean()
    if spec.slack56: pen += spec.rho * s56.mean()
    obj_total = obj + pen
    res = dict(
        checks=checks, all_ok=all(c["ok"] for c in checks.values()),
        objective_recomputed=float(obj_total), objective_terms_mean_dose=terms,
        objective_terms_weighted=weighted, slack_penalty=float(pen),
        solver_objective=solver_objective,
        objective_rel_diff=None if solver_objective is None else abs(obj_total - solver_objective) / max(abs(obj_total), 1e-12),
        dvc=dv,
        key_variables=dict(
            n_nonzero=int((x > 1e-9).sum()), n_beamlets=int(len(x)),
            n_at_upper_bound=int((x >= spec.x_max * (1 - 1e-9)).sum()),
            max_intensity=float(x.max()), mean_intensity=float(x.mean()),
            mean_nonzero_intensity=float(x[x > 1e-9].mean()) if (x > 1e-9).any() else 0.0,
            total_intensity_per_beam={str(P.angles[b]): float(x[P.beam_of_column == b].sum()) for b in range(len(P.angles))}),
        dose_stats={k: dose_stats(d, sets[k], rx) for k, rx in
                    (("PTV_68", spec.rx68), ("PTV_56", spec.rx56), ("Rectum", None), ("Bladder", None), ("BODY", None))},
    )
    if extra:
        res["solver"] = extra
    res["all_ok"] = res["all_ok"] and (res["objective_rel_diff"] is None or res["objective_rel_diff"] < 1e-5)
    return res


def _enc(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(type(o))


def update_results(key, obj, path=None):
    """Merge `obj` under `key` into results/results.json."""
    path = path or (config.RESULTS_DIR / "results.json")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[key] = obj
    path.write_text(json.dumps(data, indent=2, default=_enc))


def print_report(name, v):
    print(f"--- verify: {name}: {'PASS' if v['all_ok'] else 'FAIL'}")
    for k, c in v["checks"].items():
        extra = f"max viol {c['max_violation']:.2e} (rel {c['relative']:.1e})" if "max_violation" in c else f"over={c['n_over']} allowed={c['allowed']}"
        print(f"    [{'ok' if c['ok'] else 'XX'}] {k}: {extra}")
    print(f"    objective recomputed {v['objective_recomputed']:.6f}  solver {v['solver_objective']}  rel diff {v['objective_rel_diff']}")
    print("    key vars:", {k: v['key_variables'][k] for k in ("n_nonzero", "n_at_upper_bound", "max_intensity", "mean_intensity")})
