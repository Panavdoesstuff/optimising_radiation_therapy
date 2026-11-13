"""Exact beam selection by enumeration on the 36-degree grid (10 candidates).

For each K, every K-subset of the candidates is solved with the exact LP (row generation over all
target, organ and body-cap rows, converged against every row and the full-grid body cap) and checked with verify.py on the full dose matrices of
its beams. The minimum over all subsets is the certified optimum of "choose at most K of these 10
beams" (subsets with fewer beams are dominated by a superset). A subset whose reduced LP is
infeasible is infeasible for the full problem too (the reduced LP is a relaxation).

Resumable: one JSON line per subset in results/enum/K{K}.jsonl. Parallel: ENUM_WORKERS processes,
each seeded with the saved body-row pool.
"""
import itertools, json, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp

OUT = None
_W = {}


def _init(pool_rows):
    warnings.simplefilter("ignore")
    from src import config, problem, easy_lp, hard_bao
    sets = problem.build("prostate", config.PROSTATE_5).sets("disjoint")
    red = hard_bao.build_reduced(config.ENUM_CANDIDATES, sets)
    red.beam_cache = {}
    hard_bao.add_rows(red, pool_rows)
    _W.update(sets=sets, red=red, spec=easy_lp.Spec(), grid=problem.build("prostate", config.PROSTATE_5).grid)


def _solve(angles):
    from src import problem, hard_bao, verify
    red, sets, spec = _W["red"], _W["sets"], _W["spec"]
    t0 = time.time()
    n_body0 = len(red.body)
    beams = [red.angles.index(a) for a in angles]
    r = hard_bao.rowgen_active(red, spec, sets, beams)
    out = dict(angles=list(angles), K=len(angles), time=None, rowgen_rounds=len(r.get("rowgen", [])),
               status=r["message"], converged=bool(r.get("converged")), method="rowgen_active (all rows)")
    x_sel = None
    if r["success"] and r["converged"]:
        # independent full-grid verification on the beams' full dose matrices
        mats = [hard_bao.beam_D(red, a) for a in sorted(angles)]
        boc = np.concatenate([np.full(m.shape[1], i, np.int32) for i, m in enumerate(mats)])
        P = problem.Problem("prostate", sorted(angles), sp.hstack(mats, format="csr"), boc, _W["grid"])
        x_sel = np.concatenate([r["x"][red.beam_of_column == b] for b in sorted(beams)])
        v = verify.verify(P, sets, spec, x_sel, solver_objective=r["objective"])
        out.update(objective=v["objective_recomputed"], verified=bool(v["all_ok"]),
                   max_rel_violation=max(c.get("relative", 0.0) for c in v["checks"].values()),
                   max_body_dose=v["dose_stats"]["BODY"]["max"])
        out["outcome"] = "solved"
    elif not r["success"] and "infeasible" in r["message"].lower():
        out["outcome"] = "infeasible"          # relaxation infeasible => full problem infeasible
    else:
        out["outcome"] = "unresolved"          # numerical trouble or no convergence: breaks the certificate
    out["time"] = time.time() - t0
    new_rows = red.rows[red.body[n_body0:]].tolist()
    return out, x_sel, new_rows


def main():
    warnings.simplefilter("ignore")
    from src import config
    R = config.RESULTS_DIR
    out_dir = R / "enum"; out_dir.mkdir(exist_ok=True)
    pool_path = R / "bao_body_pool.npy"
    pool_rows = np.load(pool_path) if pool_path.exists() else np.zeros(0, np.int64)
    extra_path = out_dir / "pool_added.npy"
    if extra_path.exists():
        pool_rows = np.union1d(pool_rows, np.load(extra_path))
    print(f"candidates {config.ENUM_CANDIDATES}; body-row pool {len(pool_rows)}; workers {config.ENUM_WORKERS}", flush=True)

    best = {}
    tasks = []
    for K in config.ENUM_K_VALUES:
        f = out_dir / f"K{K}.jsonl"
        done = set()
        if f.exists():
            for line in f.read_text().splitlines():
                rec = json.loads(line); done.add(tuple(rec["angles"]))
                if rec.get("outcome") == "solved" and (K not in best or rec["objective"] < best[K]):
                    best[K] = rec["objective"]
        todo = [c for c in itertools.combinations(config.ENUM_CANDIDATES, K) if c not in done]
        print(f"K={K}: {len(done)} done, {len(todo)} to go", flush=True)
        tasks += todo
    if not tasks:
        print("nothing to do"); return

    added = np.zeros(0, np.int64)
    t_start = time.time()
    with Pool(config.ENUM_WORKERS, initializer=_init, initargs=(pool_rows,)) as pool:
        for i, (rec, x_sel, new_rows) in enumerate(pool.imap_unordered(_solve, tasks, chunksize=1), 1):
            K = rec["K"]
            with open(out_dir / f"K{K}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["outcome"] == "solved" and (K not in best or rec["objective"] < best[K] - 1e-9):
                best[K] = rec["objective"]
                np.savez(out_dir / f"best_K{K}.npz", x=x_sel, angles=np.array(sorted(rec["angles"])))
            if new_rows:
                added = np.union1d(added, new_rows)
                if i % 10 == 0:
                    np.save(extra_path, np.union1d(added, np.load(extra_path)) if extra_path.exists() else added)
            el = time.time() - t_start
            print(f"[{i}/{len(tasks)}] K={K} {rec['angles']} {rec['outcome']} "
                  f"{rec.get('objective', float('nan')):.4f} ({rec['time']:.0f}s) best K={K}: {best.get(K)} "
                  f"| elapsed {el/60:.0f} min, eta {el/i*(len(tasks)-i)/60:.0f} min", flush=True)
    np.save(extra_path, np.union1d(added, np.load(extra_path)) if extra_path.exists() else added)
    print("done")


if __name__ == "__main__":
    main()
