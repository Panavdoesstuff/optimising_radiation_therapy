"""Certify K=3 on the 18-degree grid (20 candidates) by enumerating all C(20,3)=1140 subsets.

Each subset: exact LP by row generation (src/bao_search.rowgen_cutoff). Pruning: every reduced LP is
a relaxation, so if its optimum already exceeds the incumbent cutoff (60.789 + 1e-6, the best known
plan, which is itself re-solved) the subset cannot beat it; it is recorded as 'pruned' with its valid
lower bound. Infeasible: relaxation infeasible => infeasible. Subsets of a known-infeasible subset
are not needed (K=3 is the smallest level). Certificate holds iff no 'unresolved' remain.
Resumable jsonl: results/improve/bao18/K3.jsonl. Usage: 06a_enumerate18.py [K] [cutoff]
"""
import itertools, json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, bao_search as bs

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    CUT = float(sys.argv[2]) if len(sys.argv) > 2 else 60.789 + 1e-6
    W = 4
    out_dir = config.RESULTS_DIR / "improve" / "bao18"; out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"K{K}.jsonl"
    pool_rows = bs.load_pool()
    done = set(); best = None
    if f.exists():
        for line in f.read_text().splitlines():
            rec = json.loads(line); done.add(tuple(rec["angles"]))
            if rec["outcome"] == "solved" and (best is None or rec["objective"] < best):
                best = rec["objective"]
    todo = [c for c in itertools.combinations(bs.CANDS, K) if c not in done]
    print(f"K={K}: cutoff {CUT}; {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with Pool(W, initializer=bs.init_worker, initargs=(pool_rows,)) as pool:
        tasks = [(c, CUT, CUT + 1.0) for c in todo]
        for i, (rec, x_sel, _) in enumerate(pool.imap_unordered(bs.solve_subset, tasks, chunksize=1), 1):
            with open(f, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if rec["outcome"] == "solved" and (best is None or rec["objective"] < best - 1e-9):
                best = rec["objective"]
                np.savez(out_dir / f"best_K{K}_enum.npz", x=x_sel, angles=np.array(sorted(rec["angles"])))
            el = time.time() - t0
            print(f"[{i}/{len(tasks)}] {rec['angles']} {rec['outcome']} {rec.get('objective', rec.get('lb', float('nan'))):.4f} "
                  f"({rec['time']:.0f}s) best {best} | {el/60:.0f} min, eta {el/i*(len(tasks)-i)/60:.0f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
