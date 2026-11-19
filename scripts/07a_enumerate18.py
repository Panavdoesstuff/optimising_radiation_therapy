"""Pruned enumeration on the 18-degree grid for K=4,5,6 (extends 06a_enumerate18.py; new output dir).
Usage: 07a_enumerate18.py K cutoff [sample_n [seed]]   (sample_n>0 => random sample, written to K{K}_sample.jsonl)
Resumable jsonl results/improve/cert18/K{K}.jsonl. Unresolved subsets are retried at the end with
max_iter=300 and a different seed stride. Certificate holds iff zero unresolved remain."""
import itertools, json, random, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, bao_search as bs

_orig = bs.rowgen_cutoff
def work(a):
    args, mi, ss = a
    bs.rowgen_cutoff = lambda red, spec, sets, beams, cutoff=None, **kw: _orig(red, spec, sets, beams, cutoff, seed_stride=ss, max_iter=mi)
    return bs.solve_subset(args)

def main():
    K = int(sys.argv[1]); CUT = float(sys.argv[2])
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out = config.RESULTS_DIR / "improve" / "cert18"; out.mkdir(parents=True, exist_ok=True)
    f = out / (f"K{K}_sample.jsonl" if ns else f"K{K}.jsonl")
    pool_rows = bs.load_pool()
    done = {}; 
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l); done[tuple(r["angles"])] = r
    allc = list(itertools.combinations(bs.CANDS, K))
    if ns: random.Random(seed).shuffle(allc); allc = allc[:ns]
    best = None; t0 = time.time()
    def run(pool, tasks, tag, mi=40, ss=10):
        nonlocal best
        for i, (rec, x, _) in enumerate(pool.imap_unordered(work, [(t, mi, ss) for t in tasks], chunksize=1), 1):
            rec["pass"] = tag
            with open(f, "a") as fh: fh.write(json.dumps(rec) + "\n")
            done[tuple(rec["angles"])] = rec
            if rec["outcome"] == "solved" and (best is None or rec["objective"] < best):
                best = rec["objective"]
                if x is not None and not ns: np.savez(out / f"best_K{K}.npz", x=x, angles=np.array(sorted(rec["angles"])))
            if i % 25 == 0 or rec["outcome"] == "solved":
                el = time.time() - t0
                print(f"[{tag} {i}/{len(tasks)}] {rec['angles']} {rec['outcome']} best={best} {el/60:.1f} min", flush=True)
    with Pool(4, initializer=bs.init_worker, initargs=(pool_rows,), maxtasksperchild=400) as pool:
        todo = [c for c in allc if c not in done]
        print(f"K={K} cutoff={CUT} total={len(allc)} todo={len(todo)}", flush=True)
        run(pool, [(c, CUT, CUT + 1.0) for c in todo], "main")
        for attempt, (mi, ss) in enumerate([(300, 7), (600, 3)]):
            unres = [c for c in allc if done[c]["outcome"] == "unresolved"]
            if not unres: break
            print(f"retry {len(unres)} unresolved max_iter={mi}", flush=True)
            run(pool, [(c, CUT, CUT + 1.0) for c in unres], f"retry{attempt}", mi, ss)
    from collections import Counter
    cnt = Counter(done[c]["outcome"] for c in allc)
    tot = sum(done[c]["time"] for c in allc)
    print("COUNTS", dict(cnt), "cpu_s", tot, "wall_min", (time.time()-t0)/60, flush=True)
    if ns:
        print(f"sample est: cpu/subset {tot/len(allc):.2f}s")
    json.dump(dict(K=K, cutoff=CUT, counts=dict(cnt), cpu_s=tot, wall_s=time.time()-t0, best=best), open(out / f"K{K}{'_sample' if ns else ''}_summary.json", "w"))

if __name__ == "__main__":
    main()
