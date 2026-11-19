"""Fine-angle search round 2. Usage: 07b_search.py MODE K budget_hours
MODE: k6 (scan 6th beam on K=5 best, then descend top-3), desc (resume descent from best cached K), k5r (restarts from other starts).
Reads old cache results/improve/fine/evals.jsonl, appends to results/improve/fine2/evals.jsonl."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, fine_search

R = config.RESULTS_DIR
OLD = R / "improve" / "fine" / "evals.jsonl"
OUT = R / "improve" / "fine2"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "evals.jsonl"
STEPS = [-14, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 14]
NW = 3
key = lambda a: tuple(sorted(a))


def load_cache():
    c = {}
    for f in (OLD, CACHE):
        if f.exists():
            for l in f.read_text().splitlines():
                d = json.loads(l); c[key(d["angles"])] = d.get("objective")
    return c


def work(a):
    return fine_search.evaluate(a)[0]


def main():
    mode, K, budget = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) * 3600
    t0 = time.time(); t_end = t0 + budget
    cache = load_cache(); n_new = 0
    pool = np.union1d(np.load(R / "bao_body_pool.npy"), np.load(R / "enum" / "pool_added.npy"))
    pp = Pool(NW, initializer=fine_search.init_worker, initargs=(pool,))
    log = open(CACHE, "a")

    def batch(cands):
        nonlocal n_new
        todo = [c for c in dict.fromkeys(key(c) for c in cands) if c not in cache]
        for o in pp.imap_unordered(work, todo):
            cache[key(o["angles"])] = o.get("objective")
            log.write(json.dumps(o) + "\n"); log.flush(); n_new += 1
            print(f"   {o['angles']}: {o.get('objective')} ({o['time']:.0f}s)", flush=True)
        return {c: cache[c] for c in (key(c) for c in cands)}

    def best_of(k):
        return min(((a, v) for a, v in cache.items() if len(a) == k and v is not None), key=lambda t: t[1])

    def descend(cur):
        cur = list(cur); f = batch([cur])[key(cur)]; path = [(cur, f)]; conv = False
        while True:
            if time.time() > t_end: break
            cands = []
            for i, a in enumerate(cur):
                for s in STEPS:
                    b = (a + s) % 360
                    if b not in cur: cands.append(sorted(cur[:i] + [b] + cur[i + 1:]))
            res = {k: v for k, v in batch(cands).items() if v is not None}
            # a sweep is only complete if all candidates were evaluated (they are: batch is not interrupted)
            k, v = min(res.items(), key=lambda kv: kv[1])
            if v < f - 1e-7:
                cur, f = list(k), v; path.append((cur, f)); print(f"IMPROVED {cur} {f:.4f}", flush=True)
            else:
                conv = True; break
        return cur, f, path, conv

    summ = dict(mode=mode, K=K, runs=[])
    if mode == "k6":
        base = list(best_of(5)[0]); print("K5 base", base, best_of(5)[1], flush=True)
        cands = [sorted(base + [a]) for a in range(0, 360, 20) if all(min((a - b) % 360, (b - a) % 360) >= 8 for b in base)]
        res = {k: v for k, v in batch(cands).items() if v is not None}
        # also include previously known best 6-beam plan as a start
        starts = [list(k) for k, v in sorted(res.items(), key=lambda kv: kv[1])[:2]]
        old6 = list(best_of(6)[0])
        if old6 not in starts: starts.append(old6)
        print("scan top", [(s, cache[key(s)]) for s in starts], flush=True)
        summ["scan"] = {str(k): v for k, v in res.items()}
        for s in starts:
            if time.time() > t_end: break
            c, f, p, conv = descend(s); summ["runs"].append(dict(start=s, end=c, f=f, converged=conv, path=p))
            print(f"-> {c} {f:.4f} conv={conv}", flush=True)
    elif mode == "desc":
        s = list(best_of(K)[0]); c, f, p, conv = descend(s)
        summ["runs"].append(dict(start=s, end=c, f=f, converged=conv, path=p)); print(f"-> {c} {f:.4f} conv={conv}", flush=True)
    elif mode == "k5r":
        starts = [[126, 180, 198, 234, 270], [18, 90, 126, 180, 234], [126, 162, 180, 234, 270], [72, 126, 162, 180, 270]]
        for s in starts:
            if time.time() > t_end: break
            c, f, p, conv = descend(s); summ["runs"].append(dict(start=s, end=c, f=f, converged=conv, path=p))
            print(f"-> {c} {f:.4f} conv={conv}", flush=True)
    summ["n_lps_new"] = n_new; summ["seconds"] = time.time() - t0
    json.dump(summ, open(OUT / f"search_{mode}_K{K}.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
