"""Fine-angle coordinate descent + perturbation restarts. Usage: 06c_fine_search.py K [budget_hours]
Starts from the current best plan for K; caches every evaluation in results/improve/fine/evals.jsonl."""
import itertools, json, sys, time, random
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, fine_search

R = config.RESULTS_DIR
OUT = R / "improve" / "fine"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "evals.jsonl"
START = {5: [90, 126, 180, 198, 234], 4: [90, 126, 180, 270], 3: [90, 180, 270],
         6: [72, 108, 144, 180, 216, 252]}
STEPS = [-14, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 14]
NW = 3


def key(a): return tuple(sorted(a))


def load_cache():
    c = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            d = json.loads(l); c[key(d["angles"])] = d.get("objective")
    return c


def work(a):
    o, _ = fine_search.evaluate(a)
    return o


def main():
    K = int(sys.argv[1]); budget = float(sys.argv[2]) * 3600 if len(sys.argv) > 2 else 3600
    t_end = time.time() + budget
    cache = load_cache(); n_new = 0
    pool = np.load(R / "bao_body_pool.npy")
    pool = np.union1d(pool, np.load(R / "enum" / "pool_added.npy"))
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

    def neighbours(cur):
        for i, a in enumerate(cur):
            for s in STEPS:
                b = (a + s) % 360
                if b in cur: continue
                yield sorted(cur[:i] + [b] + cur[i + 1:])

    def descend(cur):
        f = batch([cur])[key(cur)]
        path = [(list(cur), f)]
        while time.time() < t_end:
            res = batch(list(neighbours(cur)))
            res = {k: v for k, v in res.items() if v is not None}
            k, v = min(res.items(), key=lambda kv: kv[1])
            if v < f - 1e-7:
                cur, f = list(k), v; path.append((cur, f)); print(f"IMPROVED {cur} {f:.4f}", flush=True)
            else:
                break
        return cur, f, path

    t0 = time.time()
    cur, f, path = descend(START[K])
    best, bf, paths = cur, f, {"main": path}
    print(f"local optimum {best} {bf:.4f}", flush=True)
    rng = random.Random(K); r = 0
    while time.time() < t_end:       # perturbation restarts: shift two beams at once
        r += 1
        p = list(best)
        for i in rng.sample(range(K), 2):
            p[i] = (p[i] + rng.choice([-12, -8, -6, -4, 4, 6, 8, 12])) % 360
        if len(set(p)) < K: continue
        print(f"restart {r}: perturbed {sorted(p)}", flush=True)
        c2, f2, pa = descend(sorted(p)); paths[f"restart{r}"] = pa
        print(f"  -> {c2} {f2:.4f}", flush=True)
        if f2 < bf - 1e-7: best, bf = c2, f2
    json.dump(dict(K=K, best=best, objective=bf, paths=paths, n_lps_new=n_new, n_cached=len(cache),
                   seconds=time.time() - t0), open(OUT / f"search_K{K}.json", "w"), indent=1)
    print("DONE", best, bf, flush=True)

if __name__ == "__main__":
    main()
