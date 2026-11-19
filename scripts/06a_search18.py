"""Multi-start steepest-descent beam search on the 18-degree grid (20 candidates), K=4,5,6 (K=3 is
enumerated by 06a). Starts: top-10 36-degree enumeration plans of that K, earlier best plans, greedy
add (best of K-1 plus one beam) and greedy drop (certified 36-degree best of K+1 minus one beam).
Moves: swap any one beam for ANY unused candidate (all K*(20-K) neighbours evaluated per round,
steepest descent). Final polish (skipped if env NOPOLISH is set; env MAXSTART caps the number of starts, earlier-best/greedy first): all two-beam swaps of the best plan. Every evaluated subset is cached
(results/improve/bao18/search_cache.jsonl); neighbours whose relaxation already exceeds the current
objective are pruned (valid: relaxations are lower bounds) and stored with their lower bound.
All reported plans are verified on the full grid. Usage: 06a_search18.py K [K ...]
"""
import itertools, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, bao_search as bs

OUT = config.RESULTS_DIR / "improve" / "bao18"
CACHE = OUT / "search_cache.jsonl"
cache = {}      # tuple(angles) -> dict(exact=obj or None, lb=float)


def load_cache():
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            r = json.loads(l); cache[tuple(r["angles"])] = r


def known(key, cutoff):
    r = cache.get(key)
    if r is None:
        return False
    return r["exact"] is not None or (cutoff is not None and r["lb"] >= cutoff)


def evaluate_many(pool, subsets, cutoff):
    todo = [tuple(sorted(s)) for s in subsets]
    todo = [s for s in dict.fromkeys(todo) if not known(s, cutoff)]
    for rec, _, _ in pool.imap_unordered(bs.solve_subset, [(s, cutoff, None) for s in todo]):
        key = tuple(rec["angles"])
        if rec["outcome"] == "solved":
            e = dict(angles=rec["angles"], exact=rec["objective"], lb=rec["objective"])
        elif rec["outcome"] == "pruned":
            e = dict(angles=rec["angles"], exact=None, lb=rec["lb"])
        else:   # infeasible / unresolved: treat as unusable
            e = dict(angles=rec["angles"], exact=float("inf"), lb=float("inf"), outcome=rec["outcome"])
        cache[key] = e
        with open(CACHE, "a") as f:
            f.write(json.dumps(e) + "\n")
    return len(todo)


def obj(pool, s):
    s = tuple(sorted(s))
    if cache.get(s, {}).get("exact") is None:
        evaluate_many(pool, [s], None)
    return cache[s]["exact"]


def descend(pool, start, log):
    cur = tuple(sorted(start)); co = obj(pool, cur)
    while True:
        nb = [tuple(sorted(cur[:i] + (b,) + cur[i + 1:])) for i in range(len(cur))
              for b in bs.CANDS if b not in cur]
        evaluate_many(pool, nb, co)
        cands = [(cache[n]["exact"], n) for n in nb if cache[n]["exact"] is not None and cache[n]["exact"] < co - 1e-7]
        if not cands:
            return cur, co
        co, cur = min(cands)
        log(f"    -> {list(cur)} {co:.4f}")


def main():
    load_cache()
    Ks = [int(k) for k in sys.argv[1:]] or [4, 5, 6]
    pool_rows = bs.load_pool()
    R = config.RESULTS_DIR
    enum = {}
    for K in (3, 4, 5, 6, 7):
        f = R / "enum" / f"K{K}.jsonl"
        if f.exists():
            enum[K] = sorted((json.loads(l) for l in f.read_text().splitlines() if '"solved"' in l),
                             key=lambda r: r["objective"])
            for r in enum[K]:     # exact LP values on the 36-degree grid are exact on the 18 grid too
                cache.setdefault(tuple(sorted(r["angles"])), dict(angles=sorted(r["angles"]), exact=r["objective"], lb=r["objective"]))
    # K=3 certified plan from enumeration (if present) used as the greedy base for K=4
    best = {}
    f3 = OUT / "K3_final.json"
    if f3.exists():
        m = json.loads(f3.read_text()); best[3] = (tuple(sorted(m["angles"])), m["objective"])
    f4 = OUT / "search_K4.json"
    if f4.exists():
        m = json.loads(f4.read_text())["4"]; best[4] = (tuple(sorted(m["angles"])), m["objective"])
    summary = {}
    logf = open(OUT / "search_log.txt", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    with Pool(4, initializer=bs.init_worker, initargs=(pool_rows,)) as pool:
        for K in Ks:
            t0 = time.time()
            starts = {}
            for r in enum.get(K, [])[:10]:
                starts[tuple(sorted(r["angles"]))] = "36deg-top10"
            for prev in (R / f"bao_K{K}_solution.npz", R / "swap_K5_solution.npz" if K == 5 else None):
                if prev is not None and prev.exists():
                    starts[tuple(sorted(int(a) for a in np.load(prev)["angles"]))] = "earlier-best"
            if K - 1 in best:   # greedy add
                base = best[K - 1][0]
                evaluate_many(pool, [base + (b,) for b in bs.CANDS if b not in base], None)
                ad = sorted((cache[tuple(sorted(base + (b,)))]["exact"], tuple(sorted(base + (b,))))
                            for b in bs.CANDS if b not in base)[:3]
                for _, s in ad:
                    starts[s] = "greedy-add"
            if K + 1 in enum:   # greedy drop from certified 36-degree best of K+1
                base = tuple(sorted(enum[K + 1][0]["angles"]))
                dr = [tuple(a for a in base if a != d) for d in base]
                evaluate_many(pool, dr, None)
                for s in sorted(dr, key=lambda s: cache[s]["exact"])[:3]:
                    starts[s] = "greedy-drop"
            order = {"earlier-best": 0, "greedy-add": 1, "greedy-drop": 2, "36deg-top10": 3}
            starts = dict(sorted(starts.items(), key=lambda kv: order[kv[1]]))
            if os.environ.get("MAXSTART"):
                starts = dict(list(starts.items())[:int(os.environ["MAXSTART"])])
            log(f"== K={K}: {len(starts)} starts")
            runs = []
            top = None
            for s, why in starts.items():
                log(f"  start {list(s)} ({why}), obj {obj(pool, s):.4f}")
                fin, fo = descend(pool, s, log)
                log(f"  final {list(fin)} {fo:.4f}")
                runs.append(dict(start=list(s), why=why, final=list(fin), objective=fo))
                if top is None or fo < top[1]:
                    top = (fin, fo)
            # two-swap polish around the best plan
            cur, co = top
            while not os.environ.get('NOPOLISH'):
                nb = []
                for out_ in itertools.combinations(cur, 2):
                    rest = [a for a in cur if a not in out_]
                    for inn in itertools.combinations([b for b in bs.CANDS if b not in cur], 2):
                        nb.append(tuple(sorted(rest + list(inn))))
                log(f"  2-swap polish: {len(nb)} neighbours of {list(cur)} {co:.4f}")
                evaluate_many(pool, nb, co)
                c = [(cache[n]["exact"], n) for n in nb if cache[n]["exact"] is not None and cache[n]["exact"] < co - 1e-7]
                if not c:
                    break
                co, cur = min(c); fin2, co = descend(pool, cur, log); cur = fin2
                log(f"  polish improved -> {list(cur)} {co:.4f}")
            best[K] = (cur, co)
            n_exact = sum(1 for k, v in cache.items() if len(k) == K and v["exact"] is not None)
            # verify the final plan on the full grid
            rec, x_sel, _ = pool.apply(bs.solve_subset, ((cur, None, 1e9),))
            assert rec["outcome"] == "solved" and rec["verified"], rec
            np.savez(OUT / f"best_K{K}_search.npz", x=x_sel, angles=np.array(sorted(cur)))
            summary[K] = dict(angles=list(cur), objective=rec["objective"], verified=rec["verified"],
                              max_rel_violation=rec["max_rel_violation"], max_body_dose=rec["max_body_dose"],
                              n_subsets_cached=sum(1 for k in cache if len(k) == K), n_exact_solved=n_exact,
                              runs=runs, time_s=time.time() - t0)
            log(f"== K={K} best {list(cur)} {rec['objective']:.4f} verified={rec['verified']} "
                f"viol={rec['max_rel_violation']:.2e} ({(time.time()-t0)/60:.0f} min)")
            (OUT / f"search_K{'_'.join(map(str, Ks))}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
