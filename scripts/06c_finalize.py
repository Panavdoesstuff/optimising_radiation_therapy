"""Verify the best plan per K found in results/improve/fine/evals.jsonl on the full grid; save npz + json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, fine_search

R = config.RESULTS_DIR; OUT = R / "improve" / "fine"
rows = [json.loads(l) for l in (OUT / "evals.jsonl").read_text().splitlines()]
Ks = [int(k) for k in sys.argv[1:]] or [5, 4, 6, 3]
pool = np.union1d(np.load(R / "bao_body_pool.npy"), np.load(R / "enum" / "pool_added.npy"))
fine_search.init_worker(pool)
res = {}
for K in Ks:
    cand = {tuple(r["angles"]): r["objective"] for r in rows if len(r["angles"]) == K and r.get("objective") is not None}
    if not cand: continue
    a, f = min(cand.items(), key=lambda kv: kv[1])
    o, x = fine_search.evaluate(list(a), verify_full=True)
    P = o.pop("P"); d = P.dose(x); sets = fine_search._W["sets"]
    Rm = sets["Rectum"]
    o["rectum_mean"] = float(d[Rm].mean()); o["rectum_V50"] = float((d[Rm] > 50).mean())
    o["bladder_mean"] = float(d[sets["Bladder"]].mean()); o["body_mean"] = float(d[sets["BODY"]].mean())
    o["n_evals_K"] = len(cand)
    np.savez(OUT / f"best_K{K}.npz", x=x, angles=np.array(a))
    res[K] = o
    print(K, a, o["objective_verified"], o["verified"], o["max_rel_violation"], o["rectum_mean"], o["rectum_V50"],
          o["bladder_mean"], o["body_mean"], flush=True)
json.dump(res, open(OUT / ("final_" + "_".join(map(str, Ks)) + ".json"), "w"), indent=1)
