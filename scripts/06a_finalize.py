"""Pick the best cached plan per K (from search_cache.jsonl), re-solve and verify on the full grid,
save best_K{K}_search.npz, and write final_table.json. Usage: 06a_finalize.py K [K ...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, bao_search as bs
OUT = config.RESULTS_DIR / "improve" / "bao18"
LB = 32.67
ents = {}
for l in (OUT / "search_cache.jsonl").read_text().splitlines():
    r = json.loads(l); ents[tuple(r["angles"])] = r
bs.init_worker(bs.load_pool())
table = {}
tf = OUT / "final_table.json"
if tf.exists():
    table = json.loads(tf.read_text())
for K in [int(k) for k in sys.argv[1:]]:
    ex = sorted((v["exact"], k) for k, v in ents.items() if len(k) == K and v["exact"] is not None and np.isfinite(v["exact"]))
    top = [dict(angles=list(k), objective=o) for o, k in ex[:5]]
    o, k = ex[0]
    rec, x, _ = bs.solve_subset((k, None, 1e9))
    assert rec["outcome"] == "solved" and rec["verified"], rec
    np.savez(OUT / f"best_K{K}_search.npz", x=x, angles=np.array(sorted(k)))
    table[str(K)] = dict(angles=list(k), objective=rec["objective"], verified=True,
                         max_rel_violation=rec["max_rel_violation"], max_body_dose=rec["max_body_dose"],
                         gap_to_32_67=(rec["objective"] - LB) / rec["objective"], top5=top,
                         n_exact_in_cache=len(ex))
    print(K, table[str(K)], flush=True)
tf.write_text(json.dumps(table, indent=1))
