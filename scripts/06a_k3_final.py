"""Solve the K=3 minimiser exactly (no cutoff) and verify it; writes K3_final.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, bao_search as bs
OUT = config.RESULTS_DIR / "improve" / "bao18"
rows = [json.loads(l) for l in (OUT / "K3.jsonl").read_text().splitlines()]
m = min((r for r in rows if r["outcome"] == "pruned"), key=lambda r: r["lb"])
print("lowest lower bound:", m["angles"], m["lb"])
bs.init_worker(bs.load_pool())
rec, x, _ = bs.solve_subset((tuple(m["angles"]), None, 1e9))
print(rec)
assert rec["outcome"] == "solved" and rec["verified"]
np.savez(OUT / "best_K3_enum.npz", x=x, angles=np.array(sorted(m["angles"])))
json.dump(rec, open(OUT / "K3_final.json", "w"), indent=1)
