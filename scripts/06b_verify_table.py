"""Independent re-verification of every saved plan in results/improve/dvc/plan_*.npz on the full grid.
Writes results/improve/dvc/verified_table.json and prints a markdown table."""
import glob, json, re, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config, problem, easy_lp, verify
warnings.simplefilter("ignore")
OUT = config.RESULTS_DIR / "improve" / "dvc"
spec = easy_lp.Spec()
easy = 52.53441633869798
rows = []
for f in sorted(glob.glob(str(OUT / "plan_*.npz"))):
    m = re.match(r"plan_(.+)_a(\d+)\.npz", Path(f).name); tag, a = m.group(1), int(m.group(2)) / 1000
    beams = config.PROSTATE_5 if not tag.startswith("b5") else [108, 144, 180, 216, 252]
    P = problem.build("prostate", beams) if "P" not in globals() or P.angles != beams else P
    sets = P.sets("disjoint"); R = sets["Rectum"]
    x = np.load(f)["x"]; v = verify.verify(P, sets, spec, x, dvcs=[("Rectum", 50.0, a)])
    d = P.dose(x); n = int((d[R] > 50 * (1 + 1e-6)).sum())
    rows.append(dict(file=Path(f).name, alpha=a, objective=v["objective_recomputed"], V50_n=n, V50=n / len(R),
                     ok=bool(v["all_ok"]), checks={k: c["ok"] for k, c in v["checks"].items()}))
    print(f"| {a:.1%} | {tag} | {v['objective_recomputed']:.4f} | {n} ({n/len(R):.2%}) | {'PASS' if v['all_ok'] else 'FAIL'} |", flush=True)
(OUT / "verified_table.json").write_text(json.dumps(rows, indent=2))
