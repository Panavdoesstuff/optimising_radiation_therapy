"""Paths, beam angle sets, prescriptions, weights. Single source of truth."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Data is read-only and lives outside the repo. Override with CORT_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("CORT_DATA_ROOT", Path.home() / "Downloads"))
CASE_DIRS = {
    "tg119": DATA_ROOT / "TG119",
    "prostate": DATA_ROOT / "PROSTATE",
}
CODE_SNIPPETS_DIR = DATA_ROOT / "codeSnippetsCortDataset"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

# Structure name -> file stem (file is f"{stem}_VOILIST.mat")
STRUCTURES = {
    "tg119": {"Core": "Core", "OuterTarget": "OuterTarget", "BODY": "BODY"},
    "prostate": {"PTV_68": "PTV_68", "PTV_56": "PTV_56", "Rectum": "Rectum",
                 "Bladder": "Bladder", "BODY": "BODY"},
}

# Beam angle sets (couch 0)
TG119_ANGLES = [0, 72, 144, 216, 288]
PROSTATE_5 = [0, 72, 144, 216, 288]
PROSTATE_ALL = list(range(0, 360, 2))  # 180 beams

# Expected facts (checked in 00_inspect_data.py)
EXPECTED = {
    "prostate": dict(n_beams=180, n_beamlets=25404, grid=(184, 184, 90)),
    "tg119": dict(n_beams=5, n_beamlets=418, grid=(167, 167, 129)),
}

# Easy/hard problem prescriptions (Gy)
RX_T68 = 68.0
RX_T56 = 56.0
D_CAP_FACTOR = 1.1          # d_cap = 1.1 * 68 = 74.8
WEIGHTS = {"Rectum": 1.0, "Bladder": 0.6, "BODY": 0.6}
X_MAX_PAPER = 50.0          # paper's bound, in normalised-dose units
SEED = 0

# Easy problem: slack penalty used only on rungs 3/4 of the feasibility ladder
RHO_SLACK = 100.0           # cost per Gy of mean target shortfall (only if the ladder needs it)
D_CAP_LADDER = 1.15         # rung 2
# Paper's bound (50, normalised dose) scaled by 68 is 3400; that BINDS (78 beamlets at bound, objective +4.7%).
# The unbounded optimum peaks at ~7605, so 4x the scaled paper value never binds. Deviation noted in SUMMARY.md.
X_MAX_EASY = 4 * X_MAX_PAPER * RX_T68   # 13600

# Beam angle selection (hard problem, promoted from stretch).
# 18-degree spacing (20 candidates) is used instead of the brief's 20/10 degrees because it is
# the only nearby spacing that contains the easy plan's beams (0, 72, 144, 216, 288), so the easy
# plan is a feasible point of the K=5 MILP.
BAO_CANDIDATES = list(range(0, 360, 18))
BAO_K_VALUES = [3, 4, 5, 6, 7, 8, 9]
BAO_TIME_LIMIT = 600.0

# Dose-volume constraint kept as a secondary, bracketed hard result (see SUMMARY.md):
# the exact MILP stalled at the root for every tighter limit tried; rectum <=17% above 50 Gy
# is the tightest limit for which a feasible plan was found (fix-and-resolve).
DVC_FINAL = [("Rectum", 50.0, 0.17)]
