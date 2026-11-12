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
