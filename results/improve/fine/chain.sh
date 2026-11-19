#!/bin/bash
cd /Users/panavrg/optimising_radiation_therapy
while [ ! -f results/improve/fine/search_K5.json ]; do sleep 20; done
.venv/bin/python scripts/06c_fine_search.py 4 0.55 > results/improve/fine/K4.log 2>&1
.venv/bin/python scripts/06c_fine_search.py 6 0.55 > results/improve/fine/K6.log 2>&1
.venv/bin/python scripts/06c_fine_search.py 3 0.35 > results/improve/fine/K3.log 2>&1
