#!/bin/bash
cd /Users/panavrg/optimising_radiation_therapy
while [ ! -f results/improve/fine/search_K6.json ]; do sleep 20; done
.venv/bin/python scripts/06c_fine_search.py 4 0.3 > results/improve/fine/K4b.log 2>&1
.venv/bin/python scripts/06c_fine_search.py 3 0.15 > results/improve/fine/K3.log 2>&1
