cd /Users/panavrg/optimising_radiation_therapy
while [ ! -f results/improve/fine2/alldone ]; do sleep 20; done
cp results/improve/fine2/search_desc_K6.json results/improve/fine2/search_desc_K6.json 2>/dev/null
.venv/bin/python scripts/07b_search.py desc 6 0.7 > results/improve/fine2/k6b.log 2>&1
echo done > results/improve/fine2/k6bdone
