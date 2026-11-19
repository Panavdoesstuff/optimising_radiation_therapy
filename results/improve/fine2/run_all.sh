cd /Users/panavrg/optimising_radiation_therapy
P=.venv/bin/python
$P scripts/07b_search.py k6 6 1.7 > results/improve/fine2/k6.log 2>&1
$P scripts/07b_search.py desc 4 0.9 > results/improve/fine2/k4.log 2>&1
$P scripts/07b_search.py desc 3 0.5 > results/improve/fine2/k3.log 2>&1
$P scripts/07b_search.py k5r 5 0.5 > results/improve/fine2/k5.log 2>&1
echo ALLDONE > results/improve/fine2/alldone
