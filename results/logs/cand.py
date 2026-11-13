import sys; sys.path.insert(0,'/Users/panavrg/optimising_radiation_therapy')
import numpy as np
from src import config, problem, easy_lp, hard_dvc
P=problem.build("prostate", config.PROSTATE_5); S=P.sets("disjoint")
m=easy_lp.build_model(P, easy_lp.Spec())
x0=np.load("/Users/panavrg/optimising_radiation_therapy/results/easy_solution.npz")["x"]
d=P.dose(x0)
for t,a in [(40,0.25),(40,0.20),(30,0.30),(30,0.25),(50,0.16)]:
    print(f"t={t} alpha={a}: easy frac above = {(d[S['Rectum']]>t).mean():.3f}", flush=True)
    hard_dvc.fix_and_resolve(P,m,[hard_dvc.DVC("Rectum",float(t),a)],x0,n_iter=1)
