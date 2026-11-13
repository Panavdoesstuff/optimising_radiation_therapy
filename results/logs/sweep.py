import sys; sys.path.insert(0,'/Users/panavrg/optimising_radiation_therapy')
import numpy as np, time
from src import config, problem, easy_lp, hard_dvc
P=problem.build("prostate", config.PROSTATE_5)
m=easy_lp.build_model(P, easy_lp.Spec())
x0=np.load("/Users/panavrg/optimising_radiation_therapy/results/easy_solution.npz")["x"]
for a in [0.17,0.15,0.13,0.11]:
    dv=[hard_dvc.DVC("Rectum",50.0,a)]
    print("alpha",a)
    fr=hard_dvc.fix_and_resolve(P,m,dv,x0,n_iter=1)
    cv=hard_dvc.cvar_lp(P,m,dv)
