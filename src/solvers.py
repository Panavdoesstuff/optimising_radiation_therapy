"""MIP backend behind one interface. Default: scipy.optimize.milp (HiGHS).
If highspy is installed it is used instead, because it accepts a start solution."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    import highspy
    HAVE_HIGHSPY = True
except ImportError:
    HAVE_HIGHSPY = False


def solve_mip(c, A, hi, lb, ub, integrality, time_limit=600.0, mip_rel_gap=1e-4, start=None,
              disp=False, backend="auto"):
    """min c@v s.t. A v <= hi, lb <= v <= ub, v_i integer where integrality==1.
    Returns dict(status, message, success, x, objective, best_bound, mip_gap, node_count, solve_time, backend)."""
    use_hs = HAVE_HIGHSPY if backend == "auto" else backend == "highspy"
    t0 = time.time()
    if not use_hs:
        r = milp(c, integrality=integrality, bounds=Bounds(lb, ub),
                 constraints=LinearConstraint(A, -np.inf, hi),
                 options=dict(time_limit=time_limit, mip_rel_gap=mip_rel_gap, disp=disp))
        out = dict(backend="scipy.milp", status=int(r.status), message=r.message, success=r.x is not None,
                   x=r.x, objective=None if r.x is None else float(r.fun),
                   best_bound=getattr(r, "mip_dual_bound", None), mip_gap=getattr(r, "mip_gap", None),
                   node_count=getattr(r, "mip_node_count", None), start_used=False)
    else:
        A = A.tocsc()
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(disp))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
        lp = highspy.HighsLp()
        lp.num_col_ = len(c); lp.num_row_ = A.shape[0]
        lp.col_cost_ = np.asarray(c, float)
        lp.col_lower_ = np.asarray(lb, float); lp.col_upper_ = np.asarray(ub, float)
        lp.row_lower_ = np.full(A.shape[0], -highspy.kHighsInf); lp.row_upper_ = np.asarray(hi, float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in integrality]
        h.passModel(lp)
        if start is not None:
            s = highspy.HighsSolution(); s.col_value = list(np.asarray(start, float))
            h.setSolution(s)
        h.run()
        info = h.getInfo()
        has_x = info.primal_solution_status == 2
        x = np.array(h.getSolution().col_value) if has_x else None
        out = dict(backend="highspy", status=int(h.getModelStatus()),
                   message=h.modelStatusToString(h.getModelStatus()), success=has_x, x=x,
                   objective=float(info.objective_function_value) if has_x else None,
                   best_bound=float(info.mip_dual_bound), mip_gap=float(info.mip_gap),
                   node_count=int(info.mip_node_count), start_used=start is not None)
    out["solve_time"] = time.time() - t0
    return out
