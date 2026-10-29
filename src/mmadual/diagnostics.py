"""
Independent checks of a subproblem solution, written from the equations of the subproblem (see
approximations.py) and not from the internal criteria of any solver, so that solutions of the dual solver
and of other solvers (for instance the interior-point routine of mmapy) can be assessed alike.

    subproblem_objective(sub, x, y, z)     objective of the subproblem at (x, y, z)
    exact_penalty_objective(sub, x)        objective with the artificial variables chosen optimally for x
                                           (y_i = max(0, g_i - b_i) for d_i = 0 and a = 0): an upper bound
                                           of the optimal value for any x in the box
    dual_value(sub, lam)                   W(lam): a lower bound of the optimal value for any lam in the dual
                                           domain (checked)
    duality_gap(sub, x, lam)               exact_penalty_objective(x) - W(lam) >= 0, absolute and relative:
                                           certifies x, with the artificial variables re-optimized for it
    duality_gap_returned(sub, x, y, z, lam) F(x, y, z) - W(lam) with the artificial variables as returned by the
                                           solver: certifies the whole returned primal point
    kkt_residuals(sub, x, y, z, lam, ...)  stationarity, feasibility (primal and dual) and complementarity
                                           residuals of the subproblem
    constraint_values(sub, x)              g_i(x) - b_i (residuals of the approximate constraints)
"""
import numpy as np

from .dual_solver import dual_function


def constraint_values(sub, x):
    "g_i(x) - b_i for the approximate constraints"
    return sub['P'] @ (1.0/(sub['upp'] - x)) + sub['Q'] @ (1.0/(x - sub['low'])) - sub['b']


def subproblem_objective(sub, x, y, z=0.0):
    "objective of the subproblem at (x, y, z), without the GCMMA constant r0"
    return float(np.sum(sub['p0']/(sub['upp'] - x) + sub['q0']/(x - sub['low'])) + sub['a0']*z
                 + np.sum(sub['c']*y + 0.5*sub['d']*y*y))


def optimal_artificial(sub, x):
    """artificial variables (y, z) that minimize the objective for the given x subject to the constraints:
       y_i = max(0, g_i - b_i - a_i z) (the cost c y + d y^2/2 is increasing in y), and z >= 0 minimizing the
       resulting piecewise-convex function a0 z + sum_i [c_i y_i + d_i y_i^2/2]; with a = 0 this gives z = 0, with
       a != 0 the minimum is at z = 0 or at a breakpoint (g_i - b_i)/a_i when d = 0, and is found by a bracketed
       search otherwise"""
    g = constraint_values(sub, x)
    a, a0, c, d = sub['a'], sub['a0'], sub['c'], sub['d']

    def cost(z):
        y = np.maximum(g - a*z, 0.0)
        return a0*z + np.sum(c*y + 0.5*d*y*y), y

    if not np.any(a > 0):
        return np.maximum(g, 0.0), 0.0
    cand = [0.0] + [gi/ai for gi, ai in zip(g, a) if ai > 0 and gi > 0]
    if np.any(d > 0):                                                       # smooth pieces: refine by golden section on [0, max cand]
        zmax = max(cand)
        lo, hi = 0.0, zmax
        phi = (np.sqrt(5) - 1)/2
        for _ in range(200):
            z1, z2 = hi - phi*(hi - lo), lo + phi*(hi - lo)
            if cost(z1)[0] < cost(z2)[0]:
                hi = z2
            else:
                lo = z1
        cand.append(0.5*(lo + hi))
    z = min(cand, key=lambda zz: cost(zz)[0])
    return cost(z)[1], float(z)


def exact_penalty_objective(sub, x):
    "f_pen(x) = objective at x with the optimal artificial variables: an upper bound of the optimal value"
    y, z = optimal_artificial(sub, x)
    return subproblem_objective(sub, x, y, z)


def dual_feasible(sub, lam, rtol=1e-12):
    "lam >= 0, lam <= c where d = 0 (and a^T lam <= a0 when a != 0)"
    c = sub['c']
    if np.any(lam < -rtol*np.maximum(1.0, c)):
        return False
    if np.any((sub['d'] == 0) & (lam > c*(1 + rtol))):
        return False
    return bool(sub['a'] @ lam <= sub['a0']*(1 + rtol) + rtol)


def dual_value(sub, lam, check=True):
    "W(lam): lower bound of the optimal value of the subproblem for any dual-feasible lam"
    lam = np.asarray(lam, dtype=float).ravel()
    if check and not dual_feasible(sub, lam):
        raise ValueError("dual_value: lam is not dual feasible (lam < 0, lam > c with d = 0, or a^T lam > a0)")
    lam = np.clip(lam, 0.0, np.where(sub['d'] > 0, np.inf, sub['c']))
    return float(dual_function(sub, lam)[0])


def duality_gap(sub, x, lam):
    """gap G = f_pen(x) - W(lam) >= 0 (weak duality), which bounds the suboptimality of x and of lam at once;
       returned as (absolute, relative) with the relative gap normalized by max(1, |f_pen(x)|). This certifies
       the design variables x (with the artificial variables chosen optimally for it), not the (y, z) a solver
       returned: for those use duality_gap_returned"""
    f = exact_penalty_objective(sub, x)
    G = f - dual_value(sub, lam)
    return G, G/max(1.0, abs(f))


def duality_gap_returned(sub, x, y, z, lam):
    """gap F(x, y, z) - W(lam) >= 0 with the objective evaluated at the artificial variables actually returned
       (y, z must be feasible with x: y_i >= g_i(x) - a_i z - b_i, else the point is infeasible and the gap is
       meaningless; the violation is returned as well). Zero only if (x, y, z) and lam are both optimal"""
    g = constraint_values(sub, x) - sub['a']*z
    viol = float(np.max(np.maximum(g - y, 0.0)))
    F = subproblem_objective(sub, x, y, z)
    G = F - dual_value(sub, lam)
    return G, G/max(1.0, abs(F)), viol


def kkt_residuals(sub, x, y, z, lam, xsi, eta, mu, zet, s):
    """residuals of the KKT conditions of the subproblem (all as 1-D arrays or floats):
       stat_x  : d/dx of the Lagrangian, p/(U - x)^2 - q/(x - L)^2 - xsi + eta
       stat_y  : c + d y - mu - lam
       stat_z  : a0 - zet - a^T lam
       feas    : g(x) - a z - y + s - b          (= 0 with the slacks s >= 0)
       viol    : max(0, g(x) - a z - y - b)      (violation of the constraints with the artificial variables)
       box     : max(0, alfa - x), max(0, x - beta)
       dual    : max(0, -lam), max(0, -xsi), ... (negative multipliers)
       comp    : xsi (x - alfa), eta (beta - x), mu y, zet z, lam s
       proj_x  : projected-gradient stationarity in x, |x - clip(x - dpsi/dx, alfa, beta)|, which does not use
                 the bound multipliers (independent check of x as a minimizer of the Lagrangian at lam)"""
    p, q = sub['p0'] + lam @ sub['P'], sub['q0'] + lam @ sub['Q']
    dpsi = p/(sub['upp'] - x)**2 - q/(x - sub['low'])**2
    g = constraint_values(sub, x) - sub['a']*z
    return dict(
        stat_x=dpsi - xsi + eta,
        stat_y=sub['c'] + sub['d']*y - mu - lam,
        stat_z=float(sub['a0'] - zet - sub['a'] @ lam),
        feas=g - y + s,
        viol=np.maximum(g - y, 0.0),
        box=(np.maximum(sub['alfa'] - x, 0.0), np.maximum(x - sub['beta'], 0.0)),
        dual=(np.maximum(-lam, 0.0), np.maximum(-xsi, 0.0), np.maximum(-eta, 0.0), np.maximum(-mu, 0.0),
              max(-zet, 0.0), np.maximum(-s, 0.0)),
        comp=(xsi*(x - sub['alfa']), eta*(sub['beta'] - x), mu*y, float(zet*z), lam*s),
        proj_x=np.abs(x - np.clip(x - dpsi, sub['alfa'], sub['beta'])),
    )


def kkt_summary(res):
    "largest absolute value of each group of residuals of kkt_residuals, as a dict of floats"
    def big(v):
        return float(max(np.max(np.abs(np.atleast_1d(t))) for t in (v if isinstance(v, tuple) else (v,))))
    return {k: big(v) for k, v in res.items()}


def check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s):
    "kkt_summary plus the duality gap and the exact-penalty objective, as one dict"
    out = kkt_summary(kkt_residuals(sub, x, y, z, lam, xsi, eta, mu, zet, s))
    out['f_pen'] = exact_penalty_objective(sub, x)
    out['gap'], out['gap_rel'] = duality_gap(sub, x, lam)
    out['F_returned'] = subproblem_objective(sub, x, y, z)
    out['gap_returned'], out['gap_returned_rel'], _ = duality_gap_returned(sub, x, y, z, lam)
    return out
