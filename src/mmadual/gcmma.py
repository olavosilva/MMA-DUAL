"""
GCMMA (Svanberg 2002): public `gcmmasub` and the routines of the outer/inner iterations, `asymp`,
`concheck` and `raaupdate`, with the calling signatures of GCMMA-MMA-Python (mmapy), so that a complete
GCMMA loop runs without that package:

    for outer iteration k:
        low, upp, raa0, raa = asymp(k, n, x, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0dx, dfdx)
        xnew, ..., f0app, fapp = gcmmasub(m, n, k, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
        evaluate f0new, fnew at xnew
        while not concheck(m, epsimin, f0app, f0new, fapp, fnew) and inner < innermax:    # inner iterations
            raa0, raa = raaupdate(xnew, x, xmin, xmax, low, upp, f0new, fnew, f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
            xnew, ..., f0app, fapp = gcmmasub(...)                                          # more conservative
            evaluate f0new, fnew at xnew
        xold2, xold1, x = xold1, x, xnew

The approximations are made more conservative (raa0, raa increased) until they bound the true functions
at the new point, which guarantees a monotone decrease of the objective (Svanberg 2002). `epsimin` is the
tolerance of the conservativeness test; it is unrelated to the tolerance `tol_dual` of the dual solver.
"""
import numpy as np

from .approximations import gcmma_subproblem, approximation_values, XMAMI_MIN
from .dual_solver import solve_dual, recover_multipliers

RAACOF_MIN = 1e-12


def _col(v):
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _flat(v):
    return np.asarray(v, dtype=float).ravel()


def gcmmasub(m, n, iter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d,
             albefa=0.1, move=None, tol_dual=1e-10, max_newton=100, return_info=False, **kwargs):
    """GCMMA subproblem at xval with the current asymptotes and conservativeness parameters, solved by the dual
       method. Same positional arguments and the same returned tuple as mmapy.gcmmasub,
           xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp
       where f0app and fapp are the values of the approximations at xmma, used by concheck and raaupdate.
       `move` (optional) adds the move limit |x - xval| <= move (xmax - xmin), absent in mmapy. `iter` and
       `epsimin` are accepted for compatibility (they do not enter the subproblem)."""
    sub = gcmma_subproblem(xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d, albefa, move)
    x, y, z, lam, info = solve_dual(sub, tol_dual, max_newton)
    xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
    f0s, gs = approximation_values(x, sub['low'], sub['upp'], sub['p0'], sub['q0'], sub['P'], sub['Q'])
    f0app, fapp = sub['r0'] + f0s, sub['r'] + gs
    out = (_col(x), _col(y), _col(z), _col(lam), _col(xsi), _col(eta), _col(mu), _col(zet), _col(s),
           _col(f0app), _col(fapp))
    return out + (info,) if return_info else out


def asymp(outeriter, n, xval, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0dx, dfdx,
          asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10, **kwargs):
    """start of an outer GCMMA iteration: initial conservativeness parameters and asymptotes.
       raa0 and raa_i are set to 0.1/n times the weighted l1 norm of the gradients, |df/dx| . (xmax - xmin),
       with the floors raa0eps and raaeps; the asymptotes follow the same rule as in the MMA (iterations 1 and
       2 at asyinit (xmax - xmin) from xval, then expanded or contracted according to the last two moves).
       Returns low, upp (column vectors), raa0 (float), raa (column vector)"""
    xval, xold1, xold2, xmin, xmax, low, upp, df0dx = (_flat(v) for v in (xval, xold1, xold2, xmin, xmax, low, upp, df0dx))
    dfdx = np.asarray(dfdx, dtype=float).reshape(-1, n)
    xmami = np.maximum(xmax - xmin, XMAMI_MIN)
    raa0_new = max(float(raa0eps), (0.1/n)*float(np.abs(df0dx) @ xmami))
    raa_new = np.maximum(_flat(raaeps), (0.1/n)*(np.abs(dfdx) @ xmami))
    if outeriter <= 2:
        low_new, upp_new = xval - asyinit*xmami, xval + asyinit*xmami
    else:
        trend = (xval - xold1)*(xold1 - xold2)
        factor = np.where(trend > 0, asyincr, np.where(trend < 0, asydecr, 1.0))
        low_new = np.clip(xval - factor*(xold1 - low), xval - asymax*xmami, xval - asymin*xmami)
        upp_new = np.clip(xval + factor*(upp - xold1), xval + asymin*xmami, xval + asymax*xmami)
    return _col(low_new), _col(upp_new), raa0_new, _col(raa_new)


def concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew, **kwargs):
    """conservativeness test of the inner GCMMA iteration: 1 if the approximations, plus epsimin, bound the true
       objective and constraints at the new point, 0 otherwise"""
    ok0 = float(np.asarray(f0app).ravel()[0]) + epsimin >= float(np.asarray(f0valnew).ravel()[0])
    ok = np.all(_flat(fapp) + epsimin >= _flat(fvalnew))
    return int(ok0 and ok)


def raaupdate(xmma, xval, xmin, xmax, low, upp, f0valnew, fvalnew, f0app, fapp, raa0, raa, raa0eps, raaeps,
              epsimin, **kwargs):
    """update of raa0 and raa in the inner GCMMA iteration (Svanberg 2002, eqs. for the conservative update):
       for each function whose approximation underestimates its value at xmma by more than epsimin/2, the
       parameter is increased to 1.1 (raa + (f_new - f_app)/raacof), capped at 10 raa, where
       raacof = sum_j (xmma_j - xval_j)^2 (U_j - L_j) / ((U_j - xmma_j)(xmma_j - L_j)(xmax_j - xmin_j))
       is the curvature weight of the step. Returns raa0 (float) and raa (column vector)"""
    xmma, xval, xmin, xmax, low, upp = (_flat(v) for v in (xmma, xval, xmin, xmax, low, upp))
    xmami = np.maximum(xmax - xmin, XMAMI_MIN)
    dx = xmma - xval
    raacof = max(float(np.sum(dx*dx/((upp - xmma)*(xmma - low))*(upp - low)/xmami)), RAACOF_MIN)
    f0app, f0valnew = float(np.asarray(f0app).ravel()[0]), float(np.asarray(f0valnew).ravel()[0])
    raa0_new = float(np.asarray(raa0).ravel()[0]) if np.ndim(raa0) else float(raa0)
    if f0valnew > f0app + 0.5*epsimin:
        raa0_new = min(1.1*(raa0_new + (f0valnew - f0app)/raacof), 10.0*raa0_new)
    fapp, fvalnew, raa_new = _flat(fapp), _flat(fvalnew), _flat(raa).copy()
    grow = fvalnew > fapp + 0.5*epsimin
    raa_new[grow] = np.minimum(1.1*(raa_new[grow] + (fvalnew[grow] - fapp[grow])/raacof), 10.0*raa_new[grow])
    return raa0_new, _col(raa_new)


def kktcheck(m, n, x, y, z, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d, **kwargs):
    """residuals of the KKT conditions of the ORIGINAL problem at (x, y, z) with the multipliers returned by
       mmasub/gcmmasub (same interface as mmapy.kktcheck): the vector of residuals, its Euclidean norm and
       its largest absolute entry. Used as the stopping criterion of the outer iterations."""
    x, y, lam, xsi, eta, mu, s, xmin, xmax, df0dx, fval, a, c, d = (_flat(v) for v in (x, y, lam, xsi, eta, mu, s, xmin, xmax, df0dx, fval, a, c, d))
    dfdx = np.asarray(dfdx, dtype=float).reshape(m, n)
    z, zet = float(np.asarray(z).ravel()[0]), float(np.asarray(zet).ravel()[0])
    res = np.concatenate((
        df0dx + dfdx.T @ lam - xsi + eta,                                   # stationarity in x
        c + d*y - mu - lam,                                                 # stationarity in y
        [a0 - zet - a @ lam],                                               # stationarity in z
        fval - a*z - y + s,                                                 # primal feasibility
        xsi*(x - xmin), eta*(xmax - x), mu*y, [zet*z], lam*s))              # complementarity
    return _col(res), float(np.linalg.norm(res)), float(np.max(np.abs(res)))
