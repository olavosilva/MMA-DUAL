"""
MMA outer iteration: public `mmasub`, with the calling signature and the returned tuple of
GCMMA-MMA-Python (mmapy), built on approximations.mma_subproblem and dual_solver.solve_dual.
"""
import numpy as np

from .approximations import mma_subproblem
from .dual_solver import solve_dual, recover_multipliers


def _col(v):
    "1-D array -> column vector, as returned by mmapy"
    return np.asarray(v, dtype=float).reshape(-1, 1)


def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
           move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10, raa0=0.00001, albefa=0.1,
           tol_dual=1e-10, max_newton=100, return_info=False, **kwargs):
    """one MMA iteration: builds the subproblem at xval and solves it by the dual method.
       Same positional arguments and the same returned tuple as mmapy.mmasub,
           xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
       (column vectors; z and zet as (1, 1) arrays, as in mmapy). `tol_dual` and `max_newton` control the dual iteration; with return_info=True the
       info dict of solve_dual (convergence, reason, residual, steps) is appended to the tuple.
       f0val is not needed by the MMA subproblem and is accepted for compatibility."""
    sub = mma_subproblem(iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a, c, d,
                         move, asyinit, asydecr, asyincr, asymin, asymax, raa0, albefa)
    x, y, z, lam, info = solve_dual(sub, tol_dual, max_newton)
    xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
    out = (_col(x), _col(y), _col(z), _col(lam), _col(xsi), _col(eta), _col(mu), _col(zet), _col(s),
           _col(sub['low']), _col(sub['upp']))
    return out + (info,) if return_info else out
