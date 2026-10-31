"""
mmadual: MMA and GCMMA subproblems built from the published formulations (Svanberg 1987, 2002) and solved
by the dual method, for optimization problems with few constraints (m << n).

Public interface (compatible with GCMMA-MMA-Python, mmapy):
    mmasub, gcmmasub, asymp, concheck, raaupdate, kktcheck
Package interface:
    mma_subproblem, gcmma_subproblem      construction of the subproblem dict      (approximations)
    solve_dual, recover_multipliers       dual solver                              (dual_solver)
    diagnostics                           KKT residuals, duality gap, objectives   (diagnostics)
    subsolv_dual                          0.1.x calling convention of the dual solver (compatibility)
Subproblems with a != 0 (minimax form, a_i >= 0) are supported since 0.2.0: see dual_solver.py.

License: GPL-3.0-or-later. The dual solver is original to this package; the construction of the
subproblems follows Svanberg's published formulations and reproduces the subproblems of his MATLAB codes
and of mmapy, from which the 0.1.x versions of this package were derived (see README).
"""
import numpy as np

from .approximations import mma_subproblem, gcmma_subproblem
from .dual_solver import solve_dual, recover_multipliers
from .mma import mmasub
from .gcmma import gcmmasub, asymp, concheck, raaupdate, kktcheck
from . import diagnostics

__all__ = ['mmasub', 'gcmmasub', 'asymp', 'concheck', 'raaupdate', 'kktcheck',
           'mma_subproblem', 'gcmma_subproblem', 'solve_dual', 'recover_multipliers', 'subsolv_dual', 'diagnostics']
__version__ = '0.2.1'


def subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol=1e-10, max_newton=100):
    """dual solver with the calling convention of versions 0.1.x: coefficient arrays in, column vectors
       x, y, z, lam out. New code should build the subproblem dict and call solve_dual"""
    f = lambda v: np.asarray(v, dtype=float).ravel()
    sub = dict(low=f(low), upp=f(upp), alfa=f(alfa), beta=f(beta), p0=f(p0), q0=f(q0),
               P=np.asarray(P, dtype=float).reshape(m, n), Q=np.asarray(Q, dtype=float).reshape(m, n),
               b=f(b), a0=float(a0), a=f(a), c=f(c), d=f(d))
    x, y, z, lam, _ = solve_dual(sub, tol, max_newton)
    return x.reshape(-1, 1), y.reshape(-1, 1), np.array([[z]]), lam.reshape(-1, 1)
