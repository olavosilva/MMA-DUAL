"""
Construction of the separable convex subproblems of the MMA and of the GCMMA.

Both methods replace the problem

    min f0(x)   s.t.  fi(x) <= 0 (i = 1..m),   xmin <= x <= xmax

at the current point xval by the approximation (Svanberg 1987, 2002)

    min  sum_j [p0j/(Uj - xj) + q0j/(xj - Lj)] + a0 z + sum_i (ci yi + di yi^2/2)
    s.t. sum_j [Pij/(Uj - xj) + Qij/(xj - Lj)] - ai z - yi <= bi
         alfa_j <= xj <= beta_j,  yi >= 0,  z >= 0

with moving asymptotes L < alfa <= beta < U. The MMA coefficients are those of Svanberg's 2007 MATLAB
implementation (with the regularization 0.001 (p + q) + raa0/(xmax - xmin)); the GCMMA coefficients are
those of Svanberg (2002), with the conservativeness parameters raa0, raa updated between inner
iterations. Both constructions are written here from the published formulas and reproduce, to rounding,
the subproblems built by GCMMA-MMA-Python (mmapy), which is the Python translation of the MATLAB codes;
see tests/test_approximations.py.

A subproblem is represented by a dict with 1-D arrays (design variables of length n, constraints of
length m) and the (m, n) matrices P and Q:

    low, upp, alfa, beta : asymptotes and bounds of the variables          (n,)
    p0, q0               : coefficients of the objective                   (n,)
    P, Q                 : coefficients of the constraints                 (m, n)
    b                    : right-hand sides                                (m,)
    a0, a, c, d          : data of the artificial variables z and y        scalar, (m,), (m,), (m,)
    r0, r                : constant terms of the GCMMA approximations (only for the GCMMA; r0 is the
                           constant of the objective, r_i of the constraints; b = -r)

The subproblem dict is independent of the procedure that built it; the dual solver (dual_solver.py)
and the diagnostics (diagnostics.py) take it as their only description of the problem.
"""
import numpy as np

XMAMI_MIN = 1e-5        # lower bound of xmax - xmin used in the regularization terms (Svanberg's MATLAB codes)
CONVEX_REG = 1e-3       # fraction of |df/dx| added to both p and q so that every term is strictly convex


def _scalar(v):
    "scalar given as float, 0-d array or (1, 1) array -> float"
    return float(np.asarray(v, dtype=float).ravel()[0])


def _as_1d(v, n):
    "column vector, row vector or scalar -> 1-D array of length n"
    v = np.asarray(v, dtype=float)
    return np.broadcast_to(v.ravel(), (n,)).copy() if v.size == 1 else v.ravel()


def moving_asymptotes(iter, xval, xold1, xold2, xmin, xmax, low, upp,
                      asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10.0):
    """asymptotes L (low) and U (upp) of the MMA at outer iteration `iter` (1-based).
       Iterations 1 and 2: xval -/+ asyinit (xmax - xmin). Afterwards the distance to the asymptote is
       multiplied by asyincr when the variable moved in the same direction in the last two iterations and by
       asydecr when it oscillated, and kept within asymin (xmax - xmin) and asymax (xmax - xmin)"""
    xmami = xmax - xmin
    if iter <= 2:
        return xval - asyinit*xmami, xval + asyinit*xmami
    trend = (xval - xold1)*(xold1 - xold2)
    factor = np.where(trend > 0, asyincr, np.where(trend < 0, asydecr, 1.0))
    low_new = xval - factor*(xold1 - low)
    upp_new = xval + factor*(upp - xold1)
    low_new = np.clip(low_new, xval - asymax*xmami, xval - asymin*xmami)
    upp_new = np.clip(upp_new, xval + asymin*xmami, xval + asymax*xmami)
    return low_new, upp_new


def variable_bounds(xval, xmin, xmax, low, upp, albefa=0.1, move=None):
    """bounds alfa <= x <= beta of the subproblem: at least albefa of the distance to the asymptotes, within
       the move limit move (xmax - xmin) when given, and within [xmin, xmax]"""
    alfa = low + albefa*(xval - low)
    beta = upp - albefa*(upp - xval)
    if move is not None:
        alfa = np.maximum(alfa, xval - move*(xmax - xmin))
        beta = np.minimum(beta, xval + move*(xmax - xmin))
    return np.maximum(alfa, xmin), np.minimum(beta, xmax)


def separable_coefficients(xval, xmin, xmax, low, upp, df0dx, dfdx, raa0, raa):
    """coefficients p0, q0 (n,) and P, Q (m, n) of the separable approximations of f0 and fi at xval:
       the positive part of the gradient goes to the U side, the negative part to the L side, both
       regularized by CONVEX_REG |df/dx| and by raa0 (objective) or raa_i (constraints) divided by the
       variable range, and scaled by (U - x)^2 and (x - L)^2 so that the derivative of the approximation
       at xval equals the gradient plus the regularization"""
    xmami_inv = 1.0/np.maximum(xmax - xmin, XMAMI_MIN)
    ux2, xl2 = (upp - xval)**2, (xval - low)**2
    pos0, neg0 = np.maximum(df0dx, 0.0), np.maximum(-df0dx, 0.0)
    reg0 = CONVEX_REG*(pos0 + neg0) + raa0*xmami_inv
    p0, q0 = (pos0 + reg0)*ux2, (neg0 + reg0)*xl2
    pos, neg = np.maximum(dfdx, 0.0), np.maximum(-dfdx, 0.0)
    reg = CONVEX_REG*(pos + neg) + np.outer(np.asarray(raa, dtype=float).ravel(), xmami_inv)
    P, Q = (pos + reg)*ux2[None, :], (neg + reg)*xl2[None, :]
    return p0, q0, P, Q


def approximation_values(xval, low, upp, p0, q0, P, Q):
    "values at xval of the separable terms of the objective and of the constraints (without constants)"
    uxinv, xlinv = 1.0/(upp - xval), 1.0/(xval - low)
    return p0 @ uxinv + q0 @ xlinv, P @ uxinv + Q @ xlinv


def mma_subproblem(iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a, c, d,
                   move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10.0,
                   raa0=1e-5, albefa=0.1):
    """MMA subproblem at outer iteration `iter` (Svanberg 1987, with the 2007 MATLAB parameters).
       All vector arguments are accepted as (n,), (n, 1) or (1, n) and dfdx as (m, n). Returns the
       subproblem dict described in the module docstring; the asymptotes low, upp returned are the updated
       ones and must be passed back at the next iteration"""
    n = np.asarray(xval).size
    m = np.asarray(fval).size
    xval, xmin, xmax = _as_1d(xval, n), _as_1d(xmin, n), _as_1d(xmax, n)
    xold1, xold2, low, upp = _as_1d(xold1, n), _as_1d(xold2, n), _as_1d(low, n), _as_1d(upp, n)
    df0dx, fval = _as_1d(df0dx, n), _as_1d(fval, m)
    dfdx = np.asarray(dfdx, dtype=float).reshape(m, n)
    low, upp = moving_asymptotes(iter, xval, xold1, xold2, xmin, xmax, low, upp, asyinit, asydecr, asyincr, asymin, asymax)
    alfa, beta = variable_bounds(xval, xmin, xmax, low, upp, albefa, move)
    p0, q0, P, Q = separable_coefficients(xval, xmin, xmax, low, upp, df0dx, dfdx, raa0, np.full(m, raa0))
    _, gval = approximation_values(xval, low, upp, p0, q0, P, Q)
    b = gval - fval                                                         # g_i(xval) - b_i = f_i(xval)
    return dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, P=P, Q=Q, b=b,
                a0=_scalar(a0), a=_as_1d(a, m), c=_as_1d(c, m), d=_as_1d(d, m))


def gcmma_subproblem(xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d,
                     albefa=0.1, move=None):
    """GCMMA subproblem (Svanberg 2002) for the current asymptotes low, upp and conservativeness parameters
       raa0 (objective) and raa (constraints), which are updated by `asymp` and `raaupdate` (gcmma.py).
       Returns the subproblem dict with the constants r0 and r of the approximations, so that
       f0app = r0 + p0/(U - x) + q0/(x - L) and fapp = r + P/(U - x) + Q/(x - L) at any x"""
    n = np.asarray(xval).size
    m = np.asarray(fval).size
    xval, xmin, xmax, low, upp = (_as_1d(v, n) for v in (xval, xmin, xmax, low, upp))
    df0dx, fval, raa = _as_1d(df0dx, n), _as_1d(fval, m), _as_1d(raa, m)
    dfdx = np.asarray(dfdx, dtype=float).reshape(m, n)
    alfa, beta = variable_bounds(xval, xmin, xmax, low, upp, albefa, move)
    p0, q0, P, Q = separable_coefficients(xval, xmin, xmax, low, upp, df0dx, dfdx, _scalar(raa0), raa)
    f0s, gs = approximation_values(xval, low, upp, p0, q0, P, Q)
    r0 = _scalar(f0val) - f0s                                                 # approximations interpolate at xval
    r = fval - gs
    return dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, P=P, Q=Q, b=-r, r0=r0, r=r,
                a0=_scalar(a0), a=_as_1d(a, m), c=_as_1d(c, m), d=_as_1d(d, m))
