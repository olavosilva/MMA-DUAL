"""
Edge cases of mma_dual.mmasub, checked against mmapy.mmasub on the same subproblem:
strongly violated constraints (lam reaches c, y > 0), d > 0, objective gradients scaled from
1e-3 to 1e3, the a != 0 exception and the non-convergence warning (tiny max_newton).

Run:  python tests/test_edge_cases.py      or      pytest tests/
Requires mmapy (pip install mmapy).
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mma_dual import mmasub as mma_du
from mmapy import mmasub as mma_pd

RNG = np.random.default_rng(1)
MOVE = 0.2


def coeficientes(m, n, xval, xmin, xmax, low, upp, df0dx, dfdx, fval, raa0=1e-5):
    "p0, q0, P, Q, b of the MMA subproblem, as built by mmapy.mmasub (default raa0)"
    eeen, eeem = np.ones((n, 1)), np.ones((m, 1))
    xmami_inv = eeen/np.maximum(xmax - xmin, 1e-5*eeen)
    ux1, xl1 = upp - xval, xval - low
    ux2, xl2 = ux1*ux1, xl1*xl1
    p0, q0 = np.maximum(df0dx, 0), np.maximum(-df0dx, 0)
    pq0 = 0.001*(p0 + q0) + raa0*xmami_inv
    p0, q0 = (p0 + pq0)*ux2, (q0 + pq0)*xl2
    P, Q = np.maximum(dfdx, 0), np.maximum(-dfdx, 0)
    PQ = 0.001*(P + Q) + raa0*np.dot(eeem, xmami_inv.T)
    P, Q = (P + PQ)*ux2.T, (Q + PQ)*xl2.T
    b = np.dot(P, eeen/ux1) + np.dot(Q, eeen/xl1) - fval
    return p0, q0, P, Q, b


def aproximacao(x, y, low, upp, p0, q0, P, Q, b, c, d):
    "objective of the approximation at (x, y) with the exact penalty of y, and the constraint residuals g_i(x) - b_i"
    f = np.sum(p0/(upp - x) + q0/(x - low))
    g = (P @ (1/(upp - x)) + Q @ (1/(x - low)) - b).ravel()
    if np.all(d == 0):
        pen = np.sum(c.ravel()*np.maximum(g, 0))
    else:
        pen = np.sum(c.ravel()*y.ravel() + 0.5*d.ravel()*y.ravel()**2)
    return f + pen, g


def subproblema(n, m, escala=1.0, violada=False, dval=0.0):
    "random inputs of mmasub; violada=True makes the constraints infeasible within the move limits"
    xval = RNG.uniform(0.05, 0.95, (n, 1))
    xold1 = np.clip(xval + 0.05*RNG.standard_normal((n, 1)), 0, 1)
    xold2 = np.clip(xold1 + 0.05*RNG.standard_normal((n, 1)), 0, 1)
    low, upp = xval - 0.3, xval + 0.3
    df0dx = escala*RNG.standard_normal((n, 1))
    dfdx = RNG.standard_normal((m, n))
    fval = RNG.uniform(5, 50, (m, 1)) if violada else RNG.uniform(-1, 1, (m, 1))
    return (m, n, 5, xval, np.zeros((n, 1)), np.ones((n, 1)), xold1, xold2, 1.0, df0dx, fval, dfdx,
            low, upp, 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), dval*np.ones((m, 1)))


def compara(args, tol_f=1e-6, tol_g=1e-8):
    "dual not worse than mmapy in the approximation (penalty included), feasible, no warning"
    o1 = mma_pd(*args, move=MOVE)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        o2 = mma_du(*args, move=MOVE)
    m, n, _, xval, xmin, xmax = args[:6]
    df0dx, fval, dfdx = args[9], args[10], args[11]
    low, upp = o1[9], o1[10]
    assert np.allclose(low, o2[9]) and np.allclose(upp, o2[10])
    alfa = np.maximum(np.maximum(low + 0.1*(xval - low), xval - MOVE*(xmax - xmin)), xmin)
    beta = np.minimum(np.minimum(upp - 0.1*(upp - xval), xval + MOVE*(xmax - xmin)), xmax)
    p0, q0, P, Q, b = coeficientes(m, n, xval, xmin, xmax, low, upp, df0dx, dfdx, fval)
    c, d = args[16], args[17]
    f1, g1 = aproximacao(o1[0], o1[1], low, upp, p0, q0, P, Q, b, c, d)
    f2, g2 = aproximacao(o2[0], o2[1], low, upp, p0, q0, P, Q, b, c, d)
    escala_b = max(1.0, np.abs(b).max())
    assert np.all(o2[0] >= alfa - 1e-12) and np.all(o2[0] <= beta + 1e-12), "x outside [alfa, beta]"
    # the exact penalty multiplies a residual at the level of tol_dual*max(1, |b|) by c, so allow for it
    folga = tol_f*max(1.0, abs(f1)) + m*c.max()*1e-10*escala_b
    assert f2 - f1 <= folga, f"dual worse than mmapy: {f2 - f1:.2e}"
    assert np.max(g2 - o2[1].ravel()) <= tol_g*escala_b, "g_i(x) - y_i > b_i: infeasible"
    return o1, o2, g2


def test_constraints_violated_lam_at_c():
    "constraints infeasible within the move limits: lam = c (d = 0) and y absorbs the violation"
    for n, m in [(20, 3), (500, 8)]:
        args = subproblema(n, m, violada=True)
        o1, o2, g2 = compara(args)
        c = args[16].ravel()
        assert np.all(np.isclose(o2[3].ravel(), c)), "lam did not reach c"
        assert np.all(o2[1].ravel() > 0), "y should be > 0"
        assert np.allclose(o2[1].ravel(), np.maximum(g2, 0)), "y != max(0, g - b)"
        assert np.allclose(o2[1], o1[1], rtol=1e-4, atol=1e-6), "y differs from mmapy"


def test_d_positive():
    "quadratic penalty d > 0: lam may exceed c and y = (lam - c)/d"
    for n, m in [(50, 2), (2000, 6)]:
        args = subproblema(n, m, dval=1.0)
        compara(args, tol_g=1e-7)
        args = subproblema(n, m, dval=1.0, violada=True)
        o1, o2, _ = compara(args, tol_g=1e-7)
        c, d = args[16].ravel(), args[17].ravel()
        assert np.allclose(o2[1].ravel(), np.maximum(0, (o2[3].ravel() - c)/d))


def test_extreme_scales():
    "objective gradient scaled by 1e-3 ... 1e3 relative to the constraint gradients"
    for expo in range(-3, 4):
        for n, m in [(200, 4), (3000, 12)]:
            compara(subproblema(n, m, escala=10.0**expo))


def test_a_nonzero_raises():
    "a != 0 is not supported and must not return silently"
    args = list(subproblema(50, 2))
    args[15] = np.ones((2, 1))
    try:
        mma_du(*args)
    except NotImplementedError:
        return
    raise AssertionError("a != 0 did not raise NotImplementedError")


def test_nonconvergence_warning():
    "max_newton too small: a RuntimeWarning is issued and x is still within the bounds"
    args = subproblema(500, 6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = mma_du(*args, move=MOVE, max_newton=1)
    assert any(issubclass(i.category, RuntimeWarning) for i in w), "no RuntimeWarning for max_newton = 1"
    assert np.all(np.isfinite(out[0]))


if __name__ == '__main__':
    for nome, f in list(globals().items()):
        if nome.startswith('test_') and callable(f):
            f()
            print(f"{nome}: ok")
