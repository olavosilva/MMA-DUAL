"""
Comparison of mmadual with mmapy on random MMA and GCMMA subproblems.

Both build the SAME subproblem (asymptotes, alfa/beta, p0, q0, P, Q, b). At the point
returned by each solver, the value of the separable approximation is evaluated, including
the exact penalty sum_i c_i max(0, g_i) of the artificial variables (d = 0) or
sum_i (c_i y_i + d_i y_i^2/2) (d > 0). The dual solution must be at least as good as the
interior-point one (up to a tolerance) and feasible with respect to the subproblem.

Run:  python tests/test_against_mmapy.py      or      pytest tests/
Requires mmapy (pip install mmapy).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mmapy import mmasub as mma_pd, gcmmasub as gc_pd, asymp      # noqa: E402
from mmadual import mmasub as mma_du, gcmmasub as gc_du          # noqa: E402

TOL_OBJ = 1e-6          # dual may be worse than mmapy by at most this (relative to |f|)
TOL_VIOL = 1e-6         # feasibility of the subproblem at the dual solution


def coefficients(m, n, xval, xmin, xmax, df0dx, fval, dfdx, low, upp, raa0=1e-5, raa=None, gc=False):
    "p0, q0, P, Q, b of the approximation, as built by mmapy (mmasub or gcmmasub)"
    xmami_inv = 1/np.maximum(xmax - xmin, 1e-5)
    ux1, xl1 = upp - xval, xval - low
    p0, q0 = np.maximum(df0dx, 0), np.maximum(-df0dx, 0)
    P, Q = np.maximum(dfdx, 0), np.maximum(-dfdx, 0)
    if gc:
        pq0 = p0 + q0
        p0, q0 = (p0 + 0.001*pq0 + raa0*xmami_inv)*ux1**2, (q0 + 0.001*pq0 + raa0*xmami_inv)*xl1**2
        PQ = P + Q
        P, Q = (P + 0.001*PQ + raa @ xmami_inv.T)*ux1.T**2, (Q + 0.001*PQ + raa @ xmami_inv.T)*xl1.T**2
    else:
        pq0 = 0.001*(p0 + q0) + raa0*xmami_inv
        p0, q0 = (p0 + pq0)*ux1**2, (q0 + pq0)*xl1**2
        PQ = 0.001*(P + Q) + raa0*xmami_inv.T
        P, Q = (P + PQ)*ux1.T**2, (Q + PQ)*xl1.T**2
    b = P @ (1/ux1) + Q @ (1/xl1) - fval
    return p0, q0, P, Q, b


def approximation(coef, x, low, upp, y, c, d):
    "objective of the approximation at x, penalty of the artificial variables, and max violation"
    p0, q0, P, Q, b = coef
    f = float(np.sum(p0/(upp - x) + q0/(x - low)))
    g = (P @ (1/(upp - x)) + Q @ (1/(x - low)) - b).ravel()
    y = y.ravel()
    if d.max() > 0:
        pen = float(np.sum(c.ravel()*y + 0.5*d.ravel()*y**2))
        viol = float(max((g - y).max(), 0.0))
    else:
        pen = float(np.sum(c.ravel()*np.maximum(g, 0)))
        viol = float(max((g - y).max(), 0.0))
    return f, pen, viol


def random_problem(rng, m, n, viol0, with_d=False):
    xval = rng.uniform(0.05, 0.95, (n, 1)); xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
    xold1 = np.clip(xval + 0.03*rng.standard_normal((n, 1)), 0, 1)
    xold2 = np.clip(xval + 0.03*rng.standard_normal((n, 1)), 0, 1)
    df0dx = rng.standard_normal((n, 1))*10**rng.uniform(-3, 1, (n, 1))
    dfdx = rng.standard_normal((m, n))*10**rng.uniform(-2, 0, (m, n))
    fval = np.full((m, 1), viol0) + 0.1*rng.standard_normal((m, 1))
    a0, a, c = 1.0, np.zeros((m, 1)), 1e4*np.ones((m, 1))
    d = np.ones((m, 1)) if with_d else np.zeros((m, 1))
    return xval, xmin, xmax, xold1, xold2, df0dx, dfdx, fval, a0, a, c, d


MMA_CASES = [('m=1 inactive', 1, 800, 1, -50.0, False), ('m=4 active', 4, 800, 1, 0.0, False),
             ('m=4, iter > 2', 4, 800, 5, 0.0, False), ('m=10, n=3000', 10, 3000, 7, 0.05, False),
             ('m=4 infeasible (y > 0)', 4, 800, 3, 50.0, False), ('m=2, d > 0', 2, 500, 4, 0.1, True)]


def run_mma(verbose=True):
    rng = np.random.default_rng(0)
    rows = []
    for name, m, n, it, viol0, with_d in MMA_CASES:
        xval, xmin, xmax, xold1, xold2, df0dx, dfdx, fval, a0, a, c, d = random_problem(rng, m, n, viol0, with_d)
        low, upp = xval - 0.5, xval + 0.5
        args = (m, n, it, xval, xmin, xmax, xold1, xold2, 0.0, df0dx, fval, dfdx, low, upp, a0, a, c, d)
        kw = dict(move=0.05, asymin=1e-3)
        t = time.perf_counter(); o1 = mma_pd(*args, **kw); t1 = time.perf_counter() - t
        t = time.perf_counter(); o2 = mma_du(*args, **kw); t2 = time.perf_counter() - t
        assert np.allclose(o1[9], o2[9]) and np.allclose(o1[10], o2[10]), 'asymptotes differ'
        coef = coefficients(m, n, xval, xmin, xmax, df0dx, fval, dfdx, o1[9], o1[10])
        f1, pen1, _ = approximation(coef, o1[0], o1[9], o1[10], o1[1], c, d)
        f2, pen2, viol2 = approximation(coef, o2[0], o2[9], o2[10], o2[1], c, d)
        gap = (f1 + pen1) - (f2 + pen2)
        rows.append((name, t1, t2, gap, viol2))
        assert gap >= -TOL_OBJ*max(1.0, abs(f1 + pen1)), f'{name}: dual worse than mmapy by {-gap:.3e}'
        assert viol2 <= TOL_VIOL, f'{name}: dual solution infeasible ({viol2:.3e})'
    if verbose:
        print(f"MMA subproblems\n{'case':26s} | {'t mmapy [s]':>11s} | {'t dual [s]':>10s} | {'f_mmapy - f_dual':>16s} | viol dual")
        for name, t1, t2, gap, viol2 in rows:
            print(f"{name:26s} | {t1:11.3f} | {t2:10.4f} | {gap:16.2e} | {viol2:.1e}")
    return rows


GC_CASES = [('m=4, n=800', 4, 800, 0.0), ('m=10, n=3000', 10, 3000, 0.05), ('m=3 infeasible', 3, 800, 50.0)]


def run_gcmma(verbose=True):
    rng = np.random.default_rng(1)
    rows = []
    for name, m, n, viol0 in GC_CASES:
        xval, xmin, xmax, xold1, xold2, df0dx, dfdx, fval, a0, a, c, d = random_problem(rng, m, n, viol0)
        f0val = np.array([[1.0]])
        raa0eps, raaeps = 1e-6, 1e-6*np.ones((m, 1))
        low, upp, raa0, raa = asymp(3, n, xval, xold1, xold2, xmin, xmax, xval - 0.5, xval + 0.5, 0.01,
                                    0.01*np.ones((m, 1)), raa0eps, raaeps, df0dx, dfdx)
        args = (m, n, 3, 1e-7, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
        t = time.perf_counter(); o1 = gc_pd(*args); t1 = time.perf_counter() - t
        t = time.perf_counter(); o2 = gc_du(*args); t2 = time.perf_counter() - t
        coef = coefficients(m, n, xval, xmin, xmax, df0dx, fval, dfdx, low, upp, raa0, raa, gc=True)
        f1, pen1, _ = approximation(coef, o1[0], low, upp, o1[1], c, d)
        f2, pen2, viol2 = approximation(coef, o2[0], low, upp, o2[1], c, d)
        gap = (f1 + pen1) - (f2 + pen2)
        dapp = (float(abs(o1[9] - o2[9]).max()), float(abs(o1[10] - o2[10]).max()))
        rows.append((name, t1, t2, gap, viol2, dapp))
        assert gap >= -TOL_OBJ*max(1.0, abs(f1 + pen1)), f'{name}: dual worse than mmapy by {-gap:.3e}'
        assert viol2 <= TOL_VIOL, f'{name}: dual solution infeasible ({viol2:.3e})'
        assert dapp[0] <= 1e-3*max(1.0, float(np.abs(o1[9]).max())) and dapp[1] <= 1e-3, f'{name}: f0app/fapp differ'
    if verbose:
        print(f"\nGCMMA subproblems\n{'case':26s} | {'t mmapy [s]':>11s} | {'t dual [s]':>10s} | {'f_mmapy - f_dual':>16s} | viol dual | |df0app|, |dfapp|")
        for name, t1, t2, gap, viol2, dapp in rows:
            print(f"{name:26s} | {t1:11.3f} | {t2:10.4f} | {gap:16.2e} | {viol2:9.1e} | {dapp[0]:.1e}, {dapp[1]:.1e}")
    return rows


def test_mma_subproblems():
    run_mma(verbose=False)


def test_gcmma_subproblems():
    run_gcmma(verbose=False)


if __name__ == '__main__':
    run_mma(); run_gcmma()
    print("\nall checks passed: the dual solution is never worse than mmapy and is feasible")
