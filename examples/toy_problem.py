"""
Svanberg's five-variable cantilever beam (Svanberg 1987, Section 6; the test problem of GCMMA-MMA-Python),
solved with the MMA and with the GCMMA using only mmadual: mmasub for the MMA, and asymp, gcmmasub,
concheck and raaupdate for the GCMMA outer/inner iterations. If mmapy is installed, the MMA is also run
with mmapy.mmasub for comparison; otherwise that run is skipped.

    minimize   f0(x) = 0.0624 (x1 + x2 + x3 + x4 + x5)
    subject to f1(x) = 61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 - 1 <= 0,   1 <= xj <= 10
    optimum    x* = (6.016, 5.309, 4.494, 3.502, 2.153),  f0* = 1.340

Run:  python examples/toy_problem.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import mmadual

try:
    import mmapy
except ImportError:
    mmapy = None

W = np.array([61.0, 37.0, 19.0, 7.0, 1.0])


def beam(x):
    "objective, constraint and their gradients (column vectors / (1, n) matrix, as mmasub expects)"
    x = np.asarray(x, dtype=float).ravel()
    f0 = 0.0624*np.sum(x)
    f = np.array([[np.sum(W/x**3) - 1.0]])
    df0 = 0.0624*np.ones((5, 1))
    df = (-3*W/x**4).reshape(1, 5)
    return f0, df0, f, df


def run_mma(mmasub, label, maxiter=60, tol=1e-6):
    "MMA loop with the given mmasub (mmadual.mmasub or mmapy.mmasub); stops on the KKT residual of the original problem"
    m, n = 1, 5
    x = 5.0*np.ones((n, 1)); xold1, xold2 = x.copy(), x.copy()
    xmin, xmax = np.ones((n, 1)), 10.0*np.ones((n, 1))
    low, upp = xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1000.0*np.ones((m, 1)), np.ones((m, 1))
    for it in range(1, maxiter + 1):
        f0, df0, f, df = beam(x)
        xnew, y, z, lam, xsi, eta, mu, zet, s, low, upp = mmasub(m, n, it, x, xmin, xmax, xold1, xold2, f0, df0, f, df, low, upp, a0, a, c, d)
        xold2, xold1, x = xold1, x, xnew
        f0, df0, f, df = beam(x)
        _, _, kktmax = mmadual.kktcheck(m, n, x, y, z, lam, xsi, eta, mu, zet, s, xmin, xmax, df0, f, df, a0, a, c, d)
        if kktmax <= tol:
            break
    print(f"{label:28s}: {it:3d} iterations, f0 = {f0:.6f}, f1 = {f[0, 0]:+.2e}, kkt = {kktmax:.1e}, x = {np.round(x.ravel(), 4)}")
    return x.ravel()


def run_gcmma(label, maxiter=60, tol=1e-6, innermax=15):
    "GCMMA loop with mmadual's asymp, gcmmasub, concheck and raaupdate (no mmapy)"
    m, n = 1, 5
    x = 5.0*np.ones((n, 1)); xold1, xold2 = x.copy(), x.copy()
    xmin, xmax = np.ones((n, 1)), 10.0*np.ones((n, 1))
    low, upp = xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1000.0*np.ones((m, 1)), np.ones((m, 1))
    epsimin, raa0, raa, raa0eps, raaeps = 1e-7, 0.01, 0.01*np.ones((m, 1)), 1e-6, 1e-6*np.ones((m, 1))
    inner_total = 0
    f0, df0, f, df = beam(x)
    for it in range(1, maxiter + 1):
        low, upp, raa0, raa = mmadual.asymp(it, n, x, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0, df)
        out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
        xnew, f0app, fapp = out[0], out[9], out[10]
        f0new, _, fnew, _ = beam(xnew)
        inner = 0
        while not mmadual.concheck(m, epsimin, f0app, f0new, fapp, fnew) and inner < innermax:
            raa0, raa = mmadual.raaupdate(xnew, x, xmin, xmax, low, upp, f0new, fnew, f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
            out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
            xnew, f0app, fapp = out[0], out[9], out[10]
            f0new, _, fnew, _ = beam(xnew)
            inner += 1
        inner_total += inner
        xold2, xold1, x = xold1, x, xnew
        f0, df0, f, df = beam(x)
        _, _, kktmax = mmadual.kktcheck(m, n, x, *out[1:9], xmin, xmax, df0, f, df, a0, a, c, d)
        if kktmax <= tol:
            break
    print(f"{label:28s}: {it:3d} iterations ({inner_total} inner), f0 = {f0:.6f}, f1 = {f[0, 0]:+.2e}, kkt = {kktmax:.1e}, x = {np.round(x.ravel(), 4)}")
    return x.ravel()


if __name__ == '__main__':
    print("Svanberg's beam, optimum f0* = 1.340 at x* = (6.016, 5.309, 4.494, 3.502, 2.153)")
    xa = run_mma(mmadual.mmasub, 'MMA, mmadual')
    xb = run_gcmma('GCMMA, mmadual')
    if mmapy is not None:
        xc = run_mma(mmapy.mmasub, 'MMA, mmapy (reference)')
        print(f"max |x_mmadual - x_mmapy| = {np.abs(xa - xc).max():.2e}")
    print(f"max |x_MMA - x_GCMMA| = {np.abs(xa - xb).max():.2e}")
