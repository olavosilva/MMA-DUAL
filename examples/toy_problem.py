"""
Cantilever beam of Svanberg (1987), the classical MMA test problem:
    minimize  0.0624 (x1 + x2 + x3 + x4 + x5)
    s.t.      61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 <= 1,   1 <= xj <= 10
Known optimum: x* = (6.016, 5.309, 4.494, 3.502, 2.153), f* = 1.340.
Solved with the MMA (mmapy's mmasub and mmadual's mmasub) and with the GCMMA (mmapy's
helpers + mmadual's gcmmasub). All three should converge to the same point.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import mmapy                       # noqa: E402
import mmadual                    # noqa: E402


def funcs(x):
    x = x.ravel()
    f0 = 0.0624*np.sum(x)
    df0 = 0.0624*np.ones((5, 1))
    w = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
    f = np.array([[np.sum(w/x**3) - 1.0]])
    df = (-3*w/x**4)[None, :]
    return f0, df0, f, df


def run(method, name, max_iter=60, tol=1e-6):
    m, n = 1, 5
    x = 5.0*np.ones((n, 1)); xmin, xmax = np.ones((n, 1)), 10.0*np.ones((n, 1))
    xold1, xold2 = x.copy(), x.copy(); low, upp = xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1000.0*np.ones((m, 1)), np.zeros((m, 1))
    raa0, raa = 0.01, 0.01*np.ones((m, 1)); raa0eps, raaeps = 1e-6, 1e-6*np.ones((m, 1)); epsimin = 1e-7
    for it in range(1, max_iter + 1):
        f0, df0, f, df = funcs(x)
        if method == 'mma_mmapy':
            xnew, *_, low, upp = mmapy.mmasub(m, n, it, x, xmin, xmax, xold1, xold2, f0, df0, f, df, low, upp, a0, a, c, d)
        elif method == 'mmadual':
            xnew, *_, low, upp = mmadual.mmasub(m, n, it, x, xmin, xmax, xold1, xold2, f0, df0, f, df, low, upp, a0, a, c, d)
        else:                                                           # GCMMA with the dual subproblem
            low, upp, raa0, raa = mmapy.asymp(it, n, x, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0, df)
            out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
            xnew, f0app, fapp = out[0], out[9], out[10]
            f0n, _, fn, _ = funcs(xnew); inner = 0
            while not mmapy.concheck(m, epsimin, f0app, f0n, fapp, fn) and inner < 15:
                inner += 1
                raa0, raa = mmapy.raaupdate(xnew, x, xmin, xmax, low, upp, f0n, fn, f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
                out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
                xnew, f0app, fapp = out[0], out[9], out[10]
                f0n, _, fn, _ = funcs(xnew)
        change = float(np.max(np.abs(xnew - x)))
        xold2, xold1, x = xold1, x, xnew
        if change < tol:
            break
    f0, _, f, _ = funcs(x)
    print(f"{name:28s}: {it:3d} iterations, f0 = {f0:.6f}, g = {float(f[0, 0]): .2e}, x = {np.round(x.ravel(), 4)}")
    return x


if __name__ == '__main__':
    print("Svanberg's beam, optimum f* = 1.340 at x* = (6.016, 5.309, 4.494, 3.502, 2.153)")
    xa = run('mma_mmapy', 'MMA, mmapy subsolv')
    xb = run('mmadual', 'MMA, dual subproblem')
    xc = run('gcmmadual', 'GCMMA, dual subproblem')
    print(f"max |x_mmapy - x_dual| = {np.max(np.abs(xa - xb)):.2e};  max |x_mma - x_gcmma| = {np.max(np.abs(xa - xc)):.2e}")
