"""
Complete GCMMA iterations on Svanberg's beam without mmapy, and iterate-by-iterate agreement with the
same loop driven by mmapy's helpers (asymp, concheck, raaupdate) when mmapy is installed. Also the
KKT-based stopping with mmadual.kktcheck against mmapy.kktcheck.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import mmadual

try:
    import mmapy
except ImportError:
    mmapy = None

W = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
X_STAR = np.array([6.016, 5.309, 4.494, 3.502, 2.153])


def beam(x):
    x = np.asarray(x, dtype=float).ravel()
    return 0.0624*np.sum(x), 0.0624*np.ones((5, 1)), np.array([[np.sum(W/x**3) - 1.0]]), (-3*W/x**4).reshape(1, 5)


def gcmma_loop(helpers, iters=12, innermax=15):
    "returns the list of outer iterates and the number of inner iterations, with asymp/concheck/raaupdate from `helpers`"
    m, n = 1, 5
    x = 5.0*np.ones((n, 1)); xold1, xold2 = x.copy(), x.copy()
    xmin, xmax = np.ones((n, 1)), 10.0*np.ones((n, 1))
    low, upp = xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1000.0*np.ones((m, 1)), np.ones((m, 1))
    epsimin, raa0, raa, raa0eps, raaeps = 1e-7, 0.01, 0.01*np.ones((m, 1)), 1e-6, 1e-6*np.ones((m, 1))
    hist, inner_total = [], 0
    f0, df0, f, df = beam(x)
    for it in range(1, iters + 1):
        low, upp, raa0, raa = helpers.asymp(it, n, x, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0, df)
        out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
        xnew, f0app, fapp = out[0], out[9], out[10]
        f0new, _, fnew, _ = beam(xnew)
        inner = 0
        while not helpers.concheck(m, epsimin, f0app, f0new, fapp, fnew) and inner < innermax:
            raa0, raa = helpers.raaupdate(xnew, x, xmin, xmax, low, upp, f0new, fnew, f0app, fapp, raa0, raa.copy(), raa0eps, raaeps, epsimin)
            out = mmadual.gcmmasub(m, n, it, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0, df0, f, df, a0, a, c, d)
            xnew, f0app, fapp = out[0], out[9], out[10]
            f0new, _, fnew, _ = beam(xnew)
            inner += 1
        inner_total += inner
        xold2, xold1, x = xold1, x, xnew
        f0, df0, f, df = beam(x)
        hist.append(x.ravel().copy())
    return hist, inner_total, out


def test_gcmma_converges_without_mmapy():
    hist, inner, out = gcmma_loop(mmadual, iters=25)
    assert np.abs(hist[-1] - X_STAR).max() <= 2e-3, f"GCMMA did not reach the optimum: {hist[-1]}"
    assert inner > 0, "the beam problem needs inner iterations (conservativeness)"
    f0, df0, f, df = beam(hist[-1])
    assert f[0, 0] <= 1e-6, "constraint must be satisfied at the end"


def test_monotone_decrease_of_objective():
    """from a feasible iterate, the conservative approximations guarantee f0(x_new) <= f0app(x_new) <= f0(x_k)
       (Svanberg 2002); the conservativeness test allows epsimin, so the decrease is checked to that tolerance,
       and only while the constraint is not active (afterwards the iterates fluctuate within the tolerance)"""
    hist, _, _ = gcmma_loop(mmadual, iters=20)
    f0s = np.array([beam(x)[0] for x in hist])
    f1s = np.array([beam(x)[2][0, 0] for x in hist])
    assert np.all(f1s[:5] < 0), "the first iterates of the beam problem are strictly feasible"
    assert np.all(np.diff(f0s[:9]) < 0), "f0 must decrease strictly while the constraint is inactive"
    assert np.all(np.diff(f0s) <= 1e-7), "f0 must not increase beyond epsimin"


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
def test_same_iterates_with_mmapy_helpers():
    "the loop driven by mmapy's asymp/concheck/raaupdate gives the same iterates (same subproblem solver)"
    h1, i1, _ = gcmma_loop(mmadual, iters=12)
    h2, i2, _ = gcmma_loop(mmapy, iters=12)
    assert i1 == i2
    for a, b in zip(h1, h2):
        assert np.abs(a - b).max() <= 1e-10, f"iterates differ: {np.abs(a - b).max():.1e}"


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
def test_kktcheck_matches_mmapy():
    hist, _, out = gcmma_loop(mmadual, iters=10)
    x = hist[-1].reshape(-1, 1)
    f0, df0, f, df = beam(x)
    args = (1, 5, x, *out[1:9], np.ones((5, 1)), 10.0*np.ones((5, 1)), df0, f, df, 1.0, np.zeros((1, 1)), 1000.0*np.ones((1, 1)), np.ones((1, 1)))
    r1, n1, m1 = mmadual.kktcheck(*args)
    r2, n2, m2 = mmapy.kktcheck(*args)
    assert abs(n1 - n2) <= 1e-12*max(1.0, n2) and abs(m1 - m2) <= 1e-12*max(1.0, m2)
    assert np.abs(r1.ravel() - r2.ravel()).max() <= 1e-12*max(1.0, np.abs(r2).max())
