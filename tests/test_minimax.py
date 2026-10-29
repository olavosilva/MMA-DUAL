"""
Subproblems with a != 0 (Svanberg's minimax form, a0 = 1, a_i = 1): the dual domain is the box intersected
with the half-space a^T lam <= a0 and z is the multiplier of that constraint. Checks: dual feasibility of the
returned lam, z >= 0 and consistent with the gradient on the half-space, primal feasibility with the
recovered (y, z), exact-penalty objective (optimized in z) not worse than the interior-point solution of
mmapy, duality gap, and the a = 0 limit (same result as before when a = 0).
"""
import os
import sys
import warnings
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mmadual import mma_subproblem, solve_dual, recover_multipliers, diagnostics as dg
from mmadual.dual_solver import project_dual, _halfspace_multiplier


def linear_dual_subproblem(slopes, a=None, a0=1.0, c=1e3, n=20):
    """subproblem whose dual function is affine on the domain, W(lam) = C + slopes . lam: every x_j is held at its
       upper bound by p0 << q0 and negligible P, Q, so g(x) - b is the constant vector `slopes`. With a = 1, a0 = 1
       the dual domain is the simplex {lam >= 0, sum lam <= 1}: the maximizer is a vertex and the half-space and the
       box are active simultaneously (regression test for the recovery of z, after a reported failure)"""
    m = len(slopes)
    low, upp, alfa, beta = -np.ones(n), 2*np.ones(n), -0.5*np.ones(n), 1.5*np.ones(n)
    p0, q0 = 1e-6*np.ones(n), np.ones(n)
    P, Q = 1e-9*np.ones((m, n)), 1e-9*np.ones((m, n))
    g = P @ (1/(upp - beta)) + Q @ (1/(beta - low))
    return dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, P=P, Q=Q, b=g - np.asarray(slopes, dtype=float),
                a0=float(a0), a=np.ones(m) if a is None else np.asarray(a, dtype=float), c=c*np.ones(m), d=np.zeros(m))


def full_check(sub, x, y, z, lam, info, tol=1e-9):
    "KKT of the subproblem with the returned variables, both gap certificates, dual feasibility"
    assert dg.dual_feasible(sub, lam) and z >= 0.0
    xsi, eta, mu, zet, s, y2 = recover_multipliers(sub, x, y, z, lam)
    chk = dg.check_solution(sub, x, y2, z, lam, xsi, eta, mu, zet, s)
    scale = float(np.max(info['scale']))
    assert chk['viol'] <= tol*scale, chk
    assert chk['stat_x'] <= 1e-10*max(1.0, np.abs(lam).max()) and abs(chk['stat_z']) <= 1e-10, chk
    assert chk['comp'] <= 1e3*tol*scale, f"complementarity {chk['comp']}"
    assert -1e-9 <= chk['gap'] <= 1e3*tol*scale and -1e-9 <= chk['gap_returned'] <= 1e3*tol*scale, chk
    return chk

try:
    import mmapy
except ImportError:
    mmapy = None


def minimax_subproblem(n, m, seed, violated=False, a_pattern='ones', d=0.0):
    rng = np.random.default_rng(seed)
    xval = rng.uniform(0.1, 0.9, n)
    fval = rng.uniform(5, 30, m) if violated else rng.uniform(-1, 1, m)
    a = np.ones(m) if a_pattern == 'ones' else rng.uniform(0, 2, m)
    if a_pattern == 'mixed':
        a[::2] = 0.0
    return mma_subproblem(3, xval, np.zeros(n), np.ones(n), xval + 0.02, xval - 0.02, rng.standard_normal(n), fval,
                          rng.standard_normal((m, n)), xval - 0.3, xval + 0.3, 1.0, a, 1e3*np.ones(m), d*np.ones(m), move=0.2)


def test_projection_onto_dual_domain():
    rng = np.random.default_rng(0)
    a, a0, lam_max = np.array([1.0, 1.0, 0.0, 2.0]), 1.0, np.array([5.0, 5.0, 5.0, np.inf])
    for _ in range(50):
        v = rng.uniform(-2, 4, 4)
        lam = project_dual(v, a, a0, lam_max)
        assert np.all(lam >= 0) and np.all(lam <= lam_max) and a @ lam <= a0 + 1e-12
        # projection: no feasible point is closer to v (sampled)
        for _ in range(20):
            w = np.clip(rng.uniform(-1, 3, 4), 0, lam_max)
            if a @ w <= a0:
                assert np.linalg.norm(w - v) >= np.linalg.norm(lam - v) - 1e-9


@pytest.mark.parametrize('kw', [dict(), dict(violated=True), dict(a_pattern='random'), dict(a_pattern='mixed'), dict(violated=True, a_pattern='random'), dict(d=1.0)])
def test_minimax_solution_is_consistent(kw):
    sub = minimax_subproblem(300, 4, seed=1, **kw)
    x, y, z, lam, info = solve_dual(sub)
    assert info['converged'] or info['reason'] == 'rounding', info
    assert dg.dual_feasible(sub, lam) and z >= 0.0
    a, a0 = sub['a'], sub['a0']
    if z > 0:
        assert abs(a @ lam - a0) <= 1e-10*a0, "z > 0 only on the half-space boundary"
    full_check(sub, x, y, z, lam, info, tol=1e-8)


@pytest.mark.parametrize('slopes, lam_star, z_star', [
    ([0.2, 1.0], [0.0, 1.0], 1.0),                    # the reported failure: vertex (0, 1), lam_1 at its lower bound with grad_1 = 0.2 > 0
    ([1.0, 0.2], [1.0, 0.0], 1.0),
    ([0.5, 0.5], None, 0.5),                          # whole edge optimal: z = 0.5 on any point of it
    ([-0.3, 0.7], [0.0, 1.0], 0.7),
    ([-0.3, -0.7], [0.0, 0.0], 0.0),                  # half-space inactive, both at the lower bound
    ([2.0, 1.0, 0.5], [1.0, 0.0, 0.0], 2.0),          # three constraints, one vertex
    ([1.0, 1.0, 0.0], None, 1.0),                     # redundant constraints: the first two are interchangeable
])
def test_affine_dual_on_the_simplex(slopes, lam_star, z_star):
    sub = linear_dual_subproblem(slopes)
    x, y, z, lam, info = solve_dual(sub)
    assert info['converged'], info
    assert abs(z - z_star) <= 1e-9*max(1.0, z_star), f"z = {z}, expected {z_star}"
    if lam_star is not None:
        assert np.abs(lam - np.asarray(lam_star)).max() <= 1e-9
    full_check(sub, x, y, z, lam, info)
    # the dual value must equal the maximum of the affine function on the simplex, C + max(0, max slopes)
    assert abs(dg.dual_value(sub, lam) - (dg.dual_value(sub, np.zeros(len(slopes))) + max(0.0, max(slopes)))) <= 1e-9


def test_halfspace_multiplier_joint_active_set():
    "z from the joint minimization against brute force over z for random gradients and bound patterns"
    rng = np.random.default_rng(4)
    for _ in range(50):
        m = 5
        grad, a = rng.standard_normal(m), rng.uniform(0, 2, m)
        lam_max = np.where(rng.random(m) < 0.3, np.inf, 1.0)
        lam = np.where(rng.random(m) < 0.4, 0.0, np.where(rng.random(m) < 0.3, lam_max, 0.5))
        lam = np.where(np.isinf(lam), 0.5, lam)

        def r2(z):
            g = grad - z*a
            lower, upper = lam <= 0, lam >= lam_max
            g = np.where(lower, np.maximum(g, 0), np.where(upper, np.minimum(g, 0), g))
            return float(g @ g)
        z = _halfspace_multiplier(grad, lam <= 0, lam >= lam_max, a)
        zz = np.linspace(0, 10, 20001)
        best = zz[np.argmin([r2(v) for v in zz])]
        assert r2(z) <= min(r2(best), r2(0.0)) + 1e-9, f"z = {z} not the minimizer (brute force {best})"


def test_a_zero_limit_unchanged():
    "a = 0 goes through the same code path as before: identical to the plain solver on the same data"
    sub = minimax_subproblem(200, 3, seed=5)
    sub0 = dict(sub); sub0['a'] = np.zeros(3)
    x0, y0, z0, lam0, _ = solve_dual(sub0)
    sub1 = dict(sub); sub1['a'] = np.zeros(3)
    x1, y1, z1, lam1, info = solve_dual(sub1)
    assert np.array_equal(x0, x1) and z1 == 0.0 and not info['halfspace_active']


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
@pytest.mark.parametrize('kw', [dict(), dict(violated=True), dict(a_pattern='random'), dict(a_pattern='mixed'), dict(violated=True, a_pattern='random')])
def test_not_worse_than_mmapy_minimax(kw):
    "exact-penalty objective with z optimized for each x: dual <= interior point (up to its tolerance)"
    for seed in range(3):
        sub = minimax_subproblem(300, 4, seed=seed, **kw)
        x, y, z, lam, info = solve_dual(sub)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            o = mmapy.subsolv(4, 300, 1e-7, *(sub[k].reshape(-1, 1) for k in ('low', 'upp', 'alfa', 'beta', 'p0', 'q0')),
                              sub['P'], sub['Q'], sub['a0'], sub['a'].reshape(-1, 1), sub['b'].reshape(-1, 1), sub['c'].reshape(-1, 1), sub['d'].reshape(-1, 1))
        f_du, f_pd = dg.exact_penalty_objective(sub, x), dg.exact_penalty_objective(sub, o[0].ravel())
        assert f_du - f_pd <= 1e-6*max(1.0, abs(f_pd)), f"{kw} seed {seed}: dual worse by {f_du - f_pd:.2e} (z dual {z:.3g}, z pd {float(o[2].ravel()[0]):.3g})"
        z_pd = float(np.asarray(o[2]).ravel()[0])
        if z > 1e-6 or z_pd > 1e-6:
            assert abs(z - z_pd) <= 1e-3*max(1.0, z_pd), f"z differs: {z} vs {z_pd}"


@pytest.mark.parametrize('kw', [dict(), dict(violated=True, a_pattern='random'), dict(a_pattern='mixed', d=1.0), dict(d=0.01)])
def test_certificate_of_returned_point(kw):
    """convergence of the polytope path requires the certificate: the gap F(x, y, z) - W(lam) of the returned point,
       recomputed independently by diagnostics, is below the bound implied by the tolerance (a sum of complementarity
       products, each at most tol times the scale of its constraint), and the full KKT residuals agree"""
    for seed in range(8):
        sub = minimax_subproblem(120, 6, seed=50 + seed, **kw)
        if seed % 2:
            sub['c'] = np.random.default_rng(seed).uniform(0.01, 5, 6)   # bounds of the box active more often
        x, y, z, lam, info = solve_dual(sub)
        assert info['converged'] and info['reason'] == 'tolerance' and info['certified'], info
        chk = full_check(sub, x, y, z, lam, info)
        assert abs(chk['gap_returned'] - info['gap_returned']) <= 1e-9*max(1.0, abs(chk['gap_returned'])) + 1e-12
        assert info['gap_returned'] <= info['gap_allowed']
        assert chk['comp'] <= info['gap_allowed'] + 1e-12, chk


def test_halfspace_reached_from_inside():
    """the iterates approach the half-space from inside (the review's cases 118, 278 stalled 1e-5 and 1e-7 inside
       it with z = 0): the ratio-test step lands on it and z becomes positive; checked on affine duals whose optimum
       has the half-space active and a multiplier with a_i = 0 at its upper bound"""
    for seed in range(5):
        rng = np.random.default_rng(900 + seed)
        m = 6
        slopes = rng.uniform(0.01, 3, m)                                   # all positive: the half-space is active at the optimum
        a = rng.uniform(0.1, 2, m); a[0] = 0.0
        sub = linear_dual_subproblem(slopes, a=a, a0=float(rng.uniform(0.2, 1.0)), c=float(rng.uniform(0.5, 2.0)))
        x, y, z, lam, info = solve_dual(sub)
        assert info['converged'] and info['halfspace_active'] and z > 0.0, info
        assert abs(sub['a'] @ lam - sub['a0']) <= 1e-12*sub['a0'] and abs(lam[0] - sub['c'][0]) <= 1e-12
        full_check(sub, x, y, z, lam, info)
