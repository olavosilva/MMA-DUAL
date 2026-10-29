"""
Dual solver: analytic gradient and Hessian of W against finite differences, behaviour across active-set
transitions, agreement of the reorganized solver with the 0.1.2 reference module (benchmarks/
mmadual_v012_reference.py) and with the interior-point solver of mmapy on random subproblems, and the
information returned about the stopping reason.
"""
import os
import sys
import warnings
import importlib.util
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from mmadual import mma_subproblem, solve_dual, recover_multipliers, subsolv_dual, diagnostics
from mmadual.dual_solver import dual_function, dual_hessian, primal_x

RNG = np.random.default_rng(3)


def random_subproblem(n, m, scale=1.0, violated=False, d=0.0, seed=None):
    "MMA subproblem (iteration 5, fixed asymptotes) from random data"
    rng = RNG if seed is None else np.random.default_rng(seed)
    xval = rng.uniform(0.05, 0.95, n)
    xold1 = np.clip(xval + 0.05*rng.standard_normal(n), 0, 1)
    xold2 = np.clip(xold1 + 0.05*rng.standard_normal(n), 0, 1)
    fval = rng.uniform(5, 50, m) if violated else rng.uniform(-1, 1, m)
    return mma_subproblem(5, xval, np.zeros(n), np.ones(n), xold1, xold2, scale*rng.standard_normal(n), fval,
                          rng.standard_normal((m, n)), xval - 0.3, xval + 0.3, 1.0, np.zeros(m), 1e3*np.ones(m), d*np.ones(m), move=0.2)


def interior_point(sub, lam):
    "True if no x_j(lam) is at a bound (W is smooth there) and no lam_i at c with d = 0"
    x, _, _ = primal_x(sub, lam)
    return bool(np.all(x > sub['alfa']) and np.all(x < sub['beta']))


def test_gradient_finite_differences():
    "dW/dlam against central differences at interior points, several subproblems and multipliers"
    for seed in range(5):
        sub = random_subproblem(300, 4, seed=seed)
        rng = np.random.default_rng(100 + seed)
        for _ in range(3):
            lam = rng.uniform(0.01, 0.3, 4)
            if not interior_point(sub, lam):
                continue
            W, grad, *_ = dual_function(sub, lam)
            h = 1e-6
            for i in range(4):
                e = np.zeros(4); e[i] = h
                fd = (dual_function(sub, lam + e)[0] - dual_function(sub, lam - e)[0])/(2*h)
                assert abs(fd - grad[i]) <= 1e-6*max(1.0, abs(grad[i])), f"gradient component {i}: {grad[i]} vs FD {fd}"


def free_set(sub, lam):
    x, _, _ = primal_x(sub, lam)
    return (x > sub['alfa']) & (x < sub['beta'])


def test_hessian_finite_differences_piecewise():
    """d2W/dlam2 against central differences of the gradient at points where the set of free variables does not
       change within the finite-difference step (W is C^2 on each such piece); points where it changes are skipped"""
    h = 1e-6
    for seed in range(5):
        sub = random_subproblem(300, 4, seed=seed)
        rng = np.random.default_rng(200 + seed)
        checked = 0
        for _ in range(20):
            lam = rng.uniform(0.01, 0.3, 4)
            x, p, q = primal_x(sub, lam)
            free = free_set(sub, lam)
            H = dual_hessian(sub, lam, x, p, q)
            assert np.allclose(H, H.T), "analytic Hessian must be symmetric"
            assert np.all(np.linalg.eigvalsh(H) <= 1e-12*max(1.0, np.abs(H).max())), "Hessian must be negative semidefinite"
            Hfd = np.zeros((4, 4))
            same_piece = True
            for i in range(4):
                e = np.zeros(4); e[i] = h
                if not (np.array_equal(free_set(sub, lam + e), free) and np.array_equal(free_set(sub, lam - e), free)):
                    same_piece = False
                    break
                Hfd[:, i] = (dual_function(sub, lam + e)[1] - dual_function(sub, lam - e)[1])/(2*h)
            if not same_piece:
                continue
            assert np.abs(H - Hfd).max() <= 1e-5*max(1.0, np.abs(H).max()), f"Hessian vs FD: {np.abs(H - Hfd).max():.2e}"
            checked += 1
        assert checked >= 5, "too few points with a locally constant active set"


def test_transition_between_free_and_bounded():
    """across a value of lam where a variable reaches a bound, W stays C^1: the change of the gradient over a small
       interval is bounded by the Hessian magnitude times the interval, and the finite difference of W equals the
       mean gradient; the Hessian itself jumps (one rank-one term appears or disappears)"""
    sub = random_subproblem(50, 1, seed=7)
    lams = np.linspace(0.0, 2.0, 4001)
    nfree = np.array([free_set(sub, np.array([l])).sum() for l in lams])
    k = np.flatnonzero(np.diff(nfree) != 0)
    assert k.size > 0, "no transition found"
    lam_t = lams[k[0] + 1]
    h = 1e-5
    ll, lr = np.array([lam_t - 2*h]), np.array([lam_t + 2*h])
    Wl, gl, xl, _, pl, ql = dual_function(sub, ll)
    Wr, gr, xr, _, pr, qr = dual_function(sub, lr)
    Hl, Hr = dual_hessian(sub, ll, xl, pl, ql)[0, 0], dual_hessian(sub, lr, xr, pr, qr)[0, 0]
    assert abs(gr[0] - gl[0]) <= 1.5*max(abs(Hl), abs(Hr))*4*h + 1e-9, "gradient change not bounded by the curvature: discontinuity"
    assert abs((Wr - Wl)/(4*h) - 0.5*(gl[0] + gr[0])) <= 1e-6*max(1.0, abs(gl[0])) + max(abs(Hl), abs(Hr))*4*h
    assert abs(Hl - Hr) > 1e-6*abs(Hl), "the Hessian must jump at the transition"


def load_v012():
    path = os.path.join(HERE, '..', 'benchmarks', 'mmadual_v012_reference.py')
    spec = importlib.util.spec_from_file_location('mmadual_v012', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_same_solution_as_v012():
    "the reorganized solver reproduces the 0.1.2 solver on the same subproblems (same algorithm, same path)"
    v012 = load_v012()
    for n, m, kw in [(200, 4, {}), (3000, 12, dict(scale=1e-3)), (500, 6, dict(violated=True)), (500, 6, dict(d=1.0))]:
        sub = random_subproblem(n, m, **kw)
        x, y, z, lam, info = solve_dual(sub, 1e-10, 100)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x2, y2, z2, lam2 = v012.subsolv_dual(m, n, sub['low'], sub['upp'], sub['alfa'], sub['beta'], sub['p0'], sub['q0'],
                                                 sub['P'], sub['Q'], sub['a0'], sub['a'], sub['b'], sub['c'], sub['d'], 1e-10, 100)
        assert np.abs(x - x2.ravel()).max() <= 1e-12, f"x differs from 0.1.2 by {np.abs(x - x2.ravel()).max():.1e}"
        assert np.abs(lam - lam2.ravel()).max() <= 1e-12*max(1.0, np.abs(lam).max())
        assert info['converged'] or info['reason'] == 'rounding'


try:
    import mmapy
except ImportError:
    mmapy = None


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
def test_not_worse_than_mmapy():
    """on random subproblems the dual solution has an exact-penalty objective no larger than the interior-point
       one (up to its tolerance) and zero constraint violation; the duality gap certifies both"""
    for n, m, kw in [(200, 4, {}), (2000, 8, dict(scale=1e-3)), (500, 6, dict(violated=True)), (300, 3, dict(d=1.0))]:
        sub = random_subproblem(n, m, **kw)
        x, y, z, lam, info = solve_dual(sub, 1e-10, 100)
        xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
        chk = diagnostics.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            o = mmapy.subsolv(m, n, 1e-7, *(sub[k].reshape(-1, 1) if sub[k].ndim == 1 else sub[k] for k in ('low', 'upp', 'alfa', 'beta', 'p0', 'q0')),
                              sub['P'], sub['Q'], sub['a0'], sub['a'].reshape(-1, 1), sub['b'].reshape(-1, 1), sub['c'].reshape(-1, 1), sub['d'].reshape(-1, 1))
        f_pd = diagnostics.exact_penalty_objective(sub, o[0].ravel())
        assert chk['viol'] <= 1e-9*np.max(info['scale']) and chk['stat_x'] <= 1e-12*max(1.0, np.abs(lam).max())
        assert chk['gap'] >= -1e-12 and chk['gap_rel'] <= 1e-6
        assert chk['f_pen'] - f_pd <= 1e-6*max(1.0, abs(f_pd)), f"dual worse than mmapy by {chk['f_pen'] - f_pd:.2e}"


def test_info_and_warning():
    "max_newton too small: info says so, a RuntimeWarning is issued, and the solution is still finite and feasible"
    sub = random_subproblem(500, 6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x, y, z, lam, info = solve_dual(sub, 1e-10, 1)
    assert not info['converged'] and info['reason'] == 'max_newton' and info['newton_steps'] == 1
    assert any(issubclass(i.category, RuntimeWarning) for i in w)
    assert np.all(np.isfinite(x)) and np.all(x >= sub['alfa']) and np.all(x <= sub['beta'])
    x, y, z, lam, info = solve_dual(sub, 1e-10, 100)
    assert info['converged'] and info['reason'] == 'tolerance' and info['residual'] <= 1e-10


def test_compat_subsolv_dual():
    "0.1.x calling convention: column vectors in and out"
    sub = random_subproblem(100, 2)
    col = lambda v: v.reshape(-1, 1)
    x, y, z, lam = subsolv_dual(2, 100, col(sub['low']), col(sub['upp']), col(sub['alfa']), col(sub['beta']), col(sub['p0']), col(sub['q0']),
                                sub['P'], sub['Q'], sub['a0'], col(sub['a']), col(sub['b']), col(sub['c']), col(sub['d']))
    assert x.shape == (100, 1) and y.shape == (2, 1) and z.shape == (1, 1) and lam.shape == (2, 1)
    x2, *_ = solve_dual(sub)
    assert np.array_equal(x.ravel(), x2)


def test_a_negative_raises():
    sub = random_subproblem(50, 2)
    sub['a'] = -np.ones(2)
    with pytest.raises(ValueError):
        solve_dual(sub)


def test_certificate_reported_for_a_zero():
    """the gap of the returned point (computed independently by diagnostics) is within what the tolerance allows,
       and the a = 0 path reports it without changing its stopping rule (converged is the projected-gradient test)"""
    for seed in range(6):
        sub = random_subproblem(150, 4, scale=[1.0, 1e-3, 1e3][seed % 3], violated=seed % 2 == 1, d=[0.0, 0.0, 1.0][seed % 3], seed=100 + seed)
        x, y, z, lam, info = solve_dual(sub)
        assert info['converged'] or info['reason'] == 'rounding', info
        xsi, eta, mu, zet, s, y2 = recover_multipliers(sub, x, y, z, lam)
        G, G_rel, viol = diagnostics.duality_gap_returned(sub, x, y2, z, lam)
        assert abs(G - info['gap_returned']) <= 1e-9*max(1.0, abs(G)) + 1e-12
        assert info['certified'] and info['gap_returned'] <= info['gap_allowed'], info
        F = diagnostics.subproblem_objective(sub, x, y2, z)
        assert info['gap_allowed'] <= 1e-10*np.sum(np.maximum(sub['c'] + sub['d']*y2, lam)*info['scale']) + 3e-14*abs(F)   # tol x scales + rounding of F, W


def test_flat_components_reach_the_bound():
    """a = 0, constraints without any free design variable (every x_j at a bound: the dual function is exactly affine
       in those multipliers): the optimum lam_i = c_i for a positive slope, 0 otherwise, must be reached within
       max_newton = 100 with certified convergence. 0.1.x took unit gradient steps, so a slope of 0.01 and a bound at
       1.8 needed 180 steps (cases 87, 133 and 205 of `benchmarks/minimax_stress.py 5291 350 box`)"""
    sys.path.insert(0, os.path.join(HERE, '..', 'benchmarks'))
    from minimax_stress import affine_case
    rng = np.random.default_rng(5291)
    cases = {}
    for k in range(206):
        sub, slopes = affine_case(rng, box=True)
        if k in (87, 133, 205):
            cases[k] = (sub, slopes)
    for k, (sub, slopes) in cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            x, y, z, lam, info = solve_dual(sub, 1e-10, 100)
        assert info['converged'] and info['certified'], (k, info)
        assert np.abs(lam - np.where(slopes > 0, sub['c'], 0.0)).max() <= 1e-12, (k, lam)
        assert info['newton_steps'] <= 3, (k, info['newton_steps'])
