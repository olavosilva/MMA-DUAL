"""
diagnostics module: KKT residuals, duality gap and objectives computed from the subproblem equations, on
solutions of the dual solver, on perturbed points (gap must grow, violation appear) and, if mmapy is
installed, on interior-point solutions (same checks, with the accuracy they reach).
"""
import os
import sys
import warnings
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mmadual import mma_subproblem, solve_dual, recover_multipliers, diagnostics as dg

try:
    import mmapy
except ImportError:
    mmapy = None


def subproblem(n=400, m=5, seed=0, violated=False, d=0.0):
    rng = np.random.default_rng(seed)
    xval = rng.uniform(0.1, 0.9, n)
    fval = rng.uniform(5, 30, m) if violated else rng.uniform(-1, 1, m)
    return mma_subproblem(3, xval, np.zeros(n), np.ones(n), xval + 0.02, xval - 0.02, rng.standard_normal(n), fval,
                          rng.standard_normal((m, n)), xval - 0.3, xval + 0.3, 1.0, np.zeros(m), 1e3*np.ones(m), d*np.ones(m), move=0.2)


def solved(sub):
    x, y, z, lam, info = solve_dual(sub)
    xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
    return x, y, z, lam, xsi, eta, mu, zet, s, info


@pytest.mark.parametrize('kw', [dict(), dict(violated=True), dict(d=1.0), dict(violated=True, d=1.0)])
def test_kkt_of_dual_solution(kw):
    sub = subproblem(**kw)
    x, y, z, lam, xsi, eta, mu, zet, s, info = solved(sub)
    chk = dg.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)
    scale = float(np.max(info['scale']))
    assert chk['stat_x'] <= 1e-12*max(1.0, np.abs(lam).max())           # x minimizes the Lagrangian exactly
    assert chk['stat_y'] <= 1e-12*sub['c'].max() and chk['stat_z'] == 0.0
    assert chk['box'] == 0.0 and chk['dual'] == 0.0
    assert chk['viol'] <= 1e-9*scale and chk['feas'] <= 1e-9*scale
    assert chk['comp'] <= 1e3*1e-9*scale                                  # lam s and mu y: residual times c
    curvature = np.max((sub['p0'] + lam @ sub['P'])/(sub['upp'] - x)**2)
    assert chk['proj_x'] <= 1e-12*max(1.0, curvature)              # rounding of p/(U-x)^2 - q/(x-L)^2 at the closed-form minimizer
    assert 0.0 <= chk['gap'] <= 1e3*1e-9*scale and chk['gap_rel'] <= 1e-6


def test_gap_detects_suboptimality():
    "moving x away from the solution raises the exact-penalty objective and the gap; the gap bounds the difference"
    sub = subproblem()
    x, y, z, lam, *_ = solved(sub)
    f_star = dg.exact_penalty_objective(sub, x)
    G0, _ = dg.duality_gap(sub, x, lam)
    rng = np.random.default_rng(1)
    for eps in (1e-3, 1e-2, 1e-1):
        xp = np.clip(x + eps*rng.standard_normal(x.size), sub['alfa'], sub['beta'])
        fp = dg.exact_penalty_objective(sub, xp)
        Gp, _ = dg.duality_gap(sub, xp, lam)
        assert fp >= f_star - 1e-12*abs(f_star)
        assert Gp >= fp - f_star - 1e-12*abs(fp), "the gap must bound the suboptimality"
        assert Gp > G0


def test_dual_value_is_a_lower_bound_and_checks_feasibility():
    sub = subproblem()
    x, y, z, lam, *_ = solved(sub)
    f_star = dg.exact_penalty_objective(sub, x)
    rng = np.random.default_rng(2)
    for _ in range(20):
        lam_r = rng.uniform(0, 1, lam.size)
        assert dg.dual_value(sub, lam_r) <= f_star + 1e-12*abs(f_star), "weak duality violated"
    with pytest.raises(ValueError):
        dg.dual_value(sub, -np.ones(lam.size))
    with pytest.raises(ValueError):
        dg.dual_value(sub, 2*sub['c'])                                       # above c with d = 0


def test_constraint_values_and_artificial():
    sub = subproblem(violated=True)
    x, y, z, lam, *_ = solved(sub)
    g = dg.constraint_values(sub, x)
    yopt, zopt = dg.optimal_artificial(sub, x)
    assert np.allclose(yopt, np.maximum(g, 0)) and zopt == 0.0
    assert np.allclose(y, yopt), "with d = 0 the recovered y must equal the optimal artificial variables"
    big = yopt > 1e-6                                                      # (y of the order of the residual is allowed at free multipliers)
    assert np.any(big), "the violated case must have y > 0"
    assert np.allclose(lam[big], sub['c'][big]), "lam = c where y > 0"


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
def test_diagnostics_on_interior_point_solution():
    "the same checks applied to mmapy's solution: feasible to its tolerance, gap small but larger than the dual's"
    sub = subproblem()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = mmapy.subsolv(sub['b'].size, sub['p0'].size, 1e-7, *(sub[k].reshape(-1, 1) for k in ('low', 'upp', 'alfa', 'beta', 'p0', 'q0')),
                          sub['P'], sub['Q'], sub['a0'], sub['a'].reshape(-1, 1), sub['b'].reshape(-1, 1), sub['c'].reshape(-1, 1), sub['d'].reshape(-1, 1))
    x, y, z, lam, xsi, eta, mu, zet, s = (np.asarray(v).ravel() for v in o[:9])
    chk = dg.check_solution(sub, x, y, float(z[0]), lam, xsi, eta, mu, float(zet[0]), s)
    xd, yd, zd, lamd, *_ = solved(sub)
    Gd, _ = dg.duality_gap(sub, xd, lamd)
    assert chk['viol'] <= 1e-5 and chk['stat_x'] <= 1e-5 and chk['gap'] >= 0
    assert chk['gap'] >= Gd, "the dual solution should certify at least as tightly"
