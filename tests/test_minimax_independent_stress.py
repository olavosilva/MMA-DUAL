"""Independent minimax stress/regression tests (a != 0), with scipy.optimize.linprog (HiGHS) as the reference.

The subproblems have strictly positive p0, q0 and nonnegative P, Q, but every x_j is held at its upper bound
(p0 << q0), so the reduced dual function is affine up to ~1e-10 and its maximizer on the box intersected with
the half-space is that of a linear program. The generator and the three cases 22, 118, 278 of seed 5291 were
provided by the second review of 0.2.0: before the active-set iteration of the polytope path, case 22 reached
the optimal dual value but a multiplier 4e-13 below its upper bound was classified as free (wrong z, KKT
complementarity 0.016), and cases 118 and 278 stalled a small distance inside the half-space (z = 0, dual
value errors 8e-3 and 3e-2 after 500 Newton steps).

Run with the package installed (`pip install -e ".[test]"`) or from the repository (`python -m pytest -q tests`).
"""
import os
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mmadual import solve_dual, recover_multipliers, diagnostics


def affine_case(rng):
    "one subproblem of the review's generator from the given generator state"
    m = int(rng.integers(2, 9))
    slopes = rng.uniform(-2, 3, m)
    a = rng.uniform(0, 2, m)
    a[rng.random(m) < .25] = 0
    if not np.max(a):
        a[0] = 1
    c = rng.uniform(.05, 2, m)
    a0 = rng.uniform(.2, 2)
    n = 7
    low, upp = -np.ones(n), 2*np.ones(n)
    alfa, beta = -.5*np.ones(n), 1.5*np.ones(n)
    p0, q0 = 1e-6*np.ones(n), np.ones(n)
    P, Q = 1e-11*np.ones((m, n)), 1e-11*np.ones((m, n))
    b = P @ (1/(upp-beta)) + Q @ (1/(beta-low)) - slopes
    sub = dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0,
               P=P, Q=Q, b=b, a=a, a0=a0, c=c, d=np.zeros(m))
    return sub, slopes


def sample_from_seed(index, seed=5291):
    "case `index` of the review's sequence (the generator is advanced index + 1 times)"
    rng = np.random.default_rng(seed)
    for k in range(index + 1):
        sub, slopes = affine_case(rng)
    return sub, slopes


def solve_and_check(sub, slopes, max_newton=100):
    "dual solution, LP reference value, recovered multipliers and the KKT/gap summary"
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)                     # no accuracy warning allowed
        x, y, z, lam, info = solve_dual(sub, tol=1e-10, max_newton=max_newton)
    lp = linprog(-slopes, A_ub=sub['a'][None, :], b_ub=[sub['a0']],
                 bounds=[(0, cap) for cap in sub['c']], method='highs')
    assert lp.success, lp.message
    # independent LP optimum of the affine dual (the difference due to P, Q is of order 1e-10)
    value_gap = diagnostics.dual_value(sub, lp.x) - diagnostics.dual_value(sub, lam)
    xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
    chk = diagnostics.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)
    return info, value_gap, chk


@pytest.mark.parametrize('case', [22, 118, 278])
def test_minimax_against_independent_linear_program(case):
    "the three cases of the review, with the review's acceptance thresholds"
    sub, slopes = sample_from_seed(case)
    info, value_gap, chk = solve_and_check(sub, slopes, max_newton=500)
    assert info['converged'], (case, info)
    assert value_gap <= 1e-6, (case, 'dual optimality error', value_gap)
    assert chk['comp'] < 1e-6, (case, 'KKT complementarity', chk)
    assert chk['viol'] < 1e-6, (case, 'primal feasibility', chk)
    assert chk['gap_returned_rel'] < 1e-6, (case, 'returned-point gap', chk)


@pytest.mark.parametrize('seed', [5291, 1, 2])
def test_affine_dual_sweep(seed):
    """200 cases per seed with the default max_newton = 100: convergence certified, LP value matched to 1e-9,
       KKT of the returned point to 1e-9, in a small number of Newton steps (dual_value checks dual feasibility)"""
    rng = np.random.default_rng(seed)
    steps = []
    for k in range(200):
        sub, slopes = affine_case(rng)
        info, value_gap, chk = solve_and_check(sub, slopes)
        assert info['converged'] and info['certified'], (seed, k, info)
        assert abs(value_gap) <= 1e-9, (seed, k, value_gap)
        assert chk['comp'] <= 1e-9 and chk['viol'] <= 1e-9 and chk['gap_returned_rel'] <= 1e-9, (seed, k, chk)
        steps.append(info['newton_steps'])
    assert max(steps) <= 20, max(steps)                                     # observed: at most 10 over 2100 cases


def test_vertex_of_box_and_halfspace_with_tolerance_classification():
    """a multiplier placed 4e-13 below its upper bound (as the projection can leave it) must count as bounded:
       the half-space multiplier z and the complementarity of the returned point do not depend on it"""
    sub, slopes = sample_from_seed(22)
    info, value_gap, chk = solve_and_check(sub, slopes)
    lam_opt = linprog(-slopes, A_ub=sub['a'][None, :], b_ub=[sub['a0']],
                      bounds=[(0, cap) for cap in sub['c']], method='highs').x
    assert info['halfspace_active'] and info['z'] > 0.3 and abs(value_gap) <= 1e-10
    assert chk['comp'] <= 1e-12 and chk['gap_returned_rel'] <= 1e-12, chk
    assert np.sum(np.isclose(lam_opt, sub['c']) | np.isclose(lam_opt, 0.0)) >= sub['b'].size - 1   # a vertex: box and half-space active
