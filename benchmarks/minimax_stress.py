"""
Stress test of the dual solver on subproblems whose dual function is affine to ~1e-10 (every x_j held at
its upper bound), with the linear program solved by scipy.optimize.linprog (HiGHS) as the reference.
The generator is the one of the second review of 0.2.0 (seed 5291: cases 22, 118, 278 failed before the
active-set iteration of the a != 0 path).

Run:  python benchmarks/minimax_stress.py [SEED] [CASES] [box]

    SEED   generator seed (default 5291), CASES number of subproblems (default 350)
    box    set a = 0 (the iteration on the box): the optimum is then lam_i = c_i for a positive slope and
           0 otherwise, reached in one step since the multipliers without curvature step to their bound
           (0.1.2 took unit gradient steps and cases 87, 133, 205 of seed 5291 reached max_newton)

Reported: failures (not converged, LP value gap > 1e-6, KKT complementarity, violation or returned gap
> 1e-6), Newton steps and evaluations, worst residuals.
"""
import sys
import time
import warnings
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/src')
from mmadual import solve_dual, recover_multipliers, diagnostics


def affine_case(rng, box=False):
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
    b = P @ (1/(upp - beta)) + Q @ (1/(beta - low)) - slopes
    if box:
        a = np.zeros(m)
    return dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, P=P, Q=Q, b=b, a=a, a0=a0, c=c, d=np.zeros(m)), slopes


def main(seed=5291, cases=350, box=False):
    rng = np.random.default_rng(seed)
    fails, steps, evals = [], [], []
    worst = dict(gap=0.0, comp=0.0, viol=0.0, gret=0.0)
    t0 = time.perf_counter()
    for k in range(cases):
        sub, slopes = affine_case(rng, box)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x, y, z, lam, info = solve_dual(sub, tol=1e-10, max_newton=100)
        lp = linprog(-slopes, A_ub=sub['a'][None, :], b_ub=[sub['a0']], bounds=[(0, cap) for cap in sub['c']], method='highs')
        vgap = diagnostics.dual_value(sub, lp.x) - diagnostics.dual_value(sub, lam)
        xsi, eta, mu, zet, s, y2 = recover_multipliers(sub, x, y, z, lam)
        chk = diagnostics.check_solution(sub, x, y2, z, lam, xsi, eta, mu, zet, s)
        steps.append(info['newton_steps'])
        evals.append(info['evaluations'])
        worst['gap'] = max(worst['gap'], vgap)
        worst['comp'] = max(worst['comp'], chk['comp'])
        worst['viol'] = max(worst['viol'], chk['viol'])
        worst['gret'] = max(worst['gret'], chk['gap_returned_rel'])
        if not info['converged'] or vgap > 1e-6 or chk['comp'] > 1e-6 or chk['viol'] > 1e-6 or chk['gap_returned_rel'] > 1e-6:
            fails.append((k, info['reason'], info['newton_steps'], f"{vgap:.1e}", f"{chk['comp']:.1e}", f"{chk['gap_returned_rel']:.1e}"))
    print(f"seed {seed}, {cases} cases{' (a = 0)' if box else ''}: {len(fails)} failures, {time.perf_counter() - t0:.1f} s")
    print(f"  Newton steps: median {np.median(steps):.0f}, max {max(steps)}; evaluations: median {np.median(evals):.0f}, max {max(evals)}")
    print(f"  worst: LP value gap {worst['gap']:.1e}, complementarity {worst['comp']:.1e}, violation {worst['viol']:.1e}, "
          f"returned gap (relative) {worst['gret']:.1e}")
    for f in fails[:10]:
        print('  FAIL (case, reason, steps, LP gap, complementarity, returned gap):', f)


if __name__ == '__main__':
    args = [v for v in sys.argv[1:] if v != 'box']
    main(int(args[0]) if args else 5291, int(args[1]) if len(args) > 1 else 350, box='box' in sys.argv[1:])
