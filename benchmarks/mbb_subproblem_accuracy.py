"""
Accuracy of the interior-point subproblem solver of mmapy as a function of its barrier tolerance epsimin,
on one fixed MMA subproblem of the MBB beam, against the dual solver.

The MBB optimization of examples/mbb_topopt.py is run for `k` iterations with the dual solver, and the MMA
subproblem of iteration k + 1 is built with mmadual.mma_subproblem. That same subproblem is then solved by
mmapy.subsolv with epsimin = 1e-7 (the value hard-wired in mmapy.mmasub), 1e-9 and 1e-11, and by the dual
solver. For each solution the script reports the median density returned in the elements that the dual
solution puts at the lower bound of the subproblem (the "void" region, x_j = alfa_j), the exact-penalty
objective, the duality gap, the largest deviation from the dual solution and the time.

The interior-point iterates stay at a distance of the order of epsimin/xi_j from a bound with multiplier
xi_j, so weakly active bounds are resolved only to that distance; the dual solver returns the bound exactly.
This is a check of the accuracy of the subproblem solution, not a statement about the local optimum reached
by the complete optimization, which depends on the whole sequence of iterations.

Run:  python benchmarks/mbb_subproblem_accuracy.py [nelx] [nely] [k]       (default 120 x 40, k = 30; requires mmapy)
"""
import os
import sys
import time
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'examples'))
import mmadual
from mmadual import mma_subproblem, solve_dual, diagnostics as dg
import mmapy
import mbb_topopt as mbb

NELX = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NELY = int(sys.argv[2]) if len(sys.argv) > 2 else 40
K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
RMIN = 2.4


def subproblem_at_iteration(nelx, nely, k, rmin):
    "runs k MMA iterations of the MBB problem with the dual solver and returns the subproblem of iteration k + 1"
    nel, edof, iK, jK = mbb.malha(nelx, nely)
    ndof = 2*(nelx + 1)*(nely + 1)
    KE = mbb.elemento_q4(mbb.E0, mbb.NU)
    H, Hs = mbb.filtro(nelx, nely, rmin)
    F = np.zeros(ndof); F[1] = -1.0
    fixos = np.union1d(np.arange(0, 2*(nely + 1), 2), [2*(nelx + 1)*(nely + 1) - 1])
    livres = np.setdiff1d(np.arange(ndof), fixos)
    m, n = 1, nel
    x = mbb.VOLFRAC*np.ones(n); xold1, xold2 = x.copy(), x.copy()
    low, upp = np.zeros(n), np.ones(n)
    c_ref = None
    for it in range(1, k + 2):
        xPhys = (H @ x)/Hs
        U = mbb.monta_e_resolve(xPhys, KE, iK, jK, ndof, livres, F, {'fe': 0.0})
        ce = np.einsum('ij,jk,ik->i', U[edof], KE, U[edof])
        comp = np.sum((mbb.EMIN + xPhys**mbb.PENAL*(mbb.E0 - mbb.EMIN))*ce)
        dc = H.T @ ((-mbb.PENAL*xPhys**(mbb.PENAL - 1)*(mbb.E0 - mbb.EMIN)*ce)/Hs)
        dv = H.T @ (np.ones(n)/Hs)
        c_ref = comp if c_ref is None else c_ref
        fval = np.array([np.sum(xPhys)/(mbb.VOLFRAC*nel) - 1.0])
        sub = mma_subproblem(it, x, np.zeros(n), np.ones(n), xold1, xold2, dc/c_ref, fval, (dv/(mbb.VOLFRAC*nel)).reshape(1, -1),
                             low, upp, 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m), move=mbb.MOVE)
        if it == k + 1:
            return sub, comp
        xnew, y, z, lam, info = solve_dual(sub)
        low, upp = sub['low'], sub['upp']
        xold2, xold1, x = xold1, x, xnew


if __name__ == '__main__':
    print(mbb.configuracao())
    sub, comp = subproblem_at_iteration(NELX, NELY, K, RMIN)
    n, m = sub['p0'].size, sub['b'].size
    print(f"MBB {NELX} x {NELY} ({n} elements), subproblem of iteration {K + 1} (compliance {comp:.3f} at iteration {K}), m = {m}")
    t = time.perf_counter(); xd, yd, zd, lamd, info = solve_dual(sub); td = time.perf_counter() - t
    void = xd <= sub['alfa'] + 1e-12                                        # elements at the lower bound in the dual solution
    solid = xd >= sub['beta'] - 1e-12
    print(f"dual solution: {void.sum()} elements at the lower bound, {solid.sum()} at the upper bound, {n - void.sum() - solid.sum()} free; "
          f"{info['newton_steps']} Newton steps, residual {info['residual']:.1e}, {td*1e3:.1f} ms")
    fd, Gd = dg.exact_penalty_objective(sub, xd), dg.duality_gap(sub, xd, lamd)[0]
    print(f"\n{'solver':24s} | {'median x in void':>16} | {'max x in void':>13} | {'f_pen':>14} | {'gap':>9} | {'max |x - x_dual|':>16} | {'time [ms]':>9}")
    print(f"{'dual (tol 1e-10)':24s} | {np.median(xd[void] - sub['alfa'][void]):16.2e} | {np.max(xd[void] - sub['alfa'][void]):13.2e} | {fd:14.8f} | {Gd:9.1e} | {0.0:16.1e} | {td*1e3:9.1f}")
    col = lambda v: v.reshape(-1, 1)
    for eps in (1e-7, 1e-9, 1e-11):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = time.perf_counter()
            o = mmapy.subsolv(m, n, eps, col(sub['low']), col(sub['upp']), col(sub['alfa']), col(sub['beta']), col(sub['p0']), col(sub['q0']),
                              sub['P'], sub['Q'], sub['a0'], col(sub['a']), col(sub['b']), col(sub['c']), col(sub['d']))
            tp = time.perf_counter() - t
        xp, lamp = o[0].ravel(), o[3].ravel()
        fp, Gp = dg.exact_penalty_objective(sub, xp), dg.duality_gap(sub, xp, lamp)[0]
        print(f"{'mmapy epsimin ' + f'{eps:.0e}':24s} | {np.median(xp[void] - sub['alfa'][void]):16.2e} | {np.max(xp[void] - sub['alfa'][void]):13.2e} | {fp:14.8f} | {Gp:9.1e} | {np.abs(xp - xd).max():16.1e} | {tp*1e3:9.1f}")
    print("\n(x in void: distance of the returned density to the lower bound alfa_j of the subproblem, over the elements that the dual puts at that bound)")
