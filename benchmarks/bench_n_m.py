"""
Benchmark of mmapy.mmasub (primal-dual interior point) against mmadual.mmasub (dual method)
on random MMA subproblems: median wall time of `REP` subproblems per (n, m).

Gradients ~ N(0, 1), constraint values ~ U(-1, 1), c = 1e3, d = 0, move = 0.2. With
ESCALA = 1e-3 the objective gradient is scaled down, which makes the subproblem hard for the
interior-point method (hundreds of Newton steps per barrier level) and reproduces the second
table of the README.

Run:  python benchmarks/bench_n_m.py [ESCALA] [matched] [v012]      (requires mmapy)
      python benchmarks/bench_n_m.py matched
      python benchmarks/bench_n_m.py 1e-3 matched v012

The time of mmadual.mmasub is split into the construction of the subproblem (mma_subproblem) and its
solution (solve_dual plus the recovery of the multipliers). With the argument "v012" the 0.1.2 module kept
in benchmarks/mmadual_v012_reference.py is timed as well (same subproblems), as the reference of the
reorganized package.

When the argument "matched" is present, three accuracy measures are computed in the same way for the
solutions of both solvers (relative constraint violation, stationarity in x and relative duality gap
f_pen(x) - W(lam), see `medidas` below), and the dual solver is re-run with the loosest tol_dual, over the
decades 1e-2 ... 1e-12, whose *absolute* duality gap is no larger than that of mmapy on the same subproblem
(equal-accuracy comparison on a measure that is external to both solvers). The table then has extra columns:
the time and ratio of that run, the tolerance used, violation and stationarity for both solutions, the gap of
mmapy, of the default dual run and of the matched run, all normalized by D = max(1, |f_pen(x_du)|), and
delta = (f_pen(x_pd) - f_pen(x_du))/D, the suboptimality of the mmapy point up to the dual gap.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import mmadual
from mmadual import mmasub as mma_du
from mmadual.approximations import mma_subproblem
from mmapy import mmasub as mma_pd

V012 = None
if 'v012' in sys.argv[1:]:
    import importlib.util
    _spec = importlib.util.spec_from_file_location('mmadual_v012', os.path.join(HERE, 'mmadual_v012_reference.py'))
    V012 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V012)

REP = 5
arguments = sys.argv[1:]
MATCHED = 'matched' in arguments
escalas = [argument for argument in arguments if argument not in ('matched', 'v012')]
if len(escalas) > 1:
    raise SystemExit('Usage: bench_n_m.py [ESCALA] [matched]')
ESCALA = float(escalas[0]) if escalas else 1.0
N_LIST = [1_000, 10_000, 100_000]
M_LIST = [1, 4, 16, 64]
rng = np.random.default_rng(7)


def configuracao():
    "CPU, cores, Python, numpy, BLAS library and number of BLAS threads, for the caption of the table"
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(l.split(':', 1)[1].strip() for l in f if l.startswith('model name'))
    except (OSError, StopIteration):
        pass
    blas = 'unknown BLAS'
    try:
        import threadpoolctl
        info = [d for d in threadpoolctl.threadpool_info() if d['user_api'] == 'blas']
        if info:
            blas = ', '.join(f"{d['internal_api']} {d['version']} ({d['num_threads']} thread{'s' if d['num_threads'] > 1 else ''})" for d in info)
    except ImportError:
        blas += ' (pip install threadpoolctl to report the BLAS library and its threads)'
    return f"{cpu}, {os.cpu_count()} logical cores; {platform.system()} {platform.release()}; Python {platform.python_version()}, numpy {np.__version__}, {blas}; mmadual {mmadual.__version__}"


def medidas(o, args):
    """Accuracy measures of a subproblem solution (x, y, lam, xsi, eta as returned by the solver), computed in the
       same way for both solvers:
       viol : relative constraint violation  max_i max(0, g_i(x) - y_i - b_i) / max(1, |b_i|)  (a constraint satisfied
              with slack is not a violation)
       stat : projected-gradient stationarity in x, max_j |x_j - proj_[alfa,beta](x_j - dpsi/dx_j)|; independent of returned bound multipliers
       gap  : absolute duality gap G = f_pen(x) - W(lam), with f_pen = f(x) + sum_i c_i max(0, g_i - b_i) (exact
              penalty, d = 0) and W the dual function at the returned lam (checked to be dual feasible), evaluated at
              the exact minimizer x(lam) of the Lagrangian: W(lam) <= f* <= f_pen(x). The absolute gap is used for
              the accuracy matching; in the table all gaps and the objective difference are normalized by the same
              D = max(1, |f_pen(x_du)|) of the default dual run, so that the printed values satisfy
              max(0, delta) <= (f_pen(x_pd) - f*)/D <= delta + gamma_du  with delta = (f_pen(x_pd) - f_pen(x_du))/D and
              gamma_du = G_du/D."""
    m, n, _, xval, xmin, xmax = args[:6]
    low, upp, x, y, lam, xsi, eta, c = o[9], o[10], o[0], o[1], o[3], o[4], o[5], args[16]
    eeen, eeem = np.ones((n, 1)), np.ones((m, 1))
    xmami_inv = eeen/np.maximum(xmax - xmin, 1e-5*eeen)
    ux1, xl1 = upp - xval, xval - low
    df0dx, dfdx, fval = args[9], args[11], args[10]
    p0, q0 = np.maximum(df0dx, 0), np.maximum(-df0dx, 0)
    pq0 = 0.001*(p0 + q0) + 1e-5*xmami_inv
    p0, q0 = (p0 + pq0)*ux1**2, (q0 + pq0)*xl1**2
    P, Q = np.maximum(dfdx, 0), np.maximum(-dfdx, 0)
    PQ = 0.001*(P + Q) + 1e-5*np.dot(eeem, xmami_inv.T)
    P, Q = (P + PQ)*(ux1**2).T, (Q + PQ)*(xl1**2).T
    b = P @ (eeen/ux1) + Q @ (eeen/xl1) - fval
    g = P @ (eeen/(upp - x)) + Q @ (eeen/(x - low)) - b
    escala = np.maximum(1, np.abs(b))
    viol = float(np.max(np.maximum(g - y, 0)/escala))
    f_pen = float(np.sum(p0/(upp - x) + q0/(x - low)) + np.sum(c*np.maximum(g, 0)))
    d = args[17]
    if np.any(lam < -1e-12*c) or np.any((d == 0) & (lam > c*(1 + 1e-12))):
        raise RuntimeError('returned lam is not dual feasible (lam < 0 or lam > c with d = 0): W(lam) is not a valid bound')
    lam = np.clip(lam, 0, np.where(d > 0, np.inf, c))
    pl, ql = p0 + P.T @ lam, q0 + Q.T @ lam
    gradx = pl/(upp - x)**2 - ql/(x - low)**2
    alfa = np.maximum(np.maximum(low + 0.1*(xval - low), xval - 0.2*(xmax - xmin)), xmin)
    beta = np.minimum(np.minimum(upp - 0.1*(upp - xval), xval + 0.2*(xmax - xmin)), xmax)
    stat = float(np.max(np.abs(x - np.clip(x - gradx, alfa, beta))))
    sp, sq = np.sqrt(pl), np.sqrt(ql)
    xl_ = np.clip((sp*low + sq*upp)/(sp + sq), alfa, beta)
    W = float(np.sum(pl/(upp - xl_) + ql/(xl_ - low)) - (lam.T @ b).item())
    abs_gap = f_pen - W
    if abs_gap < -1e-9*max(1.0, abs(f_pen), abs(W)):
        raise RuntimeError(f'Negative primal-dual gap {abs_gap:g}: check dual feasibility and reconstruction')
    abs_gap = max(0.0, abs_gap)  # discard only insignificant negative round-off
    return viol, stat, abs_gap, f_pen

print(configuracao())
print(f"objective gradient scale {ESCALA:g}; median of {REP} subproblems"
      + ("; matched: dual run with the loosest tol_dual (decades from 1e-2) whose ABSOLUTE duality gap is <= that of mmapy" if MATCHED else ""))
cab = f"{'n':>7} | {'m':>3} | {'mmapy [s]':>10} | {'build [s]':>9} {'solve [s]':>9} {'mmadual [s]':>11} | {'ratio':>6}" + (f" | {'v0.1.2 [s]':>10}" if V012 else "")
if MATCHED:
    cab += f" | {'matched [s]':>11} | {'ratio':>6} | {'tol':>7} | {'viol pd':>8} {'viol du':>8} | {'stat pd':>8} {'stat du':>8} | {'gap pd':>8} {'gap du':>8} {'gap mat':>8} | {'delta':>8}"
print(cab)
for n in N_LIST:
    for m in M_LIST:
        t1s, t2s, t3s, tols, med1, med2, med3, dfs, tbs, tvs = [], [], [], [], [], [], [], [], [], []
        for _ in range(REP):
            xval = rng.uniform(0.05, 0.95, (n, 1))
            xold1 = np.clip(xval + 0.05*rng.standard_normal((n, 1)), 0, 1)
            xold2 = np.clip(xold1 + 0.05*rng.standard_normal((n, 1)), 0, 1)
            args = (m, n, 5, xval, np.zeros((n, 1)), np.ones((n, 1)), xold1, xold2, 1.0,
                    ESCALA*rng.standard_normal((n, 1)), rng.uniform(-1, 1, (m, 1)), rng.standard_normal((m, n)),
                    xval - 0.3, xval + 0.3, 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1)))
            t = time.perf_counter(); o1 = mma_pd(*args, move=0.2); t1s.append(time.perf_counter() - t)
            t = time.perf_counter(); o2 = mma_du(*args, move=0.2); t2s.append(time.perf_counter() - t)
            t = time.perf_counter(); mma_subproblem(5, xval, np.zeros(n), np.ones(n), xold1, xold2, args[9], args[10], args[11], xval - 0.3, xval + 0.3, 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m), move=0.2); tbs.append(time.perf_counter() - t)
            if V012 is not None:
                t = time.perf_counter(); V012.mmasub(*args, move=0.2); tvs.append(time.perf_counter() - t)
            if MATCHED:
                m1 = medidas(o1, args)
                m2 = medidas(o2, args)
                D = max(1.0, abs(m2[3]))                                    # common normalization
                med1.append((m1[0], m1[1], m1[2]/D)); med2.append((m2[0], m2[1], m2[2]/D))
                dfs.append((m1[3] - m2[3])/D)                               # delta: suboptimality of the mmapy point
                found_match = False
                for k in range(2, 13):                                      # loosest tolerance that matches the absolute gap of mmapy
                    tol = 10.0**(-k)
                    t = time.perf_counter(); o3 = mma_du(*args, move=0.2, tol_dual=tol); dt = time.perf_counter() - t
                    m3 = medidas(o3, args)
                    if m3[2] <= m1[2]:
                        found_match = True
                        break
                if not found_match:
                    raise RuntimeError(f'No matched absolute gap found (n={n}, m={m}, scale={ESCALA:g}): '
                                       f'dual={m3[2]:g}, mmapy={m1[2]:g}')
                t3s.append(dt); tols.append(tol); med3.append((m3[0], m3[1], m3[2]/D))
        t1, t2, tb = np.median(t1s), np.median(t2s), np.median(tbs)
        linha = f"{n:7d} | {m:3d} | {t1:10.4f} | {tb:9.4f} {t2 - tb:9.4f} {t2:11.4f} | {t1/t2:6.1f}" + (f" | {np.median(tvs):10.4f}" if V012 else "")
        if MATCHED:
            t3 = np.median(t3s); a, bd, bm = np.median(med1, axis=0), np.median(med2, axis=0), np.median(med3, axis=0)
            linha += f" | {t3:11.4f} | {t1/t3:6.1f} | {np.median(tols):7.0e} | {a[0]:8.1e} {bd[0]:8.1e} | {a[1]:8.1e} {bd[1]:8.1e} | {a[2]:8.1e} {bd[2]:8.1e} {bm[2]:8.1e} | {np.median(dfs):8.1e}"
        print(linha)
