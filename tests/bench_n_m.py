"""
Benchmark of mmapy.mmasub (primal-dual interior point) against mma_dual.mmasub (dual method)
on random MMA subproblems: median wall time of `REP` subproblems per (n, m).

Gradients ~ N(0, 1), constraint values ~ U(-1, 1), c = 1e3, d = 0, move = 0.2. With
ESCALA = 1e-3 the objective gradient is scaled down, which makes the subproblem hard for the
interior-point method (hundreds of Newton steps per barrier level) and reproduces the second
table of the README.

Run:  python benchmarks/bench_n_m.py [ESCALA]      (requires mmapy)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mma_dual import mmasub as mma_du
from mmapy import mmasub as mma_pd

REP = 5
ESCALA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
N_LIST = [1_000, 10_000, 100_000]
M_LIST = [1, 4, 16, 64]
rng = np.random.default_rng(7)

print(f"objective gradient scale {ESCALA:g}; median of {REP} subproblems")
print(f"{'n':>7} | {'m':>3} | {'mmapy [s]':>10} | {'mma_dual [s]':>12} | {'ratio':>6}")
for n in N_LIST:
    for m in M_LIST:
        t1s, t2s = [], []
        for _ in range(REP):
            xval = rng.uniform(0.05, 0.95, (n, 1))
            xold1 = np.clip(xval + 0.05*rng.standard_normal((n, 1)), 0, 1)
            xold2 = np.clip(xold1 + 0.05*rng.standard_normal((n, 1)), 0, 1)
            args = (m, n, 5, xval, np.zeros((n, 1)), np.ones((n, 1)), xold1, xold2, 1.0,
                    ESCALA*rng.standard_normal((n, 1)), rng.uniform(-1, 1, (m, 1)), rng.standard_normal((m, n)),
                    xval - 0.3, xval + 0.3, 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1)))
            t = time.perf_counter(); mma_pd(*args, move=0.2); t1s.append(time.perf_counter() - t)
            t = time.perf_counter(); mma_du(*args, move=0.2); t2s.append(time.perf_counter() - t)
        t1, t2 = np.median(t1s), np.median(t2s)
        print(f"{n:7d} | {m:3d} | {t1:10.4f} | {t2:12.4f} | {t1/t2:6.1f}")
