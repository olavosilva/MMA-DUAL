# Benchmark campaign of version 0.2.0: protocol, environment and results

This document records the subproblem benchmark and the topology-optimization runs that the README
summarizes. They were made for the manuscript that accompanies the package; the scripts
(`bench_campaign.py`, `make_tables_figures.py`, `frozen_subproblems.py`, `mbb_topopt.py`,
`topopt_snapshot.py`, with a shared `common.py`), the per-run results (`runs.csv`, one line per
solver call with its diagnostics), the frozen subproblems (`.npz`) and the environment metadata
(`metadata.json`) are in the reproduction archive of the manuscript, not in this repository. The
repository scripts `benchmarks/bench_n_m.py` and `examples/mbb_topopt.py` are lighter versions of the
same measurements; the timing tables of the 0.1.x README, made with `bench_n_m.py` on a shared virtual
machine, are superseded by the campaign below and are not mixed with it.

## 1. Software under test and provenance

- MMA-DUAL 0.2.0, tag `v0.2.0`, commit `42b434c`; `src/mmadual/` is unchanged in the later commits
  of `main` (documentation only). The campaign imported the installed package, not a working copy.
- `mmapy` 0.3.1 (GCMMA-MMA-Python), the reference, used through its public `mmasub` and `subsolv`.
- No source of either package was modified. The barrier parameter of `mmapy.subsolv` is the
  `epsimin` argument of that function; `mmapy.mmasub` fixes it at $10^{-7}$.

## 2. Subproblem benchmark

### Generator

Random MMA subproblems built by the MMA construction at iteration 5 with asymptotes at
$x^{(k)} \pm 0.3$: $x^{(k)}_j$ uniform in $[0.05, 0.95]$; the two previous designs obtained by Gaussian
perturbations of standard deviation 0.05 clipped to $[0, 1]$; $\partial f_0/\partial x_j$ standard
Gaussian multiplied by a scale factor $\sigma$; $f_i(x^{(k)})$ uniform in $[-1, 1]$;
$\partial f_i/\partial x_j$ standard Gaussian; move limit 0.2; $a = 0$, $a_0 = 1$, $c_i = 10^3$,
$d_i = 0$. $\sigma = 1$ gives regularly scaled subproblems; $\sigma = 10^{-3}$ makes the objective
coefficients small against the constraint coefficients, a situation in which `mmapy.subsolv` reaches
its iteration limit. Each subproblem is a deterministic function of $(\sigma, n, m)$ and of the seed
$1000 k + r$ (configuration $k$, repetition $r$).

| $k$ | $\sigma$ | $n$ | $m$ | repetitions |
|---|---|---|---|---|
| 1–2 | 1 | $10^3$ | 4, 16 | 10 |
| 3–5 | 1 | $10^4$ | 4, 16, 64 | 10 |
| 6–8 | 1 | $10^5$ | 4, 16, 64 | 10 |
| 9–10 | $10^{-3}$ | $10^4$ | 4, 16 | 5 |
| 11–12 | $10^{-3}$ | $10^5$ | 4, 16 | 5 |
| 13 | $10^{-3}$ | $10^3$ | 64 | 5 |

### Measurements

Three quantities are distinguished and must not be confused:

1. **Complete call**: the time of `mmasub` of each package on the same inputs, including the
   construction of the approximation and, for MMA-DUAL, the recovery of the multipliers.
2. **Frozen subproblem**: the coefficient arrays built by MMA-DUAL (identical to `mmapy`'s to
   rounding) are passed to `mmapy.subsolv` and to `solve_dual`; the construction time of MMA-DUAL
   is measured separately. The dual solver is then rerun on the same coefficients with tolerances
   $10^{-2}, \ldots, 10^{-14}$, and `mmapy.subsolv` with `epsimin` $10^{-9}$ and $10^{-11}$ in
   configurations 3, 4, 9 and 10.
3. **Matched gap**: for every subproblem, the loosest dual tolerance whose absolute gap
   $\mathcal G_x$ does not exceed that of `mmapy.subsolv` at `epsimin` $= 10^{-7}$. The MMA-DUAL
   time at that tolerance is *estimated* as construction time plus frozen solution time, measured
   separately; it excludes the multiplier recovery that `mmasub` performs (at the default tolerance
   the median estimate is within 8 % of the median measured complete call in every configuration).
   Matching means "gap not larger than"; the two gaps are not equal.

The order of the two solvers alternates between repetitions, three solves of each precede the first
measurement, and each call is timed with `time.perf_counter`. The same diagnostics
(`mmadual.diagnostics`, which use only the subproblem equations) are applied to both solutions:
projected stationarity in $x$, violation of the relaxed and of the unrelaxed constraints,
complementarity, both gaps, objectives. Gaps are reported divided by the common
$D = \max(1, |F_{\rm du}|)$ of the default dual run on the same subproblem ($\gamma_{\rm pd}$,
$\gamma_{\rm du}$); $\delta$ is the exact-penalty objective of the `mmapy` point minus that of the
dual point, divided by $D$. Medians over the repetitions; ratios are formed per paired subproblem
before the median; interquartile ranges in brackets.

### Environment

Workstation with an AMD Ryzen 9 5950X (16 cores, 32 logical processors), Windows 11, Python 3.12.9,
NumPy 2.5.4 with OpenBLAS 0.3.34, SciPy 1.18.1 with OpenBLAS 0.3.31, both BLAS libraries restricted
to one thread (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS` set before NumPy is
imported and verified with `threadpoolctl` at the start of the run; the process was not pinned to a
core). Campaign of 2026-10-30.

### Times and ratios

| $k$ | $\sigma$ | $n$ | $m$ | `mmapy` complete call [s] | MMA-DUAL complete call [s] | MMA-DUAL build [s] | MMA-DUAL solve [s] | `mmapy.subsolv` [s] | ratio, complete call | ratio, frozen | Newton steps | evaluations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 000 | 4 | 0.0149 | 0.0026 | 0.0001 | 0.0025 | 0.0147 | 5.6 [5.2, 5.7] | 5.8 | 8 | 13 |
| 2 | 1 | 1 000 | 16 | 0.0137 | 0.0037 | 0.0002 | 0.0033 | 0.0136 | 3.6 [3.3, 4.2] | 3.9 | 9 | 16 |
| 3 | 1 | 10 000 | 4 | 0.0488 | 0.0088 | 0.0011 | 0.0079 | 0.0443 | 5.5 [5.3, 5.9] | 5.5 | 9 | 10 |
| 4 | 1 | 10 000 | 16 | 0.0871 | 0.0279 | 0.0036 | 0.0244 | 0.0966 | 3.2 [3.1, 3.3] | 4.0 | 9 | 10 |
| 5 | 1 | 10 000 | 64 | 0.279 | 0.114 | 0.014 | 0.103 | 0.308 | 2.4 [2.3, 2.6] | 3.1 | 9 | 16.5 |
| 6 | 1 | 100 000 | 4 | 1.23 | 0.107 | 0.013 | 0.096 | 1.20 | 11.2 [11.0, 11.4] | 12.4 | 9 | 10 |
| 7 | 1 | 100 000 | 16 | 1.30 | 0.316 | 0.041 | 0.274 | 1.45 | 4.1 [4.1, 4.2] | 5.3 | 10 | 11 |
| 8 | 1 | 100 000 | 64 | 3.83 | 1.33 | 0.15 | 1.15 | 4.29 | 2.9 [2.8, 3.1] | 3.7 | 10.5 | 11.5 |
| 9 | $10^{-3}$ | 10 000 | 4 | 2.22 | 0.0079 | 0.0009 | 0.0069 | 1.91 | 280 [278, 292] | 279 | 8 | 9 |
| 10 | $10^{-3}$ | 10 000 | 16 | 3.52 | 0.0294 | 0.0039 | 0.0263 | 3.96 | 119 [118, 120] | 151 | 10 | 11 |
| 11 | $10^{-3}$ | 100 000 | 4 | 34.8 | 0.105 | 0.013 | 0.089 | 32.7 | 332 [332, 337] | 367 | 8 | 9 |
| 12 | $10^{-3}$ | 100 000 | 16 | 62.3 | 0.335 | 0.040 | 0.288 | 62.6 | 187 [183, 190] | 218 | 10 | 11 |
| 13 | $10^{-3}$ | 1 000 | 64 | 0.448 | 0.0113 | 0.0007 | 0.0098 | 0.441 | 41.2 [38.4, 41.3] | 45.5 | 11 | 12 |

The construction of the approximation is 5 % of the MMA-DUAL complete call at $n = 10^3$ and 11 to
13 % at $n \ge 10^4$. The dual solver takes 8 to 11 Newton steps in every configuration; its time grows
linearly with $n$ at fixed $m$ and faster than linearly with $m$ (the Hessian costs $O(nm^2)$). For
$m < n$, `mmapy.subsolv` also assembles an $m \times m$ condensed system in $O(nm^2)$ and solves it in
$O(m^3)$, so the decrease of the ratio with $m$ is an observation about the constants and iteration
counts of the two implementations, not a consequence of different asymptotic orders.

### Accuracy and matched gap

| $k$ | $\gamma_{\rm pd}$ | $\gamma_{\rm du}$ | stat. $x$ `mmapy` | stat. $x$ MMA-DUAL | $\delta$ | tol matched, median (range) | $\gamma$ at matched tol | ratio, matched gap | $\gamma_{\rm du}$ at tol $10^{-12}$ (max) | $\gamma_{\rm du}$ at tol $10^{-14}$ | `mmapy` reference |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | $8.9\times10^{-8}$ | $6.4\times10^{-12}$ | $2\times10^{-4}$ | $6\times10^{-15}$ | $8.9\times10^{-8}$ | $10^{-9}$ ($10^{-7}$ to $10^{-11}$) | $2.2\times10^{-12}$ | 5.9 [5.7, 6.1] | $2.2\times10^{-12}$ ($1.1\times10^{-9}$) | $5.7\times10^{-13}$ | converged |
| 2 | $1.6\times10^{-8}$ | $2.5\times10^{-12}$ | $6\times10^{-5}$ | $8\times10^{-15}$ | $1.3\times10^{-8}$ | $10^{-10}$ ($10^{-9}$ to $10^{-12}$) | $1.5\times10^{-12}$ | 3.8 [3.4, 4.0] | $1.5\times10^{-12}$ ($3.2\times10^{-12}$) | $1.5\times10^{-12}$ | converged |
| 3 | $9.0\times10^{-8}$ | $3.2\times10^{-11}$ | $3\times10^{-4}$ | $1\times10^{-14}$ | $8.8\times10^{-8}$ | $10^{-8}$ ($10^{-7}$ to $10^{-11}$) | $1.8\times10^{-11}$ | 6.0 [5.8, 6.2] | $1.8\times10^{-12}$ ($1.0\times10^{-9}$) | $9.1\times10^{-13}$ | converged |
| 4 | $9.2\times10^{-9}$ | $7.3\times10^{-11}$ | $3\times10^{-4}$ | $1\times10^{-14}$ | $8.8\times10^{-9}$ | $10^{-9}$ ($10^{-8}$ to $10^{-11}$) | $2.0\times10^{-11}$ | 3.1 [3.0, 3.3] | $2.0\times10^{-11}$ ($7.2\times10^{-10}$) | $2.0\times10^{-12}$ | converged |
| 5 | $3.7\times10^{-9}$ | $1.3\times10^{-10}$ | $1\times10^{-4}$ | $2\times10^{-14}$ | $3.5\times10^{-9}$ | $10^{-10}$ ($10^{-9}$ to $10^{-13}$) | $4.8\times10^{-12}$ | 2.4 [2.3, 2.5] | $4.9\times10^{-12}$ ($2.1\times10^{-8}$) | $4.8\times10^{-12}$ | converged |
| 6 | $9.1\times10^{-8}$ | $1.0\times10^{-9}$ | $3\times10^{-4}$ | $2\times10^{-14}$ | $9.0\times10^{-8}$ | $10^{-8}$ ($10^{-7}$ to $10^{-8}$) | $1.0\times10^{-9}$ | 11.2 [10.9, 11.4] | $5.8\times10^{-10}$ ($2.1\times10^{-9}$) | $5.3\times10^{-13}$ | converged |
| 7 | $8.3\times10^{-9}$ | $7.2\times10^{-7}$ | $3\times10^{-4}$ | $2\times10^{-14}$ | $-7.1\times10^{-7}$ | $10^{-11}$ ($10^{-11}$ to $10^{-13}$) | $1.9\times10^{-12}$ | 3.8 [3.8, 3.8] | $2.1\times10^{-12}$ ($3.1\times10^{-8}$) | $1.9\times10^{-12}$ | converged |
| 8 | $1.0\times10^{-9}$ | $1.1\times10^{-8}$ | $3\times10^{-4}$ | $2\times10^{-14}$ | $-9.7\times10^{-9}$ | $10^{-11}$ ($10^{-9}$ to $10^{-14}$) | $7.0\times10^{-12}$ | 2.8 [2.8, 2.9] | $1.9\times10^{-11}$ ($1.9\times10^{-8}$) | $7.0\times10^{-12}$ | converged |
| 9 | $7.0\times10^{-2}$ | $1.8\times10^{-6}$ | $3\times10^{-3}$ | $4\times10^{-19}$ | $6.4\times10^{-2}$ | $10^{-7}$ | $1.8\times10^{-6}$ | 300 [295, 311] | $1.8\times10^{-6}$ ($3.7\times10^{-6}$) | $1.2\times10^{-16}$ | iteration limit |
| 10 | $2.1\times10^{-3}$ | $9.2\times10^{-10}$ | $2\times10^{-3}$ | $2\times10^{-18}$ | $1.9\times10^{-3}$ | $10^{-10}$ | $9.2\times10^{-10}$ | 117 [117, 119] | $9.2\times10^{-10}$ ($1.8\times10^{-9}$) | $9.2\times10^{-10}$ | iteration limit |
| 11 | $1.5\times10^{-1}$ | $2.1\times10^{-6}$ | $4\times10^{-3}$ | $9\times10^{-19}$ | $1.4\times10^{-1}$ | $10^{-7}$ | $2.1\times10^{-6}$ | 340 [338, 342] | $2.1\times10^{-6}$ ($2.8\times10^{-6}$) | $2.0\times10^{-10}$ | iteration limit |
| 12 | $6.3\times10^{-2}$ | $1.4\times10^{-9}$ | $4\times10^{-3}$ | $2\times10^{-18}$ | $5.7\times10^{-2}$ | $10^{-6}$ | $8.7\times10^{-3}$ | 213 [205, 216] | $1.4\times10^{-9}$ ($1.8\times10^{-9}$) | $1.4\times10^{-9}$ | iteration limit |
| 13 | $6.5\times10^{-6}$ | $1.4\times10^{-7}$ | $5\times10^{-5}$ | $4\times10^{-19}$ | $6.3\times10^{-6}$ | $10^{-9}$ ($10^{-8}$ to $10^{-9}$) | $1.4\times10^{-7}$ | 43.3 [43.2, 45.1] | $1.4\times10^{-7}$ ($1.7\times10^{-6}$) | $1.1\times10^{-9}$ | converged |

Both solvers returned zero violation of the relaxed constraints in every run; the dual solutions
violate the unrelaxed constraints $g_i \le b_i$ by at most $5 \times 10^{-11}$ relative, which the
auxiliary variables absorb. Every repetition was matched within the sweep.

What the two groups of configurations mean:

- **Configurations 1–8 and 13** compare converged solutions. The matched-gap ratios are within 15 %
  of the default ones, because matching changes the dual time by at most one or two Newton steps in
  either direction. At the default tolerance the dual gap is the smaller one in eleven of thirteen
  configurations and the larger one in configurations 7 and 8, where the constraint scales
  $\kappa_i$ (mean about 55 000 at $n = 10^5$) turn a projected gradient of $10^{-10}\kappa_i$ into a
  gap of order $10^{-3}$ absolute through $c_i = 10^3$. Tolerances of $10^{-12}$ to $10^{-14}$ bring
  the median gap below $10^{-11}$ in the eight regular configurations at a cost of 0 to 16 % of the
  solution time; $10^{-12}$ is not sufficient in every repetition (three of 80 keep gaps of
  $2$ to $3 \times 10^{-8}$, and configurations 9 and 11 take their extra step only at $10^{-13}$).
- **Configurations 9–12** compare a converged dual solve with a `mmapy` reference that stopped at its
  iteration limit at `epsimin` $= 10^{-7}$ (stationarity $2$ to $4 \times 10^{-3}$, gaps
  $2 \times 10^{-3}$ to $0.15$). The ratios of hundreds measure that stopping behaviour, not a gain at
  equal accuracy. With `epsimin` $= 10^{-9}$ and $10^{-11}$, `mmapy.subsolv` converges in configuration
  10 (gaps $8 \times 10^{-8}$ and $8 \times 10^{-10}$ at about 4.3 s against 4.0 s at $10^{-7}$) but
  not in configuration 9 (gap from $7 \times 10^{-2}$ to $1.5 \times 10^{-2}$, time from 1.9 to
  3.8 s); the sweep was not run for 11 and 12. In configurations 3 and 4 the `mmapy` gap decreases in
  proportion to `epsimin` ($9 \times 10^{-8}$ to $9 \times 10^{-12}$, time 44 to 53 ms at $m = 4$)
  while the dual solver spans gaps from 1 to $10^{-12}$ between 5 and 8 ms.

## 3. Frozen subproblems at $n = 1000$ (verification machine)

Eight frozen subproblems of the same generator ($n = 1000$, $m \in \{4, 16\}$, $\sigma \in \{1, 10^{-3}\}$)
were solved with `mmapy.subsolv` at `epsimin` $10^{-7}$, $10^{-9}$, $10^{-11}$ and with the dual solver
at tolerances $10^{-10}$ and $10^{-12}$ on a shared cloud virtual machine (Intel Xeon 2.8 GHz, two
logical cores, Python 3.13.18, NumPy 2.5.3, one BLAS thread; times not comparable with the campaign).
The interior-point solutions have complementarity equal to `epsimin` when the routine converges, and
gaps that decrease in proportion to it in the regular cases; in the poorly scaled $m = 4$ cases the
routine stops at its iteration limit at $10^{-7}$ and $10^{-9}$ and converges at $10^{-11}$. The dual
solutions have stationarity at rounding level and gaps that follow the residual times $c_i \kappa_i$:
up to $5.3 \times 10^{-4}$ absolute at tolerance $10^{-10}$ with $m = 16$ and $\sigma = 1$, one more
Newton step at $10^{-12}$ bringing them to $10^{-10}$. In the poorly scaled $m = 4$ case both
tolerances stop at the same iterate (residual $4.5 \times 10^{-13}$, gap $6.7 \times 10^{-7}$), and a
tolerance of $10^{-13}$ takes the further step.

On the regular $n = 1000$, $m = 4$ subproblem, the dual solution has 62 variables exactly at their
lower bounds and 91 at their upper bounds; the interior-point solutions have none at a bound, and on
the 26 coordinates with $\alpha_j = 0$ the median returned value is $1.8 \times 10^{-7}$,
$1.8 \times 10^{-9}$ and $1.8 \times 10^{-11}$ for the three barrier parameters, with
$\xi_j x_j$ equal to `epsimin`.

## 4. Topology optimization (MBB beam)

MBB beam of the 88-line code (Andreassen et al. 2011), half model of $900 \times 300$ bilinear
elements ($n = 270\,000$), SIMP with $p = 3$ and $E_{\min} = 10^{-9}$, linear density filter of radius
10 elements, volume fraction 0.5, move limit 0.2, uniform initial design, $f_0 = c/c_{\rm ref}$,
$g = V/(0.5 V_0) - 1$, $c_1 = 10^3$, $d_1 = 0$; stop at a design change below 0.05 (this criterion
does not certify stationarity of the original problem). Same workstation, PyPardiso (MKL Pardiso
2025.3) on one thread, Python 3.12.9, NumPy 2.3.4, SciPy 1.16.3, all BLAS libraries on one thread. At
every iteration the subproblem is rebuilt from the same inputs and the returned solution of each
routine is checked with `mmadual.diagnostics` (not timed). Two runs: default parameters
(`epsimin` $10^{-7}$, `tol_dual` $10^{-10}$) and tighter ones (`epsimin` $10^{-9}$, `tol_dual` $10^{-12}$).

| run | solver | iterations | final compliance | $\min \tilde x$ | elements with $\tilde x < 0.01$ | $\mathcal G_x$ median (max), units of $f_0$ | `mmasub` [s/it] | core time [s/it] | `mmasub` share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | `mmapy`, $10^{-7}$ | 79 | 226.2 | 0.048 | 0 % | $1.6\times10^{-2}$ ($1.7\times10^{-2}$) | 2.01 | 6.46 | 31 % |
| default | MMA-DUAL, $10^{-10}$ | 143 | 205.6 | 0 | 37 % | $1.6\times10^{-10}$ ($4.8\times10^{-6}$) | 0.18 | 4.61 | 4 % |
| tight | `mmapy`, $10^{-9}$ | 149 | 205.8 | $7.7\times10^{-4}$ | 37 % | $2.6\times10^{-4}$ ($2.6\times10^{-4}$) | 3.58 | 8.03 | 45 % |
| tight | MMA-DUAL, $10^{-12}$ | 143 | 205.6 | 0 | 37 % | $6.8\times10^{-12}$ ($4.3\times10^{-8}$) | 0.19 | 4.64 | 4 % |

Core time is analysis (4.2 s per iteration) plus sensitivities and filter (0.2 s) plus `mmasub`. At
the default parameters the two runs separate from the third iteration: the dual solutions return a
growing number of variables exactly at a subproblem bound (266 600 of 270 000 at the end, 133 400 of
them at $x_j = 0$), the interior-point solutions never return one, and their smallest filtered density
settles at 0.048; the `mmapy` run stops at a design with no element below 0.01 and compliance 226.2,
the MMA-DUAL run at 205.6. With the tighter parameters the two runs give similar topologies and
compliances within 0.08 % (final filtered densities differ by 0.006 on average and 0.12 at most). The
MMA-DUAL run with tolerance $10^{-12}$ reproduces the one with $10^{-10}$ to $10^{-6}$ in the
compliance, with mean Newton steps 6.4 against 5.9 and three subproblems stopping with reason
`'rounding'` (final residuals below $10^{-15}$, gaps below $4 \times 10^{-11}$).

The subproblems of iteration 70 of both default runs were saved and solved on identical coefficients
with both routines (verification machine). On the subproblem of the `mmapy` trajectory the dual
solution puts 121 176 variables at the lower bound (all with $\alpha_j = 0$) and 123 602 at the upper
one, gap $2.8 \times 10^{-11}$ after five Newton steps; the interior-point solutions keep every
variable strictly inside, with the $2n$ bound complementarity products at the barrier level (sum
$2 n\, \epsilon_{\min}$: $5.4 \times 10^{-2}$, $5.4 \times 10^{-4}$, $5.4 \times 10^{-6}$) and gaps
$1.7 \times 10^{-2}$, $2.4 \times 10^{-4}$, $2.4 \times 10^{-6}$. For the elements that the dual
solution puts at $\alpha_j = 0$, the recovered bound multipliers $\xi_j$ have median
$8.5 \times 10^{-7}$, of the order of $\lambda/(0.5 n) = 1.1 \times 10^{-6}$ with $\lambda = 0.153$;
the median offset $x_j - \alpha_j$ of the interior-point solution is $1.2 \times 10^{-3}$ at
`epsimin` $= 10^{-9}$ and $1.2 \times 10^{-5}$ at $10^{-11}$, equal to $\varepsilon/\xi_j$ to 2.3 %,
and $5.1 \times 10^{-2}$ at $10^{-7}$, where the multiplier itself differs (0.200 against 0.153) and
the prediction (0.12) holds in order of magnitude. As a scale estimate valid when the volume term
dominates $\xi_j$, the offset at fixed `epsimin` is of order $0.5\, n\, \varepsilon/\lambda$ and grows
with the number of elements; the comparison with the $n = 1000$ subproblem of Section 3 does not
isolate the size effect, since the two problems differ in more than $n$.

What these runs do and do not show: the subproblem accuracy of the two routines, at their defaults,
leads to different outer trajectories on this problem, and the difference is explained by the
finite-barrier offsets; exact bound attainment is a property of the analytical minimizer, available
to any dual solver that evaluates it, and a zero density in a subproblem does not by itself
guarantee a better optimum of the original problem; the gain in the complete iteration (1.4 and 1.7)
is bounded by the share of the subproblem, which depends on the analysis (a direct solver on one
thread here).
