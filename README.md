# MMA-DUAL

[![PyPI](https://img.shields.io/pypi/v/mma-dual.svg)](https://pypi.org/project/mma-dual/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23741623.svg)](https://doi.org/10.5281/zenodo.23741623)

MMA and GCMMA subproblems solved by the **dual method**, as drop-in replacements for
`mmasub` and `gcmmasub` of [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python)
(`pip install mmapy`), for optimization problems with few constraints (m ≪ n).

## Installation

```
pip install mma-dual
```

The package is a single module, `mmadual.py`, with numpy as the only dependency:

```python
from mmadual import mmasub, gcmmasub
```

To run the tests, the example and the benchmark, `mmapy` is also needed (`pip install mmapy`,
or `pip install "mma-dual[test]"`). Alternatively, copy `mmadual.py` next to your code, or
install from a clone of this repository with `pip install -e .`.

## Origin and license

This module is an adaptation of GCMMA-MMA-Python by Arjen Deetman, which is the Python
translation of the MATLAB implementation of the Method of Moving Asymptotes by
Krister Svanberg ([smoptit.se](http://www.smoptit.se/)). The construction of the
subproblem (moving asymptotes, bounds `alfa`/`beta`, coefficients `p0`, `q0`, `P`, `Q`, `b`)
is copied verbatim from `mmapy`, so that both codes build exactly the same subproblem.
What changes is how the subproblem is solved. Both ancestors are distributed under the
GNU General Public License v3, and so is this package (see `LICENSE`).

## Why

`mmapy` solves the subproblem with a primal-dual interior-point Newton method in
n + 2m + 3 unknowns. For problems with hundreds or thousands of design variables and a
handful of constraints, the cost of this step can exceed the cost of the analysis of the
original problem, and in difficult subproblems (large multipliers, many variables at the
move limits) the interior-point iterations reach their limit of 200 Newton steps per
accuracy level.

The dual method (Svanberg 1987, Section 5) exploits the separability of the subproblem:
for fixed multipliers λ of the m constraints, the subproblem is minimized in closed form in
each design variable, and the concave dual function W(λ) is maximized by a projected Newton
method with analytical Hessian on the box 0 ≤ λ ≤ c. Each evaluation costs O(n m).

## Method

### The subproblem

At each outer iteration, MMA and GCMMA replace the original problem by a separable convex
subproblem in the design variables $x_j$ ($j = 1,\dots,n$) and the artificial variables
$y_i \ge 0$ ($i = 1,\dots,m$) and $z \ge 0$ (notation of GCMMA-MMA-Python):

$$
\min_{x,y,z}\ \sum_{j=1}^{n}\left(\frac{p_{0j}}{U_j - x_j} + \frac{q_{0j}}{x_j - L_j}\right) + a_0 z + \sum_{i=1}^{m}\left(c_i y_i + \tfrac{1}{2} d_i y_i^2\right)
$$

$$
\text{s.t.}\quad g_i(x) - a_i z - y_i \le b_i,\qquad g_i(x) = \sum_{j=1}^{n}\left(\frac{P_{ij}}{U_j - x_j} + \frac{Q_{ij}}{x_j - L_j}\right),\qquad \alpha_j \le x_j \le \beta_j .
$$

$L_j < \alpha_j$ and $U_j > \beta_j$ are the moving asymptotes, and $p_{0j}, q_{0j}, P_{ij}, Q_{ij} \ge 0$
and $b_i$ (of either sign: $b_i < 0$ when the constraint is strongly violated at the current design)
are built from the function values and gradients at the current design. This construction is
copied verbatim from `mmapy`, so the subproblem solved here is exactly the one solved there.

### Dual function

Let $\lambda_i \ge 0$ be the multipliers of the $m$ constraints. The Lagrangian is separable
in $x_j$, $y_i$ and $z$, so for fixed $\lambda$ it is minimized in closed form
(Svanberg 1987, Section 5). With

$$
p_j(\lambda) = p_{0j} + \sum_i \lambda_i P_{ij},\qquad q_j(\lambda) = q_{0j} + \sum_i \lambda_i Q_{ij},
$$

the unconstrained minimizer of $p_j/(U_j - x_j) + q_j/(x_j - L_j)$ is the point where the two
terms have equal and opposite slopes,

$$
x_j(\lambda) = \frac{\sqrt{p_j}\,L_j + \sqrt{q_j}\,U_j}{\sqrt{p_j} + \sqrt{q_j}},
$$

clipped to $[\alpha_j, \beta_j]$ (the function is convex in $x_j$, so clipping gives the
constrained minimizer). For the artificial variables,

$$
y_i(\lambda) = \max\left(0, \frac{\lambda_i - c_i}{d_i}\right)\ \ (d_i > 0),\qquad z(\lambda) = 0\ \ (a = 0,\ a_0 > 0),
$$

and when $d_i = 0$ (the usual choice) the term $(c_i - \lambda_i) y_i$ is bounded below only if
$\lambda_i \le c_i$, which becomes an upper bound on the multiplier. At the solution, if
$\lambda_i = c_i$, the constraint cannot be satisfied within the bounds and $y_i$ is recovered
from the residual, $y_i = \max(0, g_i(x) - b_i)$, as in `mmapy`; otherwise $y_i = 0$. The dual
function

$$
W(\lambda) = \sum_j\left(\frac{p_j}{U_j - x_j} + \frac{q_j}{x_j - L_j}\right) - \lambda^{\mathsf{T}} b + \sum_i\left(c_i y_i + \tfrac{1}{2} d_i y_i^2\right) - \lambda^{\mathsf{T}} y,
$$

evaluated at $x(\lambda), y(\lambda)$, is concave, and by strong duality (the
subproblem is convex with a strictly feasible point) its maximum over the box
$0 \le \lambda_i \le c_i$ (or $\lambda_i \ge 0$ when $d_i > 0$) gives the solution of the
subproblem.

Only $a = 0$ is supported, which is the standard choice ($a_0 = 1$, $a_i = 0$). With $a \ne 0$
(the minimax form, $a_0 = 1$, $a_i = 1$) the dual domain is the box intersected with the
half-space $\sum_i a_i \lambda_i \le a_0$, whose boundary is where $z > 0$ is recovered; the
projected Newton method below does not handle this intersection, and `subsolv_dual` raises
`NotImplementedError` for $a \ne 0$. Use `mmapy` for such problems.

### Gradient and Hessian

By the envelope theorem, the gradient of $W$ is the residual of the constraints at the
primal minimizer,

$$
\frac{\partial W}{\partial \lambda_i} = g_i\big(x(\lambda)\big) - y_i - b_i .
$$

Its Hessian is obtained by differentiating $x_j(\lambda)$. For the variables strictly inside
$[\alpha_j, \beta_j]$, the stationarity condition $\partial_{x_j}\mathcal{L} = 0$ gives

$$
\frac{\partial x_j}{\partial \lambda_i} = -\frac{G_{ij}}{D_j},\qquad G_{ij} = \frac{\partial g_i}{\partial x_j} = \frac{P_{ij}}{(U_j - x_j)^2} - \frac{Q_{ij}}{(x_j - L_j)^2},\qquad D_j = \frac{2 p_j}{(U_j - x_j)^3} + \frac{2 q_j}{(x_j - L_j)^3} > 0,
$$

while the variables at a bound do not move. Hence

$$
\nabla^2 W = -\sum_{j\ \text{free}} \frac{G_{\cdot j} G_{\cdot j}^{\mathsf{T}}}{D_j} - \mathrm{diag}\left(\frac{[\lambda_i > c_i]}{d_i}\right),
$$

which is negative semidefinite, as expected for a concave function. The Hessian is defined
piecewise: the set of free variables changes with $\lambda$ (a variable entering or leaving a
bound removes or adds its rank-one term), so $W$ is $C^1$ but only piecewise $C^2$. Both the
gradient and the Hessian cost $O(nm)$ and $O(nm^2)$ operations, respectively, and the Hessian
is a dense $m \times m$ matrix.

### Maximization

$W$ is maximized by a projected Newton method with Levenberg-Marquardt damping. At each step
the active set is formed by the multipliers at a bound of the box whose gradient points
outwards. In the free multipliers, the Hessian is brought to the Jacobi scaling
$S H S$, $S = \mathrm{diag}(|H_{ii}|^{-1/2})$, which makes the step invariant to the scale of
each constraint, and decomposed in eigenvalues $-w_k$ ($w_k \ge 0$). The step is

$$
\Delta\lambda = S\,V\,\mathrm{diag}\!\left(\frac{1}{w_k + \mu\, w_{\max}}\right) V^{\mathsf{T}} S\, \nabla W ,
$$

which is the Newton step for $\mu \to 0$ and a scaled gradient step for $\mu \to \infty$;
components without curvature ($H_{ii} = 0$, which happens when no $x_j$ is free) take the
gradient, damped by the same $\mu$. The step is projected onto the box and accepted when it
satisfies the Armijo condition $W(\lambda^+) - W(\lambda) \ge 10^{-4}\,\nabla W(\lambda)^{\mathsf{T}}(\lambda^+ - \lambda)$,
up to a rounding allowance of $10^{-14}|W|$ (proportional to $|W|$ because $W$ is a sum of $n$
positive terms whose rounding error scales with $|W|$; in tests the allowance never admitted
an actual decrease of $W$). Otherwise $\mu$ is multiplied by 10 and the step recomputed; after
an accepted step $\mu$ is divided by 100, down to $10^{-12}$. The damping only touches the flat
directions of $W$, which is what makes the method robust when the Hessian is singular or
nearly so: redundant or nearly redundant constraints, more constraints than free variables
($m > |\mathcal{F}|$), or constraints whose scales differ by many orders of magnitude.

The iteration stops when the projected gradient of each constraint is smaller than `tol_dual`
relative to the scale of that constraint, $s_i = \max(1, |b_i|, g_i(x(0)))$. Stationarity in
$x$, $y$ and $z$ holds by construction of $x(\lambda)$, $y(\lambda)$ and $z$, so the projected
dual gradient is the feasibility and complementarity residual of the subproblem *with* its
artificial variables: $|g_i(x) - y_i - b_i| \le$ `tol_dual` $s_i$ for the free multipliers, with
$g_i(x) - y_i < b_i$ allowed only at $\lambda_i = 0$. This is feasibility of the subproblem, not
of the approximate constraints $g_i(x) \le b_i$ themselves: when $\lambda_i = c_i$ the latter
are violated at $x$ and $y_i > 0$ carries the violation, exactly as in `mmapy`. The
complementarity residuals $\lambda_i s_i$ and $\mu_i y_i$ are the gradient residual multiplied
by $\lambda_i$ or $\mu_i \le c_i$, so with $c = 10^3$ they can reach $10^3$ times the gradient
tolerance; `tests/test_edge_cases.py` checks all the KKT residuals of the subproblem,
including those of the multipliers $\xi, \eta, \mu, \zeta, s$ recovered for compatibility
with `mmapy`.

The iteration also stops when two consecutive undamped Newton steps increase $W$ by less than
$10^{-13}|W|$: the maximum is then resolved to the rounding of $W$, and the residual cannot be
reduced further. This happens with nearly redundant constraints (rows of the constraint
Jacobian equal to $10^{-8}$), where $W$ is flat to $10^{-16}$ along $m - 1$ directions and a
residual of $10^{-10}$ would need $\Delta\lambda \sim 10^{6}$; a tight L-BFGS-B run stops at the
same point. A `RuntimeWarning` reporting the residual actually reached is issued when the
iteration stops with a residual above the square root of `tol_dual` ($10^{-5}$ by default, the level at
which `mmapy` itself stops), whether by `max_newton` or by the rounding criterion. In
practice a few steps suffice: 5 to 10 on regular subproblems, 10 to 25 on the degenerate
cases above, and up to `max_newton` on nearly redundant constraints, where the dual value
reached is within $10^{-10}$ of the optimum from the tenth step on. Once $\lambda^\star$ is found,
$x(\lambda^\star)$ is the solution of the subproblem.

### What is different from `mmapy`

`mmapy` solves the KKT conditions of the subproblem in the full primal-dual space
$(x, y, z, \lambda, \xi, \eta, \mu, \zeta, s)$, with $n + 2m + 3$ unknowns, by a Newton method
with a barrier parameter $\epsilon$ decreased from $1$ to $10^{-7}$, up to 200 Newton steps
per level. Each step solves a dense $(m+1) \times (m+1)$ system, but the number of steps grows
in difficult subproblems (large multipliers, many variables at the move limits), and the
cost of the surrounding vector operations in $n$ is paid at every step. The dual method
reduces the problem to $m$ unknowns and reaches the solution with a few Newton steps,
each costing one pass over the $n$ variables. The approximation reached at the returned point
is the same to within the stopping tolerances of the two methods (about $10^{-5}$ in the
tests).

### Benchmark

Median wall time of five random MMA subproblems per $(n, m)$ (`benchmarks/bench_n_m.py`,
one core). Gradients $\sim N(0,1)$, constraint values $\sim U(-1,1)$,
$c = 10^3$, $d = 0$, `move = 0.2`. The gain depends on how hard the subproblem is for the
interior-point method, more than on $n$ and $m$: on well-scaled subproblems it is a factor of
2.5 to 18, and it grows with $n$ and shrinks with $m$ (the $O(nm^2)$ Hessian); on badly scaled
subproblems (objective gradient $\sim 10^{-3}$, second table), where `mmapy` needs hundreds of
Newton steps per barrier level, it is two to three orders of magnitude. In that regime `mmapy`
also returns less accurate subproblem solutions (it stops at its iteration limit: feasibility
residuals of $10^{-4}$ to $10^{-3}$ in the second table, against $10^{-12}$ to $10^{-9}$ for the
dual, which always reaches `tol_dual`).

| $n$ | $m$ | `mmapy` [s] | `mmadual` [s] | ratio |
|---:|---:|---:|---:|---:|
| 1 000 | 1 | 0.015 | 0.0015 | 9.8 |
| 1 000 | 4 | 0.015 | 0.0033 | 4.5 |
| 1 000 | 16 | 0.015 | 0.0042 | 3.6 |
| 1 000 | 64 | 0.027 | 0.011 | 2.5 |
| 10 000 | 1 | 0.054 | 0.0056 | 9.5 |
| 10 000 | 4 | 0.063 | 0.0082 | 7.6 |
| 10 000 | 16 | 0.083 | 0.027 | 3.1 |
| 10 000 | 64 | 0.20 | 0.087 | 2.3 |
| 100 000 | 1 | 0.87 | 0.048 | 18 |
| 100 000 | 4 | 0.96 | 0.075 | 13 |
| 100 000 | 16 | 1.21 | 0.17 | 6.9 |
| 100 000 | 64 | 3.43 | 0.96 | 3.6 |

| $n$ | $m$ | `mmapy` [s] | `mmadual` [s] | ratio |
|---:|---:|---:|---:|---:|
| 1 000 | 4 | 0.73 | 0.0017 | 420 |
| 10 000 | 4 | 3.66 | 0.0087 | 420 |
| 10 000 | 16 | 4.87 | 0.018 | 270 |
| 100 000 | 4 | 34.2 | 0.074 | 460 |

In an optimization run the subproblem is only part of the cost, so the gain in total time is
smaller than the ratios above.

## Usage

```python
from mmadual import mmasub, gcmmasub          # instead of: from mmapy import mmasub, gcmmasub
from mmapy import asymp, concheck, raaupdate   # GCMMA helpers, unchanged

xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
    m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx,
    low, upp, a0, a, c, d, move=0.1, asymin=1e-3)
```

The positional arguments and the returned tuples are those of `mmapy`. Differences:

- `gcmmasub(..., move=None)`: optional move limit |x − xval| ≤ move (xmax − xmin), as in
  `mmasub`; `None` reproduces `mmapy`.
- `tol_dual=1e-10` and `max_newton=100`: tolerance on the projected dual gradient and
  maximum number of Newton steps.
- The multipliers `lam` returned are the dual solution; `xsi`, `eta`, `mu`, `zet`, `s` are
  recovered from the KKT conditions for compatibility and are not used by the method.
- A `RuntimeWarning` is issued if the dual iteration stops before `tol_dual` is reached
  (`max_newton` steps, or no further progress in the line search); the warning reports the
  residual actually reached. Silence it with `warnings.filterwarnings` if it is acceptable.

Limitations: only `a = 0` is supported (`NotImplementedError` otherwise, see
[Dual function](#dual-function)); the dual Hessian is a dense m × m matrix, so the method is
meant for m up to a few tens of constraints. With `d = 0` (the usual choice), the multipliers
are bounded by `c`; if the constraints cannot be satisfied within the move limits, `lam`
reaches `c` and the returned `y` carries the violation, as in `mmapy`.

## Tests and example

`tests/test_against_mmapy.py` compares the two solvers on random MMA and GCMMA subproblems
(objective of the approximation plus the penalty of the artificial variables, feasibility,
`f0app`/`fapp`); `tests/test_edge_cases.py` covers strongly violated constraints (multipliers
at `c`, `y > 0`), `d > 0`, objective gradients scaled from $10^{-3}$ to $10^{3}$, the full KKT
residuals of the subproblem at the returned point, degenerate subproblems (weakly active
constraints, constraint scales from $10^{-6}$ to $10^{6}$, identical and nearly identical
constraints, a multiplier exactly at `c` with zero gradient, more constraints than variables,
no free variable at the start), the `a != 0` exception and the non-convergence warning; `examples/toy_problem.py` runs the
five-variable test problem of GCMMA-MMA-Python with both subproblem solvers and prints the
iteration histories; `benchmarks/bench_n_m.py` reproduces the tables above.

```
pip install -e ".[test]"  # from a clone: the module plus mmapy, needed by the tests, the example and the benchmark
python tests/test_against_mmapy.py
python tests/test_edge_cases.py
python examples/toy_problem.py
python benchmarks/bench_n_m.py
```

## Citing

This repository is archived on Zenodo (DOI [10.5281/zenodo.23741623](https://doi.org/10.5281/zenodo.23741623)) and has a `CITATION.cff` file (use the "Cite this repository" button on GitHub).
Please cite the version used, together with GCMMA-MMA-Python, from which the construction of
the subproblem is taken, and the papers by Svanberg that define the methods:

```bibtex
@software{silva2026mmadual,
  author  = {Silva, Olavo M.},
  title   = {{MMA-DUAL}: {MMA} and {GCMMA} subproblems solved by the dual method},
  version = {0.1.1},
  year    = {2026},
  doi     = {10.5281/zenodo.23741623},
  url     = {https://github.com/olavosilva/MMA-DUAL},
  note    = {GPL-3.0; derived from GCMMA-MMA-Python}
}

@software{deetman2025gcmmammapython,
  author  = {Deetman, Arjen},
  title   = {{GCMMA-MMA-Python}: {Python} implementation of the Method of Moving Asymptotes},
  version = {0.3.1},
  year    = {2025},
  doi     = {10.5281/zenodo.15459165},
  url     = {https://github.com/arjendeetman/GCMMA-MMA-Python}
}

@article{svanberg1987mma,
  author  = {Svanberg, Krister},
  title   = {The method of moving asymptotes - a new method for structural optimization},
  journal = {International Journal for Numerical Methods in Engineering},
  volume  = {24},
  number  = {2},
  pages   = {359--373},
  year    = {1987},
  doi     = {10.1002/nme.1620240207}
}

@article{svanberg2002gcmma,
  author  = {Svanberg, Krister},
  title   = {A class of globally convergent optimization methods based on conservative convex separable approximations},
  journal = {SIAM Journal on Optimization},
  volume  = {12},
  number  = {2},
  pages   = {555--573},
  year    = {2002},
  doi     = {10.1137/S1052623499362822}
}
```

The DOI of GCMMA-MMA-Python above is the one of version 0.3.1; the concept DOI that always
resolves to the latest version is 10.5281/zenodo.13197565.
