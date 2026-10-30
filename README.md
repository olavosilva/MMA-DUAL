# MMA-DUAL

[![PyPI](https://img.shields.io/pypi/v/mma-dual.svg)](https://pypi.org/project/mma-dual/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23741623.svg)](https://doi.org/10.5281/zenodo.23741623)

An implementation of the MMA and GCMMA subproblems (Svanberg 1987, 2002) in which the separable
convex subproblem of each iteration is solved by the **dual method**: the design variables are
eliminated analytically for fixed multipliers, and the concave dual function of the $m$ multipliers
is maximized by a projected Newton method with analytic Hessian, Jacobi scaling and
Levenberg-Marquardt damping. The package is meant for problems with many design variables and few
constraints ($m \ll n$), such as topology optimization, where the cost of the subproblem is then
negligible against the analysis.

The public functions `mmasub`, `gcmmasub`, `asymp`, `concheck`, `raaupdate` and `kktcheck` have the
calling signatures of [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python) (`mmapy`),
so that an existing MMA or GCMMA loop runs unchanged after `from mmadual import ...`; `mmapy` is not
needed to run it. The package also exposes the construction of the subproblems, the dual solver and a
set of diagnostics (KKT residuals, duality gap) that can be applied to the solution of any subproblem
solver.

## Installation

```
pip install mma-dual
```

```python
from mmadual import mmasub, gcmmasub, asymp, concheck, raaupdate, kktcheck
```

numpy is the only dependency. The tests, examples and benchmarks compare with `mmapy`
(`pip install "mma-dual[test]"` installs it together with scipy and pytest). To work from a clone:
`pip install -e .`.

## Usage

Drop-in use in an MMA loop (the arguments and the returned tuple are those of `mmapy.mmasub`, column
vectors; `move` and the asymptote parameters have the same defaults):

```python
xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
    m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx,
    low, upp, a0, a, c, d, move=0.2)
```

Options beyond `mmapy`: `tol_dual=1e-10` and `max_newton=100` control the dual iteration;
`return_info=True` appends to the tuple a dict with the convergence status of the dual iteration
(`converged`, `reason`, `newton_steps`, `residual`, the duality gap of the returned point `gap_returned` with the bound `gap_allowed` that the tolerance implies and `certified`, and `z`/`halfspace_active` for $a \ne 0$);
`gcmmasub(..., move=...)` adds an optional move limit, absent in `mmapy`.

A GCMMA loop with the helpers (see `examples/toy_problem.py` for the complete loop with inner iterations):

```python
low, upp, raa0, raa = asymp(k, n, x, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0dx, dfdx)
out = gcmmasub(m, n, k, epsimin, x, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
xnew, f0app, fapp = out[0], out[9], out[10]
while not concheck(m, epsimin, f0app, f0new, fapp, fnew):
    raa0, raa = raaupdate(xnew, x, xmin, xmax, low, upp, f0new, fnew, f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
    ...
```

Package interface, for work on the subproblem itself:

```python
from mmadual import mma_subproblem, gcmma_subproblem, solve_dual, recover_multipliers, diagnostics

sub = mma_subproblem(iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a, c, d, move=0.2)
x, y, z, lam, info = solve_dual(sub, tol=1e-10, max_newton=100)
xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
check = diagnostics.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)   # KKT residuals, f_pen, duality gap
```

`sub` is a plain dict with the coefficients `low, upp, alfa, beta, p0, q0, P, Q, b, a0, a, c, d` (and
`r0, r` for the GCMMA), independent of how it was built; `solve_dual` and `diagnostics` take it as the
only description of the problem, so a subproblem built by other code, or solved by another solver,
can be checked with the same functions.

## Package layout

```
src/mmadual/
  approximations.py   moving asymptotes, bounds alfa/beta, coefficients p0, q0, P, Q, b (MMA) and r0, r (GCMMA)
  dual_solver.py      dual function, gradient, Hessian; projected Newton with LM damping; recovery of the multipliers
  mma.py              mmasub
  gcmma.py            gcmmasub, asymp, concheck, raaupdate, kktcheck
  diagnostics.py      KKT residuals, exact-penalty objective, dual value, duality gap
tests/                pytest suite; tests/data/reference_mmapy.npz holds coefficients captured from mmapy 0.3.1
benchmarks/           bench_n_m.py (timing and matched-accuracy comparison), mbb_subproblem_accuracy.py,
                      mmadual_v012_reference.py (the single-module version 0.1.2, kept as reference)
examples/             toy_problem.py (Svanberg's beam, MMA and GCMMA), mbb_topopt.py (MBB topology optimization)
```

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
are built from the function values and gradients at the current design. Since version 0.2.0 this
construction is written from Svanberg's published formulations (`approximations.py`) and reproduces, to
rounding, the coefficients that `mmapy` builds, so the subproblem solved here is the one solved there
(checked against stored `mmapy` coefficients in `tests/test_approximations.py`).

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
x_j(\lambda) = \frac{\sqrt{p_j} L_j + \sqrt{q_j} U_j}{\sqrt{p_j} + \sqrt{q_j}},
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

With $a \ne 0$ (the minimax form of Svanberg, $a_0 = 1$, $a_i = 1$; $a_i \ge 0$ in general), the
minimization in $z \ge 0$ gives $W(\lambda) = \phi(\lambda)$ on the half-space
$\sum_i a_i \lambda_i \le a_0$, $\phi$ being the dual function with $z = 0$, and $-\infty$ outside
it: the dual domain is the box intersected with that half-space, and $z$ is the multiplier of the
half-space constraint in the dual problem ($\partial W/\partial\lambda_i = a_i z$ on the free
components at the solution). The solver is then an active-set projected Newton method on that
domain: the multipliers at a bound and the half-space are identified with a tolerance, $z$ is
determined together with the box active set (it minimizes the projected gradient of
$W - z\, a^{\mathsf T}\lambda$, a convex piecewise-quadratic function of $z$), the Newton step is
taken in the tangent space of the active constraints up to the first bound met, stationarity is
measured on the identified active set and by the projection onto the whole domain, and the returned
point is certified by its duality gap before `converged` is reported. Supported since version 0.2.0
and tested against `mmapy` on minimax subproblems and against `scipy.optimize.linprog` on affine
dual functions (`tests/test_minimax.py`, `tests/test_minimax_independent_stress.py`). The $a = 0$
iteration of 0.1.x is kept, with one change: a multiplier whose constraint has no free design variable
($W$ affine in it) steps to its bound instead of taking the gradient as a step of unit length.

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
\Delta\lambda = S V \mathrm{diag}\left(\frac{1}{w_k + \mu w_{\max}}\right) V^{\mathsf{T}} S \nabla W ,
$$

which is the Newton step for $\mu \to 0$ and a scaled gradient step for $\mu \to \infty$.
A component without curvature ($H_{ii} = 0$, which happens when no $x_j$ of that constraint is
free, so that $W$ is affine in $\lambda_i$ until a variable leaves its bound) steps to its bound
in the direction of the gradient ($c_i - \lambda_i$ upwards, $\lambda_i$ downwards; $d_i |\partial W/\partial \lambda_i|$
beyond $c_i$ when $d_i > 0$, where the next piece has curvature $1/d_i$), damped by $1/\max(1, \mu)$
(`_flat_step`; versions 0.1.x took the gradient itself as the step, a length in the unit of $W$ that
needed many steps when the gradient was small and the bound far). The step is projected onto the box and accepted when it
satisfies the Armijo condition $W(\lambda^+) - W(\lambda) \ge 10^{-4} \nabla W(\lambda)^{\mathsf{T}}(\lambda^+ - \lambda)$,
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


### Stopping criteria and diagnostics

The dual iteration stops when the projected gradient of each constraint, relative to its scale
$\max(1, |b_i|, g_i(x(0)))$, is below `tol_dual`; or when two consecutive undamped Newton steps
increase $W$ by less than $10^-13|W|$ (the maximum is then resolved to the rounding of $W$); or at
`max_newton` steps. `solve_dual` returns which of these happened in `info['reason']`
(`'tolerance'`, `'rounding'`, `'max_newton'`, `'no_progress'`) with the residual reached, and a
`RuntimeWarning` is issued when the iteration stops with a residual above $\sqrt{\text{tol}}$.
Stationarity in $x$, $y$ and $z$ holds by construction, so the projected dual gradient is the
feasibility and complementarity residual of the subproblem with its artificial variables; when
$\lambda_i = c_i$ the approximate constraint is violated at $x$ and $y_i$ carries the violation, as in
`mmapy`. The complementarity residuals $\lambda_i s_i$ and $\mu_i y_i$ are the gradient residual times
$\lambda_i$ or $\mu_i \le c_i$, so with $c = 10^3$ they are up to $10^3$ times the gradient tolerance.

`diagnostics.check_solution` evaluates, from the equations of the subproblem and for the solution of
any solver: the stationarity residuals with the returned multipliers, a projected-gradient stationarity
measure in $x$ that does not use them, the constraint violation, the complementarity products, the
exact-penalty objective $f_{\rm pen}(x)$ (objective with the artificial variables chosen optimally for
$x$, an upper bound of the optimal value) and the duality gap $f_{\rm pen}(x) - W(\lambda) \ge 0$,
which bounds the suboptimality of $x$ and of $\lambda$ at once ($W(\lambda)$ is checked to be
evaluated at a dual-feasible $\lambda$).

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


### Subproblem accuracy of the interior-point solver

`benchmarks/mbb_subproblem_accuracy.py` builds one MMA subproblem of the MBB beam and solves it with
`mmapy.subsolv` for `epsimin` $= 10^{-7}$ (the value hard-wired in `mmapy.mmasub`), $10^{-9}$ and
$10^{-11}$, and with the dual solver. The interior-point solution stays at a distance of the order of
`epsimin`$/\xi_j$ from a bound with multiplier $\xi_j$: on a 120 x 40 mesh at iteration 31, the median
density returned in the elements that the dual solver puts at the lower bound is $1.2\times10^{-3}$,
$1.2\times10^{-5}$ and $1.2\times10^{-7}$ for the three values of `epsimin`, with duality gaps of
$4.5\times10^{-4}$, $4.5\times10^{-6}$ and $4.5\times10^{-8}$, against zero and $10^{-15}$ for the
dual solver. This measures the accuracy of the subproblem solution; a more accurate subproblem does
not by itself guarantee a better local optimum of the complete optimization.

## Tests, examples and benchmarks

```
pip install -e ".[test]"
pytest                                        # the whole suite (74 tests, about a minute)
python examples/toy_problem.py                # Svanberg's beam with the MMA and the GCMMA, no mmapy needed
python examples/mbb_topopt.py 300 100 500 3.0 0.05   # MBB beam, mmapy against mmadual, time per component
python benchmarks/bench_n_m.py [1e-3] [matched] [v012]
python benchmarks/mbb_subproblem_accuracy.py
```

The tests cover: the construction of the MMA and GCMMA subproblems and the helpers against coefficients
captured from `mmapy` 0.3.1 and stored in `tests/data/` (agreement to $10^{-13}$; `make_reference.py`
regenerates the file); the gradient and the Hessian of the dual function against finite differences,
including the behaviour across a change of the active set; the agreement of the reorganized solver with
version 0.1.2; KKT residuals and duality gap of the solutions, including strongly violated constraints,
$d > 0$, extreme scales, redundant constraints, singular Hessians and the invariance of the solution under
a rescaling of the constraints; the $a \ne 0$ (minimax) case against `mmapy` and against `scipy.optimize.linprog`
on affine dual functions (`test_minimax_independent_stress.py`), and the step of the components without
curvature on affine constraints (`test_flat_components_reach_the_bound`); complete GCMMA iterations
without `mmapy`; and, when `mmapy` is installed, the exact-penalty objective of the dual solution against
the interior-point one on random subproblems.

## Origin and license

The dual method for the MMA subproblem is described by Svanberg (1987, Section 5); the approximations
and their parameters (asymptote rules, the regularization $0.001 |\partial f/\partial x| + r_{aa0}/(x_{\max} - x_{\min})$,
the GCMMA update of the conservativeness parameters) are those of Svanberg's published MATLAB codes
(2007) and of the GCMMA paper (2002). Versions 0.1.x of this package were derived from
GCMMA-MMA-Python by Arjen Deetman, the Python translation of those codes, from which the construction of
the subproblems was taken verbatim and the solution procedure replaced by the dual solver. From version
0.2.0 the construction is written from the published formulations and reproduces, to rounding, the
subproblems of `mmapy`, which remains the reference of the tests and benchmarks. The dual solver is
original to this package. The package is distributed under the GNU General Public License v3 or later,
the license of its ancestors (see `LICENSE`).

## Citing

This repository is archived on Zenodo: the concept DOI [10.5281/zenodo.23741623](https://doi.org/10.5281/zenodo.23741623) always resolves to the latest version, and each release has its own DOI (0.2.0: [10.5281/zenodo.23785895](https://doi.org/10.5281/zenodo.23785895), the one in `CITATION.cff` and in the entry below; use the "Cite this repository" button on GitHub).
Please cite the version used, together with GCMMA-MMA-Python, from which the construction of
the subproblem is taken, and the papers by Svanberg that define the methods:

```bibtex
@software{silva2026mmadual,
  author  = {Silva, Olavo M.},
  title   = {{MMA-DUAL}: {MMA} and {GCMMA} subproblems solved by the dual method},
  version = {0.2.0},
  year    = {2026},
  doi     = {10.5281/zenodo.23785895},
  url     = {https://github.com/olavosilva/MMA-DUAL},
  note    = {GPL-3.0}
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
