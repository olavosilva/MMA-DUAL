# MMA-DUAL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23741623.svg)](https://doi.org/10.5281/zenodo.23741623)

MMA and GCMMA subproblems solved by the **dual method**, as drop-in replacements for
`mmasub` and `gcmmasub` of [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python)
(`pip install mmapy`), for optimization problems with few constraints (m ≪ n).

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

On the duct-network optimization problems this package was written for (n ≈ 800, m = 4),
the subproblem time per iteration dropped from 0.3 to 0.9 s to 5 to 30 ms. The two solvers
return the same approximate objective to within about 1e-5, the difference reflecting their
stopping tolerances (see `tests/`), and the optimization histories coincide (see `examples/`).

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

$L_j < \alpha_j$ and $U_j > \beta_j$ are the moving asymptotes, and $p_{0j}, q_{0j}, P_{ij}, Q_{ij}, b_i \ge 0$
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
y_i(\lambda) = \max\left(0, \frac{\lambda_i - c_i}{d_i}\right)\ \ (d_i > 0),\qquad z(\lambda) = 0\ \ \text{if } a_0 > \textstyle\sum_i a_i \lambda_i ,
$$

and when $d_i = 0$ (the usual choice) the term $(c_i - \lambda_i) y_i$ is bounded below only if
$\lambda_i \le c_i$, which becomes an upper bound on the multiplier, with $y_i = 0$. The dual
function

$$
W(\lambda) = \sum_j\left(\frac{p_j}{U_j - x_j} + \frac{q_j}{x_j - L_j}\right) - \lambda^{\mathsf{T}} b + a_0 z - \lambda^{\mathsf{T}}(a z) + \sum_i\left(c_i y_i + \tfrac{1}{2} d_i y_i^2\right) - \lambda^{\mathsf{T}} y,
$$

evaluated at $x(\lambda), y(\lambda), z(\lambda)$, is concave, and by strong duality (the
subproblem is convex with a strictly feasible point) its maximum over the box
$0 \le \lambda_i \le c_i$ (or $\lambda_i \ge 0$ when $d_i > 0$) gives the solution of the
subproblem.

### Gradient and Hessian

By the envelope theorem, the gradient of $W$ is the residual of the constraints at the
primal minimizer,

$$
\frac{\partial W}{\partial \lambda_i} = g_i\big(x(\lambda)\big) - a_i z - y_i - b_i .
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

which is negative semidefinite, as expected for a concave function. Both the gradient and the
Hessian cost $O(nm)$ and $O(nm^2)$ operations, respectively, and the Hessian is a dense
$m \times m$ matrix.

### Maximization

$W$ is maximized by a projected Newton method: at each step the active set is formed by the
multipliers at a bound of the box whose gradient points outwards, the Newton direction is
computed in the free multipliers, replaced by the projected gradient when it is not an ascent
direction, and the step is halved until $W$ does not decrease, with projection onto the box.
The iteration stops when the projected gradient is smaller than `tol_dual` relative to
$\max(1, \|b\|_\infty)$. The dual function is piecewise smooth (the sets of free variables
and of active bounds change with $\lambda$), so the Newton direction is exact within each
piece and the line search handles the transitions; in practice a few steps suffice. Once
$\lambda^\star$ is found, $x(\lambda^\star)$ is the solution of the subproblem, and the remaining
multipliers returned for compatibility with `mmapy` ($\xi, \eta, \mu, \zeta, s$) are recovered
from the KKT conditions.

### What is different from `mmapy`

`mmapy` solves the KKT conditions of the subproblem in the full primal-dual space
$(x, y, z, \lambda, \xi, \eta, \mu, \zeta, s)$, with $n + 2m + 3$ unknowns, by a Newton method
with a barrier parameter $\epsilon$ decreased from $1$ to $10^{-7}$, up to 200 Newton steps
per level. Each step solves a dense $(m+1) \times (m+1)$ system, but the number of steps grows
in difficult subproblems (large multipliers, many variables at the move limits), and the
cost of the surrounding vector operations in $n$ is paid at every step. The dual method
reduces the problem to $m$ unknowns and reaches the solution with a few Newton steps,
each costing one pass over the $n$ variables, which is why it is faster by one to two orders
of magnitude when $m \ll n$. The approximation reached at the returned point is the same to
within the stopping tolerances of the two methods (about $10^{-5}$ in the tests).

## Usage

```python
from mma_dual import mmasub, gcmmasub          # instead of: from mmapy import mmasub, gcmmasub
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

Limitations: the dual Hessian is a dense m × m matrix, so the method is meant for m up to a
few tens of constraints. With `d = 0` (the usual choice), the multipliers are bounded by
`c`; if the constraints cannot be satisfied within the move limits, `lam` reaches `c` and
the returned `y` carries the violation, as in `mmapy`.

## Tests and example

`tests/test_against_mmapy.py` compares the two solvers on random MMA and GCMMA subproblems
(objective of the approximation plus the penalty of the artificial variables, feasibility,
`f0app`/`fapp`); `examples/toy_problem.py` runs the five-variable test problem of
GCMMA-MMA-Python with both subproblem solvers and prints the iteration histories.

```
pip install -e .          # numpy only
pip install mmapy         # needed for the tests, the example and the GCMMA helpers
python tests/test_against_mmapy.py
python examples/toy_problem.py
```

## Citing

This repository is archived on Zenodo (DOI [10.5281/zenodo.23741623](https://doi.org/10.5281/zenodo.23741623)) and has a `CITATION.cff` file (use the "Cite this repository" button on GitHub).
Please cite the version used, together with GCMMA-MMA-Python, from which the construction of
the subproblem is taken, and the papers by Svanberg that define the methods:

```bibtex
@software{silva2026mmadual,
  author  = {Silva, Olavo M.},
  title   = {{MMA-DUAL}: {MMA} and {GCMMA} subproblems solved by the dual method},
  version = {0.1.0},
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
