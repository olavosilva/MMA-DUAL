# The method: subproblem, dual function, derivatives and iteration

This document holds the derivations that the README summarizes. Notation is that of GCMMA-MMA-Python
(`mmapy`). Everything here refers to version 0.2.0 of the package (`src/mmadual/`).

## 1. The subproblem

At each outer iteration, MMA and GCMMA replace the original problem by a separable convex subproblem
in the design variables $x_j$ ($j = 1,\dots,n$) and the auxiliary variables $y_i \ge 0$
($i = 1,\dots,m$) and $z \ge 0$:

$$
\min_{x,y,z}\ F(x, y, z) = \sum_{j=1}^{n}\left(\frac{p_{0j}}{U_j - x_j} + \frac{q_{0j}}{x_j - L_j}\right) + a_0 z + \sum_{i=1}^{m}\left(c_i y_i + \tfrac{1}{2} d_i y_i^2\right)
$$

$$
\text{s.t.}\quad g_i(x) - a_i z - y_i \le b_i,\qquad g_i(x) = \sum_{j=1}^{n}\left(\frac{P_{ij}}{U_j - x_j} + \frac{Q_{ij}}{x_j - L_j}\right),\qquad \alpha_j \le x_j \le \beta_j .
$$

$L_j < \alpha_j$ and $U_j > \beta_j$ are the moving asymptotes; $\alpha_j$, $\beta_j$ combine the
global bounds, the move limit and the asymptotes. The coefficients $p_{0j}, q_{0j}, P_{ij}, Q_{ij} \ge 0$
and $b_i$ (of either sign; $b_i < 0$ when the constraint is strongly violated at the current design) are
built from the function values and gradients at the current design, with Svanberg's regularization
$\rho_i > 0$ (`raa0` in the MMA, `raa0`, `raa` in the GCMMA), which makes $p_{0j}$ and $q_{0j}$
strictly positive even where the objective gradient vanishes. Since version 0.2.0 this construction
(`approximations.py`) is written from Svanberg's published formulations and reproduces, to rounding,
the coefficients that `mmapy` builds (checked against stored `mmapy` coefficients in
`tests/test_approximations.py`), so the subproblem solved here is the one solved there.

The relaxed constraints above, with the auxiliary variables, are the constraints of the subproblem;
the unrelaxed approximate constraints are $g_i(x) \le b_i$. The two differ whenever $y_i > 0$.

## 2. Dual function

Let $\lambda_i \ge 0$ be the multipliers of the $m$ relaxed constraints. The Lagrangian is separable
in $x_j$, $y_i$ and $z$, so for fixed $\lambda$ it is minimized in closed form (Svanberg 1987,
Section 5). With

$$
p_j(\lambda) = p_{0j} + \sum_i \lambda_i P_{ij},\qquad q_j(\lambda) = q_{0j} + \sum_i \lambda_i Q_{ij},
$$

both strictly positive, the minimizer of $p_j/(U_j - x_j) + q_j/(x_j - L_j)$ on $(L_j, U_j)$ is the
point where the two terms have equal and opposite slopes,

$$
x_j(\lambda) = \frac{\sqrt{p_j}\, L_j + \sqrt{q_j}\, U_j}{\sqrt{p_j} + \sqrt{q_j}},
$$

clipped to $[\alpha_j, \beta_j]$; the function is strictly convex in $x_j$, so clipping gives the
constrained minimizer, and a clipped variable sits exactly on its bound. For the auxiliary variables,

$$
y_i(\lambda) = \max\left(0, \frac{\lambda_i - c_i}{d_i}\right)\ \ (d_i > 0),\qquad z(\lambda) = 0\ \ (a = 0,\ a_0 > 0).
$$

When $d_i = 0$ (the usual choice) the term $(c_i - \lambda_i) y_i$ is bounded below only if
$\lambda_i \le c_i$, which becomes an upper bound on the multiplier, and the minimizing $y_i$ is zero
for $\lambda_i < c_i$ and undetermined at $\lambda_i = c_i$. The dual function

$$
W(\lambda) = \sum_j\left(\frac{p_j}{U_j - x_j} + \frac{q_j}{x_j - L_j}\right) - \lambda^{\mathsf{T}} b + \sum_i\left(c_i y_i + \tfrac{1}{2} d_i y_i^2\right) - \lambda^{\mathsf{T}} y,
$$

evaluated at $x(\lambda)$, $y(\lambda)$, is concave, and by strong duality (the subproblem is convex
with a strictly feasible point) its maximum over the box $0 \le \lambda_i \le c_i$ (or
$\lambda_i \ge 0$ where $d_i > 0$) gives the solution of the subproblem.

**Recovery of the auxiliary variables.** Once $\lambda^\star$ is found, $x(\lambda^\star)$ is the
solution of the subproblem in the design variables. For $d_i = 0$ the returned $y_i$ is recovered from
the residual of the constraint, $y_i = \max(0, g_i(x) - b_i - a_i z)$, as in `mmapy`: when
$\lambda_i = c_i$ this value may be positive (the constraint is violated at $x$ and $y_i$ carries the
violation) or zero (the constraint is active or satisfied); when $\lambda_i < c_i$ it is zero up to
the stopping residual. For $d_i > 0$ the returned $y_i$ is $y_i(\lambda^\star)$. The remaining
multipliers of `mmapy`'s return tuple ($\xi$, $\eta$, $\mu$, $\zeta$, $s$) are recovered from the
stationarity conditions by `recover_multipliers`.

**Minimax form ($a \ne 0$).** With $a_i \ge 0$ (Svanberg's minimax form has $a_0 = 1$, $a_i = 1$),
the minimization in $z \ge 0$ gives $W(\lambda) = \phi(\lambda)$ on the half-space
$\sum_i a_i \lambda_i \le a_0$, $\phi$ being the dual function with $z = 0$, and $-\infty$ outside it.
The dual domain is the box intersected with that half-space, and $z$ is the multiplier of the
half-space constraint in the dual problem ($\partial W/\partial\lambda_i = a_i z$ on the free
components at the solution). This is where the formulation differs from that of Aage and Lazarov
(2013), whose quadratic penalties in $y$ and $z$ leave the dual defined on the whole nonnegative
orthant.

## 3. Gradient and Hessian

By the envelope theorem, the gradient of the finite part $\phi$ of the dual function is the residual of
the constraints at the primal minimizer with $z = 0$,

$$
\frac{\partial \phi}{\partial \lambda_i} = g_i\big(x(\lambda)\big) - y_i - b_i ,
$$

continuous in $\lambda$ because $x(\lambda)$ and $y(\lambda)$ are. For $a = 0$ this is the gradient of $W$.
For $a \ne 0$ the optimality conditions on the half-space involve the corrected gradient
$\nabla\phi - z\, a$, $z$ being the multiplier of the half-space; the code keeps the two separate
(`dual_function` returns $\nabla\phi$, the polytope path subtracts $z\, a$). The Hessian is obtained by
differentiating $x_j(\lambda)$. For the variables strictly inside $(\alpha_j, \beta_j)$, the
stationarity condition in $x_j$ gives

$$
\frac{\partial x_j}{\partial \lambda_i} = -\frac{G_{ij}}{D_j},\qquad G_{ij} = \frac{\partial g_i}{\partial x_j} = \frac{P_{ij}}{(U_j - x_j)^2} - \frac{Q_{ij}}{(x_j - L_j)^2},\qquad D_j = \frac{2 p_j}{(U_j - x_j)^3} + \frac{2 q_j}{(x_j - L_j)^3} > 0,
$$

while the variables at a bound do not move. Hence

$$
\nabla^2 W = -\sum_{j\ \text{free}} \frac{G_{\cdot j} G_{\cdot j}^{\mathsf{T}}}{D_j} - \mathrm{diag}\left(\frac{[\lambda_i > c_i]}{d_i}\right),
$$

negative semidefinite. It is defined piecewise: the set of free variables changes with $\lambda$ (a
variable entering or leaving a bound removes or adds its rank-one term, and for $d_i > 0$ the diagonal
term appears at $\lambda_i = c_i$), so $W$ is $C^1$ but only piecewise $C^2$. The Hessian is dense,
$m \times m$, and can be rank deficient: with $d_i = 0$ its rank is at most the number of free design
variables, and it drops with identical or nearly identical constraints; for $d_i > 0$ and
$\lambda_i > c_i$ the diagonal term $-1/d_i$ adds curvature in that component. The gradient costs $O(nm)$ operations and the
Hessian $O(nm^2)$; its symmetric eigendecomposition costs $O(m^3)$. The positivity of $p_j$, $q_j$
and $D_j$ follows from $\rho_i > 0$; it guarantees a unique design minimizer, not a nonsingular
Hessian.

## 4. Maximization

$W$ is maximized by a projected Newton method with Levenberg–Marquardt damping. At each step the
active set is formed by the multipliers at a bound of the box whose gradient points outwards. On the
free multipliers, the Hessian block is brought to the Jacobi scaling $S H S$,
$S = \mathrm{diag}(|H_{ii}|^{-1/2})$, and decomposed in eigenvalues $-w_k$ ($w_k \ge 0$). The step is

$$
\Delta\lambda = S V \mathrm{diag}\left(\frac{1}{w_k + \mu\, w_{\max}}\right) V^{\mathsf{T}} S\, \nabla W ,
$$

the Newton step for $\mu \to 0$ and a scaled gradient step for $\mu \to \infty$. The Jacobi scaling
reduces the effect of differences of scale between the constraints on the step; it does not make the
complete iteration exactly invariant to a rescaling, since the floor of 1 in the per-constraint
tolerance scale and the rounding allowances are not invariant (the solution is invariant to rounding
when the path is and to `tol_dual` otherwise, `tests/test_edge_cases.py`). The damping term
$\mu\, w_{\max}$ is added to every eigenvalue of the scaled block, so it modifies the whole
regularized block; its relative effect is largest on the directions of small curvature, which is what
keeps the step bounded when the Hessian is singular or nearly so (redundant or nearly redundant
constraints, more constraints than free variables, constraints whose scales differ by many orders
of magnitude).

A free component without curvature ($H_{ii} = 0$: no design variable of that constraint is free, so
$W$ is affine in $\lambda_i$ until a variable leaves its bound) steps to its bound in the direction of
the gradient, $c_i - \lambda_i$ upwards and $\lambda_i$ downwards ($d_i |\partial W/\partial\lambda_i|$
beyond $c_i$ when $d_i > 0$, where the next piece has curvature $1/d_i$), damped by $1/\max(1, \mu)$
(`_flat_step`). Versions 0.1.x took the gradient itself as the step, a length in the unit of $W$ that
needed many steps when the gradient was small and the bound far; this is the only change of the
$a = 0$ iteration in 0.2.0.

The step is projected onto the box and accepted when it satisfies the Armijo condition
$W(\lambda^+) - W(\lambda) \ge 10^{-4}\, \nabla W(\lambda)^{\mathsf{T}}(\lambda^+ - \lambda)$, up to a
rounding allowance of $10^{-14}|W|$ ($W$ is a sum of $n$ positive terms whose rounding error scales
with $|W|$). Otherwise $\mu$ is multiplied by 10 and the step recomputed; after an accepted step $\mu$
is divided by 100, down to $10^{-12}$.

**The $a \ne 0$ path** (`_solve_polytope`) is an active-set projected Newton method on the box
intersected with the half-space: the multipliers within a relative tolerance of $10^{-12}$ of a bound,
and the half-space within the same tolerance, count as active; $z \ge 0$ and the box active set are
determined together ($z$ minimizes the norm of the projected gradient of $W - z\, a^{\mathsf T}\lambda$,
a convex piecewise-quadratic function of $z$, by bisection; $z > 0$ means the half-space is active);
the Newton/LM direction is computed on the free multipliers in the tangent space of the half-space
when it is active, a free multiplier whose direction points out of its bound being fixed and the
direction recomputed; the step goes along that direction up to the first bound or the half-space
(ratio test), never beyond, so that no non-gradient direction is projected; the trial point is
accepted under the Armijo condition, with a projected-gradient arc with backtracking when every
damping fails. The least-squares estimate of $z$ on the components treated as free would release
multipliers that the joint identification keeps at their bounds (the two-variable simplex example of
`tests/test_minimax.py`).

## 5. Stopping, residuals and certificates

The iteration stops when the projected gradient of each constraint is below `tol_dual` relative to
the scale $\kappa_i = \max(1, |b_i|, g_i(x(0)))$ of that constraint (reason `'tolerance'`); or when
two consecutive undamped Newton steps increase $W$ by less than $10^{-13}|W|$ (reason `'rounding'`, a
stagnation test on the value of $W$, taken before the residual is tested again, which does not by
itself show that the residual or the gap reached machine precision); or at `max_newton` steps; or
when a trial point is identical to the current one (`'no_progress'`). For $a \ne 0$, stationarity is
measured both on the identified active set and by the projection onto the whole domain,
$|\lambda - P_D(\lambda + \nabla W)|$, which depends on no active set, and both must be below the
tolerance.

Stationarity in $x$, $y$ and $z$ holds by construction of $x(\lambda)$, $y(\lambda)$ and $z$, so the
projected dual gradient is the feasibility and complementarity residual of the subproblem with its
auxiliary variables: $|g_i(x) - a_i z - y_i - b_i| \le$ `tol_dual` $\kappa_i$ for the free multipliers (with
$z = 0$ when $a = 0$),
with $g_i(x) - a_i z - y_i < b_i$ allowed only at $\lambda_i = 0$. The complementarity residuals
$\lambda_i s_i$ and $\mu_i y_i$ are the gradient residual multiplied by $\lambda_i$ or
$\mu_i \le c_i$, so with $c_i = 10^3$ they can reach $10^3 \kappa_i$ times the tolerance. This is why
`tol_dual` controls a residual and not a gap.

**Two gaps.** Let $F_{\rm pen}(x)$ be the exact-penalty objective, the subproblem objective with the
auxiliary variables chosen optimally for the returned $x$ (an upper bound of the optimal value). The
gap $\mathcal G_x = F_{\rm pen}(x) - W(\lambda) \ge 0$ (`gap` in the diagnostics) bounds the
suboptimality of $x$ and of $\lambda$ at once and does not depend on the returned auxiliary variables.
The gap of the returned point, $\mathcal G_{\rm ret} = F(x, y, z) - W(\lambda)$ (`gap_returned`),
uses the $y$ and $z$ actually returned; for a primal-feasible returned point and a dual-feasible
$\lambda$, $0 \le \mathcal G_x \le \mathcal G_{\rm ret}$, and the difference is
the excess cost of those variables. A solution with the correct design variables and the correct dual
value but a wrong $z$ has $\mathcal G_x$ at rounding level and $\mathcal G_{\rm ret}$ large
(0.49 in the vertex case of the verification), which is why convergence on the $a \ne 0$ path
requires the second gap.

**Decomposition of the gap.** For a dual-feasible $\lambda$ and a primal-feasible point, with
$\ell_\lambda(x)$ the $x$-dependent part of the Lagrangian and $h_i(t) = (c_i - \lambda_i) t + d_i t^2/2$,

$$
\mathcal G_{\rm ret} = \big[\ell_\lambda(x) - \ell_\lambda(x(\lambda))\big] + \sum_i \big[h_i(y_i) - h_i(y_i(\lambda))\big] + \sum_i \lambda_i s_i + \zeta z,
$$

with $s_i = y_i + b_i + a_i z - g_i(x)$ and $\zeta = a_0 - a^{\mathsf T}\lambda$; the four terms are
nonnegative (minimality, minimality, primal feasibility, dual feasibility). At the analytical minimizer
the first term vanishes, and for $d_i = 0$ the second equals $\sum_i \mu_i y_i$ with
$\mu_i = c_i - \lambda_i$, so that $\mathcal G_{\rm ret} = \sum_i(\lambda_i s_i + \mu_i y_i) + \zeta z$,
a sum of complementarity products. This identity is what `gap_allowed` bounds: each product is at most
`tol` $\kappa_i$ times $\max(c_i + d_i y_i, \lambda_i)$, plus the classification slack of the
multipliers near a bound and the rounding of $F$ and $W$. For a point that does not minimize the
Lagrangian in $x$, such as an interior-point iterate, the first term is present and the gap is not a
sum of complementarity products; the diagnostics evaluate the gap directly from $F$ and $W$ in every
case.

**Certificate.** `certified` is `gap_returned <= gap_allowed`; for $a \ne 0$ it also requires the
violation of the relaxed constraints by the returned point to be within `tol` times the largest scale,
and `converged` requires both the tolerance and the certificate. For $a = 0$ the certificate is
reported but does not control termination (the stopping rule of 0.1.x is kept). Negative values of
$\mathcal G_{\rm ret}$ at rounding level occur when the returned point is slightly infeasible
($s_i < 0$ of order $10^{-13}$ with $\lambda_i$ near $c_i$), which is why the diagnostics report the
violation next to the gap.

## 6. Comparison with `mmapy.subsolv`

`mmapy.subsolv` solves the KKT conditions of the subproblem in the full primal–dual space
$(x, y, z, \lambda, \xi, \eta, \mu, \zeta, s)$, $3n + 4m + 2$ unknowns, by a Newton method on the
perturbed complementarity equations with a barrier parameter $\epsilon$ decreased from 1 to
$\epsilon_{\min}$ ($10^{-7}$ inside `mmapy.mmasub`), with at most 200 Newton iterations per barrier
level. For $m < n$ the Newton system is condensed algebraically to an $(m+1) \times (m+1)$ system in
$(\Delta\lambda, \Delta z)$, assembled in $O(nm^2)$ operations and solved densely; for $m \ge n$ it
is condensed to $(n+1) \times (n+1)$. The dual method reduces the problem to $m$ unknowns from the
outset and takes a few Newton steps (8 to 11 in the benchmark campaign), each costing one or more
evaluations of the dual function, $O(nm)$ each, plus the Hessian, $O(nm^2)$, and its decomposition,
$O(m^3)$. The leading orders per iteration are therefore the same for $m < n$; the measured
differences in time come from the constants and the number of iterations of the two implementations.

The two solvers also differ in what they return. The interior-point iterate keeps the complementarity
products of the design bounds at the barrier level, $\xi_j(x_j - \alpha_j) \simeq \varepsilon$, so a
design variable stops at a distance of about $\varepsilon/\xi_j$ from a bound with multiplier $\xi_j$;
with small multipliers (of the order of $\lambda/(0.5 n)$ for the volume constraint of a topology
optimization with $n$ elements) that distance can be appreciable. The dual solver returns the
analytical minimizer, exactly at the bound. This property belongs to the analytical elimination, not
to the active-set treatment of the dual: a dual interior-point solver such as that of Aage and Lazarov
(2013), whose barrier acts on the multipliers only, returns the same minimizer. Exact bound attainment
in a subproblem does not by itself guarantee a better optimum of the original problem.

## References

Svanberg, K. (1987). The method of moving asymptotes: a new method for structural optimization. *Int. J. Numer. Methods Eng.* 24, 359–373.
Svanberg, K. (2002). A class of globally convergent optimization methods based on conservative convex separable approximations. *SIAM J. Optim.* 12, 555–573.
Svanberg, K. (2007). MMA and GCMMA: two methods for nonlinear optimization. Technical note, KTH.
Aage, N., Lazarov, B. S. (2013). Parallel framework for topology optimization using the method of moving asymptotes. *Struct. Multidisc. Optim.* 47, 493–505.
Deetman, A. (2025). GCMMA-MMA-Python, version 0.3.1. Zenodo, 10.5281/zenodo.15459165.
