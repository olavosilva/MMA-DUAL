# MMA-DUAL 0.2.0: mathematical equivalence and verification report

Branch `refactor-package`, compared with the released version 0.1.2 (kept as
`benchmarks/mmadual_v012_reference.py`, identical to the file of tag `v0.1.2`) and with GCMMA-MMA-Python
(`mmapy`) 0.3.1. Environment of all runs: Intel Xeon 2.30 GHz (cloud VM, 2 logical cores), Linux 6.18,
Python 3.13.18, numpy 2.5.3 with OpenBLAS 0.3.34 on one thread, scipy with its own OpenBLAS (one thread),
mmapy 0.3.1, PyPardiso for the MBB example. The host was shared and busier than during the 0.1.2
measurements: absolute times are 1.5 to 2 times those reported in the README tables; ratios between
solvers measured in the same run are the quantity to compare.

## 1. What was reimplemented and what was reorganized

| Component | 0.1.2 | 0.2.0 |
|---|---|---|
| Asymptotes, bounds, coefficients of the MMA subproblem | copied from `mmapy.mmasub` | `approximations.moving_asymptotes`, `variable_bounds`, `separable_coefficients`, `mma_subproblem`, written from Svanberg (1987) with the parameters of the 2007 MATLAB code |
| GCMMA subproblem ($r_0$, $r$, `raa0`, `raa`) | copied from `mmapy.gcmmasub` | `approximations.gcmma_subproblem` |
| `asymp`, `concheck`, `raaupdate`, `kktcheck` | not in the package (imported from `mmapy` by the example) | `gcmma.py`, written from Svanberg (2002) |
| Dual solver | `subsolv_dual`, 107 lines with nested functions | `dual_solver.py`: `primal_x`, `primal_y`, `dual_function`, `dual_hessian`, `solve_dual`, `recover_multipliers`; same algorithm |
| Diagnostics | none in the package (in the benchmark script) | `diagnostics.py` |
| $a \ne 0$ | `NotImplementedError` | solved (Section 4) |

## 2. Equivalence of the subproblems

Reference: the coefficients `low, upp, alfa, beta, p0, q0, P, Q, b` that `mmapy` 0.3.1 passes to its
`subsolv`, captured by intercepting that call (`tests/data/make_reference.py`), for five MMA cases
(iterations 1 and 2; iteration 3 with oscillating, monotone and still variables and five constraints;
iteration 7 with variables at both bounds and asymptotes beyond the admissible band; iteration 3 with
bounds of different ranges including one below the $10^{-5}$ floor, and non-default `asyinit`,
`asydecr`, `asyincr`, `raa0`, `albefa`, `move`) and one GCMMA case (with `asymp` at iterations 1 and 3,
`concheck`, `raaupdate`). Stored in `tests/data/reference_mmapy.npz` and compared by
`tests/test_approximations.py`.

Result: every coefficient agrees to a relative difference below $10^{-13}$ (the tolerance of the test);
the GCMMA values `f0app`, `fapp` at the reference point, which are differences of $O(10)$ terms of
magnitude $10^{-7}$, agree to $5\times10^{-15}$ absolute. The same test is repeated live against the
installed `mmapy` on fresh random data. One behavioural difference was found and is documented:
`mmapy.raaupdate` modifies its `raa` argument in place; `mmadual.raaupdate` returns a new array.

## 3. Dual solver

- `test_same_solution_as_v012`: on four subproblems (regular, poorly scaled $10^{-3}$, strongly violated,
  $d = 1$) the reorganized solver returns $x$ and $\lambda$ equal to those of 0.1.2 within $10^{-12}$.
- Gradient of $W$ against central differences at interior points: agreement to $10^{-6}$ relative
  (15 points, 5 subproblems). Hessian against central differences of the gradient on points where the
  set of free variables is locally constant: agreement to $10^{-5}$ relative, symmetry, negative
  semidefiniteness (at least 5 points per subproblem). Across a change of the active set: the gradient
  change over an interval of $4\times10^{-5}$ is bounded by the curvature times the interval, the finite
  difference of $W$ equals the mean gradient, and the Hessian jumps (`test_transition_between_free_and_bounded`).
- KKT residuals of the dual solutions (`test_diagnostics.py`, `test_edge_cases.py`): stationarity in
  $x$ to $10^{-12}$, zero violation of the constraints with the artificial variables, complementarity
  below $c \cdot$ tolerance, duality gap below $10^{-6}$ relative, in regular, violated ($\lambda = c$,
  $y > 0$), $d > 0$, extreme-scale and degenerate cases (redundant constraints, more constraints than
  variables, no free variable at the start, multiplier at $c$ with zero gradient).
- Against `mmapy.subsolv` on random subproblems: the exact-penalty objective of the dual solution is
  never larger than the interior-point one by more than $10^{-6}$ relative; the gap of the dual solution
  is never larger than the gap of the interior-point one.

### 3.1 Components without curvature in the $a = 0$ path (`_flat_step`)

The only change to the 0.1.x iteration, made after the second review and authorized separately. A
multiplier whose constraint has no free design variable has $H_{ii} = 0$ and $W$ is affine in it until a
variable leaves its bound; 0.1.x stepped by the gradient itself, a length $|\partial W/\partial\lambda_i|$
in the unit of $W$, so an exactly affine constraint with slope 0.01 and $c_i = 1.8$ needed 180 steps and
stopped at `max_newton` (cases 87, 133, 205 of `benchmarks/minimax_stress.py 5291 350 box`). The step of
such a component is now to its bound in the direction of the gradient ($c_i - \lambda_i$ upwards, with
$d_i |g_i|$ beyond $c_i$ when $d_i > 0$ since the next piece has curvature $1/d_i$; $\lambda_i$ downwards),
damped by $1/\max(1, \mu)$ on rejection, as in the polytope path. Everything else is as before: the
Jacobi-scaled Newton/LM step of the curved components, the projection onto the box, the Armijo test and
the active-set changes through the projected gradient, since a variable may leave its bound along the step
and the Hessian stop being zero. A trial point identical to the one just rejected is no longer evaluated
again (the damping of the flat step is inactive below $\mu = 1$, so the trial point repeats until then;
the same repetition occurred in 0.1.x when a long Newton step was clipped at the same point).

Comparison with 0.1.2 on 315 subproblems (300 random: $n$ 20 to 1500, $m$ 1 to 16, objective scales 1,
$10^{-3}$, $10^3$, 40 % violated, $d \in \{0, 1\}$, small $c_i$ in 30 %; 15 from the degenerate generators
of `test_edge_cases.py`: constraint scales $10^{-6}$ to $10^6$, no free variable at the start, violated,
$d = 1$):

- the flat step was not used in 162 of them: $x$ and $\lambda$ identical to 0.1.2 (difference exactly
  zero), as expected since no other operation changed;
- it was used in 153 (the first step from $\lambda = 0$ is flat whenever every $x_j(0)$ is at a bound, as
  with a large objective gradient or strongly violated constraints): the dual values $W(\lambda)$ coincide
  with those of 0.1.2 to $1.2\times10^{-15}$ relative in every case, the duality gaps of both are within
  the bound that the tolerance allows in every case, and the gap of the new iterate is smaller in 51
  cases, equal in 53 and larger in 49 (differences within the tolerance band: in the extreme-scale cases,
  for instance, the two runs end at the same $\lambda$ to four digits and the same $W$ to $10^{-15}$, and
  the gaps $8\times10^{-2}$ against $2\times10^{-6}$, or the reverse, come from where each stops inside the
  band $10^{-10}\max(1, |b_i|, g_i(x(0)))$ of the $10^7$-scaled constraint; at `tol` $= 10^{-12}$ both give
  the same gap). Newton steps: median 6, at most 13; dual-function evaluations 1430 in total against 2526
  for 0.1.2 (the affine cases that needed up to 100 steps now take 1; 22 cases take 1 to 2 steps more than
  0.1.2, when the step to the bound overshoots the point where variables become free and Newton has to
  return; at most 13 evaluations more). In the 162 cases where the step was not used the evaluations are
  2620 against 2780, from the trial-point cache alone.
- time: median ratio 0.97 (flat step used) and 1.05 (not used) against 0.1.2 on these small subproblems
  (the 5 % is the certificate and the package layering, see the benchmarks of Section 6 for the sizes of
  interest, where the two are within noise).

Tests: `test_flat_components_reach_the_bound` (the three cases: certified convergence, $\lambda$ at the
bounds to $10^{-12}$, at most 3 steps; they take 1); `test_same_solution_as_v012` still at $10^{-12}$
(flat step not triggered on its four subproblems); `test_scale_invariance`, `test_degenerate_subproblems`
and the rest of the suite unchanged.

## 4. $a \ne 0$

$W(\lambda) = \phi(\lambda) + \min_{z \ge 0}(a_0 - a^{\mathsf T}\lambda) z$, so the dual domain is the box
intersected with $a^{\mathsf T}\lambda \le a_0$ and $z$ is the multiplier of that constraint
($\nabla\phi_F = z\, a_F$ at the solution). Implementation: `project_dual` (bisection on the multiplier of the half-space; exact since $a \ge 0$
makes $t \mapsto a^{\mathsf T}\mathrm{clip}(v - ta)$ monotone), Newton/LM step in the tangent space of
the half-space when it is active (orthonormal basis of the null space of $a_F^{\mathsf T}$), and $z$
determined together with the box active set: since a component of $\lambda$ at a bound is active or not
according to the sign of $\partial W/\partial\lambda_i - z a_i$, $z$ is taken as the minimizer over $z \ge 0$
of the squared projected gradient of $W - z\,a^{\mathsf T}\lambda$, a convex piecewise-quadratic function
of $z$ whose derivative is monotone (bisection, `_halfspace_multiplier`); $z = 0$ means the half-space is
not active. A first version estimated $z$ by least squares on a free set computed from the raw gradient,
which fails at vertices of the dual domain where a component at its lower bound has a positive raw
gradient but a negative corrected one (counterexample of the review: $W = C + 0.2\lambda_1 + \lambda_2$ on
the simplex, optimum $(0, 1)$ with $z = 1$, returned $z = 0.6$); the counterexample and six other affine
dual functions on the simplex are now regression tests (`test_affine_dual_on_the_simplex`), together
with a brute-force check of the joint minimization (`test_halfspace_multiplier_joint_active_set`).
`diagnostics.optimal_artificial` minimizes $a_0 z + \sum_i [c_i y_i + d_i y_i^2/2]$ in $z$ (breakpoints
for $d = 0$, golden section otherwise) so that the exact-penalty objective is defined for $a \ne 0$.

Two certificates are distinguished: `duality_gap` uses $f_{\rm pen}(x)$ with the artificial variables
re-optimized for $x$ and certifies $x$; `duality_gap_returned` uses $F(x, y, z)$ with the artificial
variables actually returned and certifies the whole primal point (it was 400 on the counterexample while
the first was 0, which is how the failure escaped the original tests). `check_solution` reports both.

### 4.1 Second review: stalls near the half-space and at vertices (active-set iteration)

The second review solved 350 subproblems whose dual function is affine to $10^{-10}$ (every $x_j$ held at
its upper bound, $p_0 \ll q_0$, negligible $P, Q$; 2 to 8 constraints, $a_i \ge 0$ with zero components,
distinct $c_i$, $d = 0$) and compared with the linear program solved by HiGHS: six did not converge in
100 Newton steps and three (cases 22, 118, 278 of seed 5291) not even in 500. Reproduced here, and the
two mechanisms confirmed:

- Case 22: optimal dual value to $10^{-12}$, but $\lambda_2 = c_2 - 4\times10^{-13}$ (left there by the
  bisection of the projection) was classified as free by the exact comparisons, so $z$ and the
  complementarity of the returned point were wrong (0.016).
- Cases 118 and 278: the projection onto the half-space of a long Newton step (the Hessian is at the
  level of rounding, so the Jacobi-scaled Newton direction is arbitrary and of length $10^{16}$) moves the
  iterate along the half-space and decreases $W$; the step is rejected until $\mu \approx 10^{16}$, where
  the damped step is a gradient step of length $10^{-4}$: the iteration crawls and stops $3\times10^{-5}$
  and $10^{-7}$ inside the half-space, where it is classified as inactive ($z = 0$). A third mechanism,
  found while fixing these: a component without curvature took a gradient step of length $|\partial W/\partial\lambda_i|$
  (the rule of 0.1.x for flat components), which with a slope of $0.01$ and a bound at distance $1.8$
  needs 180 steps.

The polytope path (`_solve_polytope`, used only when $a \ne 0$; `_solve_box`, used when $a = 0$, is the
0.1.x iteration, which this commit did not touch; its flat step was changed afterwards, Section 3.1) is now
an active-set projected Newton method:

1. Classification with tolerances: a multiplier within $10^{-12}\max(1, c_i)$ of a bound counts as at the
   bound, the half-space within $10^{-12}\max(1, a_0)$ of $a^{\mathsf T}\lambda = a_0$ counts as reached;
   multipliers within the tolerance are snapped onto the bound when that keeps $a^{\mathsf T}\lambda \le a_0$.
   `_halfspace_multiplier` takes the classification as input.
2. Direction: Newton/LM on the free multipliers, in the tangent space of the half-space when it is
   active ($z > 0$); a free multiplier at a bound whose direction points outward is fixed and the
   direction recomputed (at most $m$ times); the same for the half-space when the iterate is on it, it is
   inactive and the direction leaves the domain. Reduced directions without curvature take the gradient
   scaled to reach the first bound exactly (damped by $1/\max(1, \mu)$ on rejection), not a unit step.
3. Step: along the direction up to the first bound or the half-space (ratio test), never beyond, so that
   no non-gradient direction is projected; the trial point is projected onto the domain (only rounding is
   removed) and accepted under the Armijo condition with $\nabla W^{\mathsf T}\Delta\lambda > 0$ required.
   A trial point identical to the previous one is not re-evaluated. When every damping fails, a
   projected-gradient arc with backtracking from the first bound is used.
4. Stationarity: both the projected gradient of $W - z\,a^{\mathsf T}\lambda$ on the identified active set
   and the full-domain projection $\|\lambda - P_D(\lambda + \nabla W)\|$, which depends on no active set,
   must be below `tol` relative to the constraint scales.
5. Certification: the gap $F(x, y, z) - W(\lambda)$ of the returned point, computed directly (not from the
   complementarity identity), must be below what the tolerance allows,
   $\mathrm{tol}\sum_i \max(c_i + d_i y_i, \lambda_i)\,s_i$ plus the classification slack and the rounding of
   $F$ and $W$ (the gap equals $\sum_i [(c_i + d_i y_i - \lambda_i) y_i + \lambda_i \max(0, -u_i)] + (a_0 - a^{\mathsf T}\lambda) z$
   with $u = g(x) - b - a z$, a sum of complementarity products); otherwise `converged` is False with
   reason `'certification'`. The $a = 0$ path reports the same quantities (`gap_returned`, `gap_allowed`,
   `certified`) for information and keeps its stopping rule.

Results. The three cases: converged in 4, 7 and 8 Newton steps (5, 8, 9 evaluations), dual value equal to
the LP optimum to $10^{-14}$, complementarity $10^{-14}$, returned gap zero to rounding. The review's
generator over 2100 cases (seeds 5291, 1 to 5, 350 each, `max_newton = 100`): no failure, at most 9 Newton
steps (median 3), LP value gap at most $3\times10^{-14}$, complementarity $3\times10^{-14}$, returned gap
$4\times10^{-15}$ relative. 400 curved minimax subproblems from the MMA approximation ($n$ 5 to 300, $m$ 1
to 11, $a_i$ all one, random or mixed with zeros, $d \in \{0, 0.01, 1\}$, regular and violated, small
$c_i$ and small $a_0$ in 30% of them): all converged and certified, median 9 Newton steps (maximum 73, with
$d = 0.01$ and $z \approx 10^2$), exact-penalty objective never worse than `mmapy.subsolv` at
`epsimin` $10^{-11}$ by more than $1.5\times10^{-6}$ relative (the level allowed by `tol` $= 10^{-10}$ with
$c = 10^3$ and the constraint scales of the violated cases; in those cases the half-space was inactive).

The third mechanism also existed in the $a = 0$ path: on the same generator with $a = 0$ (optimum
$\lambda_i = c_i$ for positive slopes, $0$ otherwise), 3 of 350 exactly affine cases stopped at
`max_newton` with unit gradient steps (`benchmarks/minimax_stress.py box`). It was corrected afterwards,
in a separate commit, by the same rule (Section 3.1): all 350 cases now converge in one step.

Tests (`tests/test_minimax.py`, 27 tests; `tests/test_minimax_independent_stress.py`, 7 tests): the
review's three cases with the review's thresholds and 600 cases of its generator against `linprog`
(convergence certified, LP value to $10^{-9}$, KKT to $10^{-9}$, at most 20 steps); the vertex case with
the tolerance classification; the half-space reached from inside on affine duals with a zero $a_i$;
the certificate against the independent `diagnostics.duality_gap_returned` on curved subproblems; 18
random configurations with $a_i = 1$, random $a_i \in [0, 2]$, mixed zero and positive $a_i$, regular and
violated constraints, $d = 1$; 7 affine cases on the simplex including redundant constraints, a whole
optimal edge and an inactive half-space; the projection; the brute-force check of $z$: dual feasibility,
$z \ge 0$ only on the boundary, full KKT residuals with the returned variables including
complementarity, both gaps below $c$ times the tolerance, and, against `mmapy.subsolv`, exact-penalty
objective not worse and $z$ equal to $10^{-3}$ relative. The $a = 0$ path is not affected by the
polytope path (`test_a_zero_limit_unchanged`: $a = 0$ goes through `_solve_box`; the identity with 0.1.2 of
Section 3 where the flat step of Section 3.1 is not triggered; the certificate reported in
`test_certificate_reported_for_a_zero`).

## 5. GCMMA

`tests/test_gcmma_loop.py`: a complete GCMMA run on Svanberg's beam with `mmadual` only reaches the
optimum to $2\times10^{-3}$ in 25 outer iterations with inner iterations; the objective decreases
strictly while the constraint is inactive and never increases by more than `epsimin` afterwards; the
same loop driven by `mmapy.asymp`/`concheck`/`raaupdate` produces the same iterates to $10^{-10}$ with
the same number of inner iterations; `kktcheck` equals `mmapy.kktcheck` to $10^{-12}$.
`examples/toy_problem.py`: MMA 9 iterations, GCMMA 16 outer + 13 inner, both at $f_0 = 1.339956$,
$x = (6.016, 5.309, 4.494, 3.502, 2.153)$; `mmapy.mmasub` differs by $4\times10^{-7}$ in $x$.

## 6. Benchmarks (same subproblems, same run; median of 5)

Regular scaling, `python benchmarks/bench_n_m.py v012` (subset $m \in \{4, 16\}$):

| $n$ | $m$ | `mmapy` | build | solve | `mmadual` 0.2.0 | ratio | 0.1.2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $10^3$ | 4 | 0.034 | 0.0003 | 0.0064 | 0.0067 | 5.0 | 0.0065 |
| $10^3$ | 16 | 0.033 | 0.0005 | 0.0096 | 0.0101 | 3.3 | 0.0094 |
| $10^4$ | 4 | 0.129 | 0.0014 | 0.0153 | 0.0166 | 7.7 | 0.0161 |
| $10^4$ | 16 | 0.193 | 0.0038 | 0.0311 | 0.0349 | 5.5 | 0.0357 |
| $10^5$ | 4 | 3.02 | 0.019 | 0.153 | 0.173 | 17.5 | 0.178 |
| $10^5$ | 16 | 2.85 | 0.052 | 0.381 | 0.433 | 6.6 | 0.561 |

The construction of the subproblem is 5 to 12 % of the time of `mmasub`; 0.2.0 and 0.1.2 agree within
the noise (the one 25 % difference, $n = 10^5$, $m = 16$, reverses between repetitions).

### 6.1 Poorly scaled subproblems with the matched-gap comparison

("Matched": the dual solver rerun with the loosest tolerance, over decades, whose absolute duality gap is no
larger than that of `mmapy` on the same subproblem; the gaps are not made equal.)

`python benchmarks/bench_n_m.py 1e-3 matched v012` (subset $m \in \{4, 16\}$; `mmapy` was 2 to 3 times
slower than in the README table because of the host load, `mmadual` 1.5 to 2 times):

| $n$ | $m$ | `mmapy` | `mmadual` 0.2.0 | ratio | 0.1.2 | tol matched | ratio matched | $\gamma_{\rm pd}$ | $\gamma_{\rm du}$ | $\delta$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $10^3$ | 4 | 1.89 | 0.0050 | 379 | 0.0043 | $10^{-7}$ | 424 | $5.3\times10^{-3}$ | $8\times10^{-7}$ | $4.9\times10^{-3}$ |
| $10^3$ | 16 | 1.83 | 0.0076 | 241 | 0.0074 | $10^{-10}$ | 229 | $3.6\times10^{-6}$ | $3\times10^{-10}$ | $3.6\times10^{-6}$ |
| $10^4$ | 4 | 6.79 | 0.0171 | 396 | 0.0168 | $10^{-7}$ | 418 | $6.7\times10^{-2}$ | $2\times10^{-6}$ | $6.2\times10^{-2}$ |
| $10^4$ | 16 | 11.05 | 0.0377 | 293 | 0.0361 | $10^{-10}$ | 201 | $2.2\times10^{-3}$ | $1\times10^{-9}$ | $2.0\times10^{-3}$ |
| $10^5$ | 4 | 113.2 | 0.195 | 579 | 0.185 | $10^{-7}$ | 613 | $1.5\times10^{-1}$ | $2\times10^{-6}$ | $1.4\times10^{-1}$ |
| $10^5$ | 16 | 175.7 | 0.516 | 341 | 0.560 | $10^{-6}$ | 366 | $6.3\times10^{-2}$ | $1\times10^{-9}$ | $5.7\times10^{-2}$ |

Same picture as with 0.1.2: zero violation and projected-gradient stationarity below $10^{-17}$ for the
dual solutions ($2\times10^{-3}$ to $4\times10^{-3}$ for the interior-point ones), gaps of the dual
solutions $10^{-10}$ to $10^{-6}$ against $10^{-6}$ to $0.15$, speedups at matched gap within 30 % of
those at the default tolerance (the largest deviation, $n = 10^4$, $m = 16$, 293 against 201, is timing
noise on a 40 ms measurement: the matched run used the default tolerance). 0.2.0 and 0.1.2 agree within
the noise (rerun after the active-set iteration of Section 4.1: $a = 0$ unaffected, e.g. $n = 10^5$, $m = 16$: 0.455 s against 0.438 s of 0.1.2, $m = 64$: 2.25 s against 2.22 s).

MBB beam (`examples/mbb_topopt.py 60 20 500 1.5 0.05`, PyPardiso): subproblem ratio 9.3, final
compliances 210.83 (`mmapy`, 127 iterations) and 210.66 (`mmadual`, 114 iterations), as with 0.1.2.

Rerun after the flat step of Section 3.1 (same seeds, same host), eight matched instances and the MBB:

| scale | $n$ | $m$ | `mmapy` | `mmadual` | ratio | 0.1.2 | tol matched | ratio matched | $\gamma_{\rm pd}$ | $\gamma_{\rm du}$ | $\gamma_{\rm mat}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $10^{-3}$ | $10^4$ | 4 | 6.95 | 0.0172 | 404 | 0.0163 | $10^{-7}$ | 448 | $6.8\times10^{-2}$ | $2.7\times10^{-6}$ | $2.7\times10^{-6}$ |
| $10^{-3}$ | $10^4$ | 16 | 10.32 | 0.0457 | 226 | 0.0413 | $10^{-10}$ | 256 | $1.8\times10^{-3}$ | $1.4\times10^{-9}$ | $1.4\times10^{-9}$ |
| $10^{-3}$ | $10^5$ | 4 | 107.4 | 0.174 | 616 | 0.182 | $10^{-7}$ | 596 | $1.5\times10^{-1}$ | $2.0\times10^{-6}$ | $2.0\times10^{-6}$ |
| $10^{-3}$ | $10^5$ | 16 | 166.5 | 0.474 | 352 | 0.488 | $10^{-6}$ | 369 | $6.3\times10^{-2}$ | $1.5\times10^{-9}$ | $8.4\times10^{-3}$ |
| 1 | $10^4$ | 4 | 0.143 | 0.0168 | 8.5 | 0.0165 | $10^{-8}$ | 7.7 | $9.1\times10^{-8}$ | $5.2\times10^{-9}$ | $1.6\times10^{-11}$ |
| 1 | $10^4$ | 16 | 0.198 | 0.0368 | 5.4 | 0.0391 | $10^{-10}$ | 5.5 | $8.7\times10^{-9}$ | $8.5\times10^{-10}$ | $2.3\times10^{-12}$ |
| 1 | $10^5$ | 4 | 2.86 | 0.177 | 16.1 | 0.175 | $10^{-8}$ | 16.1 | $9.2\times10^{-8}$ | $1.0\times10^{-10}$ | $1.0\times10^{-10}$ |
| 1 | $10^5$ | 16 | 2.79 | 0.423 | 6.6 | 0.448 | $10^{-11}$ | 5.2 | $8.4\times10^{-9}$ | $4.6\times10^{-7}$ | $1.1\times10^{-12}$ |

Zero violation and stationarity below $10^{-13}$ for every dual solution, 0.2.0 within the noise of 0.1.2
in every cell (ratios 0.95 to 1.06), the same accuracy picture as before. The flat step is not triggered on
these subproblems (every constraint keeps free variables), so the dual iterates are those of 0.1.2. The
regular table of Section 6 was also rerun with the same outcome (for instance $n = 10^5$, $m = 64$: 1.94 s
against 1.90 s of 0.1.2; $m = 1$: 0.110 against 0.109). At scale $10^{-3}$ the `matched` option raises
"no matched absolute gap" for $n = 10^3$, $m = 64$ (not part of the eight instances): the default dual
run already stops with a residual $6\times10^{-13}$, its gap $9.5\times10^{-6}$ against $6.4\times10^{-6}$
of `mmapy` does not change down to `tol` $= 10^{-12}$, and 0.1.2 gives the same numbers; the loop over
decades would need $10^{-13}$. Pre-existing and independent of the change; left as is.

MBB beam rerun: `mmapy` 127 iterations, compliance 210.829178; `mmadual` 114 iterations, 210.661293;
subproblem time per iteration 0.033 against 0.0043 s (ratio 7.8); identical to the run before the change.


Subproblem accuracy (`benchmarks/mbb_subproblem_accuracy.py`, 120 x 40, iteration 31): median density
at the lower bound $1.2\times10^{-3}$, $1.2\times10^{-5}$, $1.2\times10^{-7}$ for `epsimin` $10^{-7}$,
$10^{-9}$, $10^{-11}$ (proportional to `epsimin`, as $x_j - \alpha_j = \epsilon/\xi_j$), gaps
$4.5\times10^{-4}$ to $4.5\times10^{-8}$; dual solver: zero and $10^{-15}$, 6 Newton steps, 6 ms against
57 to 64 ms.

## 7. Test suite

`pytest`: 74 tests, all passing (about 50 s; Python 3.13, numpy 2.5.3, scipy, mmapy 0.3.1). `LinAlgWarning` raised by scipy inside `mmapy.subsolv` on the
degenerate reference runs is filtered in `pyproject.toml` with a comment; it is not raised by `mmadual`.
