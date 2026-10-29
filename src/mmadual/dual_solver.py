"""
Solution of the MMA/GCMMA subproblem by the dual method (Svanberg 1987, Section 5).

For fixed multipliers lam >= 0 of the m constraints, the Lagrangian of the subproblem (see
approximations.py) is separable and is minimized in closed form in each design variable,

    pj = p0j + sum_i lam_i Pij,  qj = q0j + sum_i lam_i Qij,
    xj(lam) = clip( (sqrt(pj) Lj + sqrt(qj) Uj) / (sqrt(pj) + sqrt(qj)), alfa_j, beta_j ),

in the artificial variables yi (yi = max(0, (lam_i - ci)/di) for di > 0; for di = 0 the dual is finite only
for lam_i <= ci, which becomes an upper bound of the multiplier) and in z (z = 0 when a = 0). The dual
function W(lam) is concave; its gradient is the residual of the constraints at the primal minimizer,

    dW/dlam_i = gi(x(lam)) - yi - bi,

and its Hessian, assembled analytically from the variables strictly inside (alfa_j, beta_j),

    d2W = - sum_{j free} G_j G_j^T / Dj - diag([lam_i > ci]/di),
    Gij = Pij/(Uj - xj)^2 - Qij/(xj - Lj)^2,   Dj = 2 pj/(Uj - xj)^3 + 2 qj/(xj - Lj)^3,

is negative semidefinite and discontinuous where a variable enters or leaves a bound. W is maximized
by a projected Newton method with Levenberg-Marquardt damping: the Hessian block of the free multipliers
is brought to the Jacobi scaling S H S, S = diag(|H_ii|^(-1/2)), and decomposed in eigenvalues -w_k; the
step S V diag(1/(w_k + mu w_max)) V^T S grad is the Newton step for mu -> 0 and a scaled gradient step
for mu -> infinity (components without curvature take the gradient, damped by the same mu). The step is
projected onto the box and accepted under an Armijo condition, mu growing by 10 on rejection and
shrinking by 100 on acceptance. Each evaluation costs O(n m), the Hessian O(n m^2) plus an O(m^3)
eigendecomposition; the method is meant for m << n.

Stopping: the projected gradient of each constraint, relative to its scale max(1, |b_i|, g_i(x(0))), is
below `tol`; or two consecutive undamped Newton steps increase W by less than 1e-13 |W| (the maximum is
resolved to the rounding of W); or `max_newton` steps. The `info` dict returned by `solve_dual` says which.
This is the iteration of version 0.1.x (`_solve_box`), used whenever a = 0, with one change: a free multiplier
without curvature (no free design variable in its constraint, W affine in it) steps to its bound instead of
taking the gradient as a step of unit length (`_flat_step`), damped like the rest on rejection.

With a != 0 (a_i >= 0, the minimax form of Svanberg), the minimization in z >= 0 gives W(lam) = phi(lam) on
the half-space a^T lam <= a0 and -infinity outside, phi being the dual function with z = 0; the dual domain
D is the box intersected with that half-space. The maximization on D (`_solve_polytope`) is an active-set
projected Newton method:

  * the multipliers within BOUND_TOL max(1, c_i) of a bound, and the half-space within BOUND_TOL max(1, a0)
    of a^T lam = a0, count as at the bound (no exact comparisons); the multipliers are snapped onto the bounds
    when that keeps a^T lam <= a0;
  * on the half-space, its multiplier z >= 0 and the box active set are determined together: z minimizes the
    norm of the projected gradient of W - z a^T lam, a convex piecewise-quadratic function of z, by bisection
    (`_halfspace_multiplier`); z > 0 means the half-space is active;
  * the Newton/Levenberg-Marquardt direction is computed on the free multipliers, in the tangent space of
    the half-space when it is active; a free multiplier at a bound whose direction points outward is fixed
    and the direction recomputed (at most m times);
  * the step goes along that direction up to the first bound or the half-space (ratio test), never beyond,
    so that no non-gradient direction is projected: the projection of a long Newton step onto D can decrease W
    and stall the iteration a small distance from the half-space. The trial point is projected onto D exactly
    (`project_dual`, which only removes rounding) and accepted under the Armijo condition; when every damping
    fails, a projected-gradient arc with backtracking is used;
  * stationarity is measured both by the projected gradient of W - z a^T lam on the identified active set and
    by the projection onto the whole domain, |lam - P_D(lam + grad W)|, which depends on no active set; both
    must be below `tol` (relative to the constraint scales);
  * convergence is then certified by the gap F(x, y, z) - W(lam) of the returned primal point (`_certificate`),
    which must be bounded by what the tolerance allows (the gap is a sum of complementarity products, each
    bounded by tol times the scale of its constraint), else `converged` is False with reason 'certification'.
"""
import warnings
import numpy as np

LM_MU_MIN, LM_MU_MAX = 1e-12, 1e30
ARMIJO = 1e-4
ROUNDING_W = 1e-14                 # relative rounding allowance in the Armijo test (W is a sum of n terms)
STAGNATION_W = 1e-13               # two undamped steps below this relative gain: resolved to rounding
BOUND_TOL = 1e-12                  # relative distance to a bound below which a variable/multiplier counts as bounded
ARC_HALVINGS = 60                  # backtracking steps of the projected-gradient arc (fallback of the polytope path)


def check_subproblem(sub):
    "basic consistency of the subproblem dict and the restriction to a = 0"
    if np.any(sub['a'] < 0):
        raise ValueError("mmadual: the coefficients a_i must be >= 0 (Svanberg's form; the subproblem is not bounded otherwise)")
    if sub['a0'] < 0:
        raise ValueError("mmadual: a0 must be >= 0 (the subproblem is unbounded in z otherwise)")
    if np.any(sub['low'] >= sub['alfa']) or np.any(sub['beta'] >= sub['upp']):
        raise ValueError("mmadual: the asymptotes must satisfy low < alfa and beta < upp")


def primal_x(sub, lam):
    "minimizer x(lam) of the Lagrangian in x, and the coefficients p(lam), q(lam)"
    p = sub['p0'] + lam @ sub['P']
    q = sub['q0'] + lam @ sub['Q']
    sp, sq = np.sqrt(p), np.sqrt(q)
    x = np.clip((sp*sub['low'] + sq*sub['upp'])/(sp + sq), sub['alfa'], sub['beta'])
    return x, p, q


def primal_y(sub, lam):
    "minimizer y(lam) of the Lagrangian in y (zero for d = 0, where lam <= c is enforced by the box)"
    d = sub['d']
    return np.where(d > 0, np.maximum(0.0, (lam - sub['c'])/np.where(d > 0, d, 1.0)), 0.0)


def dual_function(sub, lam):
    "W(lam), its gradient, and the primal quantities x, y, p, q at lam (z = 0)"
    x, p, q = primal_x(sub, lam)
    y = primal_y(sub, lam)
    ux, xl = sub['upp'] - x, x - sub['low']
    W = np.sum(p/ux + q/xl) - lam @ sub['b'] + np.sum(sub['c']*y + 0.5*sub['d']*y*y) - lam @ y
    grad = sub['P'] @ (1.0/ux) + sub['Q'] @ (1.0/xl) - y - sub['b']
    return W, grad, x, y, p, q


def dual_hessian(sub, lam, x, p, q):
    "analytic Hessian of W at lam (negative semidefinite); only the variables strictly inside (alfa, beta) contribute"
    ux, xl = sub['upp'] - x, x - sub['low']
    width = sub['beta'] - sub['alfa']
    free = (x > sub['alfa'] + BOUND_TOL*width) & (x < sub['beta'] - BOUND_TOL*width)
    G = (sub['P']/ux**2 - sub['Q']/xl**2)[:, free]
    D = 2*p[free]/ux[free]**3 + 2*q[free]/xl[free]**3
    H = -(G/D) @ G.T
    d = sub['d']
    H -= np.diag(np.where((d > 0) & (lam > sub['c']), 1.0/np.where(d > 0, d, 1.0), 0.0))
    return H


def _scaled_spectrum(Hl):
    """Jacobi scaling S Hl S, S = diag(|H_ii|^-1/2), and eigendecomposition of the scaled block; components with
       H_ii = 0 (no curvature: no free variable in that constraint) are left out and take the gradient"""
    diag = np.abs(np.diag(Hl))
    curved = diag > 0
    if not np.any(curved):
        return curved, None, None, None
    S = 1.0/np.sqrt(diag[curved])
    w, V = np.linalg.eigh((S[:, None]*Hl[np.ix_(curved, curved)])*S[None, :])
    return curved, S, np.abs(w), V


def _lm_step(gl, curved, S, w, V, mu):
    "Levenberg-Marquardt step (-Hl + mu w_max I)^-1 gl in the scaled spectral basis; flat components damped alike"
    step = np.zeros(len(gl))
    step[~curved] = gl[~curved]/(1.0 + mu)
    if S is not None:
        gs = S*gl[curved]
        step[curved] = S*(V @ ((V.T @ gs)/(w + mu*max(np.max(w), 1e-300))))
    return step


def project_dual(v, a, a0, lam_max):
    """projection of v onto {0 <= lam <= lam_max, a^T lam <= a0} (a >= 0): clip(v - t a) with the smallest
       t >= 0 that satisfies the half-space; t -> a^T clip(v - t a) is non-increasing, so t is found by bisection
       on the breakpoints (exact up to rounding; m is small)"""
    lam = np.clip(v, 0.0, lam_max)
    if a @ lam <= a0 or not np.any(a > 0):
        return lam
    t_lo, t_hi = 0.0, 1.0
    while a @ np.clip(v - t_hi*a, 0.0, lam_max) > a0:
        t_hi *= 2.0
        if t_hi > 1e300:
            break
    for _ in range(200):
        t = 0.5*(t_lo + t_hi)
        if a @ np.clip(v - t*a, 0.0, lam_max) > a0:
            t_lo = t
        else:
            t_hi = t
        if t_hi - t_lo <= 1e-15*max(1.0, t_hi):
            break
    return np.clip(v - t_hi*a, 0.0, lam_max)


def _projected_gradient(grad, lam, lam_max, z, a):
    "gradient of W - z a^T lam with the components of lam at a bound that point outward set to zero (exact comparisons; a = 0 path)"
    g = grad - z*a
    active = ((lam <= 0) & (g < 0)) | ((lam >= lam_max) & (g > 0))
    return np.where(active, 0.0, g), active


def _halfspace_multiplier(grad, lower, upper, a):
    """multiplier z >= 0 of the half-space a^T lam <= a0 and the box active set, determined together: z minimizes
       the squared projected gradient r(z)^2 = sum_i phi_i(z) with phi_i = (g_i - z a_i)^2 for lam_i strictly inside
       the box, max(0, g_i - z a_i)^2 at the lower bound and min(0, g_i - z a_i)^2 at the upper bound (`lower`,
       `upper`: boolean arrays of the multipliers at a bound). r^2 is convex and piecewise quadratic in z, so its
       derivative is non-decreasing and the minimizer is found by bisection (a least-squares estimate on a fixed
       free set is wrong when a component at a bound changes status with z)"""
    if not np.any(a > 0):
        return 0.0

    def dr2(z):
        g = grad - z*a
        contrib = np.where(lower, g > 0, np.where(upper, g < 0, True))
        return float(-2*np.sum(a[contrib]*g[contrib]))

    if dr2(0.0) >= 0.0:
        return 0.0
    z_hi = float(np.max(grad[a > 0]/a[a > 0])) + 1.0
    z_hi = max(z_hi, 1.0)
    while dr2(z_hi) < 0.0:
        z_hi *= 2.0
    z_lo = 0.0
    for _ in range(200):
        z = 0.5*(z_lo + z_hi)
        if dr2(z) < 0.0:
            z_lo = z
        else:
            z_hi = z
        if z_hi - z_lo <= 1e-15*max(1.0, z_hi):
            break
    return 0.5*(z_lo + z_hi)


def _certificate(sub, lam, W, grad, x, y, z, scale, tol, eps, eps_plane):
    """gap F(x, y, z) - W(lam) of the point that will be returned (y as recover_multipliers returns it) and the
       largest gap the tolerance allows: the gap equals sum_i [(c_i + d_i y_i - lam_i) y_i + lam_i max(0, -u_i)]
       + (a0 - a^T lam) z with u = g(x) - b - a z, each term a complementarity product bounded by tol scale_i times
       max(c_i + d_i y_i, lam_i) when the projected gradient is below tol scale_i, plus the classification slack
       (multipliers within eps of a bound, half-space within eps_plane) and the rounding of F and W. Also returns
       the violation of the constraints by (x, y, z) (zero when d = 0)"""
    c, d, a, a0 = sub['c'], sub['d'], sub['a'], sub['a0']
    g = grad + y                                                            # g(x) - b (y = y(lam) is subtracted in grad)
    u = g - a*z
    y_ret = np.where(d > 0, y, np.maximum(u, 0.0))
    viol = float(np.max(np.maximum(u - y_ret, 0.0)))
    F = float(np.sum(sub['p0']/(sub['upp'] - x) + sub['q0']/(x - sub['low'])) + a0*z + np.sum(c*y_ret + 0.5*d*y_ret*y_ret))
    gap = F - W
    allowed = (tol*np.sum(np.maximum(c + d*y_ret, lam)*scale) + np.sum(eps*np.abs(u)) + eps_plane*z
               + ROUNDING_W*(abs(F) + abs(W)))
    return float(gap), float(allowed), viol


def solve_dual(sub, tol=1e-10, max_newton=100, warn=True):
    """maximize the dual function on the box 0 <= lam <= c (lam unbounded above where d > 0), intersected with
       the half-space a^T lam <= a0 when a != 0.
       Returns x, y, z, lam (1-D arrays, z float) and an info dict:
         converged      : the projected-gradient tolerance was met (and, for a != 0, the gap certificate holds)
         reason         : 'tolerance', 'rounding' (W cannot be increased further), 'max_newton', 'no_progress',
                          or 'certification' (a != 0: tolerance met but the returned point is not certified)
         newton_steps   : accepted Newton/LM steps
         evaluations    : evaluations of the dual function
         residual       : largest projected gradient, relative to the constraint scales, at exit (for a != 0 the
                          larger of the active-set measure and the full-domain projection measure)
         scale          : the per-constraint scales max(1, |b_i|, g_i(x(0)))
         z, halfspace_active : multiplier of the half-space and whether it is active (a != 0; else 0, False)
         gap_returned   : F(x, y, z) - W(lam) of the returned point (with y as recover_multipliers returns it) and
         gap_allowed    : the largest value consistent with the tolerance; certified = gap_returned <= gap_allowed
                          (informative for a = 0, where the stopping rule of 0.1.x is kept; required for a != 0)
       A RuntimeWarning is issued (warn=True) when the iteration stops without meeting the tolerance and the
       residual is above sqrt(tol), or when the certificate fails."""
    check_subproblem(sub)
    if np.any(sub['a'] > 0):
        return _solve_polytope(sub, tol, max_newton, warn)
    return _solve_box(sub, tol, max_newton, warn)


def _flat_step(g, lam, c, d, mu):
    """step of the free multipliers without curvature (no free design variable in the constraint, so W is affine in
       that multiplier until a variable leaves a bound): to the bound in the direction of the gradient, c - lam
       upwards (for d > 0 the curvature 1/d of the next piece gives d g beyond c) and lam downwards, damped by
       1/max(1, mu) on rejection. 0.1.x took the gradient itself, a step of length |g| in the unit of W, which
       with a small gradient and a distant bound needs many steps (the only change to the a = 0 iteration in 0.2.0)"""
    dist = np.where(g > 0, np.where(lam < c, c - lam, d*np.abs(g)), lam)
    return np.sign(g)*dist/max(1.0, mu)


def _solve_box(sub, tol, max_newton, warn):
    """the iteration of 0.1.x on the box (a = 0). Changes in 0.2.0: the step of the components without curvature
       (_flat_step), and a trial point identical to the one just rejected is not evaluated again (the damping of the
       flat step is inactive for mu < 1, so the trial point repeats until mu reaches 1)"""
    m = sub['b'].size
    c, d, a = sub['c'], sub['d'], sub['a']
    lam_max = np.where(d > 0, np.inf, c)
    lam = np.zeros(m)
    W, grad, x, y, p, q = dual_function(sub, lam)
    scale = np.maximum(1.0, np.maximum(np.abs(sub['b']), grad + sub['b']))
    info = dict(converged=False, reason='max_newton', newton_steps=0, evaluations=1, residual=np.inf, scale=scale,
                z=0.0, halfspace_active=False)
    mu = LM_MU_MIN
    stagnant = 0
    last_trial = None
    for it in range(max_newton + 1):
        gp, active = _projected_gradient(grad, lam, lam_max, 0.0, a)
        free = ~active
        info['residual'] = float(np.max(np.abs(gp)/scale))
        if info['residual'] <= tol:
            info['converged'], info['reason'] = True, 'tolerance'
            break
        if it == max_newton:
            break
        H = dual_hessian(sub, lam, x, p, q)[np.ix_(free, free)]
        g_free = grad[free]
        curved, S, w, V = _scaled_spectrum(H)
        while True:                                                         # LM: raise mu until Armijo holds
            step = np.zeros(m)
            step_free = _lm_step(g_free, curved, S, w, V, mu)
            if not np.all(curved):
                flat = ~curved
                step_free[flat] = _flat_step(g_free[flat], lam[free][flat], c[free][flat], d[free][flat], mu)
            step[free] = step_free
            if not np.all(np.isfinite(step)) or step @ gp <= 0:
                step = gp
            lam_new = np.clip(lam + step, 0.0, lam_max)
            if last_trial is None or not np.array_equal(lam_new, last_trial):   # a trial point already rejected is not re-evaluated
                last_trial = lam_new
                W_new, grad_new, x_new, y_new, p_new, q_new = dual_function(sub, lam_new)
                info['evaluations'] += 1
                if W_new - W >= ARMIJO*(grad @ (lam_new - lam)) - ROUNDING_W*abs(W) or np.all(lam_new == lam):
                    break
            if mu >= LM_MU_MAX:
                lam_new = lam
                break
            mu *= 10.0
        if np.all(lam_new == lam):
            info['reason'] = 'no_progress'
            break
        stagnant = stagnant + 1 if (W_new - W <= STAGNATION_W*abs(W) and mu <= LM_MU_MIN) else 0
        mu = max(mu/100.0, LM_MU_MIN)
        lam, W, grad, x, y, p, q = lam_new, W_new, grad_new, x_new, y_new, p_new, q_new
        last_trial = None
        info['newton_steps'] += 1
        if stagnant >= 2:
            info['reason'] = 'rounding'
            break
    gap, allowed, _ = _certificate(sub, lam, W, grad, x, y, 0.0, scale, tol, np.zeros(m), 0.0)
    info['gap_returned'], info['gap_allowed'], info['certified'] = gap, allowed, gap <= allowed
    if not info['converged']:
        gp, _ = _projected_gradient(grad, lam, lam_max, 0.0, a)
        info['residual'] = float(np.max(np.abs(gp)/scale))
        if warn and info['residual'] > np.sqrt(tol):
            warnings.warn(f"mmadual: dual iteration stopped ({info['reason']}) after {info['newton_steps']} steps with "
                          f"projected gradient {info['residual']:.1e} (tol = {tol:.1e}); the subproblem solution may "
                          "be less accurate than requested", RuntimeWarning, stacklevel=3)
    return x, y, 0.0, lam, info


def _snap(lam, lam_max, eps, a, a0):
    "multipliers within eps of a bound are put on it, the upper bounds only when a^T lam <= a0 is kept"
    lam = np.where(lam <= eps, 0.0, lam)
    up = lam >= lam_max - eps
    if np.any(up):
        trial = np.where(up, lam_max, lam)
        if a @ trial <= a0:
            lam = trial
    return lam


def _ratio(lam, s, lam_max, fixed, hold_plane, a, a0):
    "largest alpha with lam + alpha s in the domain (bounds of the free multipliers; the half-space unless held)"
    free = ~fixed
    alpha = np.inf
    dn = free & (s < 0)
    if np.any(dn):
        alpha = min(alpha, float(np.min(lam[dn]/(-s[dn]))))
    up = free & (s > 0) & np.isfinite(lam_max)
    if np.any(up):
        alpha = min(alpha, float(np.min((lam_max[up] - lam[up])/s[up])))
    if not hold_plane:
        a_s = a @ s
        if a_s > 0:
            alpha = min(alpha, max(a0 - a @ lam, 0.0)/a_s)
    return alpha


def _to_first_bound(lam, s, lam_max, fixed, hold_plane, a, a0):
    """s scaled to reach the first bound (or the half-space) along it: the length of a direction without curvature
       (an affine piece of W) is set by the geometry of the domain, not by the unit of W"""
    alpha = _ratio(lam, s, lam_max, fixed, hold_plane, a, a0)
    return s*alpha if (np.isfinite(alpha) and alpha > 0) else s


def _tangent_direction(H, grad, lam, lam_max, free, a, a0, on_plane, mu):
    """LM direction on the free multipliers, in the tangent space of the half-space when `on_plane` (and a_free != 0).
       The reduced directions without curvature take the gradient scaled to reach the first bound exactly, damped by
       1/max(1, mu) (so that a small mu does not leave the step short of the bound); the curved ones the
       Jacobi-scaled LM step"""
    m = grad.size
    s = np.zeros(m)
    if not np.any(free):
        return s
    Hf, gf = H[np.ix_(free, free)], grad[free]
    af = a[free]
    N = None
    if on_plane and np.any(af > 0):
        N = np.linalg.svd(af.reshape(1, -1), full_matrices=True)[2][1:].T  # orthonormal basis of {u : af^T u = 0}
        Hf, gf = N.T @ Hf @ N, N.T @ gf
    curved, S, w, V = _scaled_spectrum(Hf)
    sf = _lm_step(gf, curved, S, w, V, mu)
    if not np.all(curved):                                                 # flat reduced directions: gradient to the first bound
        flat = np.where(curved, 0.0, gf)
        f = np.zeros(m)
        f[free] = N @ flat if N is not None else flat
        alpha = _ratio(lam, f, lam_max, ~free, on_plane, a, a0)
        sf[~curved] = gf[~curved]*(alpha if (np.isfinite(alpha) and alpha > 0) else 1.0)/max(1.0, mu)
    s[free] = N @ sf if N is not None else sf
    return s


def _feasible_direction(direction, lam, lam_max, gp, active, lower, upper, plane_active, on_plane, a, a0):
    """direction of the polytope path: `direction(free, on_plane)` (LM or gradient) on the free multipliers; a free
       multiplier at a bound whose component points outward is fixed and the direction recomputed, as is the
       half-space when lam is on it, it is inactive and the direction leaves the domain. Returns the direction
       (zero on the fixed multipliers, a^T s = 0 when the half-space is held) and the flags"""
    fixed, hold_plane = active.copy(), plane_active
    while True:
        free = ~fixed
        s = direction(free, hold_plane)
        if not np.all(np.isfinite(s)) or s @ gp <= 0:
            s = _to_first_bound(lam, np.where(free, gp, 0.0), lam_max, fixed, hold_plane, a, a0)
        blocked = free & ((lower & (s < 0)) | (upper & (s > 0)))
        block_plane = (not hold_plane) and on_plane and (a @ s > 0)
        if not np.any(blocked) and not block_plane:
            return s, fixed, hold_plane
        fixed |= blocked
        hold_plane = hold_plane or block_plane
        if np.all(fixed):
            return np.zeros(len(s)), fixed, hold_plane


def _solve_polytope(sub, tol, max_newton, warn):
    "active-set projected Newton on the box intersected with the half-space a^T lam <= a0 (a != 0): see the module docstring"
    m = sub['b'].size
    c, d, a, a0 = sub['c'], sub['d'], sub['a'], sub['a0']
    lam_max = np.where(d > 0, np.inf, c)
    eps = BOUND_TOL*np.maximum(1.0, c)
    eps_plane = BOUND_TOL*max(1.0, a0)
    lam = np.zeros(m)
    W, grad, x, y, p, q = dual_function(sub, lam)
    scale = np.maximum(1.0, np.maximum(np.abs(sub['b']), grad + sub['b']))
    info = dict(converged=False, reason='max_newton', newton_steps=0, evaluations=1, residual=np.inf, scale=scale,
                z=0.0, halfspace_active=False)
    mu = LM_MU_MIN
    stagnant = 0
    z = 0.0
    last_trial = None

    def trial(s, alpha):
        return _snap(project_dual(lam + alpha*s, a, a0, lam_max), lam_max, eps, a, a0)

    for it in range(max_newton + 1):
        lower, upper = lam <= eps, lam >= lam_max - eps
        on_plane = a0 - a @ lam <= eps_plane
        z = _halfspace_multiplier(grad, lower, upper, a) if on_plane else 0.0
        plane_active = z > 0.0
        u = grad - z*a
        active = (lower & (u < 0)) | (upper & (u > 0))
        gp = np.where(active, 0.0, u)
        r = project_dual(lam + grad, a, a0, lam_max) - lam
        info['residual'] = float(max(np.max(np.abs(gp)/scale), np.max(np.abs(r)/scale)))
        info['z'], info['halfspace_active'] = z, plane_active
        if info['residual'] <= tol:
            info['reason'] = 'tolerance'
            break
        if it == max_newton:
            break
        H = dual_hessian(sub, lam, x, p, q)
        accepted = False
        while True:                                                         # LM: raise mu until Armijo holds
            s, fixed, hold_plane = _feasible_direction(
                lambda free, hold: _tangent_direction(H, grad, lam, lam_max, free, a, a0, hold, mu),
                lam, lam_max, gp, active, lower, upper, plane_active, on_plane, a, a0)
            alpha = min(1.0, _ratio(lam, s, lam_max, fixed, hold_plane, a, a0)) if np.any(s != 0) else 0.0
            if alpha > 0:
                lam_new = trial(s, alpha)
                if last_trial is None or not np.array_equal(lam_new, last_trial):
                    last_trial = lam_new
                    W_new, grad_new, x_new, y_new, p_new, q_new = dual_function(sub, lam_new)
                    info['evaluations'] += 1
                    pred = grad @ (lam_new - lam)
                    if pred > 0 and W_new - W >= ARMIJO*pred - ROUNDING_W*abs(W):
                        accepted = True
                        break
            if mu >= LM_MU_MAX or np.all(fixed):
                break
            mu *= 10.0
        if not accepted:                                                    # projected-gradient arc with backtracking
            gpt = gp.copy()
            if plane_active and np.any(a[~active] > 0):                     # keep the arc on the half-space
                af, uf = a[~active], gp[~active]
                gpt[~active] = uf - af*(af @ uf)/(af @ af)
            s, fixed, hold_plane = _feasible_direction(
                lambda free, hold: np.where(free, gpt, 0.0),
                lam, lam_max, gp, active, lower, upper, plane_active, on_plane, a, a0)
            alpha = _ratio(lam, s, lam_max, fixed, hold_plane, a, a0) if np.any(s != 0) else 0.0
            if not np.isfinite(alpha):
                alpha = 1.0
            for _ in range(ARC_HALVINGS):
                if alpha <= 0:
                    break
                lam_new = trial(s, alpha)
                W_new, grad_new, x_new, y_new, p_new, q_new = dual_function(sub, lam_new)
                info['evaluations'] += 1
                pred = grad @ (lam_new - lam)
                if pred > 0 and W_new - W >= ARMIJO*pred - ROUNDING_W*abs(W):
                    accepted = True
                    break
                alpha *= 0.5
        if not accepted:
            info['reason'] = 'no_progress'
            break
        stagnant = stagnant + 1 if (W_new - W <= STAGNATION_W*abs(W) and mu <= LM_MU_MIN) else 0
        mu = max(mu/100.0, LM_MU_MIN)
        lam, W, grad, x, y, p, q = lam_new, W_new, grad_new, x_new, y_new, p_new, q_new
        last_trial = None
        info['newton_steps'] += 1
        if stagnant >= 2:
            info['reason'] = 'rounding'
            break
    # final multiplier of the half-space (tolerance-based classification) and certificate of the returned point
    lower, upper = lam <= eps, lam >= lam_max - eps
    on_plane = a0 - a @ lam <= eps_plane
    z = _halfspace_multiplier(grad, lower, upper, a) if on_plane else 0.0
    u = grad - z*a
    gp = np.where((lower & (u < 0)) | (upper & (u > 0)), 0.0, u)
    r = project_dual(lam + grad, a, a0, lam_max) - lam
    info['residual'] = float(max(np.max(np.abs(gp)/scale), np.max(np.abs(r)/scale)))
    info['z'], info['halfspace_active'] = z, z > 0.0
    gap, allowed, viol = _certificate(sub, lam, W, grad, x, y, z, scale, tol, eps, eps_plane)
    info['gap_returned'], info['gap_allowed'] = gap, allowed
    info['certified'] = gap <= allowed and viol <= tol*float(np.max(scale))
    if info['reason'] == 'tolerance':
        if info['certified']:
            info['converged'] = True
        else:
            info['reason'] = 'certification'
    if not info['converged']:
        if warn and (info['residual'] > np.sqrt(tol) or info['reason'] == 'certification'):
            warnings.warn(f"mmadual: dual iteration stopped ({info['reason']}) after {info['newton_steps']} steps with "
                          f"projected gradient {info['residual']:.1e} (tol = {tol:.1e}) and returned-point gap "
                          f"{gap:.1e} (allowed {allowed:.1e}); the subproblem solution may be less accurate than "
                          "requested", RuntimeWarning, stacklevel=3)
    return x, y, z, lam, info


def recover_multipliers(sub, x, y, z, lam):
    """multipliers of the bounds (xsi, eta), of y >= 0 (mu), of z >= 0 (zet) and the slacks s of the
       constraints, from the KKT conditions of the subproblem, for compatibility with the interior-point
       solver; with d = 0 the returned y is max(0, g(x) - b - a z), which carries the violation when lam = c"""
    p, q = sub['p0'] + lam @ sub['P'], sub['q0'] + lam @ sub['Q']
    dpsi = p/(sub['upp'] - x)**2 - q/(x - sub['low'])**2               # d(Lagrangian)/dx: = xsi - eta at the solution
    xsi, eta = np.maximum(dpsi, 0.0), np.maximum(-dpsi, 0.0)
    g = sub['P'] @ (1.0/(sub['upp'] - x)) + sub['Q'] @ (1.0/(x - sub['low'])) - sub['a']*z - sub['b']
    y = np.where(sub['d'] > 0, y, np.maximum(g, 0.0))
    mu = np.maximum(sub['c'] + sub['d']*y - lam, 0.0)
    zet = max(sub['a0'] - sub['a'] @ lam, 0.0)
    s = np.maximum(y - g, 0.0)
    return xsi, eta, mu, zet, s, y
