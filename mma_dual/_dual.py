"""
mma_dual - MMA and GCMMA subproblems solved by the dual method
==============================================================

Drop-in replacements for ``mmasub`` and ``gcmmasub`` of GCMMA-MMA-Python
(Arjen Deetman, <https://github.com/arjendeetman/GCMMA-MMA-Python>), which is
the Python translation of the MATLAB code of Krister Svanberg
(<http://www.smoptit.se/>). Both are distributed under the GNU General Public
License v3, and so is this module.

What is taken from GCMMA-MMA-Python (verbatim, so that both codes build the
SAME subproblem): the update of the asymptotes ``low``/``upp``, the bounds
``alfa``/``beta`` and the coefficients ``p0``, ``q0``, ``P``, ``Q`` and ``b``
of the separable approximations, in ``mmasub`` and ``gcmmasub``.

What is new: the solution of the subproblem. GCMMA-MMA-Python solves it with a
primal-dual interior-point Newton method in n + 2m + 3 unknowns (``subsolv``).
Here it is solved by the dual method described by Svanberg (1987, Section 5):
for fixed Lagrange multipliers ``lam`` of the m constraints the subproblem is
minimized in closed form in each x_j, and the resulting concave dual function
W(lam), which depends only on the m multipliers, is maximized by a projected
Newton method with analytical Hessian on the box 0 <= lam <= c. Each dual
evaluation costs O(n m), so for m << n the subproblem becomes negligible
compared with the analysis of the original problem.

The subproblem (notation of GCMMA-MMA-Python):
    min  sum_j [p0j/(Uj - xj) + q0j/(xj - Lj)] + a0 z + sum_i (ci yi + di yi^2/2)
    s.t. sum_j [Pij/(Uj - xj) + Qij/(xj - Lj)] - ai z - yi <= bi
         alfa_j <= xj <= beta_j ,  yi >= 0 ,  z >= 0
Closed-form minimizers for fixed lam:
    pj = p0j + sum_i lam_i Pij ,  qj = q0j + sum_i lam_i Qij
    xj = (sqrt(pj) Lj + sqrt(qj) Uj)/(sqrt(pj) + sqrt(qj)), clipped to [alfa_j, beta_j]
    yi = max(0, (lam_i - ci)/di)  if di > 0 ;  lam_i <= ci (and yi = 0) if di = 0
    z  = 0 (with a = 0 and a0 > 0, the standard choice: see the restriction below)
The dual gradient is the residual of the approximate constraints and the dual
Hessian is assembled analytically from the x_j strictly inside [alfa_j, beta_j].

Restriction: only a = 0 is supported (a0 > 0 arbitrary). With a != 0 (minimax
problems) the dual domain is the box intersected with the half-space
sum_i ai lam_i <= a0 and z is recovered on its boundary; this case is not
implemented and raises NotImplementedError. Use mmapy for such problems.

Interface: identical to GCMMA-MMA-Python (same positional arguments, same
returned tuple). ``gcmmasub`` accepts an optional ``move`` limit, absent in the
original. The auxiliary GCMMA functions ``asymp``, ``concheck`` and
``raaupdate`` do not solve subproblems and are used unchanged from mmapy. A
RuntimeWarning is issued when the dual Newton iteration stops before reaching
``tol_dual`` (``max_newton`` reached or no further progress).

References
    K. Svanberg, The method of moving asymptotes - a new method for structural
    optimization, Int. J. Numer. Meth. Engng 24 (1987) 359-373.
    K. Svanberg, A class of globally convergent optimization methods based on
    conservative convex separable approximations, SIAM J. Optim. 12 (2002)
    555-573.

Copyright (C) 2026 Olavo M. Silva (LVA-MOPT, UFSC). Derived from
GCMMA-MMA-Python, Copyright (C) Arjen Deetman, and from the MATLAB code of
Krister Svanberg. This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
LICENSE file or <https://www.gnu.org/licenses/>.
"""
import warnings
import numpy as np


def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp,
           a0, a, c, d, move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10,
           raa0=0.00001, albefa=0.1, tol_dual=1e-10, max_newton=100, **kwargs):
    "mesma interface do mmapy.mmasub; retorna xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp"
    eeen, eeem, zeron = np.ones((n, 1)), np.ones((m, 1)), np.zeros((n, 1))

    # --- assíntotas (igual ao mmapy)
    if iter <= 2:
        low = xval - asyinit*(xmax - xmin)
        upp = xval + asyinit*(xmax - xmin)
    else:
        zzz = (xval - xold1)*(xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor*(xold1 - low)
        upp = xval + factor*(upp - xold1)
        low = np.minimum(np.maximum(low, xval - asymax*(xmax - xmin)), xval - asymin*(xmax - xmin))
        upp = np.maximum(np.minimum(upp, xval + asymax*(xmax - xmin)), xval + asymin*(xmax - xmin))

    # --- limites alfa e beta (igual ao mmapy)
    alfa = np.maximum(np.maximum(low + albefa*(xval - low), xval - move*(xmax - xmin)), xmin)
    beta = np.minimum(np.minimum(upp - albefa*(upp - xval), xval + move*(xmax - xmin)), xmax)

    # --- coeficientes p0, q0, P, Q, b (igual ao mmapy)
    xmami_inv = eeen/np.maximum(xmax - xmin, 0.00001*eeen)
    ux1, xl1 = upp - xval, xval - low
    ux2, xl2 = ux1*ux1, xl1*xl1
    p0, q0 = np.maximum(df0dx, 0), np.maximum(-df0dx, 0)
    pq0 = 0.001*(p0 + q0) + raa0*xmami_inv
    p0, q0 = (p0 + pq0)*ux2, (q0 + pq0)*xl2
    P, Q = np.maximum(dfdx, 0), np.maximum(-dfdx, 0)
    PQ = 0.001*(P + Q) + raa0*np.dot(eeem, xmami_inv.T)
    P, Q = (P + PQ)*ux2.T, (Q + PQ)*xl2.T
    b = np.dot(P, eeen/ux1) + np.dot(Q, eeen/xl1) - fval

    # --- solução do subproblema pelo dual
    x, y, z, lam = subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol_dual, max_newton)

    # --- multiplicadores dos limites e folgas (para compatibilidade de retorno)
    pl, ql = p0 + P.T @ lam, q0 + Q.T @ lam
    dpsidx = pl/(upp - x)**2 - ql/(x - low)**2
    xsi, eta = np.maximum(-dpsidx, 0), np.maximum(dpsidx, 0)
    g = P @ (eeen/(upp - x)) + Q @ (eeen/(x - low)) - a*z - b                 # restrições aproximadas em x
    y = np.where(d > 0, y, np.maximum(g, 0))                                   # d = 0: y absorve a violação (como no mmapy)
    mu = np.maximum(c + d*y - lam, 0)
    zet = np.array([[max(a0 - float((a.T @ lam).item()), 0.0)]])
    s = np.maximum(y - g, 0)
    return x, y, z, lam, xsi, eta, mu, zet, s, low, upp


def gcmmasub(m, n, iter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx,
             a0, a, c, d, albefa=0.1, move=None, tol_dual=1e-10, max_newton=100, **kwargs):
    """mesma interface do mmapy.gcmmasub; retorna xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp.
       move (opcional): limite de movimento |x - xval| <= move (xmax - xmin), como no mmasub
       (o gcmmasub do mmapy não tem esse limite; None reproduz o mmapy)"""
    eeen, eeem = np.ones((n, 1)), np.ones((m, 1))
    alfa = np.maximum(low + albefa*(xval - low), xmin)
    beta = np.minimum(upp - albefa*(upp - xval), xmax)
    if move is not None:
        alfa = np.maximum(alfa, xval - move*(xmax - xmin))
        beta = np.minimum(beta, xval + move*(xmax - xmin))
    xmami_inv = eeen/np.maximum(xmax - xmin, 0.00001*eeen)
    ux1, xl1 = upp - xval, xval - low
    ux2, xl2 = ux1*ux1, xl1*xl1
    p0, q0 = np.maximum(df0dx, 0), np.maximum(-df0dx, 0)
    pq0 = p0 + q0
    p0, q0 = (p0 + 0.001*pq0 + raa0*xmami_inv)*ux2, (q0 + 0.001*pq0 + raa0*xmami_inv)*xl2
    r0 = f0val - np.dot(p0.T, eeen/ux1) - np.dot(q0.T, eeen/xl1)
    P, Q = np.maximum(dfdx, 0), np.maximum(-dfdx, 0)
    PQ = P + Q
    P, Q = (P + 0.001*PQ + np.dot(raa, xmami_inv.T))*ux2.T, (Q + 0.001*PQ + np.dot(raa, xmami_inv.T))*xl2.T
    r = fval - np.dot(P, eeen/ux1) - np.dot(Q, eeen/xl1)
    b = -r
    x, y, z, lam = subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol_dual, max_newton)
    pl, ql = p0 + P.T @ lam, q0 + Q.T @ lam
    dpsidx = pl/(upp - x)**2 - ql/(x - low)**2
    xsi, eta = np.maximum(-dpsidx, 0), np.maximum(dpsidx, 0)
    g = P @ (eeen/(upp - x)) + Q @ (eeen/(x - low)) - a*z - b
    y = np.where(d > 0, y, np.maximum(g, 0))
    mu = np.maximum(c + d*y - lam, 0)
    zet = np.array([[max(a0 - float((a.T @ lam).item()), 0.0)]])
    s = np.maximum(y - g, 0)
    f0app = r0 + np.dot(p0.T, eeen/(upp - x)) + np.dot(q0.T, eeen/(x - low))
    fapp = r + np.dot(P, eeen/(upp - x)) + np.dot(Q, eeen/(x - low))
    return x, y, z, lam, xsi, eta, mu, zet, s, f0app, fapp


def subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol=1e-10, max_newton=100):
    "maximiza a função dual W(lam) na caixa 0 <= lam <= c (Newton projetado com busca linear)"
    low, upp, alfa, beta = low.ravel(), upp.ravel(), alfa.ravel(), beta.ravel()
    p0, q0, a, b, c, d = p0.ravel(), q0.ravel(), a.ravel(), b.ravel(), c.ravel(), d.ravel()
    if np.any(a != 0):
        raise NotImplementedError("mma_dual: subproblems with a != 0 (minimax form) are not supported; the dual "
                                  "domain is then the box intersected with a^T lam <= a0. Use mmapy for these.")
    if a0 < 0:
        raise ValueError("mma_dual: a0 must be >= 0 (the subproblem is unbounded in z otherwise)")
    lam_max = np.where(d > 0, np.inf, c)                                    # d_i = 0: y_i = 0 exige lam_i <= c_i
    z = 0.0                                                                 # a = 0, a0 >= 0: z = 0 minimiza a0 z

    def primal(lam):
        "x(lam), y(lam) e os termos auxiliares"
        p, q = p0 + lam @ P, q0 + lam @ Q
        sp, sq = np.sqrt(p), np.sqrt(q)
        x = np.clip((sp*low + sq*upp)/(sp + sq), alfa, beta)
        y = np.where(d > 0, np.maximum(0.0, (lam - c)/np.where(d > 0, d, 1.0)), 0.0)
        return x, y, z, p, q

    def dual(lam):
        "W(lam) e o gradiente g_i(x(lam)) - y_i - b_i"
        x, y, z, p, q = primal(lam)
        ux, xl = upp - x, x - low
        W = np.sum(p/ux + q/xl) - lam @ b + np.sum(c*y + 0.5*d*y*y) - lam @ y
        grad = P @ (1/ux) + Q @ (1/xl) - y - b
        return W, grad, x, y, z, p, q

    def hessiana(lam, x, p, q):
        "d^2 W/d lam^2 (semidefinida negativa): só os x_j no interior contribuem"
        ux, xl = upp - x, x - low
        livre = (x > alfa + 1e-12*(beta - alfa)) & (x < beta - 1e-12*(beta - alfa))
        G = (P/ux**2 - Q/xl**2)[:, livre]                                    # d g_i/d x_j
        D = 2*p[livre]/ux[livre]**3 + 2*q[livre]/xl[livre]**3               # d^2 (lagrangiano)/d x_j^2
        H = -(G/D) @ G.T
        H -= np.diag(np.where((d > 0) & (lam > c), 1/np.where(d > 0, d, 1.0), 0.0))
        return H

    lam = np.zeros(m)
    W, grad, x, y, z, p, q = dual(lam)
    escala = max(1.0, np.max(np.abs(b)))
    convergiu = False
    for it in range(max_newton + 1):
        # conjunto ativo: lam no limite e gradiente apontando para fora
        ativo = ((lam <= 0) & (grad < 0)) | ((lam >= lam_max) & (grad > 0))
        gp = np.where(ativo, 0.0, grad)
        if np.max(np.abs(gp)) <= tol*escala:
            convergiu = True
            break
        if it == max_newton:
            break
        H = hessiana(lam, x, p, q)
        livre = ~ativo
        dlam = np.zeros(m)
        Hl = H[np.ix_(livre, livre)]
        Hl = Hl - 1e-12*max(1.0, np.max(np.abs(np.diag(Hl))))*np.eye(len(Hl))
        try:
            dlam[livre] = np.linalg.solve(Hl, -grad[livre])
        except np.linalg.LinAlgError:
            dlam[livre] = grad[livre]
        if not np.all(np.isfinite(dlam)) or dlam @ gp <= 0:                 # não é direção de subida: gradiente projetado
            dlam = gp
        t = 1.0
        for _ in range(60):                                                 # busca linear (Armijo) com projeção na caixa
            lam_n = np.clip(lam + t*dlam, 0.0, lam_max)
            W_n, grad_n, x_n, y_n, z_n, p_n, q_n = dual(lam_n)
            if W_n - W >= 1e-4*(grad @ (lam_n - lam)) - 1e-14*abs(W) or np.all(lam_n == lam):
                break                                                       # 1e-14|W|: tolerância de arredondamento de W
            t *= 0.5
        if np.all(lam_n == lam):
            break
        lam, W, grad, x, y, z, p, q = lam_n, W_n, grad_n, x_n, y_n, z_n, p_n, q_n
    if not convergiu:
        ativo = ((lam <= 0) & (grad < 0)) | ((lam >= lam_max) & (grad > 0))
        residuo = np.max(np.abs(np.where(ativo, 0.0, grad)))/escala
        warnings.warn(f"mma_dual: dual Newton stopped after {it} steps with projected gradient "
                      f"{residuo:.1e} (tol_dual = {tol:.1e}); the subproblem solution may be "
                      "less accurate than requested", RuntimeWarning, stacklevel=3)
    return x.reshape(-1, 1), y.reshape(-1, 1), np.array([[z]]), lam.reshape(-1, 1)
