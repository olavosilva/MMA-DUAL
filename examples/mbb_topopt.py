"""
Minimum-compliance topology optimization of the MBB beam (half model, plane stress, SIMP, linear
density filter, volume constraint), run twice with identical settings and initial design: once with
mmapy.mmasub (primal-dual interior-point subproblem solver) and once with mmadual.mmasub (dual
subproblem solver). The time of each outer iteration is split into finite-element analysis (assembly
and linear solve), sensitivities and filtering, and MMA subproblem, so that the gain in the subproblem
can be put against the total cost of the optimization.

Linear solver: PyPardiso (Intel MKL Pardiso, pip install pypardiso) if available, otherwise
scipy.sparse.linalg.spsolve (SuperLU, much slower for large meshes).

Run:  python examples/mbb_topopt.py [nelx] [nely] [maxiter] [rmin] [tol]
      python examples/mbb_topopt.py 600 200 500 3.0 0.01   (120 000 elements, 241 602 dofs)
Defaults: nelx = 300, nely = 100, maxiter = 500, rmin = 3.0 elements, tol = 0.01.

Stopping criterion: the iteration stops when the largest change of a design variable between consecutive
iterations, max_j |x_j^(k) - x_j^(k-1)|, falls below tol (0.01 in the 88-line code), or when maxiter is
reached, in which case the run is flagged as not converged. With tol > 0 the two solvers are compared
on the cost of reaching the same tolerance, each with its own number of iterations; with tol = 0 they
are compared on a fixed number of iterations (maxiter).

The two runs start from the same design and use the same approximations, so they differ only through
the accuracy of the subproblem solutions. The compliance histories agree to three or four digits; the
difference comes from the interior-point routine, whose solution stays at a distance epsilon/xi_j from
a weakly active move limit (barrier parameter epsilon over the bound multiplier xi_j), while the dual
solver returns the exact bound; two runs of the dual solver with tol_dual = 1e-10 and 1e-6 agree to
about 1e-7. Over many iterations these small differences are amplified by the asymptote update and by
the non-convexity of the problem, so the two runs can end in different local optima with slightly
different members and compliances within about 1% of each other (300 x 100 elements, 100 iterations:
202.04 against 199.81 for the final designs); the comparison is one of cost per iteration, not of identical designs. The
summary at the end gives the time per component for both solvers, the final compliances and the
largest difference between the two designs. Since the two runs do not follow the same path, they may
need different numbers of iterations to reach the tolerance; the summary gives the number of iterations,
the total time and the time per iteration for each solver. The final compliance in the summary is that of the design after the last update (the one shown in the
figure and saved), obtained with one additional analysis that is not included in the timings; the
history 'c' holds the compliance of the design at the start of each iteration.
Results (histories and final designs) are saved to mbb_<nelx>x<nely>.npz; a PNG of the two designs is
saved if matplotlib is available.

Formulation (Bendsoe and Sigmund; Andreassen et al. 2011, 88-line code), with the objective and the
constraint scaled as recommended by Svanberg: f0 = c/c_ref with c_ref the compliance of the uniform
initial design, g = V/(volfrac V0) - 1.
"""
import os
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import mmadual
import mmapy

try:
    from pypardiso import spsolve as pardiso_spsolve
    SOLVER = 'PyPardiso (MKL Pardiso)'
    def solve(K, F):
        return pardiso_spsolve(K, F)
except ImportError:
    from scipy.sparse.linalg import spsolve as scipy_spsolve
    SOLVER = 'scipy.sparse.linalg.spsolve (SuperLU)'
    def solve(K, F):
        return scipy_spsolve(K, F)

try:
    from scipy.linalg import LinAlgWarning                                  # emitted by mmapy's subsolv when its Newton
except ImportError:                                                         # system is ill-conditioned; counted, not silenced
    LinAlgWarning = Warning

# ---------------------------------------------------------------- parameters
VOLFRAC, PENAL, E0, EMIN, NU = 0.5, 3.0, 1.0, 1e-9, 0.3
MOVE = 0.2


def argumentos(argv):
    "nelx, nely, maxiter, rmin, tol from the command line, with the defaults of the module docstring"
    return (int(argv[1]) if len(argv) > 1 else 300, int(argv[2]) if len(argv) > 2 else 100,
            int(argv[3]) if len(argv) > 3 else 500, float(argv[4]) if len(argv) > 4 else 3.0,
            float(argv[5]) if len(argv) > 5 else 0.01)


def versao(pacote):
    try:
        from importlib.metadata import version
        return version(pacote)
    except Exception:
        return '?'


def configuracao():
    "machine, Python, numpy, BLAS threads and linear solver, for the record"
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(l.split(':', 1)[1].strip() for l in f if l.startswith('model name'))
    except (OSError, StopIteration):
        pass
    blas = ''
    try:
        import threadpoolctl
        blas = '; ' + ', '.join(f"{d['internal_api']} {d['version']} ({d['num_threads']} threads)"
                                for d in threadpoolctl.threadpool_info() if d['user_api'] == 'blas')
    except ImportError:
        pass
    return (f"{cpu}, {os.cpu_count()} logical cores; {platform.system()} {platform.release()}; "
            f"Python {platform.python_version()}, numpy {np.__version__}{blas}; "
            f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS', 'unset')}, "
            f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'unset')}, "
            f"OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS', 'unset')}; "
            f"linear solver: {SOLVER}; "
            f"mmapy {versao('mmapy')}, mmadual {mmadual.__version__}")


# ---------------------------------------------------------------- finite element model
def elemento_q4(E, nu):
    "stiffness matrix of the bilinear square element, plane stress, unit side and thickness (88-line code)"
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = E/(1 - nu**2)*np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


def malha(nelx, nely):
    "element dof table (8 dofs per element), nodes numbered column by column, top to bottom"
    nel = nelx*nely
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    n1 = (nely + 1)*elx + ely                                               # upper-left node of each element
    n1 = n1.ravel()
    edof = np.column_stack((2*n1 + 2, 2*n1 + 3, 2*(n1 + nely + 1) + 2, 2*(n1 + nely + 1) + 3,
                            2*(n1 + nely + 1), 2*(n1 + nely + 1) + 1, 2*n1, 2*n1 + 1))
    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    return nel, edof, iK, jK


def filtro(nelx, nely, rmin):
    "linear density filter H (sparse, row-normalized by Hs), built from neighbour offsets, no Python loop over elements"
    r = int(np.ceil(rmin)) - 1
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    ex, ey = ex.ravel(), ey.ravel()
    rows, cols, vals = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            w = rmin - np.sqrt(di*di + dj*dj)
            if w <= 0:
                continue
            nx, ny = ex + di, ey + dj
            ok = (nx >= 0) & (nx < nelx) & (ny >= 0) & (ny < nely)
            rows.append(np.flatnonzero(ok))
            cols.append(nx[ok]*nely + ny[ok])
            vals.append(np.full(ok.sum(), w))
    nel = nelx*nely
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nel, nel))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def monta_e_resolve(xPhys, KE, iK, jK, ndof, livres, F, cronometro):
    "global stiffness (SIMP), linear solve on the free dofs; returns U and the element compliances"
    t = time.perf_counter()
    sK = (KE.ravel()[None, :]*(EMIN + xPhys**PENAL*(E0 - EMIN))[:, None]).ravel()
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = K[livres, :][:, livres]
    U = np.zeros(ndof)
    U[livres] = solve(K, F[livres])
    cronometro['fe'] += time.perf_counter() - t
    return U


def otimiza(nome, mmasub, nelx, nely, maxiter, rmin, x0, tol):
    "MMA loop; also re-evaluates final design (excluded from timed optimization)"
    nel, edof, iK, jK = malha(nelx, nely)
    ndof = 2*(nelx + 1)*(nely + 1)
    KE = elemento_q4(E0, NU)
    H, Hs = filtro(nelx, nely, rmin)
    # MBB half beam: unit downward load at the top-left node, symmetry (ux = 0) on the left edge,
    # roller (uy = 0) at the bottom-right node
    F = np.zeros(ndof)
    F[1] = -1.0
    fixos = np.union1d(np.arange(0, 2*(nely + 1), 2), [2*(nelx + 1)*(nely + 1) - 1])
    livres = np.setdiff1d(np.arange(ndof), fixos)
    # MMA data
    m, n = 1, nel
    x = x0.copy().reshape(-1, 1)
    xold1, xold2 = x.copy(), x.copy()
    xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
    low, upp = xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1))
    cron = {'fe': 0.0, 'sens': 0.0, 'mma': 0.0}
    n_avisos = 0
    hist = {'c': [], 'vol': [], 'change': [], 't_fe': [], 't_sens': [], 't_mma': []}
    xPhys = (H @ x.ravel())/Hs
    c_ref = None
    # warm-up: library loading and thread start-up of the linear solver are not timed; the matrix is perturbed
    # so that PyPardiso, which reuses the last factorization when the matrix is unchanged, factorizes again in
    # iteration 1
    monta_e_resolve(0.999*xPhys, KE, iK, jK, ndof, livres, F, {'fe': 0.0})
    print(f"\n== {nome}")
    print(f"{'it':>4} | {'compliance':>12} | {'vol':>6} | {'change':>7} | {'t FE [s]':>8} | {'t sens [s]':>10} | {'t MMA [s]':>9}")
    for it in range(1, maxiter + 1):
        t_fe0, t_s0, t_m0 = cron['fe'], cron['sens'], cron['mma']
        U = monta_e_resolve(xPhys, KE, iK, jK, ndof, livres, F, cron)
        t = time.perf_counter()
        ce = np.einsum('ij,jk,ik->i', U[edof], KE, U[edof])                 # element compliances u_e^T KE u_e
        comp = np.sum((EMIN + xPhys**PENAL*(E0 - EMIN))*ce)
        dc = -PENAL*xPhys**(PENAL - 1)*(E0 - EMIN)*ce
        dv = np.ones(nel)
        dc = H.T @ (dc/Hs)                                                  # chain rule through the filter
        dv = H.T @ (dv/Hs)
        if c_ref is None:
            c_ref = comp
        f0val = comp/c_ref
        df0dx = (dc/c_ref).reshape(-1, 1)
        fval = np.array([[np.sum(xPhys)/(VOLFRAC*nel) - 1.0]])
        dfdx = (dv/(VOLFRAC*nel)).reshape(1, -1)
        cron['sens'] += time.perf_counter() - t
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            xnew, _, _, _, _, _, _, _, _, low, upp = mmasub(m, n, it, x, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx,
                                                            low, upp, a0, a, c, d, move=MOVE)
        cron['mma'] += time.perf_counter() - t
        n_avisos += sum(issubclass(w.category, LinAlgWarning) for w in avisos)
        for w in avisos:                                                    # anything else is shown
            if not issubclass(w.category, LinAlgWarning):
                print(f"  iteration {it}: {w.category.__name__}: {w.message}")
        change = np.abs(xnew - x).max()
        # Compliance and volume must refer to the SAME (pre-update) design.
        hist['c'].append(comp); hist['vol'].append(float(np.mean(xPhys)))
        hist['change'].append(change)
        xold2, xold1, x = xold1, x, xnew
        # Include this filter application in the sensitivity/filter timing.
        t = time.perf_counter()
        xPhys = (H @ x.ravel())/Hs
        cron['sens'] += time.perf_counter() - t
        hist['t_fe'].append(cron['fe'] - t_fe0); hist['t_sens'].append(cron['sens'] - t_s0); hist['t_mma'].append(cron['mma'] - t_m0)
        convergiu = change <= tol
        if it <= 5 or it % 10 == 0 or it == maxiter or convergiu:
            print(f"{it:4d} | {comp:12.4f} | {hist['vol'][-1]:6.3f} | {change:7.4f} | {hist['t_fe'][-1]:8.3f} | {hist['t_sens'][-1]:10.3f} | {hist['t_mma'][-1]:9.4f}")
        if convergiu:
            break
    print(f"  {'converged' if convergiu else 'NOT converged'}: change {change:.4f} {'<=' if convergiu else '>'} tol {tol} after {it} iterations")

    # The last MMA update produces a new design. Re-evaluate that design to
    # report compliance corresponding to the PNG and x_mmadual/x_mmapy arrays.
    # Exclude this additional verification from the 100-iteration timing.
    t_validation = time.perf_counter()
    U_final = monta_e_resolve(xPhys, KE, iK, jK, ndof, livres, F, {'fe': 0.0})
    ce_final = np.einsum('ij,jk,ik->i', U_final[edof], KE, U_final[edof])
    comp_final = float(np.sum((EMIN + xPhys**PENAL*(E0 - EMIN))*ce_final))
    validation_time = time.perf_counter() - t_validation
    if n_avisos:
        print(f"  {n_avisos} LinAlgWarning(s) raised by the subproblem solver (ill-conditioned Newton system)")
    return hist, xPhys.reshape(nelx, nely).T, cron, comp_final, validation_time, it, convergiu


if __name__ == '__main__':
    NELX, NELY, MAXITER, RMIN, TOL = argumentos(sys.argv)
    print(configuracao())
    nel = NELX*NELY
    print(f"MBB beam, {NELX} x {NELY} = {nel} elements, {2*(NELX + 1)*(NELY + 1)} dofs, volfrac {VOLFRAC}, penal {PENAL}, "
          f"rmin {RMIN}, move {MOVE}; stop at max change <= {TOL} or {MAXITER} iterations; m = 1 constraint")
    x0 = VOLFRAC*np.ones(nel)
    resultados = {}
    for nome, sub in [('mmapy (interior point)', mmapy.mmasub), ('mmadual (dual)', mmadual.mmasub)]:
        hist, xf, cron, comp_final, validation_time, nit, conv = otimiza(nome, sub, NELX, NELY, MAXITER, RMIN, x0, TOL)
        # total = sum of the timed components (the warm-up solve, the mesh and filter set-up and the final
        # verification solve are not MMA iterations and are excluded)
        resultados[nome] = (hist, xf, cron, cron['fe'] + cron['sens'] + cron['mma'], comp_final, nit, conv)

    print(f"\n== summary: each solver run until max change <= {TOL} (or {MAXITER} iterations)")
    print(f"{'solver':24s} | {'iters':>5} | {'conv':>4} | {'FE [s]':>8} | {'sens [s]':>8} | {'MMA [s]':>8} | {'total [s]':>9} | {'s/iter':>7} | {'MMA share':>9} | {'final compliance':>16} | {'final vol':>9}")
    for nome, (hist, xf, cron, total, comp_final, nit, conv) in resultados.items():
        print(f"{nome:24s} | {nit:5d} | {'yes' if conv else 'NO':>4} | {cron['fe']:8.2f} | {cron['sens']:8.2f} | {cron['mma']:8.3f} | {total:9.2f} | {total/nit:7.3f} | {cron['mma']/total:9.1%} | {comp_final:16.6f} | {np.mean(xf):9.4f}")
    (h1, x1, c1, t1, f1, n1, _), (h2, x2, c2, t2, f2, n2, _) = resultados.values()
    print(f"\nMMA subproblem time per iteration: mmapy {c1['mma']/n1:.4f} s, mmadual {c2['mma']/n2:.4f} s, ratio {(c1['mma']/n1)/(c2['mma']/n2):.1f}; "
          f"total time ratio {t1/t2:.2f} ({n1} against {n2} iterations), time per iteration ratio {(t1/n1)/(t2/n2):.2f}")
    k = min(n1, n2)
    print(f"max |compliance_mmapy - compliance_mmadual| over the first {k} iterations: {np.abs(np.array(h1['c'][:k]) - np.array(h2['c'][:k])).max():.3e} "
          f"(relative {np.abs(np.array(h1['c'][:k]) - np.array(h2['c'][:k])).max()/np.abs(h1['c'][:k]).max():.1e}); "
          f"max |xPhys_mmapy - xPhys_mmadual| at the end: {np.abs(x1 - x2).max():.3e}")
    out = f"mbb_{NELX}x{NELY}.npz"
    np.savez(out, x_mmapy=x1, x_mmadual=x2, iterations_mmapy=n1, iterations_mmadual=n2, tol=TOL,
             final_compliance_mmapy=f1, final_compliance_mmadual=f2,
             final_volume_mmapy=float(np.mean(x1)), final_volume_mmadual=float(np.mean(x2)),
             **{f"{k}_mmapy": np.array(v) for k, v in h1.items()},
             **{f"{k}_mmadual": np.array(v) for k, v in h2.items()}, config=configuracao())
    print(f"histories and designs saved to {out}")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 2*8*NELY/NELX + 0.8))
        for a_, xf, nome in zip(ax, (x1, x2), resultados):
            a_.imshow(1 - xf, cmap='gray', vmin=0, vmax=1, interpolation='none')
            a_.set_title(nome, fontsize=9); a_.axis('off')
        fig.tight_layout(); fig.savefig(out.replace('.npz', '.png'), dpi=150)
        print(f"designs saved to {out.replace('.npz', '.png')}")
    except ImportError:
        pass
