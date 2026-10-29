"""
Equivalence of the subproblem construction with the reference data of tests/data/reference_mmapy.npz
(coefficients captured from GCMMA-MMA-Python 0.3.1, see tests/data/make_reference.py), for the MMA and
the GCMMA, and of the GCMMA helpers asymp, concheck and raaupdate. Agreement is expected to rounding
(relative 1e-13), the only differences being the order of the floating-point operations.

If mmapy is installed, the same comparison is also made live against it.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mmadual import mma_subproblem, gcmma_subproblem, asymp, concheck, raaupdate

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference_mmapy.npz')
REF = np.load(DATA)
RTOL = 1e-13
MMA_CASES = sorted({k.split('/')[0] for k in REF.files if k.startswith('mma_')})


def case(name):
    "inputs and outputs of a reference case as two dicts"
    ins = {k.split('/')[2]: REF[k] for k in REF.files if k.startswith(f'{name}/in/')}
    outs = {k.split('/')[2]: REF[k] for k in REF.files if k.startswith(f'{name}/out/')}
    return ins, outs


def close(a, b, name, scale=None):
    """relative agreement to RTOL; `scale` gives the magnitude of the terms whose difference is compared when the
       result itself is small (fapp = r + g(x) is a difference of O(10) terms that nearly cancel)"""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ref = np.abs(b) if scale is None else np.maximum(np.abs(b), np.abs(np.asarray(scale, dtype=float).ravel()))
    err = np.max(np.abs(a - b)/np.maximum(1e-300, ref))
    assert err <= RTOL, f"{name}: relative difference {err:.2e}"


@pytest.mark.parametrize('name', MMA_CASES)
def test_mma_subproblem_matches_reference(name):
    "low, upp, alfa, beta, p0, q0, P, Q, b of the MMA subproblem against the captured mmapy coefficients"
    ins, outs = case(name)
    kw = {k: float(ins[k]) for k in ('move', 'asyinit', 'asydecr', 'asyincr', 'asymin', 'asymax', 'raa0', 'albefa') if k in ins}
    m, n = int(ins['m']), int(ins['n'])
    sub = mma_subproblem(int(ins['iter']), ins['xval'], ins['xmin'], ins['xmax'], ins['xold1'], ins['xold2'], ins['df0dx'],
                         ins['fval'], ins['dfdx'], ins['low'], ins['upp'], 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m), **kw)
    for k in ('low', 'upp', 'alfa', 'beta', 'p0', 'q0', 'P', 'Q', 'b'):
        close(sub[k], outs[k], f"{name}/{k}")
    close(sub['low'], outs['low_out'], f"{name}/low returned")
    close(sub['upp'], outs['upp_out'], f"{name}/upp returned")
    assert sub['P'].shape == (m, n) and sub['b'].shape == (m,)


def test_gcmma_subproblem_matches_reference():
    "coefficients and constants r0, r (through f0app, fapp at the reference xmma) of the GCMMA subproblem"
    ins, outs = case('gcmma_m3')
    m = int(ins['m'])
    sub = gcmma_subproblem(ins['xval'], ins['xmin'], ins['xmax'], outs['asymp3_low'], outs['asymp3_upp'], outs['asymp3_raa0'].item(),
                           outs['asymp3_raa'], ins['f0val'].item(), ins['df0dx'], ins['fval'], ins['dfdx'], 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m))
    for k in ('low', 'upp', 'alfa', 'beta', 'p0', 'q0', 'P', 'Q', 'b'):
        close(sub[k], outs[k], f"gcmma/{k}")
    x = outs['xmma'].ravel()
    f0app = sub['r0'] + np.sum(sub['p0']/(sub['upp'] - x) + sub['q0']/(x - sub['low']))
    fapp = sub['r'] + sub['P'] @ (1/(sub['upp'] - x)) + sub['Q'] @ (1/(x - sub['low']))
    close(f0app, outs['f0app'], 'gcmma/f0app', scale=sub['r0'])
    close(fapp, outs['fapp'], 'gcmma/fapp', scale=sub['r'])


def test_gcmma_helpers_match_reference():
    "asymp at iterations 1 and 3, concheck and raaupdate against the captured mmapy outputs"
    ins, outs = case('gcmma_m3')
    m, n = int(ins['m']), int(ins['n'])
    raa0eps, raaeps, epsimin = ins['raa0eps'].item(), ins['raaeps'], ins['epsimin'].item()
    low1, upp1, raa0_1, raa_1 = asymp(1, n, ins['x1'], ins['x1'], ins['x1'], ins['xmin'], ins['xmax'], np.zeros(n), np.ones(n),
                                      0.01, 0.01*np.ones(m), raa0eps, raaeps, ins['df0dx'], ins['dfdx'])
    close(low1, outs['asymp1_low'], 'asymp1/low'); close(upp1, outs['asymp1_upp'], 'asymp1/upp')
    close(raa0_1, outs['asymp1_raa0'], 'asymp1/raa0'); close(raa_1, outs['asymp1_raa'], 'asymp1/raa')
    low3, upp3, raa0_3, raa_3 = asymp(3, n, ins['xval'], ins['xold1'], ins['xold2'], ins['xmin'], ins['xmax'], ins['low_in'], ins['upp_in'],
                                      0.01, 0.01*np.ones(m), raa0eps, raaeps, ins['df0dx'], ins['dfdx'])
    close(low3, outs['asymp3_low'], 'asymp3/low'); close(upp3, outs['asymp3_upp'], 'asymp3/upp')
    close(raa0_3, outs['asymp3_raa0'], 'asymp3/raa0'); close(raa_3, outs['asymp3_raa'], 'asymp3/raa')
    assert concheck(m, epsimin, outs['f0app'], ins['f0new'], outs['fapp'], ins['fnew']) == int(outs['concheck'])
    assert concheck(m, epsimin, outs['f0app'], outs['f0app'] - 1.0, outs['fapp'], outs['fapp'] - 1.0) == 1
    assert concheck(m, epsimin, outs['f0app'], outs['f0app'] + 2*epsimin, outs['fapp'], outs['fapp']) == 0
    raa_3_copy = raa_3.copy()
    raa0_u, raa_u = raaupdate(outs['xmma'], ins['xval'], ins['xmin'], ins['xmax'], low3, upp3, ins['f0new'], ins['fnew'],
                              outs['f0app'], outs['fapp'], raa0_3, raa_3, raa0eps, raaeps, epsimin)
    close(raa0_u, outs['raaupdate_raa0'], 'raaupdate/raa0'); close(raa_u, outs['raaupdate_raa'], 'raaupdate/raa')
    grow = ins['fnew'].ravel() > outs['fapp'].ravel() + 0.5*epsimin
    assert np.all((raa_u.ravel() > raa_3.ravel()) == grow), "only the non-conservative constraints are updated"
    assert (raa0_u > raa0_3) == (ins['f0new'].item() > outs['f0app'].item() + 0.5*epsimin)
    assert np.array_equal(raa_3, raa_3_copy), "raaupdate must not modify its input raa (mmapy does)"


def test_subproblem_properties():
    "structural properties of the construction: strict convexity, interpolation of the gradient, bounds"
    ins, outs = case('mma_iter3_m5_osc')
    m = int(ins['m'])
    sub = mma_subproblem(3, ins['xval'], ins['xmin'], ins['xmax'], ins['xold1'], ins['xold2'], ins['df0dx'], ins['fval'], ins['dfdx'],
                         ins['low'], ins['upp'], 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m), move=float(ins['move']))
    x = ins['xval'].ravel()
    assert np.all(sub['p0'] > 0) and np.all(sub['q0'] > 0) and np.all(sub['P'] > 0) and np.all(sub['Q'] > 0)
    assert np.all(sub['low'] < sub['alfa']) and np.all(sub['alfa'] <= x) and np.all(x <= sub['beta']) and np.all(sub['beta'] < sub['upp'])
    # derivative of the approximation at xval = gradient + regularization (which is symmetric in p and q, so the
    # regularization adds (reg/ux^2 - reg/xl^2)*... ; check the exact identity of the construction instead)
    ux, xl = sub['upp'] - x, x - sub['low']
    dapprox0 = sub['p0']/ux**2 - sub['q0']/xl**2
    pos, neg = np.maximum(ins['df0dx'].ravel(), 0), np.maximum(-ins['df0dx'].ravel(), 0)
    reg = 1e-3*(pos + neg) + 1e-5/np.maximum((ins['xmax'] - ins['xmin']).ravel(), 1e-5)
    assert np.allclose(dapprox0, (pos + reg) - (neg + reg), rtol=1e-12)
    g_at_x = sub['P'] @ (1/ux) + sub['Q'] @ (1/xl) - sub['b']
    assert np.allclose(g_at_x, ins['fval'].ravel(), rtol=1e-12), "g_i(xval) - b_i must equal f_i(xval)"


try:
    import mmapy
    from mmapy import mma as _mmapy_mma
except ImportError:
    mmapy = None


@pytest.mark.skipif(mmapy is None, reason="mmapy not installed")
def test_live_against_mmapy():
    "same comparison against the installed mmapy, on fresh random data (both constructions, iteration 4)"
    rng = np.random.default_rng(11)
    n, m = 30, 4
    got = {}
    orig = _mmapy_mma.subsolv

    def capture(*args, **kwargs):
        got.update(dict(zip(('low', 'upp', 'alfa', 'beta', 'p0', 'q0', 'P', 'Q'), args[3:11])), b=args[13])
        return orig(*args, **kwargs)
    _mmapy_mma.subsolv = capture
    try:
        x = rng.uniform(0.2, 0.8, (n, 1)); x1 = x + 0.05*rng.standard_normal((n, 1)); x2 = x1 + 0.05*rng.standard_normal((n, 1))
        low, upp = x - 0.3, x + 0.3
        df0dx, dfdx, fval = rng.standard_normal((n, 1)), rng.standard_normal((m, n)), rng.uniform(-1, 1, (m, 1))
        z = np.zeros((n, 1)); o = np.ones((n, 1))
        mmapy.mmasub(m, n, 4, x2, z, o, x1, x, 0.0, df0dx, fval, dfdx, low, upp, 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1)), move=0.2)
        sub = mma_subproblem(4, x2, z, o, x1, x, df0dx, fval, dfdx, low, upp, 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m), move=0.2)
        for k in got:
            close(sub[k], got[k], f"live mma/{k}")
        raa0, raa = 0.02, 0.03*np.ones((m, 1))
        mmapy.gcmmasub(m, n, 4, 1e-7, x2, z, o, low, upp, raa0, raa, np.array([[1.0]]), df0dx, fval, dfdx, 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1)))
        sub = gcmma_subproblem(x2, z, o, low, upp, raa0, raa, 1.0, df0dx, fval, dfdx, 1.0, np.zeros(m), 1e3*np.ones(m), np.zeros(m))
        for k in got:
            close(sub[k], got[k], f"live gcmma/{k}")
    finally:
        _mmapy_mma.subsolv = orig
