"""
Generates tests/data/reference_mmapy.npz: the subproblem coefficients built by GCMMA-MMA-Python (mmapy)
for a set of inputs, captured by intercepting the call to its interior-point routine `subsolv`, plus the
outputs of its GCMMA helpers asymp, concheck and raaupdate. The file is the independent numerical
reference of tests/test_approximations.py; it was generated with mmapy 0.3.1 and is committed so that the
tests do not need mmapy installed.

Run:  python tests/data/make_reference.py      (requires mmapy)

Cases (all with xmin = 0, xmax = 1 except where noted):
  mma_iter1_m1, mma_iter2_m1   iterations 1 and 2 (initial asymptotes), one constraint
  mma_iter3_m5_osc             iteration 3 with oscillating and monotone variables, five constraints
  mma_iter7_m3_bounds          iteration 7 with variables at xmin and xmax and asymptotes at their limits
  mma_iter3_m2_range           iteration 3 with xmin, xmax of different ranges and a tiny range (xmami floor)
  gcmma_m3                     gcmmasub with raa0, raa given; asymp at iterations 1 and 3; raaupdate; concheck
"""
import os
import numpy as np
import mmapy
from mmapy import mma as _mma

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_mmapy.npz')
rng = np.random.default_rng(2026)
captured = {}


def capture_subsolv(*args, **kwargs):
    "records the coefficients passed to mmapy's subsolv and then calls it"
    m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d = args[:16]
    captured['last'] = dict(low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, P=P, Q=Q, b=b)
    return _orig_subsolv(*args, **kwargs)


_orig_subsolv = _mma.subsolv
_mma.subsolv = capture_subsolv
ref = {}


def store(name, inputs, coef):
    for k, v in inputs.items():
        ref[f'{name}/in/{k}'] = np.asarray(v, dtype=float)
    for k, v in coef.items():
        ref[f'{name}/out/{k}'] = np.asarray(v, dtype=float)


def mma_case(name, m, n, it, xval, xold1, xold2, low, upp, xmin=None, xmax=None, move=0.5, **kw):
    xmin = np.zeros((n, 1)) if xmin is None else xmin
    xmax = np.ones((n, 1)) if xmax is None else xmax
    df0dx, dfdx, fval = rng.standard_normal((n, 1)), rng.standard_normal((m, n)), rng.uniform(-1, 1, (m, 1))
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1))
    out = mmapy.mmasub(m, n, it, xval, xmin, xmax, xold1, xold2, 0.0, df0dx, fval, dfdx, low, upp, a0, a, c, d, move=move, **kw)
    coef = dict(captured['last'])
    coef['low_out'], coef['upp_out'] = out[9], out[10]
    store(name, dict(m=m, n=n, iter=it, xval=xval, xold1=xold1, xold2=xold2, xmin=xmin, xmax=xmax, low=low, upp=upp,
                     df0dx=df0dx, dfdx=dfdx, fval=fval, move=move, **{k: v for k, v in kw.items()}), coef)


n = 40
x = rng.uniform(0.1, 0.9, (n, 1))
mma_case('mma_iter1_m1', 1, n, 1, x, x, x, np.zeros((n, 1)), np.ones((n, 1)))
mma_case('mma_iter2_m1', 1, n, 2, x, x, x, np.zeros((n, 1)), np.ones((n, 1)))
# iteration 3: half of the variables oscillate, half move monotonically, a few did not move
x1 = np.clip(x + 0.05*rng.standard_normal((n, 1)), 0, 1); x2 = np.clip(x1 + 0.05*np.sign(rng.standard_normal((n, 1)))*np.r_[np.ones(n//2), -np.ones(n - n//2)].reshape(-1, 1), 0, 1)
x2[:3] = x1[:3]
low, upp = x - 0.4, x + 0.4
mma_case('mma_iter3_m5_osc', 5, n, 3, x2, x1, x, low, upp, move=0.3)
# iteration 7: variables at the bounds, asymptotes outside the admissible band (to be clipped)
xb = x.copy(); xb[:5] = 0.0; xb[5:10] = 1.0
mma_case('mma_iter7_m3_bounds', 3, n, 7, xb, xb + 0.02, xb - 0.02, xb - 20.0, xb + 0.001, move=0.1, asymin=0.01, asymax=10)
# different ranges, one tiny range (xmax - xmin below the 1e-5 floor)
xmin = -2.0*np.ones((n, 1)); xmax = 3.0*np.ones((n, 1)); xmin[0], xmax[0] = 0.5, 0.5 + 1e-7
xr = xmin + (xmax - xmin)*rng.uniform(0.2, 0.8, (n, 1))
mma_case('mma_iter3_m2_range', 2, n, 3, xr, xr + 0.1*(xmax - xmin)*0.1, xr - 0.1*(xmax - xmin)*0.1, xr - 0.5*(xmax - xmin), xr + 0.5*(xmax - xmin),
         xmin=xmin, xmax=xmax, move=0.2, asyinit=0.4, asydecr=0.6, asyincr=1.3, raa0=1e-4, albefa=0.2)

# ---- GCMMA
m = 3
xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
df0dx, dfdx, fval, f0val = rng.standard_normal((n, 1)), rng.standard_normal((m, n)), rng.uniform(-1, 1, (m, 1)), 2.5
raa0eps, raaeps, epsimin = 1e-6, 1e-6*np.ones((m, 1)), 1e-7
low1, upp1, raa0_1, raa_1 = mmapy.asymp(1, n, x, x, x, xmin, xmax, np.zeros((n, 1)), np.ones((n, 1)), 0.01, 0.01*np.ones((m, 1)), raa0eps, raaeps, df0dx, dfdx)
low3, upp3, raa0_3, raa_3 = mmapy.asymp(3, n, x2, x1, x, xmin, xmax, low, upp, 0.01, 0.01*np.ones((m, 1)), raa0eps, raaeps, df0dx, dfdx)
a0, a, c, d = 1.0, np.zeros((m, 1)), 1e3*np.ones((m, 1)), np.zeros((m, 1))
out = mmapy.gcmmasub(m, n, 3, epsimin, x2, xmin, xmax, low3, upp3, raa0_3, raa_3, f0val, df0dx, fval, dfdx, a0, a, c, d)
coef = dict(captured['last']); coef['f0app'], coef['fapp'] = out[9], out[10]
xmma = out[0]
f0new, fnew = np.array([[f0val + 0.3]]), fval + np.array([[0.2], [-0.1], [0.05]])                 # new values: two not conservative
conserv = mmapy.concheck(m, epsimin, out[9], f0new, out[10], fnew)
# mmapy.raaupdate modifies its `raa` argument in place: pass a copy so that asymp3_raa stays the asymp output
raa0_u, raa_u = mmapy.raaupdate(xmma, x2, xmin, xmax, low3, upp3, f0new, fnew, out[9], out[10], raa0_3, raa_3.copy(), raa0eps, raaeps, epsimin)
store('gcmma_m3', dict(m=m, n=n, xval=x2, xold1=x1, xold2=x, xmin=xmin, xmax=xmax, low_in=low, upp_in=upp, df0dx=df0dx, dfdx=dfdx,
                       fval=fval, f0val=f0val, raa0eps=raa0eps, raaeps=raaeps, epsimin=epsimin, f0new=f0new, fnew=fnew, x1=x),
      dict(asymp1_low=low1, asymp1_upp=upp1, asymp1_raa0=raa0_1, asymp1_raa=raa_1,
           asymp3_low=low3, asymp3_upp=upp3, asymp3_raa0=raa0_3, asymp3_raa=raa_3,
           **coef, xmma=xmma, concheck=conserv, raaupdate_raa0=raa0_u, raaupdate_raa=raa_u))

np.savez(OUT, **ref)
_mma.subsolv = _orig_subsolv
print(f"{len(ref)} arrays written to {OUT} with mmapy {__import__('importlib.metadata').metadata.version('mmapy')}")
