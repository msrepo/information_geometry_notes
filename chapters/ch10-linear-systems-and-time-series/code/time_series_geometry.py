#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 10, checked by hand.

Linear systems and time series. Every number quoted in the notes comes from here.

The idea of the chapter: a stationary Gaussian time series is the output of a linear filter H fed with white noise; all that
matters is its power spectrum S(w) = |H(e^{iw})|^2, which is one variance per frequency. Variances at different frequencies are
independent, so the set L of spectra is a product of one-dimensional Gaussian scale families, an exponential family indexed by w:
natural parameter 1/S, expectation parameter S, dual potentials (1/2) int log S and its negative, a diagonal metric, a Euclidean
geometry in log S, and dual flatness for every alpha. Fourier coefficients of S are the autocovariances (m-affine coordinates),
Fourier coefficients of 1/S the inverse autocovariances (e-affine coordinates): AR models are the e-flat family with finitely many
inverse autocovariances, MA models the m-flat family with finitely many autocovariances, ARMA models neither, and the Pythagorean
theorem gives Yule-Walker, maximum entropy and its dual.

Running examples, with the book's sign convention (10.15)/(10.64), H(z) = (1 + b z^-1)/(1 + a z^-1), i.e. x_t = -a x_{t-1} + e_t + b e_{t-1}:
  AR(1):     a = -0.6 (so x_t = 0.6 x_{t-1} + e_t), free innovation variance s;     MA(1): b = 0.5;     ARMA(1,1): (a, b) = (-0.5, 0.3);
  AR(2): (a_0, a_1, a_2) = (1, -0.9, 0.5) and (1, 0.4, 0.3);   MA(2): (1, 0.5, 0.2) and (1, -0.6, 0.3).
All integrals over frequency are done on a uniform grid of M = 4096 points, where the trapezoid rule is exact for trigonometric
polynomials and geometrically convergent for the analytic spectra used here; Toeplitz computations (finite stretches of n
observations) are exact linear algebra.

Checked here (the headers of the printout carry the same numbers; the notes cite them by topic):

  1.  the spectrum of a system (10.6)-(10.14), (10.40)-(10.42): autocovariances are the cosine coefficients; the sign mismatch between
      (10.63) and (10.64); minimum phase and what Gaussian noise cannot see (same covariance matrix, different third cumulant);
  1b. the periodogram is not deterministic (10.12); neighbouring ordinates and phases (10.25);
  2.  the model families as printed (10.15)-(10.23): index slips in (10.18), (10.21);
  3a. the Gaussian likelihood against its spectral form (10.26); the circulant model, where it is exact;
  3b. potentials and the Legendre relation (10.30)-(10.32), the closed-form AR(1) cone, the coordinates r, r* and the pairing (10.43);
  3c. the metric (10.34)-(10.36), the divergence (10.37) and the entropy (10.38) against exact Toeplitz computations (three different
      normalisations); the Fisher information of ARMA(1,1) in closed form;
  3d. Theorem 10.1: alpha-connections of the scale family, the alpha-divergence (10.45);
  4a. e- and m-geodesics, Fejer-Riesz and the convex cone of stable AR(p), the exact matrix e-geodesic of a Toeplitz family;
  4b. embedding curvatures of AR(1), MA(1), ARMA(1,1) (spectral limit and exact finite length);
  5a. the Yule-Walker projection, the Pythagorean theorem (10.57), prediction error; 5b. Theorems 10.2 and 10.3, the exact identities
      behind them, the hypotheses the printed statements leave out, the factor in (10.62); 5c. the two projections of an MA(1) onto AR(1);
  6a. ARMA(1,1): the singular line a = b, the Fisher determinant, the two sheets, any common factor; 6b. the boundary of the stable region;
  6c. the last remark: a two-state Markov chain observed for 0 <= t <= T is a curved exponential family, with e-curvature of order 1/T.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Monte Carlo uses fixed seeds (one generator per check); the whole script takes about five seconds.

Run:  python3 time_series_geometry.py            (checks)
      python3 time_series_geometry.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

SEED = 20261002                             # every check that simulates builds its own generator from SEED + a constant, so the checks do not disturb one another
STORE = {}                                  # numbers computed by the checks, reused by the figures

M = 4096                                    # frequency grid
W = 2 * np.pi * np.arange(M) / M
EJ = np.exp(-1j * W)
LOG2PIE = math.log(2 * math.pi * math.e)


def head(s):
    print("\n" + s)


def fv(v, d=4):
    """Format a vector for printing, writing 0 for round-off."""
    return "[" + ", ".join("0" if abs(x) < 5e-13 else f"{x:.{d}f}" for x in v) + "]"


# ------------------------------------------------------------------ spectral toolbox

def pol(c, E=EJ):
    """sum_k c_k e^{-i k w} on the grid."""
    return sum(ck * E ** k for k, ck in enumerate(c))


def spec(A, B=(1.0,), s2=1.0):
    """S(w) = s2 |B(e^{-iw})|^2 / |A(e^{-iw})|^2: the transfer function (B/A)(z), z = e^{iw}, with a_0 = 1 unless A says otherwise."""
    return s2 * np.abs(pol(B)) ** 2 / np.abs(pol(A)) ** 2


def mean(f):
    """(1/2 pi) int_{-pi}^{pi} f dw."""
    return float(np.mean(f))


def cosc(f, K):
    """c_t = (1/2 pi) int f(w) cos(t w) dw, t = 0..K. For f = S these are the autocovariances r*_t = E[x_s x_{s-t}] of (10.42), because
    S = r*_0 + sum_{t>=1} r*_t 2 cos(t w), the basis (10.40)-(10.41)."""
    return np.fft.rfft(f).real[:K + 1] / M


def ent(S):
    """Entropy rate (10.38): (1/4 pi) int log S + (1/2) log(2 pi e)."""
    return 0.5 * mean(np.log(S)) + 0.5 * LOG2PIE


def kl_book(S1, S2):
    """The divergence (10.37) exactly as printed: (1/2 pi) int (S1/S2 - 1 - log(S1/S2)) dw."""
    return mean(S1 / S2 - 1 - np.log(S1 / S2))


def kl_rate(S1, S2):
    """The Kullback-Leibler rate per observation, (1/4 pi) int (...) dw: half of (10.37)."""
    return 0.5 * kl_book(S1, S2)


def levinson(g, pmax):
    """Yule-Walker/Levinson-Durbin on autocovariances g. Returns [(a, s2)] for p = 0..pmax with a = (a_1..a_p), the model
    x_t = -sum a_k x_{t-k} + e_t (book's sign), prediction error variance s2 = sigma_p^2."""
    out = [(np.zeros(0), float(g[0]))]
    a = np.zeros(0)
    s2 = float(g[0])
    for p in range(1, pmax + 1):
        acc = g[p] + sum(a[k] * g[p - 1 - k] for k in range(p - 1))
        kap = -acc / s2
        a = np.concatenate([a + kap * a[::-1], [kap]])
        s2 *= 1 - kap ** 2
        out.append((a.copy(), s2))
    return out


def ar_spec(a, s2):
    return s2 / np.abs(pol(np.concatenate([[1.0], a]))) ** 2


def toep(g, n):
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.asarray(g)[idx]


def logdet(A):
    return 2 * float(np.log(np.diag(np.linalg.cholesky(A))).sum())


def kl_gauss(A, B):
    """KL[N(0,A) || N(0,B)] for covariance matrices."""
    n = len(A)
    return 0.5 * (float(np.trace(np.linalg.solve(B, A))) - n - logdet(A) + logdet(B))


def impulse(A, B, K):
    """First K impulse-response coefficients of B(z^-1)/A(z^-1), a_0 = 1: h_k = b_k - sum_{j>=1} a_j h_{k-j}."""
    h = np.zeros(K)
    for k in range(K):
        h[k] = (B[k] if k < len(B) else 0.0) - sum(A[j] * h[k - j] for j in range(1, min(k, len(A) - 1) + 1))
    return h


# the running examples
A1, B1M, AB = -0.6, 0.5, (-0.5, 0.3)        # AR(1) coefficient, MA(1) coefficient, ARMA(1,1) pair (a, b)
S_AR1 = spec([1, A1])
S_MA1 = spec([1], [1, B1M])
S_ARMA = spec([1, AB[0]], [1, AB[1]])


# ------------------------------------------------------------------ 1. the spectrum of a system (section 10.1)

def check_system():
    head("1. The spectrum of a system (section 10.1)")
    print(f"   a tiny example, AR(1) with a = {A1} (x_t = {-A1} x_(t-1) + e_t, unit-variance white noise): S(w) = 1/(1.36 - 1.2 cos w), S(0) = {S_AR1[0]:.4f}, S(pi/2) = {S_AR1[M // 4]:.4f}, S(pi) = {S_AR1[M // 2]:.4f};")
    print(f"      autocovariances gamma_0, gamma_1, gamma_2, gamma_3 = {fv(cosc(S_AR1, 3), 4)} (= 1/(1 - 0.36) times 1, 0.6, 0.36, 0.216); inverse autocovariances r_0, r_1, r_2, r_3 = {fv(cosc(1 / S_AR1, 3), 4)}")
    print(f"      MA(1) with b = {B1M}: S(0) = {S_MA1[0]:.4f}, S(pi/2) = {S_MA1[M // 4]:.4f}, S(pi) = {S_MA1[M // 2]:.4f}; gamma_0..gamma_3 = {fv(cosc(S_MA1, 3), 4)}; r_0..r_3 = {fv(cosc(1 / S_MA1, 3), 4)}")
    a, b = AB
    K = 14
    h = impulse([1, a], [1, b], K)
    print(f"   ARMA(1,1), (a, b) = ({a}, {b}); (10.64): H(z) = (1 + b z^-1)/(1 + a z^-1), impulse response by long division h = {np.round(h[:6], 4).tolist()} (h_1 = b - a)")
    h_print = impulse([1, -a], [1, b], K)
    print(f"   the recursion printed in (10.63), x_t = a x_(t-1) + e_t + b e_(t-1), has H = (1 + b z^-1)/(1 - a z^-1) and h = {np.round(h_print[:6], 4).tolist()} (h_1 = a + b): a different system for the same (a, b);")
    print(f"   cancellation of numerator and denominator (H = 1, h_1 = 0) is on a = b for (10.64) but on a = -b for (10.63); at a = b = 0.4 the (10.64) response is {np.round(impulse([1, 0.4], [1, 0.4], 4), 4).tolist()}, the (10.63) one is {np.round(impulse([1, -0.4], [1, 0.4], 4), 4).tolist()}")
    hl = impulse([1, a], [1, b], 400)
    g_closed = [(1 + b * b - 2 * a * b) / (1 - a * a), (b - a) * (1 - a * b) / (1 - a * a)]
    g_time = [float(hl[: 400 - t] @ hl[t:]) for t in range(4)]
    g_spec = cosc(S_ARMA, 3)
    print(f"   (10.6): E x_t^2 = sum h_i^2 = {g_time[0]:.6f}; closed form (1 + b^2 - 2ab)/(1 - a^2) = {g_closed[0]:.6f}; (1/2pi) int S dw = {g_spec[0]:.6f}")
    print(f"   autocovariances E[x_s x_(s-t)], t = 0..3: from h {np.round(g_time, 6).tolist()}, from the spectrum (10.40)-(10.42) {np.round(g_spec, 6).tolist()}; max difference {max(abs(np.array(g_time) - g_spec)):.1e}; closed form gamma_1 = (b - a)(1 - ab)/(1 - a^2) = {g_closed[1]:.6f}")
    g200 = cosc(S_ARMA, 200)
    rebuilt = g200[0] + 2 * sum(g200[t] * np.cos(t * W) for t in range(1, 201))
    print(f"   S = r*_0 + sum_t r*_t 2 cos(t w) with r*_t the autocovariances (the basis (10.41)): rebuilt from t <= 200 with max error {np.max(np.abs(rebuilt - S_ARMA)):.1e}; note (1/2pi) int S = r*_0, not int S")
    # minimum phase
    bb = 0.4
    S_min = spec([1, a], [1, bb]); S_non = spec([1, a], [1, 1 / bb])
    print(f"   minimum phase: |1 + b e^-iw|^2 = b^2 |1 + e^-iw/b|^2, so the systems with b = {bb} and 1/b = {1 / bb} have spectra in the ratio b^2: max |S(b) - b^2 S(1/b)| = {np.max(np.abs(S_min - bb ** 2 * S_non)):.1e};")
    hA = np.array([1.0, bb]); hB = np.array([bb, 1.0])               # MA(1) with zero inside (minimum phase) and its reverse (zero outside)
    nn = 8
    cov = lambda hh: toep(np.concatenate([[hh @ hh, hh[0] * hh[1]], np.zeros(nn - 2)]), nn)
    print(f"      the two MA(1) filters h = (1, {bb}) and (b, 1) = ({bb}, 1) give the same covariance matrix of {nn} observations (max difference {np.max(np.abs(cov(hA) - cov(hB))):.1e}), hence the same Gaussian likelihood: Gaussian data cannot tell them apart")
    R = 4_000_000
    e = np.random.default_rng(SEED + 1).exponential(1.0, R + 1) - 1.0                       # centred exponential noise, third cumulant 2
    eA = e[1:] + bb * e[:-1]; eA1 = e[2:] + bb * e[1:-1]                # x_t = e_t + b e_(t-1) and x_(t+1)
    xA0 = e[1:-1] + bb * e[:-2]
    xB0 = bb * e[1:-1] + e[:-2]; xB1 = bb * e[2:] + e[1:-1]
    cA = float(np.mean(xA0 ** 2 * eA1)); cB = float(np.mean(xB0 ** 2 * xB1))
    print(f"      with skewed noise (kappa_3 = 2) the third-order cumulant cum(x_t, x_t, x_(t+1)) = kappa_3 sum_i h_i^2 h_(i+1) = kappa_3 b (= {2 * bb:.3f}) for the minimum-phase filter and kappa_3 b^2 (= {2 * bb * bb:.3f}) for its reverse;"
          f" simulated ({R // 1000000}e6 samples): {cA:.3f} and {cB:.3f}. Only non-Gaussian input can tell them apart, as the book says at the end of the chapter")
    S2x = 2.0 * S_ARMA
    print(f"   Szego: the leading coefficient of the minimum-phase filter is h_0 = exp((1/4 pi) int log S dw): {math.exp(0.5 * mean(np.log(S_ARMA))):.4f} for S above (h_0 = 1), {math.exp(0.5 * mean(np.log(S2x))):.4f} = sqrt(2) = {math.sqrt(2):.4f} for 2 S; the sign of h_0 is the one freedom left (H and -H have the same spectrum)")
    STORE["mp"] = (bb, S_min, S_non)


def check_periodogram():
    head("1b. The 'power spectrum' S = |X|^2 of (10.12) is not deterministic; independence in (10.25)")
    phi = -A1
    print(f"   AR(1) x_t = {phi} x_(t-1) + e_t, I_n(w) = |sum_t x_t e^-iwt|^2 / n at the Fourier frequency w = 2 pi (n/8)/n = pi/4; 4000 paths per n (2000 for n = 8192), exact stationary start:")
    print("      n        mean I/S   std I/S    corr(I(w), I(w + 2pi/n))   |mean of e^(i phase)|   std of the average of I/S over 9 ordinates")
    out = {}
    rng_p = np.random.default_rng(SEED + 2)
    for n, R in ((128, 4000), (1024, 4000), (8192, 2000)):
        rows = []
        for ch in range(R // 500):
            x = np.empty((500, n))
            x[:, 0] = rng_p.normal(size=500) / math.sqrt(1 - phi ** 2)
            ee = rng_p.normal(size=(500, n))
            for t in range(1, n):
                x[:, t] = phi * x[:, t - 1] + ee[:, t]
            X = np.fft.fft(x, axis=1)
            I = np.abs(X) ** 2 / n
            j = n // 8
            Sw = lambda jj: 1 / abs(1 - phi * np.exp(-2j * np.pi * jj / n)) ** 2
            rows.append((I[:, j] / Sw(j), np.angle(X[:, j]), I[:, j + 1] / Sw(j + 1), np.mean([I[:, j + d] / Sw(j + d) for d in range(-4, 5)], axis=0)))
        r0, ph, r1, sm = (np.concatenate([q[i] for q in rows]) for i in range(4))
        out[n] = (r0.mean(), r0.std())
        print(f"      {n:<8d} {r0.mean():.4f}     {r0.std():.4f}     {np.corrcoef(r0, r1)[0, 1]:+.4f}                    {abs(np.mean(np.exp(1j * ph))):.4f}                  {sm.std():.4f}")
    print("   the mean of I/S is 1 (E|X|^2 = S) but its standard deviation stays at 1 however long the record: |X|^2 has an exponential law with mean S, not a deterministic value;")
    print("   it is E|X(w)|^2 that equals S(w) (this is what (10.28) uses). Ordinates at different Fourier frequencies are uncorrelated and the phase is uniform (mean resultant length ~ 1/sqrt(R)), as (10.25) and the text say;")
    print("   averaging 9 neighbouring ordinates cuts the standard deviation to about 1/3 (1/sqrt 9), which is how a spectrum is estimated in practice")
    print("   (10.25) as printed has an absolute value, E|X(w) X(w')| = 0 for w' != w, impossible for a non-negative variable; the conjugate is meant, E[X(w) conj X(w')]; and for a real series X(-w) = conj X(w), so independence holds only for w' != +-w")
    STORE["periodogram"] = out


# ------------------------------------------------------------------ 2. the model families as printed (section 10.2)

def check_models():
    head("2. The model families as printed (section 10.2): index conventions")
    bq = np.array([0.5, -0.4, 0.3])                    # printed MA(3): x_t = b_1 e_(t-1) + b_2 e_(t-2) + b_3 e_(t-3)
    ac = [float(bq[: max(3 - t, 0)] @ bq[t:]) if t < 3 else 0.0 for t in range(5)]
    print(f"   (10.18)-(10.20) as printed: x_t = sum_(i=1)^q b_i e_(t-i). For q = 3, b = {bq.tolist()}: autocovariances E[x_s x_(s-t)], t = 0..4: {np.round(ac, 4).tolist()}:")
    print("      the lag-3 autocovariance is already 0, so the printed 'MA(3)' has 3 parameters and is a delayed MA(2); (10.52)-(10.53) describe MA(q) by q + 1 coordinates r*_0..r*_q with r*_(q+1) = 0, which needs b_0 as well, as in AR(p) of (10.15), whose a_0 is the gain")
    a0, a1, b1 = 1.0, -0.5, 0.3
    K = 6
    hp = np.zeros(K)
    for t in range(K):                                   # printed (10.21), p = q = 1: x_t = -a_0 x_t - a_1 x_(t-1) + b_1 e_(t-1)  (impulse input)
        prev = hp[t - 1] if t >= 1 else 0.0
        hp[t] = (-a1 * prev + (b1 if t == 1 else 0.0)) / (1 + a0)
    h22 = np.zeros(K)
    for t in range(K):                                   # (10.22): H = b_1 z^-1/(a_0 + a_1 z^-1), i.e. a_0 x_t = -a_1 x_(t-1) + b_1 e_(t-1)
        prev = h22[t - 1] if t >= 1 else 0.0
        h22[t] = (-a1 * prev + (b1 if t == 1 else 0.0)) / a0
    print(f"   (10.21) as printed has the sum -sum_(i=0)^p a_i x_(t-i) on the right, which contains -a_0 x_t: for p = q = 1, (a_0, a_1, b_1) = ({a0}, {a1}, {b1}) its impulse response is {np.round(hp, 4).tolist()},")
    print(f"      while the transfer function (10.22) b_1 z^-1/(a_0 + a_1 z^-1) has {np.round(h22, 4).tolist()}: the printed recursion is not the system of (10.22) (the sum should start at i = 1 with a_0 x_t on the left)")
    print("   dimension count: (a_0..a_p, b_0..b_q) overparametrise ARMA(p,q) by one scale (H is unchanged under (a, b) -> (c a, c b)), so the family has p + q + 1 parameters, as AR(p) has p + 1 and MA(q) has q + 1")


# ------------------------------------------------------------------ 3. the dual geometry (section 10.3)

def gamma_fn(kind):
    """Autocovariance vector (length n) of a parametrised family, computed from the spectrum on the grid (accurate to ~1e-15)."""
    if kind == "ar1":                                       # xi = (a, s): s/|1 + a e^-iw|^2
        return lambda xi, n: cosc(spec([1, xi[0]], [1], xi[1]), n - 1)
    if kind == "ma1":                                       # xi = (b, s): s|1 + b e^-iw|^2
        return lambda xi, n: cosc(spec([1], [1, xi[0]], xi[1]), n - 1)
    if kind == "arma":                                      # xi = (a, b), unit gain (10.64)
        return lambda xi, n: cosc(spec([1, xi[0]], [1, xi[1]]), n - 1)
    raise ValueError(kind)


def fisher_exact(gfun, xi, n, h=1e-5):
    """Exact Fisher information of n consecutive observations of the Gaussian process, (1/2) tr(S^-1 dS_i S^-1 dS_j), derivatives by central differences."""
    xi = np.array(xi, float)
    k = len(xi)
    S0 = toep(gfun(xi, n), n)
    Si = np.linalg.inv(S0)
    D = []
    for i in range(k):
        e = np.zeros(k); e[i] = h
        D.append((toep(gfun(xi + e, n), n) - toep(gfun(xi - e, n), n)) / (2 * h))
    return np.array([[0.5 * np.trace(Si @ D[i] @ Si @ D[j]) for j in range(k)] for i in range(k)])


def fisher_spec(Sfun, xi, h=1e-5):
    """Whittle's per-observation Fisher information (1/4 pi) int d_i log S d_j log S dw, and the same integral without the 1/(4 pi) normalisation changed."""
    xi = np.array(xi, float)
    k = len(xi)
    L = []
    for i in range(k):
        e = np.zeros(k); e[i] = h
        L.append((np.log(Sfun(*(xi + e))) - np.log(Sfun(*(xi - e)))) / (2 * h))
    return np.array([[0.5 * mean(L[i] * L[j]) for j in range(k)] for i in range(k)])


def arma_G(a, b):
    """Closed-form per-observation Fisher information of ARMA(1,1) in (a, b): [[1/(1-a^2), -1/(1-ab)], [-1/(1-ab), 1/(1-b^2)]]."""
    return np.array([[1 / (1 - a * a), -1 / (1 - a * b)], [-1 / (1 - a * b), 1 / (1 - b * b)]])


def check_likelihood():
    head("3a. The Gaussian likelihood and its spectral form (10.26): exact against Whittle, and the circulant model where it is exact")
    phi = -A1
    print(f"   AR(1), phi = {phi}: log-likelihood of n observations, exact (Toeplitz) minus the spectral form -(1/2) sum_j [log S(w_j) + I(w_j)/S(w_j)] (the circulant model), 2000 simulated records per n:")
    print("      n        mean(exact - spectral)   sd(exact - spectral)   sd of the exact log-likelihood")
    rng_l = np.random.default_rng(SEED + 3)
    for n in (32, 128, 512):
        R = 2000
        x = np.empty((R, n)); x[:, 0] = rng_l.normal(size=R) / math.sqrt(1 - phi ** 2)
        ee = rng_l.normal(size=(R, n))
        for t in range(1, n):
            x[:, t] = phi * x[:, t - 1] + ee[:, t]
        ex = -0.5 * n * math.log(2 * math.pi) + 0.5 * math.log(1 - phi ** 2) - 0.5 * ((1 - phi ** 2) * x[:, 0] ** 2 + ((x[:, 1:] - phi * x[:, :-1]) ** 2).sum(axis=1))
        wk = 2 * np.pi * np.arange(n) / n
        Sk = 1 / np.abs(1 - phi * np.exp(-1j * wk)) ** 2
        I = np.abs(np.fft.fft(x, axis=1)) ** 2 / n
        wh = -0.5 * n * math.log(2 * math.pi) - 0.5 * (np.log(Sk).sum() + (I / Sk).sum(axis=1))
        d = ex - wh
        print(f"      {n:<8d} {d.mean():+.4f}                  {d.std():.4f}                 {ex.std():.2f}")
    print("   the difference stays of order 1 while the log-likelihood itself fluctuates like sqrt(n): the symbol '~' in (10.26) is a statement about boundary terms, which do not grow with the length")
    n = 64
    wk = 2 * np.pi * np.arange(n) / n
    f1 = lambda w: np.abs(1 + 0.3 * np.exp(-1j * w)) ** 2 / np.abs(1 - 0.5 * np.exp(-1j * w)) ** 2
    f2 = lambda w: 1 / np.abs(1 - 0.3 * np.exp(-1j * w)) ** 2
    circ = lambda lam: np.fft.ifft(lam).real[(np.subtract.outer(np.arange(n), np.arange(n))) % n]
    C1, C2 = circ(f1(wk)), circ(f2(wk))
    l1, l2 = f1(wk), f2(wk)
    print(f"   circulant (periodic) series of period n = {n}, spectra S_1 = ARMA(1,1)(-0.5, 0.3) and S_2 = AR(1)(a = -0.3) at w_k = 2 pi k/n: KL[S_1:S_2] from the matrices {kl_gauss(C1, C2):.6f}, from (1/2) sum_k (S_1/S_2 - 1 - log S_1/S_2) {0.5 * np.sum(l1 / l2 - 1 - np.log(l1 / l2)):.6f}:"
          " exact. In this model every Fourier frequency is an independent Gaussian with variance S(w_k), (10.26) is an identity, and the geometry of the book is exactly dually flat")
    h = 1e-5
    Ci = np.linalg.inv(circ(spec([1, A1])[::M // n][:n]))
    dC = (circ(spec([1, A1 + h])[::M // n][:n]) - circ(spec([1, A1 - h])[::M // n][:n])) / (2 * h)
    mat = 0.5 * np.trace(Ci @ dC @ Ci @ dC)
    sp = 0.5 * np.sum(((np.log(spec([1, A1 + h])) - np.log(spec([1, A1 - h]))) / (2 * h))[::M // n][:n] ** 2)
    print(f"   its Fisher information for a: from the matrices {mat:.6f}, from (1/2) sum_k (d log S(w_k)/da)^2 {sp:.6f}; the Toeplitz (stationary, non-periodic) value differs from n/(1 - a^2) by an O(1) boundary term (see 3c)")


def check_potentials():
    head("3b. The potentials (10.30)-(10.32), the coordinates r, r* and the pairing (10.43)")
    S = 2.0 * S_ARMA                                                      # innovation variance 2, so that int log S != 0
    th, eta = 1 / S, -S / 2
    psi_s = math.pi * mean(np.log(S)) - math.pi / 2                       # (1/2) int log S dw - pi/2
    phi_s = -math.pi * mean(np.log(S)) - math.pi / 2                      # -(1/2) int log S dw - pi/2
    cross = 2 * math.pi * mean(th * eta)                                  # int theta eta dw
    print(f"   S = 2 x ARMA(1,1)(-0.5, 0.3): psi = (1/2) int log S - pi/2 = {psi_s:.6f}; phi = -(1/2) int log S - pi/2 = {phi_s:.6f}; int theta eta dw = -pi = {cross:.6f}; psi + phi - int theta eta dw = {psi_s + phi_s - cross:.1e} (10.32 holds)")
    print(f"   the first expression of (10.30), (1/2) int log(-theta) dw with theta = 1/S > 0, is not even real; the real part of what it asks for, (1/2) int log theta dw, is {math.pi * mean(np.log(th)):.6f} = -(1/2) int log S, the opposite sign of the second expression, {math.pi * mean(np.log(S)):.6f}:")
    print("      the first form should read -(1/2) int log theta dw - pi/2 (or, if theta = -1/S were used, -(1/2) int log(-theta) dw - pi/2); (10.31) and the second forms are right")
    # explicit AR(1) potential in the coordinates (r_0, r_1), innovation variance 2
    sg = 2.0
    S_ar = sg * S_AR1
    r0, r1 = (1 + A1 ** 2) / sg, A1 / sg                                  # 1/S = |a_0 + a_1 e^-iw|^2 with a_0 = 1/sqrt(s), a_1 = a/sqrt(s): r = (a_0^2 + a_1^2, a_0 a_1)
    psi_c = lambda p, q: -math.pi * math.log((p + math.sqrt(p * p - 4 * q * q)) / 2) - math.pi / 2
    g = cosc(S_ar, 2)
    hh = 1e-5
    grad = [(psi_c(r0 + hh, r1) - psi_c(r0 - hh, r1)) / (2 * hh), (psi_c(r0, r1 + hh) - psi_c(r0, r1 - hh)) / (2 * hh)]
    hess = np.array([[(psi_c(r0 + hh, r1) - 2 * psi_c(r0, r1) + psi_c(r0 - hh, r1)) / hh ** 2, (psi_c(r0 + hh, r1 + hh) - psi_c(r0 + hh, r1 - hh) - psi_c(r0 - hh, r1 + hh) + psi_c(r0 - hh, r1 - hh)) / (4 * hh * hh)],
                     [0, (psi_c(r0, r1 + hh) - 2 * psi_c(r0, r1) + psi_c(r0, r1 - hh)) / hh ** 2]])
    hess[1, 0] = hess[0, 1]
    e0, e1 = np.ones(M), 2 * np.cos(W)
    Gq = np.array([[math.pi * mean(S_ar ** 2 * u * v) for v in (e0, e1)] for u in (e0, e1)])
    print(f"   AR(1), a = {A1}, innovation variance s = {sg:.0f}: the whole free-gain family is the cone r_0 > 2|r_1| of the plane (r_0, r_1) = (a_0^2 + a_1^2, a_0 a_1), and psi(r) = (1/2) int log S - pi/2 = -pi log((r_0 + sqrt(r_0^2 - 4 r_1^2))/2) - pi/2 in closed form:")
    print(f"      at r = ({r0:.2f}, {r1:.2f}): closed form {psi_c(r0, r1):.6f}, quadrature {math.pi * mean(np.log(S_ar)) - math.pi / 2:.6f};"
          f" gradient {np.round(grad, 6).tolist()} = (-pi gamma_0, -2 pi gamma_1) = {np.round([-math.pi * g[0], -2 * math.pi * g[1]], 6).tolist()} (the expectation coordinates eta_t = int e_t eta dw);")
    print(f"      Hessian {np.round(hess, 3).tolist()} = the metric (10.34) in the r chart, (1/2) int S^2 e_t e_s dw = {np.round(Gq, 3).tolist()}")
    print(f"      gamma_0 = 1/sqrt(r_0^2 - 4 r_1^2) = {1 / math.sqrt(r0 ** 2 - 4 * r1 ** 2):.6f} = s/(1 - a^2) = {sg / (1 - A1 ** 2):.6f}")
    # coordinates of AR(2), MA(2), ARMA(1,1)
    K = 7
    print("   the two coordinate systems for three systems (r_t = inverse autocovariances, e-affine; r*_t = autocovariances, m-affine), t = 0..6:")
    A2 = [1, -0.9, 0.5]; B2 = [1, 0.5, 0.2]
    rows = {"AR(2) a = (1, -0.9, 0.5)": spec(A2), "MA(2) b = (1, 0.5, 0.2)": spec([1], B2), "ARMA(1,1) (-0.5, 0.3)": S_ARMA}
    for nm, Sx in rows.items():
        r = cosc(1 / Sx, K - 1); rs = cosc(Sx, K - 1)
        print(f"      {nm:26s} r  = {fv(r)}")
        print(f"      {'':26s} r* = {fv(rs)}")
    ra = np.array([A2[0] ** 2 + A2[1] ** 2 + A2[2] ** 2, A2[0] * A2[1] + A2[1] * A2[2], A2[0] * A2[2]])
    print(f"   AR(2): r_t = sum_s a_s a_(s+t) = {np.round(ra, 4).tolist()} for t = 0, 1, 2 and r_t = 0 beyond: the linear constraints (10.50); MA(2) likewise r*_t = sum_s b_s b_(s+t) (10.53)")
    # the pairing (10.43): <e_t, e*_s> = (1/2) int S^2 (d theta) (d theta_s) with d theta = e_t, d theta_s = -e_s / S^2
    Kp = 4
    E_ = [np.ones(M)] + [2 * np.cos(t * W) for t in range(1, Kp)]
    P = np.array([[0.5 * 2 * math.pi * mean(S_ARMA ** 2 * E_[t] * (-E_[s] / S_ARMA ** 2)) for s in range(Kp)] for t in range(Kp)])
    print(f"   (10.43): the pairing <e_t, e*_s> of the tangent vector of the r_t axis (d theta = e_t) and that of the r*_s axis (d S = e_s, d theta = -e_s/S^2), computed with (10.34), t, s = 0..3:")
    for row in P:
        print("      " + "  ".join(f"{v:+.4f}" for v in row))
    print(f"   = -pi, -2 pi, -2 pi, -2 pi on the diagonal ({-math.pi:.4f}, {-2 * math.pi:.4f}) and 0 off it, whatever S is: the two axes systems are biorthogonal (dual bases), not orthogonal; (10.43) as printed (= 0 for all t, s) would make the metric degenerate and is true only for t != s")
    STORE["pair"] = P


def check_normalisations():
    head("3c. The metric (10.34)-(10.36), the divergence (10.37) and the entropy (10.38) against exact Toeplitz computations")
    n_list = (8, 16, 32, 64, 128, 256, 512)
    S1, S2 = S_ARMA, spec([1, -0.3])
    Db = kl_book(S1, S2)
    print(f"   KL divergence: S_1 = ARMA(1,1)(-0.5, 0.3), S_2 = AR(1)(a = -0.3). (10.37) as printed, (1/2pi) int (S_1/S_2 - 1 - log S_1/S_2) dw = {Db:.6f}; the same with (1/4pi): {Db / 2:.6f}")
    print("      n      exact KL of n observations / n     2 x that      n (KL_n/n - (1/4pi) int ...)")
    g1, g2 = cosc(S1, 600), cosc(S2, 600)
    rows = []
    for n in n_list:
        k = kl_gauss(toep(g1, n), toep(g2, n))
        rows.append((n, k / n)); print(f"      {n:<6d} {k / n:.6f}                          {2 * k / n:.6f}      {n * (k / n - Db / 2):+.4f}")
    print(f"   so KL per observation tends to {Db / 2:.6f}, half of (10.37) (the error times n is the same constant, -0.0824, for every n listed: a boundary term); for white noises with variances s_1, s_2 (10.37) gives s_1/s_2 - 1 - log(s_1/s_2), twice the KL per observation (1/2)(...)")
    STORE["kl_rows"] = rows; STORE["Db"] = Db
    print("   entropy: (1/n) of the exact differential entropy (1/2) log det(2 pi e Sigma_n) of n observations against (10.38):")
    Hs = ent(S1)
    print(f"      H_S = {Hs:.6f} = log h_0 + (1/2) log(2 pi e) with h_0 = {math.exp(0.5 * mean(np.log(S1))):.4f} the first impulse-response coefficient (Szego);")
    print("      n      exact entropy / n     difference      n x difference")
    for n in n_list:
        hn = 0.5 * (n * LOG2PIE + logdet(toep(g1, n))) / n
        print(f"      {n:<6d} {hn:.6f}              {hn - Hs:+.2e}       {n * (hn - Hs):+.4f}")
    print("   (10.38) is right: it is the entropy rate per observation")
    # Fisher information
    a, b = AB
    G = arma_G(a, b)
    Gs = fisher_spec(lambda x, y: spec([1, x], [1, y]), [a, b])
    print(f"   Fisher information per observation of ARMA(1,1) at (a, b) = ({a}, {b}) in the coordinates (a, b):")
    print(f"      closed form [[1/(1-a^2), -1/(1-ab)], [-1/(1-ab), 1/(1-b^2)]] = {np.round(G, 6).tolist()}; (1/4pi) int d_i log S d_j log S dw = {np.round(Gs, 6).tolist()}")
    print("      exact Fisher information of n observations divided by n (central differences of the Toeplitz covariance), and n times the difference from the closed form:")
    ex_rows = {}
    for n in (16, 64, 256):
        I = fisher_exact(gamma_fn("arma"), [a, b], n) / n
        ex_rows[n] = I
        print(f"      n = {n:<4d} {np.round(I, 5).tolist()}   n (I/n - G) = {np.round((I - G) * n, 4).tolist()}")
    print(f"   so the Fisher information per observation is (1/4pi) int (d log S)^2; the metric (10.36), (1/2) int (d log S)^2, is {math.pi * 2:.4f} = 2 pi times that: {np.round(2 * math.pi * G, 4).tolist()} here;")
    print(f"   the metric induced by the divergence (10.37) (its Hessian, (1/2pi) int (d log S)^2) is 2 times the per-observation value; the three normalisations (10.34), (10.37), (10.38) differ, which affects no conclusion (they are constants) but not all of them can be 'per observation'")
    print("      in terms of record length: (10.36) is the Fisher information of 2 pi observations (Fourier frequencies spaced by exactly 1), (10.37) is the Kullback-Leibler divergence of 2 observations, (10.38) is the entropy of 1")
    STORE["fisher_ex"] = ex_rows; STORE["G"] = G
    # AR(1) and MA(1): the single-parameter cases
    print(f"   one-parameter cases at a = b = 0.5 (per observation): AR(1) g_aa = 1/(1 - a^2) = {1 / (1 - 0.25):.4f}; MA(1) g_bb = 1/(1 - b^2) = {1 / (1 - 0.25):.4f}; exact n = 256: AR(1) {fisher_exact(gamma_fn('ar1'), [0.5, 1.0], 256)[0, 0] / 256:.5f}, MA(1) {fisher_exact(gamma_fn('ma1'), [0.5, 1.0], 256)[0, 0] / 256:.5f} (gain s fixed in the derivative, s = 1)")
    # estimator variance
    phi = -A1; n, R = 400, 20000
    rng_e = np.random.default_rng(SEED + 4)
    x = np.empty((R, n)); x[:, 0] = rng_e.normal(size=R) / math.sqrt(1 - phi ** 2)
    ee = rng_e.normal(size=(R, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + ee[:, t]
    phat = (x[:, 1:] * x[:, :-1]).sum(axis=1) / (x[:, :-1] ** 2).sum(axis=1)
    print(f"   the metric is the right one for estimation: AR(1), phi = {phi}, n = {n}, {R} records: n Var(phi_hat) = {n * phat.var():.4f} (+- {n * phat.var() * math.sqrt(2 / R):.4f}), Cramer-Rao 1/g_aa = 1 - a^2 = {1 - A1 ** 2:.4f}")


# ------------------------------------------------------------------ 4. e-flat AR, m-flat MA (sections 10.3-10.4)

def fejer_riesz(r):
    """Minimum-phase factor of the trigonometric polynomial R(w) = r_0 + 2 sum_{t>=1} r_t cos(t w) > 0: a = (a_0, ..., a_p), a_0 > 0, with
    |sum_k a_k e^{-ikw}|^2 = R and all zeros of sum_k a_k w^k outside the unit disc, i.e. a stable AR(p) model (10.17) with 1/S = R."""
    p = len(r) - 1
    z = np.roots(np.concatenate([r[:0:-1], r]))                 # roots of w^p R(w): reciprocal pairs
    wk = z[np.abs(z) > 1]                                        # the p roots outside the unit disc
    al = (np.poly(wk)[::-1] * ((-1) ** p / np.prod(wk))).real     # prod (1 - w/w_k), ascending powers
    rho = np.array([al[: p + 1 - t] @ al[t:] for t in range(p + 1)])
    a = math.sqrt(r[0] / rho[0]) * al
    return a, float(np.max(np.abs(rho * r[0] / rho[0] - r)))


def acf_of(a):
    """r_t = sum_s a_s a_(s+t): the inverse autocovariances of the AR model with coefficients a (or the autocovariances of the MA model with b = a)."""
    a = np.asarray(a, float)
    return np.array([a[: len(a) - t] @ a[t:] for t in range(len(a))])


def rand_stable_batch(rng, p, n):
    """n monic polynomials z^p + a_1 z^(p-1) + ... + a_p (rows (1, a_1..a_p), p = 1, 2, 3) with all roots inside the unit disc: a quadratic factor with either a complex pair
    (modulus 0.97 sqrt(U), argument pi U) or two real roots in (-0.97, 0.97), times, for p = 3, one more real root."""
    one = np.ones(n)
    r = 0.97 * np.sqrt(rng.random(n)); th = np.pi * rng.random(n)
    x1, x2 = 0.97 * (2 * rng.random(n) - 1), 0.97 * (2 * rng.random(n) - 1)
    if p == 1:
        return np.stack([one, -x1], axis=1)                                   # z - x1
    pair = rng.random(n) < 0.5
    q1 = np.where(pair, -2 * r * np.cos(th), -(x1 + x2)); q0 = np.where(pair, r * r, x1 * x2)
    if p == 2:
        return np.stack([one, q1, q0], axis=1)
    x = 0.97 * (2 * rng.random(n) - 1)
    return np.stack([one, q1 - x, q0 - x * q1, -x * q0], axis=1)           # (z - x)(z^2 + q1 z + q0)


def max_root_modulus(C):
    """Largest root modulus of each monic polynomial z^p + a_1 z^(p-1) + ... + a_p (rows of C), through the eigenvalues of the companion matrices."""
    n, p1 = C.shape
    p = p1 - 1
    comp = np.zeros((n, p, p)); comp[:, 0, :] = -C[:, 1:]
    for i in range(1, p):
        comp[:, i, i - 1] = 1
    return np.abs(np.linalg.eigvals(comp)).max(axis=1)


def check_flatness():
    head("4a. AR is e-flat, MA is m-flat: geodesics, the cone of AR(p), and what the flatness means in the coefficients")
    A2a, A2b = [1, -0.9, 0.5], [1, 0.4, 0.3]
    S1, S2 = spec(A2a), spec(A2b)
    print(f"   AR(2) a = {A2a} and {A2b} (both stable: largest root moduli {max(abs(np.roots(A2a))):.4f}, {max(abs(np.roots(A2b))):.4f}); inverse autocovariances r_t, t = 0..6, along the geodesics from the first to the second:")
    for lab, mix in (("e-geodesic: 1/S_t = (1-t)/S_1 + t/S_2", lambda t: 1 / ((1 - t) / S1 + t / S2)), ("m-geodesic: S_t = (1-t) S_1 + t S_2    ", lambda t: (1 - t) * S1 + t * S2)):
        for t in (0.5,):
            print(f"      {lab}, t = {t}: r = {fv(cosc(1 / mix(t), 6))}")
    ea, err = fejer_riesz(0.5 * acf_of(A2a) + 0.5 * acf_of(A2b))
    print(f"   so the e-geodesic stays in AR(2) (r_3 = r_4 = ... = 0, the linear constraints (10.50)) and the m-geodesic leaves it (r_3 = {cosc(1 / (0.5 * S1 + 0.5 * S2), 6)[3]:.5f}, r_4 = {cosc(1 / (0.5 * S1 + 0.5 * S2), 6)[4]:.5f}, ...)")
    print(f"   in the coefficients a the e-geodesic is not a straight line: its midpoint (a_0, a_1, a_2) = {fv(ea, 5)} (Fejer-Riesz factorisation of r_mid, error {err:.1e}), the straight midpoint of the coefficient vectors is {fv(0.5 * np.array(A2a) + 0.5 * np.array(A2b), 5)};"
          " the coefficients a are not affine coordinates, r_t = sum_s a_s a_(s+t) are")
    n_ = 200
    Pm = 0.5 * np.linalg.inv(toep(cosc(S1, n_), n_)) + 0.5 * np.linalg.inv(toep(cosc(S2, n_), n_))
    Sm_ = np.linalg.inv(Pm)
    gsp = cosc(1 / (0.5 / S1 + 0.5 / S2), 3)
    far = np.abs(np.subtract.outer(np.arange(n_), np.arange(n_))) > 2
    print(f"   the same e-geodesic in the exact model of n = {n_} observations (mixture of the two precision matrices, an e-geodesic of the Gaussian family): the precision matrix stays banded (max |P_ij| for |i - j| > 2: {np.max(np.abs(Pm[far])):.1e}),")
    print(f"      in the bulk its covariance has the autocovariances of the spectral e-geodesic: gamma_0..gamma_3 from row {n_ // 2}: {fv([Sm_[n_ // 2, n_ // 2 + k] for k in range(4)], 6)} against {fv(gsp, 6)}; the diagonal at i = 0, 1, 2, 5, {n_ // 2} is {fv([Sm_[i, i] for i in (0, 1, 2, 5, n_ // 2)], 6)}: it differs at the ends of the record, so the Toeplitz family is not e-flat at finite length")
    B2a, B2b = [1, 0.5, 0.2], [1, -0.6, 0.3]
    M1, M2 = spec([1], B2a), spec([1], B2b)
    print(f"   MA(2) b = {B2a} and {B2b}: autocovariances r*_t, t = 0..6, at t = 0.5: m-geodesic {fv(cosc(0.5 * M1 + 0.5 * M2, 6))} stays (r*_3 = ... = 0); e-geodesic {fv(cosc(1 / (0.5 / M1 + 0.5 / M2), 6))} leaves")
    sP, sQ = spec([1, -0.6]), spec([1, 0.5], [1], 2.0)
    mP, mQ = spec([1], [1, 0.5]), spec([1], [1, -0.6], 2.0)
    print(f"   Figure 1: AR(1) pair P = (a = -0.6, s = 1), Q = (a = 0.5, s = 2) with r = {fv(cosc(1 / sP, 2), 3)}, {fv(cosc(1 / sQ, 2), 3)}; at t = 0.5 the e-geodesic has r = {fv(0.5 * cosc(1 / sP, 2) + 0.5 * cosc(1 / sQ, 2), 3)}, the m-geodesic r = {fv(cosc(1 / (0.5 * sP + 0.5 * sQ), 3), 3)} (r_2 != 0: not AR(1));")
    print(f"      MA(1) pair P' = (b = 0.5, s = 1), Q' = (b = -0.6, s = 2) with gamma = {fv(cosc(mP, 2), 3)}, {fv(cosc(mQ, 2), 3)}; at t = 0.5 the m-geodesic has gamma = {fv(0.5 * cosc(mP, 2) + 0.5 * cosc(mQ, 2), 3)}, the e-geodesic gamma = {fv(cosc(1 / (0.5 / mP + 0.5 / mQ), 3), 3)} (gamma_2 != 0: not MA(1))")
    P1, P2 = spec([1, -0.5], [1, 0.3]), spec([1, 0.6], [1, -0.4])
    print("   ARMA(1,1): an ARMA(1,1) spectrum has gamma_(k+1)/gamma_k = -a for all k >= 2. Ratios for k = 2..6:")
    for nm, Sx in (("S_1 (a = -0.5, b = 0.3)", P1), ("S_2 (a = 0.6, b = -0.4)", P2), ("m-midpoint (S_1 + S_2)/2", 0.5 * P1 + 0.5 * P2), ("e-midpoint 1/(1/S_1 + 1/S_2)*2", 1 / (0.5 / P1 + 0.5 / P2))):
        g = cosc(Sx, 7)
        print(f"      {nm:32s} {fv(g[3:8] / g[2:7])}")
    print("      neither geodesic stays in ARMA(1,1), consistent with 'neither e-flat nor m-flat' (the proof is the curvature computed in 4b)")
    # Fejer-Riesz and the cone
    rng = np.random.default_rng(10)
    for p, nn in ((2, 20000), (3, 20000)):
        C1, C2 = rand_stable_batch(rng, p, nn), rand_stable_batch(rng, p, nn)
        mr = max_root_modulus(0.5 * (C1 + C2))
        bad = int(np.sum(mr >= 1))
        poly_str = "z^2 + a_1 z + a_2" if p == 2 else "z^3 + a_1 z^2 + a_2 z + a_3"
        print(f"   pairs of stable AR({p}) models ({poly_str}, roots drawn at random inside the unit disc, coefficient vectors (1, a_1..a_{p})): the straight midpoint in the coefficients is unstable in {bad} of {nn} pairs", end="")
        if bad:
            i = int(np.argmax(mr))
            print(f"; the worst case, {np.round(C1[i], 3).tolist()} and {np.round(C2[i], 3).tolist()}, has the unstable midpoint {np.round(0.5 * (C1[i] + C2[i]), 3).tolist()} (largest root modulus {mr[i]:.4f})")
            STORE["unstable_pair"] = (C1[i], C2[i])
        else:
            print(" (the stability region of AR(2) is the triangle |a_2| < 1, |a_1| < 1 + a_2: convex)")
    worst, pos = 0.0, 0
    Cs1, Cs2 = rand_stable_batch(rng, 3, 1000), rand_stable_batch(rng, 3, 1000)
    for c1, c2 in zip(Cs1, Cs2):
        ra = 0.37 * acf_of(c1) + 0.63 * acf_of(c2)
        a_, e_ = fejer_riesz(ra)
        worst = max(worst, e_)
        pos += int(np.min(0.37 * np.abs(pol(c1)) ** 2 + 0.63 * np.abs(pol(c2)) ** 2) > 0 and np.min(np.abs(np.roots(a_[::-1]))) > 1)
    print(f"   the same kind of pairs along the e-geodesic (weights 0.37, 0.63, in the inverse autocovariances r): {pos} of 1000 midpoints are positive spectra with a stable AR(3) factor (Fejer-Riesz), largest reconstruction error {worst:.1e}: the r-coordinates make the stable AR(p) models a convex cone, the a-coordinates do not")


def curv_spec(Sfun, xi, h=1e-4):
    """Squared norms of the e- and m-embedding curvatures per observation of a family xi -> S(.; xi) in the limit of infinite length:
    normal parts of d^2 theta (theta = 1/S, metric (1/4 pi) int S^2 u v) and of d^2 S (metric (1/4 pi) int u v / S^2), contracted with g^-1 g^-1."""
    xi = np.array(xi, float); k = len(xi)
    S0 = Sfun(*xi)
    D1, D2 = [], [[None] * k for _ in range(k)]
    for i in range(k):
        e = np.zeros(k); e[i] = h
        D1.append((Sfun(*(xi + e)) - Sfun(*(xi - e))) / (2 * h))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h; ej = np.zeros(k); ej[j] = h
            if i == j:
                d = (Sfun(*(xi + ei)) - 2 * S0 + Sfun(*(xi - ei))) / h ** 2
            else:
                d = (Sfun(*(xi + ei + ej)) - Sfun(*(xi + ei - ej)) - Sfun(*(xi - ei + ej)) + Sfun(*(xi - ei - ej))) / (4 * h * h)
            D2[i][j] = D2[j][i] = d
    th1 = [-D1[i] / S0 ** 2 for i in range(k)]
    th2 = [[-D2[i][j] / S0 ** 2 + 2 * D1[i] * D1[j] / S0 ** 3 for j in range(k)] for i in range(k)]
    ipth = lambda u, w: 0.5 * mean(S0 ** 2 * u * w)
    ipet = lambda u, w: 0.5 * mean(u * w / S0 ** 2)
    g = np.array([[ipet(D1[i], D1[j]) for j in range(k)] for i in range(k)])
    gi = np.linalg.inv(g)

    def normal(v, T, ip):
        c = gi @ np.array([ip(v, T[d]) for d in range(k)])
        return v - sum(c[d] * T[d] for d in range(k))

    def norm2(N, ip):
        return sum(gi[a, c] * gi[b, d] * ip(N[a][b], N[c][d]) for a in range(k) for b in range(k) for c in range(k) for d in range(k))

    Ne = [[normal(th2[i][j], th1, ipth) for j in range(k)] for i in range(k)]
    Nm = [[normal(D2[i][j], D1, ipet) for j in range(k)] for i in range(k)]
    return g, norm2(Ne, ipth), norm2(Nm, ipet)


def sigma_ar1(phi, s, n):
    return s * toep(phi ** np.arange(n), n) / (1 - phi ** 2)


def sigma_ma1(b, s, n):
    g = np.zeros(n); g[0] = s * (1 + b * b); g[1] = s * b
    return toep(g, n)


def curv_exact(fun, xi, h=1e-4):
    """The same two squared curvatures for the exact Gaussian family of n observations with covariance fun(xi) (Toeplitz), inside the family of all
    zero-mean Gaussians: theta = -Sigma^-1/2 with <u, w> = 2 tr(u Sigma w Sigma); eta = Sigma with <u, w> = (1/2) tr(Sigma^-1 u Sigma^-1 w)."""
    xi = np.array(xi, float); k = len(xi)
    S = fun(*xi)
    D1 = []
    D2 = [[None] * k for _ in range(k)]
    for i in range(k):
        e = np.zeros(k); e[i] = h
        D1.append((fun(*(xi + e)) - fun(*(xi - e))) / (2 * h))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h; ej = np.zeros(k); ej[j] = h
            if i == j:
                d = (fun(*(xi + ei)) - 2 * S + fun(*(xi - ei))) / h ** 2
            else:
                d = (fun(*(xi + ei + ej)) - fun(*(xi + ei - ej)) - fun(*(xi - ei + ej)) + fun(*(xi - ei - ej))) / (4 * h * h)
            D2[i][j] = D2[j][i] = d
    Si = np.linalg.inv(S)
    th1 = [0.5 * Si @ D1[i] @ Si for i in range(k)]
    th2 = [[-0.5 * (-Si @ D2[i][j] @ Si + Si @ D1[i] @ Si @ D1[j] @ Si + Si @ D1[j] @ Si @ D1[i] @ Si) for j in range(k)] for i in range(k)]
    ipth = lambda u, w: 2 * float(np.trace(u @ S @ w @ S))
    ipet = lambda u, w: 0.5 * float(np.trace(Si @ u @ Si @ w))
    g = np.array([[ipet(D1[i], D1[j]) for j in range(k)] for i in range(k)])
    gi = np.linalg.inv(g)

    def normal(v, T, ip):
        c = gi @ np.array([ip(v, T[d]) for d in range(k)])
        return v - sum(c[d] * T[d] for d in range(k))

    def norm2(N, ip):
        return sum(gi[a, c] * gi[b, d] * ip(N[a][b], N[c][d]) for a in range(k) for b in range(k) for c in range(k) for d in range(k))

    Ne = [[normal(th2[i][j], th1, ipth) for j in range(k)] for i in range(k)]
    Nm = [[normal(D2[i][j], D1, ipet) for j in range(k)] for i in range(k)]
    return g, norm2(Ne, ipth), norm2(Nm, ipet)


def kappa_e2_three_statistics(phi, s, n, h=1e-4):
    """The exact AR(1) likelihood of n observations is a (3, 2) curved exponential family: -(1/2s)[T_1 - 2 phi T_2 + phi^2 T_3] with T_1 = sum x_t^2,
    T_2 = sum x_t x_(t-1), T_3 = sum_(t=2)^(n-1) x_t^2, i.e. statistics (T_3, T_2, D = x_1^2 + x_n^2) and natural parameters
    (-(1 + phi^2)/(2s), phi/s, -1/(2s)). The e-curvature of the surface is computed in that three-dimensional family (a different route from curv_exact)."""
    Sg = sigma_ar1(phi, s, n)
    A3 = np.zeros((n, n)); A3[np.arange(1, n - 1), np.arange(1, n - 1)] = 1
    A2 = np.zeros((n, n)); A2[np.arange(1, n), np.arange(n - 1)] = 0.5; A2 = A2 + A2.T
    AD = np.zeros((n, n)); AD[0, 0] = 1; AD[n - 1, n - 1] = 1
    As = [A3, A2, AD]
    G = np.array([[2 * np.trace(As[i] @ Sg @ As[j] @ Sg) for j in range(3)] for i in range(3)])
    th = lambda ph, ss: np.array([-(1 + ph ** 2) / (2 * ss), ph / ss, -1 / (2 * ss)])
    xi = np.array([phi, s]); J = np.zeros((3, 2)); H = np.zeros((3, 2, 2))
    for i in range(2):
        e = np.zeros(2); e[i] = h
        J[:, i] = (th(*(xi + e)) - th(*(xi - e))) / (2 * h)
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = h; ej = np.zeros(2); ej[j] = h
            H[:, i, j] = (th(*(xi + ei)) - 2 * th(*xi) + th(*(xi - ei))) / h ** 2 if i == j else (th(*(xi + ei + ej)) - th(*(xi + ei - ej)) - th(*(xi - ei + ej)) + th(*(xi - ei - ej))) / (4 * h * h)
    gi = np.linalg.inv(J.T @ G @ J)
    N = np.zeros((3, 2, 2))
    for i in range(2):
        for j in range(2):
            N[:, i, j] = H[:, i, j] - J @ gi @ (J.T @ G @ H[:, i, j])
    return sum(gi[a, c] * gi[b, d] * (N[:, a, b] @ G @ N[:, c, d]) for a in range(2) for b in range(2) for c in range(2) for d in range(2))


def check_curvature():
    head("4b. Embedding curvatures: AR(1) is e-flat and MA(1) is m-flat, ARMA(1,1) is neither (infinite length), and what finite length does")
    f_ar = lambda a, s: spec([1, a], [1], s)
    f_ma = lambda b, s: spec([1], [1, b], s)
    f_arma = lambda a, b, s: spec([1, a], [1, b], s)
    print("   infinite length, per observation: squared norms |H^e|^2, |H^m|^2 of the e- and m-embedding curvature of the free-gain families (parameters (a, s), (b, s), (a, b, s), s = 1):")
    print("      family            parameter        |H^e|^2        |H^m|^2        4/(1 - a^2)")
    for a in (-0.6, 0.3, 0.8):
        _, he, hm = curv_spec(f_ar, [a, 1.0])
        print(f"      AR(1)             a = {a:+.1f}        {he:.1e}      {hm:.4f}        {4 / (1 - a * a):.4f}")
    for b in (0.5, -0.3, 0.8):
        _, he, hm = curv_spec(f_ma, [b, 1.0])
        print(f"      MA(1)             b = {b:+.1f}        {he:.4f}       {hm:.1e}      {4 / (1 - b * b):.4f}")
    _, he, hm = curv_spec(f_arma, [AB[0], AB[1], 1.0])
    print(f"      ARMA(1,1)         (a, b) = {AB}   {he:.4f}        {hm:.4f}       (both non-zero: neither e-flat nor m-flat)")
    STORE["curv_arma"] = (he, hm)
    print("   with the gain pinned (a_0 = 1, the one-parameter family S = 1/|1 + a e^-iw|^2, a curve in the flat plane (r_0, r_1): the parabola r_0 = 1 + r_1^2) AR(1) is no longer e-flat, and the extra curvature is a constant:")
    print("      a        |H^e|^2 (pinned)    |H^m|^2 (pinned)    2 + 4/(1 - a^2)")
    for a in (-0.9, -0.6, 0.0, 0.3, 0.8):
        _, he1, hm1 = curv_spec(lambda a_: spec([1, a_]), [a])
        print(f"      {a:+.1f}      {he1:.4f}              {hm1:.4f}              {2 + 4 / (1 - a * a):.4f}")
    print("      (and, dually, for the pinned MA(1) |H^m|^2 = 2 and |H^e|^2 = 2 + 4/(1 - b^2), checked at b = 0.5: "
          f"{curv_spec(lambda b_: spec([1], [1, b_]), [0.5])[2]:.4f}, {curv_spec(lambda b_: spec([1], [1, b_]), [0.5])[1]:.4f} against {2 + 4 / 0.75:.4f})")
    print("   exact length n (Toeplitz covariances, inside the Gaussian family of n observations); AR(1) at (phi, s) = (0.6, 1) and MA(1) at (b, s) = (0.5, 1):")
    print("      n      AR(1): n^2 |H^e|^2   n |H^m|^2       MA(1): n |H^e|^2   |H^m|^2        |H^e|^2(n/2)/|H^e|^2(n)")
    prev = None
    rows = []
    for n in (10, 20, 40, 80, 160):
        _, he_a, hm_a = curv_exact(lambda p_, s_: sigma_ar1(p_, s_, n), [0.6, 1.0])
        _, he_m, hm_m = curv_exact(lambda b_, s_: sigma_ma1(b_, s_, n), [0.5, 1.0])
        rows.append((n, he_a, hm_a, he_m, hm_m))
        ratio = "" if prev is None else f"{prev / he_a:.3f}"
        print(f"      {n:<6d} {n * n * he_a:<20.4f} {n * hm_a:<14.4f} {n * he_m:<18.4f} {hm_m:<13.1e} {ratio}")
        prev = he_a
    print(f"   limits per observation: AR(1) |H^m|^2 = 4/(1 - a^2) = {4 / (1 - 0.36):.4f} (n |H^m|^2 -> it); MA(1) |H^e|^2 = 4/(1 - b^2) = {4 / (1 - 0.25):.4f} (n |H^e|^2 -> it); the exact AR(1) e-curvature is O(1/n^2) in the squared norm: halving it takes a ratio 4 per doubling of n (the ratios above tend to 4);")
    print(f"   |H^m|^2 of MA(1) is zero for every n (the covariance matrices gamma_0 I + gamma_1 (J + J^T) are a linear space: MA(q) is exactly m-flat at finite length), while AR(1) is e-flat only up to boundary terms")
    n = 40
    k3 = kappa_e2_three_statistics(0.6, 1.0, n)
    k2 = curv_exact(lambda p_, s_: sigma_ar1(p_, s_, n), [0.6, 1.0])[1]
    print(f"   cross-check by a second route (AR(1) as a (3, 2) curved exponential family with statistics sum x_t^2 over the inner points, sum x_t x_(t-1), x_1^2 + x_n^2): n = {n}: |H^e|^2 = {k3:.8f} against the matrix computation {k2:.8f}")
    STORE["curv_rows"] = rows


# ------------------------------------------------------------------ 5. projections, Pythagoras, maximum and minimum entropy (section 10.4)

E4 = EJ[::4]                                  # a coarser grid (1024 points) for the derivative-free optimisations


def nelder_mead(f, x0, step=0.2, iters=500):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float); x[i] += step; pts.append(x)
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = np.argsort(vals); pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def pacf_to_ar(u):
    """Stable AR coefficients (a_1..a_p) from unconstrained numbers u via reflection coefficients tanh(u) and the Levinson step-up."""
    a = np.zeros(0)
    for k in np.tanh(u):
        a = np.concatenate([a + k * a[::-1], [k]])
    return a


def check_projections():
    head("5a. The m-projection onto AR(p) is the Yule-Walker model; Pythagoras (10.57)")
    b = B1M
    S = S_MA1
    sinf = math.exp(mean(np.log(S)))
    g = cosc(S, 20)
    lv = levinson(g, 6)
    S0 = np.ones(M)
    print(f"   S = MA(1), b = {b} (gamma_0, gamma_1 = {g[0]:.4f}, {g[1]:.4f}), sigma_inf^2 = exp((1/2pi) int log S) = {sinf:.4f}; the m-projection S_p of S onto AR(p) has the same gamma_0..gamma_p (Yule-Walker, Levinson-Durbin):")
    print("      p   a_1..a_p                         sigma_p^2   closed form (1-b^(2p+4))/(1-b^(2p+2))   D[S:S_p] (1/4pi)   (1/2) log(sigma_p^2/sigma_inf^2)   match of gamma_0..gamma_p")
    rows = []
    for p_, (a, s2) in enumerate(lv):
        Sp = ar_spec(a, s2)
        D = kl_rate(S, Sp)
        mt = np.max(np.abs(cosc(Sp, p_) - g[: p_ + 1]))
        rows.append((p_, a, s2, D))
        print(f"      {p_}   {fv(a[:3], 4):30s} {s2:.6f}    {(1 - b ** (2 * p_ + 4)) / (1 - b ** (2 * p_ + 2)):.6f}                              {D:.6f}           {0.5 * math.log(s2 / sinf):.6f}                          {mt:.1e}")
    STORE["proj_rows"] = rows
    S0w = np.ones(M)
    print(f"   Pythagoras (10.57) for this S with S_0 = white noise, p = 0..4: D[S:S_0] = {kl_rate(S, S0w):.6f} = D[S:S_p] + D[S_p:S_0] with")
    for p_ in range(0, 5):
        Sp_ = ar_spec(*lv[p_])
        print(f"      p = {p_}: D[S:S_p] = {kl_rate(S, Sp_):.6f}, D[S_p:S_0] = {kl_rate(Sp_, S0w):.6f}, sum {kl_rate(S, Sp_) + kl_rate(Sp_, S0w):.6f}")
    print("   the divergence of the m-projection is (1/2) log(sigma_p^2/sigma_inf^2): the log-ratio of the p-th order and the infinite-order prediction error variances (classical prediction theory, a different route to the same number);")
    print("   the printed (10.37) is twice these numbers")
    S1m = ar_spec(*lv[1])
    print(f"   the higher autocovariances of the stochastic realisation are NOT zero: S = MA(1) has gamma_0..gamma_5 = {fv(cosc(S, 5), 4)}, its AR(1) projection {fv(cosc(S1m, 5), 4)} (equal up to lag 1, then geometric);")
    print(f"      it is the inverse autocovariances of an AR(p) that vanish beyond lag p: r_0..r_4 of S_1 = {fv(cosc(1 / S1m, 4), 4)}, of S = MA(1) {fv(cosc(1 / S, 4), 4)}; the paragraph before (10.57) says both things (once with r_(p+1) not 0, once 'are 0') because r and r* are mixed")
    # brute force: m-projection by derivative-free minimisation of the divergence
    print("   brute force: minimise D[S:S'] over stable AR(p) models S' = s/|A|^2 by Nelder-Mead (profile over s: s = mean(S|A|^2), D = (1/2) log(mean(S|A|^2)/sigma_inf^2)), parametrised by reflection coefficients:")
    for nm, Sx in (("MA(1) b = 0.5", S_MA1), ("ARMA(1,1) (-0.5, 0.3)", S_ARMA)):
        gx = cosc(Sx, 8)
        for p_ in (1, 2):
            f = lambda u: math.log(mean(Sx[::4] * np.abs(pol(np.concatenate([[1.0], pacf_to_ar(u)]), E4)) ** 2))
            u, v = nelder_mead(f, np.zeros(p_) + 0.05)
            a_nm = pacf_to_ar(u); a_yw = levinson(gx, p_)[p_][0]
            print(f"      {nm:22s} p = {p_}: Nelder-Mead a = {fv(a_nm, 6)}, Yule-Walker a = {fv(a_yw, 6)}, max difference {np.max(np.abs(a_nm - a_yw)):.1e}; minimum of mean(S |A|^2) {math.exp(v):.8f} = sigma_p^2 {levinson(gx, p_)[p_][1]:.8f}")
    # Pythagoras
    rng = np.random.default_rng(3)
    print("   Pythagoras (10.57), D[S:S_0] = D[S:S_p] + D[S_p:S_0], for S = ARMA(1,1)(-0.5, 0.3), S_0 = white noise and S_0 = random stable AR(p) models (10.57 holds for every S_0 in the e-flat AR(p)):")
    gA = cosc(S_ARMA, 8)
    for p_ in (1, 2, 3):
        a, s2 = levinson(gA, p_)[p_]
        Sp = ar_spec(a, s2)
        res = [abs(kl_rate(S_ARMA, S0) - kl_rate(S_ARMA, Sp) - kl_rate(Sp, S0))]
        for _ in range(5):
            c = rand_stable_batch(rng, p_, 1)[0]; S0r = np.exp(rng.normal()) / np.abs(pol(c)) ** 2
            res.append(abs(kl_rate(S_ARMA, S0r) - kl_rate(S_ARMA, Sp) - kl_rate(Sp, S0r)))
        print(f"      p = {p_}: D[S:S_0] = {kl_rate(S_ARMA, S0):.6f}, D[S:S_p] = {kl_rate(S_ARMA, Sp):.6f}, D[S_p:S_0] = {kl_rate(Sp, S0):.6f}; largest violation over white noise and 5 random S_0: {max(res):.1e}")
    pp = []
    for p_ in range(0, 6):
        a, s2 = levinson(gA, p_)[p_]
        Sp = ar_spec(a, s2)
        pp.append((p_, kl_rate(S_ARMA, Sp), kl_rate(Sp, S0)))
    STORE["pyth_p"] = pp


def random_cosine_perturbations(rng, order0, n, chunk=500):
    """Chunks of n random trigonometric polynomials sum_(k=0)^5 c_k 2 cos((order0 + k) w), c_k ~ N(0, u^2) with a random scale u: perturbations with no
    cosine component of order below order0, i.e. that keep the first order0 coefficients of whatever they are added to."""
    COS = np.array([2 * np.cos((order0 + k) * W) for k in range(6)])
    left = n
    while left > 0:
        m = min(chunk, left); left -= m
        c = rng.normal(size=(m, 6)) * rng.random((m, 1))
        yield c @ COS


def check_entropy():
    head("5b. Theorem 10.2 (maximum entropy), Theorem 10.3 (minimum entropy) and the proofs' constants")
    S = S_MA1
    g = cosc(S, 20)
    sinf = math.exp(mean(np.log(S)))
    print("   (10.58) with c_0 = r*_0 = (1/2pi) int S dw: D[S:S_0] = -2 H_S + log(2 pi e) + c_0 - 1; checked for S = MA(1) b = 0.5 and ARMA(1,1):")
    for nm, Sx in (("MA(1)", S_MA1), ("ARMA(1,1)", S_ARMA)):
        c0 = mean(Sx)
        print(f"      {nm:10s} D[S:S_0] (10.37) = {kl_book(Sx, np.ones(M)):.6f}; -2 H_S + log(2 pi e) + c_0 - 1 = {-2 * ent(Sx) + LOG2PIE + c0 - 1:.6f}")
    print("   Theorem 10.2: among spectra with the same r*_0..r*_p, S_p (the m-projection) has the largest entropy rate. H(S_p) - H(S) = (1/2) log(sigma_p^2/sigma_inf^2) > 0 and decreases to 0 as p grows (Burg's proof: sigma_inf^2 <= sigma_p^2, more past cannot hurt the predictor):")
    for p_ in range(0, 5):
        a, s2 = levinson(g, p_)[p_]
        print(f"      p = {p_}: H(S_p) - H(S) = {ent(ar_spec(a, s2)) - ent(S):.6f}")
    rng = np.random.default_rng(11)
    p_ = 1
    a, s2 = levinson(g, p_)[p_]
    Sp = ar_spec(a, s2)
    for frac, lab in ((0.6, "up to 60% of min S_1"), (0.05, "up to 5% of min S_1")):
        worst, ratios, ident = -1.0, [], 0.0
        for delta in random_cosine_perturbations(rng, p_ + 1, 4000):
            delta = delta * (frac * Sp.min() / np.maximum(np.abs(delta).max(axis=1, keepdims=True), 1e-12))
            Sx = Sp + delta
            d = 0.5 * np.mean(np.log(Sx), axis=1) - 0.5 * mean(np.log(Sp))                            # H(S') - H(S_1)
            klr = 0.5 * np.mean(Sx / Sp - 1 - np.log(Sx / Sp), axis=1)                                 # D[S':S_1] (true rate)
            worst = max(worst, float(d.max()))
            ratios.append(d / (-0.25 * np.mean(delta ** 2 / Sp ** 2, axis=1)))
            ident = max(ident, float(np.abs(-d - klr).max()))
        print(f"   4000 random perturbations of S_1 inside M_1(r) (cosines of order 2..7 added, so r*_0 and r*_1 are unchanged; size {lab}): the largest H(S') - H(S_1) is {worst:.2e} (all negative); ratio to the second-order prediction -(1/4) mean(delta^2/S_1^2): {np.mean(np.concatenate(ratios)):.4f}")
        print(f"      exact identity H(S_1) - H(S') = D[S':S_1] (the true Kullback-Leibler rate, (1/4pi) int), the one-line proof of Theorem 10.2: largest violation {ident:.1e}")
    print("      the first-order term vanishes by the orthogonality of M_p(r) and AR(p) and the second-order term is the metric, so S_p is a strict maximum")
    for order in (2, 3, 5):
        dl = 0.4 * np.cos(order * W)
        print(f"   specific direction delta = 0.4 cos({order} w) (Figure 4, interactive page): H(S_1 + t delta) - H(S_1) at t = 1: {ent(Sp + dl) - ent(Sp):+.6f} (second-order prediction {-0.25 * mean(dl ** 2 / Sp ** 2):+.6f}), at t = -1: {ent(Sp - dl) - ent(Sp):+.6f}; gamma_0, gamma_1 of S_1 + delta: {fv(cosc(Sp + dl, 1), 6)} against {fv(cosc(Sp, 1), 6)}")
    print("   the proof needs the zeroth coefficient too. Without r*_0 fixed (only r*_1..r*_p, as printed in Theorem 10.2) there is no maximum: adding white noise c to S_1 keeps r*_1 and raises H:")
    print("      c        H(S_1 + c) - H(S_1)      (1/2) log(c) for comparison")
    for cc in (0.0, 1.0, 10.0, 100.0, 1000.0):
        print(f"      {cc:<8g} {ent(Sp + cc) - ent(Sp):>10.4f}               {0.5 * math.log(cc) if cc > 0 else float('nan'):.4f}")
    print(f"   if r_1..r_p are read as normalised correlations r*_t/r*_0, scaling S -> c S keeps them and changes the entropy by exactly (1/2) log c: c = 1000: {ent(1000 * Sp) - ent(Sp):.4f} against {0.5 * math.log(1000):.4f}")
    # dual: Theorem 10.3
    a_ = A1
    Sa = S_AR1
    ra = cosc(1 / Sa, 20)
    T = 1 / Sa
    lvT = levinson(ra, 6)
    sinfT = math.exp(mean(np.log(T)))
    print(f"   dual: S = AR(1), a = {a_}; the dual stochastic realisation S_q^MA is the MA(q) spectrum with the same inverse autocovariances r_0..r_q, i.e. S_q^MA = 1/(AR(q) fit of 1/S) = |A'|^2/sigma'^2 with (A', sigma'^2) the Yule-Walker fit of the sequence r_t:")
    print("      q   D[S_q:S] (1/4pi)   (1/2) log(sigma'_q^2/sigma'_inf^2)   closed form (1/2) log((1-a^(2q+4))/(1-a^(2q+2)))   r_0..r_q of S_q minus those of S   gamma_k of S_q for k > q")
    mas = {}
    for q in range(0, 5):
        ap, s2p = lvT[q]
        Tq = ar_spec(ap, s2p)                      # AR(q) fit to T = 1/S
        Smq = 1 / Tq                               # MA(q) spectrum
        mas[q] = Smq
        mt = np.max(np.abs(cosc(1 / Smq, q) - ra[: q + 1]))
        tail = np.max(np.abs(cosc(Smq, q + 8)[q + 1:]))
        print(f"      {q}   {kl_rate(Smq, Sa):.6f}          {0.5 * math.log(s2p / sinfT):.6f}                            {0.5 * math.log((1 - a_ ** (2 * q + 4)) / (1 - a_ ** (2 * q + 2))):.6f}                                          {mt:.1e}                            {tail:.1e}")
    q = 1
    Smq = mas[q]
    S0 = np.ones(M)
    ap1, s2p1 = lvT[1]
    print(f"   the MA(1) dual realisation of S = AR(1)(a = {a_}) is S_1^MA = s' |1 + b' e^-iw|^2 with b' = {ap1[0]:.4f}, s' = 1/sigma'^2 = {1 / s2p1:.4f}: its inverse autocovariances r_0, r_1 = {fv(cosc(1 / Smq, 1), 4)} are those of S = {fv(ra[:2], 4)}; (it cannot match gamma_1/gamma_0 = {-a_:.2f}: an MA(1) has at most 1/2)")
    print(f"   Pythagoras (10.61), D[S_0:S] = D[S_0:S_q] + D[S_q:S] for q = 1: {kl_rate(S0, Sa):.6f} = {kl_rate(S0, Smq):.6f} + {kl_rate(Smq, Sa):.6f} (violation {abs(kl_rate(S0, Sa) - kl_rate(S0, Smq) - kl_rate(Smq, Sa)):.1e})")
    th_q = 1 / Smq
    worst, ident = 1.0, 0.0
    for delta in random_cosine_perturbations(rng, q + 1, 4000):
        delta = delta * np.minimum(1.0, 0.6 * th_q.min() / np.maximum(np.abs(delta).max(axis=1, keepdims=True), 1e-12))
        Sx = 1 / (th_q + delta)
        d = 0.5 * np.mean(np.log(Sx), axis=1) - 0.5 * mean(np.log(Smq))                               # H(S') - H(S_1^MA)
        klr = 0.5 * np.mean(Smq / Sx - 1 - np.log(Smq / Sx), axis=1)                                   # D[S_1^MA:S'] (true rate)
        worst = min(worst, float(d.min()))
        ident = max(ident, float(np.abs(d - klr).max()))
    print(f"   Theorem 10.3 (minimum entropy): 4000 random perturbations of 1/S_1 inside E_1(r) (added cosines of order 2..7, so r_0, r_1 unchanged): the smallest H(S') - H(S_1^MA) is {worst:.2e} (all positive);")
    print(f"      exact identity H(S') - H(S_1^MA) = D[S_1^MA:S'] (true Kullback-Leibler rate): largest violation {ident:.1e}")
    print("   without r_0 fixed (only r_1..r_q, as printed) there is no minimum: adding c to the inverse spectrum 1/S_1 keeps r_1 and lowers H without bound:")
    print("      c        H(1/(1/S_1 + c)) - H(S_1)     -(1/2) log(c) for comparison")
    for cc in (0.0, 1.0, 10.0, 100.0, 1000.0):
        print(f"      {cc:<8g} {ent(1 / (th_q + cc)) - ent(Smq):>10.4f}               {-0.5 * math.log(cc) if cc > 0 else float('nan'):.4f}")
    # (10.62): factor 2
    print("   (10.62) as printed says D[S_0:S] = H_S + const. With the printed (10.37): D[S_0:S] = r_0 - 1 + (1/2pi) int log S = 2 H_S + r_0 - 1 - log(2 pi e). On the AR(1)-type family with a fixed r_0 = a_0^2 + a_1^2 = 1.36 (inverse spectrum |a_0 + a_1 e^iw|^2):")
    print("      r_1      H_S          D[S_0:S] (10.37)    D - 2 H_S     D - H_S")
    for r1 in (-0.6, -0.3, 0.0, 0.3, 0.6):
        pp = 1.36
        # |a_0 + a_1 e^{iw}|^2 = r_0 + 2 r_1 cos w with a_0^2 + a_1^2 = r_0, a_0 a_1 = r_1
        Sx = 1 / (pp + 2 * r1 * np.cos(W))
        D = kl_book(S0, Sx)
        print(f"      {r1:+.1f}     {ent(Sx):.6f}    {D:.6f}            {D - 2 * ent(Sx):.6f}      {D - ent(Sx):.6f}")
    print(f"   D - 2 H_S is the constant r_0 - 1 - log(2 pi e) = {1.36 - 1 - LOG2PIE:.6f}; D - H_S is not constant: the factor in (10.62) should be 2 (with the true Kullback-Leibler rate it would be 1 in (10.58)-(10.59) as well); the conclusion of Theorem 10.3 is unaffected")
    STORE["entropy_demo"] = (Sp, th_q)


def check_two_projections():
    head("5c. The best AR(1) approximation of an MA(1), in each of the two senses")
    b = B1M
    S = S_MA1
    g = cosc(S, 4)
    am = -g[1] / g[0]
    sm = g[0] * (1 - (g[1] / g[0]) ** 2)
    Sm = ar_spec(np.array([am]), sm)
    # e-projection: minimise D[S':S] over (a', s'): profile s' = 1/mean(1/(S|1 + a e|^2)); criterion J(a) = mean(1/(S |1 + a e|^2))
    J = lambda a: mean(1 / (S * np.abs(1 + a * EJ) ** 2))
    lo, hi = -0.99, 0.0
    for _ in range(200):                              # golden-section search on a convex-looking 1-d criterion
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        lo, hi = (lo, m2) if J(m1) < J(m2) else (m1, hi)
    ae = 0.5 * (lo + hi)
    se = 1 / J(ae)
    Se = ar_spec(np.array([ae]), se)
    cub = np.roots([b * b, b, -1, -b])
    cub = cub[np.abs(cub.imag) < 1e-12].real
    root = cub[np.argmin(np.abs(cub - ae))]
    print(f"   S = MA(1), b = {b}. m-projection (minimises D[S:S'] over AR(1), the Yule-Walker model, a = -gamma_1/gamma_0 = -b/(1 + b^2)): a = {am:.6f}, s = {sm:.6f}")
    print(f"   e-projection (minimises D[S':S] over AR(1)): golden-section search gives a = {ae:.6f}, s = {se:.6f}; it is the root of b^2 a^3 + b a^2 - a - b = 0 in (-1, 0): {root:.6f}")
    print(f"      D[S:S_m] = {kl_rate(S, Sm):.6f} <= D[S:S_e] = {kl_rate(S, Se):.6f};   D[S_e:S] = {kl_rate(Se, S):.6f} <= D[S_m:S] = {kl_rate(Sm, S):.6f}: each projection wins in its own divergence, and they are different models")
    S0 = np.ones(M)
    print(f"   Pythagoras for the m-projection (AR(1) is e-flat): D[S:S_0] = {kl_rate(S, S0):.6f} = D[S:S_m] + D[S_m:S_0] = {kl_rate(S, Sm) + kl_rate(Sm, S0):.6f};")
    print(f"      the dual relation for the e-projection fails (AR(1) is not m-flat): D[S_0:S] = {kl_rate(S0, S):.6f}, D[S_0:S_e] + D[S_e:S] = {kl_rate(S0, Se) + kl_rate(Se, S):.6f}, difference {kl_rate(S0, S) - kl_rate(S0, Se) - kl_rate(Se, S):+.6f}")
    # brute force for e-projection onto AR(2)
    f = lambda u: math.log(mean(1 / (S[::4] * np.abs(pol(np.concatenate([[1.0], pacf_to_ar(u)]), E4)) ** 2)))
    u, v = nelder_mead(f, np.zeros(2) + 0.05)
    a2 = pacf_to_ar(u)
    s2e = 1 / mean(1 / (S * np.abs(pol(np.concatenate([[1.0], a2]))) ** 2))
    print(f"   e-projection onto AR(2) (Nelder-Mead): a = {fv(a2, 5)}, D[S_e:S] = {kl_rate(ar_spec(a2, s2e), S):.6f}; m-projection onto AR(2): a = {fv(levinson(g, 2)[2][0], 5)}, D[S:S_m] = {kl_rate(S, ar_spec(*levinson(g, 2)[2])):.6f}")
    STORE["two_proj"] = (am, sm, ae, se)


# ------------------------------------------------------------------ 3d. the alpha-structure (Theorem 10.1)

def d_alpha(S1, S2, al):
    """The alpha-divergence (10.45) of the book: (1/(2 pi alpha^2)) int ((S2/S1)^alpha - 1 - alpha log(S2/S1)) dw, and (1/4 pi) int (log S2 - log S1)^2 dw at alpha = 0."""
    if abs(al) < 1e-12:
        return 0.5 * mean((np.log(S2) - np.log(S1)) ** 2)
    r = S2 / S1
    return mean(r ** al - 1 - al * np.log(r)) / al ** 2


def check_alpha():
    head("3d. Theorem 10.1 and the alpha-structure of L")
    th0 = 1.7                                                       # one frequency: the Gaussian scale family N(0, S), theta = 1/S
    psi = lambda t: -0.5 * np.log(t)
    h = 1e-2
    g = (psi(th0 + h) - 2 * psi(th0) + psi(th0 - h)) / h ** 2
    T = (psi(th0 + 2 * h) - 2 * psi(th0 + h) + 2 * psi(th0 - h) - psi(th0 - 2 * h)) / (2 * h ** 3)
    print(f"   at one frequency, theta = {th0}: g = psi'' = {g:.5f} (1/(2 theta^2) = {1 / (2 * th0 ** 2):.5f}), T = psi''' = {T:.5f} (-1/theta^3 = {-1 / th0 ** 3:.5f});")
    print("   the alpha-connection is Gamma^(alpha)_(theta theta)^theta = ((1 - alpha)/2) T/g, and an alpha-affine coordinate rho must satisfy rho''/rho' = Gamma:")
    print("      alpha    Gamma^(alpha)    rho''/rho' for rho = -(1/alpha) S^(-alpha) = -(1/alpha) theta^alpha, the representation (10.46)    (alpha - 1)/theta")
    hh = 1e-3
    for al in (-1.0, -0.5, 0.0, 0.5, 1.0):
        rho = (lambda t: -(t ** al) / al) if al != 0 else (lambda t: np.log(t))
        r1 = (rho(th0 + hh) - rho(th0 - hh)) / (2 * hh)
        r2 = (rho(th0 + hh) - 2 * rho(th0) + rho(th0 - hh)) / hh ** 2
        print(f"      {al:+.1f}      {(1 - al) / 2 * T / g:+.5f}         {r2 / r1:+.5f}                                                                                     {(al - 1) / th0:+.5f}")
    print("   so S^(-alpha) (log S at alpha = 0) is alpha-affine at each frequency, and so are its Fourier coefficients: L is flat for every alpha because it is a product of one-dimensional families (like the positive measures, not like a probability simplex)")
    S1, S2 = S_ARMA, spec([1, -0.3])
    print("   the alpha-divergence (10.45) between S_1 = ARMA(1,1)(-0.5, 0.3) and S_2 = AR(1)(a = -0.3):")
    print(f"      alpha = -1: {d_alpha(S1, S2, -1.0):.6f} = (10.37) KL[S_1:S_2] = {kl_book(S1, S2):.6f};   alpha = +1: {d_alpha(S1, S2, 1.0):.6f} = KL[S_2:S_1] = {kl_book(S2, S1):.6f};   alpha = 0: {d_alpha(S1, S2, 0.0):.6f}; alpha = +-0.01: {d_alpha(S1, S2, 0.01):.6f}, {d_alpha(S1, S2, -0.01):.6f} (continuous at 0)")
    print(f"      duality D^(alpha)(S_1||S_2) = D^(-alpha)(S_2||S_1): largest difference over alpha = 0.3, 0.7, 1: {max(abs(d_alpha(S1, S2, al) - d_alpha(S2, S1, -al)) for al in (0.3, 0.7, 1.0)):.1e}")
    hfun = np.cos(2 * W) + 0.5 * np.sin(W)
    eps = 0.02
    Sb = S1 * np.exp(eps * hfun)
    quad = 0.5 * eps ** 2 * mean(hfun ** 2)
    print(f"   small displacement S_2 = S_1 exp(eps h), eps = {eps}: D^(alpha) / [(1/4pi) int (eps h)^2] = " + ", ".join(f"{d_alpha(S1, Sb, al) / quad:.4f} (alpha = {al:+.1f})" for al in (-1.0, -0.5, 0.0, 0.5, 1.0)) + ": every alpha gives the same metric to second order")
    print(f"      that metric is (1/2pi) int (d log S)^2, 1/pi = {1 / math.pi:.4f} times the metric (10.36), (1/2) int (d log S)^2 (D = (1/2) g_D): the normalisations of (10.45) and of (10.34)-(10.36) differ by a factor pi")


# ------------------------------------------------------------------ 6. ARMA(1,1): the singular line; the boundary

def li2(x, K=200000):
    k = np.arange(1, K + 1, dtype=float)
    return float(np.sum(x ** k / k ** 2))


def check_arma11():
    head("6a. ARMA(1,1) (10.63)-(10.64): the singular line a = b, the Fisher determinant and the two sheets (Figure 10.3)")
    a0 = AB[0]
    print(f"   per-observation Fisher information in (a, b): G = [[1/(1-a^2), -1/(1-ab)], [-1/(1-ab), 1/(1-b^2)]], det G = (a - b)^2 / ((1-a^2)(1-b^2)(1-ab)^2); at a = {a0} and b = a + eps:")
    print("      eps      b        det G (closed form)   det G (quadrature)   smallest eigenvalue   det G / eps^2   exact n = 64: det(I/n)")
    rows = []
    for eps in (0.8, 0.4, 0.1, 0.02, 0.0):
        b = a0 + eps
        G = arma_G(a0, b)
        Gs = fisher_spec(lambda x, y: spec([1, x], [1, y]), [a0, b])
        cf = (a0 - b) ** 2 / ((1 - a0 ** 2) * (1 - b ** 2) * (1 - a0 * b) ** 2)
        ex = fisher_exact(gamma_fn("arma"), [a0, b], 64) / 64
        rows.append((eps, b, cf, float(np.linalg.eigvalsh(G)[0])))
        print(f"      {eps:<7g} {b:+.3f}    {cf:.8f}            {abs(np.linalg.det(Gs)):.8f}           {np.linalg.eigvalsh(G)[0]:.6f}              {cf / eps ** 2 if eps else float('nan'):.5f}        {np.linalg.det(ex):.8f}")
    STORE["arma_rows"] = rows
    print(f"   on the diagonal a = b the model is H = 1 whatever a is: d log S/da + d log S/db = 0 there, so G (1, 1)^T = 0: kernel direction (1, 1); det G vanishes like (a - b)^2, with det G/eps^2 -> 1/(1 - a^2)^4 = {1 / (1 - a0 ** 2) ** 4:.4f}")
    print("   two sheets: the sign of h_1 = b - a tells the triangles apart; the cepstrum (Fourier coefficients of log S = sum_k 2 (-1)^(k+1) (b^k - a^k) cos(k w)/k) gives Euclidean coordinates for (10.36):")
    print(f"      c_1 = 2 (b - a), c_2 = -(b^2 - a^2), Jacobian d(c_1, c_2)/d(a, b) = 4 (b - a): at (a, b) = {AB}: c = ({2 * (AB[1] - AB[0]):.4f}, {-(AB[1] ** 2 - AB[0] ** 2):.4f}), Jacobian {4 * (AB[1] - AB[0]):.4f}; the whole diagonal goes to the origin (white noise)")
    cep = cosc(np.log(S_ARMA), 3)
    print(f"      numerical cepstrum of S (twice the cosine coefficients of log S): c_1 = {2 * cep[1]:.4f}, c_2 = {2 * cep[2]:.4f}")
    d2 = math.pi * mean(np.log(S_ARMA) ** 2)
    a_, b_ = AB
    print(f"   distance from white noise in the metric (10.36): d^2 = (1/2) int (log S)^2 dw = {d2:.6f} = 2 pi [Li_2(a^2) + Li_2(b^2) - 2 Li_2(ab)] = {2 * math.pi * (li2(a_ ** 2) + li2(b_ ** 2) - 2 * li2(a_ * b_)):.6f}; for b = a + eps, d ~ eps sqrt(2 pi/(1 - a^2)):")
    for eps in (0.1, 0.01):
        bb = a0 + eps
        d2e = math.pi * mean(np.log(spec([1, a0], [1, bb])) ** 2)
        print(f"      eps = {eps}: d = {math.sqrt(d2e):.6f}, eps sqrt(2 pi/(1 - a^2)) = {eps * math.sqrt(2 * math.pi / (1 - a0 ** 2)):.6f}")
    # the same cancellation in ARMA(2,2): a common factor (1 + beta w) in numerator and denominator
    al_, be_, ga_ = -0.3, 0.5, 0.4
    A22 = np.polymul([1, be_], [1, al_]); B22 = np.polymul([1, be_], [1, ga_])
    G22 = fisher_spec(lambda a1, a2, b1, b2: spec([1, a1, a2], [1, b1, b2]), [A22[1], A22[2], B22[1], B22[2]])
    ev, vec = np.linalg.eigh(G22)
    nv = np.array([1.0, al_, 1.0, ga_]); nv /= np.linalg.norm(nv)
    print(f"   the same happens for any common factor: ARMA(2,2) with A = (1 + {be_} w)(1 {al_:+.1f} w) = {np.round(A22, 3).tolist()}, B = (1 + {be_} w)(1 + {ga_} w) = {np.round(B22, 3).tolist()} (reduced model (1 + {ga_} w)/(1 {al_:+.1f} w)):")
    print(f"      eigenvalues of the 4 x 4 Fisher information in (a_1, a_2, b_1, b_2): {fv(ev, 6)}; the zero eigenvector is {fv(vec[:, 0] * np.sign(vec[0, 0]), 4)}, the direction that moves the common factor in numerator and denominator together, (1, {al_}, 1, {ga_}) normalised = {fv(nv, 4)}")
    avals = (-0.8, -0.4, 0.0, 0.4, 0.8)
    Gr = np.array([[math.sqrt((1 - x * x) * (1 - y * y)) / (1 - x * y) for y in avals] for x in avals])
    print(f"   the directions in which S leaves white noise when b -> a are f_a = d log S/db = 2 Re[e^-iw/(1 + a e^-iw)]; their normalised inner products are sqrt((1-a^2)(1-a'^2))/(1-a a'). For a = {avals} the Gram matrix has smallest eigenvalue {np.linalg.eigvalsh(Gr)[0]:.4f} > 0:")
    print(f"      five directions are linearly independent (the angle between those of a = -0.5 and a = 0.5 is {math.degrees(math.acos(0.75 / 1.25)):.2f} degrees), so near white noise the image is a cone over a curve, not a smooth surface: a genuine singular point, as Figure 10.3 draws it")


def check_boundary():
    head("6b. The boundary of the stable region: |a| -> 1 and |b| -> 1")
    Mb = 1 << 16
    Eb = np.exp(-1j * 2 * np.pi * np.arange(Mb) / Mb)
    mb = lambda f: float(np.mean(f))
    print("   AR(1), x_t = phi x_(t-1) + e_t (a = -phi): the Fisher information per observation 1/(1 - phi^2) diverges at the unit root, but the distance does not: the arc length from 0 is arcsin(phi) -> pi/2 (the whole interval has length pi)")
    print("      phi       g = 1/(1-phi^2)   arc length asin(phi)   KL[white:AR] (1/4pi)   closed form phi^2/2    KL[AR:white]     closed form phi^2/(2(1-phi^2))")
    for phi in (0.5, 0.9, 0.99, 0.999):
        Sx = 1 / np.abs(1 - phi * Eb) ** 2
        f1 = 0.5 * mb(1 / Sx - 1 + np.log(Sx)); f2 = 0.5 * mb(Sx - 1 - np.log(Sx))
        print(f"      {phi:<8g}  {1 / (1 - phi ** 2):<16.4f}  {math.asin(phi):<21.4f}  {f1:<21.5f}  {phi ** 2 / 2:<20.5f}  {f2:<15.4f}  {phi ** 2 / (2 * (1 - phi ** 2)):.4f}")
    print("   the unit-root model is at finite distance, and the divergence from white noise to it is finite (tending to 1/2), but the divergence from it to the stable models is infinite")
    print("   MA(1), b -> 1: the spectrum S = |1 + b e^-iw|^2 acquires a zero at w = pi, the inverse spectrum stops being integrable, the e-coordinates r_t = (-b)^t/(1 - b^2) blow up while the m-coordinates stay finite:")
    print("      b         r_0 = (1/2pi) int 1/S    1/(1 - b^2)    gamma_0 = 1 + b^2    (1/2) int (log S)^2 dw    2 pi Li_2(b^2)")
    for b in (0.5, 0.9, 0.99, 0.999):
        Sx = np.abs(1 + b * Eb) ** 2
        print(f"      {b:<8g}  {mb(1 / Sx):<22.4f}  {1 / (1 - b ** 2):<13.4f}  {mb(Sx):<19.4f}  {math.pi * mb(np.log(Sx) ** 2):<24.4f}  {2 * math.pi * li2(b * b):.4f}")
    print(f"   at b = 1 exactly: log S = log(2 + 2 cos w) is square integrable ((1/2) int (log S)^2 = pi^3/3 = {math.pi ** 3 / 3:.4f}), so (10.13) holds, but 1/S is not integrable: the e-chart (r_t) does not exist there although the point is a legitimate spectrum")


# ------------------------------------------------------------------ 6c. a finite Markov chain (the last remark of the chapter)

def check_markov():
    head("6c. The last remark: a Markov chain observed on 0 <= t <= T is a curved exponential family, with e-curvature of order 1/T")
    print("   two-state chain x_0..x_T with P(0->1) = alpha, P(1->0) = beta. Every such chain (any initial law, any transition matrix) lies in the 4-dimensional exponential family")
    print("   with statistics (x_0, N_01, N_10, N_11) (N_ij = number of transitions i -> j, N_00 = T - the rest); enumerating all 2^(T+1) sequences gives the Fisher metric exactly.")

    def stats_all(T):
        X = ((np.arange(2 ** (T + 1))[:, None] >> np.arange(T, -1, -1)) & 1).astype(np.int8)
        a, b = X[:, :-1], X[:, 1:]
        Nij = lambda i, j: ((a == i) & (b == j)).sum(axis=1).astype(float)
        return np.stack([X[:, 0].astype(float), Nij(0, 1), Nij(1, 0), Nij(1, 1)], axis=1), Nij(0, 0)

    def curvature(T, xi, theta_fun, logp_fun, h=1e-4):
        Sx, N00 = stats_all(T)
        p = np.exp(logp_fun(Sx, N00, *xi))
        mu = p @ Sx
        G = (Sx - mu).T @ ((Sx - mu) * p[:, None])
        xi = np.array(xi, float); k = len(xi)
        J = np.zeros((4, k)); H = np.zeros((4, k, k))
        for i in range(k):
            e = np.zeros(k); e[i] = h
            J[:, i] = (theta_fun(*(xi + e)) - theta_fun(*(xi - e))) / (2 * h)
        for i in range(k):
            for j in range(k):
                ei = np.zeros(k); ei[i] = h; ej = np.zeros(k); ej[j] = h
                H[:, i, j] = ((theta_fun(*(xi + ei)) - 2 * theta_fun(*xi) + theta_fun(*(xi - ei))) / h ** 2 if i == j
                              else (theta_fun(*(xi + ei + ej)) - theta_fun(*(xi + ei - ej)) - theta_fun(*(xi - ei + ej)) + theta_fun(*(xi - ei - ej))) / (4 * h * h))
        gi = np.linalg.inv(J.T @ G @ J)
        N = np.zeros((4, k, k))
        for i in range(k):
            for j in range(k):
                N[:, i, j] = H[:, i, j] - J @ gi @ (J.T @ G @ H[:, i, j])
        return sum(gi[a, c] * gi[b, d] * (N[:, a, b] @ G @ N[:, c, d]) for a in range(k) for b in range(k) for c in range(k) for d in range(k))

    def logp_stat(Sx, N00, al, be):                       # stationary start: pi = (beta, alpha)/(alpha + beta)
        pi1 = al / (al + be)
        return np.where(Sx[:, 0] == 1, math.log(pi1), math.log(1 - pi1)) + N00 * math.log(1 - al) + Sx[:, 1] * math.log(al) + Sx[:, 2] * math.log(be) + Sx[:, 3] * math.log(1 - be)

    def theta_stat(al, be):
        pi1 = al / (al + be)
        return np.array([math.log(pi1 / (1 - pi1)), math.log(al / (1 - al)), math.log(be / (1 - al)), math.log((1 - be) / (1 - al))])

    def logp_free(Sx, N00, pi1, al, be):                   # free initial distribution
        return np.where(Sx[:, 0] == 1, math.log(pi1), math.log(1 - pi1)) + N00 * math.log(1 - al) + Sx[:, 1] * math.log(al) + Sx[:, 2] * math.log(be) + Sx[:, 3] * math.log(1 - be)

    def theta_free(pi1, al, be):
        return np.array([math.log(pi1 / (1 - pi1)), math.log(al / (1 - al)), math.log(be / (1 - al)), math.log((1 - be) / (1 - al))])

    print("   squared norm |H^e|^2 of the e-embedding curvature in that family, at (alpha, beta) = (0.3, 0.4) (stationary start, 2 parameters) and at (pi_1, alpha, beta) = (0.43, 0.3, 0.4) (free initial law, 3 parameters):")
    print("      T      stationary: |H^e|^2   T^2 |H^e|^2      free initial law: |H^e|^2   T^2 |H^e|^2")
    rows = []
    for T in (4, 6, 8, 10, 12, 14, 16, 18):
        c2 = curvature(T, [0.3, 0.4], theta_stat, logp_stat)
        c3 = curvature(T, [0.43, 0.3, 0.4], theta_free, logp_free)
        rows.append((T, c2, c3))
        print(f"      {T:<6d} {c2:<20.6e} {T * T * c2:<15.4f} {c3:<24.6e} {T * T * c3:.4f}")
    print("   T^2 |H^e|^2 rises and levels off (its increments shrink steadily), so |H^e|^2 ~ 1/T^2 and the curvature itself falls like 1/T: the book's statement for Markov chains holds in this example, as it did for AR(1)")
    STORE["markov"] = rows


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100} .s0{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100} .f0{fill:#8a8880}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .thin{stroke-width:1;fill:none}
  .dash{stroke-width:1.4;fill:none;stroke-dasharray:5 3}
  .con{stroke-width:0.9;fill:none;opacity:.75}
  .fillS{fill:#2a78d6;opacity:.08}
  .ring{stroke:#fdfdfc;stroke-width:1.5}
  .fillB{fill:#2a78d6;opacity:.85} .fillG{fill:#1baf7a;opacity:.85} .fillO{fill:#eb6834;opacity:.85}
  .sk{stroke:#1a1a19} .fk{fill:#1a1a19}
  .dot{stroke-width:1.2;fill:none;stroke-dasharray:1.5 3.5;stroke-linecap:round}
  .halo{paint-order:stroke;stroke:#fdfdfc;stroke-width:3.5px;stroke-linejoin:round}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .fillS{fill:#3987e5}
    .ring{stroke:#161615}
    .fillB{fill:#3987e5} .fillG{fill:#199e70} .fillO{fill:#d95926}
    .sk{stroke:#eceae3} .fk{fill:#eceae3}
    .halo{stroke:#161615}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fmt(v):
    return f"{v:g}".replace("-", "−")


class Panel:
    def __init__(self, body, x0, y0, w, h, xr, yr):
        self.b, self.x0, self.y0, self.w, self.h, self.xr, self.yr = body, x0, y0, w, h, xr, yr

    def X(self, x):
        return self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, y):
        return self.y0 + self.h - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def frame(self, xt, yt, xlab, ylab, title, grid=True, xtl=None):
        b = self.b
        for y in yt:
            if grid: b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{fmt(y)}</text>')
        for x in xt:
            if grid: b.append(f'<line class="gd" x1="{self.X(x):.1f}" y1="{self.y0}" x2="{self.X(x):.1f}" y2="{self.y0 + self.h}"/>')
            b.append(f'<text class="sm" x="{self.X(x):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{xtl[x] if xtl else fmt(x)}</text>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0 + self.h}" x2="{self.x0 + self.w}" y2="{self.y0 + self.h}"/>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" y2="{self.y0 + self.h}"/>')
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlab}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{title}</text>')

    def line(self, xs, ys, cls):
        keep = [(x, y) for x, y in zip(xs, ys) if self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9]
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in keep)
        self.b.append(f'<polyline class="{cls}" points="{pts}"/>')

    def dot(self, x, y, cls, r=4.5):
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{s}</text>')


import re


def T(s):
    """Tiny text markup for the figures: _x or _{xy} is a subscript, ^x or ^{xy} a superscript."""
    parts = re.split(r"([_^])(?:\{([^}]*)\}|(.))", s)
    out, cur, i = [], 0.0, 0
    # re.split with three groups yields: text, kind, braced, single, text, ...
    while i < len(parts):
        txt = parts[i]
        if txt:
            if cur != 0.0:
                out.append(f'<tspan dy="{-cur:g}" xml:space="preserve">{txt}</tspan>'); cur = 0.0      # preserve: a space right after a sub/superscript is otherwise dropped
            else:
                out.append(txt)
        if i + 3 < len(parts):
            kind, body = parts[i + 1], parts[i + 2] or parts[i + 3]
            t = 3.5 if kind == "_" else -4.5
            out.append(f'<tspan dy="{t - cur:g}" font-size="8.5">{body}</tspan>'); cur = t
        i += 4
    return "".join(out)


def note(b, x, y, lines, cls="sm", lh=15):
    for k, ln in enumerate(lines):
        b.append(f'<text class="{cls}" x="{x}" y="{y + k * lh}">{T(ln)}</text>')


def arrow(b, x1, y1, x2, y2, k, sw=2.0, head=7.0, ink=False):
    """An arrow in pixel coordinates; k in '0'..'4' picks the colour class, ink=True the text colour."""
    dx, dy = x2 - x1, y2 - y1; L = math.hypot(dx, dy)
    if L < 1e-9:
        return
    ux, uy = dx / L, dy / L; hx, hy = x2 - ux * head, y2 - uy * head
    sc, fc = ("sk", "fk") if ink else (f"s{k}", f"f{k}")
    b.append(f'<line class="{sc}" style="stroke-width:{sw};stroke-linecap:round" x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}"/>')
    b.append(f'<polygon class="{fc}" points="{x2:.1f},{y2:.1f} {hx - uy * head * 0.42:.1f},{hy + ux * head * 0.42:.1f} {hx + uy * head * 0.42:.1f},{hy - ux * head * 0.42:.1f}"/>')


def poly(b, pts, cls, style=""):
    st = f' style="{style}"' if style else ""
    b.append(f'<polyline class="{cls}"{st} points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')


def legend(b, x, y, entries, col=170, lh=17):
    """entries: (class, text); two columns."""
    for k, (cls, txt) in enumerate(entries):
        cx, cy = x + (k % 2) * col, y + (k // 2) * lh
        b.append(f'<line class="{cls}" style="stroke-width:2.2" x1="{cx}" y1="{cy}" x2="{cx + 22}" y2="{cy}"/>')
        b.append(f'<text class="sm" x="{cx + 28}" y="{cy + 3.5}">{T(txt)}</text>')


def legend_col(b, x, y, entries, lh=16):
    """Single-column legend: entries are (class, text); the sample is a short line."""
    for k, (cls, txt) in enumerate(entries):
        yy = y + k * lh
        b.append(f'<line class="{cls}" style="stroke-width:2.2" x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}"/>')
        b.append(f'<text class="sm" x="{x + 28}" y="{yy + 3.5}">{T(txt)}</text>')


def ptext(P, x, y, s, cls="sm", anchor="start", dx=0, dy=0):
    """Text at data coordinates with the sub/superscript markup of T()."""
    P.b.append(f'<text class="{cls}" x="{P.X(x) + dx:.1f}" y="{P.Y(y) + dy:.1f}" text-anchor="{anchor}">{T(s)}</text>')


def shade(P, pts, cls="fillS"):
    P.b.append(f'<polygon class="{cls}" points="' + " ".join(f"{P.X(x):.1f},{P.Y(y):.1f}" for x, y in pts) + '"/>')


def fig_charts(path):
    """AR(1) in the e-chart and MA(1) in the m-chart, with the two kinds of geodesic."""
    b = []
    W_, H_ = 800, 450
    # AR(1), free gain: coordinates (r_0, r_1) = ((1 + a^2)/s, a/s)
    Pa = Panel(b, 70, 56, 300, 270, (0, 2.3), (-1.15, 1.15))
    Pa.frame([0, 0.5, 1, 1.5, 2], [-1, -0.5, 0, 0.5, 1], T("r_0"), T("r_1"), "AR(1), free gain: the e-chart is a flat cone", True)
    shade(Pa, [(0, 0), (2.3, 1.15), (2.3, -1.15)])
    Pa.line([0, 2.3], [0, 1.15], "dash s0"); Pa.line([0, 2.3], [0, -1.15], "dash s0")
    ptext(Pa, 0.9, -1.04, "unit root, |a| = 1", "sm", "start")
    bs = np.linspace(-0.95, 0.95, 120)
    Pa.line(1 + bs ** 2, bs, "dot s0")
    sP, sQ = spec([1, -0.6]), spec([1, 0.5], [1], 2.0)
    rP, rQ = cosc(1 / sP, 2), cosc(1 / sQ, 2)
    ts = np.linspace(0, 1, 60)
    Pa.line([(1 - t) * rP[0] + t * rQ[0] for t in ts], [(1 - t) * rP[1] + t * rQ[1] for t in ts], "ln s1")
    mm = [cosc(1 / ((1 - t) * sP + t * sQ), 2) for t in ts]
    Pa.line([m_[0] for m_ in mm], [m_[1] for m_ in mm], "ln s2")
    Pa.dot(rP[0], rP[1], "f0", 5); Pa.dot(rQ[0], rQ[1], "f0", 5); Pa.dot(1, 0, "f4", 5)
    ptext(Pa, rP[0], rP[1], "P", "v", "start", 8, 4); ptext(Pa, rQ[0], rQ[1], "Q", "v", "start", 8, -2); ptext(Pa, 1, 0, "white noise", "sm halo", "start", 7, 14)
    Pa.dot(0.5 * (rP[0] + rQ[0]), 0.5 * (rP[1] + rQ[1]), "f1", 4.5)
    mid = cosc(1 / (0.5 * sP + 0.5 * sQ), 3)
    Pa.dot(mid[0], mid[1], "f2", 4.5)
    # MA(1), free gain: coordinates (gamma_0, gamma_1) = (s (1 + b^2), s b)
    Pb = Panel(b, 450, 56, 300, 270, (0, 3.2), (-1.65, 1.65))
    Pb.frame([0, 1, 2, 3], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], T("γ_0"), T("γ_1"), "MA(1), free gain: the m-chart is a flat cone", True)
    shade(Pb, [(0, 0), (3.2, 1.6), (3.2, -1.6)])
    Pb.line([0, 3.2], [0, 1.6], "dash s0"); Pb.line([0, 3.2], [0, -1.6], "dash s0")
    ptext(Pb, 1.3, -1.55, "S(π) = 0, b = 1", "sm", "start")
    Pb.line(1 + bs ** 2, bs, "dot s0")
    mP, mQ = spec([1], [1, 0.5]), spec([1], [1, -0.6], 2.0)
    gP, gQ = cosc(mP, 2), cosc(mQ, 2)
    Pb.line([(1 - t) * gP[0] + t * gQ[0] for t in ts], [(1 - t) * gP[1] + t * gQ[1] for t in ts], "ln s1")
    ee = [cosc(1 / ((1 - t) / mP + t / mQ), 2) for t in ts]
    Pb.line([e_[0] for e_ in ee], [e_[1] for e_ in ee], "ln s2")
    Pb.dot(gP[0], gP[1], "f0", 5); Pb.dot(gQ[0], gQ[1], "f0", 5); Pb.dot(1, 0, "f4", 5)
    ptext(Pb, gP[0], gP[1], "P'", "v", "start", 8, -4); ptext(Pb, gQ[0], gQ[1], "Q'", "v", "start", 8, 4); ptext(Pb, 1, 0, "white noise", "sm halo", "start", 7, 14)
    Pb.dot(0.5 * (gP[0] + gQ[0]), 0.5 * (gP[1] + gQ[1]), "f1", 4.5)
    emid = cosc(1 / (0.5 / mP + 0.5 / mQ), 3)
    Pb.dot(emid[0], emid[1], "f2", 4.5)
    legend_col(b, 70, 372, [("s1", "straight in this chart: stays in the family"), ("s2", "the other geodesic: bends, leaves the family"), ("s0", "dotted: gain pinned to 1 (a_0 = 1 or b_0 = 1)")], 17)
    note(b, 450, 372, ["At t = 1/2 the m-geodesic of the AR(1) pair has", f"r = ({mid[0]:.3f}, {mid[1]:.3f}, {mid[2]:.3f}, ...): r_2 is not 0, so it left AR(1).", "The e-geodesic of the MA(1) pair has", f"γ = ({emid[0]:.3f}, {emid[1]:.3f}, {emid[2]:.3f}, ...): γ_2 is not 0, so it left MA(1)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W_, H_, "AR is e-flat, MA is m-flat",
        "Left: the AR(1) models with free innovation variance fill the cone r_0 > 2|r_1| of the plane of the first two inverse autocovariances; the e-geodesic between two of them is a straight line inside the cone, while the m-geodesic, the straight mixture of the spectra, is a curve that has non-zero r_2 and so leaves AR(1). Right: the MA(1) models fill the cone gamma_0 > 2|gamma_1| of the plane of the first two autocovariances; the m-geodesic is straight and stays in MA(1) while the e-geodesic bends and leaves it. The dotted parabola is the gain-pinned one-parameter family in each chart.", b))


def fig_normalisations(path):
    """Left: KL per observation of n observations against the two candidate limits; right: finite-length embedding curvatures."""
    b = []
    W_, H_ = 800, 440
    rows = STORE["kl_rows"]; Db = STORE["Db"]
    P1 = Panel(b, 75, 56, 310, 270, (2.7, 9.3), (0.12, 0.30))
    P1.frame([3, 4, 5, 6, 7, 8, 9], [0.14, 0.18, 0.22, 0.26, 0.30], "n (observations)", "per observation", "Kullback-Leibler divergence", True, {k: str(2 ** k) for k in range(3, 10)})
    P1.line([2.7, 9.3], [Db / 2, Db / 2], "dash s0"); P1.line([2.7, 9.3], [Db, Db], "dash s2")
    xs = [math.log2(n) for n, _ in rows]
    P1.line(xs, [k for _, k in rows], "ln s1"); P1.line(xs, [2 * k for _, k in rows], "ln s2")
    for x_, (_, k) in zip(xs, rows):
        P1.dot(x_, k, "f1", 3.8); P1.dot(x_, 2 * k, "f2", 3.8)
    ptext(P1, 4.6, Db / 2, f"limit (1/4π) ∫(…)dω = {Db / 2:.4f}", "sm", "start", 0, -8)
    ptext(P1, 3.0, Db, f"(10.37) as printed, (1/2π) ∫(…)dω = {Db:.4f}", "sm", "start", 0, -8)
    ptext(P1, 9.25, 0.128, "exact KL of n observations, divided by n", "sm", "end", 0, 0)
    ptext(P1, 3.0, 0.256, "twice that", "sm", "start", 0, 0)
    rows_c = STORE["curv_rows"]
    P2 = Panel(b, 470, 56, 300, 270, (3.0, 7.7), (0, 11))
    P2.frame([math.log2(n) for n in (10, 20, 40, 80, 160)], [0, 2, 4, 6, 8, 10], "n (observations)", "scaled squared curvature", "Embedding curvature at finite length", True, {math.log2(n): str(n) for n in (10, 20, 40, 80, 160)})
    ns = [math.log2(r[0]) for r in rows_c]
    P2.line(ns, [r[0] ** 2 * r[1] for r in rows_c], "ln s1")
    P2.line(ns, [r[0] * r[2] for r in rows_c], "ln s3")
    P2.line(ns, [r[0] * r[3] for r in rows_c], "ln s2")
    for r in rows_c:
        P2.dot(math.log2(r[0]), r[0] ** 2 * r[1], "f1", 3.8); P2.dot(math.log2(r[0]), r[0] * r[2], "f3", 3.8); P2.dot(math.log2(r[0]), r[0] * r[3], "f2", 3.8)
    P2.line([3.0, 7.7], [4 / 0.64, 4 / 0.64], "dash s3"); P2.line([3.0, 7.7], [4 / 0.75, 4 / 0.75], "dash s2")
    ptext(P2, 7.65, 10.3, "AR(1): n² |H^e|²", "sm", "end")
    ptext(P2, 7.65, 6.25, "AR(1): n |H^m|², limit 6.25", "sm", "end", 0, -6)
    ptext(P2, 7.65, 5.33, "MA(1): n |H^e|², limit 5.33", "sm", "end", 0, 14)
    ptext(P2, 7.65, 0.35, "MA(1): |H^m|² = 0 for every n", "sm", "end")
    note(b, 75, 390, ["Left: the printed (10.37) is twice the Kullback-Leibler divergence per observation of a Gaussian series", "(exact Toeplitz covariances); the error of KL/n times n is a constant.", "Right: AR(1) is e-flat only up to boundary terms: its squared e-curvature falls like 1/n² (the curve rises", "to a constant); the m-curvature of AR(1) and the e-curvature of MA(1) fall like 1/n."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W_, H_, "Normalisation of (10.37) and finite-length curvature",
        "Left: the Kullback-Leibler divergence per observation between two exact Gaussian time series of n observations converges to one half of the printed divergence (10.37), not to (10.37) itself; the doubled values converge to (10.37). Right: for exact Toeplitz covariances, the squared e-embedding curvature of AR(1) times n squared tends to a constant, the m-curvature of AR(1) times n tends to 4/(1 - a squared), the e-curvature of MA(1) times n tends to 4/(1 - b squared), and the m-curvature of MA(1) is zero at every n.", b))


def fig_projections(path):
    b = []
    W_, H_ = 800, 450
    S = S_MA1
    am, sm, ae, se = STORE["two_proj"]
    g = cosc(S, 4)
    lv = levinson(g, 3)
    P1 = Panel(b, 75, 56, 310, 270, (0, math.pi), (0, 3.4))
    P1.frame([0, math.pi / 2, math.pi], [0, 1, 2, 3], "ω", "S(ω)", "MA(1), b = 0.5, and AR approximations", True, {0: "0", math.pi / 2: "π/2", math.pi: "π"})
    w = np.linspace(0, math.pi, 200)
    ev = lambda f: [float(f(x)) for x in w]
    Sf = lambda x: abs(1 + B1M * np.exp(-1j * x)) ** 2
    ARf = lambda a, s: (lambda x: s / abs(1 + sum(a[k] * np.exp(-1j * (k + 1) * x) for k in range(len(a)))) ** 2)
    P1.line(w, ev(Sf), "ln sk")
    P1.line(w, ev(ARf(lv[1][0], lv[1][1])), "ln s1")
    P1.line(w, ev(ARf(lv[2][0], lv[2][1])), "ln s3")
    P1.line(w, ev(ARf(np.array([ae]), se)), "dash s2")
    legend_col(b, 130, 82, [("sk", "S = MA(1), b = 0.5"), ("s1", "m-projection, AR(1)"), ("s3", "m-projection, AR(2)"), ("s2", "e-projection, AR(1)")], 16)
    pr = STORE["pyth_p"]
    P2 = Panel(b, 470, 56, 300, 270, (-0.6, 5.6), (0, 0.5))
    P2.frame([0, 1, 2, 3, 4, 5], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "p (order of the AR model)", "D (1/4π)", "Pythagoras (10.57), S = ARMA(1,1)", True)
    for p_, d_sp, d_p0 in pr:
        for val, cls, y0 in ((d_p0, "fillB", 0.0), (d_sp, "fillO", d_p0)):
            b.append(f'<rect class="{cls}" x="{P2.X(p_) - 14:.1f}" y="{P2.Y(y0 + val):.1f}" width="28" height="{max(P2.Y(y0) - P2.Y(y0 + val), 0.5):.1f}"/>')
    b.append(f'<rect class="fillB" x="474" y="388" width="12" height="9"/><text class="sm" x="492" y="396">{T("D[S_p : S_0]")}: the model against white noise</text>')
    b.append(f'<rect class="fillO" x="474" y="405" width="12" height="9"/><text class="sm" x="492" y="413">{T("D[S : S_p]")}: what the model misses (the m-projection)</text>')
    note(b, 75, 388, ["Left: Yule-Walker matches γ_0..γ_p exactly; the", "e-projection is another AR(1): a = −0.428, not −0.400.", "Right: every bar has the same height D[S : S_0];", "the m-projection moves it from 'misses' to 'model'."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W_, H_, "Projections onto AR(p) and Pythagoras",
        "Left: the spectrum of the MA(1) process with b = 0.5 and the spectra of its m-projections onto AR(1) and AR(2) (Yule-Walker) and of its e-projection onto AR(1), which is a different AR(1) model. Right: stacked bars for p = 0 to 5 showing the Pythagorean relation of (10.57) for an ARMA(1,1) spectrum: the sum of the divergence from the process to its AR(p) projection and from the projection to white noise is the same for every p.", b))


def fig_entropy(path):
    b = []
    W_, H_ = 800, 440
    Sp, th_q = STORE["entropy_demo"]
    # S_1 = Sp is the AR(1) m-projection of MA(1) b = 0.5
    P1 = Panel(b, 75, 56, 310, 270, (-1, 1), (-0.04, 0.005))
    P1.frame([-1, -0.5, 0, 0.5, 1], [-0.04, -0.03, -0.02, -0.01, 0], "t", T("H(S_1 + t δ) − H(S_1)"), T("Entropy in M_1(r) is maximal at the AR(1) model"), True)
    ts = np.linspace(-1, 1, 81)
    for k, (order, amp, cls) in enumerate(((2, 0.4, "ln s1"), (3, 0.4, "ln s2"), (5, 0.4, "ln s3"))):
        delta = amp * np.cos(order * W)
        P1.line(ts, [ent(Sp + t * delta) - ent(Sp) for t in ts], cls)
    legend_col(b, 150, 262, [("s1", "δ = 0.4 cos 2ω"), ("s2", "δ = 0.4 cos 3ω"), ("s3", "δ = 0.4 cos 5ω")], 16)
    P2 = Panel(b, 470, 56, 300, 270, (-2, 3), (-4, 4))
    P2.frame([-2, -1, 0, 1, 2, 3], [-4, -2, 0, 2, 4], "log₁₀ c", "entropy change", T("No maximum, no minimum without r*_0 or r_0"), True)
    lc = np.linspace(-2, 3, 80)
    P2.line(lc, [ent(Sp + 10 ** x) - ent(Sp) for x in lc], "ln s1")
    P2.line(lc, [ent(1 / (th_q + 10 ** x)) - ent(1 / th_q) for x in lc], "ln s2")
    P2.line(lc, [0.5 * math.log(10 ** x) for x in lc], "dash s0"); P2.line(lc, [-0.5 * math.log(10 ** x) for x in lc], "dash s0")
    ptext(P2, -1.9, 3.1, "H(S_1 + c): r*_1 kept, r*_0 raised", "sm", "start"); ptext(P2, -1.9, -3.5, "H(1/(1/S_1^{MA} + c)): r_1 kept, r_0 raised", "sm", "start")
    note(b, 75, 390, ["Left: moving inside M_1(r) (autocovariances γ_0, γ_1 held) can only lower the entropy: the first-order term vanishes by orthogonality.", "Right: if γ_0 (or r_0) is not held, entropy is unbounded above (below); the dashed lines are ±(1/2) log c."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W_, H_, "Maximum entropy needs the zeroth coefficient",
        "Left: the entropy rate along three directions inside the set of spectra with the same first two autocovariances as an AR(1) model is largest at the AR(1) model, which is Theorem 10.2. Right: if the zeroth autocovariance is not held fixed, adding white noise c to the spectrum raises the entropy without bound, like one half of the logarithm of c; dually, adding c to the inverse spectrum lowers it without bound, so the printed Theorems 10.2 and 10.3, which list only r_1 to r_p, need r_0 as well.", b))


def fig_arma(path):
    b = []
    W_, H_ = 800, 515
    P1 = Panel(b, 70, 56, 310, 310, (-1, 1), (-1, 1))
    P1.frame([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], "a", "b", T("ARMA(1,1): log₁₀ det G (pale = little information)"), False)
    N_ = 32
    xe = [round(P1.X(-1 + 2 * i / N_), 1) for i in range(N_ + 1)]                      # shared cell edges: neighbouring cells touch without overlapping
    ye = [round(P1.Y(-1 + 2 * j / N_), 1) for j in range(N_ + 1)]
    for i in range(N_):
        for j in range(N_):
            a_ = -1 + (i + 0.5) * 2 / N_; b_ = -1 + (j + 0.5) * 2 / N_
            dg = (a_ - b_) ** 2 / ((1 - a_ ** 2) * (1 - b_ ** 2) * (1 - a_ * b_) ** 2)
            f = min(max((math.log10(max(dg, 1e-12)) + 3) / 5, 0), 1)
            b.append(f'<rect class="fillS" style="opacity:{0.06 + 0.8 * f:.2f};shape-rendering:crispEdges" x="{xe[i]}" y="{ye[j + 1]}" width="{xe[i + 1] - xe[i]:.1f}" height="{ye[j] - ye[j + 1]:.1f}"/>')
    P1.line([-1, 1], [-1, 1], "dash s2")
    ptext(P1, -0.92, 0.82, "b &gt; a: h_1 &gt; 0", "v halo", "start"); ptext(P1, 0.1, -0.85, "b &lt; a: h_1 &lt; 0", "v halo", "start")
    ptext(P1, 0.35, 0.62, "a = b: H = 1", "sm halo", "start")
    P1.dot(AB[0], AB[1], "f4", 5); ptext(P1, AB[0], AB[1], "(−0.5, 0.3)", "sm halo", "start", 8, 4)
    P2 = Panel(b, 450, 56, 310, 310, (-4, 4), (-1.1, 1.1))
    P2.frame([-4, -2, 0, 2, 4], [-1, -0.5, 0, 0.5, 1], T("c_1 = 2(b − a)"), T("c_2 = a² − b²"), "Cepstral image: all curves meet at white noise", True)
    vals = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)
    tt = np.linspace(-0.98, 0.98, 90)
    for v in vals:
        P2.line([2 * (x - v) for x in tt], [v * v - x * x for x in tt], "thin s1")      # a = v fixed, b = x
        P2.line([2 * (v - x) for x in tt], [x * x - v * v for x in tt], "thin s3")      # b = v fixed, a = x
    P2.dot(0, 0, "f2", 5.5); ptext(P2, 0, 0, "white noise", "sm halo", "start", 8, 18)
    P2.dot(2 * (AB[1] - AB[0]), AB[0] ** 2 - AB[1] ** 2, "f4", 5)
    legend_col(b, 470, 440, [("s1", "a fixed, b varies"), ("s3", "b fixed, a varies")], 16)
    note(b, 70, 432, ["Left: the Fisher determinant vanishes on the", "diagonal a = b (pole-zero cancellation).", "Right: the map (a, b) → (c_1, c_2) has Jacobian 4(b − a):", "it folds the two triangles onto two half-planes and", "sends the whole diagonal to one point."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W_, H_, "The singular line of ARMA(1,1)",
        "Left: a heat map over the square of stable parameters (a, b) of the logarithm of the determinant of the Fisher information of the ARMA(1,1) model; it vanishes quadratically on the diagonal a = b, where numerator and denominator of the transfer function cancel. The two triangles b greater than a and b less than a are separated by the sign of the first impulse-response coefficient. Right: the image of the square in the first two cepstral coordinates (c_1, c_2) = (2(b - a), a squared minus b squared): lines of constant a and of constant b all pass through the origin, white noise, which is the singular point where the two sheets meet.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_charts(out / "charts.svg")
    fig_normalisations(out / "normalisations.svg")
    fig_projections(out / "projections.svg")
    fig_entropy(out / "entropy.svg")
    fig_arma(out / "arma-singularity.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_system()
    check_periodogram()
    check_models()
    check_likelihood()
    check_potentials()
    check_normalisations()
    check_alpha()
    check_flatness()
    check_curvature()
    check_projections()
    check_entropy()
    check_two_projections()
    check_arma11()
    check_boundary()
    check_markov()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
