#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 7, checked by hand.

Asymptotic theory of statistical inference. Every number quoted in the notes comes from here.

The running example is a curved exponential family: S = {N(u, u^2)}, a Gaussian whose standard deviation equals its mean,
inside the two-parameter Gaussian exponential family M with sufficient statistics (x, x^2). An observed sample is summarised by
the observed point eta_bar = (xbar, mean of x^2) in the eta-plane; S is the parabola eta_2 = 2 eta_1^2. Everything about an
estimator of u is then a statement about a family of curves ("leaves") in that plane, and all the estimators used here have closed
forms, which makes exact simulation cheap: the sufficient statistics of N Gaussian observations are sampled directly
(xbar normal, the sum of squares chi-square), no samples are generated.

Convention as in the book: lower indices differentiate in theta, upper in eta; the second-order terms are the squares of the
e-embedding curvature of S, the m-embedding curvature of the ancillary leaves and the m-connection coefficient of the coordinate u.

Checked here, in the order the notes use them:

  1. estimation and the Cramer-Rao bound (7.1): the exponential distribution exactly, the coefficient 4 in (7.10), and Hodges' estimator
     as the counterexample to Theorem 7.1 read pointwise;
  2. exponential families (7.2): the MLE is the observed point; Theorem 7.3 exactly on the binomial; the asymptotic bias of the MLE in
     any chart is -(1/2N) g^{bc} Gamma^{(m)a}_{bc}, checked on the Gaussian against exact bias in three charts;
  3. the curved family N(u, u^2): theta(u), eta(u), the Fisher metric, the m-connection and the e-curvature of the curve (Efron's
     statistical curvature); the MLE as the m-projection (KL minimisation) and its leaves, which are straight lines in eta;
  4. first-order theory (7.4): five estimators as five families of leaves; consistency, the Schur complement (7.58), efficiency as
     orthogonality; the angle between a leaf and S; Monte Carlo at N = 4000;
  5. the bias formula (7.54) for four leaf families, against the delta method and Monte Carlo;
  6. second-order theory (7.5): the three terms of (7.65) for the MLE, for an estimator with curved leaves and for the MLE of log u;
     Monte Carlo of the bias-corrected estimators; Theorem 7.6;
  7. hypothesis testing (7.6): local power of a test is the efficiency of its leaf; Rao, Wald, likelihood-ratio and two inefficient tests;
  8. a non-regular model: the uniform location family and the midrange.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Monte Carlo uses a fixed seed; the whole script takes about twenty seconds.

Run:  python3 asymptotics.py            (checks)
      python3 asymptotics.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20261002)
CHUNK = 1_000_000
STORE = {}                         # numbers computed by the checks, reused by the figures


def head(s):
    print("\n" + s)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


Z05 = 1.6448536269514722          # upper 5% point of the standard normal


# ------------------------------------------------------------------ the Gaussian exponential family, statistics (x, x^2)

def gauss_G(mu, s):
    """Fisher information in theta = Cov[(x, x^2)] for N(mu, s^2)."""
    return np.array([[s * s, 2 * mu * s * s], [2 * mu * s * s, 2 * s * s * (s * s + 2 * mu * mu)]])


def gauss_theta(mu, s):
    return np.array([mu / s ** 2, -1 / (2 * s ** 2)])


def gauss_eta(mu, s):
    return np.array([mu, mu * mu + s * s])


def psi_third(th):
    """psi_{ijk}: the cubic tensor T of the Gaussian family in theta."""
    t1, t2 = th
    P = np.zeros((2, 2, 2))
    P[0, 0, 1] = P[0, 1, 0] = P[1, 0, 0] = 1 / (2 * t2 ** 2)
    P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = -t1 / t2 ** 3
    P[1, 1, 1] = 3 * t1 ** 2 / (2 * t2 ** 4) - 1 / t2 ** 3
    return P


# ------------------------------------------------------------------ the curved family S = N(u, u^2) and curves in M

class Curve:
    """A curve in M with its first two derivatives at one parameter value, in theta and in eta coordinates."""

    def __init__(self, th, thd, thdd, et, etd, etdd):
        self.th, self.thd, self.thdd, self.et, self.etd, self.etdd = th, thd, thdd, et, etd, etdd


def curve_S(u):
    """theta(u) = (1/u, -1/(2u^2)), eta(u) = (u, 2u^2) and derivatives."""
    return Curve(np.array([1 / u, -1 / (2 * u * u)]), np.array([-1 / u ** 2, 1 / u ** 3]), np.array([2 / u ** 3, -3 / u ** 4]),
                 np.array([u, 2 * u * u]), np.array([1.0, 4 * u]), np.array([0.0, 4.0]))


def reparam(c, kd, kdd):
    """New parameter t with u = k(t): derivatives by the chain rule, kd = k'(t), kdd = k''(t)."""
    return Curve(c.th, c.thd * kd, c.thdd * kd ** 2 + c.thd * kdd, c.et, c.etd * kd, c.etdd * kd ** 2 + c.etd * kdd)


def S_G(u):
    return gauss_G(u, u)


def second_order_terms(c, G, etv, etvv):
    """The three terms of (7.65) for a one-dimensional S in a two-dimensional M with a one-dimensional leaf.
    c: S as a Curve; G: Fisher information in theta at the point; etv, etvv: velocity and m-acceleration of the leaf (eta chart)."""
    g = c.thd @ G @ c.thd                                         # g_uu
    gam = (c.etdd @ c.thd) / g                                    # Gamma^{(m) u}_{uu}: tangential part of the m-acceleration
    n2 = c.thdd @ G @ c.thdd - (c.thd @ G @ c.thdd) ** 2 / g      # |normal part of theta''|^2 in the Fisher metric
    hm = (etvv @ c.thd) / g                                       # H^{(m) u}_{vv}
    gvv = etv @ np.linalg.inv(G) @ etv
    out = dict(g=g, gam=gam, n2=n2, gamma2=n2 / g ** 2, Gm2=gam ** 2 / g ** 2, He2=n2 / g ** 3, Hm2=hm ** 2 / gvv ** 2)
    out["coef"] = 0.5 * (out["Gm2"] + 2 * out["He2"] + out["Hm2"])
    return out


# ------------------------------------------------------------------ five estimators of u, as functions of the observed point (a, b) = (xbar, mean x^2)

EST = {
    "MLE": lambda a, b: (np.sqrt(a * a + 4 * b) - a) / 2,                    # root of u^2 + a u - b = 0
    "e-orth": lambda a, b: (a + np.sqrt(8 * b - 7 * a * a)) / 4,             # leaves are e-straight lines orthogonal to S
    "xbar": lambda a, b: a,
    "m2": lambda a, b: np.sqrt(b / 2),
    "1.1 xbar": lambda a, b: 1.1 * a,
}


def leaf_chart(kind, u):
    """The leaf through eta(u) in the chart w = (u, v), eta(u, v), with derivatives at v = 0: (eta_u, eta_v, eta_uu, eta_uv, eta_vv)."""
    z = np.zeros(2)
    if kind == "MLE":              # straight lines eta_2 = u eta_1 + u^2
        return np.array([1, 4 * u]), np.array([1, u]), np.array([0, 4.0]), np.array([0, 1.0]), z
    if kind == "e-orth":           # parabolas eta_2 = eta_1^2 - u eta_1 + 2 u^2
        return np.array([1, 4 * u]), np.array([1, u]), np.array([0, 4.0]), np.array([0, 1.0]), np.array([0, 2.0])
    if kind == "xbar":             # vertical lines eta_1 = u
        return np.array([1, 4 * u]), np.array([0, 1.0]), np.array([0, 4.0]), z, z
    if kind == "m2":               # horizontal lines eta_2 = 2 u^2
        return np.array([1, 4 * u]), np.array([1, 0.0]), np.array([0, 4.0]), z, z
    raise ValueError(kind)


def ab_samples(N, R, mu, s, rng=None):
    """(xbar, mean of x^2) for R samples of size N from N(mu, s^2): the exact joint law, xbar normal and the sum of squares chi-square."""
    rng = rng or RNG
    a = rng.normal(mu, s / math.sqrt(N), R)
    chi = rng.chisquare(N - 1, R)
    return a, a * a + s * s * chi / N


def mc_stats(N, R, mu, s, fn, rng=None):
    """Monte Carlo means and standard errors of the arrays returned by fn(a, b) (a dict name -> array), in chunks."""
    acc, n = {}, 0
    left = R
    while left > 0:
        r = min(CHUNK, left); left -= r
        a, b = ab_samples(N, r, mu, s, rng)
        for k, v in fn(a, b).items():
            sx, sxx = acc.get(k, (0.0, 0.0))
            acc[k] = (sx + float(v.sum()), sxx + float((v * v).sum()))
        n += r
    out = {}
    for k, (sx, sxx) in acc.items():
        m = sx / n
        out[k] = (m, math.sqrt(max(sxx / n - m * m, 0.0) / n))
    return out


# ------------------------------------------------------------------ 1. estimation and the Cramer-Rao bound

def hodges_mse(N, u):
    """Exact mean squared error of Hodges' estimator of the mean of N(u, 1): xbar if |xbar| > N^(-1/4), else 0."""
    c = N ** -0.25; sq = math.sqrt(N)
    a, b = -sq * (c + u), sq * (c - u)
    p = Phi(b) - Phi(a)
    inner = p - (b * phi(b) - a * phi(a))                   # int_a^b z^2 phi(z) dz
    return 1 / N - inner / N + u * u * p


def check_estimation():
    head("1. Estimation and the Cramer-Rao bound (section 7.1, Theorems 7.1)")
    print("   Exponential distribution with rate lam: eta = 1/lam is the mean, theta = -lam; xbar ~ Gamma(N, rate N lam).")
    print("   For the mean eta the MLE is xbar: unbiased, Var = eta^2/N exactly = G^-1/N with G = 1/eta^2 (Theorem 7.2).")
    print("   For the rate (a different chart) the MLE is lam_hat = 1/xbar, with exact E = N lam/(N-1) and Var = N^2 lam^2/((N-1)^2 (N-2)):")
    print("      N      N*bias/lam    Var/(lam^2/N)    unbiased (N-1)/(N xbar): Var/(lam^2/N)    N (Var/(lam^2/N) - 1)")
    for N in (5, 10, 20, 50, 200, 1000):
        r = N ** 3 / ((N - 1) ** 2 * (N - 2)); ru = N / (N - 2)
        print(f"      {N:<6d} {N / (N - 1):<13.4f} {r:<16.4f} {ru:<42.4f} {N * (r - 1):.4f}")
    print("   (7.10) holds: V = lam^2/N + O(1/N^2), and the O(1/N^2) coefficient is 4 lam^2 (the last column tends to 4), which depends on the chart; no estimator of lam attains the bound exactly,")
    print("   even the unbiased one, whose variance exceeds it by N/(N-2); only the estimator of the expectation parameter does")
    print("   Hodges' estimator of the mean of N(u, 1), threshold N^(-1/4) (asymptotically unbiased and consistent at every fixed u), G = 1:")
    for N in (100, 10_000, 1_000_000):
        us = np.linspace(0, 3 * N ** -0.25, 3001)
        vals = np.array([N * hodges_mse(N, u) for u in us])
        k = int(vals.argmax())
        print(f"      N = {N:>9d}: N*MSE at u = 0 is {N * hodges_mse(N, 0.0):.5f} (the Cramer-Rao bound says 1); the worst case over u is {vals[k]:.3f} at u = {us[k]:.4f} = {us[k] * N ** 0.25:.2f} N^(-1/4)")
    print("   so an estimator can beat G^-1/N at a point while the bias tends to 0 everywhere: Theorem 7.1 needs uniformity (or exact unbiasedness), not only lim b = 0 (7.4)")


# ------------------------------------------------------------------ 2. exponential families

def check_exp_family():
    head("2. Estimation in an exponential family (section 7.2): Theorems 7.2 and 7.3, and the bias in other charts")
    mu, s, N = 1.0, 2.0, 10
    G = gauss_G(mu, s); eta = gauss_eta(mu, s)
    res = mc_stats(N, 4_000_000, mu, s, lambda a, b: {"a": a, "b": b, "aa": a * a, "ab": a * b, "bb": b * b})
    m1, m2 = res["a"][0], res["b"][0]
    cov = np.array([[res["aa"][0] - m1 * m1, res["ab"][0] - m1 * m2], [res["ab"][0] - m1 * m2, res["bb"][0] - m2 * m2]]) * N
    print(f"   Gaussian N(1, 2^2), N = {N}: eta_hat = (xbar, mean x^2) has mean {m1:.4f}, {m2:.4f} (eta = {eta[0]:.0f}, {eta[1]:.0f}) and N Cov = {np.round(cov, 3).tolist()}; G = {G.tolist()} (exact for every N, Monte Carlo 4e6 samples)")
    # Theorem 7.3 exactly on the binomial: x in {0,1}, p = 0.3, N = 10, error e = sqrt(N)(xbar - p)
    p, n = 0.3, 10
    pmf = np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])
    e = math.sqrt(n) * (np.arange(n + 1) / n - p)
    g, T = p * (1 - p), p * (1 - p) * (1 - 2 * p)
    print(f"   Theorem 7.3 on the binomial (p = 0.3, N = 10, exact enumeration): E[e] = {pmf @ e:.1e}, E[e^2] = {pmf @ e ** 2:.6f} = g = {g:.6f}, E[e^3] = {pmf @ e ** 3:.6f} = T/sqrt(N) = {T / math.sqrt(n):.6f}")
    print(f"      the third moment is the skewness of the observed point: it falls like 1/sqrt(N) (skewness {T / g ** 1.5 / math.sqrt(n):.4f} here)")
    # the bias in other charts: -(1/2N) g^{bc} Gamma^{(m) a}_{bc}
    Gi = np.linalg.inv(G); th = gauss_theta(mu, s)
    P = psi_third(th)
    b_theta = -0.5 * np.einsum("ai,ibg,bg->a", Gi, P, Gi)
    Gam = np.zeros((2, 2, 2)); Gam[0, 0, 1] = 1 / s; Gam[1, 1, 1] = 1 / s            # Gamma^{(m)}_{ij}^k in (mu, sigma), Chapter 5
    b_ms = -0.5 * np.einsum("bga,bg->a", Gam, np.diag([s * s, s * s / 2]))
    print(f"   Asymptotic bias of the MLE in a chart w is -(1/2N) g^(bc) Gamma^(m)a_(bc), the contraction of the m-connection of that chart (7.54, 7.53):")
    print(f"      eta chart (Gamma^m = 0): 0, exactly unbiased;  (mu, sigma) chart with Gamma^m_(mu mu)^sigma = Gamma^m_(sigma sigma)^sigma = 1/sigma (Chapter 5): N*bias = {np.round(b_ms, 4).tolist()} = (0, -3 sigma/4);")
    print(f"      theta chart with Gamma^m_(bc)^a = g^(ai) T_(ibc): N*bias = {np.round(b_theta, 4).tolist()}")
    print("      exact finite-N values at (1, 2) (s^2 = mean square deviation, N s^2/sigma^2 ~ chi^2_(N-1)):")
    print("      N      N(E sigma_hat - sigma)   N(E theta1_hat - theta1)  N(E theta2_hat - theta2)")
    for N_ in (10, 40, 160, 1000):
        ex = s * math.sqrt(2 / N_) * math.exp(math.lgamma(N_ / 2) - math.lgamma((N_ - 1) / 2))
        t1 = mu / s ** 2 * (N_ / (N_ - 3) - 1); t2 = -1 / (2 * s ** 2) * (N_ / (N_ - 3) - 1)
        print(f"      {N_:<6d} {N_ * (ex - s):<24.4f} {N_ * t1:<25.4f} {N_ * t2:.4f}")
    print("   all three converge to the formula: the bias is not a tensor, it carries the non-tensorial m-connection of the chart (Chapter 5)")


# ------------------------------------------------------------------ 3. the curved family

def kl_gauss(m, s2, u):
    """KL[N(m, s2) || N(u, u^2)]."""
    return 0.5 * np.log(u * u / s2) + (s2 + (m - u) ** 2) / (2 * u * u) - 0.5


def check_curved():
    head("3. The curved exponential family N(u, u^2) inside the Gaussian family (section 7.3)")
    u = 1.0; c = curve_S(u); G = S_G(u)
    print(f"   at u = 1: theta = {c.th.tolist()}, eta = {c.et.tolist()}, G = {G.tolist()}, theta' = {c.thd.tolist()}, eta' = {c.etd.tolist()} = G theta' = {(G @ c.thd).tolist()}")
    t = second_order_terms(c, G, *leaf_chart("MLE", u)[1::3][:1], leaf_chart("MLE", u)[4])
    print(f"   Fisher information g_uu = theta'.eta' = {t['g']:.4f} = 3/u^2; the m-connection of S in u: Gamma^(m)u_(uu) = eta''.theta'/g = {t['gam']:.4f} = 4/(3u);")
    print(f"   e-embedding curvature: |normal part of theta''|^2 = {t['n2']:.4f} = 2/(3u^4); Efron's statistical curvature gamma^2 = |.|^2/g^2 = {t['gamma2']:.4f} = 2/27 = {2 / 27:.4f}")
    # the MLE is the m-projection: minimise KL[p_etabar || p_u] over u
    a, b = ab_samples(20, 1500, 1.0, 1.0)
    s2 = b - a * a
    ugrid = np.linspace(0.05, 4.0, 40001)
    best = np.empty(len(a)); kmin = np.empty(len(a))
    for i in range(len(a)):
        kl = kl_gauss(a[i], s2[i], ugrid)
        j = int(kl.argmin()); best[i] = ugrid[j]; kmin[i] = kl[j]
    mle = EST["MLE"](a, b)
    print(f"   MLE = m-projection: for 1500 observed points (N = 20, true u = 1) the minimiser over u of KL[p_etabar || p_u] on a 1e-4 grid agrees with the closed form (sqrt(a^2 + 4b) - a)/2 to {np.abs(best - mle).max():.1e};"
          f" KL at the closed form is never above the grid minimum (largest difference {np.max(kl_gauss(a, s2, mle) - kmin):.1e}, i.e. <= 0)")
    thd_hat = np.stack([-1 / mle ** 2, 1 / mle ** 3], axis=1)
    orth = np.sum((np.stack([a - mle, b - 2 * mle ** 2], axis=1)) * thd_hat, axis=1)
    print(f"      the likelihood equation is the orthogonality (eta_bar - eta(u_hat)).theta'(u_hat) = 0: largest residual {np.abs(orth).max():.1e}; it reads eta_2 = u eta_1 + u^2:"
          f" the leaf of the MLE through eta(u) is the straight line of slope u, so the leaves are m-geodesics (the m-projection is along straight lines in eta)")
    print(f"   foliation (7.24)-(7.25): two leaves u_1, u_2 meet at eta_1 = -(u_1 + u_2) < 0, so for eta_1 > 0 the MLE leaves do not cross; their envelope is eta_2 = -eta_1^2/4, outside the model M (eta_2 >= eta_1^2)")
    print(f"   leaves of the other estimators through eta(u): xbar vertical (eta_1 = u), sqrt(m2/2) horizontal (eta_2 = 2u^2), the e-orthogonal estimator eta_2 = eta_1^2 - u eta_1 + 2u^2 (a parabola)")


# ------------------------------------------------------------------ 4. first-order theory

def leaf_metric(kind, u):
    """g_{alpha beta} of (7.36) in the chart w = (u, v) at v = 0: B_eta^T G^-1 B_eta (rows of B_eta: d eta / d w^alpha)."""
    eu, ev, _, _, _ = leaf_chart(kind, u)
    Be = np.stack([eu, ev])
    return Be @ np.linalg.inv(S_G(u)) @ Be.T


def check_first_order():
    head("4. First-order theory (section 7.4): consistency, the Schur complement (7.58) and efficiency as orthogonality")
    u = 1.0
    print(f"   at u = 1 the estimators and the leaves A(u) through eta(u) = (1, 2) (tangent direction in the eta chart):")
    print(f"      {'estimator':10s} {'leaf':36s} g_uu   g_uv   g_vv   gbar_uu = g_uu - g_uv^2/g_vv   1/gbar   angle(A,S)   efficiency")
    N = 4000
    def stat4(a, b):
        out = {}
        for k, f in EST.items():
            e = f(a, b) - u
            out[k] = e; out[k + "^2"] = e * e * N
        return out
    mc = mc_stats(N, 4_000_000, u, u, stat4)
    leaves = {"MLE": "straight line eta_2 = u eta_1 + u^2", "e-orth": "parabola eta_2 = eta_1^2 - u eta_1 + 2u^2", "xbar": "vertical line eta_1 = u", "m2": "horizontal line eta_2 = 2u^2"}
    res = {}
    for kind in ("MLE", "e-orth", "xbar", "m2"):
        gw = leaf_metric(kind, u)
        gbar = gw[0, 0] - gw[0, 1] ** 2 / gw[1, 1]
        cosang = gw[0, 1] / math.sqrt(gw[0, 0] * gw[1, 1])
        ang = math.degrees(math.acos(abs(cosang)))
        res[kind] = (gw, gbar)
        print(f"      {kind:10s} {leaves[kind]:36s} {gw[0, 0]:.3f}  {gw[0, 1]:+.3f}  {gw[1, 1]:.3f}   {gbar:.4f}                       {1 / gbar:.4f}   {ang:6.2f} deg   {gbar / gw[0, 0]:.4f}")
    print(f"   efficiency = gbar/g_uu = sin^2(angle between the leaf and S): 1 for the orthogonal leaves (MLE, e-orth), 8/9 for sqrt(m2/2) (angle 70.53 deg), 1/3 for xbar (35.26 deg)")
    print(f"   Monte Carlo at N = {N} (4e6 samples): N*Var(u_hat) against the asymptotic 1/gbar of (7.57):")
    for kind in ("MLE", "e-orth", "xbar", "m2"):
        m, se = mc[kind + "^2"]
        print(f"      {kind:10s} N E[(u_hat - u)^2] = {m:.4f} +- {se:.4f}   1/gbar = {1 / res[kind][1]:.4f}   (g^uu = 1/g_uu = {1 / 3:.4f} is the Cramer-Rao bound)")
    m, se = mc["1.1 xbar"]
    print(f"   inconsistent: u_hat = 1.1 xbar has E[u_hat] = {m + u:.4f} +- {se:.4f} at N = {N}, not 1: its leaf eta_1 = u/1.1 does not pass through eta(u) = (u, 2u^2) (Theorem 7.4 (1))")
    # (7.61) and the reading of g^{kappa lambda} in (7.58)
    rng = np.random.default_rng(3); worst = 1.0; cnt = 0
    for _ in range(2000):
        A_ = rng.normal(size=(3, 3)); Gm = A_ @ A_.T + 0.1 * np.eye(3)
        S_ = Gm[:1, :1] - Gm[:1, 1:] @ np.linalg.inv(Gm[1:, 1:]) @ Gm[1:, :1]
        worst = min(worst, float(np.linalg.eigvalsh(Gm[:1, :1] - S_)[0])); cnt += 1
    print(f"   (7.61): over {cnt} random positive-definite 3x3 metrics split 1 + 2, g_ab - gbar_ab = g_(a kappa) g_(b lambda) g^(kappa lambda) has smallest eigenvalue {worst:.2e} >= 0, and it is 0 exactly when g_(a kappa) = 0 (orthogonal leaves)")
    gw = res["xbar"][0]; full = np.linalg.inv(gw)
    print(f"   notation: for xbar, g = {np.round(gw, 3).tolist()}; (7.58) with g^(kappa lambda) the inverse of the kappa-lambda block gives gbar = {gw[0, 0] - gw[0, 1] ** 2 / gw[1, 1]:.4f} (the right marginal precision, = 1/Var(sqrt(N) xbar));"
          f" read as the kappa-lambda entry of the full inverse g^(alpha beta) of (7.55) it would give {gw[0, 0] - gw[0, 1] ** 2 * full[1, 1]:.4f}, a negative 'precision'")


# ------------------------------------------------------------------ 5. the bias

def bias_formula(kind, u):
    """N times the asymptotic bias of u_hat from (7.54): -(1/2) C_{bc}^u g^{bc}, C_{bc}^a = B^{a i} d^2 eta_i / dw^b dw^c, g^{bc} the full inverse of g_{ab}."""
    eu, ev, euu, euv, evv = leaf_chart(kind, u)
    Be = np.stack([eu, ev]); J = np.linalg.inv(Be.T)                 # J[alpha, i] = d w^alpha / d eta_i
    E2 = np.array([[euu, euv], [euv, evv]])                           # E2[b, c, i]
    C = np.einsum("ai,bci->abc", J, E2)
    ginv = np.linalg.inv(leaf_metric(kind, u))
    return -0.5 * np.einsum("bc,bc->", C[0], ginv), -0.5 * C[0, 0, 0] * ginv[0, 0], -0.5 * C[0, 1, 1] * ginv[1, 1]


def delta_bias(f, u, h=1e-4):
    """N times the second-order bias of f(a, b) from the delta method: (1/2) sum f_ij G_ij at eta(u)."""
    a0, b0 = u, 2 * u * u
    fa = (f(a0 + h, b0) - 2 * f(a0, b0) + f(a0 - h, b0)) / h ** 2
    fb = (f(a0, b0 + h) - 2 * f(a0, b0) + f(a0, b0 - h)) / h ** 2
    fab = (f(a0 + h, b0 + h) - f(a0 + h, b0 - h) - f(a0 - h, b0 + h) + f(a0 - h, b0 - h)) / (4 * h * h)
    G = S_G(u)
    return 0.5 * (fa * G[0, 0] + 2 * fab * G[0, 1] + fb * G[1, 1])


def check_bias():
    head("5. The asymptotic bias (7.54) is the m-connection contracted with the inverse metric")
    u = 1.0
    print("   N*bias of u_hat at u = 1: formula (7.54) = -(1/2) g^(bc) Gamma^(m)u_(bc) [split into the S part and the leaf part], delta method (Hessian of the estimator), Monte Carlo:")
    print(f"      {'estimator':10s} formula   (S part, leaf part)    delta method   MC N=40 (6e6)        MC N=160 (6e6)")
    mc = {}
    for N in (40, 160):
        mc[N] = mc_stats(N, 6_000_000, u, u, lambda a, b: {k: (f(a, b) - u) * 1.0 for k, f in EST.items() if k != "1.1 xbar"})
    for kind in ("MLE", "e-orth", "xbar", "m2"):
        tot, s_part, l_part = bias_formula(kind, u)
        d = delta_bias(EST[kind], u)
        m40, se40 = mc[40][kind]; m160, se160 = mc[160][kind]
        print(f"      {kind:10s} {tot:+.4f}   ({s_part:+.4f}, {l_part:+.4f})      {d:+.4f}        {40 * m40:+.4f} +- {40 * se40:.4f}   {160 * m160:+.4f} +- {160 * se160:.4f}")
    print("   exact values: -2/9 = -0.2222 (MLE), -4/9 = -0.4444 (leaves with m-curvature double the leaf part), 0 (xbar is exactly unbiased), -3/16 = -0.1875 (sqrt(m2/2));"
          " the S part is the same for every estimator, the leaf part is the m-curvature of the leaf (zero for straight leaves in eta)")


# ------------------------------------------------------------------ 6. second-order theory

def check_second_order():
    head("6. Second-order theory (section 7.5): the three terms of (7.65), bias-corrected estimators, Theorem 7.6")
    u = 1.0; c = curve_S(u); G = S_G(u)
    c_log = reparam(c, u, u)                      # t = log u:  u = e^t, k' = k'' = u
    cases = [("MLE of u", c, "MLE"), ("e-orth estimator of u", c, "e-orth"), ("MLE of log u", c_log, "MLE")]
    terms = {}
    print("   the three terms at u = 1 (units: the one-parameter case of (7.66)-(7.68); the e-curvature term is 2 (H^e)^2 and the coefficient is (1/2) times the sum):")
    print(f"      {'estimator':24s} (Gamma^m)^2   2 (H^e)^2   (H^m_A)^2   coefficient c = (1/2)(sum)     g^uu")
    for lab, cc, kind in cases:
        _, ev, _, _, evv = leaf_chart(kind, u)
        t = second_order_terms(cc, G, ev, evv)
        terms[lab] = t
        print(f"      {lab:24s} {t['Gm2']:.4f}       {2 * t['He2']:.4f}      {t['Hm2']:.4f}      {t['coef']:.4f} (= {t['coef'] * 162:.0f}/162)         {1 / t['g']:.4f}")
    t0 = terms["MLE of u"]
    print(f"   Gamma^(m)u_(uu) = {t0['gam']:.4f} in u and {terms['MLE of log u']['gam']:.4f} = 7/3 in log u: the m-connection term depends on the parametrisation; the e-curvature term transforms as a tensor"
          f" (2 (H^e)^2 = {2 * t0['He2']:.4f} in u, {2 * terms['MLE of log u']['He2']:.4f} in log u, ratio (du'/du)^2 = 1 here since u = 1); the leaf term is 0 for the straight leaves of the MLE")
    print(f"   statistical curvature gamma^2 = {t0['gamma2']:.4f} = 2/27; (H^e)^2 = gamma^2 g^uu = {t0['He2']:.4f} = 2/81")
    # Monte Carlo of bias-corrected estimators: u* = u_hat - b(u_hat)
    print("   Monte Carlo (6e6 exact samples each): N E[(u* - u)^2] for the bias-corrected estimators u* = u_hat - b1(u_hat)/N, with b1 = -2u/9 (MLE), -4u/9 (e-orth), and for log u: b1 = b_u/u - g^uu/(2u^2):")
    b1m, b1e = -2 / 9, -4 / 9
    b1ml, b1el = b1m - 0.5 / 3, b1e - 0.5 / 3
    print(f"      b1 for log u: MLE {b1ml:.4f} = -7/18 from (7.54) in the chart log u (-(1/2) g^tt Gamma^t_tt = {-0.5 * (1 / 3) * terms['MLE of log u']['gam']:.4f}); e-orth {b1el:.4f} = -11/18")
    g3 = 1 / 3
    print(f"      {'N':>4s} {'estimator':22s} {'measured':>18s}  {'prediction 1/3 + c/N':>22s}  {'fraction of the 2nd-order term':>30s}")
    for N in (20, 40, 80):
        def fn(a, b, N=N):
            m = EST["MLE"](a, b); e = EST["e-orth"](a, b)
            ms = m * (1 + 2 / (9 * N)); es = e * (1 + 4 / (9 * N))
            lm = np.log(m) - b1ml / N; le = np.log(e) - b1el / N
            return {"mle": N * (ms - u) ** 2, "eorth": N * (es - u) ** 2, "diff": N * ((es - u) ** 2 - (ms - u) ** 2), "frozen": N * (m + 2 / (9 * N) - u) ** 2,
                    "log mle": N * lm ** 2, "log eorth": N * le ** 2, "log diff": N * (le ** 2 - lm ** 2)}
        mc = mc_stats(N, 6_000_000, u, u, fn)
        STORE.setdefault("second", {})[N] = mc
        STORE["terms"] = terms
        for key, lab, coef in (("mle", "MLE of u", terms["MLE of u"]["coef"]), ("eorth", "e-orth of u", terms["e-orth estimator of u"]["coef"]), ("log mle", "MLE of log u", terms["MLE of log u"]["coef"])):
            m, se = mc[key]; pred = g3 + coef / N
            print(f"      {N:>4d} {lab:22s} {m:.5f} +- {se:.5f}   {pred:.5f}                 {(m - g3) / (coef / N):.3f}")
        d, sed = mc["diff"]; pd = (terms["e-orth estimator of u"]["coef"] - terms["MLE of u"]["coef"]) / N
        print(f"      {N:>4d} {'e-orth minus MLE (paired)':22s} {d:.5f} +- {sed:.5f}   {pd:.5f}                 {d / pd:.3f}")
        fz, sefz = mc["frozen"]
        pfz = g3 + (terms["MLE of u"]["coef"] + 2 * (-2 / 9) * g3) / N
        print(f"      {N:>4d} {'MLE, b1 frozen at u = 1':22s} {fz:.5f} +- {sefz:.5f}   {pfz:.5f}                 (subtracting the constant 2/(9N), which uses the true u, instead of the function 2 u_hat/(9N): c becomes c + 2 b1' g^uu = 10/81 - 4/27 = -2/81)")
    print("   the second-order term of the MLE is c = 10/81 = 0.1235: 'Gamma^m' and 'e-curvature' only; any other efficient estimator adds (H^m_A)^2/2 >= 0 (Theorem 7.6);"
          " here the e-orthogonal estimator adds 8/81 = 0.0988, observed in the paired difference to within 3 percent at N = 20, 40, 80, and the parametrisation log u adds 0.2037 to the MLE's coefficient")


def trigamma(x):
    """psi'(x) by its asymptotic series, accurate to 1e-12 for x >= 5."""
    return 1 / x + 1 / (2 * x ** 2) + 1 / (6 * x ** 3) - 1 / (30 * x ** 5) + 1 / (42 * x ** 7) - 1 / (30 * x ** 9)


def check_isolated_terms():
    head("6b. Isolating one term at a time: the m-affine parameter, and exactly solvable one-parameter families")
    u = 1.0; c = curve_S(u); G = S_G(u)
    # the m-affine parameter of S: Gamma^(m) = 0 for tau = u^(7/3), since d^2u/dtau^2 = -Gamma^u_uu (du/dtau)^2 with Gamma^u_uu = 4/(3u)
    kd, kdd = 3 / 7, -12 / 49                      # u = tau^(3/7) at tau = 1
    ct = reparam(c, kd, kdd); _, ev, _, _, evv = leaf_chart("MLE", u)
    t = second_order_terms(ct, G, ev, evv)
    b1 = (7 / 3) * (-2 / 9) + 0.5 * (28 / 9) / 3
    print(f"   S = N(u, u^2) in the m-affine parameter tau = u^(7/3): Gamma^(m) = {t['gam']:.1e}, so the m-connection term vanishes; b1 = h' b1_u + (1/2) h'' g^uu = (7/3)(-2/9) + (1/2)(28/9)(1/3) = {b1:.1e} (no first-order bias);"
          f" g^(tau tau) = {1 / t['g']:.4f} = 49/27, and what is left is c = (H^e)^2 = {t['coef']:.4f} = 98/729 (the MLE has straight leaves)")
    g0 = 1 / t["g"]
    print(f"      Monte Carlo of the plain MLE tau_hat = u_hat^(7/3) (no correction needed): N E[(tau_hat - 1)^2] against g^(tau tau) + c/N, and the fraction of the second-order term observed:")
    for N, R in ((10, 8_000_000), (15, 8_000_000), (20, 8_000_000)):
        mc = mc_stats(N, R, u, u, lambda a, b: {"bias": EST["MLE"](a, b) ** (7 / 3) - 1.0, "sq": (EST["MLE"](a, b) ** (7 / 3) - 1.0) ** 2})
        mb, seb = mc["bias"]; m, se = mc["sq"]; m *= N; se *= N; pred = g0 + t["coef"] / N
        print(f"      N = {N:>3d}: N*bias = {N * mb:+.4f} +- {N * seb:.4f} (prediction 0); N E[(tau-1)^2] = {m:.5f} +- {se:.5f}, prediction {pred:.5f}, fraction {(m - g0) / (t['coef'] / N):.3f} +- {se / (t['coef'] / N):.3f}")
    print("      the e-curvature term is the only second-order term for this estimator; had it only half its printed weight (coefficient 1 instead of 2 in (7.65)) the same data would sit at about twice the predicted second-order term, which they do not")
    # exactly solvable: N(0, s) with sufficient statistic x^2, n = m = 1: no leaves, no e-curvature; c = (1/2) Gamma^2 / g^2
    print("   Exactly solvable: S = M = {N(0, s)} with sufficient statistic x^2 (n = m = 1: no leaves, no e-curvature), sigma = 1, N m_2 ~ chi^2_N; only the m-connection term remains, c = (1/2) Gamma^2/g^2:")
    print(f"      {'parameter':12s} theta'  eta'   eta''   g = theta' eta'   Gamma = eta'' theta'/g   c      exact N^2 (Var - g^-1/N) of the unbiased estimator for N = 10, 100, 1000")
    for lab, thd, etd, etdd in (("s = sigma^2", 0.5, 1.0, 0.0), ("sigma", 1.0, 2.0, 2.0), ("log sigma", 1.0, 2.0, 4.0)):
        g_ = thd * etd; gam = etdd * thd / g_; cc = 0.5 * gam ** 2 / g_ ** 2
        vals = []
        for N in (10, 100, 1000):
            if lab == "s = sigma^2":
                var = 2.0 / N                                          # m_2 itself: unbiased, Var = 2 s^2/N exactly
            elif lab == "sigma":
                cN = math.sqrt(2 / N) * math.exp(math.lgamma((N + 1) / 2) - math.lgamma(N / 2))
                var = (1 / cN ** 2 - 1)                                # sigma_hat / c_N is exactly unbiased: Var = sigma^2 (1/c_N^2 - 1)
            else:
                var = 0.25 * trigamma(N / 2)                           # (1/2) log m_2 minus its bias: Var = (1/4) psi'(N/2)
            vals.append(N * (N * var - 1 / g_))
        print(f"      {lab:12s} {thd:.2f}    {etd:.1f}    {etdd:.1f}     {g_:.2f}              {gam:.4f}                  {cc:.4f}  " + ", ".join(f"{v:.4f}" for v in vals))
    print("      the exact second-order coefficients 0, 1/8, 1/2 are the m-connection terms: it depends on how the unknown is parametrised, and the book's (7.68) is right with the 1/2 of (7.65)")
    print("   and when S is e-flat and the parameter m-affine, nothing is left: S = {N(u, 1)} has theta = (u, -1/2) a straight line (e-flat), Gamma^m = 0, and the MLE xbar has Var = 1/N exactly")


# ------------------------------------------------------------------ 7. hypothesis testing

def check_tests():
    head("7. Hypothesis testing (section 7.6): H0: u = 1 against u > 1, level 0.05, local alternatives u = 1 + delta/sqrt(N)")
    u0 = 1.0; c = curve_S(u0); G = S_G(u0); g = c.thd @ G @ c.thd
    rows = {"Rao / efficient": c.thd, "xbar test": np.array([1.0, 0.0]), "m2 test": np.array([0.0, 1.0])}
    shifts = {}
    print("   a test that rejects when n.(eta_bar - eta(u0)) is large has boundary a straight line with normal n; its local power is Phi(delta * n.eta'/sqrt(n^T G n) - z), z = 1.6449:")
    for lab, n in rows.items():
        sh = (n @ c.etd) / math.sqrt(n @ G @ n); shifts[lab] = sh
        print(f"      {lab:16s} n = {np.round(n, 1).tolist()}: shift factor {sh:.4f}, efficiency (shift^2/g_uu) = {sh * sh / g:.4f}  (the same numbers as the estimators with the same leaf directions: 1, 1/3, 8/9)")
    print("   (Cauchy-Schwarz: the shift is largest, sqrt(g_uu) = 1.7321, exactly when n is proportional to theta'(u0), i.e. when the boundary is orthogonal to S: Theorem 7.7)")
    lg = lambda u, a, b: a / u - b / (2 * u * u) - 0.5 - np.log(u) - 0.5 * math.log(2 * math.pi)       # per observation log-likelihood of N(u, u^2)
    def stats(a, b, N):
        m = EST["MLE"](a, b)
        lr = 2 * N * np.maximum(lg(m, a, b) - lg(1.0, a, b), 0)
        return {"Rao": b - a - 1.0, "Wald": m, "LR": np.where(m > 1, lr, -lr), "xbar": a, "m2": b}
    print(f"      {'N':>5s} {'delta':>5s}  {'Rao':>7s} {'Wald':>7s} {'LR':>7s} | {'xbar':>7s} {'m2':>7s} | asymptotic: efficient   xbar     m2")
    for N in (50, 500, 5000):
        R = 2_000_000
        a0, b0 = ab_samples(N, R, u0, u0); st0 = stats(a0, b0, N)
        crit = {k: np.quantile(v, 0.95) for k, v in st0.items()}
        del a0, b0, st0
        for delta in (1.0, 2.0, 3.0):
            a1, b1 = ab_samples(N, R, u0 + delta / math.sqrt(N), u0 + delta / math.sqrt(N)); st1 = stats(a1, b1, N)
            pw = {k: float(np.mean(v > crit[k])) for k, v in st1.items()}
            STORE.setdefault("tests", {})[(N, delta)] = pw
            STORE["shifts"] = shifts
            asy = [Phi(delta * shifts[k] - Z05) for k in ("Rao / efficient", "xbar test", "m2 test")]
            print(f"      {N:>5d} {delta:>5.1f}  {pw['Rao']:.4f}  {pw['Wald']:.4f}  {pw['LR']:.4f} | {pw['xbar']:.4f}  {pw['m2']:.4f} |             {asy[0]:.4f}   {asy[1]:.4f}  {asy[2]:.4f}")
    print("   Rao, Wald (any monotone function of the MLE, whichever information is used) and likelihood-ratio tests have the same power to within 0.0005 at every N here:"
          " their second-order differences, which the book attributes to the m-curvature and the angle of A_N(u), are too small to see in this model;"
          " the first-order efficiency of the three and the loss for the xbar and m2 tests, 1/3 and 8/9 of the information, are as predicted")


# ------------------------------------------------------------------ 8. a non-regular model

def check_nonregular():
    head("8. A non-regular model (closing remarks): the uniform location family U[u - 1/2, u + 1/2]")
    print("   the Fisher information diverges; the midrange (max + min)/2 is an excellent estimator, with error of order 1/N rather than 1/sqrt(N):")
    print(f"      {'N':>5s} {'N^2 Var':>10s} {'exact N^2/(2(N+1)(N+2))':>25s}  {'excess kurtosis':>16s}")
    last = None
    for N in (10, 50, 500, 5000):
        R = 4_000_000
        V = RNG.random(R); W = RNG.random(R)
        mx = V ** (1 / N); mn = mx * (1 - W ** (1 / (N - 1)))
        x = N * ((mx + mn) / 2 - 0.5)
        k = float(np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3)
        print(f"      {N:>5d} {np.var(x):>10.4f} {N * N / (2 * (N + 1) * (N + 2)):>25.4f}  {k:>16.3f}")
        last = x
        STORE.setdefault("mid_kurt", {})[N] = k
        if N == 500:
            STORE["mid_hist"] = np.histogram(x, bins=np.linspace(-3, 3, 61), density=True)
    x2 = last[: len(last) // 2] + last[len(last) // 2:]
    k2 = float(np.mean(x2 ** 4) / np.mean(x2 ** 2) ** 2 - 3)
    print(f"   limit of N (midrange - u): the difference of two independent exponentials over 2, a Laplace law with density exp(-2|x|): variance 1/2 and excess kurtosis 3 (not Gaussian, not 0);"
          f" the sum of two independent copies has excess kurtosis {k2:.2f} (= 3/2 for a Laplace), so the limit is not a stable law as the book says: a stable law's sums keep the shape")




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
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .fillS{fill:#3987e5}
    .ring{stroke:#161615}
    .fillB{fill:#3987e5} .fillG{fill:#199e70} .fillO{fill:#d95926}
    .sk{stroke:#eceae3} .fk{fill:#eceae3}
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
                out.append(f'<tspan dy="{-cur:g}">{txt}</tspan>'); cur = 0.0
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


def eta_leaf(kind, u, n=90):
    """Points of the leaf A(u) in the eta-plane, restricted to the admissible region eta_2 >= eta_1^2."""
    if kind == "xbar":
        ys = np.linspace(u * u, 5.2, n); xs = np.full(n, u)
    else:
        xs = np.linspace(0.1, 2.0, 4 * n)
        ys = {"MLE": u * xs + u * u, "e-orth": xs ** 2 - u * xs + 2 * u * u, "m2": np.full_like(xs, 2 * u * u)}[kind]
        keep = ys >= xs ** 2
        xs, ys = xs[keep], ys[keep]
    return xs, ys


def fig_cramer_rao(path):
    b = []
    W, H = 800, 405
    P1 = Panel(b, 60, 56, 320, 250, (4, 60), (0.9, 2.8))
    P1.frame([10, 20, 30, 40, 50, 60], [1, 1.5, 2, 2.5], "N", "variance / (Cramér–Rao bound)", "Exponential distribution: the rate", True)
    Ns = np.arange(4.0, 60.1, 0.5)
    P1.line([4, 60], [1, 1], "dash s0")
    P1.line(Ns, Ns ** 3 / ((Ns - 1) ** 2 * (Ns - 2)), "ln s1")
    P1.line(Ns, Ns / (Ns - 2), "ln s2")
    P1.line(Ns, 1 + 4 / Ns, "dot s1")
    legend_col(b, 66 + 120, 80, [("s1", "MLE 1/x̄ of the rate"), ("s2", "unbiased (N−1)/(N x̄)"), ("dot s1", "1 + 4/N"), ("dash s0", "x̄, for the mean η: exactly 1")])
    P2 = Panel(b, 450, 56, 320, 250, (0, 3), (0, 6.5))
    P2.frame([0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], "u · N^{1/4}  (units of the threshold)", "N · MSE(u)", "Hodges' estimator, N = 100", True)
    us = np.linspace(0, 3 * 100 ** -0.25, 400)
    P2.line(us * 100 ** 0.25, [100 * hodges_mse(100, u) for u in us], "ln s3")
    P2.line([0, 3], [1, 1], "dash s2")
    P2.text(1.55, 1.15, "Cramér–Rao bound G⁻¹ = 1", "sm", "start", 0, -3); P2.text(0.12, 0.45, "0.019 at u = 0", "sm", "start"); P2.text(1.15, 5.9, "5.6 near u ≈ 0.3", "sm", "start")
    note(b, 60, 376, ["Left: no estimator of the rate attains the bound at finite N, the MLE and the unbiased one both exceed it; the mean parameter does.", "Right: Hodges' estimator is below the bound at u = 0 and far above it nearby."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The Cramér–Rao bound and its fine print",
        "Left: for the exponential distribution the variance of the maximum likelihood estimator of the rate divided by the Cramér–Rao bound is N cubed over (N minus 1) squared times (N minus 2), tending to one as one plus four over N; the unbiased estimator has N over N minus 2; the sample mean attains the bound for the mean parameter. Right: Hodges' superefficient estimator of a Gaussian mean has N times mean squared error far below one at zero and far above one close to it.", b))


def fig_ancillary(path):
    rng = np.random.default_rng(5)
    a, bb = ab_samples(15, 70, 1.0, 1.0, rng)
    kinds = [("MLE", "MLE: straight leaves, orthogonal to S (efficiency 1)"), ("e-orth", "e-orthogonal: curved leaves, orthogonal (1)"),
             ("xbar", "x̄: vertical leaves, 35° to S (efficiency 1/3)"), ("m2", "√(m₂/2): horizontal, 71° to S (efficiency 8/9)")]
    b = []
    W, H = 800, 745
    pos = [(60, 60), (440, 60), (60, 400), (440, 400)]
    for (kind, title), (x0, y0) in zip(kinds, pos):
        P = Panel(b, x0, y0, 310, 250, (0.2, 1.9), (0, 5.2))
        P.frame([0.5, 1.0, 1.5], [0, 1, 2, 3, 4, 5], "η₁ = E[x]", "η₂ = E[x²]", title, True)
        xs = np.linspace(0.2, 1.9, 200)
        P.line(xs, xs ** 2, "dash s0")
        for u in (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6):
            lx, ly = eta_leaf(kind, u)
            P.line(lx, ly, "ln s1" if abs(u - 1.0) < 1e-9 else "thin s1")
        us = np.linspace(0.3, 1.62, 200)
        P.line(us, 2 * us ** 2, "ln sk")
        est = EST[kind](a, bb)
        for i in range(len(a)):
            P.dot(a[i], bb[i], "f0", 2.2)
        for i in range(0, len(a), 9):
            fx, fy = est[i], 2 * est[i] ** 2
            if 0.2 < fx < 1.9 and 0 < fy < 5.2:
                b.append(f'<line class="thin s2" x1="{P.X(a[i]):.1f}" y1="{P.Y(bb[i]):.1f}" x2="{P.X(fx):.1f}" y2="{P.Y(fy):.1f}"/>')
                P.dot(fx, fy, "f2", 3.2)
        P.dot(1.0, 2.0, "f4", 4.6)
    note(b, 60, 712, ["Each grey dot is the observed point (x̄, mean of x²) of N = 15 observations from N(1, 1²); the model S is the black parabola η₂ = 2η₁²,", "dashed: η₂ = η₁² (zero variance). Blue: the leaves A(u) of each estimator; orange: where the estimator sends a few of the points."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Four estimators as four foliations",
        "In the plane of the observed point (sample mean of x, sample mean of x squared), the model N(u, u squared) is a parabola. Four estimators of u are shown as families of leaves: the maximum likelihood estimator has straight leaves orthogonal to the model, the e-orthogonal estimator has curved leaves orthogonal to the model, the sample mean has vertical leaves and the second-moment estimator has horizontal leaves. Grey dots are observed points of 15 observations; orange lines send a few of them to the model along their leaf.", b))


def fig_second_order(path):
    st = STORE["second"]; terms = STORE["terms"]
    b = []
    W, H = 800, 405
    P1 = Panel(b, 70, 56, 320, 250, (10, 90), (0.3315, 0.3565))
    P1.frame([20, 40, 60, 80], [0.335, 0.34, 0.345, 0.35, 0.355], "N", "N · E[(u* − u)²]", "Bias-corrected estimators", True)
    Ns = np.linspace(12, 90, 120)
    cm, ce, cl = terms["MLE of u"]["coef"], terms["e-orth estimator of u"]["coef"], terms["MLE of log u"]["coef"]
    P1.line([10, 90], [1 / 3, 1 / 3], "dash s0")
    P1.line(Ns, 1 / 3 + cm / Ns, "ln s1"); P1.line(Ns, 1 / 3 + ce / Ns, "ln s2"); P1.line(Ns, 1 / 3 + cl / Ns, "ln s3")
    for N, mc in st.items():
        for key, cls in (("mle", "f1"), ("eorth", "f2"), ("log mle", "f3")):
            m, se = mc[key]
            P1.dot(N, m, cls, 3.8)
            b.append(f'<line class="thin s0" x1="{P1.X(N):.1f}" y1="{P1.Y(m - 2 * se):.1f}" x2="{P1.X(N):.1f}" y2="{P1.Y(m + 2 * se):.1f}"/>')
    P1.text(88, 1 / 3 + 0.0002, "1/g_uu = 1/3: the Cramér–Rao bound", "sm", "end", 0, 12)
    legend_col(b, P1.X(44), P1.Y(0.3555), [("s3", "MLE of log u"), ("s2", "e-orthogonal, u"), ("s1", "MLE of u")])
    P2 = Panel(b, 470, 56, 300, 250, (0, 3), (0, 0.36))
    P2.frame([0.5, 1.5, 2.5], [0, 0.1, 0.2, 0.3], "", "coefficient c of 1/N", "Where the second-order term comes from", True,
             {0.5: "MLE of u", 1.5: "e-orthogonal", 2.5: "MLE of log u"})
    for k, lab in enumerate(("MLE of u", "e-orth estimator of u", "MLE of log u")):
        t = terms[lab]; y = 0.0
        for val, cls in ((0.5 * t["Gm2"], "fillB"), (t["He2"], "fillG"), (0.5 * t["Hm2"], "fillO")):
            if val <= 0:
                continue
            b.append(f'<rect class="{cls}" x="{P2.X(k + 0.5) - 26:.1f}" y="{P2.Y(y + val):.1f}" width="52" height="{P2.Y(y) - P2.Y(y + val):.1f}"/>')
            y += val
        b.append(f'<text class="v" x="{P2.X(k + 0.5):.1f}" y="{P2.Y(y) - 6:.1f}" text-anchor="middle">{y:.3f}</text>')
    for k, (cls, lab) in enumerate((("fillB", "½ (Γᵐ)²: the parametrisation"), ("fillG", "(Hᵉ)²: statistical curvature of S"), ("fillO", "½ (Hᵐ_A)²: curvature of the leaf"))):
        b.append(f'<rect class="{cls}" x="474" y="{342 + 15 * k}" width="12" height="9"/><text class="sm" x="492" y="{350 + 15 * k}">{lab}</text>')
    note(b, 70, 372, ["Dots: Monte Carlo (6·10⁶ samples, bars ±2 s.e.);", "curves: 1/3 + c/N from (7.65)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The second-order terms of Theorem 7.5, checked",
        "Left: N times the mean squared error of three bias-corrected estimators against N, from simulation, with the curves one third plus c over N predicted by Theorem 7.5. Right: the coefficient c as a stack of the m-connection term, the e-curvature term and the leaf's m-curvature term: the MLE of u has the first two, an estimator with curved leaves adds the third, and parametrising by the logarithm of u makes the first much larger.", b))


def fig_tests(path):
    b = []
    W, H = 800, 405
    sh = STORE["shifts"]; tests = STORE["tests"]
    P1 = Panel(b, 70, 56, 320, 250, (0, 4), (0, 1.02))
    P1.frame([0, 1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "δ  (u = 1 + δ/√N)", "power", "Local power at level 0.05", True)
    ds = np.linspace(0, 4, 100)
    for key, cls in (("Rao / efficient", "ln s1"), ("m2 test", "ln s3"), ("xbar test", "ln s2")):
        P1.line(ds, [Phi(d * sh[key] - Z05) for d in ds], cls)
    for (N, d), pw in tests.items():
        if N != 500:
            continue
        P1.dot(d, pw["Rao"], "f1", 3.8); P1.dot(d, pw["m2"], "f3", 3.8); P1.dot(d, pw["xbar"], "f2", 3.8)
    P1.text(2.2, 0.99, "efficient: Rao, Wald, LR", "sm", "start", 0, 14); P1.text(0.15, 0.62, "m₂ test (8/9)", "sm", "start"); P1.text(2.7, 0.62, "x̄ test (1/3)", "sm", "start")
    P2 = Panel(b, 470, 56, 300, 250, (0.6, 1.6), (1.0, 3.6))
    P2.frame([0.8, 1.0, 1.2, 1.4], [1, 2, 3], "η₁", "η₂", "Rejection regions at N = 100", True)
    us = np.linspace(0.6, 1.5, 100)
    P2.line(us, 2 * us ** 2, "ln sk")
    z, N = Z05, 100
    xs = np.array([0.6, 1.6])
    P2.line(xs, xs + 1 + z * math.sqrt(3 / N), "ln s1")
    P2.line([1 + z / math.sqrt(N)] * 2, [1.0, 3.6], "ln s2")
    P2.line(xs, [2 + z * math.sqrt(6 / N)] * 2, "ln s3")
    P2.dot(1.0, 2.0, "f4", 4.6)
    P2.text(0.64, 2.18, "Rao: ⟂ S", "sm", "start"); P2.text(1.19, 1.25, "x̄", "sm", "start"); P2.text(0.64, 2.52, "m₂", "sm", "start")
    note(b, 70, 376, ["Dots: Monte Carlo at N = 500. A test rejects on the far side of its boundary line;", "the efficient test is the one whose line is orthogonal to S at η(u₀) (Theorem 7.7)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Efficiency of a test is the angle of its boundary",
        "Left: local power against delta for the efficient tests, whose boundary is orthogonal to the model, and for the tests based on the sample mean and the second moment, which lose two thirds and one ninth of the information; dots are simulations at N = 500. Right: in the plane of the observed point, the model N(u, u squared) and the boundaries of the three rejection regions at N = 100.", b))


def fig_midrange(path):
    b = []
    W, H = 800, 405
    h, edges = STORE["mid_hist"]
    P1 = Panel(b, 70, 56, 320, 250, (-3, 3), (0, 1.1))
    P1.frame([-3, -2, -1, 0, 1, 2, 3], [0, 0.25, 0.5, 0.75, 1], "N (midrange − u)", "density", "Uniform location family, N = 500", True)
    pts = []
    for k, hv in enumerate(h):
        pts += [(edges[k], hv), (edges[k + 1], hv)]
    P1.line([p[0] for p in pts], [p[1] for p in pts], "thin s0")
    xs = np.linspace(-3, 3, 400)
    P1.line(xs, np.exp(-2 * np.abs(xs)), "ln s1")
    P1.line(xs, np.exp(-xs ** 2) / math.sqrt(2 * math.pi * 0.5), "ln s2")
    P1.text(0.35, 1.04, "Laplace, exp(−2|x|)", "sm", "start"); P1.text(1.1, 0.42, "Gaussian, same variance ½", "sm", "start")
    P2 = Panel(b, 470, 56, 300, 250, (0, 4), (0, 3.4))
    P2.frame([0.5, 1.5, 2.5, 3.5], [0, 1, 2, 3], "", "excess kurtosis", "Not Gaussian, not stable", True, {0.5: "N=10", 1.5: "50", 2.5: "500", 3.5: "5000"})
    for k, N in enumerate((10, 50, 500, 5000)):
        P2.dot(k + 0.5, STORE["mid_kurt"][N], "f1", 4.6)
    P2.line([0, 4], [3, 3], "dash s2"); P2.line([0, 4], [0, 0], "dash s0")
    P2.text(0.1, 3.08, "Laplace: 3", "sm", "start"); P2.text(0.1, 0.12, "Gaussian: 0", "sm", "start")
    note(b, 70, 376, ["Left: the error is of order 1/N and its limit is a Laplace law, with a sharp peak. Right: its kurtosis tends to 3;", "a sum of two independent copies has kurtosis 1.5, so the limit is not stable."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "A non-regular model: the midrange of a uniform sample",
        "Left: histogram of N times the error of the midrange for the uniform location family at N = 500, with the limiting Laplace density and a Gaussian of the same variance. Right: the excess kurtosis of that error tends to 3, the Laplace value, as N grows.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_cramer_rao(out / "cramer-rao.svg")
    fig_ancillary(out / "ancillary-families.svg")
    fig_second_order(out / "second-order.svg")
    fig_tests(out / "tests.svg")
    fig_midrange(out / "uniform-midrange.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_estimation()
    check_exp_family()
    check_curved()
    check_first_order()
    check_bias()
    check_second_order()
    check_isolated_terms()
    check_tests()
    check_nonregular()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
