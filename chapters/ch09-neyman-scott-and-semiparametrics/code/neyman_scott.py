#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 9, checked by hand.

Neyman-Scott problem, estimating functions and semiparametric models. Every number quoted in the notes comes from here.

Running examples. (i) The coefficient of proportionality of section 9.2: x = v + e, y = u v + e' with unit Gaussian noise, u the parameter of
interest and the specimen weight v the nuisance parameter, either one v shared by all pairs (section 9.1) or one v per pair (Neyman-Scott).
(ii) Two more Neyman-Scott problems with exact solutions: boxes of Gaussian measurements with one mean per box, and matched pairs of binary
outcomes with one log-odds per pair. (iii) A finite model for the geometry of section 9.4: x, y in {0..m}, binomial pairs with odds ratio u and a
nuisance parameter that is the natural parameter of s = x + y, with a mixing law k on a nine-point grid; every statement about tangent spaces,
transports, the efficient score and the set of estimating functions is then linear algebra on 16 outcomes. (iv) The gamma shape of a spike train (9.5.4).

Checked here, in the order the notes use them:

  1. section 9.1 (9.4)-(9.19): the Fisher matrix of the proportionality model with one shared v, the efficient information as a Schur complement
     and as the inverse of a block of the inverse, the efficient score, the MLE ybar/xbar against 1/gbar; orthogonal nuisance parameters: the ODE
     for a scalar u (two models), and a vector-u counterexample in which no orthogonal nuisance coordinate exists (holonomy of the horizontal lift);
  2. section 9.2 (9.20)-(9.28): the MLE settling on the wrong value in the Gaussian-variance problem (limit (m-1)/m) and for matched pairs
     (limit u^2); the five solutions of (9.22)-(9.26): the limit of least squares, the Cauchy limit of the averaging method, the gross average,
     MLE = TLS and the second root of (9.26); Stonehenge (Fig. 9.2): the MLE of a radius with one unknown angle per stone;
  3. section 9.3 (9.29)-(9.43): Theorem 9.1 on the class (9.97)/(9.102) with the corrected (9.103) against the printed one, (9.105), the sign of (9.39),
     the second root of the population equation against (9.30)/(9.32), the gross average when E v = 0, the matrix sandwich (9.43);
  4. section 9.4 (9.44)-(9.80): the finite model: dimensions of T_U, T_K, T_A (9.57), (9.82), the duality of the transports (9.65), invariance of T_K
     (9.76), (9.74)-(9.75), Theorems 9.3, 9.5, 9.6 and the efficient score (9.90)-(9.91), the printed denominator of (9.80), the hierarchy of informations,
     matched pairs (m = 1) and a model with no estimating function; the signs in (9.56), (9.71)-(9.74);
  5. section 9.5 (9.81)-(9.134): the Lemma (9.82) and the efficient score (9.95) in the continuous proportionality model; the proportionality problem against the semiparametric bound for Gaussian and two-atom mixing laws (Cauchy-Schwarz
     form of Theorem 9.6), TLS against the gross average, simulation, wrong mixing laws; the scale problems (equal and unequal box sizes, common mean with
     unknown precisions); the spike train: the efficient score, the limit of the joint MLE, S against L_V, overlapping pairs, robustness.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Monte Carlo uses fixed seeds and exact sampling of sufficient statistics where possible; the whole script takes about a dozen seconds.

Run:  python3 neyman_scott.py            (checks)
      python3 neyman_scott.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

CHUNK = 1_000_000
STORE = {}                         # numbers computed by the checks, reused by the figures


def head(s):
    print("\n" + s)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


# ------------------------------------------------------------------ numerical tools

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(48)
GH_W = GH_W / math.sqrt(2 * math.pi)             # E[g(Z)] = sum GH_W g(GH_X) for Z ~ N(0, 1)


def Enorm(fn, mu=0.0, sd=1.0):
    """E[fn(X)], X ~ N(mu, sd^2), by 48-point Gauss-Hermite quadrature (exact for polynomials of degree < 96)."""
    return float(np.sum(GH_W * fn(mu + sd * GH_X)))


def digamma(x):
    r = 0.0
    while x < 8:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def trigamma(x):
    r = 0.0
    while x < 8:
        r += 1 / (x * x)
        x += 1
    f = 1 / (x * x)
    return r + 1 / x + f / 2 + (1 / x) * f * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))


def dawson(z):
    """Dawson's integral F(z) = exp(-z^2) int_0^z exp(t^2) dt, by Simpson's rule (odd in z)."""
    if z < 0:
        return -dawson(-z)
    if z == 0:
        return 0.0
    n = 4000
    t = np.linspace(0, z, n + 1)
    y = np.exp(t * t)
    h = z / n
    return math.exp(-z * z) * h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def rk4(f, y0, t0, t1, n):
    """Classical Runge-Kutta for y' = f(t, y), y a numpy array; returns y(t1)."""
    y = np.array(y0, float)
    h = (t1 - t0) / n
    t = t0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def Phi_inv(p):
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ 1. a model with a nuisance parameter (section 9.1)

def check_nuisance():
    head("1. A statistical model with a nuisance parameter (section 9.1)")
    u, v = 1.5, 2.0
    q = 1 + u * u
    print(f"   Coefficient of proportionality (9.2)-(9.3), sigma = 1, ONE specimen weight v shared by all N pairs: x ~ N(v, 1), y ~ N(u v, 1); (u, v) = ({u}, {v}), 1 + u^2 = {q}")
    Z1, Z2 = np.meshgrid(GH_X, GH_X, indexing="ij")
    W2 = np.outer(GH_W, GH_W)
    x, y = v + Z1, u * v + Z2
    eu = v * (y - u * v)                                  # score of u:  d/du log p
    ev = (x - v) + u * (y - u * v)                        # score of v
    g = np.array([[np.sum(W2 * eu * eu), np.sum(W2 * eu * ev)], [np.sum(W2 * eu * ev), np.sum(W2 * ev * ev)]])
    gcl = np.array([[v * v, u * v], [u * v, q]])
    print(f"   Fisher matrix (9.4)-(9.7) by 2-D Gauss-Hermite quadrature: {np.round(g, 6).tolist()}; closed form [[v^2, uv], [uv, 1+u^2]] = {gcl.tolist()}; largest difference {np.abs(g - gcl).max():.1e}")
    gbar = g[0, 0] - g[0, 1] ** 2 / g[1, 1]
    ginv = np.linalg.inv(g)
    print(f"   efficient information (9.10): gbar = g_uu - g_uv^2 / g_vv = {gbar:.6f} = v^2/(1+u^2) = {v * v / q:.6f};  the (u,u) entry of the INVERSE matrix is g^(uu) = {ginv[0, 0]:.6f} = 1/gbar = {1 / gbar:.6f} (9.9)-(9.10)")
    print(f"      the inverse of the (u,u) BLOCK of g is 1/g_uu = {1 / g[0, 0]:.4f}, which differs by the factor 1+u^2 = {q}: information about u is divided by {g[0, 0] / gbar:.4f} when v is unknown (9.11)")
    print(f"      the other reading of g^(kappa lambda), the nuisance entry of the full inverse g^(alpha beta) = {ginv[1, 1]:.4f}, would give g_uu - g_uv^2 * {ginv[1, 1]:.4f} = {g[0, 0] - g[0, 1] ** 2 * ginv[1, 1]:.4f}, a negative 'information'")
    ang = math.degrees(math.acos(g[0, 1] / math.sqrt(g[0, 0] * g[1, 1])))
    print(f"      in the Fisher metric the angle between e_u and e_v is {ang:.2f} degrees, |e_u| = {math.sqrt(g[0, 0]):.4f}, |e_v| = {math.sqrt(g[1, 1]):.4f}, and |ebar_u| = |e_u| sin(angle) = {math.sqrt(g[0, 0]) * math.sin(math.radians(ang)):.4f} = sqrt(gbar) = {math.sqrt(gbar):.4f}")
    # efficient score (9.14)-(9.16)
    ebar = eu - g[0, 1] / g[1, 1] * ev
    print(f"   efficient score (9.14): E[ebar^2] = {np.sum(W2 * ebar ** 2):.6f} (= gbar, 9.16); E[ebar e_v] = {np.sum(W2 * ebar * ev):.1e} (orthogonal to the nuisance score);"
          f" closed form v (y - u x)/(1+u^2) agrees on the quadrature grid to {np.abs(ebar - v * (y - u * x) / q).max():.1e}")
    # (9.12): the MLE is ybar/xbar; N Var
    N, R = 2000, 2_000_000
    rng = np.random.default_rng(91)
    left = R
    s1 = s2 = t1 = t2 = 0.0
    while left > 0:
        r = min(CHUNK, left)
        left -= r
        xb = rng.normal(v, 1 / math.sqrt(N), r)
        yb = rng.normal(u * v, 1 / math.sqrt(N), r)
        a, b = yb / xb, yb / v                           # MLE of u when v is unknown (ybar/xbar) and when v is known
        s1 += float(((a - u) ** 2).sum()); s2 += float(((a - u) ** 4).sum())
        t1 += float(((b - u) ** 2).sum()); t2 += float(((b - u) ** 4).sum())
    m1 = s1 / R; se1 = math.sqrt((s2 / R - m1 * m1) / R)
    m2 = t1 / R; se2 = math.sqrt((t2 / R - m2 * m2) / R)
    print(f"   The joint MLE solves v_hat = xbar, u_hat = ybar/xbar. N = {N}, {R} replications (xbar, ybar sampled exactly): N E(u_hat - u)^2 = {N * m1:.4f} +- {N * se1:.4f}"
          f"  against 1/gbar = (1+u^2)/v^2 = {q / (v * v):.4f} (9.12);  with v known (u_hat = ybar/v): {N * m2:.4f} +- {N * se2:.4f} against 1/g_uu = {1 / (v * v):.4f}")
    rng_pd = np.random.default_rng(3)
    worst_inv, worst_eig, cnt = 0.0, 1.0, 2000
    for _ in range(cnt):
        A_ = rng_pd.normal(size=(5, 5)); G5 = A_ @ A_.T + 0.1 * np.eye(5)                     # 2 parameters of interest and 3 nuisance parameters
        Sb = G5[:2, :2] - G5[:2, 2:] @ np.linalg.inv(G5[2:, 2:]) @ G5[2:, :2]
        worst_inv = max(worst_inv, float(np.abs(np.linalg.inv(Sb) - np.linalg.inv(G5)[:2, :2]).max()))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(G5[:2, :2] - Sb)[0]))
    print(f"   Vector case, {cnt} random positive-definite 5 x 5 information matrices (2 parameters of interest, 3 nuisance): the (a,b) block of the INVERSE equals the inverse of g_ab - g_(a kappa) g^(kappa lambda) g_(lambda b) (the inverse of the nuisance BLOCK in the middle factor) to {worst_inv:.1e};"
          f" g_ab - gbar_ab has smallest eigenvalue {worst_eig:.2e} >= 0 (9.11)")
    STORE["prop_common"] = dict(u=u, v=v, g=g, gbar=gbar)


def check_orthogonal():
    head("1b. Orthogonal nuisance parameters (9.17)-(9.19)")
    # scalar u, scalar v: dv/du = -g_uv/g_vv  (ODE), here -u v/(1+u^2)
    u0, v0, u1 = 0.0, 2.0, 1.5
    sol = rk4(lambda t, y: np.array([-t * y[0] / (1 + t * t)]), [v0], u0, u1, 400)[0]
    print(f"   Proportionality model: the curves v' = const orthogonal to the nuisance direction solve dv/du = -g_uv/g_vv = -u v/(1+u^2); RK4 from (0, {v0}) to u = {u1} gives v = {sol:.10f};"
          f" closed form v0/sqrt(1+u^2) = {v0 / math.sqrt(1 + u1 * u1):.10f}, so v' = v sqrt(1+u^2) is an orthogonal nuisance parameter")

    def gmat(u, v):
        q = 1 + u * u
        return np.array([[v * v, u * v], [u * v, q]])

    worst = 0.0
    for (u, vp) in [(0.3, 1.0), (1.5, 2.0), (-0.8, 3.0)]:
        v = vp / math.sqrt(1 + u * u)
        J = np.array([[1.0, 0.0], [-vp * u / (1 + u * u) ** 1.5, 1 / math.sqrt(1 + u * u)]])      # d(u, v)/d(u, v')
        gn = J.T @ gmat(u, v) @ J
        worst = max(worst, abs(gn[0, 1]))
        if (u, vp) == (1.5, 2.0):
            print(f"      in the chart (u, v'): g = {np.round(gn, 6).tolist()}; g'_(u v') = {gn[0, 1]:.1e} and g'_uu = {gn[0, 0]:.6f} = gbar = v^2/(1+u^2) = {v * v / (1 + u * u):.6f} (no information is lost, 9.17)")
    print(f"      largest |g'_(u v')| over three points: {worst:.1e}")
    # gamma: shape kappa (interest), scale theta (nuisance): g_kt = 1/theta, g_tt = kappa/theta^2 -> d theta/d kappa = -theta/kappa -> theta*kappa constant (the mean)
    kap, th = 3.0, 0.5
    z = np.linspace(1e-9, 70, 400001)
    pdf = z ** (kap - 1) * np.exp(-z) / math.gamma(kap)                          # density of x/theta
    dz = z[1] - z[0]
    sk = np.log(z) - digamma(kap)                                                # d/dkappa log p(x) with x = theta z: log z - psi(kappa) (the log theta terms cancel)
    st = (z - kap) / th                                                           # d/dtheta log p
    gkt = float(np.sum(pdf * sk * st) * dz); gtt = float(np.sum(pdf * st * st) * dz); gkk = float(np.sum(pdf * sk * sk) * dz)
    print(f"   Gamma(shape kappa, scale theta) at ({kap}, {th}): g_kk = {gkk:.5f} (trigamma {trigamma(kap):.5f}), g_k,theta = {gkt:.5f} (1/theta = {1 / th:.5f}), g_theta,theta = {gtt:.5f} (kappa/theta^2 = {kap / th ** 2:.5f});"
          f" dtheta/dkappa = -g_k,theta/g_theta,theta = -theta/kappa, so theta*kappa = the MEAN is the orthogonal nuisance parameter, which is why (9.125) uses the rate v = 1/mean")
    STORE["orth"] = dict()


def mu_twist(u1, u2, v):
    """Gaussian shift model x ~ N(mu(u1, u2, v), I_3) with mu = (u1, u2, 0) + v d(u), d = (-u2, u1, 1)/sqrt(1+|u|^2) (unit vectors)."""
    w = math.sqrt(1 + u1 * u1 + u2 * u2)
    return np.array([u1 - v * u2 / w, u2 + v * u1 / w, v / w])


def check_holonomy():
    head("1c. Vector u: the orthogonal parametrisation can be impossible (remark after (9.19))")
    print("   Model: x in R^3, x ~ N(mu(u1, u2, v), I), mu = (u1, u2, 0) + v d(u), d(u) = (-u2, u1, 1)/sqrt(1 + u1^2 + u2^2); the Fisher metric is g_ab = (d_a mu).(d_b mu).")
    print("   An orthogonal nuisance coordinate v'(u, v) needs the 'horizontal' curves (g_(a v) du^a + g_vv dv = 0) to be level sets of v', so a closed loop in the u-plane must lift to a closed curve.")
    h = 1e-6

    def metric_col(u1, u2, v):
        J = np.stack([(mu_twist(u1 + h, u2, v) - mu_twist(u1 - h, u2, v)) / (2 * h), (mu_twist(u1, u2 + h, v) - mu_twist(u1, u2 - h, v)) / (2 * h),
                      (mu_twist(u1, u2, v + h) - mu_twist(u1, u2, v - h)) / (2 * h)])
        return J @ J.T

    out = {}
    for rho in (0.25, 0.5, 1.0, 1.5):
        def rhs(t, s):
            u1, u2, vv = rho * math.cos(t), rho * math.sin(t), s[0]
            gm = metric_col(u1, u2, vv)
            du = np.array([-rho * math.sin(t), rho * math.cos(t)])
            return np.array([-(gm[2, 0] * du[0] + gm[2, 1] * du[1]) / gm[2, 2]])
        n = 800
        ts = np.linspace(0, 2 * math.pi, 41)
        vs = [0.7]
        for a, b in zip(ts[:-1], ts[1:]):
            vs.append(rk4(rhs, [vs[-1]], a, b, n // 40)[0])
        gap = vs[-1] - vs[0]
        cl = -2 * math.pi * rho * rho / math.sqrt(1 + rho * rho)
        out[rho] = (ts, np.array(vs), gap)
        print(f"      loop of radius {rho}: the horizontal lift of v does not close, v(2 pi) - v(0) = {gap:+.6f}; closed form -2 pi rho^2/sqrt(1+rho^2) = {cl:+.6f}")
    g0 = metric_col(0.0, 0.0, 0.7)
    print(f"   At u = 0 the metric is g_(a v) = {np.round(g0[2, :2], 8).tolist()} (orthogonal there): orthogonality at one point is possible, in a neighbourhood it is not: the obstruction is the curvature of the horizontal distribution, a density (2+|u|^2)/(1+|u|^2)^(3/2) per unit area of the u-plane (= 2 at u = 0); the gap is minus its integral over the disc")
    STORE["holonomy"] = out


# ------------------------------------------------------------------ 2. the Neyman-Scott problem (section 9.2)

def check_boxes():
    head("2. The Neyman-Scott problem (section 9.2)")
    print("   (a) N boxes of m Gaussian measurements, each box with its own unknown mean v_i, common variance sigma^2 = 2 to estimate (the scale problem of 9.5.3).")
    print("       The joint MLE of sigma^2 is (1/Nm) sum of within-box sums of squares W_i, W_i ~ sigma^2 chi^2_(m-1), so it is exactly (m-1)/m times the book's unbiased (9.114)-(9.115); sum W_i ~ sigma^2 chi^2_(N(m-1)) is sampled exactly.")
    sig2 = 2.0
    rng = np.random.default_rng(1)
    x1, x2 = 10.2, 9.6
    print(f"       Toy case, one specimen weighed twice, m = 2: readings {x1} and {x2}; the joint MLE puts the box mean at {(x1 + x2) / 2:.1f}, so both residuals are {abs(x1 - x2) / 2:.1f} and its variance estimate is {((x1 - x2) / 2) ** 2:.2f};"
          f" the unbiased within-box estimate (9.115) is {(x1 ** 2 + x2 ** 2 - (x1 + x2) ** 2 / 2) / 1:.2f}, twice as large (E (x1 - x2)^2 = 2 sigma^2, so the MLE's box estimate (x1 - x2)^2/4 averages sigma^2/2)")
    print("       m    N      MLE: mean (limit (m-1)/m sigma^2)   book (9.114): mean      N Var(book)  vs  2 sigma^4/(m-1) = bound    N Var(MLE)")
    rows = []
    for m in (2, 3, 5, 10):
        N, R = 2000, 200_000
        tot = sig2 * rng.chisquare(N * (m - 1), R)
        ml = tot / (N * m)
        bk = tot / (N * (m - 1))
        print(f"       {m:<4d} {N:<6d} {ml.mean():.4f} ({(m - 1) / m * sig2:.4f})                        {bk.mean():.4f} +- {bk.std() / math.sqrt(R):.4f}      {N * bk.var():.4f}      {2 * sig2 ** 2 / (m - 1):.4f}                 {N * ml.var():.4f}")
        rows.append((m, ml.mean(), bk.mean()))
    print("       for m = 2 the MLE settles at sigma^2/2 however many boxes there are:")
    seq = []
    for N in (10, 100, 1000, 10000, 100000):
        R = 4000
        tot = sig2 * rng.chisquare(N, R)
        seq.append((N, (tot / (2 * N)).mean(), (tot / N).mean(), (tot / (2 * N)).std(), (tot / N).std()))
        print(f"         N = {N:<7d} MLE {seq[-1][1]:.4f} +- {seq[-1][3]:.4f}    unbiased {seq[-1][2]:.4f} +- {seq[-1][4]:.4f}")
    STORE["boxes_seq"] = seq
    # the efficient score (9.113) is unbiased whatever the box means are, and its variance is the efficient information
    m, u = 4, 1 / sig2
    print(f"       Efficient score (9.113) for m = {m}, u = 1/sigma^2 = {u}: (1/u) - (1/(m-1)) (sum x^2 - (sum x)^2/m); mean over 400000 boxes for three different true means v:", end=" ")
    for mu in (0.0, 5.0, -20.0):
        x = rng.normal(mu, math.sqrt(sig2), (400_000, m))
        l = 1 / u - (np.sum(x * x, 1) - np.sum(x, 1) ** 2 / m) / (m - 1)
        print(f"v = {mu:+.0f}: {l.mean():+.4f} +- {l.std() / math.sqrt(len(l)):.4f};", end=" ")
    print()
    # unequal numbers of measurements: half the boxes with m = 2, half with m = 10 ("we can solve the problem in a similar way")
    nA = nB = 1000
    R = 100_000
    WA = sig2 * rng.chisquare(nA * 1, R); WB = sig2 * rng.chisquare(nB * 9, R)              # sums of the within-box sums of squares of each group (exact laws)
    unweighted = (WA / 1 + WB / 9) / (nA + nB)                                              # (9.114) read literally: the plain average of the box estimators
    pooled = (WA + WB) / (nA * 1 + nB * 9)                                                  # the sum of the efficient scores: total sum of squares over total degrees of freedom
    print(f"       Unequal box sizes, {nA} boxes with m = 2 and {nB} with m = 10: the plain average of the box estimators (9.114) has N Var = {(nA + nB) * unweighted.var():.4f} (theory 2 sigma^4 mean(1/(m_i - 1)) = {2 * sig2 ** 2 * (nA / 1 + nB / 9) / (nA + nB):.4f}),")
    print(f"       the sum of the efficient scores, sum W_i/sum(m_i - 1), has {(nA + nB) * pooled.var():.4f} (theory {2 * sig2 ** 2 * (nA + nB) / (nA + 9 * nB):.4f}); both unbiased (means {unweighted.mean():.4f}, {pooled.mean():.4f}): 'in a similar way' must mean the pooled estimator")
    gb = (m - 1) / (2 * u * u)
    x = rng.normal(0.0, math.sqrt(sig2), (3_000_000, m))
    lE = (m - 1) / 2 * (1 / u - (np.sum(x * x, 1) - np.sum(x, 1) ** 2 / m) / (m - 1))       # the efficient score proper r' - E[r'|s]: (m-1)/2 times (9.113)
    print(f"       the efficient score proper is (m-1)/2 times (9.113); its variance over 3000000 boxes is {lE.var():.3f} +- {lE.var() * math.sqrt(2 / len(lE)):.3f} against the efficient information (m-1)/(2u^2) = {gb:.4f}; so the bound for sigma^2 is 1/(N gbar u^4) = 2 sigma^4/(N (m-1)), which (9.114) attains")


def check_matched_pairs():
    print("   (b) Matched pairs: N pairs (x_i, y_i) of binary outcomes, P(x=1) = expit(v_i), the odds of y are u times the odds of x; v_i free (a stratum effect), u to estimate.")
    u = 2.0
    ev = np.exp(GH_X)
    pi1 = ev / (1 + ev)
    pi2 = u * ev / (1 + u * ev)
    cells = np.stack([GH_W @ ((1 - pi1) * (1 - pi2)), GH_W @ ((1 - pi1) * pi2), GH_W @ (pi1 * (1 - pi2)), GH_W @ (pi1 * pi2)])       # (x,y) = 00, 01, 10, 11, v ~ N(0,1)
    n_disc = cells[1] + cells[2]
    print(f"       Toy case, N = 1000 pairs: expected counts n00, n01, n10, n11 = {np.round(1000 * cells, 1).tolist()}; conditional estimate n01/n10 = {cells[1] / cells[2]:.3f}, joint MLE (n01/n10)^2 = {(cells[1] / cells[2]) ** 2:.3f}")
    print(f"       structural version v ~ N(0,1), u = {u}: cell probabilities (00, 01, 10, 11) = {np.round(cells, 5).tolist()}; p01/p10 = {cells[1] / cells[2]:.6f} = u for ANY mixing law, because the ratio is {u} at every v")
    print("       Unconditional joint MLE: concordant pairs push v_i to +-infinity and carry no information; for a discordant pair the best v_i gives likelihood 1/(1+sqrt u)^2 or u/(1+sqrt u)^2,")
    print("       so the profile log-likelihood is -2 n_disc log(1 + sqrt u) + n01 log u, maximal at u_hat = (n01/n10)^2. Brute-force check on 5 pairs of type (1,0) and 11 of type (0,1) (each v_i maximised numerically):")
    vgrid = np.linspace(-12, 12, 4801)

    def prof(uu):
        e = np.exp(vgrid)
        l10 = np.log(e / (1 + e) / (1 + uu * e)).max()
        l01 = np.log(1 / (1 + e) * uu * e / (1 + uu * e)).max()
        return 5 * l10 + 11 * l01
    ug = np.linspace(2.0, 8.0, 6001)
    pv = np.array([prof(x) for x in ug])
    print(f"         argmax over a grid of u: {ug[pv.argmax()]:.4f}; (11/5)^2 = {(11 / 5) ** 2:.4f}; the conditional estimator n01/n10 = {11 / 5:.4f}")
    rng = np.random.default_rng(2)
    print("       N        n01/n10 (conditional, consistent)  predicted s.e.   (n01/n10)^2 (joint MLE)")
    seq = []
    for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        c = rng.multinomial(N, cells)
        r = c[1] / c[2]
        seq.append((N, r, r * r))
        print(f"       {N:<8d} {r:.4f}                              {math.sqrt(u * (1 + u) ** 2 / (n_disc * N)):.4f}           {r * r:.4f}")
    STORE["matched_seq"] = seq
    # asymptotic variance of the conditional estimator from (9.35) with f = y(1-x) - u x(1-y)
    Ef2 = cells[1] + u * u * cells[2]
    Efp = -cells[2]
    avar = Ef2 / Efp ** 2
    N, R = 200_000, 20_000
    c = rng.multinomial(N, cells, R)
    est = c[:, 1] / c[:, 2]
    gbar = n_disc / (u * (1 + u) ** 2)
    print(f"       (9.35) for f = y(1-x) - u x(1-y): E f^2 = {Ef2:.5f}, E f' = {Efp:.5f}, N Var = {avar:.4f} = u(1+u)^2/P(discordant) = {u * (1 + u) ** 2 / n_disc:.4f} = 1/gbar = {1 / gbar:.4f};"
          f" N = {N}, {R} replications: N Var(n01/n10) = {N * est.var():.4f}; the joint MLE has mean {np.mean(est ** 2):.4f} (u^2 = {u * u})")
    STORE["matched"] = dict(u=u, cells=cells)


def rice_mean(nu, sig=1.0, h=0.02):
    """E rho for rho = |(nu + sig Z1, sig Z2)| (a Rice law), by a midpoint rule on the plane"""
    xs = np.arange(nu - 9 * sig + h / 2, nu + 9 * sig, h)
    ys = np.arange(-9 * sig + h / 2, 9 * sig, h)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dens = np.exp(-((X - nu) ** 2 + Y ** 2) / (2 * sig * sig)) / (2 * math.pi * sig * sig)
    return float(np.sum(np.hypot(X, Y) * dens) * h * h)


def check_stonehenge():
    print("   (d) Stonehenge (Fig. 9.2): points x = r cos(v_i) + e, y = r sin(v_i) + e' round a circle whose centre is known, unit noise, one unknown angle v_i per stone; r is the parameter of interest.")
    print("       The joint MLE of (r, v_1..v_N) puts each v_i in the direction of its point, leaving the residual rho_i - r with rho_i = |(x_i, y_i)|, so r_hat = mean of rho_i; rho_i has a Rice law whatever the angles are.")
    print("       Since E rho^2 = r^2 + 2 sigma^2 exactly, f = rho^2 - r^2 - 2 sigma^2 is an unbiased estimating function for every angle and (mean rho^2 - 2 sigma^2)^(1/2) is consistent.")
    rng = np.random.default_rng(14)
    N = 1_000_000
    rows = []
    for r in (1.0, 2.0, 5.0):
        lim = rice_mean(r)
        out = []
        for angles in ("uniform angles", "all angles 0.3"):
            ang = rng.uniform(0, 2 * math.pi, N) if angles == "uniform angles" else np.full(N, 0.3)
            x = r * np.cos(ang) + rng.normal(size=N); y = r * np.sin(ang) + rng.normal(size=N)
            rho = np.hypot(x, y)
            out.append((rho.mean(), rho.std() / math.sqrt(N), math.sqrt(max((rho * rho).mean() - 2.0, 0.0))))
        rows.append((r, lim))
        print(f"       r = {r}: the MLE (mean of rho) tends to E rho = {lim:.4f} (large-r approximation r + sigma^2/(2r) = {r + 1 / (2 * r):.4f}); simulated {out[0][0]:.4f} +- {out[0][1]:.4f} with uniform angles and {out[1][0]:.4f} +- {out[1][1]:.4f} with all angles equal;"
              f" the estimating-function solution sqrt(mean rho^2 - 2) gives {out[0][2]:.4f} and {out[1][2]:.4f}")
    STORE["stonehenge"] = rows


def three_sums(rng, r, N, u, mu_v, tau, sig=1.0):
    """r replications of N pairs (x_i, y_i), x = v + e, y = u v + e', v ~ N(mu_v, tau^2); the sums needed by every estimator, and the mean of y/x."""
    v = rng.normal(mu_v, tau, (r, N))
    x = v + sig * rng.normal(size=(r, N))
    y = u * v + sig * rng.normal(size=(r, N))
    return dict(Sx=x.sum(1), Sy=y.sum(1), Sxx=(x * x).sum(1), Syy=(y * y).sum(1), Sxy=(x * y).sum(1), avg=(y / x).mean(1))


def tls_root(S):
    """The root of (9.26) that is close to the slope when S_xy > 0: ((Syy - Sxx) + sqrt((Syy - Sxx)^2 + 4 Sxy^2)) / (2 Sxy)."""
    d = S["Syy"] - S["Sxx"]
    return (d + np.sqrt(d * d + 4 * S["Sxy"] ** 2)) / (2 * S["Sxy"])


def check_five_solutions():
    print("   (c) The five solutions of the coefficient-of-proportionality problem (9.20)-(9.26): u = 2, sigma = 1, v_i ~ k = N(2, 1) (the semiparametric reading (9.27)).")
    u, mu_v, tau = 2.0, 2.0, 1.0
    m2v = mu_v ** 2 + tau ** 2
    ls_lim = u * m2v / (m2v + 1)
    print(f"       least squares (9.23): u_hat -> E[xy]/E[x^2] = u E v^2/(E v^2 + sigma^2) = {u} * {m2v:.0f}/{m2v + 1:.0f} = {ls_lim:.4f}, not u (an attenuation, as in regression dilution)")
    # averaging: heavy tails
    ncd = lambda m: m * (2 * Phi(m) - 1) + 2 * phi(m)
    a = sum(w * phi(vv) * ncd(u * vv) for w, vv in zip(GH_W, mu_v + tau * GH_X))
    loc = u * sum(w * vv * math.sqrt(2) * dawson(vv / math.sqrt(2)) for w, vv in zip(GH_W, mu_v + tau * GH_X))
    print(f"       averaging (9.24): E|y/x| = infinity because x has a positive density at 0, so the mean of the ratios has no law of large numbers. Tail P(|y/x| > t) ~ 2a/t, a = E_v[phi(v) E|y|] = {a:.5f};")
    print(f"       the average of N ratios converges IN LAW (not in probability) to a Cauchy law with scale pi a = {math.pi * a:.4f} (interquartile range 2 pi a = {2 * math.pi * a:.4f}) centred at the principal-value mean u E_v[v sqrt2 Dawson(v/sqrt2)] = {loc:.4f}, not at u")
    rng = np.random.default_rng(3)
    print("       N        R      averaging: median  IQR      | gross average (9.25): median  IQR     | TLS=MLE: median  IQR       | LS: mean")
    rows = []
    for N, R, ch in ((100, 20000, 2000), (1000, 8000, 1000), (10000, 1600, 200)):
        parts = []
        left = R
        while left > 0:
            r = min(ch, left)
            left -= r
            parts.append(three_sums(rng, r, N, u, mu_v, tau))
        S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        gross = S["Sy"] / S["Sx"]
        tls = tls_root(S)
        ls = S["Sxy"] / S["Sxx"]
        qa = np.percentile(S["avg"], [25, 50, 75]); qg = np.percentile(gross, [25, 50, 75]); qt = np.percentile(tls, [25, 50, 75])
        rows.append((N, qa[1], qa[2] - qa[0], qg[1], qg[2] - qg[0], qt[1], qt[2] - qt[0], ls.mean()))
        pg, pt = 1.349 * math.sqrt(1.25 / N), 1.349 * math.sqrt(1.04 / N)
        print(f"       {N:<8d} {R:<6d} {qa[1]:.3f}            {qa[2] - qa[0]:.3f}    |      {qg[1]:.3f}            {qg[2] - qg[0]:.4f} (pred. {pg:.4f})   |       {qt[1]:.3f}           {qt[2] - qt[0]:.4f} (pred. {pt:.4f})      | {ls.mean():.4f}")
    STORE["five_rows"] = rows
    STORE["five_pred"] = dict(a=a, loc=loc, ls=ls_lim)
    print("       the IQR of the averaged ratios does not shrink with N (it stays near the Cauchy value), the IQRs of the gross average and TLS shrink like N^(-1/2) and agree with 1.349 sqrt(AVar/N), AVar = (1+u^2)/E v^2 = 1.25 and (q E v^2 + 1)/(E v^2)^2 = 1.04 (section 5);")
    print("       gross average and TLS are consistent, LS is not, averaging is not.")
    # TLS = MLE (the book says 'we can prove'); two roots
    rng2 = np.random.default_rng(4)
    n = 8
    vv = rng2.normal(mu_v, tau, n)
    x = vv + rng2.normal(size=n)
    y = u * vv + rng2.normal(size=n)
    Sxx, Syy, Sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()
    roots = np.roots([-Sxy, Syy - Sxx, Sxy])
    # joint MLE by alternating minimisation of sum (x - v)^2 + (y - u v)^2 over (u, v_1..v_n)
    uu = Sxy / Sxx
    for _ in range(20000):
        vi = (x + uu * y) / (1 + uu * uu)
        uu = (vi * y).sum() / (vi * vi).sum()
    prof = lambda t: float(np.sum((y - t * x) ** 2) / (1 + t * t))
    r_hi, r_lo = max(roots), min(roots)
    print(f"       'MLE = TLS' (9.26): joint MLE of (u, v_1..v_8) by alternating minimisation gives u = {uu:.12f}; the roots of (9.26) are {r_hi:.12f} and {r_lo:.12f}; product {roots.prod():.12f} (= -1 always: the second root is -1/u_hat)")
    print(f"       the profile sum of squared orthogonal distances is {prof(r_hi):.5f} at the first root (the MLE, a minimum) and {prof(r_lo):.5f} at the second (the worst line, a maximum); so (9.26) always has two roots and MLE = TLS = the first one")
    STORE["five_small"] = dict(x=x, y=y)
    # a sample of 40 pairs for the picture: the slopes of the estimators
    rng3 = np.random.default_rng(5)
    n = 40
    vv = rng3.normal(mu_v, tau, n)
    x = vv + rng3.normal(size=n)
    y = u * vv + rng3.normal(size=n)
    S1 = dict(Sx=x.sum(), Sy=y.sum(), Sxx=(x * x).sum(), Syy=(y * y).sum(), Sxy=(x * y).sum())
    sl = dict(LS=S1["Sxy"] / S1["Sxx"], avg=float(np.mean(y / x)), gross=S1["Sy"] / S1["Sx"], TLS=float(tls_root({k: np.array(w) for k, w in S1.items()})))
    print(f"       one sample of {n} pairs (the figure): slopes  least squares {sl['LS']:.3f}, averaging {sl['avg']:.3f}, gross average {sl['gross']:.3f}, TLS = MLE {sl['TLS']:.3f}; true u = {u}")
    STORE["five_sample"] = dict(x=x, y=y, slopes=sl)


# ------------------------------------------------------------------ 3. estimating functions (section 9.3)

def avar_class(c, q, m1, m2):
    """N Var of the root of (9.102), sigma = 1: the corrected (9.103); q = 1 + u^2, m1 = mean v, m2 = mean v^2."""
    V = m2 - m1 * m1
    return q * ((c + q * m1) ** 2 + q * q * V + q) / (c * m1 + q * m2) ** 2


def avar_printed(c, q, m1, m2):
    """(9.103) read literally (outer factor 1+u^2 on the first term only), with the missing 1/N dropped and the denominator squared."""
    V = m2 - m1 * m1
    return (q * (c + q * m1) ** 2 + q * q * V + q) / (c * m1 + q * m2) ** 2


def check_estimating_functions():
    head("3. Estimating functions (section 9.3)")
    u = 2.0
    q = 1 + u * u
    N, R = 500, 40_000
    # fixed (functional) nuisance values: the quantiles of N(2, 1)
    zq = np.array([Phi_inv((i + 0.5) / N) for i in range(N)])
    vfix = 2.0 + zq
    m1, m2 = float(vfix.mean()), float((vfix ** 2).mean())
    V = m2 - m1 * m1
    cstar = m1 / V
    print(f"   Class (9.97): f = (y - u x) h(x + u y), h(s) = s + c (9.101); estimating equation (9.102) is a quadratic in u; u = {u}, sigma = 1, N = {N} fixed v_i (quantiles of N(2,1): mean {m1:.4f}, mean of squares {m2:.4f}, variance {V:.4f}),")
    print(f"   root chosen = the one closer to the gross average; {R} replications. (9.105): c_hat = vbar/(vbar2 - vbar^2) = {cstar:.4f}")
    rng = np.random.default_rng(5)
    Sx = np.empty(R); Sy = np.empty(R); Sxx = np.empty(R); Syy = np.empty(R); Sxy = np.empty(R)
    for st in range(0, R, 2000):
        r = min(2000, R - st)
        x = vfix[None, :] + rng.normal(size=(r, N))
        y = u * vfix[None, :] + rng.normal(size=(r, N))
        Sx[st:st + r] = x.sum(1); Sy[st:st + r] = y.sum(1); Sxx[st:st + r] = (x * x).sum(1); Syy[st:st + r] = (y * y).sum(1); Sxy[st:st + r] = (x * y).sum(1)
    gross = Sy / Sx

    def root(c):
        a, b, c0 = -Sxy, Syy - Sxx - c * Sx, Sxy + c * Sy
        d = np.sqrt(b * b - 4 * a * c0)
        r1, r2 = (-b + d) / (2 * a), (-b - d) / (2 * a)
        return np.where(np.abs(r1 - gross) < np.abs(r2 - gross), r1, r2)
    print("       c           N Var (Monte Carlo)   (9.103) corrected   (9.103) as printed, 1/N and the square of the denominator restored [N Var]")
    rows = []
    for c in (0.0, 1.0, cstar, 10.0, 100.0):
        e = root(c)
        mc = N * e.var()
        rows.append((c, mc, avar_class(c, q, m1, m2), avar_printed(c, q, m1, m2)))
        print(f"       {c:<11.4f} {mc:.4f} +- {mc * math.sqrt(2 / R):.4f}      {rows[-1][2]:.4f}              {rows[-1][3]:.4f}")
    mc = N * gross.var()
    rows.append((float("inf"), mc, q / m1 ** 2, q / m1 ** 2))
    print(f"       gross average (c = infinity): {mc:.4f} +- {mc * math.sqrt(2 / R):.4f}      {q / m1 ** 2:.4f} = (1+u^2)/vbar^2")
    STORE["class_rows"] = rows
    STORE["class_par"] = dict(u=u, q=q, m1=m1, m2=m2, cstar=cstar)
    bnd = avar_class(cstar, q, m1, m2)
    lo_, hi_ = cstar, 200.0
    for _ in range(100):
        mid = 0.5 * (lo_ + hi_)
        if avar_printed(mid, q, m1, m2) < bnd:
            lo_ = mid
        else:
            hi_ = mid
    STORE["printed_cross"] = 0.5 * (lo_ + hi_)
    print(f"   The printed (9.103) lies below the minimum {bnd:.4f} of the correct curve (the efficiency bound) for every c from 0 up to {STORE['printed_cross']:.2f}, including the book's own optimum c = {cstar:.4f}, where it gives {avar_printed(cstar, q, m1, m2):.4f}.")
    lit = (q * (cstar + q * m1) ** 2 + q * q * (m2 - m1 * m1) + q) / (cstar * m1 + q * m2)
    print(f"   Read literally (no 1/N, denominator not squared) the printed (9.103) gives {lit:.4f} at c = {cstar:.4f}: not a variance of order 1/N, so the 1/N and the square are taken as typographical.")
    print(f"   The minimum of the corrected formula is at c = {cstar:.4f} (a grid search over c gives {min(((avar_class(c, q, m1, m2), c) for c in np.linspace(0, 8, 8001)))[1]:.4f}), confirming (9.105) for sigma = 1; for general sigma it is sigma^2 vbar/(vbar2 - vbar^2).")
    print(f"   (9.35) is a statement about the limiting law: the roots have no finite second moment (x-bar has a density at 0 for every N); the Monte Carlo variances above are finite only because the probability of that event is about exp(-{N * m1 ** 2 / 2:.0f}).")
    # (9.36)-(9.39): the linearisation  u_hat - u = -eps/(sqrt(N) A),  eps = N^(-1/2) sum f(x_i, u),  A = E f'
    print("   The linearisation (9.36)-(9.39), for c = 0 (TLS) and c = c_hat: eps = N^(-1/2) sum f(x_i, y_i; u), A = E f' = -(q E v^2 + c E v):")
    for c in (0.0, cstar):
        eps = (-Sxy * u * u + (Syy - Sxx - c * Sx) * u + (Sxy + c * Sy)) / math.sqrt(N)
        A = -(q * m2 + c * m1)
        e = root(c)
        slope = float(np.cov(math.sqrt(N) * (e - u), eps)[0, 1] / eps.var())
        corr = float(np.corrcoef(e - u, eps)[0, 1])
        print(f"      c = {c:.4f}: A = {A:.4f}; regression slope of sqrt(N)(u_hat - u) on eps = {slope:+.4f} against -1/A = {-1 / A:+.4f} (the printed (9.39) has +1/A, the opposite sign); correlation {corr:+.4f}; Var(eps) = {eps.var():.3f} against E f^2 = {q * (q * q * m2 + 2 * q * c * m1 + c * c + q):.3f}")


def check_roots():
    head("3b. Is the estimating function identifying? (9.30), (9.32) for the class (9.97)")
    u, mu_v, tau = 2.0, 2.0, 1.0
    m1, m2 = mu_v, mu_v ** 2 + tau ** 2
    X3, Y3, Z3 = np.meshgrid(GH_X[::1], GH_X, GH_X, indexing="ij")
    W3 = GH_W[:, None, None] * GH_W[None, :, None] * GH_W[None, None, :]
    v = mu_v + tau * X3
    x = v + Y3
    y = u * v + Z3

    def Psi(c, up):
        return float(np.sum(W3 * (y - up * x) * (up * y + x + c)))

    print(f"   Population equation E[f(x,y; u')] under the truth u = {u}, v ~ N({mu_v}, {tau}^2): closed form (u - u')[(1 + u u') E v^2 + c E v], quadrature over (v, noise, noise) in brackets")
    for c in (0.0, 2.0):
        print(f"      c = {c}:", end="  ")
        for up in (-1 / u, -0.9, 0.3, u, 3.0):
            cl = (u - up) * ((1 + u * up) * m2 + c * m1)
            print(f"u' = {up:+.2f}: {cl:+.4f} ({Psi(c, up):+.4f})", end="  ")
        print()
        r2 = -(m2 + c * m1) / (u * m2)
        print(f"      so the population equation has the roots u' = u = {u} and u' = -(E v^2 + c E v)/(u E v^2) = {r2:.4f}" + (" = -1/u" if c == 0 else ""))
    print("   (9.32) says E_(p_K)[f(x, u')] = 0 only for u' = u; for every member of the class it also vanishes at a second value u' != u. (9.30) fails too: for c = 0 the expectation vanishes at u' = -1/u for EVERY v;")
    print("   for c != 0 and a given u' it vanishes at v = -c/(1 + u u') and changes sign there, so a mixing law k centred near that v can cancel it.")
    print("   The estimator is the root of the sample equation (9.102) that lies near a consistent preliminary estimate (here the gross average); the sample equation always has two roots and the other one is not a consistent estimate of u.")
    # gross average with mean v = 0
    rng = np.random.default_rng(6)
    u, tau = 2.0, 2.0
    vx, vy, cxy = tau ** 2 + 1, u * u * tau ** 2 + 1, u * tau ** 2
    loc = cxy / vx
    scale = math.sqrt(vx * vy - cxy ** 2) / vx
    print(f"   The gross average needs E v != 0: with v ~ N(0, {tau}^2), u = {u}, (sum x, sum y) is exactly bivariate normal for every N (covariance N [[{vx:.0f}, {cxy:.0f}], [{cxy:.0f}, {vy:.0f}]]), so sum y/sum x is exactly Cauchy with location Cov/Var(x) = {loc:.4f} and scale sqrt(Var x Var y - Cov^2)/Var x = {scale:.4f} (IQR {2 * scale:.4f}), whatever N:")
    L = np.linalg.cholesky(np.array([[vx, cxy], [cxy, vy]]))
    for N in (100, 10_000, 1_000_000):
        R = 200_000
        z = rng.normal(size=(R, 2)) @ L.T * math.sqrt(N)
        qg = np.percentile(z[:, 1] / z[:, 0], [25, 50, 75])
        print(f"      N = {N:<8d} sampled exactly, R = {R}: gross average median {qg[1]:.3f}, IQR {qg[2] - qg[0]:.3f}")
    for N, R in ((100, 20000), (10000, 800)):
        parts = []
        left = R
        while left > 0:
            r = min(1000 if N == 100 else 100, left)
            left -= r
            parts.append(three_sums(rng, r, N, u, 0.0, tau))
        S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        qt = np.percentile(tls_root(S), [25, 50, 75])
        print(f"      N = {N:<8d} TLS from full simulations, R = {R}: median {qt[1]:.3f}, IQR {qt[2] - qt[0]:.4f}   (TLS is still a good estimating function: E f' = -(1+u^2) E v^2 != 0 and IQR shrinks like N^(-1/2))")


def check_matrix_sandwich():
    head("3c. The vector case (9.41)-(9.43)")
    print("   Moment estimating function f(x; mu, s) = (x - mu, x^2 - mu^2 - s) for data of ANY law with mean mu and variance s; here x ~ Gamma(shape 3, scale 1): mu = 3, s = 3.")
    kap, th = 3.0, 1.0
    z = np.linspace(1e-9, 80, 800001)
    pdf = z ** (kap - 1) * np.exp(-z) / math.gamma(kap)
    dz = z[1] - z[0]
    mu, s = kap * th, kap * th * th
    f1, f2 = z - mu, z * z - mu * mu - s
    Eff = np.array([[np.sum(pdf * f1 * f1), np.sum(pdf * f1 * f2)], [np.sum(pdf * f1 * f2), np.sum(pdf * f2 * f2)]]) * dz
    A = np.array([[-1.0, 0.0], [-2 * mu, -1.0]])
    Ai = np.linalg.inv(A)
    cov = Ai @ Eff @ Ai.T
    wrong = Ai.T @ Eff @ Ai
    rng = np.random.default_rng(7)
    N, R = 2000, 12_000
    xs = rng.gamma(kap, th, (R, N))
    mh = xs.mean(1)
    sh = (xs * xs).mean(1) - mh * mh
    C = np.cov(np.stack([mh, sh])) * N
    print(f"   E[f f^T] = {np.round(Eff, 3).tolist()}, A = {A.tolist()}; (9.43) N Cov = A^-1 E[ff^T] A^-T = {np.round(cov, 3).tolist()}")
    print(f"   Monte Carlo (N = {N}, {R} replications): N Cov = {np.round(C, 3).tolist()}; the other ordering A^-T E[ff^T] A^-1 would give {np.round(wrong, 3).tolist()}")
    STORE["sandwich"] = dict(cov=cov, mc=C)


# ------------------------------------------------------------------ 4. information geometry of estimating functions (section 9.4): an exact finite model

class Pairs:
    """Binomial pairs: x, y in {0..m}; p(x, y; u, v) = C(m,x) C(m,y) exp(v (x+y)) u^y / [(1+e^v)(1+u e^v)]^m.
    The nuisance v is the natural parameter of s = x + y (the form (9.81)); u is the odds ratio between y and x (s' = 0, r = y log u).
    The mixing law k lives on a grid of v values; with at least 2m+1 grid values the nuisance tangent space is the whole space of functions of s."""

    def __init__(self, m=3, grid=None):
        self.m = m
        self.out = [(x, y) for x in range(m + 1) for y in range(m + 1)]
        self.X = np.array([o[0] for o in self.out]); self.Y = np.array([o[1] for o in self.out]); self.S = self.X + self.Y
        self.BX = np.array([math.comb(m, x) * math.comb(m, y) for x, y in self.out], float)
        self.grid = np.linspace(-2, 2, 9) if grid is None else np.asarray(grid, float)
        self.n = len(self.out)

    def pmf(self, u, v):
        w = self.BX * np.exp(v * self.S) * u ** self.Y
        return w / w.sum()

    def comps(self, u):
        """L x n matrix of the component distributions p(.; u, w), w on the grid."""
        return np.array([self.pmf(u, w) for w in self.grid])

    def pK(self, u, k):
        return k @ self.comps(u)

    def score_u_mix(self, u, k):
        """analytic u-score of the mixture: sum_w k_w p_w (y/u - m e^w/(1+u e^w)) / p_K"""
        P = self.comps(u)
        sc = self.Y[None, :] / u - self.m * np.exp(self.grid)[:, None] / (1 + u * np.exp(self.grid))[:, None]
        return (k @ (P * sc)) / (k @ P)

    def cond_mean_given_s(self, u, g):
        """E[g | s] under any mixture (the conditional law of (x, y) given s does not depend on v): non-central hypergeometric with odds u."""
        out = np.zeros(self.n)
        for sv in range(2 * self.m + 1):
            idx = np.where(self.S == sv)[0]
            w = self.BX[idx] * u ** self.Y[idx]
            w = w / w.sum()
            out[idx] = (w * g[idx]).sum()
        return out

    def TK(self, u, k):
        """rows span the nuisance tangent space (9.56): p_w/p_K - 1"""
        P = self.comps(u)
        return P / (k @ P)[None, :] - 1

    def lE_closed(self, u):
        """Corollary (9.91): r' - E[r'|s], r' = y/u"""
        return (self.Y - self.cond_mean_given_s(u, self.Y.astype(float))) / u


def ip(a, b, p):
    return float(np.sum(p * a * b))


def orth_basis(B, p, tol=1e-9):
    """orthonormal basis (w.r.t. the p-weighted inner product) of the row space of B, as columns of the returned matrix in the weighted coordinates sqrt(p) * vector"""
    M = np.sqrt(p)[:, None] * B.T
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sv > tol * sv[0]).sum())
    return U[:, :r]


def proj_off(g, B, p):
    """g minus its L2(p) projection on the row space of B"""
    Ub = orth_basis(B, p)
    w = np.sqrt(p) * g
    return (w - Ub @ (Ub.T @ w)) / np.sqrt(p)


def check_geometry():
    head("4. Information geometry of estimating functions (section 9.4): an exact finite model")
    M = Pairs(3)
    u0 = 2.0
    k1 = np.exp(-M.grid ** 2 / 2); k1 /= k1.sum()
    k2 = np.zeros(len(M.grid)); k2[1] = 0.5; k2[7] = 0.5
    p1, p2 = M.pK(u0, k1), M.pK(u0, k2)
    print(f"   Model: x, y in {{0..3}} (16 outcomes), odds ratio u = {u0}, nuisance v = log-odds of x (natural parameter of s = x + y); mixing laws on the nine-point grid v = -2, -1.5, ..., 2:")
    print(f"   k1 = discretised N(0,1), k2 = two atoms at v = {M.grid[1]:.2f} and {M.grid[7]:.2f}. T = mean-zero functions of (x, y): dimension {M.n - 1}.")
    # 4.1 dimensions
    r1 = np.linalg.matrix_rank(M.TK(u0, k1), tol=1e-10); r2 = np.linalg.matrix_rank(M.TK(u0, k2), tol=1e-10)
    H = np.array([(M.S == sv).astype(float) for sv in range(7)])
    Hc = H - (H * p1).sum(1, keepdims=True)
    ru = np.linalg.matrix_rank(np.vstack([M.TK(u0, k1), Hc]), tol=1e-10)
    print(f"   (9.57) dimensions: dim T_K = {r1} (k1) and {r2} (k2); the mean-zero functions of s alone have dimension {np.linalg.matrix_rank(Hc, tol=1e-10)} and adding them to T_K(k1) leaves the rank at {ru}: T_K = {{h(s)}}, which is (9.89)")
    P1 = M.comps(u0)
    sv_ = np.linalg.svd(P1, compute_uv=False)
    nul = M.n - int((sv_ > 1e-10).sum())
    print(f"   unbiased functions: those f with sum_(x,y) f p(x,y; u, w) = 0 for every grid value w: the null space of the 9 x 16 matrix has dimension {nul} = 16 - 7; (9.57): dim T_U + dim T_K + dim T_A = 1 + {r1} + {M.n - 1 - 1 - r1} = {M.n - 1}")
    # 4.2 Lemma (9.82): finite-difference u-score of the mixture versus s' E[v|s] + r' - E[psi'|s]
    h = 1e-5
    lu_fd = (np.log(M.pK(u0 + h, k1)) - np.log(M.pK(u0 - h, k1))) / (2 * h)
    ew = np.exp(M.grid)
    Zw = ((1 + ew) * (1 + u0 * ew)) ** M.m
    Lam = lambda sv, fn: sum(k1[j] * fn(M.grid[j]) * math.exp(M.grid[j] * sv) / Zw[j] for j in range(len(M.grid))) / sum(k1[j] * math.exp(M.grid[j] * sv) / Zw[j] for j in range(len(M.grid)))
    Ev_s = np.array([Lam(sv, lambda w: 1.0) for sv in range(7)])                         # E[1|s] check
    Epsi_s = np.array([Lam(sv, lambda w: M.m * math.exp(w) / (1 + u0 * math.exp(w))) for sv in range(7)])
    lu_formula = M.Y / u0 - Epsi_s[M.S]
    print(f"   Lemma (9.82): u-score of the mixture by a finite difference of log p_K in u (step 1e-5) against s' E[v|s] + r' - E[psi'|s] = 0 + y/u - E[m e^v/(1+u e^v) | s]: largest difference {np.abs(lu_fd - lu_formula).max():.1e}; E[score] = {ip(lu_fd, 1.0, p1):.1e}")
    # 4.3 transports and duality
    rng = np.random.default_rng(8)
    a = rng.normal(size=M.n); a -= ip(a, 1.0, p1)
    b = rng.normal(size=M.n); b -= ip(b, 1.0, p1)
    pe = lambda r: r - ip(r, 1.0, p2)                                                    # (9.60) e-transport k1 -> k2
    pm = lambda r: p1 / p2 * r                                                           # (9.63) m-transport k1 -> k2
    print(f"   Theorem 9.2 (9.65): for random mean-zero a, b: <a, b>_k1 = {ip(a, b, p1):+.10f};  <e-transport a, m-transport b>_k2 = {ip(pe(a), pm(b), p2):+.10f};")
    print(f"      the non-dual pairings differ: <e a, e b>_k2 = {ip(pe(a), pe(b), p2):+.6f}, <m a, m b>_k2 = {ip(pm(a), pm(b), p2):+.6f}; (9.61), (9.64): E_k2[e-transport a] = {ip(pe(a), 1.0, p2):.1e}, E_k2[m-transport a] = {ip(pm(a), 1.0, p2):.1e}")
    # invariance of T_K (9.76)
    B1 = M.TK(u0, k1); B2 = M.TK(u0, k2)
    resm = max(np.abs(proj_off(pm(r), B2, p2)).max() for r in B1)
    rese = max(np.abs(proj_off(pe(r), B2, p2)).max() for r in B1)
    lu1, lu2 = M.score_u_mix(u0, k1), M.score_u_mix(u0, k2)
    resu = np.abs(proj_off(pm(lu1), lu2[None, :], p2)).max()
    print(f"   Theorem 9.4 (9.76): m-transport of every element of T_K(k1) lies in T_K(k2): largest residual {resm:.1e}; the e-transport also preserves T_K here (T_K = all functions of s): residual {rese:.1e};"
          f" but the u-score is not transported to the u-score: distance of m-transport(l_u(k1)) from span l_u(k2) = {resu:.3f}")
    # (9.74)-(9.75): T_K is spanned by the m-transports of the elementary nuisance scores d/dw log p(.; u, w)
    elem = []
    for w in M.grid:
        pw = M.pmf(u0, w)
        lw = M.S - M.m * (math.exp(w) / (1 + math.exp(w)) + u0 * math.exp(w) / (1 + u0 * math.exp(w)))      # d/dw log p(x, y; u, w)
        elem.append(pw / p1 * lw)                                                                          # m-transport from the point mass at w to k1
    elem = np.array(elem)
    print(f"   (9.74)-(9.75): the 9 transported elementary nuisance scores (p_w/p_K) d/dw log p_w lie in T_K(k1) (largest residual {max(np.abs(proj_off(e, B1, p1)).max() for e in elem):.1e}) and have rank {np.linalg.matrix_rank(elem, tol=1e-10)} = dim T_K: they span it")
    # 4.4 efficient score
    lE_bf = proj_off(lu1, B1, p1)
    lE_bf2 = proj_off(lu2, B2, p2)
    lE_cl = M.lE_closed(u0)
    gbar1, gbar2 = ip(lE_cl, lE_cl, p1), ip(lE_cl, lE_cl, p2)
    print(f"   Theorem 9.7 / Corollary (9.91): brute-force projection of the u-score off T_K (a least-squares problem in L2(p_K)) against r' - E[r'|s] = (y - E[y|s])/u: largest difference {np.abs(lE_bf - lE_cl).max():.1e} (k1), {np.abs(lE_bf2 - lE_cl).max():.1e} (k2);")
    print(f"      the efficient score is the SAME vector for k1 and k2 (it does not depend on k), but the efficient information depends on k: gbar = E l_E^2 = {gbar1:.6f} (k1), {gbar2:.6f} (k2)")
    # 4.5 information hierarchy
    gv = []
    for w in M.grid:
        pw = M.pmf(u0, w)
        lw = M.Y / u0 - M.m * math.exp(w) / (1 + u0 * math.exp(w))
        gv.append(ip(lw, lw, pw))
    gv = np.array(gv)
    print(f"   Information about u: v known (average of g_uu(w) under k1) {k1 @ gv:.6f}  >  k1 known (Fisher information of the mixture) {ip(lu1, lu1, p1):.6f}  >  k unknown (efficient information) {gbar1:.6f}")
    print(f"      for k2: {k2 @ gv:.6f} > {ip(lu2, lu2, p2):.6f} > {gbar2:.6f}; the angle between l_u and T_K(k1): sin^2 = gbar/I(k known) = {gbar1 / ip(lu1, lu1, p1):.4f} (T_U and T_K are not orthogonal)")
    STORE["geo"] = dict(lE=lE_cl, p1=p1, p2=p2, gbar1=gbar1, gbar2=gbar2, X=M.X, Y=M.Y, S=M.S, hier1=(float(k1 @ gv), ip(lu1, lu1, p1), gbar1), hier2=(float(k2 @ gv), ip(lu2, lu2, p2), gbar2))
    # 4.6 Theorem 9.3: estimating functions from projections of an arbitrary vector; (9.68)
    g0 = rng.normal(size=M.n)

    def f_family(u, k=k1, g=g0):
        """an estimating function for every u: g - E[g|s] (orthogonal to all functions of s), the L2(p_K)-projection of g off {h(s)}"""
        return g - M.cond_mean_given_s(u, g)
    f0 = f_family(u0)
    bad = max(abs(ip(f0, 1.0, M.pmf(u0, w))) for w in (-3.0, -0.37, 0.0, 0.81, 2.7))
    print(f"   Theorem 9.3: f = g - E[g|s] for a random vector g: E_(u,v)[f] at five v values (four off the grid) is at most {bad:.1e} (unbiased for every v); <f, T_K(k1)> = {max(abs(ip(f0, r, p1)) for r in B1):.1e}, <f, T_K(k2)> = {max(abs(ip(f0, r, p2)) for r in B2):.1e}")
    fd = (f_family(u0 + h) - f_family(u0 - h)) / (2 * h)
    print(f"      (9.68): E[f'] = {ip(fd, 1.0, p1):+.8f} (finite difference in u of the function f(., u), expectation under k1) and -<l_u, f> = {-ip(lu1, f0, p1):+.8f}; also under k2: {ip(fd, 1.0, p2):+.8f} and {-ip(lu2, f0, p2):+.8f}")
    # 4.7 Theorem 9.5: every estimating function is alpha l_E + a, a in T_A
    Uu, svv, Vh = np.linalg.svd(P1)
    null = Vh[int((svv > 1e-10).sum()):]                                                  # basis of the 9-dim space of unbiased functions (orthonormal in the plain sense)
    coef = rng.normal(size=null.shape[0])
    f = coef @ null
    alpha = ip(f, lE_cl, p1) / ip(lE_cl, lE_cl, p1)
    a_ = f - alpha * lE_cl
    print(f"   Theorem 9.5 (9.79): a random element of the 9-dimensional space of unbiased functions is alpha l_E + a with alpha = {alpha:+.4f}; a is orthogonal to T_K (max {max(abs(ip(a_, r, p1)) for r in B1):.1e}) and to l_u (|<a, l_u>| = {abs(ip(a_, lu1, p1)):.1e}), i.e. a in T_A;"
          f" the unbiased functions with E f' = 0 form the kernel of one more linear functional: dimension {null.shape[0] - 1} = dim T_A")
    # 4.8 Theorem 9.6 : variance of the root for f_t = l_E + t a
    gt = rng.normal(size=M.n)
    ugrid = np.linspace(0.8, 3.2, 481)
    F0 = np.empty((M.n, len(ugrid))); A0 = np.empty((M.n, len(ugrid)))
    for j, uu in enumerate(ugrid):
        pk = M.pK(uu, k1)
        le = M.lE_closed(uu)
        gc = gt - M.cond_mean_given_s(uu, gt)
        aa = gc - ip(gc, le, pk) / ip(le, le, pk) * le
        aa = aa / math.sqrt(ip(aa, aa, pk)) * math.sqrt(ip(le, le, pk))                # normalised so that ||a||^2 = gbar at every u
        F0[:, j], A0[:, j] = le, aa
    j0 = int(np.argmin(np.abs(ugrid - u0)))
    gb0 = ip(lE_cl, lE_cl, p1)
    print(f"   Theorem 9.6 (9.80): estimating functions f_t(., u) = l_E(., u) + t a(., u) with a in T_A(u) normalised to ||a||^2 = gbar; observations from p_K(.; u = {u0}, k1) (counts are multinomial, the root of sum f_t = 0 is found on a grid of 481 values of u).")
    print(f"      asymptotic N Var = E f^2/(E f')^2 = (gbar + t^2 ||a||^2)/gbar^2 = (1 + t^2)/gbar, gbar = {gb0:.6f}:")
    rng2 = np.random.default_rng(9)
    rows = []
    for N, R in ((800, 40_000), (8000, 20_000)):
        cnt = rng2.multinomial(N, p1, R)
        for t in (0.0, 0.5, 1.0, 1.5):
            Ft = F0 + t * A0
            est = np.empty(R)
            for st in range(0, R, 4000):
                Sg = cnt[st:st + 4000] @ Ft
                cross = Sg[:, :-1] * Sg[:, 1:] < 0
                dist = np.abs(0.5 * (ugrid[:-1] + ugrid[1:]) - u0)[None, :] + np.where(cross, 0.0, 1e9)
                ix = dist.argmin(1)
                r_ = np.arange(len(ix))
                s0, s1 = Sg[r_, ix], Sg[r_, ix + 1]
                est[st:st + 4000] = ugrid[ix] + (ugrid[ix + 1] - ugrid[ix]) * s0 / (s0 - s1)
            mc = N * est.var()
            rows.append((N, t, mc, (1 + t * t) / gb0))
            print(f"         N = {N:<5d} t = {t:.1f}: Monte Carlo {mc:.3f} +- {mc * math.sqrt(2 / R):.3f}   asymptotic {(1 + t * t) / gb0:.3f}   (ratio to the bound 1/gbar = {mc * gb0:.3f}, predicted {1 + t * t:.3f})")
    STORE["geo_var"] = rows
    print("      the printed (9.80) divides E[l_E^2] by {E[l_E]}^2, and E[l_E] = 0 for a tangent vector: with the derivative restored, E[l_E'] = -E[l_E^2] gives 1/(N gbar).")
    ElE = ip(lE_cl, 1.0, p1)
    dle = (M.lE_closed(u0 + h) - M.lE_closed(u0 - h)) / (2 * h)
    print(f"      numbers: E[l_E] = {ElE:.1e} (so the printed denominator is 0), E[l_E'] = {ip(dle, 1.0, p1):+.6f}, -E[l_E^2] = {-gb0:+.6f}")
    # 4.9 m = 1: the unique estimating function; no estimating function when l_u is in T_K
    M1 = Pairs(1, grid=np.linspace(-2, 2, 5))
    kk = np.ones(5) / 5
    P1m = M1.comps(2.0)
    svm = np.linalg.svd(P1m)
    nm = svm[2][int((svm[1] > 1e-10).sum()):]
    fvec = nm[0] / nm[0][1]
    fvec = np.where(np.abs(fvec) < 1e-12, 0.0, fvec)
    print(f"   m = 1 (matched pairs): dim T = 3, dim T_K = 2, so the unbiased functions form a {nm.shape[0]}-dimensional space: f (00, 01, 10, 11) = {np.round(fvec + 0.0, 6).tolist()} = (0, 1, -u, 0) with u = 2, i.e. y(1-x) - u x(1-y), the estimator n01/n10 of 2.2")
    # Bernoulli(u + v): no estimating function
    pi = 0.3 + np.linspace(0.0, 0.4, 5)
    Pb = np.stack([1 - pi, pi], 1)
    nb = Pb.shape[1] - np.linalg.matrix_rank(Pb)
    print(f"   When l_u lies in T_K nothing exists (Theorem 9.5, 'only when'): x ~ Bernoulli(u + v) with v on a grid of 5 values: the space of unbiased functions has dimension {nb} (f(0) (1 - pi) + f(1) pi = 0 for 5 different pi forces f = 0); u and v are not separately identifiable")


def check_delta_signs():
    head("4b. The delta-function bookkeeping (9.56), (9.71)-(9.74)")
    eps = 0.02
    w = np.linspace(-9, 9, 36001)
    dw = w[1] - w[0]
    ddelta = lambda v, ww: -(v - ww) / eps ** 2 * np.exp(-(v - ww) ** 2 / (2 * eps ** 2)) / (eps * math.sqrt(2 * math.pi))      # d/dv of a narrow Gaussian standing for delta(v - w)
    bfun = lambda v: v * np.exp(-v * v / 2)                                                 # a perturbation b(v) of the mixing law with integral 0 (9.54)
    B = np.exp(-w * w / 2) - 1                                                              # B(w) = - int_0^w b(v) dv  (9.73)
    print("   (9.72): b(v) = int delta'_w(v) B(w) dw with B(w) = - int_0^w b and delta'_w(v) = d/dv delta(v - w) (9.71); b(v) = v exp(-v^2/2):")
    for v in (-1.5, -0.5, 0.7, 1.9):
        val = float(np.sum(ddelta(v, w) * B) * dw)
        print(f"      v = {v:+.1f}: the right-hand side is {val:+.5f}, b(v) = {float(bfun(v)):+.5f}, ratio {val / float(bfun(v)):+.4f}")
    p = lambda v: np.exp(-(v - 0.3) ** 2 / 2)
    ww = 1.0
    lhs956 = float(np.sum(p(w) * ddelta(w, ww)) * dw)                                       # int p(v) delta'_w(v) dv, i.e. (9.56) with b = delta'_w and p_K = 1
    rhs974 = float(p(ww) * (-(ww - 0.3)))                                                   # p(w) l_v(w) / p_K, the last line of (9.74)
    print(f"   (9.56) with b = delta'_w gives (1/p_K) int p delta'_w dv = {lhs956:+.5f} (= -p'(w)/p_K); the last line of (9.74), p(w) l_v(w)/p_K, is {rhs974:+.5f}: ratio {rhs974 / lhs956:+.4f}")
    print("   so (9.73) has the wrong sign for (9.71) (or (9.71) for (9.73)) and the leading minus in (9.74) contradicts (9.56); the span of the nuisance scores, which is all that is used, is unaffected")


# ------------------------------------------------------------------ 5. solutions to Neyman-Scott problems (section 9.5)

class Mixing:
    """A mixing law k(v) as atoms (v_j, p_j); the Gaussian N(mu, tau^2) is represented by 48 Gauss-Hermite atoms."""

    def __init__(self, atoms, probs):
        self.v = np.asarray(atoms, float); self.p = np.asarray(probs, float)
        self.mean = float(self.p @ self.v); self.m2 = float(self.p @ self.v ** 2); self.var = self.m2 - self.mean ** 2

    @staticmethod
    def gauss(mu, tau):
        return Mixing(mu + tau * GH_X, GH_W)


def s_grid(k, q, n=20001):
    lo, hi = q * k.v.min() - 14 * math.sqrt(q), q * k.v.max() + 14 * math.sqrt(q)
    return np.linspace(lo, hi, n)


def s_density(k, q, sg):
    """marginal density of s = x + u y when v ~ k: sum_j p_j N(s; q v_j, q)"""
    return k.p @ (np.exp(-(sg[None, :] - q * k.v[:, None]) ** 2 / (2 * q)) / math.sqrt(2 * math.pi * q))


def post_mean(k, q, sg):
    """E[v | s], computed with the largest exponent removed (stable far in the tails)"""
    L = np.log(np.maximum(k.p, 1e-300))[:, None] - (sg[None, :] - q * k.v[:, None]) ** 2 / (2 * q)
    L -= L.max(0, keepdims=True)
    W = np.exp(L)
    return (k.v[:, None] * W).sum(0) / W.sum(0)


def avar_h(k, q, h, n=20001):
    """N Var of the root of sum (y - u x) h(x + u y) = 0 (sigma = 1): q E[h^2] / (E[v h(s)])^2, v ~ k, s | v ~ N(q v, q); h is a function of s"""
    sg = s_grid(k, q, n)
    ds = sg[1] - sg[0]
    hv = h(sg)
    ps = s_density(k, q, sg)
    Eh2 = float(np.sum(ps * hv * hv) * ds)
    Evh = float(np.sum(ps * post_mean(k, q, sg) * hv) * ds)                 # E[v h(s)] = E[E[v|s] h(s)]
    return q * Eh2 / Evh ** 2


def hfun(k, q):
    """the estimating function h(s) = E_k[v|s] of the efficient score, as an interpolated function of s"""
    sg = s_grid(k, q)
    Ev = post_mean(k, q, sg)
    return lambda s_: np.interp(s_, sg, Ev)


def check_prop_score():
    head("5a. The efficient score of the proportionality problem (9.82), (9.90), (9.95) in the continuous model")
    u, v0, v1 = 2.0, 0.0, 2.0
    p0, p1 = 0.5, 0.5
    q = 1 + u * u
    xg, wg = np.polynomial.hermite_e.hermegauss(80)                                       # for E over z1 ~ N(0, q) given s
    wg = wg / math.sqrt(2 * math.pi)

    def pK(x, y, uu):
        return p0 * np.exp(-((x - v0) ** 2 + (y - uu * v0) ** 2) / 2) / (2 * math.pi) + p1 * np.exp(-((x - v1) ** 2 + (y - uu * v1) ** 2) / 2) / (2 * math.pi)

    def post(sv, uu):
        qq = 1 + uu * uu
        w0 = p0 * math.exp(v0 * sv - qq * v0 ** 2 / 2); w1 = p1 * math.exp(v1 * sv - qq * v1 ** 2 / 2)
        return w0 / (w0 + w1), w1 / (w0 + w1)

    rng = np.random.default_rng(15)
    pts = rng.uniform(-1, 4, (6, 2))
    h = 1e-5
    worst_lemma = worst_E = 0.0
    for (x, y) in pts:
        sv = x + u * y
        a0, a1 = post(sv, u)
        Ev, Ev2 = a1 * v1 + a0 * v0, a1 * v1 ** 2 + a0 * v0 ** 2
        lu_fd = (math.log(pK(x, y, u + h)) - math.log(pK(x, y, u - h))) / (2 * h)
        lu_formula = y * Ev - u * Ev2                                                     # (9.82): s' E[v|s] + r' - E[psi'|s] with s' = y, r' = 0, psi' = u v^2
        worst_lemma = max(worst_lemma, abs(lu_fd - lu_formula))
        # E[l_u | s]: integrate over z1 = y - u x ~ N(0, q) at fixed s = z2, with x = (s - u z1)/q, y = (z1 + u s)/q
        z1 = math.sqrt(q) * xg
        xs, ys = (sv - u * z1) / q, (z1 + u * sv) / q
        lu_all = ys * Ev - u * Ev2                                                        # l_u as a function of (x, y) at the same s: only y varies
        cond = float(np.sum(wg * lu_all))
        lE_proj = lu_formula - cond                                                       # l_u minus its conditional mean given s: the projection off T_K = {h(s)}
        lE_95 = (y - u * x) / q * Ev                                                      # (9.95)
        worst_E = max(worst_E, abs(lE_proj - lE_95))
    print(f"   k = two atoms v = 0, 2 (probability 1/2 each), u = 2: the u-score of the mixture by a finite difference of log p_K against the Lemma (9.82), y E[v|s] - u E[v^2|s]: largest difference {worst_lemma:.1e} at six random points;")
    print(f"   l_u minus its conditional mean given s (a 1-D quadrature over z1 = y - u x ~ N(0, 1+u^2)) against (9.95), (y - u x) E[v|s]/(1+u^2): largest difference {worst_E:.1e}; E[y|s] = u s/(1+u^2) was used in (9.95), and E[v|s] depends on k (here a logistic function of s), so unlike the finite model of section 4 the efficient score DOES depend on k")


def check_proportionality_bound():
    head("5. Solutions to Neyman-Scott problems (section 9.5)")
    print("   (a) Coefficient of proportionality, sigma = 1: efficient score (9.95) l_E = (y - u x) E[v|s]/(1+u^2); for ANY h the root of sum (y - u x) h(x + u y) is consistent and N Var = q E[h(s)^2]/(E[v h(s)])^2, q = 1+u^2")
    print("       (E f^2 = q E h^2 and E f' = -E[v h(s)], from z1 = y - u x, z2 = x + u y independent, N(0,q) and N(q v, q)); by Cauchy-Schwarz this is >= q/E[(E[v|s])^2] = 1/gbar with equality iff h is proportional to E[v|s]: Theorem 9.6 in this example.")
    out = {}
    cases = {"u = 2, Gaussian k = N(2, 1)": (2.0, Mixing.gauss(2.0, 1.0)), "u = 2, two atoms v = 0, 2": (2.0, Mixing([0.0, 2.0], [0.5, 0.5])),
             "u = 0.25, two atoms v = 0, 1": (0.25, Mixing([0.0, 1.0], [0.5, 0.5]))}
    for name, (u, k) in cases.items():
        q = 1 + u * u
        sg = s_grid(k, q)
        ds = sg[1] - sg[0]
        Ev = post_mean(k, q, sg)
        gbar = float(np.sum(s_density(k, q, sg) * Ev * Ev) * ds) / q
        cstar = k.mean / k.var
        lin = lambda c: (lambda s_: s_ + c)
        row = dict(k=k, u=u, q=q, gbar=gbar, bound=1 / gbar, tls=avar_h(k, q, lin(0.0)), gross=q / k.mean ** 2, best_lin=avar_h(k, q, lin(cstar)), cstar=cstar, opt=avar_h(k, q, hfun(k, q)),
                   formula_tls=avar_class(0.0, q, k.mean, k.m2), formula_best=avar_class(cstar, q, k.mean, k.m2), printed_best=avar_printed(cstar, q, k.mean, k.m2))
        out[name] = row
        print(f"       {name}: E v = {k.mean:.3f}, Var v = {k.var:.3f}; efficient information gbar = (1/q) E[(E[v|s])^2] = {gbar:.5f}, bound 1/gbar = {row['bound']:.5f}")
        print(f"          N Var:  TLS (c = 0) {row['tls']:.5f} (corrected (9.103): {row['formula_tls']:.5f});  gross average {row['gross']:.5f};  best linear c = vbar/Var v = {cstar:.4f}: {row['best_lin']:.5f} (corrected (9.103): {row['formula_best']:.5f}; as printed: {row['printed_best']:.5f});"
              f"  optimal h = E[v|s]: {row['opt']:.5f}")
        print(f"          efficiency (bound/N Var): TLS = MLE {row['bound'] / row['tls']:.4f}, gross average {row['bound'] / row['gross']:.4f}, best linear {row['bound'] / row['best_lin']:.4f}")
    u = 2.0
    q = 1 + u * u
    g = out["u = 2, Gaussian k = N(2, 1)"]
    print(f"       Gaussian k: closed form gbar = (1/q)[mu^2 + q tau^4/(q tau^2 + 1)] = {(4 + q / (q + 1)) / q:.5f}; E[v|s] is LINEAR in s, so the best linear estimating function (9.105) is the efficient one: {g['formula_best']:.5f} = bound {g['bound']:.5f};")
    print(f"       the printed (9.103) at the book's own optimum (9.105) is {g['printed_best']:.5f}, BELOW the bound: impossible, so the printed formula is wrong (the corrected one meets the bound exactly)")
    # crossover TLS vs gross average
    m1 = 2.0
    Vx = (-q * m1 ** 2 + math.sqrt(q * q * m1 ** 4 + 4 * q * m1 ** 2)) / (2 * q)
    print(f"   TLS against gross average (m1 = {m1}, q = {q}): TLS = (q m2 + 1)/m2^2 and gross = q/m1^2 are equal when q V (m1^2 + V) = m1^2, i.e. V = {Vx:.5f} (spread tau = {math.sqrt(Vx):.4f}); for V below that the gross average wins, above it TLS:")
    rows = []
    for tau in (0.2, 0.4369, 0.6, 1.0, 2.0):
        V = tau * tau
        m2 = m1 * m1 + V
        rows.append((tau, avar_class(0.0, q, m1, m2), q / m1 ** 2, avar_class(m1 / V, q, m1, m2)))
        print(f"      tau = {tau:<6}: TLS {rows[-1][1]:.4f}   gross {rows[-1][2]:.4f}   best linear (c = {m1 / V:.2f}) {rows[-1][3]:.4f}")
    STORE["prop_bound"] = out
    STORE["prop_cross"] = dict(Vx=Vx, rows=rows, m1=m1, q=q)
    # Monte Carlo for the two-atom law: TLS, best linear and the optimal nonlinear h
    kk = cases["u = 2, two atoms v = 0, 2"][1]
    assert kk.v[0] == 0.0
    N = 400
    cst = kk.mean / kk.var
    rng = np.random.default_rng(11)

    def roots_quadratic(Sx, Sy, Sxx, Syy, Sxy, c):
        gross = Sy / Sx
        a_, b_, c0 = -Sxy, Syy - Sxx - c * Sx, Sxy + c * Sy
        d = np.sqrt(b_ * b_ - 4 * a_ * c0)
        r1, r2 = (-b_ + d) / (2 * a_), (-b_ - d) / (2 * a_)
        return np.where(np.abs(r1 - gross) < np.abs(r2 - gross), r1, r2)
    R1 = 60_000
    est_t = np.empty(R1); est_b = np.empty(R1)
    for st in range(0, R1, 1000):
        r = min(1000, R1 - st)
        vv = kk.v[1] * (rng.random((r, N)) < kk.p[1])
        x = vv + rng.normal(size=(r, N)); y = u * vv + rng.normal(size=(r, N))
        sums = (x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1), (x * y).sum(1))
        est_t[st:st + r] = roots_quadratic(*sums, 0.0); est_b[st:st + r] = roots_quadratic(*sums, cst)
    R2 = 12_000
    ugr = np.linspace(1.5, 2.5, 21)
    est_o = np.empty(R2)
    ratio01 = kk.p[0] / kk.p[1]
    for st in range(0, R2, 200):
        r = min(200, R2 - st)
        vv = kk.v[1] * (rng.random((r, N)) < kk.p[1])
        x = vv + rng.normal(size=(r, N)); y = u * vv + rng.normal(size=(r, N))
        Fm = np.empty((r, len(ugr)))
        for j, uu in enumerate(ugr):
            qq = 1 + uu * uu
            hh = kk.v[1] / (1 + ratio01 * np.exp(-kk.v[1] * (x + uu * y) + qq * kk.v[1] ** 2 / 2))           # E[v | s] for the atoms {0, v1}: a logistic function of s
            Fm[:, j] = ((y - uu * x) * hh).sum(1)
        cross = Fm[:, :-1] * Fm[:, 1:] < 0
        dist = np.abs(0.5 * (ugr[:-1] + ugr[1:]) - u)[None, :] + np.where(cross, 0.0, 1e9)
        ix = dist.argmin(1)
        r_ = np.arange(r)
        s0, s1 = Fm[r_, ix], Fm[r_, ix + 1]
        est_o[st:st + r] = ugr[ix] + (ugr[ix + 1] - ugr[ix]) * s0 / (s0 - s1)
    o = out["u = 2, two atoms v = 0, 2"]
    print(f"   Monte Carlo, two atoms v in {{0, 2}} (half the specimens are 'empty'), N = {N}: N Var = TLS {N * est_t.var():.3f} +- {N * est_t.var() * math.sqrt(2 / R1):.3f} (theory {o['tls']:.3f}) and best linear {N * est_b.var():.3f} +- {N * est_b.var() * math.sqrt(2 / R1):.3f} ({o['best_lin']:.3f}) from {R1} replications;"
          f" optimal E[v|s] {N * est_o.var():.3f} +- {N * est_o.var() * math.sqrt(2 / R2):.3f} ({o['bound']:.3f}) from {R2} replications")
    STORE["prop_mc"] = (N * est_t.var(), N * est_b.var(), N * est_o.var())
    # misspecified k_1 in l_E(., k_1), true k_0 = two atoms {0, 2}
    k0 = kk
    print("   Using l_E(., k1) built from a WRONG mixing law k1 when the truth is k0 = two atoms {0, 2} (Theorem 9.6: 'works well even for an approximate k1'): N Var, and efficiency = bound / N Var:")
    cand = [("k1 = k0 (optimal)", hfun(k0, q)),
            ("k1 Gaussian, same mean and variance (the best linear h)", hfun(Mixing.gauss(k0.mean, math.sqrt(k0.var)), q)),
            ("k1 Gaussian N(2, 1) (wrong mean)", hfun(Mixing.gauss(2.0, 1.0), q)),
            ("k1 Gaussian N(0.2, 0.5^2)", hfun(Mixing.gauss(0.2, 0.5), q)),
            ("k1 point mass, h = const (the gross average)", lambda s_: np.ones_like(s_)),
            ("h(s) = s (TLS)", lambda s_: s_)]
    tab = []
    for name, h_ in cand:
        av = avar_h(k0, q, h_)
        tab.append((name, av, o["bound"] / av))
        print(f"      {name:58s} {av:.4f}   efficiency {o['bound'] / av:.4f}")
    STORE["prop_mis"] = tab


def check_common_mean():
    print("   (b) Scale problem 2 (9.116)-(9.122): N scales, scale i has its own unknown precision v_i = 1/sigma_i^2, each measures the same specimen m times; u = the common weight.")
    print("       f = S1 h(s), S1 = sum_j (x_j - u), s = -Q/2, Q = sum_j (x_j - u)^2; given Q the direction of (x_j - u) is uniform on the sphere, so E[S1 | Q] = 0 and E[S1^2 | Q] = Q;")
    print("       hence E f^2 = E[Q h^2], E f' = -m E[h] - 2 E[Q dh/dQ] and N Var = E f^2/(E f')^2 per box. (The printed (9.118) has u^2 where m u^2 is needed, and (9.120)-(9.121) use the box MEAN for x-bar while (9.110) defined a sum.)")
    m, a, b = 4, 3.0, 3.0
    xg, wg = np.polynomial.legendre.leggauss(600)
    t = 0.5 * (xg + 1); wt = 0.5 * wg
    dens = math.gamma(m / 2 + a) / (math.gamma(m / 2) * math.gamma(a)) * t ** (m / 2 - 1) * (1 - t) ** (a - 1)
    Q = 2 * b * t / (1 - t)
    EQ = lambda g: float(np.sum(wt * dens * g))
    print(f"       Precisions v ~ Gamma(shape {a}, rate {b}) (mean {a / b:.2f}, E[1/v] = b/(a-1) = {b / (a - 1):.2f}); m = {m}; Q = (2b/..) beta-prime law, quadrature check: total mass {EQ(1.0):.8f}, E[Q] = m E[1/v] = {EQ(Q):.6f} vs {m * b / (a - 1):.6f}")
    cands = {
        "plain mean of box means (h = 1)": (lambda Q_: np.ones_like(Q_), lambda Q_: np.zeros_like(Q_)),
        "joint MLE (h = 1/Q)": (lambda Q_: 1 / Q_, lambda Q_: -1 / Q_ ** 2),
        "optimal h = E[v|s] = (a + m/2)/(b + Q/2)": (lambda Q_: (a + m / 2) / (b + Q_ / 2), lambda Q_: -(a + m / 2) / (2 * (b + Q_ / 2) ** 2)),
    }
    res = {}
    for name, (h, hq) in cands.items():
        Ef2 = EQ(Q * h(Q) ** 2)
        Efp = -(m * EQ(h(Q)) + 2 * EQ(Q * hq(Q)))
        res[name] = (Ef2, Efp, Ef2 / Efp ** 2)
        print(f"       {name:44s} E f^2 = {Ef2:.5f}, E f' = {Efp:+.5f}, N Var = {Ef2 / Efp ** 2:.5f}")
    o = res["optimal h = E[v|s] = (a + m/2)/(b + Q/2)"]
    print(f"       for the optimal h: E f' = {o[1]:+.5f} = -E f^2 = {-o[0]:+.5f} (as (9.68) requires for l_E), so N Var = 1/E f^2 = 1/gbar = {1 / o[0]:.5f}")
    rng = np.random.default_rng(12)
    N, R, u = 400, 2000, 0.5
    est = {k: np.empty(R) for k in cands}
    for st in range(0, R, 250):
        r = min(250, R - st)
        v = rng.gamma(a, 1 / b, (r, N))
        xb = u + rng.normal(size=(r, N)) / np.sqrt(m * v)
        W = rng.chisquare(m - 1, (r, N)) / v
        for name, (h, hq) in cands.items():
            uh = xb.mean(1)
            for _ in range(40 if name.startswith(("joint", "optimal")) else 1):
                Qi = W + m * (xb - uh[:, None]) ** 2
                w = h(Qi)
                uh = (w * xb).sum(1) / w.sum(1)
            est[name][st:st + r] = uh
    print(f"       Monte Carlo (N = {N} scales, {R} replications; weighted means solved as the fixed point (9.121) from the plain mean; true u = {u}):")
    for name in cands:
        mc = N * est[name].var()
        print(f"         {name:44s} N Var = {mc:.4f} +- {mc * math.sqrt(2 / R):.4f}  (theory {res[name][2]:.4f}), mean {est[name].mean():.4f}")
    STORE["common_mean"] = {k: (res[k][2], N * est[k].var()) for k in cands}
    print("       so the 'obvious' weighting by the joint-MLE weights 1/Q is consistent but beats the plain mean only modestly, and the optimum depends on the unknown law of the precisions (here a Gamma prior gives (a + m/2)/(b + Q/2)).")


def beta_D(rng, kappa, size):
    """D = T1/(T1 + T2) for two gamma intervals with the same shape and any common rate: exactly Beta(kappa, kappa)"""
    return rng.beta(kappa, kappa, size)


class KTable:
    """Inverse of a decreasing function of the shape kappa, tabulated on a logarithmic grid (vectorised root finding)."""

    def __init__(self, fn):
        self.k = np.exp(np.linspace(math.log(1e-2), math.log(1e3), 6001))
        self.g = np.array([fn(float(x)) for x in self.k])

    def inv(self, target):
        t = np.log(np.maximum(np.asarray(target, float), 1e-300))
        return np.exp(np.interp(t, np.log(self.g)[::-1], np.log(self.k)[::-1]))


G_eff = lambda k: digamma(2 * k) - digamma(k) - math.log(2)            # E[-1/2 log(4 D (1-D))] under Gamma shape k, D = T1/(T1+T2): the efficient estimating equation sets S = G_eff
G_mle = lambda k: math.log(k) - digamma(k)                              # what the joint MLE sets S + log 2... (see check_neuron) equal to
TAB_EFF = KTable(G_eff)
TAB_MLE = KTable(G_mle)


def check_neuron():
    print("   (c) Temporal firing pattern (9.123)-(9.134): gamma intervals with shape kappa (to estimate), rate v_k changing from box to box; m = 2 intervals per box.")
    kap = 2.0
    rng = np.random.default_rng(13)
    n = 400_000
    print("       The efficient score (9.129) for m = 2 is log T1 + log T2 - 2 log(T1 + T2) + 2 phi(2k) - 2 phi(k); it depends on the intervals only through D = T1/(T1+T2) ~ Beta(kappa, kappa), whatever v is.")
    print("       Mean of the score over 400000 boxes at the true kappa = 2, for three rates v, with phi = digamma (log Gamma)' and with phi = Gamma' as printed in (9.130):")
    for vv in (0.5, 3.0, 40.0):
        T1 = rng.gamma(kap, 1 / (kap * vv), n); T2 = rng.gamma(kap, 1 / (kap * vv), n)
        base = np.log(T1) + np.log(T2) - 2 * np.log(T1 + T2)
        ok = base + 2 * digamma(2 * kap) - 2 * digamma(kap)
        gam = lambda x: math.gamma(x) * digamma(x)
        pr = base + 2 * gam(2 * kap) - 2 * gam(kap)
        print(f"          v = {vv:<5}: digamma {ok.mean():+.4f} +- {ok.std() / math.sqrt(n):.4f};   Gamma' (as printed) {pr.mean():+.3f}   (exact: 0 and {2 * (digamma(kap) - digamma(2 * kap)) + 2 * math.gamma(2 * kap) * digamma(2 * kap) - 2 * math.gamma(kap) * digamma(kap):+.3f})")
    # (9.129) is the kappa-derivative of the log-density of the scale-free statistic D = T1/(T1+T2) ~ Beta(kappa, kappa)
    lgam = math.lgamma
    logbeta = lambda d, k: lgam(2 * k) - 2 * lgam(k) + (k - 1) * (np.log(d) + np.log(1 - d))
    Dt = rng.uniform(0.02, 0.98, 6)
    h_ = 1e-5
    fdv = (logbeta(Dt, kap + h_) - logbeta(Dt, kap - h_)) / (2 * h_)
    u_E = np.log(Dt) + np.log(1 - Dt) + 2 * digamma(2 * kap) - 2 * digamma(kap)
    print(f"       (9.129) with T1 = D, T2 = 1 - D is exactly the kappa-derivative of log Beta(D; kappa, kappa): largest difference {np.abs(fdv - u_E).max():.1e} at six values of D;"
          f" so the estimating-function solution is the maximum likelihood estimator based on D alone (a marginal likelihood), and its information is that of Beta(kappa, kappa): 2 trigamma(k) - 4 trigamma(2k) = {2 * trigamma(kap) - 4 * trigamma(2 * kap):.5f} at kappa = {kap}")
    # the three estimators and the naive pooled one, on a train whose rate changes in boxes
    nb = 200_000
    v = np.exp(rng.normal(0.0, 0.8, nb))
    T1 = rng.gamma(kap, 1 / (kap * v)); T2 = rng.gamma(kap, 1 / (kap * v))
    D = T1 / (T1 + T2)
    q_ = 4 * D * (1 - D)
    Sbar = float((-0.5 * np.log(q_)).mean())
    LVbar = float((3 * (1 - q_)).mean())
    k_S = float(TAB_EFF.inv(Sbar))
    k_LV = 0.5 * (3 / LVbar - 1)
    k_mle = float(TAB_MLE.inv(Sbar))
    Tall = np.concatenate([T1, T2])
    k_naive = float(TAB_MLE.inv(math.log(Tall.mean()) - float(np.log(Tall).mean())))
    print(f"       {nb} boxes, true kappa = {kap}, rates v_k = exp(N(0, 0.8^2)) (changing by a factor of about 5 between boxes):")
    print(f"         efficient estimating equation (9.131), i.e. S of (9.132) = G(kappa) = phi(2k) - phi(k) - log 2 : kappa_hat = {k_S:.4f}")
    print(f"         L_V of (9.133): E[L_V] = 3/(2 kappa + 1) gives kappa_hat = {k_LV:.4f}")
    print(f"         joint MLE of (kappa, v_1, ..., v_n): sets S = log kappa - phi(kappa) instead; kappa_hat = {k_mle:.4f}  (about 2 kappa: the plug-in v_k_hat bias, the Neyman-Scott inconsistency)")
    print(f"         naive: one common gamma law for all intervals (ignores that the rate changes): kappa_hat = {k_naive:.4f}  (rate changes read as irregularity)")
    sv = 0.8
    nl = float(TAB_MLE.inv(G_mle(kap) + sv ** 2 / 2))
    print(f"         exact limits: joint MLE kappa* = {float(TAB_MLE.inv(G_eff(kap))):.4f} (G_mle(kappa*) = G_eff(kappa0)); naive pooled estimate solves log k - phi(k) = log k0 - phi(k0) + sigma_v^2/2, which gives {nl:.4f}")
    STORE["neuron_est"] = dict(kap=kap, S=k_S, LV=k_LV, mle=k_mle, naive=k_naive)
    print("       limit of the joint MLE: kappa* solves log(2k*) - phi(k*) = phi(2k0) - phi(k0):")
    lims = []
    for k0 in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        ks = float(TAB_MLE.inv(digamma(2 * k0) - digamma(k0) - math.log(2)))
        lims.append((k0, ks))
        print(f"          kappa0 = {k0:<5}: kappa* = {ks:.4f}  (ratio {ks / k0:.4f})")
    STORE["neuron_lim"] = lims
    # asymptotic variances: exact sampling of D
    print("       Asymptotic variance per box of the S-based and L_V-based estimators (9.35), against the efficient bound 1/gbar, gbar = 2 trigamma(k) - 4 trigamma(2k), and the information 2 (trigamma(k) - 1/k) if the rate were one known constant:")
    rows = []
    for k0 in (1.0, 2.0, 5.0):
        gbar = 2 * trigamma(k0) - 4 * trigamma(2 * k0)
        gfull = 2 * (trigamma(k0) - 1 / k0)
        avS = 1 / gbar
        avL = k0 * (2 * k0 + 1) ** 2 / (2 * k0 + 3)
        nbx, R = 500, 40000
        nbx0 = nbx
        D_ = beta_D(rng, k0, (R, nbx))
        qq = 4 * D_ * (1 - D_)
        kS = TAB_EFF.inv((-0.5 * np.log(qq)).mean(1))
        kL = 0.5 * (3 / (3 * (1 - qq).mean(1)) - 1)
        rows.append((k0, avS, nbx * kS.var(), avL, nbx * kL.var(), gbar / gfull))
        print(f"          kappa = {k0}: gbar = {gbar:.5f} (known-rate information {gfull:.5f}, ratio {gbar / gfull:.4f});  S: theory {avS:.4f}, simulated {nbx * kS.var():.4f};  L_V: theory {avL:.4f}, simulated {nbx * kL.var():.4f};  efficiency of L_V {avS / avL:.4f}")
    k0 = 2.0
    gbar = 2 * trigamma(k0) - 4 * trigamma(2 * k0)
    fin = []
    for nbx, R in ((250, 20_000), (1000, 8_000)):
        D_ = beta_D(rng, k0, (R, nbx))
        kS = TAB_EFF.inv((-0.5 * np.log(4 * D_ * (1 - D_))).mean(1))
        fin.append((nbx, nbx * kS.var() * gbar))
    worst = max(max(abs(r[2] / r[1] - 1), abs(r[4] / r[3] - 1)) for r in rows)
    print(f"          the six simulated variances above are within {worst:.1%} of the asymptotic ones at n = {nbx0} boxes (s.e. about 0.7-1%); finite-n check at kappa = 2 for S (simulated / asymptotic): n = 250: {fin[0][1]:.4f}, n = 500: {rows[1][2] / rows[1][1]:.4f}, n = 1000: {fin[1][1]:.4f} (consistent with an O(1/n) correction; the s.e. is about 1-1.5%)")
    STORE["neuron_var"] = rows
    # overlapping pairs, rate piecewise constant on blocks
    print("       Consecutive overlapping pairs (T_i, T_(i+1)) as in (9.132) against disjoint boxes, when the rate is constant on blocks of B intervals (kappa = 2, 400000 intervals, rates exp(N(0, 0.8^2))):")
    nint = 400_000
    seq = []
    for B in (2, 10, 100):
        nbk = nint // B
        vb = np.repeat(np.exp(rng.normal(0.0, 0.8, nbk)), B)
        T = rng.gamma(kap, 1 / (kap * vb))
        Dd = T[0::2] / (T[0::2] + T[1::2])
        Do = T[:-1] / (T[:-1] + T[1:])
        kd = float(TAB_EFF.inv(float((-0.5 * np.log(4 * Dd * (1 - Dd))).mean())))
        ko = float(TAB_EFF.inv(float((-0.5 * np.log(4 * Do * (1 - Do))).mean())))
        seq.append((B, kd, ko))
        print(f"          B = {B:<4d} disjoint aligned pairs: kappa_hat = {kd:.4f};  overlapping pairs (9.132): kappa_hat = {ko:.4f}   (fraction of overlapping pairs that straddle a change: {1 / B:.2f})")
    STORE["neuron_overlap"] = seq
    # robustness
    print("       Robustness ('L_V may be more robust'): replace a fraction eps of the boxes by an artefact with D = 1e-6 (two spikes almost on top of each other), kappa = 2, 400000 boxes:")
    rob = []
    Dbase = beta_D(rng, kap, 400_000)
    Ubad = rng.random(400_000)                                                          # the same artefact boxes are used for every eps (nested), so the shifts are paired
    for eps in (0.0, 0.001, 0.005, 0.02):
        D_ = np.where(Ubad < eps, 1e-6, Dbase)
        qq = 4 * D_ * (1 - D_)
        kS = float(TAB_EFF.inv(float((-0.5 * np.log(qq)).mean())))
        kL = 0.5 * (3 / float((3 * (1 - qq)).mean()) - 1)
        rob.append((eps, kS, kL))
        print(f"          eps = {eps:<6}: S-based kappa_hat = {kS:.4f}, L_V-based kappa_hat = {kL:.4f}")
    STORE["neuron_rob"] = rob
    shS, shL = rob[0][1] - rob[1][1], rob[0][2] - rob[1][2]
    print(f"       shifts caused by eps = 0.001: S moves by {shS:.4f}, L_V by {shL:.4f} (ratio {shS / shL:.1f}); by eps = 0.005: {rob[0][1] - rob[2][1]:.4f} and {rob[0][2] - rob[2][2]:.4f} (ratio {(rob[0][1] - rob[2][1]) / (rob[0][2] - rob[2][2]):.1f})")
    STORE["neuron_shift_ratio"] = shS / shL
    print("       the influence of one artefact box is -1/2 log(4 D(1-D)) = 6.2 (unbounded as D -> 0) for S, but at most 3 for L_V: bounded influence.")


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



def log_ticks(lo, hi):
    sup = {4: "10⁴", 5: "10⁵", 6: "10⁶", 7: "10⁷"}
    return list(range(lo, hi + 1)), {e: (str(10 ** e) if e < 4 else sup[e]) for e in range(lo, hi + 1)}


def fig_efficient_score(path):
    pc = STORE["prop_common"]; g = pc["g"]
    b = []
    W, H = 800, 410
    # panel A: the tangent plane in the Fisher metric
    x0, y0 = 64, 56
    b.append(f'<text class="hd" x="{x0}" y="{y0 - 12}">The efficient score at (u, v) = ({pc["u"]}, {pc["v"]})</text>')
    ox, oy, sc = 118, 290, 98.0
    lu, lv, lb = math.sqrt(g[0, 0]), math.sqrt(g[1, 1]), math.sqrt(pc["gbar"])
    ang = math.acos(g[0, 1] / (lu * lv))
    eu = (ox + sc * lu * math.sin(ang), oy - sc * lu * math.cos(ang))
    ev = (ox, oy - sc * lv)
    foot = (ox, oy - sc * lu * math.cos(ang))
    tip = (eu[0], oy)
    b.append(f'<rect x="{x0}" y="{y0}" width="320" height="250" fill="none" class="gd"/>')
    b.append(f'<line class="dash s0" x1="{ox}" y1="{oy}" x2="{tip[0] + 30:.1f}" y2="{oy}"/>')
    b.append(f'<line class="dash s0" x1="{eu[0]:.1f}" y1="{eu[1]:.1f}" x2="{tip[0]:.1f}" y2="{tip[1]:.1f}"/>')
    b.append(f'<line class="dash s0" x1="{eu[0]:.1f}" y1="{eu[1]:.1f}" x2="{foot[0]:.1f}" y2="{foot[1]:.1f}"/>')
    arrow(b, ox, oy, ev[0], ev[1], 1, 2.4)
    arrow(b, ox, oy, eu[0], eu[1], 2, 2.4)
    arrow(b, ox, oy, tip[0], tip[1], 3, 2.8)
    b.append(f'<path class="thin sk" d="M {ox + 9} {oy} L {ox + 9} {oy - 9} L {ox} {oy - 9}"/>')
    b.append(f'<text class="v" x="{ev[0]:.1f}" y="{ev[1] - 8:.1f}" text-anchor="middle">{T("e_v")}  (length {lv:.3f})</text>')
    b.append(f'<text class="v" x="{eu[0] + 8:.1f}" y="{eu[1] + 2:.1f}">{T("e_u")}  (length {lu:.3f})</text>')
    b.append(f'<text class="v" x="{tip[0] + 8:.1f}" y="{oy - 8:.1f}">{T("ē_u")}  (length {lb:.3f})</text>')
    b.append(f'<text class="sm" x="{ox + 12}" y="{oy - 70}">{math.degrees(ang):.1f}°</text>')
    b.append(f'<text class="sm" x="{ox - 8}" y="{foot[1] + 4:.1f}" text-anchor="end">{g[0, 1] / lv:.3f}</text>')
    b.append(f'<text class="sm" x="{x0 + 8}" y="{y0 + 16}">scores drawn with the Fisher inner product;</text>')
    b.append(f'<text class="sm" x="{x0 + 8}" y="{y0 + 31}">the dashed drop is the part of the u-score along v</text>')
    # panel B: holonomy
    hol = STORE["holonomy"]
    P = Panel(b, 470, 56, 300, 250, (0, 1.3), (-8.6, 0.6))
    P.frame([0, 0.25, 0.5, 0.75, 1], [-8, -6, -4, -2, 0], "angle round the loop / 2π", "v(θ) − v(0)", "Horizontal lift of v round a loop in the u-plane", True)
    P.line([0, 1.3], [0, 0], "dash s0")
    for rho, cls in ((0.5, "s1"), (1.0, "s2"), (1.5, "s3")):
        ts, vs, gap = hol[rho]
        P.line(ts / (2 * math.pi), vs - vs[0], "ln " + cls)
        P.dot(1.0, gap, "f" + cls[1], 4.2)
        P.text(1.04, gap, f"ρ = {rho}: {gap:.2f}".replace("-", "−"), "sm", "start", 0, 4)
    P.text(1.28, 0.0, "a closed loop would end at 0", "sm", "end", 0, -6)
    note(b, 64, 372, ["Left: the efficient score is the part of the u-score orthogonal to the nuisance score; its length is the square root of the efficient", "information. Right: for two parameters of interest the orthogonal nuisance coordinate does not exist, the lift does not close."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Efficient score and the failure of orthogonal nuisance parameters",
        "Left: in the tangent plane with the Fisher inner product, the u-score e_u, the nuisance score e_v and the efficient score, the component of e_u orthogonal to e_v, for the coefficient-of-proportionality model at u = 1.5, v = 2. Right: for a model with two parameters of interest and one nuisance parameter, lifting a loop of the parameter plane horizontally (orthogonally to the nuisance direction) does not close: v ends lower by 0.38 to 7.8 depending on the radius of the loop, so no orthogonal nuisance coordinate exists.", b))


def fig_neyman_scott(path):
    b = []
    W, H = 800, 405
    seq = STORE["boxes_seq"]
    P1 = Panel(b, 70, 56, 320, 250, (0.7, 5.3), (0, 1.7))
    tk, tl = log_ticks(1, 5)
    P1.frame(tk, [0, 0.5, 1, 1.5], "N, number of boxes (m = 2)", "estimate of σ² / σ²", "Gaussian variance with one mean per box", True, tl)
    P1.line([0.7, 5.3], [0.5, 0.5], "dash s2"); P1.line([0.7, 5.3], [1.0, 1.0], "dash s0")
    for N, ml, un, sdm, sdu in seq:
        x = math.log10(N)
        for val, sd, cls in ((ml / 2, sdm / 2, "f2"), (un / 2, sdu / 2, "f1")):
            b.append(f'<line class="thin s0" x1="{P1.X(x):.1f}" y1="{P1.Y(val - sd):.1f}" x2="{P1.X(x):.1f}" y2="{P1.Y(val + sd):.1f}"/>')
            P1.dot(x, val, cls, 4)
    P1.line([math.log10(r[0]) for r in seq], [r[1] / 2 for r in seq], "thin s2"); P1.line([math.log10(r[0]) for r in seq], [r[2] / 2 for r in seq], "thin s1")
    P1.text(5.25, 1.0, "within-box estimator (9.114) → σ²", "sm", "end", 0, -9); P1.text(5.25, 0.5, "joint MLE → σ²/2", "sm", "end", 0, -9)
    ms = STORE["matched_seq"]
    P2 = Panel(b, 470, 56, 300, 250, (2.7, 7.3), (0, 5))
    tk2, tl2 = log_ticks(3, 7)
    P2.frame(tk2, [0, 1, 2, 3, 4, 5], "N, number of pairs", "estimate of the odds ratio u", "Matched pairs, u = 2", True, tl2)
    P2.line([2.7, 7.3], [4, 4], "dash s2"); P2.line([2.7, 7.3], [2, 2], "dash s0")
    P2.line([math.log10(r[0]) for r in ms], [r[2] for r in ms], "thin s2"); P2.line([math.log10(r[0]) for r in ms], [r[1] for r in ms], "thin s1")
    for N, c, mle in ms:
        P2.dot(math.log10(N), mle, "f2", 4); P2.dot(math.log10(N), c, "f1", 4)
    P2.text(7.25, 4.0, "u² = 4: limit of the MLE", "sm", "end", 0, -7); P2.text(7.25, 2.0, "u = 2: conditional estimator", "sm", "end", 0, -7)
    note(b, 70, 372, ["Left: 4000 simulated data sets per N (exact sampling of the sum of squares); dots are means, bars ± 1 standard deviation of one data set.", "Right: one data set per N. In both problems the joint MLE converges, to the wrong value; the estimating-function solution to the right one."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The Neyman-Scott MLE settles on the wrong value",
        "Left: with N boxes of two Gaussian measurements and one unknown mean per box, the maximum likelihood estimate of the variance converges to half the true variance, while the within-box estimator of the book converges to the truth. Right: for matched pairs of binary outcomes with a free log-odds per pair, the unconditional maximum likelihood estimate of the odds ratio converges to its square, 4 instead of 2, while the ratio of discordant pair counts converges to 2.", b))


def fig_five_solutions(path):
    sm = STORE["five_sample"]; sl = sm["slopes"]
    b = []
    W, H = 800, 405
    P1 = Panel(b, 70, 56, 320, 250, (-1, 6.5), (-2, 14))
    P1.frame([0, 2, 4, 6], [0, 4, 8, 12], "x", "y", "Forty specimens, v ~ N(2, 1), u = 2", True)
    xs = np.array([-1, 6.5])
    for slope, cls, dash in ((2.0, "sk", "ln"), (sl["LS"], "s1", "ln"), (sl["avg"], "s2", "dash"), (sl["gross"], "s3", "ln"), (sl["TLS"], "s4", "dash")):
        P1.line(xs, slope * xs, f"{dash} {cls}" if cls != "sk" else "ln sk")
    for xx, yy in zip(sm["x"], sm["y"]):
        P1.dot(xx, yy, "f0", 2.6)
    legend_col(b, 80, 76, [("sk", "true y = 2x"), ("s1", f"least squares {sl['LS']:.2f}"), ("s2", f"averaging of ratios {sl['avg']:.2f}"), ("s3", f"gross average {sl['gross']:.2f}"), ("s4", f"TLS = MLE {sl['TLS']:.2f}")])
    # panel B: cumulative averages of y_i / x_i
    rng = np.random.default_rng(21)
    checkpoints = np.unique(np.round(np.logspace(1, 6, 60)).astype(int))
    P2 = Panel(b, 470, 56, 300, 250, (1, 6), (-8, 8))
    tk, tl = log_ticks(1, 6)
    P2.frame(tk, [-8, -4, 0, 4, 8], "n, pairs averaged", "running estimate of u", "Running average of y/x and gross average", True, tl)
    P2.line([1, 6], [2, 2], "dash s0")
    cols = ["s2", "s2", "s2", "s2", "s2", "s2"]
    for r in range(6):
        tot, cnt, out = 0.0, 0, []
        sx = sy = 0.0
        outg = []
        pos = 0
        while cnt < 1_000_000:
            n = min(200_000, 1_000_000 - cnt)
            v = rng.normal(2.0, 1.0, n); x = v + rng.normal(size=n); y = 2 * v + rng.normal(size=n)
            cr, cx, cy = np.cumsum(y / x), np.cumsum(x), np.cumsum(y)
            while pos < len(checkpoints) and checkpoints[pos] <= cnt + n:
                j = checkpoints[pos] - cnt - 1
                out.append(((tot + cr[j]) / checkpoints[pos])); outg.append((sy + cy[j]) / (sx + cx[j]))
                pos += 1
            tot += cr[-1]; sx += cx[-1]; sy += cy[-1]; cnt += n
        P2.line(np.log10(checkpoints), np.clip(np.array(out), -7.9, 7.9), "thin s2")
        if r == 0:
            P2.line(np.log10(checkpoints), np.array(outg), "ln s3")
    P2.text(1.05, 2.0, "u = 2", "sm", "start", 0, -6)
    P2.text(1.05, 7.2, "orange: six runs of the averaging (clipped at ±8)", "sm", "start"); P2.text(1.05, 5.9, "green: one run of the gross average", "sm", "start")
    note(b, 70, 372, ["Left: one sample; the fits through the origin. The averaging line is far off because a few ratios y/x with x near 0 dominate.", "Right: the running average of y/x keeps jumping however many pairs are used; the gross average settles at u."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Five solutions of the coefficient-of-proportionality problem",
        "Left: forty simulated specimens with the true line y = 2x and four fitted lines through the origin: least squares is pulled down, the average of the ratios y over x is far off, the gross average and total least squares are close to the truth. Right: running average of the ratios y over x for six independent runs up to a million pairs, which never settles, against the running gross average, which converges to 2.", b))


def fig_estimating_class(path):
    par = STORE["class_par"]; rows = STORE["class_rows"]
    q, m1, m2 = par["q"], par["m1"], par["m2"]
    b = []
    W, H = 800, 424
    P1 = Panel(b, 70, 56, 320, 250, (0, 20), (0.8, 1.3))
    P1.frame([0, 5, 10, 15, 20], [0.8, 0.9, 1.0, 1.1, 1.2, 1.3], "c in h(s) = s + c", "N · Var of the root", "The class (9.97), u = 2, N = 500", True)
    cs = np.linspace(0, 20, 200)
    P1.line(cs, [avar_class(c, q, m1, m2) for c in cs], "ln s1")
    P1.line(cs, [avar_printed(c, q, m1, m2) for c in cs], "dash s2")
    P1.line([0, 20], [q / m1 ** 2] * 2, "dash s0")
    P1.line([0, 20], [avar_class(par["cstar"], q, m1, m2)] * 2, "dot s1")
    for c, mc, th, pr in rows:
        if c <= 12:
            P1.dot(c, mc, "f1", 3.8)
            b.append(f'<line class="thin s0" x1="{P1.X(c):.1f}" y1="{P1.Y(mc * (1 - 2 * math.sqrt(2 / 40000))):.1f}" x2="{P1.X(c):.1f}" y2="{P1.Y(mc * (1 + 2 * math.sqrt(2 / 40000))):.1f}"/>')
    P1.dot(par["cstar"], avar_class(par["cstar"], q, m1, m2), "f3", 4.8)
    P1.text(par["cstar"] + 0.6, 1.07, "ĉ of (9.105)", "sm", "start")
    legend_col(b, 236, 242, [("s1", "corrected (9.103)"), ("dash s2", "(9.103) as printed*"), ("dash s0", "gross average (c = ∞)"), ("dot s1", "efficiency bound")])
    # panel B: efficiency against the spread of k (Gaussian k = N(2, tau^2)); best linear = efficient bound
    P2 = Panel(b, 470, 56, 300, 250, (0, 2), (0.5, 1.03))
    P2.frame([0, 0.5, 1, 1.5, 2], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "τ, spread of k = N(2, τ²)", "efficiency (bound / N Var)", "TLS or gross average?", True)
    taus = np.linspace(0.02, 2, 200)
    bound = np.array([avar_class(m1 / t ** 2, q, m1, m1 ** 2 + t ** 2) for t in taus])
    P2.line(taus, bound / np.array([avar_class(0.0, q, m1, m1 ** 2 + t ** 2) for t in taus]), "ln s1")
    P2.line(taus, bound / (q / m1 ** 2), "ln s3")
    P2.line([0, 2], [1, 1], "dash s0")
    xc = math.sqrt(STORE["prop_cross"]["Vx"])
    b.append(f'<line class="thin s0" x1="{P2.X(xc):.1f}" y1="{P2.y0}" x2="{P2.X(xc):.1f}" y2="{P2.y0 + P2.h}"/>')
    P2.text(xc + 0.04, 0.53, f"τ = {xc:.3f}", "sm", "start")
    legend_col(b, 560, 238, [("s1", "TLS (c = 0)"), ("s3", "gross average (c = ∞)"), ("dash s0", "best linear c = E v / Var v")])
    note(b, 70, 372, ["Left: dots are Monte Carlo (40000 samples, bars ± 2 s.e.); solid: (9.103) with its outer bracket restored; dashed*: (9.103) as printed,",
                      f"* with the 1/N and the square of the denominator, evidently missing, put back; it lies below the efficiency bound for all c up to {STORE['printed_cross']:.1f}.",
                      "Right: with E v = 2, TLS beats the gross average once the spread of the specimen weights exceeds τ = 0.437."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The linear class of estimating functions",
        "Left: N times the variance of the root of the estimating equation with h(s) = s + c against c, for u = 2 and N = 500 specimens with weights spread as N(2,1): Monte Carlo dots agree with the corrected formula (9.103), whose minimum is at c = 2.005 as (9.105) says, and not with the formula as printed, which dips below the efficiency bound. Right: efficiency of total least squares (c = 0) and of the gross average (c = infinity) against the spread of the specimen weights, with a crossing at tau = 0.437.", b))


def fig_finite_model(path):
    geo = STORE["geo"]; rows = STORE["geo_var"]
    b = []
    W, H = 800, 405
    x0, y0, cell = 70, 66, 56
    b.append(f'<text class="hd" x="{x0}" y="{y0 - 22}">Efficient score of the 4 × 4 outcome table</text>')
    lE = geo["lE"]; mx = float(np.abs(lE).max())
    for i, (x_, y_) in enumerate(zip(geo["X"], geo["Y"])):
        cx, cy = x0 + 22 + x_ * cell, y0 + (3 - y_) * cell
        cls = "fillO" if lE[i] >= 0 else "fillB"
        b.append(f'<rect class="{cls}" style="opacity:{0.15 + 0.7 * abs(lE[i]) / mx:.2f}" x="{cx}" y="{cy}" width="{cell - 3}" height="{cell - 3}"/>')
        b.append(f'<text class="v" x="{cx + (cell - 3) / 2}" y="{cy + (cell - 3) / 2 + 4}" text-anchor="middle">{lE[i]:+.2f}</text>')
    for j in range(4):
        b.append(f'<text class="sm" x="{x0 + 22 + j * cell + (cell - 3) / 2}" y="{y0 + 4 * cell + 8}" text-anchor="middle">x = {j}</text>')
        b.append(f'<text class="sm" x="{x0 + 14}" y="{y0 + (3 - j) * cell + (cell - 3) / 2 + 4}" text-anchor="end">y = {j}</text>')
    b.append(f'<text class="lab" x="{x0 + 22 + 2 * cell}" y="{y0 + 4 * cell + 30}" text-anchor="middle">outcomes (x, y), u = 2</text>')
    P2 = Panel(b, 470, 56, 300, 250, (0, 1.6), (10, 50))
    P2.frame([0, 0.5, 1, 1.5], [10, 20, 30, 40, 50], "t, weight of the ancillary part a", "N · Var of the root", "Theorem 9.6: variance of l_E + t a", True)
    ts = np.linspace(0, 1.6, 100)
    gb = geo["gbar1"]
    P2.line(ts, (1 + ts ** 2) / gb, "ln s1")
    for (N, t, mc, th) in rows:
        P2.dot(t, mc, "f2" if N == 800 else "f3", 4.2)
    legend_col(b, 482, 82, [("s2", "Monte Carlo, N = 800"), ("s3", "Monte Carlo, N = 8000")])
    P2.text(0.55, 1 / gb + 0.5, "bound 1/ḡ = 13.52 at t = 0", "sm", "start", 0, 24)
    note(b, 70, 372, ["Left: l_E = (y − E[y|s])/u, with s = x + y; it is the same for every mixing law k (orange positive, blue negative).", "Right: any other estimating function adds an ancillary part a, orthogonal to l_E, and its variance grows with the square of it."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "An exact finite model for the geometry of estimating functions",
        "Left: the efficient score of the binomial-pairs model with m = 3 and odds ratio 2, shown on the four by four table of outcomes; it is the same vector for every mixing law of the nuisance parameter. Right: variance of the estimator from l_E plus t times an ancillary vector of the same norm: the parabola (1 plus t squared) over the efficient information from Theorem 9.6, with Monte Carlo points at N = 800 and N = 8000.", b))


def fig_neuron(path):
    b = []
    W, H = 800, 405
    P1 = Panel(b, 70, 56, 320, 250, (-0.4, 1.9), (0.8, 2.1))
    P1.frame([0, 1], [1.0, 1.5, 2.0], "shape κ₀ of the gamma intervals", "limit of the joint MLE / κ₀", "Joint MLE of κ and all box rates, m = 2", True, {0: "1", 1: "10"})
    ks = np.exp(np.linspace(math.log(0.4), math.log(80), 120))
    lim = [float(TAB_MLE.inv(digamma(2 * k0) - digamma(k0) - math.log(2))) / k0 for k0 in ks]
    P1.line(np.log10(ks), lim, "ln s2")
    P1.line([-0.4, 1.9], [2, 2], "dash s0"); P1.line([-0.4, 1.9], [1, 1], "dash s0")
    for k0, kst in STORE["neuron_lim"]:
        P1.dot(math.log10(k0), kst / k0, "f2", 3.8)
    P1.text(1.88, 2.0, "2", "sm", "end", 0, -6); P1.text(1.88, 1.0, "truth", "sm", "end", 0, -6)
    P2 = Panel(b, 470, 56, 300, 250, (-0.002, 0.0225), (1.0, 2.1))
    P2.frame([0, 0.005, 0.01, 0.015, 0.02], [1.0, 1.25, 1.5, 1.75, 2.0], "fraction ε of artefact boxes (D = 10⁻⁶)", "estimate of κ (true value 2)", "S against L_V: robustness", True)
    rob = STORE["neuron_rob"]
    P2.line([r[0] for r in rob], [r[1] for r in rob], "ln s1"); P2.line([r[0] for r in rob], [r[2] for r in rob], "ln s3")
    for e, kS, kL in rob:
        P2.dot(e, kS, "f1", 3.8); P2.dot(e, kL, "f3", 3.8)
    P2.line([-0.002, 0.0225], [2, 2], "dash s0")
    legend_col(b, 486, 262, [("s1", "S: (9.132), unbounded influence"), ("s3", "L_V: (9.133), bounded influence")])
    note(b, 70, 372, ["Left: the joint MLE tends to about 2κ₀ for large κ₀ (dots: the exact limits); the efficient estimating equation is consistent at every κ₀.", f"Right: S is the efficient statistic, the more robust one is L_V: at ε = 0.001 the artefacts move S about {STORE['neuron_shift_ratio']:.0f} times as far as L_V."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The neuron example: inconsistency and robustness",
        "Left: the limit of the joint maximum likelihood estimate of the gamma shape, divided by the true shape, when every pair of intervals has its own rate: it tends to 2 for large shape and is 1.78 at shape 1, while the estimating-function solution is consistent. Right: estimates of the shape, true value 2, when a fraction of boxes is replaced by an artefact of two nearly simultaneous spikes: the efficient statistic S is pulled down quickly, the robust statistic L_V much more slowly.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_efficient_score(out / "efficient-score.svg")
    fig_neyman_scott(out / "neyman-scott.svg")
    fig_five_solutions(out / "five-solutions.svg")
    fig_estimating_class(out / "estimating-class.svg")
    fig_finite_model(out / "finite-model.svg")
    fig_neuron(out / "neuron.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


def main():
    check_nuisance()
    check_orthogonal()
    check_holonomy()
    check_boxes()
    check_matched_pairs()
    check_five_solutions()
    check_stonehenge()
    check_estimating_functions()
    check_roots()
    check_matrix_sandwich()
    check_geometry()
    check_delta_signs()
    check_prop_score()
    check_proportionality_bound()
    check_common_mean()
    check_neuron()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
