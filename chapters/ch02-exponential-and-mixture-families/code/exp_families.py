#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 2, checked by hand.

Exponential families and mixture families. Every number quoted in the notes comes from here.

Checked here, in the order the notes use them:

  1. the exponential-family form: psi, eta = grad psi, the dual function phi = theta.eta - psi, for the Bernoulli, Poisson,
     exponential, Gaussian and Gamma families; Theorem 2.1 (Fisher information = Hessian of psi = covariance of the statistics);
     "negative entropy" is relative to the reference measure;
  2. mixture families: S_n as both, a two-component mixture family, phi convex, theta = grad phi, KL = D_phi;
  3. e- and m-geodesics: a geometric versus an arithmetic mixture of two densities, bimodality, the e-geodesic as a one-parameter
     exponential family;
  4. the function-space picture: the Pythagorean identity (2.70)-(2.71) in discretised form, the local expansion (2.72)-(2.73),
     and the discontinuity of the entropy functional;
  5. the kernel exponential family: psi convex in the weights, eta = E[k(x, y)];
  6. Bregman divergences and exponential families: Theorem 2.2 by example (Gaussian, Poisson, exponential), the inverse Laplace condition;
  7. the three applications: maximum entropy (a die, and the printed 'theta = a'), mutual information (independence is e-flat, not m-flat),
     and maximum likelihood as an m-projection (Hardy-Weinberg), with the N-fold scaling.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only.

Run:  python3 exp_families.py            (checks)
      python3 exp_families.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def head(s):
    print("\n" + s)


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def digamma(x, h=1e-5):
    return (math.lgamma(x + h) - math.lgamma(x - h)) / (2 * h)


def trigamma(x, h=1e-4):
    return (math.lgamma(x + h) - 2 * math.lgamma(x) + math.lgamma(x - h)) / (h * h)


def gauss(x, m, s):
    return np.exp(-(x - m) ** 2 / (2 * s * s)) / (math.sqrt(2 * math.pi) * s)


# ------------------------------------------------------------------ 1. the exponential-family form

def check_forms():
    head("1. The exponential-family form p(x, theta) = exp{theta . x - psi(theta)} w.r.t. a reference measure mu (section 2.1)")
    # Bernoulli
    th = 0.7; p = 1 / (1 + math.exp(-th))
    psi = lambda t: math.log1p(math.exp(t))
    fd = (psi(th + 1e-6) - psi(th - 1e-6)) / 2e-6
    phi = p * math.log(p) + (1 - p) * math.log(1 - p)
    print(f"   Bernoulli: x = indicator, theta = log(p/(1-p)) = {th}, psi = log(1 + e^theta); eta = psi' = {fd:.4f} = p = {p:.4f}; psi = {psi(th):.4f}; "
          f"theta.eta - psi = {th * p - psi(th):.4f} = p log p + (1-p) log(1-p) = {phi:.4f}; Fisher = psi'' = p(1-p) = {p * (1 - p):.4f}")
    # Poisson: reference measure counting measure / x!
    lam = 3.0; th = math.log(lam)
    xs = np.arange(0, 80); lf = np.array([math.lgamma(k + 1) for k in xs])
    pmf = np.exp(xs * th - lam - lf)
    psi_p = lam                                                                    # psi = e^theta
    rel = th * lam - psi_p                                                         # theta.eta - psi (negative entropy relative to mu)
    shannon = float(np.sum(pmf * np.log(pmf)))
    print(f"   Poisson: theta = log lambda = {th:.4f}, psi = e^theta = {psi_p}, eta = lambda = {float(np.sum(xs * pmf)):.4f}; theta.eta - psi = {rel:.4f} "
          f"(negative entropy relative to mu = 1/x!); the true Shannon negative entropy sum p log p = {shannon:.4f}; difference {shannon - rel:+.4f} = -E[log x!] = {-float(np.sum(pmf * lf)):+.4f}")
    # Exponential distribution
    lam = 2.0; th = -lam
    x = np.linspace(0, 40, 400001); dx = x[1] - x[0]; pdf = lam * np.exp(-lam * x)
    print(f"   exponential: x > 0, theta = -rate = {th}, psi = -log(-theta) = {-math.log(-th):.4f}, eta = -1/theta = {-1 / th:.4f} (the mean; quadrature {float(np.sum(x * pdf) * dx):.4f}); "
          f"theta.eta - psi = {th * (-1 / th) + math.log(-th):.4f} = integral p log p = {float(np.sum(pdf * np.log(pdf + 1e-300)) * dx):.4f}")
    # Gaussian
    mu, sg = 1.0, 2.0; t1, t2 = mu / sg ** 2, -1 / (2 * sg ** 2)
    psi_g = -t1 ** 2 / (4 * t2) + 0.5 * math.log(-math.pi / t2)
    print(f"   Gaussian: theta = ({t1}, {t2}), psi = {psi_g:.4f}, eta = (mu, mu^2 + sigma^2) = ({mu}, {mu ** 2 + sg ** 2}); "
          f"theta.eta - psi = {t1 * mu + t2 * (mu ** 2 + sg ** 2) - psi_g:.4f} = -(1/2) log(2 pi e sigma^2) = {-0.5 * math.log(2 * math.pi * math.e * sg ** 2):.4f}")
    # Gamma(k, s): statistics (log x, x), reference measure dx / x
    k, s = 3.0, 2.0; th = np.array([k, -1 / s])
    psi_gam = lambda t: math.lgamma(t[0]) - t[0] * math.log(-t[1])
    xs = np.linspace(1e-7, 150, 3_000_001); dx = xs[1] - xs[0]
    pdf = xs ** (k - 1) * np.exp(-xs / s) / (math.gamma(k) * s ** k)
    eta_fd = np.array([(psi_gam(th + e) - psi_gam(th - e)) / 2e-6 for e in (np.array([1e-6, 0]), np.array([0, 1e-6]))])
    eta_q = np.array([float(np.sum(np.log(xs) * pdf) * dx), float(np.sum(xs * pdf) * dx)])
    H = np.zeros((2, 2)); h = 1e-4; E = np.eye(2) * h
    for i in range(2):
        for j in range(2):
            H[i, j] = (psi_gam(th + E[i] + E[j]) - psi_gam(th + E[i] - E[j]) - psi_gam(th - E[i] + E[j]) + psi_gam(th - E[i] - E[j])) / (4 * h * h)
    stats = np.stack([np.log(xs), xs]); C = np.array([[float(np.sum((stats[i] - eta_q[i]) * (stats[j] - eta_q[j]) * pdf) * dx) for j in range(2)] for i in range(2)])
    print(f"   Gamma(shape {k}, scale {s}): theta = (shape, -1/scale) = {th.tolist()}, statistics (log x, x), reference measure dx/x, psi = log Gamma(theta1) - theta1 log(-theta2) = {psi_gam(th):.4f}")
    print(f"      eta = grad psi = {np.round(eta_fd, 4).tolist()} (finite differences) = (digamma({k:.0f}) + log {s:.0f}, {k * s:.0f}) = ({digamma(k) + math.log(s):.4f}, {k * s:.4f}); quadrature of (E log x, E x) = {np.round(eta_q, 4).tolist()}")
    print(f"      Hessian of psi = {np.round(H, 4).tolist()} ; Cov[(log x, x)] by quadrature = {np.round(C, 4).tolist()} ; formula [[trigamma, -1/theta2], [-1/theta2, theta1/theta2^2]] = "
          f"[[{trigamma(k):.4f}, {-1 / th[1]:.1f}], [{-1 / th[1]:.1f}, {th[0] / th[1] ** 2:.1f}]]")
    # Theorem 2.1: Fisher information = E[score score^T] = Hessian, score = x - eta
    score = stats - eta_q[:, None]
    F = np.array([[float(np.sum(score[i] * score[j] * pdf) * dx) for j in range(2)] for i in range(2)])
    print(f"   Theorem 2.1 (Gamma): score d/dtheta_i log p = x_i - eta_i; E[score score^T] = {np.round(F, 4).tolist()} equals the Hessian {np.round(H, 4).tolist()}; "
          f"check d/dtheta1 log p at x = 5: log x - (digamma(theta1) - log(-theta2)) = {math.log(5) - eta_fd[0]:.4f}")


# ------------------------------------------------------------------ 2. mixture families

XG = np.linspace(-14, 17, 620001); DXG = XG[1] - XG[0]


def check_mixture_family():
    head("2. Mixture families (section 2.3): p(x, eta) = (1 - eta) q0(x) + eta q1(x), q0 = N(0, 1), q1 = N(3, 1)")
    q0, q1 = gauss(XG, 0, 1), gauss(XG, 3, 1)
    phi = lambda e: float(np.sum(((1 - e) * q0 + e * q1) * np.log((1 - e) * q0 + e * q1 + 1e-300)) * DXG)
    dphi = lambda e: float(np.sum((q1 - q0) * np.log((1 - e) * q0 + e * q1 + 1e-300)) * DXG)          # theta = phi'(eta), since int (q1 - q0) = 0
    d2phi = lambda e: float(np.sum((q1 - q0) ** 2 / ((1 - e) * q0 + e * q1 + 1e-300)) * DXG)
    for e in (0.2, 0.5, 0.8):
        h = 1e-4
        print(f"   eta = {e}: phi = {phi(e):.4f}; theta = phi' = {dphi(e):+.4f} (finite difference {(phi(e + h) - phi(e - h)) / (2 * h):+.4f}); "
              f"phi'' = int (q1 - q0)^2 / p = {d2phi(e):.4f} (finite difference {(phi(e + h) - 2 * phi(e) + phi(e - h)) / h ** 2:.4f}) > 0, so phi is convex")
    e, e2 = 0.2, 0.7
    pe, pe2 = (1 - e) * q0 + e * q1, (1 - e2) * q0 + e2 * q1
    kl_num = float(np.sum(pe * np.log(pe / pe2)) * DXG)
    d_phi = phi(e) - phi(e2) - dphi(e2) * (e - e2)
    print(f"   KL[p_{e} : p_{e2}] = {kl_num:.4f}; the Bregman divergence of phi, phi(eta) - phi(eta') - phi'(eta')(eta - eta') = {d_phi:.4f}  (2.48)")
    print(f"   the m-coordinate is eta (the weight); the e-coordinate theta = phi'(eta) runs from {dphi(0.05):+.3f} at eta = 0.05 to {dphi(0.95):+.3f} at eta = 0.95: it is not the natural parameter of an exponential family here")
    # S_n is both: mixture of delta distributions with weights p_i
    p = np.array([0.5, 0.3, 0.2]); th = np.log(p[1:] / p[0])
    print(f"   S_2 as both: p = {p.tolist()} is the mixture of the three point masses with weights p (eta = (p1, p2) = {p[1:].tolist()}) and the exponential family with theta = log(p_i/p_0) = {np.round(th, 4).tolist()}")


# ------------------------------------------------------------------ 3. e- and m-geodesics

def check_geodesics():
    head("3. e-geodesic (geometric mixture) versus m-geodesic (arithmetic mixture) of two densities (section 2.4)")
    P, Q = (-2.0, 1.0), (2.0, 1.0)
    p, q = gauss(XG, *P), gauss(XG, *Q)
    t = 0.5
    m = (1 - t) * p + t * q
    lo = (1 - t) * np.log(p) + t * np.log(q + 1e-300)
    z = float(np.sum(np.exp(lo)) * DXG); e = np.exp(lo) / z
    peaks = lambda y: [round(float(XG[i]), 2) for i in range(1, len(XG) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 1e-3]
    print(f"   P = N{P}, Q = N{Q}, t = 0.5. Normaliser of the geometric mixture: int sqrt(p q) = {z:.4f} = exp(-(mu1 - mu2)^2/8) = {math.exp(-16 / 8):.4f}, so psi(0.5) = log of it = {math.log(z):.4f}")
    print(f"   m-mixture (arithmetic): modes at {peaks(m)}, density at 0 is {float(m[len(XG) // 2 + int(round(-XG[0] / DXG)) - len(XG) // 2]):.4f}; "
          f"e-mixture (geometric): modes at {peaks(e)}, which is N(0, 1) with peak {float(e.max()):.4f}")
    mean = float(np.sum(XG * m) * DXG); var = float(np.sum((XG - mean) ** 2 * m) * DXG); kurt = float(np.sum((XG - mean) ** 4 * m) * DXG / var ** 2 - 3)
    print(f"   the m-mixture has mean {mean:.4f}, variance {var:.4f} (= 1 + 4) and excess kurtosis {kurt:+.4f} (negative: two humps), the e-mixture is Gaussian (excess kurtosis 0)")
    # log-linearity along the e-geodesic: log p(x, t) = (1 - t) log p + t log q - psi(t)
    t2 = 0.3; lo2 = (1 - t2) * np.log(p) + t2 * np.log(q + 1e-300); psi2 = math.log(float(np.sum(np.exp(lo2)) * DXG))
    e2 = np.exp(lo2 - psi2); mask = (XG > -6) & (XG < 6)
    print(f"   the e-geodesic is a one-parameter exponential family with natural parameter t: log p(x, t) - [(1 - t) log p + t log q] = -psi(t) is constant: spread {float(np.ptp((np.log(e2) - lo2)[mask])):.1e}; "
          f"psi(0.3) = {psi2:.4f} = -0.3 * 0.7 * 16/2 = {-0.3 * 0.7 * 16 / 2:.4f}")
    # unequal widths: tails
    P2, Q2 = (0.0, 1.0), (0.0, 2.0)
    p2, q2 = gauss(XG, *P2), gauss(XG, *Q2)
    m2 = 0.5 * p2 + 0.5 * q2
    var2 = float(np.sum(XG ** 2 * m2) * DXG); k2 = float(np.sum(XG ** 4 * m2) * DXG / var2 ** 2 - 3)
    prec = 0.5 / 1 + 0.5 / 4
    print(f"   equal means, widths 1 and 2: the m-mixture has variance {var2:.4f} and excess kurtosis {k2:+.4f} (heavy tails); the e-mixture is N(0, {1 / prec:.4f}) with sigma {math.sqrt(1 / prec):.4f} (precisions average: {prec:.4f}), Gaussian, no excess")


# ------------------------------------------------------------------ 4. function space

def check_function_space():
    head("4. The function-space picture (section 2.5): Pythagoras (2.70)-(2.71), the local metric (2.72)-(2.73), and a pathology")
    rng = np.random.default_rng(4)
    p, q, r = (rng.dirichlet(np.ones(8)) for _ in range(3))
    gap = kl(p, r) - kl(p, q) - kl(q, r)
    ident = float(np.sum((p - q) * (np.log(q) - np.log(r))))
    print(f"   discretised on 8 points, random p, q, r: KL[p:r] - KL[p:q] - KL[q:r] = {gap:+.6f} = sum (p - q)(log q - log r) = {ident:+.6f}: the gap is exactly the pairing in (2.71)")
    d = p - q; v = rng.normal(size=8); v = v - (v @ d) / (d @ d) * d
    for tt in (0.5, 2.0):
        rr = q * np.exp(tt * v); rr /= rr.sum()
        print(f"   choose r = q exp(t v) / Z with sum (p - q) v = 0 (t = {tt}): pairing sum (p - q)(log r - log q) = {float(np.sum(d * (np.log(rr) - np.log(q)))):+.1e}; "
              f"KL[p:r] = {kl(p, rr):.6f} = KL[p:q] + KL[q:r] = {kl(p, q) + kl(q, rr):.6f}")
    u = rng.normal(size=8); u -= u.mean(); u *= 0.5 / np.max(np.abs(u / p))
    for eps in (1.0, 0.1, 0.01):
        dl = eps * u
        print(f"   local expansion: delta = {eps} u: KL[p : p + delta] = {kl(p, p + dl):.3e}, (1/2) sum delta^2/p = {0.5 * float(np.sum(dl ** 2 / p)):.3e}, ratio {kl(p, p + dl) / (0.5 * float(np.sum(dl ** 2 / p))):.4f}")
    phi_f = -0.5 * math.log(2 * math.pi * math.e)
    print(f"   the entropy functional is not continuous: p_n = (1 - 1/n) N(0, 1) + (1/n) Uniform[10, 10 + e^(n^2)] stays within L1 distance about 2/n of N(0, 1), yet its negative entropy is")
    for n in (2, 3, 4, 5, 6):
        eps = 1 / n; Lg = n * n
        val = (1 - eps) * (phi_f + math.log(1 - eps)) + eps * (math.log(eps) - Lg)
        print(f"      n = {n}: L1 distance about {2 * eps:.3f}, negative entropy {val:+.3f} (N(0, 1) has {phi_f:+.3f})")


# ------------------------------------------------------------------ 5. kernel exponential family

def check_kernel_family():
    head("5. The kernel exponential family (section 2.6): p(x) = exp{ sum_j alpha_j k(x, y_j) - psi(alpha) } w.r.t. mu(x) = exp(-x^2/(2 tau^2))")
    sg, tau = 0.5, 2.0
    ys = np.array([-1.0, 0.5, 2.0]); al = np.array([1.5, -1.0, 2.0])
    x = np.linspace(-14, 14, 280001); dx = x[1] - x[0]
    K = np.stack([gauss(x, yj, sg) for yj in ys]); w = np.exp(-x ** 2 / (2 * tau ** 2))
    psi = lambda a: math.log(float(np.sum(np.exp(a @ K) * w) * dx))
    pdf = np.exp(al @ K - psi(al)) * w
    eta_q = np.array([float(np.sum(K[j] * pdf) * dx) for j in range(3)])
    eta_fd = np.array([(psi(al + 1e-5 * np.eye(3)[j]) - psi(al - 1e-5 * np.eye(3)[j])) / 2e-5 for j in range(3)])
    C = np.array([[float(np.sum((K[i] - eta_q[i]) * (K[j] - eta_q[j]) * pdf) * dx) for j in range(3)] for i in range(3)])
    print(f"   Gaussian kernel width {sg}, centres y = {ys.tolist()}, weights alpha = {al.tolist()}, tau = {tau}: psi = {psi(al):.4f}; normalisation check int p dmu = {float(np.sum(pdf) * dx):.6f}")
    print(f"   eta_j = E[k(x, y_j)] by quadrature {np.round(eta_q, 4).tolist()}; grad psi by finite differences {np.round(eta_fd, 4).tolist()} (the dual parameter (2.82) is the gradient)")
    print(f"   Hessian of psi = Cov[k(x, y_j), k(x, y_l)] has eigenvalues {np.round(np.linalg.eigvalsh(C), 5).tolist()}: all positive, so psi is convex in alpha")
    ys6 = np.linspace(-2, 2, 6); G = np.array([[float(gauss(a, b, sg)) for b in ys6] for a in ys6])
    print(f"   positivity (2.78) of the Gaussian kernel: Gram matrix on 6 points has smallest eigenvalue {np.linalg.eigvalsh(G).min():.4f} > 0")
    print(f"   density p(x) is {'bimodal' if sum(1 for i in range(1, len(x) - 1) if pdf[i] > pdf[i - 1] and pdf[i] > pdf[i + 1] and pdf[i] > 1e-3) > 1 else 'unimodal'} here; "
          f"modes at {[round(float(x[i]), 2) for i in range(1, len(x) - 1) if pdf[i] > pdf[i - 1] and pdf[i] > pdf[i + 1] and pdf[i] > 1e-3]}")


# ------------------------------------------------------------------ 6. Bregman divergences and exponential families

def check_bregman_exp():
    head("6. Every Bregman divergence is the KL of an exponential family (Theorem 2.2), by example (section 2.7)")
    th = 1.3
    xs = np.linspace(-14, 14, 560001); dx = xs[1] - xs[0]
    print(f"   inverse Laplace condition (2.85): for psi = theta^2/2 the measure is N(0, 1): int e^(theta x) dN(0, 1) = {float(np.sum(np.exp(th * xs) * gauss(xs, 0, 1)) * dx):.4f} = exp(theta^2/2) = {math.exp(th * th / 2):.4f}; "
          f"for psi = e^theta it is the counting measure / x!: sum e^(theta x)/x! = {sum(math.exp(0.4 * k - math.lgamma(k + 1)) for k in range(80)):.4f} = exp(e^0.4) = {math.exp(math.exp(0.4)):.4f}")
    t, tp = 2.0, 0.5
    kl_g = float(np.sum(gauss(xs, tp, 1) * np.log(gauss(xs, tp, 1) / gauss(xs, t, 1))) * dx)
    print(f"   psi = theta^2/2 (Gaussian N(theta, 1)): D_psi[theta : theta'] = {0.5 * (t - tp) ** 2:.4f} against KL[N(theta', 1) : N(theta, 1)] = {kl_g:.4f}  (theta = {t}, theta' = {tp})")
    lam, lamp = 3.0, 1.0; th_, thp_ = math.log(lam), math.log(lamp)
    d_p = math.exp(th_) - math.exp(thp_) - math.exp(thp_) * (th_ - thp_)
    kl_p = sum((lamp ** k * math.exp(-lamp) / math.factorial(k)) * (k * math.log(lamp / lam) - lamp + lam) for k in range(60))
    print(f"   psi = e^theta (Poisson): D_psi[theta : theta'] = {d_p:.4f} against KL[Poisson({lamp}) : Poisson({lam})] = {kl_p:.4f} = lam' log(lam'/lam) - lam' + lam = {lamp * math.log(lamp / lam) - lamp + lam:.4f} (generalised KL, section 1.3)")
    m, mp = 2.0, 0.5; t_, tp_ = -1 / m, -1 / mp
    d_e = -math.log(-t_) + math.log(-tp_) - (-1 / tp_) * (t_ - tp_)
    x = np.linspace(0, 80, 800001); dxe = x[1] - x[0]
    pe, qe = np.exp(-x / mp) / mp, np.exp(-x / m) / m
    kl_e = float(np.sum(pe * np.log(pe / qe)) * dxe)
    print(f"   psi = -log(-theta) (exponential distribution, mean m = -1/theta): D_psi[theta : theta'] = {d_e:.4f} against KL[Exp(mean {mp}) : Exp(mean {m})] = {kl_e:.4f} = ln(m/m') + m'/m - 1 = {math.log(m / mp) + mp / m - 1:.4f}: "
          "the Itakura-Saito divergence of section 1.3 is the KL divergence between exponential distributions")
    print("   (the mixture family M has a dually flat structure with phi as potential, but Theorem 2.2 does not make it an exponential family: the family built from phi is a different one)")


# ------------------------------------------------------------------ 7. maximum entropy

def tilt_to_mean(base, vals, a):
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        w = base * np.exp(mid * vals); w /= w.sum()
        lo, hi = (mid, hi) if float(w @ vals) < a else (lo, mid)
    return (lo + hi) / 2


def check_maxent():
    head("7a. Maximum entropy (section 2.8.1): a six-sided die with prescribed mean a")
    vals = np.arange(1, 7, dtype=float); P0 = np.full(6, 1 / 6)
    H = lambda p: -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    for a in (2.0, 3.5, 4.5, 5.5):
        th = tilt_to_mean(P0, vals, a); ph = P0 * np.exp(th * vals); ph /= ph.sum()
        print(f"   a = {a}: maximum entropy distribution p(x) ∝ exp(theta x) with theta = {th:+.4f}: {np.round(ph, 4).tolist()}; entropy {H(ph):.4f} (uniform: {math.log(6):.4f}); KL[p_hat : uniform] = log 6 - H = {kl(ph, P0):.4f}")
    a = 4.5; th = tilt_to_mean(P0, vals, a); ph = P0 * np.exp(th * vals); ph /= ph.sum()
    rng = np.random.default_rng(7); best = 0.0; worst_gap = 0.0; worst_pair = 0.0
    for _ in range(2000):
        p = rng.dirichlet(np.ones(6) * 0.8); s = tilt_to_mean(p, vals, a); p = p * np.exp(s * vals); p /= p.sum()      # a random member of M(4.5)
        best = max(best, H(p)); worst_gap = max(worst_gap, abs(kl(p, P0) - (kl(p, ph) + kl(ph, P0)))); worst_pair = max(worst_pair, abs((p - ph) @ vals * th))
    print(f"   2000 random distributions with mean {a}: the largest entropy found is {best:.4f} < {H(ph):.4f} = the maximum entropy member")
    print(f"   Pythagoras (2.89) KL[P : P0] = KL[P : P_hat] + KL[P_hat : P0] for all 2000: largest gap {worst_gap:.1e}; orthogonality theta (E_P[x] - a) = {worst_pair:.1e}")
    print("   the printed '(2.90) natural coordinates are specified by theta = a' does not hold: the constraint fixes the EXPECTATION parameter, eta = a, and the natural parameter theta(a) is a different "
          f"number (theta = 0 at a = 3.5, {tilt_to_mean(P0, vals, 4.5):.4f} at a = 4.5, {tilt_to_mean(P0, vals, 5.5):.4f} at a = 5.5)")


# ------------------------------------------------------------------ 8. mutual information

def odds(P):
    return float(P[0, 0] * P[1, 1] / (P[0, 1] * P[1, 0]))


def mi(P):
    return kl(P.ravel(), np.outer(P.sum(1), P.sum(0)).ravel())


def check_mutual_information():
    head("7b. Mutual information (section 2.8.2): the independent distributions are e-flat, not m-flat")
    a, b = np.array([0.7, 0.3]), np.array([0.2, 0.8]); c, d = np.array([0.4, 0.6]), np.array([0.9, 0.1])
    p, q = np.outer(a, b), np.outer(c, d)
    em = np.sqrt(p * q); em /= em.sum(); mm = 0.5 * p + 0.5 * q
    print(f"   two product distributions p = a x b, q = c x d: odds ratio of p = {odds(p):.4f}, of q = {odds(q):.4f} (independent means odds ratio 1)")
    print(f"   their e-midpoint (geometric mixture) has odds ratio {odds(em):.6f}: still independent, so the independent family is e-flat; "
          f"their m-midpoint (arithmetic mixture) has odds ratio {odds(mm):.4f} and mutual information {mi(mm):.4f}: not independent, so it is not m-flat")
    P = np.array([[0.4, 0.1], [0.2, 0.3]]); Ph = np.outer(P.sum(1), P.sum(0))
    print(f"   P = {P.tolist()}: marginals {np.round(P.sum(1), 4).tolist()} and {np.round(P.sum(0), 4).tolist()}; m-projection onto independence is the product of marginals {np.round(Ph, 4).tolist()}; "
          f"mutual information I = KL[P : product] = {mi(P):.6f}; odds ratio of P {odds(P):.4f}")
    Q = np.outer([0.5, 0.5], [0.5, 0.5])
    print(f"   Pythagoras for an arbitrary independent Q = uniform x uniform: KL[P:Q] = {kl(P.ravel(), Q.ravel()):.6f} = I + KL[product : Q] = {mi(P):.6f} + {kl(Ph.ravel(), Q.ravel()):.6f}")
    fib = lambda s: np.array([[s, 0.5 - s], [0.6 - s, s - 0.1]])
    ss = np.linspace(0.1001, 0.4999, 4000); mis = np.array([mi(fib(s)) for s in ss])
    print(f"   the reverse problem: joints with the same marginals (0.5, 0.5), (0.6, 0.4) form an m-flat line parametrised by s = P00 in (0.1, 0.5); I is {abs(mi(fib(0.3))):.4f} at s = 0.3 (independent), "
          f"{mis[0]:.4f} near s = 0.1 and {mis[-1]:.4f} near s = 0.5: the maximum of KL[p : p_hat] on the line sits at an end, where a cell is empty")
    rho = 0.8; x = np.linspace(-7, 7, 1401); dx = x[1] - x[0]; X, Y = np.meshgrid(x, x)
    pdf = np.exp(-(X ** 2 - 2 * rho * X * Y + Y ** 2) / (2 * (1 - rho ** 2))) / (2 * math.pi * math.sqrt(1 - rho ** 2))
    marg = lambda z: np.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
    mi_g = float(np.sum(pdf * np.log(pdf / (marg(X) * marg(Y)))) * dx * dx)
    print(f"   bivariate Gaussian, correlation {rho}: mutual information by 2-D quadrature {mi_g:.4f} = -(1/2) log(1 - rho^2) = {-0.5 * math.log(1 - rho ** 2):.4f}")


# ------------------------------------------------------------------ 9. maximum likelihood as an m-projection

def hw(q):
    return np.array([q * q, 2 * q * (1 - q), (1 - q) ** 2])


def check_mle():
    head("7c. Repeated observations and maximum likelihood (section 2.8.3)")
    N, p = 10, 0.7
    kl_ber = lambda a_, b_: a_ * math.log(a_ / b_) + (1 - a_) * math.log((1 - a_) / (1 - b_))
    pmf = lambda n, pp: np.array([math.comb(n, k) * pp ** k * (1 - pp) ** (n - k) for k in range(n + 1)])
    print(f"   N = {N} coins: the joint density is exp(N [theta xbar - psi(theta)]): psi becomes N psi. KL between the two N-fold laws of the sufficient statistic = {kl(pmf(N, 0.7), pmf(N, 0.5)):.4f} "
          f"= N KL[Ber(0.7) : Ber(0.5)] = {N * kl_ber(0.7, 0.5):.4f}; variance of the count {float(np.sum((np.arange(N + 1) - N * p) ** 2 * pmf(N, p))):.4f} = N p (1 - p) = {N * p * (1 - p):.4f} (Fisher information scales by N)")
    print(f"   the observed point in eta-coordinates is xbar: 7 heads in 10 give eta_bar = 0.7, and the maximum likelihood theta is logit(0.7) = {math.log(0.7 / 0.3):.4f}")
    n = np.array([40, 40, 20]); pbar = n / n.sum()
    qs = np.linspace(0.0005, 0.9995, 1999001)
    ll = n[0] * 2 * np.log(qs) + n[1] * (np.log(2) + np.log(qs) + np.log(1 - qs)) + n[2] * 2 * np.log(1 - qs)
    qh = float(qs[int(np.argmax(ll))]); ph = hw(qh)
    print(f"   Hardy-Weinberg: genotype counts (AA, Aa, aa) = {n.tolist()}, observed point pbar = {pbar.tolist()}; model p(q) = (q^2, 2q(1-q), (1-q)^2). Brute-force maximum likelihood q = {qh:.4f} = (2 n_AA + n_Aa)/(2N) = {(2 * n[0] + n[1]) / (2 * n.sum()):.4f}")
    print(f"      p_hat = {np.round(ph, 4).tolist()}; KL[pbar : p_hat] = {kl(pbar, ph):.6f}; maximising the log likelihood is minimising this divergence (the two differ by the constant sum pbar log pbar)")
    th = lambda q: np.array([math.log(hw(q)[1] / hw(q)[0]), math.log(hw(q)[2] / hw(q)[0])])
    resid = [float(th(q)[1] - 2 * th(q)[0] + 2 * math.log(2)) for q in (0.1, 0.3, 0.45, 0.6, 0.9)]
    print(f"   the model is e-flat: theta(q) = (log 2 + L, 2 L) with L = log((1 - q)/q), the line theta2 = 2 theta1 - 2 log 2; theta at q = 0.3, 0.45, 0.6 is "
          f"{np.round(th(0.3), 4).tolist()}, {np.round(th(0.45), 4).tolist()}, {np.round(th(0.6), 4).tolist()}; residual theta2 - 2 theta1 + 2 log 2 at q = 0.1, 0.3, 0.45, 0.6, 0.9: "
          + ", ".join(f"{r:.0e}" for r in resid))
    qh_exact = (2 * n[0] + n[1]) / (2 * n.sum()); ph = hw(qh_exact)
    gaps = [abs(kl(pbar, hw(q)) - (kl(pbar, ph) + kl(ph, hw(q)))) for q in np.linspace(0.05, 0.95, 500)]
    dth = (th(qh_exact + 1e-6) - th(qh_exact - 1e-6)) / 2e-6
    print(f"   Pythagoras KL[pbar : p(q)] = KL[pbar : p_hat] + KL[p_hat : p(q)] over 500 values of q: largest gap {max(gaps):.1e}; orthogonality: (eta_bar - eta_hat) . d theta/dq = {float((pbar - ph)[1:] @ dth):.1e} (the score equation)")
    for nn in ((40, 40, 20), (20, 50, 30), (5, 30, 65)):
        nn = np.array(nn); print(f"      counts {nn.tolist()}: q_hat = {(2 * nn[0] + nn[1]) / (2 * nn.sum()):.4f}, KL[pbar : p_hat] = {kl(nn / nn.sum(), hw((2 * nn[0] + nn[1]) / (2 * nn.sum()))):.6f}")



# ------------------------------------------------------------------ shared figure helpers (as in the Chapter 1 script)

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
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .fillS{fill:#3987e5}
    .ring{stroke:#161615}
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


# simplex chart: a point p = (p0, p1, p2) is drawn at p0 V0 + p1 V1 + p2 V2
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def simplex_xy(p):
    return np.asarray(p) @ V


class SimplexPanel(Panel):
    def __init__(self, body, x0, y0, w, title, ty=None):
        super().__init__(body, x0, y0, w, w * math.sqrt(3) / 2, (0, 1), (0, math.sqrt(3) / 2))
        self.title, self.ty = title, (y0 - 34 if ty is None else ty)

    def frame_simplex(self):
        b = self.b
        pts = " ".join(f"{self.X(v[0]):.1f},{self.Y(v[1]):.1f}" for v in V)
        b.append(f'<polygon class="ax fillS" points="{pts}"/>')
        for k, name in enumerate(["outcome 0", "outcome 1", "outcome 2"]):
            anchor = ["start", "end", "middle"][k]
            dy = [16, 16, -8][k]
            b.append(f'<text class="sm" x="{self.X(V[k][0]):.1f}" y="{self.Y(V[k][1]) + dy:.1f}" text-anchor="{anchor}">{name}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.ty}">{self.title}</text>')

    def curve(self, ps, cls):
        xy = np.array([simplex_xy(p) for p in ps])
        self.line(xy[:, 0], xy[:, 1], cls)

    def pt(self, p, cls, r=4.5):
        xy = simplex_xy(p); self.dot(xy[0], xy[1], cls, r)

    def label(self, p, s, dx=8, dy=-8, cls="v"):
        xy = simplex_xy(p)
        self.b.append(f'<text class="{cls}" x="{self.X(xy[0]) + dx:.1f}" y="{self.Y(xy[1]) + dy:.1f}">{s}</text>')




# ------------------------------------------------------------------ figures

class HWPanel(SimplexPanel):
    names = ("AA", "Aa", "aa")

    def frame_simplex(self):
        b = self.b
        pts = " ".join(f"{self.X(v[0]):.1f},{self.Y(v[1]):.1f}" for v in V)
        b.append(f'<polygon class="ax fillS" points="{pts}"/>')
        for k, name in enumerate(self.names):
            anchor = ["start", "end", "middle"][k]; dy = [16, 16, -8][k]
            b.append(f'<text class="sm" x="{self.X(V[k][0]):.1f}" y="{self.Y(V[k][1]) + dy:.1f}" text-anchor="{anchor}">{name}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.ty}">{self.title}</text>')


def fig_mixtures(out):
    body = []
    xs = np.linspace(-6, 6, 400); P, Q = (-2.0, 1.0), (2.0, 1.0)
    p, q = gauss(xs, *P), gauss(xs, *Q)
    for k, (title, mk) in enumerate((("m-geodesic: arithmetic mixtures (1 − t) p + t q", lambda t: (1 - t) * p + t * q),
                                      ("e-geodesic: geometric mixtures ∝ p^(1−t) q^t", None))):
        A = Panel(body, 56 + 360 * k, 46, 300, 190, (-6, 6), (0, 0.48))
        A.frame([-4, -2, 0, 2, 4], [0, 0.2, 0.4], "x", "density", title, grid=False)
        A.line(xs, p, "thin s0"); A.line(xs, q, "thin s0")
        for t, cls in ((0.25, "thin s2" if k == 0 else "thin s1"), (0.5, "ln s2" if k == 0 else "ln s1"), (0.75, "dash s2" if k == 0 else "dash s1")):
            if mk is not None: y = mk(t)
            else:
                lo = (1 - t) * np.log(p) + t * np.log(q); y = np.exp(lo); y /= np.sum(y) * (xs[1] - xs[0])
            A.line(xs, y, cls)
        A.text(-2, 0.42, "p", "sm", dx=-4, dy=-4); A.text(2, 0.42, "q", "sm", dx=-4, dy=-4)
    body.append('<text class="sm" x="60" y="290">p = N(−2, 1), q = N(2, 1); curves at t = 0.25 (thin), 0.5 (bold), 0.75 (dashed). The arithmetic mixture has two humps, kurtosis −1.28 at t = 0.5;</text>')
    body.append('<text class="sm" x="60" y="308">the geometric mixture is always one Gaussian, here N(0, 1) at t = 0.5, sliding from p to q with ψ(t) = −8 t (1 − t).</text>')
    (out / "mixtures.svg").write_text(svg(800, 322, "Arithmetic versus geometric mixtures of two Gaussians",
        "Two panels with p = N(-2, 1) and q = N(2, 1) in grey. Left: the arithmetic mixtures (1-t)p + tq at t = 0.25, 0.5, 0.75 are bimodal, with two humps at -2 and 2 at t = 0.5. Right: the geometric mixtures proportional to p^(1-t) q^t are single Gaussian bumps of unit width sliding from p to q, with N(0, 1) at t = 0.5.", body))


def fig_maxent(out):
    body = []
    vals = np.arange(1, 7, dtype=float); P0 = np.full(6, 1 / 6)
    A = Panel(body, 56, 46, 290, 200, (0.7, 6.3), (0, 0.72))
    A.frame([1, 2, 3, 4, 5, 6], [0, 0.2, 0.4, 0.6], "face x", "probability", "maximum entropy dice for mean a", grid=False)
    for a, cls in ((2.0, "s1"), (3.5, "s0"), (4.5, "s2"), (5.5, "s3")):
        th = tilt_to_mean(P0, vals, a); ph = P0 * np.exp(th * vals); ph /= ph.sum()
        A.line(vals, ph, "ln " + cls)
        for x_, y_ in zip(vals, ph): A.dot(x_, y_, "f" + cls[1], 2.6)
        A.text(6, ph[-1], f"a = {a}", "sm", dx=6, dy=4)
    B = Panel(body, 450, 46, 290, 200, (1, 6), (-3, 4))
    B.frame([1, 2, 3, 4, 5, 6], [-2, 0, 2, 4], "mean a", "natural parameter θ", "θ(a): not the line θ = a", grid=False)
    aa = np.linspace(1.15, 5.85, 200)
    B.line(aa, [tilt_to_mean(P0, vals, a) for a in aa], "ln s2")
    dl = np.linspace(1, 6, 100); B.line(dl, dl, "dash s0")
    for a in (2.0, 3.5, 4.5, 5.5):
        B.dot(a, tilt_to_mean(P0, vals, a), "f2", 3.6)
    B.text(4.5, tilt_to_mean(P0, vals, 4.5), "θ = 0.3710 at a = 4.5", "sm", dx=-8, dy=-12, anchor="end")
    body.append('<text class="sm" x="60" y="296">Left: the maximum entropy member is an exponential tilt of the uniform die, p(x) ∝ exp(θ x); a = 3.5 is the uniform die (θ = 0). Right: the natural parameter θ(a)</text>')
    body.append('<text class="sm" x="60" y="314">that produces mean a (orange) is not the line θ = a (dashed): the constraint fixes the expectation parameter η = a, and θ is its Legendre partner.</text>')
    (out / "maxent-dice.svg").write_text(svg(800, 330, "Maximum entropy dice and their natural parameter",
        "Left: maximum entropy distributions on a six-sided die for means 2, 3.5, 4.5 and 5.5: exponential tilts of the uniform die. Right: the natural parameter theta as a function of the mean a, an S-shaped curve through theta = 0 at a = 3.5, theta = 0.3710 at a = 4.5 and 1.0870 at a = 5.5, compared with the dashed line theta = a.", body))


def fig_hw(out):
    body = []
    S = HWPanel(body, 56, 80, 360, "Hardy–Weinberg: genotype frequencies p(q) = (q², 2q(1−q), (1−q)²)", ty=34)
    S.frame_simplex()
    qs = np.linspace(0.01, 0.99, 200)
    S.curve([hw(q) for q in qs], "ln s2")
    cols = {(40, 40, 20): "f1", (20, 50, 30): "f3", (5, 30, 65): "f4"}
    for nn, c in cols.items():
        nn_ = np.array(nn); pb = nn_ / nn_.sum(); qh = (2 * nn_[0] + nn_[1]) / (2 * nn_.sum()); ph = hw(qh)
        S.curve([(1 - t) * pb + t * ph for t in np.linspace(0, 1, 20)], "dash s0")
        S.pt(pb, c, 4.5); S.pt(ph, "f2", 3.6)
    S.label(np.array([0.4, 0.4, 0.2]), "p̄ = (0.4, 0.4, 0.2)", 8, -6)
    S.label(hw(0.6), "p̂ (q̂ = 0.6)", 8, 14)
    body.append('<line class="ln s2" x1="450" y1="130" x2="482" y2="130"/><text class="sm" x="488" y="134">the model: an e-flat curve (a straight line in θ)</text>')
    body.append('<line class="dash s0" x1="450" y1="154" x2="482" y2="154"/><text class="sm" x="488" y="158">m-geodesic from the observed point to its m-projection:</text>')
    body.append('<text class="sm" x="488" y="174">a straight segment in the triangle</text>')
    body.append('<text class="sm" x="450" y="212">Observed frequencies p̄ (blue, green, yellow) are projected onto the curve</text>')
    body.append('<text class="sm" x="450" y="230">along straight lines; the feet (orange) are the maximum</text>')
    body.append('<text class="sm" x="450" y="248">likelihood estimates: q̂ = (2 n_AA + n_Aa)/(2N) = 0.6, 0.45, 0.2.</text>')
    body.append('<text class="sm" x="450" y="276">For p̄ = (0.4, 0.4, 0.2): KL[p̄ : p̂] = 0.013844, and</text>')
    body.append('<text class="sm" x="450" y="294">KL[p̄ : p(q)] = KL[p̄ : p̂] + KL[p̂ : p(q)] for every q.</text>')
    (out / "hardy-weinberg.svg").write_text(svg(800, 430, "Maximum likelihood is an m-projection onto the Hardy-Weinberg curve",
        "The probability triangle of genotype frequencies AA, Aa, aa with the Hardy-Weinberg curve (q^2, 2q(1-q), (1-q)^2). Three observed frequency vectors are joined by straight segments to their m-projections on the curve, which are the maximum likelihood estimates with allele frequencies 0.6, 0.45 and 0.2. For the observed (0.4, 0.4, 0.2) the divergence to the curve is 0.013844.", body))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_mixtures(out); fig_maxent(out); fig_hw(out)
    print("\nwrote", ", ".join(sorted(p.name for p in out.glob("*.svg"))))


if __name__ == "__main__":
    check_forms(); check_mixture_family(); check_geodesics(); check_function_space(); check_kernel_family(); check_bregman_exp()
    check_maxent(); check_mutual_information(); check_mle()
    if "--figures" in sys.argv:
        make_figures()
