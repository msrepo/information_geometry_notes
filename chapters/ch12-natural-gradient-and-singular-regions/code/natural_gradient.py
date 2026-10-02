#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 12, checked by hand.

Natural gradient learning and its dynamics in singular regions. Every number quoted in the notes comes from here.

Part A (section 12.1) is checked on small models in which everything is exact or cheap: a 2x2 metric, a Bernoulli model in two
charts, a three-point sample space, a three-state Markov decision process, a Gaussian linear model whose online error covariance
obeys an exact recursion, a two-dimensional logistic regression, and a single erf neuron. Part B (section 12.2) uses one model for
everything: the two-hidden-unit perceptron  f(x) = v1 phi(w1 x) + v2 phi(w2 x)  with scalar input x ~ N(0,1) and the book's
sigmoid phi(u) = erf(u / sqrt 2) of (12.103). For this phi the Gaussian averages of products are arcsin kernels,
E[phi(a x) phi(b x)] = (2/pi) arcsin(a b / sqrt((1 + a^2)(1 + b^2))), so the expected loss, its gradient, the Fisher matrix and
the Hessian are closed forms, and the averaged learning dynamics (12.129) can be integrated without sampling. The teachers are one
neuron (inside the critical region R) or two neurons (outside).

Checked here, in the order the notes use them:

  1. steepest descent in a metric (12.17)-(12.22), covariant against contravariant (12.24)-(12.25), invariance of the natural
     gradient flow under a change of chart (Bernoulli, two charts);
  2. the Fisher matrix against the Hessian (12.32)-(12.35), where they differ at a critical region, and the saddle-free
     Newton method (12.38) on a toy saddle (its flow near the critical region is in section 11);
  3. stochastic relaxation (12.40): the natural gradient of an expected cost is a constant vector, the flow is exponential tilting;
  4. the natural policy gradient (12.46)-(12.56) on a random three-state MDP, including (12.56) and the recursion (12.55);
  5. mirror descent against the natural gradient (12.57)-(12.62): first-order agreement, same flow, and its dependence on the chart;
  6. Theorem 12.1 (12.63)-(12.70): an exact recursion for the online error covariance of the Gaussian linear model with a c/t
     learning constant, vanilla against natural, then a logistic regression with an adaptive estimate of G^-1 (12.84)-(12.85);
  7. saturation and Theorem 12.2 (12.76)-(12.82) on one erf neuron, with a short online simulation;
  8. the adaptive learning constant (12.87)-(12.100);
  9. the perceptron's singular structure (12.108)-(12.128): equivalent points, rank of the Fisher matrix, the scaling of its
     eigenvalues, the layer-wise approximation of section 12.1.7.4, the blow-down chart, and the expansion (12.139) in the chart
     (u, z, s, r) of (12.130)-(12.138);
 10. the reduced dynamics (12.141)-(12.148), (12.144), (12.149) against the exact averaged flow, the stability pattern of
     Theorems 12.3-12.4, a first integral, and the basin of the stable part of R_o (Milnor attractor);
 11. plateau and critical slowdown (12.151)-(12.152) for the vanilla gradient, the natural gradient near R (12.153)-(12.158), the
     blow-down coordinates (12.155)-(12.157), the size of the natural gradient when the teacher is outside R, the saddle-free Newton
     (absolute Hessian) flow, online SGD on the plateau;
 12. a singular statistical model (12.159)-(12.162): the likelihood-ratio statistic of a neuron whose weight is unidentified.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Monte Carlo uses fixed seeds; the whole script takes under twenty seconds.

Run:  python3 natural_gradient.py            (checks)
      python3 natural_gradient.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20261002)
STORE = {}                         # numbers computed by the checks, reused by the figures
np.set_printoptions(precision=6, suppress=True, linewidth=160)


def head(s):
    print("\n" + s)


def sci(v, d=2):
    """Scientific notation for a list of numbers."""
    return "[" + ", ".join(f"{float(x):.{d}e}" for x in np.asarray(v).ravel()) + "]"


def clean(v, d=4):
    """Round an array for printing, with tiny values shown as 0."""
    return [0.0 if abs(x) < 1e-9 else round(float(x), d) for x in np.asarray(v).ravel()]


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    return np.log(p / (1 - p))


def rk4(f, y, t1, n):
    """Classical Runge-Kutta from 0 to t1 in n steps."""
    h = t1 / n
    for _ in range(n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def t_cross(fun, y0, getter, target, h0=0.01, grow=0.02, hmax=5.0, tmax=1e6):
    """First time getter(y) reaches `target` along y' = fun(y) (RK4 with steps growing like grow * t, linear interpolation at the crossing)."""
    y = np.array(y0, float); t = 0.0; g0 = getter(y)
    while t < tmax:
        h = min(hmax, max(h0, grow * t))
        y2 = rk4(fun, y, h, 1); g1 = getter(y2)
        if g1 >= target:
            return t + h * (target - g0) / (g1 - g0)
        y, t, g0 = y2, t + h, g1
    return float("inf")


# ------------------------------------------------------------------ the perceptron with erf units: closed forms

_H = 0.004                          # E over x ~ N(0,1) by the trapezoid rule on a fine grid: exponentially accurate for these smooth integrands,
GH_X = np.arange(-12.0, 12.0 + _H / 2, _H)          # and it resolves features as narrow as phi'(8x)^2 (Gauss-Hermite with 160 nodes does not)
GH_W = np.exp(-GH_X ** 2 / 2) / math.sqrt(2 * math.pi) * _H
_ERF = np.frompyfunc(math.erf, 1, 1)
S2PI = math.sqrt(2 / math.pi)


def erfv(u):
    return np.asarray(_ERF(np.asarray(u, float)), float)


def phi(u):                         # (12.103): sqrt(2/pi) int_0^u exp(-s^2/2) ds
    return erfv(np.asarray(u, float) / math.sqrt(2))


def dphi(u):
    return S2PI * np.exp(-np.asarray(u, float) ** 2 / 2)


def ddphi(u):
    u = np.asarray(u, float)
    return -u * S2PI * np.exp(-u * u / 2)


def d3phi(u):
    u = np.asarray(u, float)
    return (u * u - 1) * S2PI * np.exp(-u * u / 2)


def Ex(vals):
    """E over x ~ N(0,1) of the values tabulated at the Gauss-Hermite nodes."""
    return float(GH_W @ vals)


def kC(a, b):                       # E[phi(a x) phi(b x)]
    return 2 / math.pi * np.arcsin(a * b / np.sqrt((1 + a * a) * (1 + b * b)))


def kCa(a, b):                      # E[phi'(a x) x phi(b x)] = dC/da
    return 2 / math.pi * b / ((1 + a * a) * np.sqrt(1 + a * a + b * b))


def kCab(a, b):                     # E[phi'(a x) x phi'(b x) x] = d2C/da db
    return 2 / math.pi * (1 + a * a + b * b) ** -1.5


def kCaa(a, b):                     # E[phi''(a x) x^2 phi(b x)] = d2C/da2
    return 2 / math.pi * b * (-2 * a * (1 + a * a) ** -2 * (1 + a * a + b * b) ** -0.5 - a * (1 + a * a) ** -1 * (1 + a * a + b * b) ** -1.5)


class Teacher:
    """f_0(x) = sum_k c_k phi(om_k x)."""

    def __init__(self, om, c):
        self.om = np.array(om, float)
        self.c = np.array(c, float)

    def f(self, x):
        return sum(c * phi(o * x) for c, o in zip(self.c, self.om))


def loss_grad(xi, T):
    """xi = (w1, w2, v1, v2). Expected loss L = (1/2) E (f - f0)^2 (the noise constant is left out) and its gradient, exactly."""
    w = xi[:2]; v = xi[2:]
    Cww = kC(w[:, None], w[None, :]); Cwo = kC(w[:, None], T.om[None, :]); Coo = kC(T.om[:, None], T.om[None, :])
    L = 0.5 * (v @ Cww @ v - 2 * v @ Cwo @ T.c + T.c @ Coo @ T.c)
    gv = Cww @ v - Cwo @ T.c
    gw = v * (kCa(w[:, None], w[None, :]) @ v - kCa(w[:, None], T.om[None, :]) @ T.c)
    return L, np.concatenate([gw, gv])


def fisher(xi):
    """G = E[grad f grad f^T] (Gaussian noise of variance 1), exactly."""
    w = xi[:2]; v = xi[2:]
    G = np.zeros((4, 4))
    Cww = kC(w[:, None], w[None, :])
    G[2:, 2:] = Cww
    G[:2, 2:] = v[:, None] * kCa(w[:, None], w[None, :])
    G[2:, :2] = G[:2, 2:].T
    G[:2, :2] = v[:, None] * v[None, :] * kCab(w[:, None], w[None, :])
    return G


def hess_loss(xi, T):
    """Hessian of L: G - E[(f0 - f) grad grad f], exactly (12.35)."""
    w = xi[:2]; v = xi[2:]
    # r_i^(0) = E[(f0 - f) phi''(w_i x) x^2], r_i^(1) = E[(f0 - f) phi'(w_i x) x]
    r0 = np.array([sum(c * kCaa(w[i], o) for c, o in zip(T.c, T.om)) - sum(v[j] * kCaa(w[i], w[j]) for j in range(2)) for i in range(2)])
    r1 = np.array([sum(c * kCa(w[i], o) for c, o in zip(T.c, T.om)) - sum(v[j] * kCa(w[i], w[j]) for j in range(2)) for i in range(2)])
    E2 = np.zeros((4, 4))
    for i in range(2):
        E2[i, i] = v[i] * r0[i]
        E2[i, 2 + i] = E2[2 + i, i] = r1[i]
    return fisher(xi) - E2


def to_zeta(xi):
    """(12.131)-(12.132): u = w2 - w1, z = (v1 - v2)/(v1 + v2), s = (v1 w1 + v2 w2)/(v1 + v2), r = v1 + v2."""
    w1, w2, v1, v2 = xi
    r = v1 + v2
    return np.array([w2 - w1, (v1 - v2) / r, (v1 * w1 + v2 * w2) / r, r])


def from_zeta(ze):
    u, z, s, r = ze
    v1 = r * (1 + z) / 2; v2 = r * (1 - z) / 2
    return np.array([s - v2 / r * u, s + v1 / r * u, v1, v2])


def zeta_T(xi):
    """T = d zeta / d xi of (12.137) (here scalar inputs)."""
    w1, w2, v1, v2 = xi
    r = v1 + v2; s = (v1 * w1 + v2 * w2) / r
    return np.array([[-1, 1, 0, 0], [0, 0, 2 * v2 / r ** 2, -2 * v1 / r ** 2],
                     [v1 / r, v2 / r, (w1 - s) / r, (w2 - s) / r], [0, 0, 1, 1]], float)


def one_neuron_optimum(T):
    """Global best single neuron r phi(s x) for the teacher: (s, r, L_1, K) with K = (r/4) E[(f0 - r phi(s x)) phi''(s x) x^2] (12.143)."""
    S = np.linspace(0.05, 8, 800)
    num = sum(c * kC(S, o) for c, o in zip(T.c, T.om))
    rho2 = num ** 2 / kC(S, S)
    j = int(rho2.argmax()); a, b = S[max(j - 2, 0)], S[min(j + 2, len(S) - 1)]

    def rho(s):
        return float(sum(c * kC(s, o) for c, o in zip(T.c, T.om)) ** 2 / kC(s, s))
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c1 = b - gr * (b - a); c2 = a + gr * (b - a)
        if rho(c1) > rho(c2):
            b = c2
        else:
            a = c1
    s = (a + b) / 2
    r = float(sum(c * kC(s, o) for c, o in zip(T.c, T.om)) / kC(s, s))
    L1 = loss_grad(np.array([s, s, r, 0.0]), T)[0]
    K = float(r / 4 * (sum(c * kCaa(s, o) for c, o in zip(T.c, T.om)) - r * kCaa(s, s)))
    return s, r, L1, K


# ------------------------------------------------------------------ integrators for the averaged flows

def flow_gd(xi, T):
    return -loss_grad(xi, T)[1]


def flow_ng(xi, T, thresh=1e-13):
    """-G^+ grad L with the smallest eigenvalues (below thresh * largest) dropped: only used very close to a singular set."""
    L, g = loss_grad(xi, T)
    lam, Q = np.linalg.eigh(fisher(xi))
    inv = np.where(lam > thresh * lam.max(), 1.0 / np.where(lam > 0, lam, 1.0), 0.0)
    return -(Q * inv) @ (Q.T @ g)


def flow_sfn(xi, T, thresh=1e-9):
    """-|H|^+ grad L: the saddle-free Newton method / natural gradient with the absolute Hessian metric (12.36)-(12.38), pseudo-inverse of |H|."""
    L, g = loss_grad(xi, T); H = hess_loss(xi, T)
    lam, Q = np.linalg.eigh((H + H.T) / 2)
    a = np.abs(lam); inv = np.where(a > thresh * a.max(), 1.0 / np.where(a > 0, a, 1.0), 0.0)
    return -(Q * inv) @ (Q.T @ g)


def _fd_jac(f, xi, T, h=1e-6):
    J = np.zeros((4, 4))
    for k in range(4):
        d = np.zeros(4); d[k] = h * max(1.0, abs(xi[k]))
        J[:, k] = (f(xi + d, T) - f(xi - d, T)) / (2 * d[k])
    return J


def ros2_step(f, xi, T, dt, jac=None):
    """Second-order, L-stable linearly implicit step (Verwer's ROS2); jac(xi, T) = Jacobian of f, finite differences if None."""
    g = 1 + 1 / math.sqrt(2)
    J = _fd_jac(f, xi, T) if jac is None else jac(xi, T)
    A = np.eye(4) - g * dt * J
    k1 = np.linalg.solve(A, f(xi, T))
    k2 = np.linalg.solve(A, f(xi + dt * k1, T) - 2 * k1)
    return xi + 1.5 * dt * k1 + 0.5 * dt * k2


def jac_gd(xi, T):
    return -hess_loss(xi, T)


def integrate_gd(xi0, T, t_end, dt0=0.01, rel=0.03, dtmax=50.0, grow=1.1, stop=None):
    """Vanilla gradient flow (12.129) with steps that grow with t; records at times growing by the factor `grow`."""
    t = 0.0; xi = xi0.copy(); out = [(0.0, xi.copy())]; nxt = 1.0
    while t < t_end:
        dt = min(dtmax, max(dt0, rel * t))
        if t + dt > t_end:
            dt = t_end - t
        xi = ros2_step(flow_gd, xi, T, dt, jac_gd); t += dt
        if t >= nxt - 1e-12 or t >= t_end - 1e-12:
            out.append((t, xi.copy())); nxt = t * grow
        if stop is not None and stop(t, xi):
            out.append((t, xi.copy())); break
    return out


def integrate_ng(xi0, T, t_end, dt, rec=None):
    """Natural gradient flow with a fixed step dt, recording every `rec` time units."""
    t = 0.0; xi = xi0.copy(); out = [(0.0, xi.copy())]; nxt = rec if rec else dt
    while t < t_end - 1e-12:
        xi = ros2_step(flow_ng, xi, T, dt); t += dt
        if t >= nxt - 1e-12:
            out.append((t, xi.copy())); nxt += rec if rec else dt
    return out


# ------------------------------------------------------------------ 1. steepest descent, covariant against contravariant

def check_steepest():
    head("1. Steepest descent in a metric and reparametrisation (section 12.1.2, (12.17)-(12.26))")
    G = np.array([[4.0, 1.2], [1.2, 0.5]]); g = np.array([1.0, 0.6])
    Lc = np.linalg.cholesky(G)
    n = 200_000
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = np.linalg.solve(Lc.T, np.stack([np.cos(ang), np.sin(ang)]))      # every a with a^T G a = 1
    dec = g @ a
    k = int(dec.argmax()); a_best = a[:, k]
    a_ng = np.linalg.solve(G, g); a_ng = a_ng / math.sqrt(a_ng @ G @ a_ng)
    best = math.sqrt(g @ np.linalg.solve(G, g))
    ge = g / math.sqrt(g @ G @ g)
    print(f"   G = {G.tolist()}, grad L = {g.tolist()}. Over {n} directions on the ellipse a^T G a = 1 (12.18) the largest decrease grad L . a is {dec.max():.6f}")
    print(f"   at a = {np.round(a_best, 4).tolist()}; G^-1 grad L normalised to a^T G a = 1 is {np.round(a_ng, 4).tolist()} (difference {np.abs(a_best - a_ng).max():.1e}) with decrease sqrt(grad L^T G^-1 grad L) = {best:.6f} (12.21)")
    print(f"   the plain gradient direction, normalised the same way, decreases L by {g @ ge:.4f}, which is {g @ ge / best:.3f} of the maximum; that ratio squared is (g.g)^2/((g^T G g)(g^T G^-1 g)) = {(g @ g) ** 2 / ((g @ G @ g) * (g @ np.linalg.solve(G, g))):.4f}")
    ts = np.linspace(0, 2 * np.pi, 361)
    STORE["steep"] = dict(G=G, g=g, a_ellipse=np.linalg.solve(Lc.T, np.stack([np.cos(ts), np.sin(ts)])), dec=g @ np.linalg.solve(Lc.T, np.stack([np.cos(ts), np.sin(ts)])),
                          a_ng=a_ng, a_gd=ge, best=best)
    # contravariant against covariant (12.24)-(12.25)
    J = np.array([[1.0, 0.3], [0.0, 2.0]])                                # xi' = J xi
    Gp = np.linalg.inv(J).T @ G @ np.linalg.inv(J); gp = np.linalg.inv(J).T @ g
    print(f"   change of chart xi' = J xi, J = {J.tolist()}: G' = J^-T G J^-1, grad' L = J^-T grad L. The natural gradient transforms as a vector: |G'^-1 grad' L - J G^-1 grad L| = {np.abs(np.linalg.solve(Gp, gp) - J @ np.linalg.solve(G, g)).max():.1e};"
          f" the plain gradient does not: |grad' L - J grad L| = {np.abs(gp - J @ g).max():.2f} (it is a covector)")
    # the Bernoulli model in two charts, cross-entropy loss to the target 0.9
    Lp = lambda p: (p - 0.9) / (p * (1 - p))                               # dL/dp for L = -0.9 log p - 0.1 log(1 - p)
    f_ng_th = lambda th: np.array([-Lp(sig(th[0]))])                       # d theta/dt = -G_theta^-1 dL/dtheta, G_theta = p(1-p)
    f_ng_p = lambda p: np.array([-p[0] * (1 - p[0]) * Lp(p[0])])           # d p/dt = -G_p^-1 dL/dp, G_p = 1/(p(1-p))
    f_gd_th = lambda th: np.array([-Lp(sig(th[0])) * sig(th[0]) * (1 - sig(th[0]))])
    f_gd_p = lambda p: np.array([-Lp(p[0])])
    th0 = logit(0.01)
    print("   Bernoulli model, cross-entropy loss with target p = 0.9, start p = 0.01, flows in the logit chart theta and in the chart p (exact solution of the natural flow: p(t) = 0.9 - 0.89 e^-t):")
    for T in (1.0, 3.0):
        p_ng_th = sig(rk4(f_ng_th, np.array([th0]), T, 2000)[0]); p_ng_p = rk4(f_ng_p, np.array([0.01]), T, 2000)[0]
        p_gd_th = sig(rk4(f_gd_th, np.array([th0]), T, 3000)[0]); p_gd_p = rk4(f_gd_p, np.array([0.01]), T, 3000)[0]
        print(f"      t = {T:.0f}: natural flow {p_ng_th:.10f} (logit chart) {p_ng_p:.10f} (p chart) exact {0.9 - 0.89 * math.exp(-T):.10f};   plain gradient flow {p_gd_th:.6f} (logit chart) {p_gd_p:.6f} (p chart)")
    tt = np.linspace(0, 6, 61)
    pg_th = []; pg_p = []; y1 = np.array([th0]); y2 = np.array([0.01])
    for i in range(len(tt)):
        pg_th.append(float(sig(y1[0]))); pg_p.append(float(y2[0]))
        y1 = rk4(f_gd_th, y1, 0.1, 20); y2 = rk4(f_gd_p, y2, 0.1, 20)
    STORE["bern"] = dict(t=tt, ng=0.9 - 0.89 * np.exp(-tt), gd_th=np.array(pg_th), gd_p=np.array(pg_p))
    t_th = t_cross(f_gd_th, np.array([th0]), lambda y: sig(y[0]), 0.89); t_p = t_cross(f_gd_p, np.array([0.01]), lambda y: y[0], 0.89)
    print(f"   time to reach p = 0.89: natural flow {math.log(0.89 / 0.01):.2f} in either chart; plain gradient flow {t_th:.1f} in the logit chart,"
          f" {t_p:.2f} in the p chart: the plain gradient flow depends on the chart, the natural one does not (it is a flow of the tangent vector field G^-1 grad L)")


# ------------------------------------------------------------------ 2. Fisher matrix against Hessian; saddle-free Newton

def check_hessian():
    head("2. Fisher matrix against Hessian, Gauss-Newton and the saddle-free Newton method (section 12.1.3, (12.29)-(12.38))")
    T = Teacher([0.4, 4.0], [10.0, 5.0])                                  # a two-neuron teacher (outside R)
    xi = np.array([0.7, 1.4, 1.2, -0.6])
    L, g = loss_grad(xi, T); G = fisher(xi); H = hess_loss(xi, T)
    # finite differences of the exact gradient
    Hfd = np.zeros((4, 4))
    for k in range(4):
        d = np.zeros(4); d[k] = 1e-5
        Hfd[:, k] = (loss_grad(xi + d, T)[1] - loss_grad(xi - d, T)[1]) / 2e-5
    # quadrature version of G and of E[(f0-f) grad grad f]
    x = GH_X
    w1, w2, v1, v2 = xi
    f = v1 * phi(w1 * x) + v2 * phi(w2 * x); f0 = T.f(x)
    grad_f = np.stack([v1 * dphi(w1 * x) * x, v2 * dphi(w2 * x) * x, phi(w1 * x), phi(w2 * x)])
    Gq = (grad_f * GH_W) @ grad_f.T
    E2 = np.zeros((4, 4))
    E2[0, 0] = Ex((f0 - f) * v1 * ddphi(w1 * x) * x ** 2); E2[1, 1] = Ex((f0 - f) * v2 * ddphi(w2 * x) * x ** 2)
    E2[0, 2] = E2[2, 0] = Ex((f0 - f) * dphi(w1 * x) * x); E2[1, 3] = E2[3, 1] = Ex((f0 - f) * dphi(w2 * x) * x)
    print(f"   (12.34)-(12.35) on the two-hidden-unit perceptron (teacher outside R) at a random point: the Hessian from differentiating the exact gradient equals G - E[(f0 - f) grad grad f]"
          f" to {np.abs(Hfd - (Gq - E2)).max():.1e} (quadrature) and the closed form G to {np.abs(G - Gq).max():.1e}")
    print(f"   there |G - H| / |G| = {np.linalg.norm(G - H) / np.linalg.norm(G):.3f}: the first equality in (12.32), G = grad grad L, would say 0. G is E over the model's own distribution of grad grad l (the Fisher identity);"
          f" the Hessian of L averages over the true distribution (12.33)")
    # teacher in R: global minimum set; points of R_o and R_e with zero loss
    Tin = Teacher([1.0], [1.0])
    for name, xi0 in (("a point of R_o (w1 = w2 = 1, v1 + v2 = 1)", np.array([1.0, 1.0, 0.7, 0.3])), ("a point of R_e (v2 = 0, w1 = 1, w2 = 2.5)", np.array([1.0, 2.5, 1.0, 0.0])),
                      ("a point with a different output function", np.array([1.2, 0.5, 0.6, 0.8]))):
        L0, g0 = loss_grad(xi0, Tin)
        print(f"   teacher in R, {name}: L = {L0:.2e}, |grad L| = {np.abs(g0).max():.1e}, |G - H| = {np.abs(fisher(xi0) - hess_loss(xi0, Tin)).max():.1e}")
    # teacher outside R, the one-neuron optimum on R_o
    s, r, L1, K = one_neuron_optimum(T)
    print(f"   teacher outside R: the best single neuron is r phi(s x) with s = {s:.4f}, r = {r:.4f} (loss L_1 = {L1:.4f}); with K = (r/4) E[(f0 - f) phi''(s x) x^2] = {K:.4f} (12.143)")
    for z in (0.6, 1.5):
        xi1 = from_zeta([0.0, z, s, r])
        g1 = loss_grad(xi1, T)[1]; G1 = fisher(xi1); H1 = hess_loss(xi1, T)
        ev = np.linalg.eigvalsh(H1)
        print(f"      point of R_o with z = {z}: |grad L| = {np.abs(g1).max():.1e} (a critical point of L), eigenvalues of G {clean(np.linalg.eigvalsh(G1))}, of H {clean(ev)},"
              f" |H - G| / |G| = {np.linalg.norm(H1 - G1) / np.linalg.norm(G1):.2f}; (1 - z^2) K = {(1 - z * z) * K:+.4f}; eigenvalues of |H| {clean(np.abs(ev))}")
    print("   so 'G and H are equal at critical or singular regions of the MLP' holds when the teacher is in R (L = 0 there) and fails otherwise: at R_o the Hessian carries the number K, the Fisher matrix does not,"
          " and H has a negative eigenvalue exactly where (1 - z^2) K > 0 (the unstable part of R_o of Theorem 12.4)")
    # saddle-free Newton
    Hs = np.diag([1.0, -1.0])
    eta = 0.1
    print("   Saddle L = (x^2 - y^2)/2, update maps xi -> xi - eta M^-1 grad L near the saddle, eigenvalues of the linearisation (a value of modulus > 1 is an escape):")
    for name, M, e in (("gradient descent, eta = 0.1", np.eye(2), eta), ("natural gradient with G = diag(2, 0.5), eta = 0.1", np.diag([2.0, 0.5]), eta),
                       ("Newton, eta = 1", Hs, 1.0), ("saddle-free Newton |H|, eta = 1", np.abs(Hs), 1.0)):
        Jm = np.eye(2) - e * np.linalg.solve(M, Hs)
        print(f"      {name:50s}: {np.round(np.linalg.eigvals(Jm).real, 3).tolist()}  ->  {'attracted to the saddle' if np.abs(np.linalg.eigvals(Jm)).max() < 1 else 'escapes along y'}")


# ------------------------------------------------------------------ 3. stochastic relaxation

def check_relaxation():
    head("3. Stochastic relaxation (section 12.1.4, (12.39)-(12.40))")
    f = np.array([0.0, 0.5, 1.0])                                          # costs of the three outcomes; outcome 0 is the minimiser
    def sm(th):
        z = np.concatenate([[0.0], th]); z = z - z.max(); e = np.exp(z); return e / e.sum()
    def gradL(th):
        p = sm(th); return p[1:] * (f[1:] - p @ f)
    def Gm(p):
        return np.diag(p[1:]) - np.outer(p[1:], p[1:])
    p0 = np.array([0.001, 0.998, 0.001]); th0 = np.log(p0[1:] / p0[0])
    nat = np.linalg.solve(Gm(p0), gradL(th0))
    print(f"   p = {p0.tolist()} on three outcomes with costs {f.tolist()}, theta_i = log(p_i/p_0): the natural gradient of L = E_p[f] is G^-1 grad L = {np.round(nat, 12).tolist()}, exactly f_i - f_0 = {(f[1:] - f[0]).tolist()}"
          f" at every p (L is linear in the expectation parameters eta_i = p_i, and G = d eta/d theta). Check at 5 random points: largest deviation"
          f" {max(np.abs(np.linalg.solve(Gm(sm(t)), gradL(t)) - (f[1:] - f[0])).max() for t in RNG.normal(size=(5, 2)) * 3):.1e}")
    ng = lambda th: -(f[1:] - f[0])
    gd = lambda th: -gradL(th)
    th_T = rk4(ng, th0.copy(), 7.0, 1400)
    tilt = p0 * np.exp(-7.0 * f); tilt = tilt / tilt.sum()
    print(f"   the natural gradient flow is theta' = -(f_i - f_0): a straight line in theta, p_t proportional to p_0 exp(-t f) (Boltzmann reweighting): at t = 7 the integrated flow gives {np.round(sm(th_T), 8).tolist()}, the closed form {np.round(tilt, 8).tolist()}")
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (p0 * np.exp(-mid * f) / (p0 * np.exp(-mid * f)).sum())[0] < 0.9:
            lo = mid
        else:
            hi = mid
    t_gd = t_cross(gd, th0.copy(), lambda th: sm(th)[0], 0.9, h0=0.01, grow=0.01, hmax=5.0)
    print(f"   time until the best outcome has probability 0.9, starting at the wrong vertex p = (0.001, 0.998, 0.001): natural gradient {lo:.2f} (closed form), plain gradient in theta {t_gd:.0f}, {t_gd / lo:.0f} times longer;"
          f" the plain gradient p_j (f_j - L) carries the factor p_j that vanishes near a vertex")
    # trajectories for the figure: several starts in the triangle
    starts = [np.array(p) for p in ([0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.3, 0.3, 0.4], [0.02, 0.6, 0.38])]
    trajs = []
    for p in starts:
        th = np.log(p[1:] / p[0]); ngp = [p.copy()]; gdp = [p.copy()]; tg = th.copy(); tn = th.copy()
        for i in range(120):
            tg = rk4(gd, tg, 0.5, 5); tn = rk4(ng, tn, 0.5 / 4, 5)
            gdp.append(sm(tg)); ngp.append(sm(tn))
        trajs.append((np.array(ngp), np.array(gdp)))
    STORE["relax"] = dict(f=f, trajs=trajs, starts=starts)


# ------------------------------------------------------------------ 4. natural policy gradient

def check_policy():
    head("4. Natural policy gradient (section 12.1.5, (12.41)-(12.56))")
    rng = np.random.default_rng(11)
    nS, nA, gam = 3, 3, 0.9
    P = rng.dirichlet(np.ones(nS), size=(nS, nA)); Rw = rng.normal(size=(nS, nA)); mu0 = np.array([1.0, 0, 0])

    def policy(theta):                                                     # softmax over actions, logit of action 0 fixed at 0
        z = np.concatenate([np.zeros((nS, 1)), theta], 1); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)

    def evaluate(theta):
        pi = policy(theta)
        Ppi = np.einsum('xu,xuy->xy', pi, P); rpi = (pi * Rw).sum(1)
        V = np.linalg.solve(np.eye(nS) - gam * Ppi, rpi)
        Q = Rw + gam * np.einsum('xuy,y->xu', P, V)
        d = np.linalg.solve((np.eye(nS) - gam * Ppi).T, mu0)                # d(x) = sum_t gamma^t p(x_t), (12.45)
        return pi, V, Q, d, float(mu0 @ V)

    def feat(x, u, pi):                                                    # a(x, u) = grad_theta log pi(u|x)  (12.50)
        a = np.zeros((nS, nA - 1))
        for j in range(1, nA):
            a[x, j - 1] = (1.0 if u == j else 0.0) - pi[x, j]
        return a.ravel()

    theta = rng.normal(size=(nS, nA - 1)) * 0.5
    pi, V, Q, d, J = evaluate(theta)
    g = sum(d[x] * pi[x, u] * feat(x, u, pi) * Q[x, u] for x in range(nS) for u in range(nA))
    gfd = np.zeros_like(g); h = 1e-6
    for i in range(g.size):
        dt = np.zeros(g.size); dt[i] = h
        gfd[i] = (evaluate(theta + dt.reshape(theta.shape))[4] - evaluate(theta - dt.reshape(theta.shape))[4]) / (2 * h)
    G = sum(d[x] * pi[x, u] * np.outer(feat(x, u, pi), feat(x, u, pi)) for x in range(nS) for u in range(nA))
    A = np.array([feat(x, u, pi) for x in range(nS) for u in range(nA)]); Y = np.array([Q[x, u] for x in range(nS) for u in range(nA)])
    W = np.array([d[x] * pi[x, u] for x in range(nS) for u in range(nA)])
    M = A.T @ (W[:, None] * A); w = np.linalg.solve(M, A.T @ (W * Y))
    print(f"   random MDP with {nS} states, {nA} actions, gamma = {gam}, softmax policy with {g.size} free logits: J = {J:.4f}; the policy gradient formula (12.51) against finite differences of J: relative error {np.linalg.norm(g - gfd) / np.linalg.norm(g):.1e}")
    print(f"   with a = grad log pi, w = weighted least-squares fit of Q on a under the weights d(x) pi(u|x): grad J = G w to {np.linalg.norm(g - G @ w) / np.linalg.norm(g):.1e} (12.52); G = sum d F equals A^T W A to {np.linalg.norm(G - M):.1e} (12.46)-(12.47)")
    nat = np.linalg.solve(G, g).reshape(nS, nA - 1)
    print(f"   natural gradient G^-1 grad J = w (12.53); for the tabular softmax it is the table of Q(x,u) - Q(x,0): max |G^-1 grad J - (Q(x,u) - Q(x,0))| = {np.abs(nat - (Q[:, 1:] - Q[:, [0]])).max():.1e}, whatever the state weights d(x) are")
    a_bar = np.array([np.abs(sum(pi[x, u] * feat(x, u, pi) for u in range(nA))).max() for x in range(nS)])
    print(f"   (12.56): a(x) = int pi(u|x) a(x,u) du is the mean of the score, identically 0: largest entry over the states {a_bar.max():.1e}, so the update (12.55), w <- w + alpha delta_t a(x_t), as printed, never moves w")
    Adv = Q - V[:, None]
    Ebar = sum(W[i] * Adv.ravel()[i] * A[i] for i in range(len(W)))
    wl = np.zeros(g.size)
    for _ in range(4000):
        wl = wl + 0.5 * (Ebar - M @ wl)
    print(f"   what does work: the least-mean-squares recursion w <- w + alpha (delta_t - a . w) a(x_t, u_t), whose mean is Ebar - M w, with delta_t having mean advantage Q - V: its fixed point is G^-1 grad J to"
          f" {np.linalg.norm(wl - np.linalg.solve(G, g)) / np.linalg.norm(wl):.1e};"
          f" without the - a . w term the expected update is E[delta a] = grad J (error {np.linalg.norm(Ebar - g) / np.linalg.norm(g):.1e}), the vanilla policy gradient")


# ------------------------------------------------------------------ 5. mirror descent

def check_mirror():
    head("5. Mirror descent and natural gradient (section 12.1.6, (12.57)-(12.62))")
    fp = lambda th: th - 1.5                                                # f(theta) = (theta - 1.5)^2 / 2
    th0 = -0.5
    print("   Bernoulli model, psi(theta) = log(1 + e^theta), eta = sigma(theta), G = sigma'(theta); f = (theta - 1.5)^2/2; one step from theta = -0.5:")
    print("      eps      mirror step (12.59)-(12.60)   natural step (12.62)    difference    difference / eps^2")
    for eps in (0.02, 0.01, 0.005, 0.0025):
        eta = sig(th0) - eps * fp(th0); th_md = float(logit(eta))
        th_ng = th0 - eps * fp(th0) / (sig(th0) * (1 - sig(th0)))
        print(f"      {eps:<7} {th_md:+.8f}                   {th_ng:+.8f}            {abs(th_md - th_ng):.2e}      {abs(th_md - th_ng) / eps ** 2:.2f}")
    g0 = sig(th0) * (1 - sig(th0)); g1 = g0 * (1 - 2 * sig(th0))
    print(f"   so (12.61)-(12.62) hold to first order in eps; the difference is second order, with the coefficient (1/2) G'/G^3 f'^2 = {0.5 * g1 / g0 ** 3 * fp(th0) ** 2:.2f} (the second-order term of the Legendre map).")
    f_ng = lambda y: np.array([-fp(y[0]) / (sig(y[0]) * (1 - sig(y[0])))])
    f_md = lambda e: np.array([-fp(float(logit(e[0])))])
    a = rk4(f_ng, np.array([-0.5]), 1.0, 4000)[0]; b = float(logit(rk4(f_md, np.array([sig(-0.5)]), 1.0, 4000)[0]))
    print(f"   continuous time: the natural gradient flow and the mirror flow d eta/dt = -f'(theta) are the same flow: theta(1) = {a:.12f} and {b:.12f} from theta(0) = -0.5")
    # other charts: theta = h(tau); the mirror map becomes eta~ = d psi~/d tau with psi~(tau) = psi(h(tau)), convex when h'' >= 0
    def mirror_flow_chart(h, hp, hpp, tau0, T=1.0, n=400):
        sg = lambda t: sig(h(t))

        def tau_of(et, tg):
            ta = tg
            for _ in range(8):
                F = sg(ta) * hp(ta) - et
                dF = sg(ta) * (1 - sg(ta)) * hp(ta) ** 2 + sg(ta) * hpp(ta)
                ta = ta - F / dF
            return ta
        et = sg(tau0) * hp(tau0); tau = tau0; dt = T / n

        def rhs(e_, tg):
            ta = tau_of(e_, tg); return -fp(h(ta)) * hp(ta), ta
        for _ in range(n):
            k1, t1 = rhs(et, tau); k2, t2 = rhs(et + 0.5 * dt * k1, t1); k3, t3 = rhs(et + 0.5 * dt * k2, t2); k4, t4 = rhs(et + dt * k3, t3)
            et = et + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tau = t4
        return h(tau)
    a0 = rk4(f_ng, np.array([-0.5]), 1.0, 2000)[0]
    th_aff = mirror_flow_chart(lambda t: 2 * t + 1, lambda t: 2.0 + 0 * t, lambda t: 0.0 * t, -0.75)
    th_quad = mirror_flow_chart(lambda t: t + 0.3 * t * t, lambda t: 1 + 0.6 * t, lambda t: 0.6 + 0 * t, (-1 + math.sqrt(1 + 1.2 * (-0.5))) / 0.6)
    print(f"   chart dependence (the book: 'both eta and grad f are covariant, so it is invariant'), start theta(0) = -0.5, t = 1: the natural gradient flow gives {a0:.9f} in every chart;"
          f" the mirror flow with the potential psi(h(tau)) gives {th_aff:.9f} for the affine chart theta = 2 tau + 1 (the same) but {th_quad:.6f} for theta = tau + 0.3 tau^2 (convex in tau, a smooth change of chart)."
          f" The mirror map eta = grad psi is tied to the flat chart in which psi is the potential; the natural flow is not")


# ------------------------------------------------------------------ 6. Theorem 12.1: online natural gradient is efficient

LAM = np.array([1.0, 0.04])          # eigenvalues of the input covariance (= G) of the Gaussian linear model


def exact_cov_recursion(variants, T, rec):
    """Exact error variances (eigen-directions of G) of online learning of y = xi.x + eps, x ~ N(0, diag(LAM)), eps ~ N(0,1), learning constant c/(t + t0).
    variants: list of (kind, c, t0), kind 'NG' (gain times G^-1) or 'GD'. Returns {t: array(nv, 2)} of t * lambda_a * V_a, the ratio to the Cramer-Rao value 1/(t lambda_a).
    One step is  V -> V - 2 eta P Lam V + eta^2 [2 P Lam V Lam P + P Lam P tr(Lam V)] + eta^2 P Lam P  (Isserlis' formula for the fourth moment of x), P = G^-1 or 1."""
    lam = LAM
    ng = np.array([k == "NG" for k, c, t0 in variants]); c = np.array([c for k, c, t0 in variants], float); t0 = np.array([t0 for k, c, t0 in variants], float)
    V = np.ones((len(variants), 2)); out = {}
    for t in range(1, T + 1):
        eta = (c / (t + t0))[:, None]
        trSV = (V * lam).sum(1)[:, None]
        Vng = V * (1 - 2 * eta + 2 * eta ** 2) + eta ** 2 * (trSV / lam + 1.0 / lam)
        Vgd = V * (1 - 2 * eta * lam + 2 * eta ** 2 * lam ** 2) + eta ** 2 * (lam * trSV + lam)
        V = np.where(ng[:, None], Vng, Vgd)
        if t in rec:
            out[t] = t * lam * V
    return out


_LOG_S = np.linspace(0.0, 14.0, 1401)
_u = _LOG_S[:, None] * GH_X[None, :]
_sp = sig(_u) * (1 - sig(_u))
_LOG_A = (_sp * _u ** 2 * GH_W).sum(1)
_LOG_B = (_sp * GH_W).sum(1)


def logistic_fisher(xi, sx):
    """G(xi) = E[sigma'(xi.x) x x^T] for x ~ N(0, diag(sx^2)); xi has shape (R, 2). One-dimensional quadrature along xi (tabulated)."""
    Sig = np.diag(sx ** 2)
    Sx = xi * sx ** 2
    s2 = (Sx * xi).sum(1); s = np.sqrt(np.maximum(s2, 1e-12))
    a = np.interp(s, _LOG_S, _LOG_A); b = np.interp(s, _LOG_S, _LOG_B)
    outer = Sx[:, :, None] * Sx[:, None, :]
    return (a / s2 ** 2)[:, None, None] * outer + b[:, None, None] * (Sig[None] - outer / s2[:, None, None])


def check_online():
    head("6. Online natural gradient is Fisher efficient (section 12.1.7.1, Theorem 12.1, (12.63)-(12.70)); adaptive G^-1 (12.84)-(12.85)")
    # (12.66): the recursion for the covariance; exact on the Gaussian linear model, checked by simulation
    variants = [("NG", 1.0, 0), ("NG", 0.75, 0), ("NG", 2.0, 0), ("NG", 0.5, 0), ("GD", 1.0, 100), ("GD", 12.5, 100), ("GD", 25.0, 100)]
    rec = tuple(sorted(set([30, 100, 300, 1000, 3000, 10000, 30000] + [int(round(10 ** e)) for e in np.linspace(1, 4.45, 36)])))
    out = exact_cov_recursion(variants, 30000, rec)
    STORE["online_exact"] = dict(variants=variants, t=np.array(rec), R=np.array([out[t] for t in rec]))
    # Monte Carlo check of the recursion at t = 30 (NG, c = 1 and GD, c = 25)
    R, T = 200_000, 30
    rng = np.random.default_rng(31)
    xi = {"NG": np.ones((R, 2)), "GD": np.ones((R, 2))}
    for t in range(1, T + 1):
        x = rng.normal(size=(R, 2)) * np.sqrt(LAM); y = rng.normal(size=R)
        for kind, c, t0 in (("NG", 1.0, 0), ("GD", 25.0, 100)):
            r = y - (xi[kind] * x).sum(1)
            step = (c / (t + t0)) * r[:, None] * x
            xi[kind] = xi[kind] + (step / LAM if kind == "NG" else step)
    ex = exact_cov_recursion([("NG", 1.0, 0), ("GD", 25.0, 100)], T, (T,))[T]
    mc = np.array([T * LAM * (xi["NG"] ** 2).mean(0), T * LAM * (xi["GD"] ** 2).mean(0)])
    print(f"   Gaussian linear model y = xi.x + noise, x ~ N(0, diag(1, 0.04)) so G = diag(1, 0.04) and the Cramer-Rao value of t Var is G^-1; start error (1, 1). The exact one-step recursion for the error covariance (Isserlis for the 4th moment)"
          f" is checked by simulating {R} runs: at t = {T} t*lambda*Var = {np.round(mc[0], 3).tolist()} (natural, c = 1), {np.round(mc[1], 3).tolist()} (plain, c = 25) against the recursion {np.round(ex[0], 3).tolist()}, {np.round(ex[1], 3).tolist()}"
          f" (the same quantity: 1 would be the Cramer-Rao bound)")
    print("   ratio  t*lambda_a*Var_a  to the Cramer-Rao bound, direction a = 1 (lambda 1), 2 (lambda 0.04), learning constant c/(t + t0):")
    print("      algorithm                  t=100          t=1000         t=10^4         t=3*10^4       asymptotic prediction")
    pred = {("NG", 1.0): "1, 1", ("NG", 0.75): "c^2/(2c-1) = 1.125", ("NG", 2.0): "c^2/(2c-1) = 1.333", ("NG", 0.5): "diverges (c = 1/2)", ("GD", 1.0): "(c lam)^2/(2 c lam - 1): 1.0, c lam = 0.04: t^(1 - 2 c lam) growth",
            ("GD", 12.5): "6.51, c lam = 1/2: log growth", ("GD", 25.0): "12.76, 1.00"}
    for i, (kind, c, t0) in enumerate(variants):
        cells = "   ".join(f"{out[t][i, 0]:6.3f} {out[t][i, 1]:7.3f}" for t in (100, 1000, 10000, 30000))
        print(f"      {kind} c = {c:<5} t0 = {t0:<3}      {cells}      {pred[(kind, c)]}")
    print("   Theorem 12.1 is the case NG, c = 1: both directions tend to 1 (1.0029, 1.0028 at t = 3*10^4). With c/t and c != 1 the limit is c^2/(2c-1) > 1 (c > 1/2 needed); the plain gradient cannot do this in both directions because"
          " its factor is (c lambda_a)^2/(2 c lambda_a - 1), minimal only at c lambda_a = 1: with c = 25 it is 12.76 in the steep direction and 1 in the flat one")
    # batch: ordinary least squares has E Cov = G^-1/(t - k - 1) in this model
    print(f"   batch least squares (the MLE) in the same model: E Cov = G^-1/(t - k - 1), ratio t/(t - 3) = {10000 / 9997:.4f} at t = 10^4 (k = 2); the online value is 1.0084, so online loses 0.8 percent at t = 10^4 and 0.3 percent at 3*10^4: the O(1/t) cost of a start at error 1")
    # logistic regression
    rng = np.random.default_rng(20261002)
    sx = np.array([1.0, 1.0 / 3.0]); xi0 = np.array([1.0, 2.0])
    G0 = logistic_fisher(xi0[None], sx)[0]; CR = np.linalg.inv(G0); lam, Vv = np.linalg.eigh(G0)
    R, T, T0, cgd = 600, 1000, 20, 40.0
    xi_ng = np.tile(xi0, (R, 1)); xi_gd = xi_ng.copy(); xi_fo = xi_ng.copy(); xi_sm = xi_ng.copy()
    Gi_fo = np.tile(0.5 * CR, (R, 1, 1)); Gi_sm = Gi_fo.copy()
    sumg = np.zeros((R, 2)); Xs = np.empty((R, T, 2)); Ys = np.empty((R, T)); snap = {}
    for t in range(1, T + 1):
        x = rng.normal(size=(R, 2)) * sx; p0 = sig(x @ xi0); y = (rng.random(R) < p0).astype(float)
        Xs[:, t - 1] = x; Ys[:, t - 1] = y
        sumg += (y - p0)[:, None] * x
        eta = 1.0 / (t + T0)
        p = sig((xi_ng * x).sum(1)); xi_ng = xi_ng + eta * np.linalg.solve(logistic_fisher(xi_ng, sx), ((y - p)[:, None] * x)[:, :, None])[:, :, 0]
        p = sig((xi_gd * x).sum(1)); xi_gd = xi_gd + (cgd / (t + T0)) * (y - p)[:, None] * x
        eps = 1.0 / (t + 200)
        p = sig((xi_fo * x).sum(1)); sc = (y - p)[:, None] * x; v = np.einsum('rab,rb->ra', Gi_fo, sc); xi_fo = xi_fo + eta * v
        Gi_fo = (1 + eps) * Gi_fo - eps * v[:, :, None] * v[:, None, :]
        p = sig((xi_sm * x).sum(1)); sc = (y - p)[:, None] * x; v = np.einsum('rab,rb->ra', Gi_sm, sc); xi_sm = xi_sm + eta * v
        q = (sc * v).sum(1)
        Gi_sm = (Gi_sm - (eps / (1 - eps + eps * q))[:, None, None] * v[:, :, None] * v[:, None, :]) / (1 - eps)
        if t in (250, 500, 1000):
            snap[t] = (xi_ng.copy(), xi_gd.copy(), xi_fo.copy(), xi_sm.copy(), sumg.copy(), Gi_fo.copy(), Gi_sm.copy())

    def mle(Xt, Yt):
        xi = np.tile(xi0, (Xt.shape[0], 1))
        for _ in range(8):
            pz = sig((Xt * xi[:, None, :]).sum(2))
            g = ((Yt - pz)[:, :, None] * Xt).sum(1)
            H = np.einsum('rt,rta,rtb->rab', pz * (1 - pz), Xt, Xt)
            xi = xi + np.linalg.solve(H, g[:, :, None])[:, :, 0]
        return xi
    print(f"   Logistic regression P(y=1|x) = sigma(xi.x), x ~ N(0, diag(1, 1/9)), xi_0 = (1, 2): G(xi_0) has eigenvalues {np.round(lam, 5).tolist()}, Cramer-Rao G^-1 = {np.round(CR, 3).tolist()}."
          f" {R} runs of {T} steps started at xi_0 (the local regime of the theorem), gain 1/(t + {T0}) for the natural variants, {cgd:g}/(t + {T0}) for the plain gradient.")
    print("   ratio of t*Cov to the Cramer-Rao value in the two eigen-directions of G(xi_0) (flat, steep), with the control variate E[S S^T] = G^-1/t for the linearised estimator S = G^-1 mean(score), so the standard errors are about 0.01 for the natural variants;")
    print("   'linear' is the same recursion run on the linearised model (a scalar recursion per direction), which is what the theory predicts for this gain schedule:")
    STORE["logistic"] = {}
    for t in (250, 500, 1000):
        ng, gd, fo, sm, sg, Gifo, Gism = snap[t]
        Xt = Xs[:, :t]; Yt = Ys[:, :t]
        xm = mle(Xt, Yt)
        S = (CR @ sg.T).T / t
        def ratio(est):
            e = est - xi0
            C1 = t * (e.T @ e - S.T @ S) / R + CR
            return np.diag(Vv.T @ C1 @ Vv) * lam
        def twin(kind):
            v = np.zeros(2)
            for s_ in range(1, t + 1):
                if kind == "ng":
                    eta_s = 1.0 / (s_ + T0); v = (1 - eta_s) ** 2 * v + eta_s ** 2 / lam
                else:
                    eta_s = cgd / (s_ + T0); v = (1 - eta_s * lam) ** 2 * v + eta_s ** 2 * lam
            return t * lam * v
        rows = {"natural, exact G(xi_t)": ratio(ng), "adaptive (12.85), first order": ratio(fo), "adaptive, Sherman-Morrison": ratio(sm), f"plain gradient c = {cgd:g}": ratio(gd), "batch MLE": ratio(xm)}
        STORE["logistic"][t] = rows
        print(f"      t = {t}: " + ";  ".join(f"{k} {np.round(v, 3).tolist()}" for k, v in rows.items()) + f";  linear theory: natural {np.round(twin('ng'), 3).tolist()}, plain {np.round(twin('gd'), 3).tolist()}")
        if t == 1000:
            for name, Gi in (("first-order (12.85)", Gifo), ("Sherman-Morrison", Gism)):
                err = np.linalg.norm(Gi - CR[None], axis=(1, 2)) / np.linalg.norm(CR)
                print(f"      estimate of G^-1 by the {name} recursion at t = 1000: median relative error {np.median(err):.3f}; fraction of runs with a non-positive-definite estimate {np.mean(np.linalg.eigvalsh(Gi)[:, 0] <= 0):.3f}")
    # the printed (12.85) against Sherman-Morrison, and the positive-definiteness trouble
    A_ = rng.normal(size=(2, 2)); Gt = A_ @ A_.T + 0.5 * np.eye(2); sv = rng.normal(size=2)
    print("   (12.85) is not a Taylor expansion of G(xi_{t+1}) but the first-order-in-eps inverse of the running average G <- (1 - eps) G + eps s s^T of the gradient outer products (E[s s^T] = G); against the exact inverse the error is O(eps^2):")
    for eps in (0.1, 0.05, 0.025):
        exact = np.linalg.inv((1 - eps) * Gt + eps * np.outer(sv, sv)); Gi = np.linalg.inv(Gt)
        first = (1 + eps) * Gi - eps * Gi @ np.outer(sv, sv) @ Gi
        print(f"      eps = {eps}: |exact - (12.85)| = {np.abs(exact - first).max():.2e}, divided by eps^2: {np.abs(exact - first).max() / eps ** 2:.3f}")
    def pd_fraction(offset, R=3000, T=200):
        rng2 = np.random.default_rng(5); bad = np.zeros(R, bool)
        xi = np.tile(xi0, (R, 1)); Gi = np.tile(0.5 * CR, (R, 1, 1))
        for t in range(1, T + 1):
            x = rng2.normal(size=(R, 2)) * sx; y = (rng2.random(R) < sig(x @ xi0)).astype(float)
            p = sig((xi * x).sum(1)); sc = (y - p)[:, None] * x; v = np.einsum('rab,rb->ra', Gi, sc)
            xi = xi + (1.0 / (t + T0)) * v
            e = 1.0 / (t + offset)
            Gi = (1 + e) * Gi - e * v[:, :, None] * v[:, None, :]
            ev = np.linalg.eigvalsh(np.where(np.isfinite(Gi), Gi, 0.0))[:, 0]
            bad |= ev <= 0
            Gi[bad] = 0.5 * CR                                              # reset so that the run goes on
        return float(bad.mean())
    print(f"   positive definiteness of the first-order recursion (12.85): fraction of {3000} runs in which G^-1 loses positive definiteness within 200 steps: {pd_fraction(10):.3f} with eps_t = 1/(t + 10), {pd_fraction(200):.3f} with eps_t = 1/(t + 200);"
          f" the Sherman-Morrison form stays positive definite by construction")


# ------------------------------------------------------------------ 7. saturation

def check_saturation():
    head("7. Saturation and Theorem 12.2 (section 12.1.7.2, (12.74)-(12.82)) on one erf neuron")
    w0 = 1.0
    print("   f = phi(w x), x ~ N(0,1), y = phi(w_0 x) + noise of variance 1, w_0 = 1. G(w) = E[phi'(wx)^2 x^2] = (2/pi)(1 + 2w^2)^(-3/2); G_bar = E_{w_0}[grad l^2] = G + E[(f_0 - f)^2 phi'^2 x^2]:")
    print("      w      G(w)       E|grad l|^2 (= G_bar)   E|nat. grad|^2 = tr(G_bar G^-1)   E (G^-1 grad l)^2 = G_bar/G^2   dL/dw")
    rows = []
    for w in (0.5, 1.0, 2.0, 4.0, 8.0):
        G = float(kCab(w, w)); f = phi(w * GH_X); f0 = phi(w0 * GH_X)
        Gbar = Ex(((f0 - f) ** 2 + 1.0) * dphi(w * GH_X) ** 2 * GH_X ** 2)
        Lp = float(kCa(w, w) - kCa(w, w0))
        rows.append((w, G, Gbar, Gbar / G, Gbar / G ** 2, Lp))
        print(f"      {w:<5} {G:.5f}    {Gbar:.5f}               {Gbar / G:.4f}                         {Gbar / G ** 2:10.1f}                   {Lp:+.5f}")
    STORE["sat"] = rows
    rng = np.random.default_rng(1); w = 2.0; N = 2_000_000
    x = rng.normal(size=N); y = phi(w0 * x) + rng.normal(size=N)
    gl = -(y - phi(w * x)) * dphi(w * x) * x
    print(f"   Theorem 12.2 (12.80): at w = 2 the Monte Carlo mean of (grad l)^2 / G is {np.mean(gl * gl) / kCab(w, w):.4f} (2*10^6 samples), tr(G_bar G^-1) = {rows[2][3]:.4f}; at w = w_0 it is exactly 1 = k (12.82)."
          " The natural gradient's Riemannian length never falls below k = 1, however flat the neuron is (the plain gradient's squared length G_bar falls like w^-3), but its squared length in parameter units,"
          f" G_bar/G^2 (next-to-last column), grows like w^3: a step of {math.sqrt(rows[3][4]):.0f} units per sample at w = 4 and {math.sqrt(rows[4][4]):.0f} at w = 8 for a gain of 1")
    def iters(w, eta, kind, tol=0.05, maxit=100000):
        for k in range(maxit):
            if abs(w - w0) < tol:
                return k
            Lp = float(kCa(w, w) - kCa(w, w0))
            w = w - eta * Lp if kind == "gd" else w - eta * Lp / float(kCab(w, w))
        return maxit
    print("   averaged (expected-loss) iterations from w = 4 and w = 6 until |w - 1| < 0.05:")
    res = {}
    for start in (4.0, 6.0):
        for kind, eta in (("gd", 1.0), ("gd", 5.0), ("ng", 1.0), ("ng", 0.5), ("ng", 0.2)):
            res[(start, kind, eta)] = iters(start, eta, kind)
        print(f"      from w = {start:.0f}: plain gradient eta = 1: {res[(start, 'gd', 1.0)]} steps, eta = 5: {res[(start, 'gd', 5.0)]};  natural gradient eta = 1: {res[(start, 'ng', 1.0)]}, eta = 0.5: {res[(start, 'ng', 0.5)]}, eta = 0.2: {res[(start, 'ng', 0.2)]}")
    # trajectories for the figure (start w = 4)
    traj = {}
    for kind, eta in (("gd", 1.0), ("ng", 0.5), ("ng", 0.2)):
        w = 4.0; tr = [w]
        for k in range(300):
            Lp = float(kCa(w, w) - kCa(w, w0)); w = w - eta * Lp if kind == "gd" else w - eta * Lp / float(kCab(w, w)); tr.append(w)
        traj[(kind, eta)] = np.array(tr)
    STORE["sat_traj"] = traj
    # online simulation from a saturated start
    def sim(kind, eta, R=400, T=400, w_start=4.0):
        rng = np.random.default_rng(3)
        w = np.full(R, w_start); out = {}
        for t in range(1, T + 1):
            x = rng.normal(size=R); y = phi(w0 * x) + rng.normal(size=R)
            g = -(y - phi(w * x)) * dphi(w * x) * x
            w = w - eta * g if kind == "gd" else w - eta * g / kCab(w, w)
            if t in (10, 30, 100, 400):
                out[t] = np.abs(w - w0)
        return out
    print("   online learning from w = 4, one sample per step, 400 runs: median |w - w_0| (and the fraction of runs within 0.3) at t = 10, 30, 100, 400:")
    STORE["sat_online"] = {}
    for kind, eta in (("gd", 1.0), ("ng", 0.05), ("ng", 0.2)):
        o = sim(kind, eta)
        STORE["sat_online"][(kind, eta)] = {t: float(np.median(v)) for t, v in o.items()}
        print(f"      {'plain gradient' if kind == 'gd' else 'natural gradient'}, eta = {eta}: " + "; ".join(f"t = {t}: {np.median(v):.3f} ({np.mean(v < 0.3):.2f})" for t, v in o.items()))
    print("   the natural gradient with a small gain reaches the neighbourhood of w_0 in about 30 steps while the plain gradient needs hundreds; with the larger gain 0.2 the first saturated sample throws w far away (the step is G^-1 grad l ~ w^3) and the runs never recover:"
          " 'free of saturation' is true of the direction, and the gain has to be small enough for the step in parameter units")


# ------------------------------------------------------------------ 8. the learning constant

def check_adaptive_eta():
    head("8. The adaptive learning constant (section 12.1.7.5, (12.87)-(12.100))")
    def ode(alpha, beta, e0=0.5, eta0=0.5, tend=1e6):
        def f(y):
            e, eta = y; return np.array([-2 * eta * e, alpha * beta * eta * e - alpha * eta * eta])
        y = np.array([e0, eta0]); t = 0.0; out = {}; nxt = 1e3
        while t < tend:
            dt = min(max(1e-3, 0.005 * t), 1e3)
            k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
            if t >= nxt:
                out[nxt] = (t * y[0], t * y[1]); nxt *= 10
        return out
    print("   the averaged equations (12.97)-(12.98), t*e_t and t*eta_t at t = 10^3, 10^4, 10^5, 10^6 (the book: t e_t -> (1/beta)(1/2 - 1/alpha), t eta_t -> 1/2):")
    STORE["eta_ode"] = {}
    for alpha, beta in ((4.0, 1.0), (3.0, 2.0), (2.0, 1.0), (1.5, 1.0)):
        o = ode(alpha, beta)
        pr = (1 / beta) * (0.5 - 1 / alpha)
        STORE["eta_ode"][(alpha, beta)] = o
        print(f"      alpha = {alpha}, beta = {beta}: formula (12.99) gives {pr:+.4f}; computed t*e: " + ", ".join(f"{o[t][0]:.4f}" for t in sorted(o)) + "; t*eta: " + ", ".join(f"{o[t][1]:.4f}" for t in sorted(o)))
    print("   (12.99) needs alpha > 2: for alpha <= 2 it is zero or negative, while e_t is a square. For alpha < 2 the balance is different: eta_t ~ 1/(alpha t) (here 0.667 for alpha = 1.5) and e_t decays faster than 1/t (like t^(-2/alpha)).")
    rng = np.random.default_rng(5)
    def sim(alpha, beta, sigma, R=400, T=10000, xi1=0.7, eta1=0.05):
        xi = np.full(R, xi1); eta = np.full(R, eta1); out = {}
        for t in range(1, T + 1):
            x = rng.normal(size=R); y = sigma * rng.normal(size=R)
            r = y - xi * x; l = 0.5 * r * r
            xi = xi + eta * r * x
            eta = eta * np.exp(np.clip(alpha * (beta * l - eta), -30, 30))
            if t in (100, 1000, 10000):
                out[t] = (np.mean(t * 0.5 * xi ** 2), np.median(t * eta), np.median(eta))
        return out
    print("   the stochastic rule (12.89)-(12.90) on y = xi x + noise, x ~ N(0,1) (G = 1), true xi_0 = 0, start xi = 0.7, eta = 0.05, 400 runs: mean of t*e_t, median of t*eta_t, median eta_t:")
    for alpha, beta, sigma in ((4.0, 0.05, 0.0), (1.5, 0.05, 0.0), (4.0, 0.05, 0.3)):
        o = sim(alpha, beta, sigma)
        note = f"(12.99)-(12.100) predict t*e = {(1 / beta) * (0.5 - 1 / alpha):.2f}, t*eta = 0.5" if (alpha > 2 and sigma == 0) else (f"eta tends to alpha-independent beta sigma^2/2 = {beta * sigma ** 2 / 2:.5f}, no 1/t decay" if sigma > 0 else "alpha < 2: eta ~ 1/(alpha t) = 0.667/t")
        STORE.setdefault("eta_sim", {})[(alpha, beta, sigma)] = o
        print(f"      alpha = {alpha}, beta = {beta}, noise sd = {sigma}: " + "; ".join(f"t = {t}: {o[t][0]:.3f}, {o[t][1]:.3f}, {o[t][2]:.2e}" for t in sorted(o)) + f"   [{note}]")
    print("   In the noise-free (realisable) case the book's averaged analysis is confirmed, t eta_t -> 1/2 being the boundary value c = 1/2 of the c/t schedules of section 6. With noise the instantaneous loss never goes to 0, (12.95) replaces <l> by the error e_t"
          " (it assumes the minimum loss is 0), and eta_t stalls at beta sigma^2/2: the rule then tracks a moving target instead of converging.")



# ------------------------------------------------------------------ the teachers of part B

T_IN = Teacher([1.0], [1.0])                       # one neuron: the teacher is inside the critical region R
T_POS = Teacher([0.4, 4.0], [10.0, 5.0])           # two neurons, K > 0
T_NEG = Teacher([0.3, 1.0], [10.0, -10.0])         # two neurons, K < 0


def slow_manifold_rate(z, s, r, K):
    """Rate lambda(z) of u on the slow manifold of the plain gradient flow near R_o (my derivation, see the notes): the fast variables (s, r) are not
    slaved to dL/ds = dL/dr = 0 but follow a manifold tilted linearly in u. With H the Hessian of the one-neuron loss in (r, s) and Q = diag(2, (1 + z^2)/2):
        lambda Y = -(Q H Y + q),   lambda = z (H Y)_s + 2 (1 - z^2) K,   q = (0, z (1 - z^2) K),   Y = (r_1, s_1) the tilt."""
    a = float(kC(s, s)); b = float(r * kCa(s, s)); d = float(r * r * kCab(s, s) - 4 * K)
    H = np.array([[a, b], [b, d]])
    Q = np.diag([2.0, (1 + z * z) / 2]); q = np.array([0.0, z * (1 - z * z) * K])
    lam = 2 * (1 - z * z) * K / (1 + z * z)
    for _ in range(500):
        Y = -np.linalg.solve(Q @ H + lam * np.eye(2), q)
        ln = z * (H @ Y)[1] + 2 * (1 - z * z) * K
        if abs(ln - lam) < 1e-15:
            break
        lam = ln
    return lam, Y, (a, b, d)


def measure_rate(T, s, r, z0, u0, t0, t1):
    """d ln|u| / dt of the plain gradient flow started at (u0, z0, s, r), fitted on [t0, t1]."""
    out = integrate_gd(from_zeta([u0, z0, s, r]), T, t1, dt0=0.005, rel=0.02, dtmax=0.2, grow=1.02)
    ts = np.array([o[0] for o in out]); us = np.array([abs(o[1][1] - o[1][0]) for o in out])
    m = (ts >= t0) & (ts <= t1)
    return float(np.polyfit(ts[m], np.log(us[m]), 1)[0]), out


# ------------------------------------------------------------------ 9. the singular structure of the perceptron

def check_structure():
    head("9. The two-hidden-unit perceptron: equivalent points, the Fisher matrix on R, its scaling, and the chart (u, z, s, r) (section 12.2.1-12.2.3, (12.103)-(12.140))")
    T = T_POS
    xi = np.array([0.7, 1.4, 1.2, -0.6]); L0 = loss_grad(xi, T)[0]
    dev = max(abs(loss_grad(np.array([-0.7, 1.4, -1.2, -0.6]), T)[0] - L0), abs(loss_grad(np.array([1.4, 0.7, -0.6, 1.2]), T)[0] - L0), abs(loss_grad(np.array([-1.4, -0.7, 0.6, -1.2]), T)[0] - L0))
    print(f"   phi is odd, so (w_i, v_i) -> (-w_i, -v_i) and the swap of the two neurons leave f unchanged (12.109)-(12.110): losses agree to {dev:.1e}. The symmetry group has 2^m m! = 8 elements for m = 2;"
          " the book writes xi ~ -xi, which is only the flip of all neurons at once")
    r0 = 0.9
    Lo = [loss_grad(np.array([1.3, 1.3, r0 * (1 + z) / 2, r0 * (1 - z) / 2]), T)[0] for z in (-3.0, -0.5, 0.0, 0.7, 2.0)]
    Le = [loss_grad(np.array([1.3, w2, r0, 0.0]), T)[0] for w2 in (-2.0, 0.1, 1.3, 3.0)]
    print(f"   on R_o (w_1 = w_2 = 1.3, v_1 + v_2 = {r0}, z = -3, ..., 2) the loss takes one value, spread {max(Lo) - min(Lo):.1e}; on R_e (v_2 = 0, w_1 = 1.3, v_1 = {r0}, w_2 anything) spread {max(Le) - min(Le):.1e}; the two sets meet at w_2 = w_1")
    # (12.138) is the same function as the chart in xi
    ze = np.array([0.37, 0.4, 1.1, 2.0]); xi_z = from_zeta(ze); x = GH_X
    f_xi = xi_z[2] * phi(xi_z[0] * x) + xi_z[3] * phi(xi_z[1] * x)
    u, z, s, r = ze
    f_138 = 0.5 * r * (1 + z) * phi((s + 0.5 * (z - 1) * u) * x) + 0.5 * r * (1 - z) * phi((s + 0.5 * (z + 1) * u) * x)
    print(f"   (12.138): the output written in (u, z, s, r) equals v_1 phi(w_1 x) + v_2 phi(w_2 x) with w_1 = s + (z-1)u/2, w_2 = s + (z+1)u/2, v_1 = r(1+z)/2, v_2 = r(1-z)/2: max difference {np.abs(f_xi - f_138).max():.1e}")
    # expansion (12.139)
    s_, r_, z_ = 1.0, 2.0, 0.4
    print("   Expansion (12.139) of the output around R_o, for r = 2, s = 1, z = 0.4 (L2 distance over x ~ N(0,1) between the exact output and the expansions):")
    print("      u        |f - [r phi(sx) + (1-z^2) u^2 J/8]| (as printed)    |f - [r phi(sx) + r (1-z^2) u^2 J/8]| (with the factor r)    ratio")
    for uu in (0.2, 0.1, 0.05):
        xi_u = from_zeta([uu, z_, s_, r_]); f = xi_u[2] * phi(xi_u[0] * x) + xi_u[3] * phi(xi_u[1] * x)
        Jx = ddphi(s_ * x) * x ** 2
        e_pr = math.sqrt(Ex((f - (r_ * phi(s_ * x) + (1 - z_ ** 2) * uu ** 2 * Jx / 8)) ** 2)); e_ok = math.sqrt(Ex((f - (r_ * phi(s_ * x) + r_ * (1 - z_ ** 2) * uu ** 2 * Jx / 8)) ** 2))
        print(f"      {uu:<6}   {e_pr:.3e}                                                {e_ok:.3e}                                                          {e_pr / e_ok:.1f}")
    print("   so (12.139) as printed lacks the factor r (the second-order coefficient of r-weighted neurons is r(1 - z^2)/8); (12.143) K = (r/4)<(y - f) J> is consistent with the corrected form.")
    # Fisher matrix on R
    G_gen = fisher(np.array([0.7, 1.4, 1.2, -0.6])); xo = np.array([1.3, 1.3, 0.7, 0.2]); xe = np.array([1.3, 2.4, 0.9, 0.0])
    Go = fisher(xo); Ge = fisher(xe)
    ev = lambda G: np.linalg.eigvalsh(G)
    a1 = np.array([0, 0, 1.0, -1.0]); a2 = np.array([xo[3], -xo[2], 0, 0])
    print(f"   Fisher matrix eigenvalues: generic point {clean(ev(G_gen), 5)};  on R_o {clean(ev(Go), 5)};  on R_e {clean(ev(Ge), 5)}")
    print(f"   on R_o the null space is two-dimensional: G a = 0 for a = (0, 0, 1, -1) (along R_o, |G a| = {np.abs(Go @ a1).max():.1e}) and for a = (v_2, -v_1, 0, 0) (moves the two input weights apart at fixed s, |G a| = {np.abs(Go @ a2).max():.1e}),"
          f" while R_o is a one-dimensional set. (12.126)-(12.127) identify the null directions with the directions inside R; on R_o there is one more. On R_e: rank 3, one null direction (the weight w_2 of the dead neuron), as the book says (n = 1).")
    # scaling
    print("   scaling of the eigenvalues of G as u -> 0 at fixed (s, r, z) = (1, 1, 0.4): the four eigenvalues and the log-log slopes of the two smallest between successive rows")
    us = [0.4, 0.2, 0.1, 0.05, 0.025]; prev = None; rows = []
    for uu in us:
        G = fisher(from_zeta([uu, 0.4, 1.0, 1.0])); e = ev(G); dt = np.linalg.det(G)
        sl = "" if prev is None else f"   slopes {math.log(e[0] / prev[1][0]) / math.log(uu / prev[0]):.2f}, {math.log(e[1] / prev[1][1]) / math.log(uu / prev[0]):.2f}, det {math.log(dt / prev[2]) / math.log(uu / prev[0]):.2f}"
        print(f"      u = {uu:<6} eigenvalues {e[0]:.3e} {e[1]:.3e} {e[2]:.4f} {e[3]:.4f}{sl}")
        rows.append((uu, e)); prev = (uu, e, dt)
    STORE["fisher_scaling"] = rows
    print("   the two smallest eigenvalues scale like u^6 and u^2 (the determinant like u^8): the Fisher matrix is nearly singular much faster than the 'u^2, u^4' that the u- and z-directions suggest, because the derivatives of f with respect to u and z are both proportional"
          " to J(x) at leading order (df/du = r(1-z^2)uJ/4, df/dz = -r z u^2 J/4) and only their difference sees the next term (J_3).")
    # a layer-wise block-diagonal approximation of G (the first stage of K-FAC, section 12.1.7.4)
    Gb = np.zeros((4, 4)); Gb[:2, :2] = Go[:2, :2]; Gb[2:, 2:] = Go[2:, 2:]
    print(f"   layer-wise block-diagonal approximation of G (input weights | output weights), the coarse part of the approximations of section 12.1.7.4: on R_o its eigenvalues are {clean(ev(Gb), 4)} and both null vectors survive (|Gb a_1| = {np.abs(Gb @ a1).max():.1e}, |Gb a_2| = {np.abs(Gb @ a2).max():.1e});"
          " near R_o, at (s, r, z) = (1, 1, 0.4):")
    prev = None
    for uu in (0.2, 0.1, 0.05, 0.025):
        Gq = fisher(from_zeta([uu, 0.4, 1.0, 1.0])); Gbq = np.zeros((4, 4)); Gbq[:2, :2] = Gq[:2, :2]; Gbq[2:, 2:] = Gq[2:, 2:]
        e = ev(Gbq)
        print(f"      u = {uu}: two smallest eigenvalues {e[0]:.3e} {e[1]:.3e}" + ("" if prev is None else f", slopes {math.log(e[0] / prev[1][0]) / math.log(uu / prev[0]):.2f} {math.log(e[1] / prev[1][1]) / math.log(uu / prev[0]):.2f}"))
        prev = (uu, e)
    print("   so the null space on R is kept, as the book says ('do not destroy most of the singular structure'), but the approximation is nearly singular like u^2 in both directions, not u^2 and u^6: its inverse blows up much more mildly than that of the true G.")
    # blow-down coordinates
    print("   blow-down coordinates (12.155)-(12.156), mu = (delta, gamma, s, r) with delta = (1 - z^2) u^2, gamma = z (1 - z^2) u^3: Fisher matrix G_mu = M^-T G M^-1, M = d mu / d xi")
    x = GH_X; sq = 1.0; r1 = 1.0
    F = [r1 / 8 * ddphi(x) * x ** 2, r1 / 24 * d3phi(x) * x ** 3, r1 * dphi(x) * x, phi(x)]
    Gram = np.array([[Ex(a * b) for b in F] for a in F])
    for uu in (0.3, 0.1, 0.03):
        xi_u = from_zeta([uu, 0.4, 1.0, 1.0]); u_, z_, s_, r_ = to_zeta(xi_u)
        dmu = np.array([[2 * (1 - z_ ** 2) * u_, -2 * z_ * u_ ** 2, 0, 0], [3 * z_ * (1 - z_ ** 2) * u_ ** 2, (1 - 3 * z_ ** 2) * u_ ** 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        M = dmu @ zeta_T(xi_u)
        Gmu = np.linalg.inv(M).T @ fisher(xi_u) @ np.linalg.inv(M)
        print(f"      u = {uu}: eigenvalues of G_xi {sci(ev(fisher(xi_u)))}  of G_mu {sci(ev(Gmu))}")
    print(f"      limit u -> 0 (Gram matrix of (r/8) J_2, (r/24) J_3, r phi' x, phi at s = 1, r = 1 by quadrature): {sci(np.linalg.eigvalsh(Gram))}: G_mu stays finite and non-degenerate near R, as (12.155)-(12.156) are meant to achieve")
    STORE["gram_limit"] = np.linalg.eigvalsh(Gram)


# ------------------------------------------------------------------ 10. the reduced dynamics against the exact flow

def check_reduced():
    head("10. The reduced dynamics (12.141)-(12.148) and Theorems 12.3-12.5 against the exact averaged flow")
    info = {}
    for name, T in (("T_pos", T_POS), ("T_neg", T_NEG)):
        s, r, L1, K = one_neuron_optimum(T); info[name] = (s, r, L1, K)
        print(f"   teacher {name}: f_0 = " + " ".join(f"{c:+g} phi({o:g} x)" for c, o in zip(T.c, T.om)) + f"; best single neuron s = {s:.4f}, r = {r:.4f}, loss L_1 = {L1:.4f}, K = {K:+.5f} (12.143)")
    STORE["info"] = info
    print("   rate lambda of u on the plain gradient flow, u = u_0 e^(lambda t), started on R_o's neighbourhood at (s, r) = (s*, r*) and measured after the fast variables have settled; 'book' is (12.141): 2 (1 - z^2) K;"
          " 'slow manifold' is my correction (the fast variables follow a manifold tilted linearly in u; see the notes), which uses the Hessian of the one-neuron loss, not only K:")
    print("      teacher   z     K (1-z^2)    measured lambda   slow manifold   book 2(1-z^2)K   measured/book   Theorem 12.4 says")
    rate_rows = []
    for name, T, zs in (("T_pos", T_POS, (0.5, 1.5, 2.5)), ("T_neg", T_NEG, (0.5, 1.5, 2.5))):
        s, r, L1, K = info[name]
        for z in zs:
            lam_sm, Y, abd = slow_manifold_rate(z, s, r, K)
            unstable = (1 - z * z) * K > 0
            lam_m, _ = measure_rate(T, s, r, z, 1e-4 if unstable else 1e-3, 10.0 if unstable else 20.0, 25.0 if unstable else 50.0)
            book = 2 * (1 - z * z) * K
            rate_rows.append((name, z, K, lam_m, lam_sm, book))
            print(f"      {name}   {z}   {(1 - z * z) * K:+.4f}      {lam_m:+.5f}         {lam_sm:+.5f}         {book:+.5f}        {lam_m / book:.3f}           {'unstable (u grows)' if unstable else 'stable (u shrinks)'} -> {'unstable' if lam_m > 0 else 'stable'}")
    STORE["rates"] = rate_rows
    print("   the sign pattern of Theorem 12.4 holds in both cases (K > 0: stable for |z| > 1; K < 0: stable for |z| < 1), but the rates of (12.141) are too large by the factors in the 'measured/book' column (0.84 down to 0.09), nowhere near 1;"
          " the slow-manifold value reproduces the measured rates to within 0.5 percent. My explanation (derived in the notes): dL/ds and dL/dr are not zero on the flow (they are O(u)), and the metric coupling M_us = -z of the plain gradient flow in the chart (u,z,s,r) feeds them back into u at the same order.")
    # where (12.141) comes from, and the foot point
    s, r, L1, K = info["T_pos"]
    out = integrate_gd(from_zeta([0.2, 1.5, s, r]), T_POS, 150.0, dt0=0.005, rel=0.02, dtmax=0.2, grow=1.3)
    D = np.array([o[1][2] - o[1][3] for o in out]); zs_ = np.array([to_zeta(o[1])[1] for o in out])
    print(f"   along an exact trajectory from (u, z) = (0.2, 1.5): v_1 - v_2 changes from {D[0]:.5f} to {D[-1]:.5f} (relative {abs(D[-1] - D[0]) / D[0]:.1e}) while z moves between {zs_.min():.5f} and {zs_.max():.5f};"
          " the foot point on R_o is set by v_1 - v_2 (a slow, nearly conserved quantity), z itself shifts by an amount of order u")
    # first integral
    def reduced(y):
        u, zz = y
        return np.array([2 * (1 - zz * zz) * K * u, -zz * (zz * zz + 3) * u * u * K / r ** 2])
    y = np.array([0.2, 1.5]); inv = lambda y_, p: 0.5 * y_[0] ** 2 - (2 * r * r / 3) * math.log((y_[1] ** 2 + 3) ** p / abs(y_[1]))
    dev2 = 0.0; dev3 = 0.0; i20 = inv(y, 2); i30 = inv(y, 3)
    for k in range(40):
        y = rk4(reduced, y, 2.5, 50)
        dev2 = max(dev2, abs(inv(y, 2) - i20)); dev3 = max(dev3, abs(inv(y, 3) - i30))
    print(f"   (12.147) against (12.148): along a solution of the printed reduced equations (12.141)-(12.142) from (u, z) = (0.2, 1.5), h(u) - (2r^2/3) log((z^2+3)^p/|z|) varies by {dev2:.1e} for p = 2 and by {dev3:.3f} for p = 3:"
          " the integral of (12.141)-(12.142) has the exponent 2, as (12.148) says; the exponent 3 in (12.147) is a misprint (dh/dz = -2r^2 (1 - z^2)/(z (z^2 + 3)) integrates to (2r^2/3) log((z^2+3)^2/|z|))")
    out = integrate_gd(from_zeta([0.2, 1.5, s, r]), T_POS, 60.0, dt0=0.005, rel=0.02, dtmax=0.2, grow=1.05)
    vals = np.array([0.5 * to_zeta(o[1])[0] ** 2 - (2 * r * r / 3) * math.log((to_zeta(o[1])[1] ** 2 + 3) ** 2 / abs(to_zeta(o[1])[1])) for o in out])
    print(f"   on the exact flow from the same point the quantity (p = 2) varies by {vals.max() - vals.min():.3f} over the trajectory, against h(u_0) = {0.5 * 0.2 ** 2:.3f}: it is not conserved to the order of h")
    # (12.144)
    xe = np.array([s, s + 0.5, r, 0.0]); ge = loss_grad(xe, T_POS)[1]
    xo = from_zeta([0.0, 0.7, s, r]); go = loss_grad(xo, T_POS)[1]
    zdot_exact = float((zeta_T(xe) @ (-ge))[1]); zdot_book = -4 * K * 0.25 / r ** 2
    print(f"   (12.144) says every point of R = R_o + R_e is an equilibrium. At a point of R_o with (s, r) = (s*, r*): |grad L| = {np.abs(go).max():.1e} (an equilibrium, for the teacher outside R too). At a point of R_e with the live neuron optimal"
          f" (v_2 = 0, w_1 = s*, v_1 = r*, w_2 = s* + 0.5): grad L = {clean(ge, 5)}: dL/dv_2 = {ge[3]:+.4f} is not 0, the dead neuron's output weight is pushed, so z moves: dz/dt = {zdot_exact:+.2e} on the exact flow"
          f" (the leading-order (12.142) at z = 1, u = 0.5 gives -4 K u^2/r^2 = {zdot_book:+.2e}). R_e consists of equilibria only when the teacher is in R (all gradients vanish there, L = 0)")
    # (12.149)
    xi1 = from_zeta([0.3, 0.4, 1.0, 1.0]); x = GH_X
    f = xi1[2] * phi(xi1[0] * x) + xi1[3] * phi(xi1[1] * x); f0 = T_IN.f(x)
    Jx = ddphi(1.0 * x) * x ** 2
    K143 = 1.0 / 4 * Ex((f0 - f) * Jx); K149 = 1.0 / 4 * Ex((f - f0) * Jx)
    u_rate, _ = measure_rate(T_IN, 1.0, 1.0, 0.4, 0.3, 3.0, 6.0)
    print(f"   (12.149): teacher in R, point (u, z, s, r) = (0.3, 0.4, 1, 1): (12.143) with y - f -> f_0 - f gives K = {K143:+.3e}; (12.149) with e = f - f_0 as in (12.150) would give {K149:+.3e}. With (1 - z^2) K < 0 u decays (stable, Theorem 12.3),"
          f" with the printed sign it would grow; the exact flow decays (d ln u/dt = {u_rate:+.2e} per unit time at this point, of the size -c u^2 with c = 5.8e-4, see section 11). The sign in (12.149) is a misprint (K = -(r/4)<e J> with e = f - f_0)")
    # the basin of the stable segment: separatrix near z = 1
    def classify(u0, z0, tmax=4000.0):
        xi = from_zeta([u0, z0, s, r]); t = 0.0; dt = 0.01
        while t < tmax:
            xi = ros2_step(flow_gd, xi, T_POS, dt, jac_gd); t += dt; dt = min(dt * 1.03, 10.0)
            u = abs(xi[1] - xi[0])
            if u < 0.01 * u0:
                return True
            if u > 1.0:
                return False
        return u < u0
    print("   Milnor attractor: the stable part of R_o (here |z| > 1) has a basin of finite measure but is not Lyapunov stable. Bisection on the exact flow for the smallest z_0 > 1 whose trajectory is trapped (u -> 0), for starting distances u_0;"
          " the printed (12.148) predicts a wedge z_c - 1 = u_0/r near z = 1:")
    wedge = []
    for u0 in (0.4, 0.2):
        lo, hi = 1.0001, 1.12
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if classify(u0, mid):
                hi = mid
            else:
                lo = mid
        zc = 0.5 * (lo + hi); wedge.append((u0, zc))
        print(f"      u_0 = {u0}: z_c = {zc:.4f}, (z_c - 1)/u_0 = {(zc - 1) / u0:.4f}  (prediction 1/r = {1 / r:.4f})")
    STORE["wedge"] = wedge
    print("   the wedge is reproduced to within 4 percent: points arbitrarily close to R_o with |z| - 1 < u_0/r leave. (At z = 1 both rates in (12.141)-(12.142) are off by the same factor 1/2, so their ratio, which sets the separatrix there, survives.)")


# ------------------------------------------------------------------ 11. plateau: vanilla against natural gradient

def check_plateau():
    head("11. Critical slowdown and natural gradient near R (section 12.2.4-12.2.5, (12.149)-(12.158))")
    x = GH_X
    # --- teacher in R, plain gradient
    xi0 = from_zeta([0.3, 0.4, 1.0, 1.0])
    out = integrate_gd(xi0, T_IN, 1e7, dt0=0.01, rel=0.05, dtmax=1e6, grow=1.3)
    ts = np.array([o[0] for o in out]); ze = np.array([to_zeta(o[1]) for o in out]); Ls = np.array([loss_grad(o[1], T_IN)[0] for o in out])
    us = np.abs(ze[:, 0])
    m = (ts >= 1e6) & (ts <= 1e7)
    sl_u = np.polyfit(np.log(ts[m]), np.log(us[m]), 1)[0]; sl_L = np.polyfit(np.log(ts[m]), np.log(Ls[m]), 1)[0]
    J2 = ddphi(x) * x ** 2
    B = [phi(x), dphi(x) * x]; Gm = np.array([[Ex(a * b) for b in B] for a in B]); coef = np.linalg.solve(Gm, np.array([Ex(a * J2) for a in B])); P2 = J2 - coef[0] * B[0] - coef[1] * B[1]
    nJ = Ex(J2 * J2); nP = Ex(P2 * P2)
    cs = []
    for lo, hi in ((3e4, 1e5), (1e6, 1e7)):
        mw = (ts >= lo) & (ts <= hi)
        pf = np.polyfit(ts[mw], us[mw] ** -2, 1); dev = np.max(np.abs(us[mw] ** -2 - np.polyval(pf, ts[mw])) / us[mw] ** -2)
        zl = float(ze[mw, 1].mean())
        cs.append((lo, hi, pf[0] / 2, dev, zl, (1 / 16) * (1 - zl ** 2) ** 2 * nJ, (1 / (1 + zl ** 2)) * (1 / 16) * (1 - zl ** 2) ** 2 * nP))
    print(f"   teacher in R (one neuron), plain gradient flow from (u, z, s, r) = (0.3, 0.4, 1, 1), t up to 10^7: u falls from 0.3 to {us[-1]:.4f} and L from {Ls[0]:.1e} to {Ls[-1]:.1e};"
          f" local slopes on [10^6, 10^7]: d ln u / d ln t = {sl_u:.3f} (u ~ t^(-1/2): du/dt = -c u^3, the exponent of (12.152)), d ln L / d ln t = {sl_L:.3f} (L ~ t^-2); z drifts to {ze[-1, 1]:.3f}")
    for lo, hi, c_meas, dev, zl, c_book, c_pred in cs:
        print(f"      window [{lo:.0e}, {hi:.0e}]: u^-2 is a straight line in t to {dev:.1e} (relative), slope 2 c with c = {c_meas:.2e} (mean z = {zl:.3f}); the printed reduction (12.141)+(12.143) gives c = (r^2/16)(1 - z^2)^2 <J^2> = {c_book:.2e},"
              f" {c_book / c_meas:.0f} times too large; with the projected curvature and the metric factor, c = (1/(1+z^2)) (r^2/16)(1-z^2)^2 <(P J)^2> = {c_pred:.2e}")
    zf, uf = ze[-1, 1], us[-1]
    print(f"      here <J^2> = {nJ:.4f} and <(P J)^2> = {nP:.4f}, P projecting out phi(sx) and phi'(sx) x: the optimal (s, r) shift by O(u^2) and absorb most of J = phi''(sx) x^2; the loss on the flow is (r^2/128)(1-z^2)^2 u^4 <(P J)^2>, not (r^2/128)(1-z^2)^2 u^4 <J^2>:"
          f" at t = 10^7 the measured L is {Ls[-1]:.3e} against {(1 / 128) * (1 - zf ** 2) ** 2 * uf ** 4 * nP:.3e} (ratio {Ls[-1] / ((1 / 128) * (1 - zf ** 2) ** 2 * uf ** 4 * nP):.3f}) with <(P J)^2> and {(1 / 128) * (1 - zf ** 2) ** 2 * uf ** 4 * nJ:.3e} with <J^2>")
    ou = integrate_gd(from_zeta([0.3, 2.0, 1.0, 1.0]), T_IN, 1e4, dt0=0.01, rel=0.05, dtmax=500.0, grow=1.5)
    print(f"      the same with z_0 = 2 (|z| > 1): u falls from 0.3 to {abs(ou[-1][1][1] - ou[-1][1][0]):.4f} at t = 10^4: Theorem 12.3 (stable for every z), the exponent 3 in (12.152) does not depend on which part of R_o it is")
    STORE["slow_gd"] = dict(t=ts, u=us, L=Ls)
    # --- teacher in R, natural gradient
    outn = integrate_ng(xi0, T_IN, 9.0, 0.05, rec=1.0)
    print("   teacher in R, natural gradient flow from the same point (12.153)-(12.158):")
    print("      t      u         z         delta=(1-z^2)u^2   ln(delta) + t   u z        L")
    rows = []
    for t, xi in outn:
        u, z, s, r = to_zeta(xi); dl = (1 - z * z) * u * u; L = loss_grad(xi, T_IN)[0]
        rows.append((t, u, z, dl, L))
        if t in (0.0, 1.0, 2.0, 3.0, 5.0, 9.0) or abs(t - 9.0) < 1e-9:
            print(f"      {t:<5.1f}  {u:.5f}   {z:.5f}   {dl:.4e}         {math.log(dl) + t:+.4f}        {u * z:.5f}    {L:.3e}")
    STORE["slow_ng"] = rows
    print(f"   delta falls like e^-t (ln delta + t changes by {(math.log(rows[-1][3]) + rows[-1][0]) - math.log(rows[0][3]):+.4f} over the run), u z stays near its initial value 0.12 ({rows[-1][1] * rows[-1][2]:.4f} at the end), z -> 1: the flow ends on R_e with u = {rows[-1][1]:.3f} != 0, not on R_o,"
          f" and L falls by the factor {rows[0][4] / rows[-1][4]:.2e} (e^18 = {math.exp(18):.2e}) as e^(-2t)")
    print("   instantaneous check of (12.153)-(12.154) on the exact natural flow at z = 0.4, (s, r) = (1, 1): du/dt / u against -(1 - z^2)/2 = -0.4200 and dz/dt against (1 - z^2) z / 2 = 0.1680:")
    for uu in (0.3, 0.1, 0.03):
        xi = from_zeta([uu, 0.4, 1.0, 1.0]); v = zeta_T(xi) @ flow_ng(xi, T_IN)
        print(f"      u = {uu}: du/dt / u = {v[0] / uu:+.4f}, dz/dt = {v[1]:+.4f}")
    print("   (12.157): <grad l> = G mu in the blow-down chart, teacher in R, at (s, r) = (1, 1): relative error |grad_mu L - G_mu (delta, gamma, 0, 0)| / |grad_mu L|:")
    for uu in (0.3, 0.1, 0.03):
        xi = from_zeta([uu, 0.4, 1.0, 1.0]); u_, z_, s_, r_ = to_zeta(xi)
        dmu = np.array([[2 * (1 - z_ ** 2) * u_, -2 * z_ * u_ ** 2, 0, 0], [3 * z_ * (1 - z_ ** 2) * u_ ** 2, (1 - 3 * z_ ** 2) * u_ ** 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        M = dmu @ zeta_T(xi); Mi = np.linalg.inv(M)
        gmu = Mi.T @ loss_grad(xi, T_IN)[1]; Gmu = Mi.T @ fisher(xi) @ Mi
        mu = np.array([(1 - z_ ** 2) * u_ ** 2, z_ * (1 - z_ ** 2) * u_ ** 3, 0, 0])
        print(f"      u = {uu}: {np.linalg.norm(gmu - Gmu @ mu) / np.linalg.norm(gmu):.3f}")
    # --- teacher outside R
    s, r, L1, K = STORE["info"]["T_pos"]
    e0 = r * phi(s * x) - T_POS.f(x)
    F = [phi(s * x), dphi(s * x) * x, ddphi(s * x) * x ** 2, d3phi(s * x) * x ** 3]
    Gr = np.array([[Ex(a * b) for b in F] for a in F]); bvec = np.array([Ex(e0 * a) for a in F])
    lim = float(bvec @ np.linalg.solve(Gr, bvec))
    print(f"   teacher outside R (T_pos, K = {K:.4f}), on the stable segment z = 1.5: rate of decrease of the loss, plain gradient |grad L|^2 against natural gradient grad L^T G^-1 grad L:")
    for uu in (0.3, 0.1, 0.03):
        xi = from_zeta([uu, 1.5, s, r]); g = loss_grad(xi, T_POS)[1]; v = flow_ng(xi, T_POS)
        print(f"      u = {uu}: plain {g @ g:.3e}, natural {-(g @ v):.4f}")
    print(f"      the natural rate stays near {lim:.4f} = |P e_0|^2, the squared length of the residual's projection on span{{phi, phi' x, phi'' x^2, phi''' x^3}} (the limiting tangent space, by quadrature), while the plain rate vanishes like u^2: in function space the natural flow keeps moving at an ordinary speed")
    def ng_velocity_blowdown(xi, T):
        """-G^-1 grad L computed in the blow-down chart (better conditioned): xi' = M^-1 (-G_mu^-1 grad_mu L)."""
        u_, z_, s_, r_ = to_zeta(xi)
        dmu = np.array([[2 * (1 - z_ ** 2) * u_, -2 * z_ * u_ ** 2, 0, 0], [3 * z_ * (1 - z_ ** 2) * u_ ** 2, (1 - 3 * z_ ** 2) * u_ ** 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Mi = np.linalg.inv(dmu @ zeta_T(xi))
        return Mi @ (-np.linalg.solve(Mi.T @ fisher(xi) @ Mi, Mi.T @ loss_grad(xi, T)[1]))
    print("   but in parameter space the natural gradient vector G^-1 grad L (largest entry) diverges as u -> 0 when the teacher is outside R, and stays ordinary when it is inside"
          " (computed in the blow-down chart, which agrees with the direct solve to 7 digits for u >= 0.06 and stays accurate where the direct solve loses the smallest eigenvalue):")
    for name, T, (ss, rr) in (("teacher outside R", T_POS, (s, r)), ("teacher inside R ", T_IN, (1.0, 1.0))):
        prev = None; cells = []
        for uu in (0.5, 0.25, 0.125, 0.0625, 0.03):
            n = np.abs(ng_velocity_blowdown(from_zeta([uu, 1.5, ss, rr]), T)).max()
            cells.append(f"u = {uu}: {n:.3g}" + ("" if prev is None else f" (slope {math.log(n / prev[1]) / math.log(uu / prev[0]):+.2f})"))
            prev = (uu, n)
        print(f"      {name}: " + "; ".join(cells))
    print("   the slopes -2.2, -2.5, -2.7, -2.8 are heading for -3, the value of the pseudo-inverse picture (G^-1 grad L = J^+ e_0 and the smallest singular value of the Jacobian J of the output scales like u^3, eigenvalue u^6; I cannot go closer in double precision)."
          " 'The natural gradient takes an ordinary value in a very small neighborhood of R' (p. 309) is a statement about the output (and holds for the weights only when the teacher is in R): the weights move at speed ~ u^-3 when the teacher is outside.")
    # trajectories
    out_a = integrate_gd(from_zeta([0.1, 1.5, s, r]), T_POS, 400.0, dt0=0.01, rel=0.05, dtmax=2.0, grow=1.3)
    out_b = integrate_gd(from_zeta([0.1, 0.6, s, r]), T_POS, 400.0, dt0=0.01, rel=0.05, dtmax=2.0, grow=1.3, stop=lambda t, xi: loss_grad(xi, T_POS)[0] < 0.5 * L1)
    ta = out_a[-1]; tb = out_b[-1]
    print(f"   plain gradient from u_0 = 0.1: at z_0 = 1.5 (stable segment) the loss is still {loss_grad(ta[1], T_POS)[0]:.5f} = L_1 = {L1:.5f} at t = 400 (u = {abs(ta[1][1] - ta[1][0]):.1e}): trapped for good;"
          f" at z_0 = 0.6 (unstable segment) it takes t = {tb[0]:.1f} to get below L_1/2")
    xs = from_zeta([0.1, 0.6, s, r]); Ls0, gs0 = loss_grad(xs, T_POS); vs = flow_ng(xs, T_POS); rate0 = -(gs0 @ vs); zd = zeta_T(xs) @ vs
    lam0 = np.linalg.eigvalsh(fisher(xs))
    print(f"   natural gradient from (u_0, z_0) = (0.1, 0.6), the unstable part: Fisher eigenvalues {lam0[0]:.1e}, {lam0[1]:.1e}, {lam0[2]:.3f}, {lam0[3]:.3f}; in function space the loss falls at the rate"
          f" grad L^T G^+ grad L = {rate0:.4f} = {rate0 / Ls0:.3f} L (the u_0 -> 0 limit {lim:.4f} quoted above is {lim / L1:.3f} L_1), so L should halve in about ln 2 / {rate0 / Ls0:.3f} = {math.log(2) / (rate0 / Ls0):.2f} time units;"
          f" but the weights move at speed max|xi'| = {np.abs(vs).max():.3g} (du/dt = {zd[0]:.0f}, dz/dt = {zd[1]:.0f}): a single step of 0.001 would change v_1 by {0.001 * abs(vs[2]):.0f}, so only the loss curve, not the parameter path, can be trusted")
    res = {}
    for dt in (0.002, 0.001, 0.0005):
        o = integrate_ng(xs, T_POS, 1.0, dt, rec=0.25)
        res[dt] = [loss_grad(xi, T_POS)[0] for t, xi in o]
    fine = res[0.0005]
    tg = [0.0, 0.25, 0.5, 0.75, 1.0]
    def t_cross(Ls):
        j = next(i for i in range(len(Ls)) if Ls[i] < L1 / 2)
        return tg[j - 1] + (tg[j] - tg[j - 1]) * (Ls[j - 1] - L1 / 2) / (Ls[j - 1] - Ls[j])
    t_half = t_cross(fine)
    loc = [math.log(fine[i] / fine[i + 1]) / 0.25 for i in range(1, 4)]
    print(f"   natural gradient flow from the same start: L at t = 0, 0.25, 0.5, 0.75, 1 is " + ", ".join(f"{v:.4f}" for v in fine) + " (step 0.0005);"
          f" the steps 0.002, 0.001, 0.0005 give {res[0.002][1]:.4f}, {res[0.001][1]:.4f}, {res[0.0005][1]:.4f} at t = 0.25 and {res[0.002][4]:.4f}, {res[0.001][4]:.4f}, {res[0.0005][4]:.4f} at t = 1 (converging);"
          f" it crosses L_1/2 = {L1 / 2:.4f} at t = {t_half:.2f}, against t = {tb[0]:.0f} for the plain gradient, and then L decays like e^(-ct) with local rates c = {loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f} (e^-2t in the over-realisable case)")
    o3 = integrate_ng(from_zeta([0.3, 0.6, s, r]), T_POS, 1.0, 0.0005, rec=0.25)
    L3 = [loss_grad(xi, T_POS)[0] for t, xi in o3]
    print(f"   the same from u_0 = 0.3 (step 0.0005): L at t = 0, 0.25, ..., 1 is " + ", ".join(f"{v:.4f}" for v in L3) + f", crossing L_1/2 at t = {t_cross(L3):.3f} (from u_0 = 0.1: {t_half:.3f}): the natural exit time hardly depends on u_0;"
          " for u_0 <= 0.05 I could not get converged integrations (in double precision the loss either blew up or changed with the step), so below 0.1 the u_0-independence rests on the function-space rate above")
    STORE["plat_pos"] = dict(L1=L1, gd_trap=[(o[0], loss_grad(o[1], T_POS)[0]) for o in out_a], gd_escape=None, ng_t=[0.0, 0.25, 0.5, 0.75, 1.0], ng_L=fine)
    out_c = integrate_gd(from_zeta([0.1, 0.6, s, r]), T_POS, 400.0, dt0=0.01, rel=0.05, dtmax=2.0, grow=1.3)
    STORE["plat_pos"]["gd_escape"] = [(o[0], loss_grad(o[1], T_POS)[0]) for o in out_c]
    print("   from z_0 = 1.5 (inside the plain gradient's basin) the natural flow also falls below L_1/2 within t < 1 in my integrations, but the parameters make huge excursions on the way and the result depended on the step size,"
          " so I do not quote that trajectory (the route probably runs round the cusp of the image of the two-neuron family in function space; unverified)")
    # saddle-free Newton (absolute Hessian metric) from the same starts
    def sfn_exit(u0, z0, dt=0.05, t_end=30.0):
        xi = from_zeta([u0, z0, s, r]); t = 0.0; Lp = loss_grad(xi, T_POS)[0]
        while t < t_end:
            xi2 = ros2_step(flow_sfn, xi, T_POS, dt); L = loss_grad(xi2, T_POS)[0]
            if L < 0.5 * L1:
                return t + dt * (Lp - 0.5 * L1) / (Lp - L), L
            xi, t, Lp = xi2, t + dt, L
        return None, Lp
    ex = [sfn_exit(u0, 0.6)[0] for u0 in (0.1, 0.01, 0.001)]
    tr_ = sfn_exit(0.1, 1.5)
    print(f"   saddle-free Newton (the book: 'the same characteristics' as the Fisher natural gradient near R, pp. 286, 302, 310), -|H|^+ grad L, from z_0 = 0.6 and u_0 = 0.1, 0.01, 0.001: loss falls below L_1/2 at t = "
          f"{ex[0]:.2f}, {ex[1]:.2f}, {ex[2]:.2f} (step 0.05; it grows by {ex[1] - ex[0]:.2f} and {ex[2] - ex[1]:.2f} per decade of u_0, ln 10 = {math.log(10):.2f}: a Newton-type exponential escape at rate 1, t ~ ln(1/u_0), unlike the natural gradient's"
          f" u_0-independent {t_half:.2f}); from the stable part z_0 = 1.5, u_0 = 0.1 it stays at L = {tr_[1]:.5f} (L_1 = {L1:.5f}) up to t = 30. Along the null direction inside R, a^T |H| a = 0 as (12.128) says, but |H| is not degenerate"
          " in the direction u when K != 0 (its eigenvalue there is |(1 - z^2) K|-sized), so it does not share the Fisher metric's blow-up of G^-1")
    # online SGD on the plateau
    rng = np.random.default_rng(2)
    R_, eta_ = 100, 0.01
    xs = np.tile(from_zeta([0.0, 1.1, s, r]), (R_, 1))
    print(f"   online SGD (one sample per step, eta = {eta_}, noise-free teacher T_pos, {R_} runs) started on the stable segment at z_0 = 1.1, u_0 = 0: the loss, |u| and z of the median run:")
    for step in range(1, 10001):
        x = rng.normal(size=R_); y = T_POS.f(x)
        w1, w2, v1, v2 = xs.T
        p1 = phi(w1 * x); p2 = phi(w2 * x); e = v1 * p1 + v2 * p2 - y
        xs = xs - eta_ * np.stack([e * v1 * dphi(w1 * x) * x, e * v2 * dphi(w2 * x) * x, e * p1, e * p2], 1)
        if step in (500, 4500, 10000):
            zz = np.array([to_zeta(q) for q in xs]); LL = np.array([loss_grad(q, T_POS)[0] for q in xs])
            print(f"      t = {step * eta_:6.1f}: median L = {np.median(LL):.4f} (L_1 = {L1:.4f}), median |u| = {np.median(np.abs(zz[:, 0])):.3f}, median z = {np.median(zz[:, 1]):.3f}, runs with L < L_1/2: {np.mean(LL < L1 / 2):.2f}")
    print("   after 100 time units no run has left the plateau: the noise keeps u at a thermal level (|u| ~ 0.04) instead of 0 and z creeps toward 1 very slowly; the book's picture (a long random walk to the edge |z| = 1 followed by escape) is consistent with this but I did not run long enough to see an escape")


# ------------------------------------------------------------------ 12. a singular statistical model

def check_singular_statistics():
    head("12. A singular statistical model: the likelihood-ratio statistic of a neuron with an unidentified weight (section 12.2.6, (12.159)-(12.162))")
    print("   y = v phi(w x) + noise, x ~ N(0,1), unit noise, true v = 0 (the truth is on the singular set: w cannot be identified). For N observations the profile statistic for H0: v = 0 is 2 log LR = max_w T_N(w)^2,")
    print("   T_N(w) = sum y_i phi(w x_i) / sqrt(sum phi(w x_i)^2), which for large N is the supremum of a Gaussian process with correlation C(w,w')/sqrt(C(w,w) C(w',w')), C the arcsin kernel; 10^5 draws on a grid of weights w in [0.05, W]:")
    rng = np.random.default_rng(12)
    c1 = 1.959963984540054 ** 2; c2 = -2 * math.log(0.05)
    print(f"      references: chi^2_1 (w known): mean 1, 95 percent point {c1:.3f}; chi^2_2 (two free parameters): mean 2, 95 percent point {c2:.3f}")
    res = []
    for W, m in ((1.0, 60), (10.0, 80), (10.0, 160), (1000.0, 120)):
        w = np.exp(np.linspace(math.log(0.05), math.log(W), m))
        Cm = kC(w[:, None], w[None, :]); d = np.sqrt(np.diag(Cm)); rho = Cm / np.outer(d, d)
        Lc = np.linalg.cholesky(rho + 1e-9 * np.eye(m))
        sup = ((rng.normal(size=(100_000, m)) @ Lc.T) ** 2).max(axis=1)
        res.append((W, m, float(sup.mean()), float(np.quantile(sup, 0.95)), float(np.quantile(sup, 0.99))))
        print(f"      w up to {W:g} ({m} grid points): mean {res[-1][2]:.3f}, 95 percent point {res[-1][3]:.3f}, 99 percent point {res[-1][4]:.3f}")
    print("   so the statistic is neither chi^2_1 nor chi^2_2, and it stays bounded as the range of w grows (the mean saturates near 1.5): for this bounded model the limit is a finite supremum and nothing diverges with N;"
          " the chi^2 recipes behind AIC and MDL do not apply here, as the book says, but a log N growth of the statistic is not what happens in this model")


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

    def frame(self, xt, yt, xlab, ylab, title, grid=True, xtl=None, ytl=None):
        b = self.b
        for y in yt:
            if grid: b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{ytl[y] if ytl else fmt(y)}</text>')
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



def log10(x):
    return math.log10(x)


def fig_steepest(path):
    st = STORE["steep"]; bern = STORE["bern"]
    b = []; W, H = 800, 405
    P1 = Panel(b, 90, 56, 250, 250, (-3, 3), (-3, 3))
    P1.frame([-2, 0, 2], [-2, 0, 2], "a₁", "a₂", "One step of unit length in the metric G", True)
    th = np.linspace(0, 2 * np.pi, 200)
    P1.line(np.cos(th), np.sin(th), "dash s0")                                       # the Euclidean unit circle
    e = st["a_ellipse"]
    P1.line(e[0], e[1], "ln sk")                                                     # a^T G a = 1
    g = st["g"]; an = st["a_ng"]; ag = st["a_gd"]
    # level line of the linear decrease through the natural-gradient point: g . a = best
    d = np.array([-g[1], g[0]]); d = d / np.linalg.norm(d)
    pts = [an + t * d for t in (-0.75, 0.75)]
    P1.line([p[0] for p in pts], [p[1] for p in pts], "dash s1")
    O = (P1.X(0), P1.Y(0))
    arrow(b, O[0], O[1], P1.X(g[0]), P1.Y(g[1]), "0", 2.0, 7.0, True)
    arrow(b, O[0], O[1], P1.X(ag[0]), P1.Y(ag[1]), "2", 2.4, 7.0)
    arrow(b, O[0], O[1], P1.X(an[0]), P1.Y(an[1]), "1", 2.4, 7.0)
    P1.text(g[0] + 0.1, g[1] - 0.45, "∇L", "sm", "start")
    P1.text(ag[0] + 0.12, ag[1] - 0.55, "plain", "sm", "start")
    P1.text(an[0] + 0.12, an[1] - 0.15, "G⁻¹∇L", "sm", "start")
    legend_col(b, 360, 80, [("sk", "aᵀGa = 1"), ("dash s0", "|a| = 1"), ("dash s1", "∇L·a = max")])
    P2 = Panel(b, 470, 56, 300, 250, (0, 6), (0, 1))
    P2.frame([0, 2, 4, 6], [0, 0.25, 0.5, 0.75, 1], "t", "p(t)", "Bernoulli model, target p = 0.9", True)
    P2.line([0, 6], [0.9, 0.9], "dash s0")
    P2.line(bern["t"], bern["ng"], "ln s1"); P2.line(bern["t"], bern["gd_th"], "ln s2"); P2.line(bern["t"], bern["gd_p"], "ln s3")
    legend_col(b, 598, 122, [("s1", "natural, either chart"), ("s2", "plain, logit chart θ"), ("s3", "plain, chart p")])
    note(b, 70, 372, ["Left: among all steps with aᵀGa = 1 the largest decrease of L is at G⁻¹∇L (blue); the plain gradient (orange) is steepest only for the Euclidean circle.",
                      "Right: the natural flow is one curve in every chart; the plain flows depend on the chart (p(1) = 0.57 against 0.02 and 0.90)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Steepest descent in a metric, and chart dependence",
        "Left: in the plane of parameter steps a, the ellipse a transpose G a equals one and the Euclidean unit circle, the gradient of the loss, the plain gradient direction scaled to the ellipse and the natural gradient direction G inverse times the gradient, which is where the line of constant decrease touches the ellipse. Right: probability of the Bernoulli model against time under three flows started at 0.01 with target 0.9: the natural gradient flow is the same curve in the logit chart and in the probability chart, 0.9 minus 0.89 exp(-t), while the plain gradient flows differ strongly between charts.", b))


def fig_online(path):
    ex = STORE["online_exact"]; t = ex["t"]; R = ex["R"]; var = ex["variants"]
    b = []; W, H = 800, 405
    P1 = Panel(b, 70, 56, 320, 250, (1, 4.5), (-0.05, 3.0))
    P1.frame([1, 2, 3, 4], [0, 1, 2, 3], "t", "t λ Var / (Cramér–Rao value)", "Online error variance, ratio to the bound", True, {1: "10", 2: "10²", 3: "10³", 4: "10⁴"}, {0: "1", 1: "10", 2: "100", 3: "1000"})
    idx = {(k, c): i for i, (k, c, t0) in enumerate(var)}
    lt = np.log10(t)
    def col(k, c, a):
        return np.log10(R[:, idx[(k, c)], a])
    P1.line(lt, col("NG", 1.0, 1), "ln s1"); P1.line(lt, col("NG", 2.0, 1), "ln s3")
    P1.line(lt, col("GD", 25.0, 0), "ln s2"); P1.line(lt, col("GD", 25.0, 1), "dash s2"); P1.line(lt, col("GD", 1.0, 1), "ln s4")
    P1.line([1, 4.5], [0, 0], "dash s0")
    legend_col(b, 86, 70, [("s1", "natural, c = 1 (→ 1)"), ("s3", "natural, c = 2 (→ 4/3)"), ("s2", "plain, c = 25, steep (→ 12.8)"), ("dash s2", "plain, c = 25, flat (→ 1)"), ("s4", "plain, c = 1, flat (grows)")], 15)
    P2 = Panel(b, 470, 56, 300, 250, (log10(0.3), log10(40)), (0, 14))
    xs = np.logspace(log10(0.52), log10(40), 200)
    P2.frame([log10(0.5), 0, 1], [0, 4, 8, 12], "x = c λ", "factor x²/(2x − 1)", "Asymptotic factor of a c/t schedule", True, {log10(0.5): "0.5", 0: "1", 1: "10"})
    P2.line(np.log10(xs), xs ** 2 / (2 * xs - 1), "ln s0")
    for x, y, cls in ((0.75, 1.125, "f1"), (1.0, 1.0, "f1"), (2.0, 4 / 3, "f1"), (25.0, 25 ** 2 / 49, "f2"), (1.0, 1.0, "f2")):
        P2.dot(log10(x), y, cls, 4.0)
    P2.text(log10(0.75), 1.9, "natural c = 0.75, 1, 2", "sm", "start", -16, -2)
    P2.text(log10(25), 12.76, "plain c = 25", "sm", "end", -8, 4)
    P2.text(log10(0.52), 12.6, "x ≤ 1/2: no 1/t rate", "sm", "start", 2, 0)
    note(b, 70, 372, ["Gaussian linear model, G = diag(1, 0.04): exact error covariance of online learning with gain c/(t + t₀). The natural gradient has x = c in both directions;",
                      "the plain gradient has x = cλ with λ = 1 and 0.04 and cannot make both factors 1: its flat direction is slow unless c ≥ 12.5, its steep one then costs 12.8."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Online learning: natural against plain gradient",
        "Left: on logarithmic axes, the ratio of the online error variance times t to its Cramér–Rao value against t for a Gaussian linear model with two directions whose Fisher eigenvalues are 1 and 0.04: the natural gradient with c = 1 tends to 1, with c = 2 to 4/3; the plain gradient with c = 25 tends to 12.8 in the steep direction and 1 in the flat one, and with c = 1 the flat direction diverges. Right: the asymptotic factor x squared over 2x minus 1, minimal and equal to 1 at x = 1, with the cases marked.", b))


def fig_saturation(path):
    rows = STORE["sat"]; tr = STORE["sat_traj"]; on = STORE["sat_online"]
    b = []; W, H = 800, 405
    ws = np.exp(np.linspace(math.log(0.4), math.log(9.0), 36)); w0 = 1.0
    G = np.array([float(kCab(w, w)) for w in ws])
    Gb = np.array([Ex(((phi(w0 * GH_X) - phi(w * GH_X)) ** 2 + 1.0) * dphi(w * GH_X) ** 2 * GH_X ** 2) for w in ws])
    P1 = Panel(b, 70, 56, 320, 250, (log10(0.4), log10(9)), (-3.5, 4))
    P1.frame([log10(0.5), 0, log10(2), log10(4), log10(8)], [-3, -1.5, 0, 1.5, 3], "w", "value", "One erf neuron, y = φ(w₀x) + noise, w₀ = 1", True,
             {log10(0.5): "0.5", 0: "1", log10(2): "2", log10(4): "4", log10(8): "8"}, {-3: "10⁻³", -1.5: "10⁻¹·⁵", 0: "1", 1.5: "10¹·⁵", 3: "10³"})
    P1.line(np.log10(ws), np.log10(Gb), "ln s2"); P1.line(np.log10(ws), np.log10(Gb / G), "ln s1"); P1.line(np.log10(ws), np.log10(Gb / G ** 2), "ln s3")
    for (w, Gw, Gbar, nat, step, Lp) in rows:
        P1.dot(log10(w), log10(Gbar), "f2", 3.6); P1.dot(log10(w), log10(nat), "f1", 3.6); P1.dot(log10(w), log10(step), "f3", 3.6)
    legend_col(b, 86, 70, [("s2", "plain: E|∇l|² = Ḡ"), ("s1", "natural: tr(Ḡ G⁻¹)"), ("s3", "natural step in w: Ḡ/G²")], 15)
    P2 = Panel(b, 470, 56, 300, 250, (0, 2.7), (-1.6, 0.7))
    P2.frame([0, 1, 2], [-1, 0], "iteration t", "|w − w₀|", "From w = 4: averaged and online", True, {0: "1", 1: "10", 2: "100"}, {-1: "0.1", 0: "1"})
    for key, cls in ((("gd", 1.0), "ln s2"), (("ng", 0.5), "ln s1")):
        a = tr[key]; ks = np.arange(len(a)); err = np.maximum(np.abs(a - 1.0), 1e-3)
        P2.line(np.log10(1 + ks), np.log10(err), cls)
    for (kind, eta), cls in ((("gd", 1.0), "f2"), (("ng", 0.05), "f1")):
        for t, med in on[(kind, eta)].items():
            P2.dot(log10(t), log10(max(med, 1e-2)), cls, 3.6)
    legend_col(b, 556, 262, [("s2", "plain η = 1"), ("s1", "natural η = 0.5")], 15)
    note(b, 70, 372, ["Left: the plain gradient's size falls like w⁻³ in saturation; the natural gradient's Riemannian size stays above k = 1, but its step in w grows like w^{3/2} (Ḡ/G² ∝ w³).",
                      "Right: lines are the averaged iteration (plain 124 steps, natural 4), dots the median of 400 online runs (one sample per step, natural gain 0.05)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Saturation: natural against plain gradient on one neuron",
        "Left: on logarithmic axes, for one erf neuron with noise variance 1 and true weight 1, the squared size of the plain gradient falls like w to the minus 3, the natural gradient's Riemannian size stays between 1 and 1.4, and the squared size of the natural step in weight units grows like w cubed, reaching 3100 at w = 8. Right: distance to the true weight against the number of iterations for the averaged plain gradient with step 1, which needs 124 iterations, and the averaged natural gradient with step 0.5, which needs 4; dots show medians of online runs.", b))


def fig_fisher(path):
    b = []; W, H = 800, 420
    us = np.exp(np.linspace(math.log(0.4), math.log(0.025), 22))
    ev = []; evb = []
    for u in us:
        G = fisher(from_zeta([u, 0.4, 1.0, 1.0])); ev.append(np.linalg.eigvalsh(G))
        Gb = np.zeros((4, 4)); Gb[:2, :2] = G[:2, :2]; Gb[2:, 2:] = G[2:, 2:]; evb.append(np.linalg.eigvalsh(Gb))
    ev = np.array(ev); evb = np.array(evb)
    P1 = Panel(b, 70, 56, 320, 250, (log10(0.025), log10(0.4)), (-14, 0.5))
    P1.frame([log10(0.03), log10(0.05), log10(0.1), log10(0.2), log10(0.4)], [-12, -8, -4, 0], "u = w₂ − w₁", "eigenvalues of G", "Fisher matrix near the overlap singularity", True,
             {log10(0.03): "0.03", log10(0.05): "0.05", log10(0.1): "0.1", log10(0.2): "0.2", log10(0.4): "0.4"}, {-12: "10⁻¹²", -8: "10⁻⁸", -4: "10⁻⁴", 0: "1"})
    lu = np.log10(us)
    P1.line(lu, np.log10(ev[:, 0]), "ln s2"); P1.line(lu, np.log10(ev[:, 1]), "ln s1"); P1.line(lu, np.log10(ev[:, 2]), "ln s3"); P1.line(lu, np.log10(ev[:, 3]), "ln s0")
    P1.line(lu, np.log10(evb[:, 0]), "dash s4"); 
    P1.text(log10(0.04), -10.2, "slope 6", "sm", "start", 0, 0); P1.text(log10(0.06), -6.4, "slope 2", "sm", "start", 0, 0)
    legend_col(b, 212, 262, [("s2", "smallest (∝ u⁶)"), ("s1", "second (∝ u²)"), ("dash s4", "block approx. (u², u²)")], 15)
    r = STORE["info"]["T_pos"][1]
    P2 = Panel(b, 470, 56, 300, 250, (0.96, 1.12), (0, 0.6))
    P2.frame([1.0, 1.04, 1.08, 1.12], [0, 0.2, 0.4, 0.6], "z", "u", "Basin of the stable part of R_o (K > 0)", True)
    zz = np.linspace(1.0001, 1.12, 120)
    g = lambda z: math.log((z * z + 3) ** 2 / abs(z))
    h = np.array([(2 * r * r / 3) * (g(z) - g(1.0)) for z in zz]); usep = np.sqrt(2 * h)
    ok = usep <= 0.6
    polyx = [P2.X(z) for z in zz[ok]] + [P2.X(1.12), P2.X(1.12), P2.X(zz[ok][0])]
    polyy = [P2.Y(u) for u in usep[ok]] + [P2.Y(usep[ok][-1]), P2.Y(0), P2.Y(0)]
    b.append('<polygon class="fillS" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(polyx, polyy)) + '"/>')
    P2.line(zz[ok], usep[ok], "ln s1")
    P2.line([1.0, 1.0 + 0.6 / r], [0, 0.6], "dash s0")
    P2.line([1.0, 1.0], [0, 0.6], "dash sk")
    for u0, zc in STORE["wedge"]:
        P2.dot(zc, u0, "f2", 4.4)
    P2.text(1.075, 0.12, "trapped: u → 0", "sm", "middle"); P2.text(0.975, 0.3, "escapes", "sm", "start"); P2.text(1.003, 0.57, "|z| = 1", "sm", "start")
    legend_col(b, 486, 90, [("s1", "(12.148) through (1, 0)"), ("dash s0", "wedge z − 1 = u/r")], 15)
    note(b, 70, 372, ["Left: at fixed (s, r, z) the two smallest eigenvalues of the Fisher matrix scale like u⁶ and u²; a layer-wise block",
                      "approximation keeps the null vectors on the overlap set but is only of order u² small. Right: the exact plain-gradient flow (dots:",
                      "bisection at u₀ = 0.4 and 0.2) has a wedge-shaped basin next to |z| = 1, as the printed separatrix says: finite measure, not stable."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The Fisher matrix near the overlap singularity and the basin of the stable part",
        "Left: on logarithmic axes the four eigenvalues of the Fisher matrix of the two-neuron perceptron against the distance u between the two input weights: two stay of order one, the second smallest falls like u squared and the smallest like u to the sixth, and a layer-wise block-diagonal approximation falls like u squared in both small directions. Right: in the plane of z and u, the basin of the stable part of the overlap singularity for a teacher with K positive is a wedge between the line z equal to 1 and the printed separatrix; dots are bisection results on the exact flow.", b))


def fig_plateau(path):
    sg = STORE["slow_gd"]; sn = STORE["slow_ng"]; pp = STORE["plat_pos"]
    b = []; W, H = 800, 420
    P1 = Panel(b, 70, 56, 320, 250, (0, 7), (-14, -5))
    P1.frame([0, 2, 4, 6], [-13, -11, -9, -7, -5], "t", "loss L", "Teacher inside R (one neuron)", True, {0: "1", 2: "10²", 4: "10⁴", 6: "10⁶"}, {-13: "10⁻¹³", -11: "10⁻¹¹", -9: "10⁻⁹", -7: "10⁻⁷", -5: "10⁻⁵"})
    m = sg["t"] > 0.9
    P1.line(np.log10(sg["t"][m]), np.log10(np.maximum(sg["L"][m], 1e-300)), "ln s2")
    tn = [r_[0] for r_ in sn if r_[0] >= 1.0]; Ln = [r_[4] for r_ in sn if r_[0] >= 1.0]
    P1.line(np.log10(tn), np.log10(np.maximum(Ln, 1e-300)), "ln s1")
    t_ref = np.array([1e4, 1e7]); P1.line(np.log10(t_ref), np.log10(6.7e-13 * (t_ref / 1e7) ** -2), "dash s0")
    P1.text(4.0, -9.6, "slope −2", "sm", "start", 0, 0)
    legend_col(b, 190, 262, [("s2", "plain gradient"), ("s1", "natural gradient")], 15)
    P2 = Panel(b, 470, 56, 300, 250, (-0.7, 2.7), (0, 0.3))
    P2.frame([-0.5, 0, 1, 2], [0, 0.1, 0.2, 0.3], "t", "loss L", "Teacher outside R, start at u₀ = 0.1", True, {-0.5: "0.3", 0: "1", 1: "10", 2: "100"})
    P2.line([-0.7, 2.7], [pp["L1"], pp["L1"]], "dash s0")
    ga = [(t, L) for t, L in pp["gd_trap"] if t > 0.2]; gb = [(t, L) for t, L in pp["gd_escape"] if t > 0.2]
    P2.line([log10(t) for t, L in ga], [L for t, L in ga], "ln s2")
    P2.line([log10(t) for t, L in gb], [L for t, L in gb], "ln s3")
    nt = list(pp["ng_t"][1:]); nl = list(pp["ng_L"][1:])
    P2.line([log10(t) for t in nt], nl, "ln s1")
    P2.dot(-0.7, pp["ng_L"][0], "f1", 3.6); P2.text(-0.66, pp["ng_L"][0] - 0.012, "natural at t = 0", "sm", "start")
    P2.text(2.6, pp["L1"] + 0.008, "L₁ (best single neuron)", "sm", "end")
    legend_col(b, 486, 120, [("s2", "plain, z₀ = 1.5: trapped"), ("s3", "plain, z₀ = 0.6: escapes late"), ("s1", "natural, z₀ = 0.6")], 15)
    note(b, 70, 372, ["Left: the plain gradient flow slows down (u ~ t^{−1/2}, L ~ t⁻²); the natural flow has L ~ e^{−2t}. Right: the plain flow",
                      "stays at the plateau value L₁ for ever (z₀ = 1.5) or leaves after t ≈ 40 (z₀ = 0.6); the natural flow from z₀ = 0.6 is below L₁/2",
                      "at t ≈ 0.4. All curves are the averaged flow (12.129) of the two-neuron perceptron, from closed forms."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Critical slowdown and plateaus: plain against natural gradient",
        "Left: loss against time on logarithmic axes for a one-neuron teacher inside the critical region: the plain gradient flow decays like t to the minus 2 over seven decades of time while the natural gradient flow decays like exp(minus 2t) and is at 10 to the minus 13 by t = 9. Right: for a two-neuron teacher outside the critical region, loss against time on a logarithmic time axis: the plain flow started on the stable part stays at the plateau value for ever, the plain flow started on the unstable part leaves after about 40 time units, and the natural flow leaves in less than one time unit.", b))


def fig_simplex(path):
    rl = STORE["relax"]; f = rl["f"]
    b = []; W, H = 800, 405
    s3 = math.sqrt(3) / 2
    V = [(0.0, 0.0), (1.0, 0.0), (0.5, s3)]                       # outcome 0 (cost 0, the minimiser), 1 (cost 0.5), 2 (cost 1)
    P1 = Panel(b, 70, 56, 300, 260, (-0.05, 1.05), (-0.05, 0.95))
    xy = lambda p: (p[0] * V[0][0] + p[1] * V[1][0] + p[2] * V[2][0], p[0] * V[0][1] + p[1] * V[1][1] + p[2] * V[2][1])
    P1.frame([], [], "", "", "Three outcomes, costs 0, 0.5, 1", False)
    P1.line([V[0][0], V[1][0], V[2][0], V[0][0]], [V[0][1], V[1][1], V[2][1], V[0][1]], "ln sk")
    for tgn, tgd in rl["trajs"]:
        a = np.array([xy(p) for p in tgn]); c = np.array([xy(p) for p in tgd])
        P1.line(a[:, 0], a[:, 1], "ln s1"); P1.line(c[:, 0], c[:, 1], "ln s2")
        P1.dot(a[0, 0], a[0, 1], "f0", 3.2)
    P1.text(-0.04, -0.045, "cost 0: the minimiser", "sm", "start"); P1.text(1.05, -0.045, "cost 0.5", "sm", "end"); P1.text(0.5, s3 + 0.03, "cost 1", "sm", "middle")
    P2 = Panel(b, 470, 56, 300, 250, (-1, 4), (0, 1))
    th0 = np.log(np.array([0.001, 0.998, 0.001])[1:] / 0.001)
    def sm(th):
        z = np.concatenate([[0.0], th]); z = z - z.max(); e = np.exp(z); return e / e.sum()
    gd = lambda th: -(sm(th)[1:] * (f[1:] - sm(th) @ f))
    ts = np.logspace(-1, 3.65, 60); th = th0.copy(); t = 0.0; ys = []
    for tt in ts:
        n = max(1, int((tt - t) / 0.5)); th = rk4(gd, th, tt - t, n); t = tt; ys.append(float(sm(th)[0]))
    p0 = np.array([0.001, 0.998, 0.001]); ng_y = []
    for tt in ts:
        q = p0 * np.exp(-tt * f); ng_y.append(float((q / q.sum())[0]))
    P2.frame([-1, 0, 1, 2, 3, 4], [0, 0.5, 1], "t", "probability of the best outcome", "Start at the wrong vertex (0.001, 0.998, 0.001)", True, {-1: "0.1", 0: "1", 1: "10", 2: "10²", 3: "10³", 4: "10⁴"})
    P2.line(np.log10(ts), ng_y, "ln s1"); P2.line(np.log10(ts), ys, "ln s2"); P2.line([-1, 4], [0.9, 0.9], "dash s0")
    legend_col(b, 486, 250, [("s1", "natural: crosses 0.9 at t = 18"), ("s2", "plain: crosses 0.9 at t = 4073")], 15)
    note(b, 70, 372, ["Stochastic relaxation of E[f] on three outcomes (12.39)-(12.40). Natural gradient flow (blue): p_t ∝ p₀ e^{−tf}, each path a straight line in the θ coordinates;",
                      "plain gradient flow in θ (orange) crawls along the edges, because its speed carries the factor p_j that vanishes near a vertex."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Stochastic relaxation on a simplex",
        "Left: trajectories on the triangle of probability vectors over three outcomes with costs 0, 0.5 and 1 for the natural gradient flow of the expected cost (blue, exponential reweighting) and the plain gradient flow in the natural parameters (orange) from four starting points. Right: probability of the best outcome against time on a logarithmic axis starting near the wrong vertex: the natural flow reaches 0.9 at time 18, the plain flow at time 4073.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_steepest(out / "steepest-descent.svg")
    fig_online(out / "online-efficiency.svg")
    fig_saturation(out / "saturation.svg")
    fig_fisher(out / "singular-fisher.svg")
    fig_plateau(out / "plateau.svg")
    fig_simplex(out / "relaxation-simplex.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


def main():
    check_steepest()
    check_hessian()
    check_relaxation()
    check_policy()
    check_mirror()
    check_online()
    check_saturation()
    check_adaptive_eta()
    check_structure()
    check_reduced()
    check_plateau()
    check_singular_statistics()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
