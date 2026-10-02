#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 6, checked by hand.

Dual affine connections and dually flat manifolds. Every number quoted in the notes comes from here.

Conventions as in Chapters 5 and 7 of these notes: Gamma_{ij}^k are the components of nabla_{e_i} e_j = Gamma_{ij}^k e_k, so the
first index is the direction of differentiation, and Gamma_{ijk} = Gamma_{ij}^m g_{mk}. In the code G[i, j, k] stores Gamma_{ij}^k
and Gl[i, j, k] stores Gamma_{ijk}. The book's pair is (Gamma, Gamma*) with T = Gamma* - Gamma (6.11); for a statistical manifold
Gamma is the e-connection (alpha = +1) and Gamma* the m-connection (alpha = -1).

Running examples. (1) The Gaussian manifold in the chart (mu, sigma), with the Fisher metric diag(1/s^2, 2/s^2) and the closed
forms of Chapter 5 for the e-, m- and Levi-Civita connections; the alpha-connections are their mixtures. (2) The probability
simplex S_2 (three outcomes) in the m-chart (p1, p2), where g = diag(1/p_i) + 1/p_3 and Gamma^(alpha)_{ijk} = -(1+alpha)/2 T_{ijk}.
(3) A random metric with a random symmetric connection on R^2, to test algebraic identities with no special structure.
(4) Two and three binary neurons, a 2 x 2 binary channel and a Gaussian channel, a 3 x 3 input-output table.

Checked here, in the order the notes use them:

  1. dual connections (6.1)-(6.14): the duality condition, (nabla*)* = nabla, the average is metric, Theorem 6.1 and the sign of
     (6.14), torsion of the dual of a symmetric connection, parallel transport along a loop, submanifolds of a dual pair
     (what Chapter 7 uses);
  2. a divergence induces (g, Gamma, Gamma*) (6.19)-(6.32), by finite differences: KL, alpha-divergences, a generic non-Bregman
     divergence, half the squared geodesic distance, the transformation law, the Bregman case and the sign of T^{ijk}(eta);
     which divergences give the same geometry (6.51)-(6.54);
  3. f-divergences (6.33)-(6.36), including f with f''(1) != 1;
  4. alpha-geometry (6.38)-(6.39): the expectation formula, duality, curvature, the Jensen-type divergence;
  5. dually flat manifolds (6.40)-(6.50): curvature duality (a stronger identity than Theorem 6.5), the symmetric hypothesis,
     biorthogonal frames;
  6. canonical divergence of a dually flat manifold (6.51)-(6.71): the potential from the metric, eta = grad psi is nabla*-affine,
     the affine change (6.69), orientation of KL;
  7. canonical divergence of a general dual pair (6.72)-(6.85): geodesic boundary-value problems, the cubic Taylor tensor,
     Theorem 6.9, the projection condition (6.81) for several divergences;
  8. mixed coordinates and foliations (6.86)-(6.116): two and three neurons, the true diagonal blocks, the decomposition (6.96)
     and its orientation;
  9. integrated information (6.117)-(6.162): the split models, GI and GI', the printed relations, the Gaussian and binary channels;
 10. input-output tables (6.164)-(6.180): the coordinates, RAS, the decomposition.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Everything is deterministic (fixed seeds); the whole script takes about six seconds.

Run:  python3 dual_connections.py            (checks)
      python3 dual_connections.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

STORE = {}                          # numbers computed by the checks, reused by the figures
np.set_printoptions(linewidth=140, precision=6, suppress=True)


def head(s):
    print("\n" + s)


def sci(v):
    return f"{v:.1e}"


# ------------------------------------------------------------------ generic geometry in a chart

def num_dg(gfun, x, h=1e-5):
    """dg[l, i, j] = d_l g_ij by central differences."""
    n = len(x); dg = np.zeros((n, n, n))
    for l in range(n):
        e = np.zeros(n); e[l] = h
        dg[l] = (gfun(x + e) - gfun(x - e)) / (2 * h)
    return dg


def lower(G, g):
    """Gamma_{ijk} = Gamma_{ij}^m g_{mk}."""
    return np.einsum("ijm,mk->ijk", G, g)


def raise_(Gl, g):
    return np.einsum("ijm,mk->ijk", Gl, np.linalg.inv(g))


def dual_lowered(gfun, Gfun, x, h=1e-5):
    """(6.6): d_i g_jk = Gamma_ijk + Gamma*_ikj, so Gamma*_ijk = d_i g_kj - Gamma_ikj."""
    g = gfun(x); dg = num_dg(gfun, x, h); Gl = lower(Gfun(x), g)
    return dg - np.transpose(Gl, (0, 2, 1))


def dual_fn(gfun, Gfun, h=1e-5):
    def f(x):
        return raise_(dual_lowered(gfun, Gfun, x, h), gfun(x))
    return f


def riemann(Gfun, x, h=1e-5):
    """R_{ijk}^l = d_i Gamma_{jk}^l - d_j Gamma_{ik}^l + Gamma_{im}^l Gamma_{jk}^m - Gamma_{jm}^l Gamma_{ik}^m  (5.66)."""
    n = len(x); G = Gfun(x); dG = np.zeros((n, n, n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        dG[i] = (Gfun(x + e) - Gfun(x - e)) / (2 * h)
    R = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    R[i, j, k, l] = dG[i][j, k, l] - dG[j][i, k, l] + sum(G[i, m, l] * G[j, k, m] - G[j, m, l] * G[i, k, m] for m in range(n))
    return R


def cov_deriv_g(gfun, Gfun, x):
    """(nabla_i g)_{jk} = d_i g_jk - Gamma_ijk - Gamma_ikj."""
    g = gfun(x); dg = num_dg(gfun, x); Gl = lower(Gfun(x), g)
    return dg - Gl - np.transpose(Gl, (0, 2, 1))


def sym3(C):
    """Full symmetrisation of a 3-index array."""
    return sum(np.transpose(C, perm) for perm in itertools.permutations(range(3))) / 6


def maxabs(a):
    return float(np.max(np.abs(a)))


# ------------------------------------------------------------------ Example 1: the Gaussian manifold, chart (mu, sigma)

def g_gauss(x):
    return np.diag([1 / x[1] ** 2, 2 / x[1] ** 2])


def G_e(x):
    s = x[1]; G = np.zeros((2, 2, 2)); G[0, 1, 0] = G[1, 0, 0] = -2 / s; G[1, 1, 1] = -3 / s
    return G


def G_m(x):
    s = x[1]; G = np.zeros((2, 2, 2)); G[0, 0, 1] = 1 / s; G[1, 1, 1] = 1 / s
    return G


def G_lc(x):
    s = x[1]; G = np.zeros((2, 2, 2)); G[0, 1, 0] = G[1, 0, 0] = -1 / s; G[0, 0, 1] = 1 / (2 * s); G[1, 1, 1] = -1 / s
    return G


def G_alpha(a):
    """alpha-connection of the Gaussian manifold: (1+a)/2 e + (1-a)/2 m (alpha = +1 is the e-connection)."""
    return lambda x: (1 + a) / 2 * G_e(x) + (1 - a) / 2 * G_m(x)


ZN, ZW = np.polynomial.hermite_e.hermegauss(60)
ZW = ZW / math.sqrt(2 * math.pi)


def gauss_scores(x):
    """First and second derivatives of log p(t; mu, sigma) at the Gauss-Hermite nodes t = mu + sigma z (exact for polynomials)."""
    m, s = x; t = m + s * ZN
    l1 = np.array([(t - m) / s ** 2, ((t - m) ** 2 - s ** 2) / s ** 3])
    l2 = np.zeros((2, 2, len(t)))
    l2[0, 0] = -1 / s ** 2; l2[0, 1] = l2[1, 0] = -2 * (t - m) / s ** 3; l2[1, 1] = -3 * (t - m) ** 2 / s ** 4 + 1 / s ** 2
    return l1, l2


def gauss_g_T(x):
    l1, _ = gauss_scores(x)
    g = np.einsum("ik,jk,k->ij", l1, l1, ZW)
    T = np.einsum("ik,jk,lk,k->ijl", l1, l1, l1, ZW)
    return g, T


def gauss_gamma_alpha_expect(x, a):
    """Gamma^(alpha)_{ijk} = E[(d_i d_j l + (1-a)/2 d_i l d_j l) d_k l]  (the usual definition of the alpha-connection)."""
    l1, l2 = gauss_scores(x)
    return np.einsum("ijt,kt,t->ijk", l2 + (1 - a) / 2 * np.einsum("it,jt->ijt", l1, l1), l1, ZW)


def T_gauss(x):
    return gauss_g_T(x)[1]


# ------------------------------------------------------------------ Example 2: the simplex S_2 in the m-chart (p1, p2)

DP = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])         # d_i p_x [x, i]


def pvec(x):
    return np.array([x[0], x[1], 1 - x[0] - x[1]])


def g_s2(x):
    p = pvec(x); return np.einsum("xi,xj,x->ij", DP, DP, 1 / p)


def T_s2(x):
    p = pvec(x); return np.einsum("xi,xj,xk,x->ijk", DP, DP, DP, 1 / p ** 2)


def G_s2(a):
    """Gamma^(alpha)_{ijk} = E[(d_i d_j l + (1-a)/2 d_i l d_j l) d_k l] = -(1+a)/2 T_ijk in the m-chart, where d_i d_j p = 0."""
    return lambda x: raise_(-(1 + a) / 2 * T_s2(x), g_s2(x))


# ------------------------------------------------------------------ Example 3: a random metric and a random symmetric connection on R^2

def make_random_pair(seed=5):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(2, 2, 2, 3)) * 0.5

    def gfun(x):
        a, b = x
        A = np.array([[1 + 0.3 * math.sin(a + 0.5 * b), 0.4 * math.cos(b)], [0.2 * a * b, 1 + 0.2 * math.cos(a - b)]])
        return A @ A.T + 0.3 * np.eye(2)

    def Gfun(x):
        a, b = x
        basis = np.array([1.0, math.sin(a + b), math.cos(a - 0.5 * b)])
        T = np.einsum("ijkm,m->ijk", Q, basis)
        return (T + np.transpose(T, (1, 0, 2))) / 2

    return gfun, Gfun


# ------------------------------------------------------------------ parallel transport

def transport_pair(Gfun, Gsfun, gfun, curve, dcurve, A0, B0, steps=2000):
    """Parallel transport (5.58) of A by Gfun and of B by Gsfun along curve(t), t in [0, 1]; returns <A, B> at each step and the end vectors."""
    A = np.array(A0, float); B = np.array(B0, float); dt = 1.0 / steps

    def rhs(t, V, Gf):
        return -np.einsum("jki,j,k->i", Gf(curve(t)), dcurve(t), V)

    def step(t, V, Gf):
        k1 = rhs(t, V, Gf); k2 = rhs(t + dt / 2, V + dt / 2 * k1, Gf); k3 = rhs(t + dt / 2, V + dt / 2 * k2, Gf); k4 = rhs(t + dt, V + dt * k3, Gf)
        return V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    ip = [A @ gfun(curve(0.0)) @ B]; AA = [A.copy()]; BB = [B.copy()]; t = 0.0
    for k in range(steps):
        A = step(t, A, Gfun); B = step(t, B, Gsfun); t += dt
        ip.append(A @ gfun(curve(t)) @ B)
        if (k + 1) % (steps // 8) == 0:
            AA.append(A.copy()); BB.append(B.copy())
    return np.array(ip), A, B, AA, BB


# ------------------------------------------------------------------ 1. dual connections

def check_dual_connections():
    head("1. Dual connections (§6.1)")
    x = np.array([1.0, 2.0]); g = g_gauss(x)
    # 1.1 the duality condition (6.6) on the e/m pair of the Gaussian manifold
    dg = num_dg(g_gauss, x)
    Ge, Gm = lower(G_e(x), g), lower(G_m(x), g)
    res = dg - Ge - np.transpose(Gm, (0, 2, 1))
    print(f"(6.6) on the Gaussians at (mu, sigma) = (1, 2): max |d_k g_ij - Gamma^e_kij - Gamma^m_kji| = {sci(maxabs(res))}")
    print(f"      Gamma^e (mu sigma, sigma sigma) = ({G_e(x)[0, 1, 0]:.4f}, {G_e(x)[1, 1, 1]:.4f}),  Gamma^m (mu mu, sigma sigma) = ({G_m(x)[0, 0, 1]:.4f}, {G_m(x)[1, 1, 1]:.4f}),  T(mu mu sigma) = {T_gauss(x)[0, 0, 1]:.4f}, T(sigma sigma sigma) = {T_gauss(x)[1, 1, 1]:.4f}")
    D = dual_fn(g_gauss, G_e)
    print(f"      the dual of e built from g by (6.6) minus the closed-form m: {sci(maxabs(D(x) - G_m(x)))}; (e*)* - e: {sci(maxabs(dual_fn(g_gauss, D)(x) - G_e(x)))}")
    print(f"      T = Gamma^m - Gamma^e against the expectation E[d_i l d_j l d_k l]: {sci(maxabs(Gm - Ge - T_gauss(x)))}")
    # 1.2 the same on the simplex for several alpha, and the dual pair is (alpha, -alpha)
    xs = np.array([0.3, 0.25]); worst = 0.0
    for a in (-1.0, 0.0, 0.5, 1.0):
        worst = max(worst, maxabs(dual_fn(g_s2, G_s2(a))(xs) - G_s2(-a)(xs)))
    print(f"      on the simplex S_2 at p = (0.3, 0.25, 0.45): the dual of the alpha-connection is the (-alpha)-connection for alpha = -1, 0, 0.5, 1: max error {sci(worst)}")
    # the average of the (0.5, -0.5) pair is the Levi-Civita connection (5.85) built from g (on S_2 and the Gaussians)
    for name, gf, Gf, pt in (("simplex S_2", g_s2, G_s2, xs), ("Gaussians", g_gauss, G_alpha, x)):
        gp = gf(pt); dgp = num_dg(gf, pt)
        lc = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    lc[i, j, k] = 0.5 * (dgp[i][j, k] + dgp[j][i, k] - dgp[k][i, j])
        avg = 0.5 * (lower(Gf(0.5)(pt), gp) + lower(Gf(-0.5)(pt), gp))
        print(f"(6.9) on the {name}: (Gamma^(0.5) + Gamma^(-0.5))/2 equals the Levi-Civita symbol (5.85) of g to {sci(maxabs(avg - lc))}")
    # 1.3 Theorem 6.1: nabla g = T and nabla* g = -T; (6.14) with the inverse metric
    T = T_gauss(x)
    ce = cov_deriv_g(g_gauss, G_e, x); cm = cov_deriv_g(g_gauss, G_m, x)
    print(f"Theorem 6.1 (6.13): (nabla^e g)_ijk - T_ijk = {sci(maxabs(ce - T))}; (nabla^m g)_ijk + T_ijk = {sci(maxabs(cm + T))}")
    gi = np.linalg.inv(g)
    Tup = np.einsum("ijk,ja,kb->iab", T, gi, gi)                    # T_i^{jk}
    nab = np.zeros((2, 2, 2)); h = 1e-5
    for l in range(2):
        e = np.zeros(2); e[l] = h
        nab[l] = (np.linalg.inv(g_gauss(x + e)) - np.linalg.inv(g_gauss(x - e))) / (2 * h)       # d_l g^{jk}
    nab = nab + np.einsum("iaj,ak->ijk", G_m(x), gi) + np.einsum("iak,ja->ijk", G_m(x), gi)      # nabla^m_i g^{jk}
    print(f"(6.14) read for the INVERSE metric tensor: nabla^m_i g^{{jk}} - (+T_i^{{jk}}) = {sci(maxabs(nab - Tup))}, whereas nabla^m_i g^{{jk}} - (-T_i^{{jk}}) = {maxabs(nab + Tup):.4f}: with g^{{jk}} the inverse metric the sign is +; "
          f"the printed minus is right only if g^{{jk}} means the components of the metric in the dual chart eta (then it is (6.13) for nabla* written there, d^i d^j d^k phi = -T^{{ijk}}, see §2)")
    STORE["thm61"] = {"inverse_plus": maxabs(nab - Tup), "inverse_minus": maxabs(nab + Tup)}
    # 1.4 a random metric with a random symmetric connection on R^2: no special structure
    gfun, Gfun = make_random_pair()
    xr = np.array([0.3, -0.4]); gr = gfun(xr)
    Gstar = dual_fn(gfun, Gfun); Gss = dual_fn(gfun, Gstar)
    G0 = lambda y: 0.5 * (Gfun(y) + Gstar(y))
    Gs = Gstar(xr)
    tors = Gs - np.transpose(Gs, (1, 0, 2))
    cod = cov_deriv_g(gfun, Gfun, xr); cod = cod - np.transpose(cod, (1, 0, 2))
    print(f"random pair on R^2: (Gamma*)* - Gamma = {sci(maxabs(Gss(xr) - Gfun(xr)))}; metric defect of the average (nabla^0 g) = {sci(maxabs(cov_deriv_g(gfun, G0, xr)))}")
    print(f"      the dual of a SYMMETRIC connection is not symmetric: max |Gamma*_ij^k - Gamma*_ji^k| = {maxabs(tors):.4f}; it equals the Codazzi defect (nabla_i g)_jk - (nabla_j g)_ik to {sci(maxabs(lower(tors, gr) - cod))}")
    print(f"      so the average (Gamma + Gamma*)/2 is metric but has torsion {maxabs(0.5 * tors):.4f}: it is the Levi-Civita connection only if both connections are symmetric")
    Tr = lower(Gs, gr) - lower(Gfun(xr), gr)
    print(f"      T = Gamma* - Gamma is symmetric in its last two indices to {sci(maxabs(Tr - np.transpose(Tr, (0, 2, 1))))} (that comes from (6.6) and the symmetry of g) but not in its first two, {maxabs(Tr - np.transpose(Tr, (1, 0, 2))):.4f}: "
          f"the second symmetry is exactly the hypothesis that Gamma* is symmetric")
    STORE["random_pair"] = {"torsion": maxabs(tors)}
    # 1.5 parallel transport round a loop: the dual pair preserves <A, B>, nothing else does
    al = 0.6
    Gf, Gsf = G_alpha(al), G_alpha(-al)
    curve = lambda t: np.array([1 + 0.6 * math.cos(2 * math.pi * t), 2 + 0.5 * math.sin(2 * math.pi * t)])
    dcurve = lambda t: np.array([-0.6 * 2 * math.pi * math.sin(2 * math.pi * t), 0.5 * 2 * math.pi * math.cos(2 * math.pi * t)])
    A0, B0 = np.array([1.0, 0.3]), np.array([-0.2, 0.8])
    ip, A1, B1, AA, BB = transport_pair(Gf, Gsf, g_gauss, curve, dcurve, A0, B0)
    ip_same, _, _, _, _ = transport_pair(Gf, Gf, g_gauss, curve, dcurve, A0, B0)
    ip_lc, _, _, _, _ = transport_pair(G_lc, G_lc, g_gauss, curve, dcurve, A0, B0)
    ip_em, Ae, Be, _, _ = transport_pair(G_e, G_m, g_gauss, curve, dcurve, A0, B0)
    print(f"(6.1) transport round the ellipse mu = 1 + 0.6 cos 2 pi t, sigma = 2 + 0.5 sin 2 pi t, A by the 0.6-connection, B by the (-0.6)-connection: <A, B> = {ip[0]:.6f} at the start, {ip[-1]:.6f} at the end, largest drift {sci(np.abs(ip - ip[0]).max())}")
    print(f"      neither vector returns (this pair is curved): |A_end - A_0| = {np.linalg.norm(A1 - A0):.4f}, |B_end - B_0| = {np.linalg.norm(B1 - B0):.4f}")
    print(f"      the same 0.6-connection for both: largest drift {np.abs(ip_same - ip_same[0]).max():.4f}; Levi-Civita for both: {sci(np.abs(ip_lc - ip_lc[0]).max())}; the flat e/m pair: {sci(np.abs(ip_em - ip_em[0]).max())}, and both vectors return exactly ({sci(np.linalg.norm(Ae - A0))}, {sci(np.linalg.norm(Be - B0))})")
    # holonomy matrices of the dual pair round the loop: H^T g H* = g, i.e. H* is the g-adjoint inverse of H (the heart of the proof of Theorem 6.5)
    H = np.zeros((2, 2)); Hs = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = 1.0
        _, Ak, Bk, _, _ = transport_pair(Gf, Gsf, g_gauss, curve, dcurve, e, e)
        H[:, k] = Ak; Hs[:, k] = Bk
    g0 = g_gauss(curve(0.0))
    print(f"      holonomy matrices round the loop: H^T g H* - g = {sci(maxabs(H.T @ g0 @ Hs - g0))}, so H* = g^-1 H^-T g and H = 1 exactly when H* = 1; here det H = {np.linalg.det(H):.6f}, det H* = {np.linalg.det(Hs):.6f}, "
          f"rotation-like deviation |H - 1| = {maxabs(H - np.eye(2)):.4f}")
    STORE["transport"] = {"curve": np.array([curve(t) for t in np.linspace(0, 1, 201)]), "stations": np.array([curve(k / 8) for k in range(9)]),
                          "A": np.array(AA), "B": np.array(BB), "ip": ip, "ip_same": ip_same, "ip_lc": ip_lc, "A0": A0, "B0": B0}


def check_submanifold():
    head("1b. Submanifolds of a dual pair (what Chapter 7 uses)")
    # the curve S = {N(u, u^2)} in the Gaussian manifold: (mu, sigma) = (u, u); at u = 1
    u = 1.0; x = np.array([u, u]); g = g_gauss(x)
    B = np.array([1.0, 1.0]); guu = B @ g @ B
    # induced connection coefficients: Gamma^S_{uu,u} = <nabla_u d_u, d_u> = Gamma_{ij k} B^i B^j B^k (the chart (mu, sigma) = (u, u) is linear, d B = 0)
    Gem = np.einsum("ijk,i,j,k->", lower(G_e(x), g), B, B, B) / guu
    Gmm = np.einsum("ijk,i,j,k->", lower(G_m(x), g), B, B, B) / guu
    # the same from the natural-parameter picture used in Chapter 7: theta(u), eta(u)
    th = lambda u_: np.array([1 / u_, -1 / (2 * u_ ** 2)]); thd = np.array([-1.0, 1.0]); thdd = np.array([2.0, -3.0])
    etd = np.array([1.0, 4.0]); etdd = np.array([0.0, 4.0])
    print(f"curve N(u, u^2) at u = 1: g_uu = {guu:.4f} (3/u^2 = 3); induced m-connection coefficient Gamma^(m)u_uu = {Gmm:.6f}; Chapter 7's formula eta''.theta'/g_uu = {etdd @ thd / guu:.6f}")
    print(f"      induced e-coefficient = {Gem:.6f}; theta''.eta'/g_uu = {thdd @ etd / guu:.6f}; their sum {Gem + Gmm:.6f} = g_uu'/g_uu = -2/u = {-2 / u:.6f}: the induced connections are again dual (6.6 on S)")
    Gm_ = np.array([[1 / (u * u), 0], [0, 2 / (u * u)]])
    # e-embedding curvature of the curve: normal part of theta'' in the Fisher metric of the theta chart
    Gth = np.array([[1.0, 2.0], [2.0, 6.0]])                              # Fisher information in theta at u = 1 = Cov[(x, x^2)]
    nrm = thdd - (thdd @ Gth @ thd) / (thd @ Gth @ thd) * thd
    gamma2 = (nrm @ Gth @ nrm) / (thd @ Gth @ thd) ** 2
    print(f"      e-embedding curvature: |normal part of theta''|^2_g = {nrm @ Gth @ nrm:.6f} (2/3), statistical curvature gamma^2 = {gamma2:.6f} (2/27 = {2 / 27:.6f}); both as used in Chapter 7")
    # a two-dimensional submanifold of the simplex S_3 (four outcomes) and duality of the induced connections
    def p4(w):
        a, b = w
        q = np.array([1.0 + 0.3 * math.sin(a), 1.5 + 0.2 * b * b + 0.1 * a, 1.2 + 0.5 * math.cos(b), 0.8 + 0.2 * a * b + 0.3])
        return q / q.sum()
    # embed through the m-chart (p1, p2, p3) of the 3-simplex
    def emb(w):
        return p4(w)[:3]
    def Tm3(xx):
        p = np.array([xx[0], xx[1], xx[2], 1 - xx.sum()]); D3 = np.vstack([np.eye(3), -np.ones((1, 3))])
        return np.einsum("xi,xj,xk,x->ijk", D3, D3, D3, 1 / p ** 2), np.einsum("xi,xj,x->ij", D3, D3, 1 / p)
    w0 = np.array([0.4, -0.3]); h = 1e-4
    Bm = np.zeros((2, 3)); H2 = np.zeros((2, 2, 3))
    for a in range(2):
        e = np.zeros(2); e[a] = h
        Bm[a] = (emb(w0 + e) - emb(w0 - e)) / (2 * h)
        for b in range(2):
            f = np.zeros(2); f[b] = h
            H2[a, b] = (emb(w0 + e + f) - emb(w0 + e - f) - emb(w0 - e + f) + emb(w0 - e - f)) / (4 * h * h)
    x3 = emb(w0); T3, g3 = Tm3(x3)
    gS = Bm @ g3 @ Bm.T
    alpha = 0.4
    # ambient Gamma^(alpha)_{ijk} = -(1+alpha)/2 T_ijk in the m-chart; induced (5.100): Gamma_abc = B B B Gamma_ijk + B_c^j (d_a B_b^i) g_ij
    def induced(al):
        amb = -(1 + al) / 2 * T3
        return np.einsum("ai,bj,ck,ijk->abc", Bm, Bm, Bm, amb) + np.einsum("cj,abi,ij->abc", Bm, H2, g3)
    Ga, Gb = induced(alpha), induced(-alpha)
    dgS = np.zeros((2, 2, 2))
    for c in range(2):
        e = np.zeros(2); e[c] = h
        Bp = np.zeros((2, 3)); Bq = np.zeros((2, 3))
        for a in range(2):
            f = np.zeros(2); f[a] = h
            Bp[a] = (emb(w0 + e + f) - emb(w0 + e - f)) / (2 * h); Bq[a] = (emb(w0 - e + f) - emb(w0 - e - f)) / (2 * h)
        _, gp = Tm3(emb(w0 + e)); _, gq = Tm3(emb(w0 - e))
        dgS[c] = (Bp @ gp @ Bp.T - Bq @ gq @ Bq.T) / (2 * h)
    res = dgS - Ga - np.transpose(Gb, (0, 2, 1))
    print(f"a 2-dimensional surface in the simplex S_3, alpha = 0.4: induced (alpha) and (-alpha) connections satisfy (6.6) with the induced metric to {sci(maxabs(res))}")


# ------------------------------------------------------------------ divergences and the geometry they induce (6.19)-(6.25)

def mixed(D, x, y, ops, h=0.01):
    """Mixed partial derivative of D(x, y) at (x, y) with respect to the variables listed in ops (0..n-1: x, n..2n-1: y);
    central differences, Richardson extrapolation in h."""
    n = len(x); z0 = np.concatenate([x, y]).astype(float)

    def one(hh):
        tot = 0.0
        for signs in itertools.product((1, -1), repeat=len(ops)):
            z = z0.copy()
            for s_, o in zip(signs, ops):
                z[o] += s_ * hh
            tot += np.prod(signs) * D(z[:n], z[n:])
        return tot / (2 * hh) ** len(ops)

    return (4 * one(h / 2) - one(h)) / 3


def geometry_from_D(D, x, h=0.01):
    """g^D_ij = -D_{i;j}, Gamma^D_ijk = -D_{ij;k}, Gamma^D*_ijk = -D_{k;ij}  (6.22)-(6.24); returns (g, Gamma, Gamma*) lowered."""
    n = len(x)
    g = np.zeros((n, n)); Gam = np.zeros((n, n, n)); Gs = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            g[i, j] = -mixed(D, x, x, [i, n + j], h)
            for k in range(n):
                Gam[i, j, k] = -mixed(D, x, x, [i, j, n + k], h)
                Gs[i, j, k] = -mixed(D, x, x, [k, n + i, n + j], h)
    return g, Gam, Gs


def kl_gauss(x, y):
    """KL[N(x) : N(y)] for x = (mu1, s1), y = (mu2, s2)."""
    m1, s1 = x; m2, s2 = y
    return math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def renyi_int(x, y, a):
    """Integral of p^a q^(1-a) for two Gaussians."""
    m1, s1 = x; m2, s2 = y
    v = a * s2 ** 2 + (1 - a) * s1 ** 2
    return s1 ** (1 - a) * s2 ** a / math.sqrt(v) * math.exp(-a * (1 - a) * (m1 - m2) ** 2 / (2 * v))


def alpha_div_gauss(al):
    """D_alpha[p:q] = 4/(1-alpha^2) (1 - int p^((1-alpha)/2) q^((1+alpha)/2)); alpha -> -1 is KL[p:q], alpha -> +1 is KL[q:p]."""
    a = (1 - al) / 2
    return lambda x, y: 4 / (1 - al ** 2) * (1 - renyi_int(x, y, a))


def fr_dist(x, y):
    """Fisher-Rao distance between N(x) and N(y): the metric is 2 x the hyperbolic metric of the half-plane (mu/sqrt 2, sigma)."""
    (m1, s1), (m2, s2) = x, y
    return math.sqrt(2) * math.acosh(1 + ((m1 - m2) ** 2 / 2 + (s1 - s2) ** 2) / (2 * s1 * s2))


def check_eguchi():
    head("2. Metric and cubic tensor from a divergence (§6.2)")
    x = np.array([1.0, 2.0]); g0 = g_gauss(x); T0 = T_gauss(x)
    # numerical quadrature check of the closed form used for D_alpha
    ts = np.linspace(-30, 30, 400001); dt = ts[1] - ts[0]
    pdf = lambda t, m, s: np.exp(-(t - m) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
    y = np.array([0.4, 1.3]); a = 0.3
    print(f"(closed form of the Renyi integral, a = 0.3: {renyi_int(x, y, a):.10f}; trapezoid rule on [-30, 30]: {np.sum(pdf(ts, *x) ** a * pdf(ts, *y) ** (1 - a)) * dt:.10f})")
    rows = []
    cases = [("KL[p:q]", kl_gauss, -1.0), ("KL[q:p]", lambda a_, b_: kl_gauss(b_, a_), 1.0)] + [(f"D_alpha, alpha = {al_:+.1f}", alpha_div_gauss(al_), al_) for al_ in (0.0, 0.5, -0.7)]
    for name, Dv, al in cases:
        g, Gam, Gs = geometry_from_D(Dv, x)
        eg = maxabs(g - g0); e1 = maxabs(Gam - lower(G_alpha(al)(x), g0)); e2 = maxabs(Gs - lower(G_alpha(-al)(x), g0)); e3 = maxabs((Gs - Gam) - al * T0)
        print(f"{name:20s}: g err {sci(eg)} | Gamma^D vs Gamma^(alpha) err {sci(e1)} | Gamma^D* vs Gamma^(-alpha) err {sci(e2)} | T^D - alpha T err {sci(e3)}")
        rows.append((name, al, eg, e1, e2, e3))
    STORE["eguchi_gauss"] = rows
    # the three expressions of the metric (6.22) and the symmetry of the induced tensors
    D = kl_gauss
    gA = np.array([[-mixed(D, x, x, [i, 2 + j]) for j in range(2)] for i in range(2)])
    gB = np.array([[mixed(D, x, x, [i, j]) for j in range(2)] for i in range(2)])
    gC = np.array([[mixed(D, x, x, [2 + i, 2 + j]) for j in range(2)] for i in range(2)])
    print(f"(6.22) -D_i;j, D_ij and D_;ij agree for KL: {sci(maxabs(gA - gB))}, {sci(maxabs(gA - gC))} (and equal the Fisher metric to {sci(maxabs(gA - g0))})")
    # the same quadratic part, a different cubic part: D(p, p + s d) = 1/2 g(d,d) s^2 + c3 s^3 + ... with c3 = 1/2 Gamma^0(d,d,d) + alpha/12 T(d,d,d)
    d = np.array([0.6, 0.8]); G0l = lower(G_lc(x), g0)
    pred = lambda al_: 0.5 * np.einsum("ijk,i,j,k->", G0l, d, d, d) + al_ / 12 * np.einsum("ijk,i,j,k->", T0, d, d, d)

    def cubic_coeff(Dv):
        u = [(Dv(x, x + s_ * d) - 0.5 * d @ g0 @ d * s_ ** 2) / s_ ** 3 for s_ in (0.02, 0.04, 0.08)]
        return (8 * u[0] - 6 * u[1] + u[2]) / 3

    sgrid = np.linspace(0.02, 0.6, 30); curves = []
    print(f"cubic Taylor coefficient along d = (0.6, 0.8) at (1, 2), D(p, p + s d) = 1/2 g(d,d) s^2 + c3 s^3 + ..., 1/2 g(d,d) = {0.5 * d @ g0 @ d:.6f}; "
          f"prediction c3 = 1/2 Gamma^0(d,d,d) + (alpha/12) T(d,d,d) with Gamma^0(d,d,d) = {np.einsum('ijk,i,j,k->', G0l, d, d, d):.6f}, T(d,d,d) = {np.einsum('ijk,i,j,k->', T0, d, d, d):.6f}:")
    for name, Dv, al in (("KL[p:q]", kl_gauss, -1.0), ("D_alpha, alpha = -0.5", alpha_div_gauss(-0.5), -0.5), ("D_alpha, alpha = 0", alpha_div_gauss(0.0), 0.0),
                         ("half the squared distance", lambda a_, b_: 0.5 * fr_dist(a_, b_) ** 2, 0.0), ("D_alpha, alpha = +0.5", alpha_div_gauss(0.5), 0.5), ("KL[q:p]", lambda a_, b_: kl_gauss(b_, a_), 1.0)):
        c3 = cubic_coeff(Dv)
        print(f"   {name:28s} alpha = {al:+.1f}: c3 = {c3:+.6f}, predicted {pred(al):+.6f} (difference {sci(abs(c3 - pred(al)))})")
        curves.append((name, al, c3, pred(al), np.array([Dv(x, x + s_ * d) / s_ ** 2 for s_ in sgrid])))
    STORE["cubic"] = {"s": sgrid, "curves": curves, "quad": 0.5 * d @ g0 @ d, "slope_T": np.einsum("ijk,i,j,k->", T0, d, d, d) / 12, "c0": 0.5 * np.einsum("ijk,i,j,k->", G0l, d, d, d)}
    # generic non-Bregman, non-f divergence D = 1/2 d.M(x).d + 1/6 C(x)[d,d,d] + 0.3/24 |d|^4,  d = y - x
    rng = np.random.default_rng(3)
    A1 = rng.normal(size=(2, 2)); c0 = sym3(rng.normal(size=(2, 2, 2)))

    def M(x_):
        Bm = A1 + 0.3 * np.array([[math.sin(x_[0]), x_[1]], [x_[0] * x_[1], math.cos(x_[1])]])
        return Bm @ Bm.T + 0.5 * np.eye(2)

    C = lambda x_: c0 * (1 + 0.4 * math.sin(x_[0] + 2 * x_[1]))

    def Dg(x_, y_):
        d = y_ - x_
        return 0.5 * d @ M(x_) @ d + np.einsum("ijk,i,j,k->", C(x_), d, d, d) / 6 + 0.3 * (d @ d) ** 2 / 24

    xg = np.array([0.4, -0.3])
    g, Gam, Gs = geometry_from_D(Dg, xg)
    dM = num_dg(M, xg); Cx = C(xg)
    pred = np.zeros((2, 2, 2)); preds = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pred[i, j, k] = dM[i][j, k] + dM[j][i, k] - Cx[i, j, k]
                preds[i, j, k] = Cx[i, j, k] - dM[k][i, j]
    T = Gs - Gam
    asym = max(maxabs(T - np.transpose(T, p)) for p in itertools.permutations(range(3)))
    dual_res = maxabs(dM - Gam - np.transpose(Gs, (0, 2, 1)))
    print(f"generic D = 1/2 d.M(x).d + 1/6 C(x)[d,d,d] + quartic (neither Bregman nor an f-divergence): g err {sci(maxabs(g - M(xg)))}; "
          f"Gamma^D = d_i M_jk + d_j M_ik - C_ijk to {sci(maxabs(Gam - pred))}; Gamma^D* = C_ijk - d_k M_ij to {sci(maxabs(Gs - preds))}")
    print(f"      Theorem 6.2: d_k g_ij = Gamma^D_kij + Gamma^D*_kji to {sci(dual_res)}; T^D = Gamma^D* - Gamma^D is symmetric in all three indices to {sci(asym)}")
    # same geometry from D + d (6.51)-(6.54) and a counterexample
    def D2(x_, y_):
        d = y_ - x_
        return Dg(x_, y_) + 3.0 * (d @ d) ** 2 + 0.7 * d[0] ** 2 * d[1] ** 2 + 0.5 * Dg(x_, y_) ** 2
    g2, Gam2, Gs2 = geometry_from_D(D2, xg)

    def D3(x_, y_):
        d = y_ - x_
        return Dg(x_, y_) + 0.5 * d[0] ** 3
    g3, Gam3, Gs3 = geometry_from_D(D3, xg)
    print(f"(6.51)-(6.54): D + 3|d|^4 + 0.7 d1^2 d2^2 + D^2/2 has the same geometry: change in g {sci(maxabs(g2 - g))}, Gamma {sci(maxabs(Gam2 - Gam))}, Gamma* {sci(maxabs(Gs2 - Gs))}; "
          f"adding the cubic term d1^3/2 instead changes Gamma by {maxabs(Gam3 - Gam):.4f} and Gamma* by {maxabs(Gs3 - Gs):.4f}")
    STORE["generic_D"] = {"gam_err": maxabs(Gam - pred), "dual_res": dual_res, "asym": asym, "same_geometry": maxabs(Gam2 - Gam), "cubic_change": maxabs(Gam3 - Gam)}
    # half the squared Fisher-Rao distance: a Riemannian divergence, T = 0
    Dh = lambda a_, b_: 0.5 * fr_dist(a_, b_) ** 2
    g, Gam, Gs = geometry_from_D(Dh, x)
    GL = lower(G_lc(x), g0)
    print(f"half the squared Fisher-Rao distance: g err {sci(maxabs(g - g0))}; Gamma^D and Gamma^D* both equal the Levi-Civita symbol to {sci(maxabs(Gam - GL))}, {sci(maxabs(Gs - GL))}; T^D = {sci(maxabs(Gs - Gam))}")
    # transformation law: Gamma^D computed in the natural parameters theta against the law (5.37) applied to the (mu, sigma) result
    def theta_of(xi): return np.array([xi[0] / xi[1] ** 2, -1 / (2 * xi[1] ** 2)])

    def xi_of_theta(th):
        s2 = -1 / (2 * th[1]); return np.array([th[0] * s2, math.sqrt(s2)])

    th = theta_of(x)
    Dth = lambda a_, b_: kl_gauss(xi_of_theta(a_), xi_of_theta(b_))
    g_t, Gam_t, Gs_t = geometry_from_D(Dth, th, 0.004)
    gx, Gamx, Gsx = geometry_from_D(kl_gauss, x)
    # Jacobian J[i, k] = d xi^i / d theta^k and Hessian by central differences of xi_of_theta
    J = np.zeros((2, 2)); Hn = np.zeros((2, 2, 2)); h = 1e-4
    for k in range(2):
        e = np.zeros(2); e[k] = h
        J[:, k] = (xi_of_theta(th + e) - xi_of_theta(th - e)) / (2 * h)
        for l in range(2):
            f = np.zeros(2); f[l] = h
            Hn[:, k, l] = (xi_of_theta(th + e + f) - xi_of_theta(th + e - f) - xi_of_theta(th - e + f) + xi_of_theta(th - e - f)) / (4 * h * h)
    law = np.einsum("ia,jb,pc,ijp->abc", J, J, J, Gamx) + np.einsum("jab,pc,jp->abc", Hn, J, gx)
    print(f"(5.37) applied to Gamma^D computed from KL[p:q] in (mu, sigma): agrees with Gamma^D computed directly in theta to a relative {maxabs(law - Gam_t) / maxabs(Gam_t):.1e} "
          f"(the tensor part alone is off by {maxabs(np.einsum('ia,jb,pc,ijp->abc', J, J, J, Gamx) - Gam_t):.1f}); in theta, Gamma^D* = {sci(maxabs(Gs_t))} and Gamma^D = (psi'''): "
          f"{Gam_t[0, 0, 0]:.3f}, {Gam_t[0, 0, 1]:.3f}, {Gam_t[0, 1, 1]:.3f}, {Gam_t[1, 1, 1]:.3f}")
    # the same in a well-conditioned chart zeta = (mu, log sigma), where the numbers are O(1)
    xi_of_zeta = lambda z: np.array([z[0], math.exp(z[1])])
    zeta = np.array([x[0], math.log(x[1])])
    Dz = lambda a_, b_: kl_gauss(xi_of_zeta(a_), xi_of_zeta(b_))
    g_z, Gam_z, Gs_z = geometry_from_D(Dz, zeta)
    Jz = np.array([[1.0, 0.0], [0.0, x[1]]]); Hz = np.zeros((2, 2, 2)); Hz[1, 1, 1] = x[1]
    law_z = np.einsum("ia,jb,pc,ijp->abc", Jz, Jz, Jz, Gamx) + np.einsum("jab,pc,jp->abc", Hz, Jz, gx)
    print(f"      and in the chart (mu, log sigma): law (5.37) against the direct computation {sci(maxabs(law_z - Gam_z))}, tensor part alone off by {maxabs(np.einsum('ia,jb,pc,ijp->abc', Jz, Jz, Jz, Gamx) - Gam_z):.3f}")
    check_bregman()


def softmax_psi(th):
    return math.log(1 + math.exp(th[0]) + math.exp(th[1]))


def check_bregman():
    # Bregman case on the simplex S_2 (softmax): theta = (log p1/p3, log p2/p3), eta = (p1, p2)
    theta_of_p = lambda p: np.array([math.log(p[0] / p[2]), math.log(p[1] / p[2])])
    p_of_theta = lambda t: np.array([math.exp(t[0]), math.exp(t[1]), 1.0]) / (1 + math.exp(t[0]) + math.exp(t[1]))
    # D_psi[theta : theta'] = psi(theta) + phi(eta') - theta.eta' = KL[p_theta' : p_theta]
    def D_th(a, b):
        pa, pb = p_of_theta(a), p_of_theta(b)
        return softmax_psi(a) + float(np.sum(pb * np.log(pb))) - float(a @ pb[:2])
    t = np.array([0.3, -0.2]); pt = p_of_theta(t)
    klv = float(np.sum(p_of_theta(np.array([0.1, 0.5])) * np.log(p_of_theta(np.array([0.1, 0.5])) / pt)))
    print(f"(6.28) Bregman divergence of the cumulant function = KL with the arguments reversed: D_psi[theta : theta'] = {D_th(t, np.array([0.1, 0.5])):.10f}, KL[p_theta' : p_theta] = {klv:.10f}")
    g, Gam, Gs = geometry_from_D(D_th, t, 0.004)
    gA = np.diag(pt[:2]) - np.outer(pt[:2], pt[:2])
    Cs = np.vstack([np.eye(2) - pt[:2][None, :], -pt[:2][None, :]])                    # (s_x - eta) for the statistic s = (1[x=1], 1[x=2])
    Tth = np.einsum("x,xi,xj,xk->ijk", pt, Cs, Cs, Cs)                                    # third central moments of s = psi'''
    print(f"(6.29)-(6.32) in theta: g = psi'' to {sci(maxabs(g - gA))}; Gamma(theta) = {sci(maxabs(Gam))}; Gamma* = psi''' to {sci(maxabs(Gs - Tth))}")
    # dual chart: eta = (p1, p2)
    eta = pt[:2]
    D_et = lambda a, b: D_th(theta_of_p(np.array([a[0], a[1], 1 - a.sum()])), theta_of_p(np.array([b[0], b[1], 1 - b.sum()])))
    ge, Game, Gse = geometry_from_D(D_et, eta, 0.004)
    # phi''' by finite differences of the metric g^{ij}(eta) = phi'' = diag(1/eta_i) + 1/p3 (the metric in the eta chart)
    gE = lambda e: g_s2(e)
    dphi3 = num_dg(gE, eta)                                                  # d_k phi_ij
    Teta = T_s2(eta)                                                         # covariant components T(d^i, d^j, d^k) of the cubic tensor in the eta chart
    print(f"in eta: g = phi'' = diag(1/eta_i) + 1/p_3 to {sci(maxabs(ge - gE(eta)))}; Gamma*(eta) = {sci(maxabs(Gse))}; Gamma(eta) = phi''' to {sci(maxabs(Game - dphi3.transpose(1, 2, 0)))}")
    print(f"      the cubic tensor in the eta chart, T(d^i, d^j, d^k) = Gamma* - Gamma = {sci(maxabs((Gse - Game) - Teta))} from -phi''' = {sci(maxabs((Gse - Game) + dphi3.transpose(1, 2, 0)))}: "
          f"T^{{ijk}}(eta) = -d^i d^j d^k phi: consistent with (6.14) read in the eta chart, but (6.32) and (6.58) print +, which is the cubic tensor of the dual pair (g, -T)")
    STORE["bregman_sign"] = {"minus_phi3": maxabs((Gse - Game) + dphi3.transpose(1, 2, 0)), "plus_phi3": maxabs((Gse - Game) - dphi3.transpose(1, 2, 0)), "value": float(Teta[0, 0, 0])}
    print(f"      numbers at p = ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}): T(d^1, d^1, d^1) = {Teta[0, 0, 0]:.6f}, phi'''_111 = {dphi3[0, 0, 0]:.6f}")


# ------------------------------------------------------------------ 3. f-divergences

def fdiv_s2(f):
    def D(x, y):
        p = pvec(x); q = pvec(y)
        return float(np.sum(p * f(q / p)))
    return D


def alpha_f(al):
    return lambda u: 4 / (1 - al ** 2) * ((1 - al) / 2 + (1 + al) / 2 * u - u ** ((1 + al) / 2))


def check_fdiv():
    head("3. Invariant metric and cubic tensor: f-divergences (§6.3)")
    x = np.array([0.3, 0.25]); g0 = g_s2(x); T0 = T_s2(x)
    cases = [("-log u + u - 1 = KL[p:q]", lambda u: -np.log(u) + u - 1),
             ("u log u - u + 1 = KL[q:p]", lambda u: u * np.log(u) - u + 1),
             ("alpha = 0 (Hellinger)", alpha_f(0.0)),
             ("alpha = 0.5", alpha_f(0.5)),
             ("alpha = -0.7", alpha_f(-0.7)),
             ("Pearson (u-1)^2/2", lambda u: (u - 1) ** 2 / 2),
             ("Neyman (u-1)^2/(2u)", lambda u: (u - 1) ** 2 / (2 * u)),
             ("Jensen-Shannon", lambda u: u * np.log(u) - (1 + u) * np.log((1 + u) / 2))]
    rows = []
    print("On the simplex S_2 at p = (0.3, 0.25, 0.45), by finite differences of D_f[p:q] = sum p f(q/p):")
    print("%-28s %9s %10s %9s %9s %9s %7s" % ("f", "f''(1)", "f'''(1)", "g^f/g", "T^f/T", "2f3+3f2", "2f3+3"))
    for name, f in cases:
        g, Gam, Gs = geometry_from_D(fdiv_s2(f), x, 0.005)
        T = Gs - Gam; h = 1e-3
        f2 = (f(1 + h) - 2 * f(1) + f(1 - h)) / h ** 2
        f3 = (f(1 + 2 * h) - 2 * f(1 + h) + 2 * f(1 - h) - f(1 - 2 * h)) / (2 * h ** 3)
        ratio_g = float(np.sum(g * g0) / np.sum(g0 * g0)); ratio_T = float(np.sum(T * T0) / np.sum(T0 * T0))
        print(f"{name:28s} {f2:9.4f} {f3:10.4f} {ratio_g:9.6f} {ratio_T:9.6f} {2 * f3 + 3 * f2:9.4f} {2 * f3 + 3:7.4f}   residual of T = ratio*T0: {sci(maxabs(T - ratio_T * T0))}")
        rows.append((name, f2, f3, ratio_g, ratio_T))
    STORE["fdiv"] = rows
    print("so g^f = f''(1) g and T^f = (2 f'''(1) + 3 f''(1)) T; with f''(1) = 1 this is the book's alpha = 2 f'''(1) + 3 (6.36). Jensen-Shannon has f''(1) = 1/2 and T^f = 0.")


# ------------------------------------------------------------------ 4. alpha-geometry

def gauss_curv_scalar(R, g):
    """K = g_{1m} R_{122}^m / det g, the coefficient used for surfaces in Chapter 5 (for a non-metric connection it is just a number)."""
    return float(sum(g[0, m] * R[0, 1, 1, m] for m in range(2)) / np.linalg.det(g))


def jensen_div(psi, al):
    a = (1 - al) / 2; b = (1 + al) / 2
    return lambda x, y: 4 / (1 - al ** 2) * (a * psi(x) + b * psi(y) - psi(a * x + b * y))


def check_alpha():
    head("4. alpha-geometry (§6.4)")
    x = np.array([1.0, 2.0]); g = g_gauss(x); T = T_gauss(x); G0 = lower(G_lc(x), g)
    worst1 = worst2 = 0.0
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        ex = gauss_gamma_alpha_expect(x, a)
        worst1 = max(worst1, maxabs(ex - lower(G_alpha(a)(x), g)))
        worst2 = max(worst2, maxabs(ex - (G0 - a / 2 * T)))
    print(f"(6.38) on the Gaussians: Gamma^(alpha)_ijk = Gamma^0_ijk - (alpha/2) T_ijk agrees with the expectation E[(d_i d_j l + (1-alpha)/2 d_i l d_j l) d_k l] "
          f"(alpha = -1, -0.5, 0, 0.5, 1) to {sci(worst2)}, and with (1+alpha)/2 e + (1-alpha)/2 m to {sci(worst1)}")
    # alpha = 1 is the e-connection of the exponential family: Gamma^(1) = E[d_i d_j l d_k l]
    print(f"      alpha = +1 gives the e-connection (-2/s, -3/s at (1, 2): {gauss_gamma_alpha_expect(x, 1.0)[0, 1, 0] / g[0, 0]:.4f}, {gauss_gamma_alpha_expect(x, 1.0)[1, 1, 1] / g[1, 1]:.4f}), alpha = -1 the m-connection")
    # curvature
    rows = []
    print("curvature of the alpha-connection, R^(alpha) against (1 - alpha^2) R^(0):")
    for name, gf, Gf, pt in (("Gaussians at (1, 2)", g_gauss, G_alpha, np.array([1.0, 2.0])), ("simplex S_2 at (0.3, 0.25)", g_s2, G_s2, np.array([0.3, 0.25]))):
        gp = gf(pt); R0 = riemann(Gf(0.0), pt); K0 = gauss_curv_scalar(R0, gp)
        for a in (-1.0, -0.3, 0.5, 0.8, 1.0):
            Ra = riemann(Gf(a), pt)
            Ka = gauss_curv_scalar(Ra, gp)
            print(f"   {name:28s} alpha = {a:+.1f}: K^(alpha) = {Ka:+.6f} (K^(0) = {K0:+.6f}, (1-alpha^2) K^(0) = {(1 - a * a) * K0:+.6f}); max |R^(alpha) - (1-alpha^2) R^(0)| = {sci(maxabs(Ra - (1 - a * a) * R0))}")
            rows.append((name, a, Ka))
        STORE.setdefault("alpha_curv", {})[name] = (K0, [(a, gauss_curv_scalar(riemann(Gf(a), pt), gp)) for a in np.linspace(-1.5, 1.5, 13)])
    # the Jensen-type divergence (6.39) on the cumulant function of the simplex
    t = np.array([0.3, -0.2])
    pt_ = np.exp(t) / (1 + np.exp(t).sum()); g0 = np.diag(pt_) - np.outer(pt_, pt_)
    S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]); C = S - pt_[None, :]
    T0 = np.einsum("x,xi,xj,xk->ijk", np.append(pt_, 1 - pt_.sum()), C, C, C)
    print("Jensen-type divergence (6.39) for psi = log(1 + e^theta1 + e^theta2) at theta = (0.3, -0.2): g, T^D = Gamma^D* - Gamma^D, Gamma^D")
    for al in (0.0, 0.5, -0.5, 0.9, 2.0, -3.0):
        Dj = jensen_div(softmax_psi, al)
        g_, Gam, Gs = geometry_from_D(Dj, t)
        print(f"   alpha = {al:+.1f}: g - psi'' {sci(maxabs(g_ - g0))}; T^D - alpha T {sci(maxabs((Gs - Gam) - al * T0))}; Gamma^D - (1-alpha)/2 T {sci(maxabs(Gam - (1 - al) / 2 * T0))}; D at a distant point {Dj(t, t + np.array([0.7, -0.4])):.4f} (>= 0)")
    th1, th2 = t, np.array([0.1, 0.5])
    p1, p2 = np.exp(th1) / (1 + np.exp(th1).sum()), np.exp(th2) / (1 + np.exp(th2).sum())
    Dpsi = lambda a, b: softmax_psi(a) + float(np.sum(np.append(np.exp(b) / (1 + np.exp(b).sum()), 1 / (1 + np.exp(b).sum())) * np.log(np.append(np.exp(b) / (1 + np.exp(b).sum()), 1 / (1 + np.exp(b).sum()))))) \
        - float(a @ (np.exp(b) / (1 + np.exp(b).sum())))
    print(f"      limits: D^(alpha) at alpha = +0.9999 is {jensen_div(softmax_psi, 0.9999)(th1, th2):.6f}, the Bregman divergence D_psi[theta : theta'] = {Dpsi(th1, th2):.6f}; at alpha = -0.9999 it is {jensen_div(softmax_psi, -0.9999)(th1, th2):.6f} = D_psi[theta' : theta] = {Dpsi(th2, th1):.6f}")
    a3 = alpha_div_gauss(0.9999); a4 = alpha_div_gauss(-0.9999)
    p, q = np.array([1.0, 2.0]), np.array([0.4, 1.3])
    print(f"      the alpha-divergence of the Gaussians: D_(+0.9999)[p:q] = {a3(p, q):.6f} and KL[q:p] = {kl_gauss(q, p):.6f}; D_(-0.9999)[p:q] = {a4(p, q):.6f} and KL[p:q] = {kl_gauss(p, q):.6f}")


# ------------------------------------------------------------------ 5. dually flat manifolds

def check_dually_flat():
    head("5. Dually flat manifolds (§6.5)")
    cases = []
    gfun, Gfun = make_random_pair()
    cases.append(("random pair on R^2", gfun, Gfun, np.array([0.3, -0.4]), 1e-4))
    cases.append(("Gaussians, alpha = 0.5", g_gauss, G_alpha(0.5), np.array([1.0, 2.0]), 1e-5))
    cases.append(("simplex S_2, alpha = 0.5", g_s2, G_s2(0.5), np.array([0.3, 0.25]), 1e-5))
    cases.append(("Gaussians, e", g_gauss, G_e, np.array([1.0, 2.0]), 1e-5))
    for name, gf, Gf, pt, h in cases:
        Gs = dual_fn(gf, Gf)
        gp = gf(pt)
        R = riemann(Gf, pt, h); Rs = riemann(Gs, pt, h)
        Rl = np.einsum("ijkm,ml->ijkl", R, gp); Rsl = np.einsum("ijkm,ml->ijkl", Rs, gp)
        print(f"{name:26s}: max|R| = {maxabs(R):.4f}, max|R*| = {maxabs(Rs):.4f}; R_ijkl + R*_ijlk = {sci(maxabs(Rl + np.transpose(Rsl, (0, 1, 3, 2))))}")
    # the symmetric hypothesis: a flat connection whose dual has torsion
    gfun = lambda x: np.diag([1.0, math.exp(x[0])])
    Gfun = lambda x: np.zeros((2, 2, 2))
    x = np.array([0.2, 0.7]); Gs = dual_fn(gfun, Gfun)(x); Rs = riemann(dual_fn(gfun, Gfun), x, 1e-4)
    print(f"flat nabla (Gamma = 0 in the chart x) with g = diag(1, e^x1): Gamma*_12^2 = {Gs[0, 1, 1]:.4f}, Gamma*_21^2 = {Gs[1, 0, 1]:.4f}: the dual has torsion {maxabs(Gs - np.transpose(Gs, (1, 0, 2))):.4f}, "
          f"yet R* = {sci(maxabs(Rs))}: Theorem 6.5 holds, but no chart makes Gamma* vanish")
    # (6.46)-(6.50) on the Gaussians, with the charts differentiated numerically
    xi = np.array([1.0, 2.0]); g = g_gauss(xi)
    th_of = lambda z: np.array([z[0] / z[1] ** 2, -1 / (2 * z[1] ** 2)])
    et_of = lambda z: np.array([z[0], z[0] ** 2 + z[1] ** 2])
    Jt = np.zeros((2, 2)); Je = np.zeros((2, 2)); h = 1e-6
    for k in range(2):
        e = np.zeros(2); e[k] = h
        Jt[:, k] = (th_of(xi + e) - th_of(xi - e)) / (2 * h); Je[:, k] = (et_of(xi + e) - et_of(xi - e)) / (2 * h)       # d theta/d xi, d eta/d xi  [a, k]
    e_th = np.linalg.inv(Jt)          # columns: e_i = d/d theta^i written in the (mu, sigma) chart
    e_et = np.linalg.inv(Je)          # columns: e^j = d/d eta_j
    G_th = e_th.T @ g @ e_th          # the metric in the theta chart
    d_eta_theta = Je @ e_th           # d eta / d theta
    d_theta_eta = Jt @ e_et           # d theta / d eta
    print(f"(6.46) on the Gaussians at (1, 2): d eta/d theta = g_ij(theta) to {sci(maxabs(d_eta_theta - G_th))} (g = [[{G_th[0, 0]:.0f}, {G_th[0, 1]:.0f}], [{G_th[1, 0]:.0f}, {G_th[1, 1]:.0f}]] = Cov[(x, x^2)]), "
          f"d theta/d eta = g^ij to {sci(maxabs(d_theta_eta - np.linalg.inv(G_th)))}")
    pairing = e_th.T @ g @ e_et
    print(f"(6.48) <d_i, d^j> = delta_i^j: max |pairing - I| = {sci(maxabs(pairing - np.eye(2)))}")
    # (6.50): transport the theta-frame by the e-connection and the eta-frame by the m-connection along a path; the pairing stays delta
    path = lambda t: np.array([1 + 1.5 * t, 2 - 0.8 * t * (1 - 0.3 * t)])
    dpath = lambda t: np.array([1.5, -0.8 * (1 - 0.6 * t)])
    A1 = np.zeros((2, 2)); B1 = np.zeros((2, 2))
    for k in range(2):
        ip, Ae, Bm_, _, _ = transport_pair(G_e, G_m, g_gauss, path, dpath, e_th[:, k], e_et[:, k], steps=800)
        A1[:, k] = Ae; B1[:, k] = Bm_
    xe = path(1.0); ge = g_gauss(xe)
    ths = th_of(xe); Jt2 = np.zeros((2, 2)); Je2 = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = h
        Jt2[:, k] = (th_of(xe + e) - th_of(xe - e)) / (2 * h); Je2[:, k] = (et_of(xe + e) - et_of(xe - e)) / (2 * h)
    print(f"(6.50) the theta-frame carried by the e-connection and the eta-frame carried by the m-connection from (1, 2) to ({xe[0]:.2f}, {xe[1]:.2f}): pairing - I = {sci(maxabs(A1.T @ ge @ B1 - np.eye(2)))}; "
          f"they coincide with the coordinate frames there to {sci(maxabs(A1 - np.linalg.inv(Jt2)))} and {sci(maxabs(B1 - np.linalg.inv(Je2)))}")


# ------------------------------------------------------------------ 6. canonical divergence of a dually flat manifold

Q0 = 0.25 * np.array([[2.0, 1.0], [1.0, 4.0]])


def psi_cv(t):
    """A convex potential that is not a cumulant function: log-sum-exp plus a positive quadratic."""
    return math.log(1 + math.exp(t[0]) + math.exp(t[1])) + 0.5 * t @ Q0 @ t


def grad_psi_cv(t):
    e = np.exp(t); return e / (1 + e.sum()) + Q0 @ t


def hess_psi_cv(t):
    e = np.exp(t); p = e / (1 + e.sum()); return np.diag(p) - np.outer(p, p) + Q0


GLX, GLW = np.polynomial.legendre.leggauss(40)
GLX = (GLX + 1) / 2; GLW = GLW / 2


def recover_psi(gfun, t):
    """(6.62)-(6.66): psi_j(t) = int_0^1 g_ji(s t) t^i ds, psi(t) = psi(0) + int_0^1 psi_j(s t) t^j ds."""
    def psi_j(u):
        return np.array([sum(w * (gfun(s * u) @ u)[j] for s, w in zip(GLX, GLW)) for j in range(2)])
    return sum(w * (psi_j(s * t) @ t) for s, w in zip(GLX, GLW))


def legendre_dual(psi_grad, psi_hess, eta, th0=None, iters=80):
    th = np.zeros(2) if th0 is None else th0.copy()
    for _ in range(iters):
        r = psi_grad(th) - eta
        if np.abs(r).max() < 1e-14:
            break
        th = th - np.linalg.solve(psi_hess(th), r)
    return th


def check_canonical_flat():
    head("6. Canonical divergence of a dually flat manifold (§6.6)")
    t = np.array([0.7, -0.4])
    # Lemma 6.1: from a flat nabla and symmetric nabla*, the metric is a Hessian
    sym = num_dg(hess_psi_cv, t)
    print(f"(6.60) d_i g_jk = d_k g_ji for g = Hessian of a convex potential (the metric of a flat chart with symmetric dual): {sci(maxabs(sym - np.transpose(sym, (2, 1, 0))))}")
    pts = [np.array(p) for p in ([0.7, -0.4], [1.0, 0.5], [-0.8, 0.9], [0.1, 0.1], [-0.5, -0.6], [1.2, -1.1])]
    err = max(abs(recover_psi(hess_psi_cv, p) - (psi_cv(p) - psi_cv(np.zeros(2)) - grad_psi_cv(np.zeros(2)) @ p)) for p in pts)
    print(f"Lemma 6.1 (6.62)-(6.66): integrating g along rays recovers psi up to the affine function psi(0) + grad psi(0).theta at 6 points to {sci(err)}")
    # eta = grad psi is nabla*-affine
    g = hess_psi_cv(t); Gs_low = sym
    d2eta = np.zeros((2, 2, 2)); h = 1e-4
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = h; ej = np.zeros(2); ej[j] = h
            d2eta[:, i, j] = (grad_psi_cv(t + ei + ej) - grad_psi_cv(t + ei - ej) - grad_psi_cv(t - ei + ej) + grad_psi_cv(t - ei - ej)) / (4 * h * h)
    Gaff = np.einsum("aij,ma->ijm", d2eta, np.linalg.inv(g))
    print(f"      eta = grad psi is affine for the dual connection: the connection that makes eta straight has Gamma_ij^m = (d2 eta_a/d theta^i d theta^j)(d theta^m/d eta_a), which equals Gamma*_ij^m = g^mk d_i g_jk to {sci(maxabs(Gaff - raise_(Gs_low, g)))}")
    # (6.69): the affine change and the dual coordinates
    A = np.array([[1.3, 0.4], [-0.6, 0.9]]); b = np.array([0.3, -0.2]); a = np.array([0.5, -0.8]); d = 0.7
    Ai = np.linalg.inv(A)
    psiT = lambda tt: psi_cv(Ai @ (tt - b)) + a @ tt + d
    gradT = lambda tt: Ai.T @ grad_psi_cv(Ai @ (tt - b)) + a
    t1, t2 = np.array([0.7, -0.4]), np.array([-0.2, 0.5])
    tt1, tt2 = A @ t1 + b, A @ t2 + b
    eta2 = grad_psi_cv(t2)
    err_T = maxabs(gradT(tt2) - (Ai.T @ eta2 + a)); err_print = maxabs(gradT(tt2) - (Ai @ eta2 + a))
    print(f"(6.69) theta~ = A theta + b with A = [[1.3, 0.4], [-0.6, 0.9]]: the dual coordinates are grad psi~ = A^(-T) eta + a (error {sci(err_T)}); the printed A^(-1) eta + c is off by {err_print:.4f}")
    As = np.array([[1.3, 0.4], [0.4, 0.9]]); Asi = np.linalg.inv(As)
    psiS = lambda tt: psi_cv(Asi @ (tt - b))
    gradS = lambda tt: Asi.T @ grad_psi_cv(Asi @ (tt - b))
    print(f"      for a symmetric A the two agree: {sci(maxabs(gradS(As @ t2 + b) - Asi @ eta2))}")
    # the canonical divergence does not depend on the chart or on the linear term of psi (6.70)-(6.71)
    th2 = legendre_dual(grad_psi_cv, hess_psi_cv, grad_psi_cv(t2)); phi2 = th2 @ eta2 - psi_cv(th2)
    D0 = psi_cv(t1) + phi2 - t1 @ eta2

    def hessT(tt):
        return Ai.T @ hess_psi_cv(Ai @ (tt - b)) @ Ai

    thT = legendre_dual(gradT, hessT, gradT(tt2)); phiT = thT @ gradT(tt2) - psiT(thT)
    DT = psiT(tt1) + phiT - tt1 @ gradT(tt2)
    print(f"      canonical divergence D[theta1 : theta2] in the old chart {D0:.12f}, in the new chart (with the added linear term and constant) {DT:.12f}")
    # orientation of the Pythagorean identity for D = (6.68)
    P, Q, R = np.array([0.7, -0.4]), np.array([-0.2, 0.5]), np.array([0.3, 0.9])
    Dc = lambda a_, b_: psi_cv(a_) + (lambda tb: tb @ grad_psi_cv(b_) - psi_cv(tb))(legendre_dual(grad_psi_cv, hess_psi_cv, grad_psi_cv(b_), b_)) - a_ @ grad_psi_cv(b_)
    lhs = Dc(P, R) + Dc(R, Q) - Dc(P, Q); rhs = (R - P) @ (grad_psi_cv(R) - grad_psi_cv(Q))
    print(f"      the identity behind every decomposition below, for D of (6.68): D[P:R] + D[R:Q] - D[P:Q] = (theta_R - theta_P).(eta_R - eta_Q): {lhs:.10f} against {rhs:.10f}")


# ------------------------------------------------------------------ 7. canonical divergence of a general dual pair (6.72)-(6.85)
# Geodesics are integrated with RK4 on closed-form accelerations (plain floats, for speed).

def acc_gauss(a):
    """x'' = -Gamma^(alpha)_ij^k x'^i x'^j on the Gaussian manifold, chart (mu, sigma)."""
    c1, c2, c3 = 2 * (1 + a), (1 - a) / 2, 1 + 2 * a

    def f(x1, x2, v1, v2):
        return c1 * v1 * v2 / x2, (-c2 * v1 * v1 + c3 * v2 * v2) / x2
    return f


def quad_gauss(x1, x2, v1, v2):
    return (v1 * v1 + 2 * v2 * v2) / (x2 * x2)


def acc_s2(a):
    """x'' on the simplex S_2 (m-chart): x''^k = (1+alpha)/2 g^{kl} T(v, v, .)_l."""
    c = (1 + a) / 2

    def f(x1, x2, v1, v2):
        p3 = 1 - x1 - x2; w3 = v1 + v2
        t1 = v1 * v1 / (x1 * x1) - w3 * w3 / (p3 * p3); t2 = v2 * v2 / (x2 * x2) - w3 * w3 / (p3 * p3)
        g11 = 1 / x1 + 1 / p3; g22 = 1 / x2 + 1 / p3; g12 = 1 / p3; det = g11 * g22 - g12 * g12
        return c * (g22 * t1 - g12 * t2) / det, c * (-g12 * t1 + g11 * t2) / det
    return f


def quad_s2(x1, x2, v1, v2):
    p3 = 1 - x1 - x2; w3 = v1 + v2
    return v1 * v1 / x1 + v2 * v2 / x2 + w3 * w3 / p3


def integrate(acc, quad, x0, v0, wmode, steps=64):
    """RK4 for x'' = acc(x, x') on t in [0, 1], with the running integral of w(t) g(x)[x', x'] dt (wmode 0: none, 1: w = t, 2: w = 1 - t)."""
    x1, x2 = float(x0[0]), float(x0[1]); v1, v2 = float(v0[0]), float(v0[1]); I = 0.0
    dt = 1.0 / steps; t = 0.0
    wf = (lambda t_: 0.0, lambda t_: t_, lambda t_: 1 - t_)[wmode]
    for _ in range(steps):
        a1, a2 = acc(x1, x2, v1, v2); q1 = quad(x1, x2, v1, v2) * wf(t)
        y1, y2, u1, u2 = x1 + dt / 2 * v1, x2 + dt / 2 * v2, v1 + dt / 2 * a1, v2 + dt / 2 * a2
        b1, b2 = acc(y1, y2, u1, u2); q2 = quad(y1, y2, u1, u2) * wf(t + dt / 2)
        y1, y2, w1, w2 = x1 + dt / 2 * u1, x2 + dt / 2 * u2, v1 + dt / 2 * b1, v2 + dt / 2 * b2
        c1, c2 = acc(y1, y2, w1, w2); q3 = quad(y1, y2, w1, w2) * wf(t + dt / 2)
        y1, y2, z1, z2 = x1 + dt * w1, x2 + dt * w2, v1 + dt * c1, v2 + dt * c2
        d1, d2 = acc(y1, y2, z1, z2); q4 = quad(y1, y2, z1, z2) * wf(t + dt)
        x1 += dt / 6 * (v1 + 2 * u1 + 2 * w1 + z1); x2 += dt / 6 * (v2 + 2 * u2 + 2 * w2 + z2)
        v1, v2 = v1 + dt / 6 * (a1 + 2 * b1 + 2 * c1 + d1), v2 + dt / 6 * (a2 + 2 * b2 + 2 * c2 + d2)
        I += dt / 6 * (q1 + 2 * q2 + 2 * q3 + q4); t += dt
    return (x1, x2), (v1, v2), I


def log_map(acc, quad, p, q, steps=64, tol=1e-13):
    """The initial velocity v at p of the geodesic with x(0) = p, x(1) = q (Newton shooting; the book's exp^-1_p(q), (6.76))."""
    p = np.asarray(p, float); q = np.asarray(q, float); v = q - p
    for _ in range(40):
        x1 = np.array(integrate(acc, quad, p, v, 0, steps)[0]); r = x1 - q
        if np.abs(r).max() < tol:
            break
        J = np.zeros((2, 2)); h = 1e-7
        for j in range(2):
            e = np.zeros(2); e[j] = h
            J[:, j] = (np.array(integrate(acc, quad, p, v + e, 0, steps)[0]) - x1) / h
        v = v - np.linalg.solve(J, r)
    return v


def dtilde(acc, quad, p, q, wmode, steps=64):
    v = log_map(acc, quad, p, q, steps)
    return integrate(acc, quad, p, v, wmode, steps)[2]


def canonical_D(acc_p, acc_d, quad, p, q, steps=64):
    """(6.77), (6.79), (6.80): D~ along the primal geodesic with weight t, D~* along the dual geodesic with weight 1 - t, and their mean."""
    a = dtilde(acc_p, quad, p, q, 1, steps); b = dtilde(acc_d, quad, p, q, 2, steps)
    return a, b, 0.5 * (a + b)


def frDistS2(x, y):
    p, q = pvec(x), pvec(y)
    return 2 * math.acos(min(1.0, float(np.sum(np.sqrt(p * q)))))


def alpha_div_s2(al):
    a = (1 - al) / 2
    return lambda x, y: 4 / (1 - al ** 2) * (1 - float(np.sum(pvec(x) ** a * pvec(y) ** (1 - a))))


def angle_deg(acc, quad, gfun, Dfun, p, q, steps=64, h=2e-3):
    """Angle between the normal n^i = g^ij d'_j D[p:q] of the divergence ball at q and -X, X = exp_q^-1(p) (condition (6.81))."""
    gq = gfun(q)
    X = log_map(acc, quad, q, p, steps, tol=1e-14)
    gr = np.zeros(2)
    for j in range(2):
        e = np.zeros(2); e[j] = h
        gr[j] = (-Dfun(p, q + 2 * e) + 8 * Dfun(p, q + e) - 8 * Dfun(p, q - e) + Dfun(p, q - 2 * e)) / (12 * h)       # five-point stencil
    n = np.linalg.solve(gq, gr); m = -X
    dot = n @ gq @ m; cross = math.sqrt(np.linalg.det(gq)) * (n[0] * m[1] - n[1] * m[0])
    return math.degrees(math.atan2(abs(cross), dot))


def check_canonical_general():
    head("7. Canonical divergence of a general dual pair (§6.7)")
    # the closed-form accelerations agree with the tensor code
    x = np.array([1.1, 2.3]); v = np.array([0.4, -0.7]); worst = 0.0
    for a in (-1.0, 0.0, 0.5):
        worst = max(worst, maxabs(np.array(acc_gauss(a)(*x, *v)) + np.einsum("ijk,i,j->k", G_alpha(a)(x), v, v)))
    xs = np.array([0.3, 0.25]); vs = np.array([0.2, -0.1])
    for a in (-1.0, 0.0, 0.5):
        worst = max(worst, maxabs(np.array(acc_s2(a)(*xs, *vs)) + np.einsum("ijk,i,j->k", G_s2(a)(xs), vs, vs)))
    print(f"(closed-form geodesic accelerations against the tensor code: {sci(worst)})")
    p = np.array([1.0, 2.0]); q = np.array([1.8, 2.7])
    # 7.1 dually flat pair: e-geodesics for nabla, m-geodesics for nabla*
    a_, b_, d_ = canonical_D(acc_gauss(1.0), acc_gauss(-1.0), quad_gauss, p, q)
    print(f"dually flat pair (nabla = e, nabla* = m) from p = (1, 2) to q = (1.8, 2.7): D~ = {a_:.9f}, D~* = {b_:.9f}, D = {d_:.9f}; the Bregman divergence D_psi[theta_p : theta_q] = KL[q:p] = {kl_gauss(q, p):.9f}")
    STORE["canon_flat"] = (a_, b_, kl_gauss(q, p))
    # 7.2 Riemannian case
    a2, b2, d2 = canonical_D(acc_gauss(0.0), acc_gauss(0.0), quad_gauss, p, q)
    hell = alpha_div_gauss(0.0)(p, q)
    print(f"Riemannian pair (T = 0): D~ = {a2:.9f}, D~* = {b2:.9f}, half the squared distance = {0.5 * fr_dist(p, q) ** 2:.9f}; the alpha = 0 divergence 4(1 - BC) = {hell:.9f} is a different divergence with the same local geometry")
    # a cleaner example: two coins p = 0.2, q = 0.8 (Bernoulli: half-line of the sphere)
    bc = math.sqrt(0.2 * 0.8) + math.sqrt(0.8 * 0.2)
    d_fr = 2 * math.acos(bc)
    print(f"      two coins, p = 0.2 and q = 0.8: Bhattacharyya coefficient {bc:.4f}; 4(1 - BC) = {4 * (1 - bc):.6f} but half the squared Fisher-Rao distance (2 arccos(BC))^2/2 = {0.5 * d_fr ** 2:.6f}")
    STORE["coins"] = (4 * (1 - bc), 0.5 * d_fr ** 2)
    # 7.3 a curved, non-metric pair: the alpha = 0.5 structure of the Gaussians
    al = 0.5
    Gp, Gd = acc_gauss(al), acc_gauss(-al)
    a3, b3, d3 = canonical_D(Gp, Gd, quad_gauss, p, q)
    print(f"alpha = 0.5 pair (R != 0, nabla not metric): D~ = {a3:.9f}, D~* = {b3:.9f} (D~* - D~ = {b3 - a3:.1e}), D = {d3:.9f}; D_0.5[p:q] = {alpha_div_gauss(al)(p, q):.9f}, D_-0.5[p:q] = {alpha_div_gauss(-al)(p, q):.9f}, "
          f"half d^2 = {0.5 * fr_dist(p, q) ** 2:.9f}, KL[p:q] = {kl_gauss(p, q):.9f}, KL[q:p] = {kl_gauss(q, p):.9f}")
    STORE["canon_a05"] = (a3, b3, d3, alpha_div_gauss(al)(p, q), alpha_div_gauss(-al)(p, q), 0.5 * fr_dist(p, q) ** 2)
    # the canonical divergence of the dual pair is D with the arguments reversed
    r1 = canonical_D(Gd, Gp, quad_gauss, p, q)[2]; r2 = canonical_D(Gp, Gd, quad_gauss, q, p)[2]
    print(f"      the canonical divergence of the dual structure (nabla*, nabla), D*[p:q] = {r1:.12f}, equals D[q:p] = {r2:.12f} (the weights t and 1 - t exchange under reversing a geodesic)")
    # 7.4 Theorem 6.9, first claim: the Taylor expansion of D(p, p + s d) to third order
    g = g_gauss(p); dg = num_dg(g_gauss, p)
    dirs = [np.array([math.cos(t_), math.sin(t_)]) for t_ in np.linspace(0, math.pi, 7)[:-1]]
    rows = []; vals = {"D~": [], "D~*": []}
    for d in dirs:
        u = {"D~": [], "D~*": []}
        for s in (0.02, 0.04, 0.08):
            qq = p + s * d
            aa = dtilde(Gp, quad_gauss, p, qq, 1, 40); bb = dtilde(Gd, quad_gauss, p, qq, 2, 40)
            quadr = 0.5 * d @ g @ d * s ** 2
            u["D~"].append((aa - quadr) / s ** 3); u["D~*"].append((bb - quadr) / s ** 3)
        for k in u:
            vals[k].append(6 * (8 * u[k][0] - 6 * u[k][1] + u[k][2]) / 3)             # = C(d, d, d)
        rows.append([d[0] ** 3, 3 * d[0] ** 2 * d[1], 3 * d[0] * d[1] ** 2, d[1] ** 3])
    Amat = np.array(rows); fit = {}
    for k in vals:
        sol, *_ = np.linalg.lstsq(Amat, np.array(vals[k]), rcond=None)
        C = np.zeros((2, 2, 2))
        for idx in itertools.product(range(2), repeat=3):
            C[idx] = sol[sum(idx)]
        fit[k] = C
    Gl, Gsl = lower(G_alpha(al)(p), g), lower(G_alpha(-al)(p), g)
    dgs = sym3(dg)
    pred_c = sym3(Gl) + 2 * sym3(Gsl)                  # claimed for both D~ and D~*: Gamma_(ijk) + 2 Gamma*_(ijk)
    pred_t = -sym3(Gl) + 2 * dgs; pred_ts = sym3(Gsl) + dgs           # from the geodesic expansion done by hand (notes, §7)
    print(f"third-order Taylor tensor C of D(p, p + s d) = 1/2 g d d s^2 + 1/6 C(d,d,d) s^3 + ..., fitted from boundary-value geodesics at 6 directions and 3 scales:")
    print(f"      D~ : C_111, C_112, C_122, C_222 = {fit['D~'].reshape(-1)[[0, 1, 3, 7]]},  D~*: {fit['D~*'].reshape(-1)[[0, 1, 3, 7]]}")
    print(f"      prediction Gamma_(ijk) + 2 Gamma*_(ijk) (the same for both): {pred_c.reshape(-1)[[0, 1, 3, 7]]}; max error D~ {sci(maxabs(fit['D~'] - pred_c))}, D~* {sci(maxabs(fit['D~*'] - pred_c))}; "
          f"-Gamma_(ijk) + 2 d_(i g_jk) = Gamma_(ijk) + 2 Gamma*_(ijk) to {sci(maxabs(pred_t - pred_c))}")
    Cm = 0.5 * (fit["D~"] + fit["D~*"]); T0 = T_gauss(p)
    T_rec = 2 * Cm - 3 * dgs
    GamD = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                GamD[i, j, k] = dg[i][j, k] + dg[j][i, k] - Cm[i, j, k]
    print(f"      so the structure of D: T^D = 2C - 3 d_(i g_jk) = {T_rec.reshape(-1)[[0, 1, 3, 7]]} against alpha T = {(al * T0).reshape(-1)[[0, 1, 3, 7]]} (error {sci(maxabs(T_rec - al * T0))}); "
          f"Gamma^D = d_i g_jk + d_j g_ik - C_ijk equals Gamma^(alpha) to {sci(maxabs(GamD - Gl))}; each of D~ and D~* alone also has T^D = alpha T (errors {sci(maxabs(2 * fit['D~'] - 3 * dgs - al * T0))}, {sci(maxabs(2 * fit['D~*'] - 3 * dgs - al * T0))})")
    STORE["taylor_err"] = maxabs(T_rec - al * T0)
    # 7.5 the projection condition (6.81)
    print("condition (6.81): angle (degrees) between the normal of the divergence ball at q and the geodesic direction -exp_q^-1(p):")
    table = []
    qs_g = [np.array([1.8, 2.7]), np.array([0.2, 1.4]), np.array([1.4, 1.5])]
    for al_, lab in ((1.0, "alpha = +1 (Bregman, e-geodesics)"), (-1.0, "alpha = -1 (Bregman, m-geodesics)")):
        Dfun = (lambda a_, b_: kl_gauss(b_, a_)) if al_ == 1.0 else kl_gauss
        angs = [angle_deg(acc_gauss(al_), quad_gauss, g_gauss, Dfun, p, qq) for qq in qs_g]
        table.append(("Gaussians", lab, angs)); print(f"   Gaussians, D_alpha, {lab}: " + ", ".join(f"{a:.1e}" for a in angs))
    for al_ in (0.0, 0.5, -0.5):
        angs = [angle_deg(acc_gauss(al_), quad_gauss, g_gauss, alpha_div_gauss(al_), p, qq) for qq in qs_g]
        table.append(("Gaussians", f"D_alpha, alpha = {al_:+.1f}", angs)); print(f"   Gaussians, D_alpha, alpha = {al_:+.1f}: " + ", ".join(f"{a:.3f}" for a in angs))
    Dcan = lambda a_, b_: canonical_D(Gp, Gd, quad_gauss, a_, b_, 64)[2]
    angs = [angle_deg(Gp, quad_gauss, g_gauss, Dcan, p, qq, 64) for qq in qs_g[::2]]
    table.append(("Gaussians", "canonical D (6.80), alpha = 0.5 pair", angs)); print("   Gaussians, canonical D (6.80), alpha = 0.5 pair, at q = (1.8, 2.7), (1.4, 1.5): " + ", ".join(f"{a:.4f}" for a in angs))
    ps = np.array([0.3, 0.25]); qs_s = [np.array([0.4, 0.3]), np.array([0.2, 0.45]), np.array([0.55, 0.15])]
    for al_ in (0.0, 0.5, -0.5, 0.8):
        angs = [angle_deg(acc_s2(al_), quad_s2, g_s2, alpha_div_s2(al_), ps, qq) for qq in qs_s]
        table.append(("S_2", f"D_alpha, alpha = {al_:+.1f}", angs)); print(f"   simplex S_2, D_alpha, alpha = {al_:+.1f}: " + ", ".join(f"{a:.1e}" for a in angs))
    Dcan2 = lambda a_, b_: canonical_D(acc_s2(0.5), acc_s2(-0.5), quad_s2, a_, b_, 64)[2]
    angs = [angle_deg(acc_s2(0.5), quad_s2, g_s2, Dcan2, ps, qq, 64) for qq in qs_s[:2]]
    table.append(("S_2", "canonical D (6.80), alpha = 0.5 pair", angs)); print("   simplex S_2, canonical D (6.80), alpha = 0.5 pair: " + ", ".join(f"{a:.1e}" for a in angs))
    STORE["projection_angles"] = table
    # how close is the gradient of D~ to g(xi'(1)) (the Riemannian value)? and the limits of Newton shooting
    gqq = g_gauss(q); rels = []
    for stp in (64, 128):
        v_ = log_map(Gp, quad_gauss, p, q, stp, 1e-14); _, v1_, _ = integrate(Gp, quad_gauss, p, v_, 0, stp)
        grad = np.zeros(2)
        for j in range(2):
            e = np.zeros(2); e[j] = 2e-3
            f_ = lambda s_: dtilde(Gp, quad_gauss, p, q + s_ * e / 2e-3 * 1.0, 1, stp)
            grad[j] = (-f_(2e-3 * 2) + 8 * f_(2e-3) - 8 * f_(-2e-3) + f_(-2e-3 * 2)) / (12 * 2e-3)
        rels.append(np.linalg.norm(grad - gqq @ np.array(v1_)) / np.linalg.norm(gqq @ np.array(v1_)))
    print(f"the gradient of D~ with respect to q at the alpha = 0.5 pair, p = (1, 2), q = (1.8, 2.7), differs from g(q) xi'(1) (exact for half the squared distance and for Bregman divergences) by a relative {rels[0]:.2e} at 64 steps and {rels[1]:.2e} at 128 steps")
    miss = []
    for (a_, b_, al_) in (((1.0, 2.0), (3.0, 1.0), 1.0), ((1.0, 2.0), (3.0, 1.0), -1.0), ((3.0, 1.0), (1.0, 2.0), 1.0), ((3.0, 1.0), (1.0, 2.0), -1.0)):
        acc_ = acc_gauss(al_); v_ = log_map(acc_, quad_gauss, np.array(a_), np.array(b_), 64)
        e_ = np.array(integrate(acc_, quad_gauss, np.array(a_), v_, 0, 64)[0]); miss.append(float(np.abs(e_ - np.array(b_)).max()) if np.isfinite(e_).all() else float("nan"))
    print(f"limits of Newton shooting from the straight-line guess between (1, 2) and (3, 1): the e-geodesic (1,2) -> (3,1) is found (miss {miss[0]:.0e}), the m-geodesic (1,2) -> (3,1) and the e-geodesic (3,1) -> (1,2) are not (miss {miss[1]}, {miss[2]}), although both exist in closed form (straight lines in eta, resp. theta)")
    # the generalized Pythagorean relation for D_alpha on the simplex (constant curvature (1 - alpha^2)/4), cited in §6.7
    print("generalized Pythagorean relation on S_2: if the alpha-geodesic from Q to P is orthogonal at Q to the (-alpha)-geodesic from Q to R, then D_alpha[P:R] = D_alpha[P:Q] + D_alpha[Q:R] - (1 - alpha^2)/4 D_alpha[P:Q] D_alpha[Q:R]:")
    Pp = np.array([0.08, 0.15]); Qq = np.array([0.5, 0.3]); gq = g_s2(Qq); pyth_rows = []
    for al_ in (0.0, 0.5, 0.8, -0.5):
        X = log_map(acc_s2(al_), quad_s2, Qq, Pp, 128)
        miss = np.abs(np.array(integrate(acc_s2(al_), quad_s2, Qq, X, 0, 128)[0]) - Pp).max()
        if miss > 1e-9:
            print(f"   alpha = {al_:+.1f}: the geodesic from Q to P was not found (Newton shooting stalls at miss {miss:.1e}); skipped")
            continue
        w = np.array([-(gq @ X)[1], (gq @ X)[0]]); w = w / math.sqrt(w @ gq @ w) * 0.5            # orthogonal to X in the Fisher metric, length 0.5
        R = np.array(integrate(acc_s2(-al_), quad_s2, Qq, w, 0, 400)[0])
        Dl = alpha_div_s2(al_); a_ = Dl(Pp, Qq); b_ = Dl(Qq, R); c_ = Dl(Pp, R); kk = (1 - al_ ** 2) / 4
        print(f"   alpha = {al_:+.1f}: D[P:Q] = {a_:.6f}, D[Q:R] = {b_:.6f}, D[P:R] = {c_:.6f}; plain sum {a_ + b_:.6f}; with the correction {a_ + b_ - kk * a_ * b_:.6f} (difference {sci(abs(c_ - (a_ + b_ - kk * a_ * b_)))}; orthogonality {sci(abs(X @ gq @ w))})")
        pyth_rows.append((al_, a_, b_, c_, a_ + b_ - kk * a_ * b_))
    STORE["pyth_s2"] = pyth_rows
    a6, b6, d6 = canonical_D(acc_s2(0.0), acc_s2(0.0), quad_s2, ps, qs_s[0])
    print(f"the sphere: S_2 with the Fisher metric is an octant of a sphere of radius 2; alpha = 0, p = (0.3, 0.25), q = (0.4, 0.3): D~ = {a6:.9f}, D~* = {b6:.9f}, half the squared distance 2 arccos(BC)^2 = {0.5 * frDistS2(ps, qs_s[0]) ** 2:.9f}, 4(1 - BC) = {alpha_div_s2(0.0)(ps, qs_s[0]):.9f}")
    a5, b5, d5 = canonical_D(acc_s2(0.5), acc_s2(-0.5), quad_s2, ps, qs_s[0])
    print(f"      on S_2, alpha = 0.5, p = (0.3, 0.25), q = (0.4, 0.3): D~ = {a5:.9f}, D~* = {b5:.9f}, D_0.5[p:q] = {alpha_div_s2(0.5)(ps, qs_s[0]):.9f}, D_-0.5[p:q] = {alpha_div_s2(-0.5)(ps, qs_s[0]):.9f}")


# ------------------------------------------------------------------ 8. mixed coordinates and foliations (neurons)

def design(n):
    """The 2^n binary states and the log-linear design matrix: one column per non-empty subset of the neurons (by size, then lexicographically)."""
    X = np.array(list(itertools.product((0, 1), repeat=n)))
    subsets = [s_ for k in range(1, n + 1) for s_ in itertools.combinations(range(n), k)]
    F = np.array([[np.prod(x[list(s_)]) for s_ in subsets] for x in X], float)
    return X, subsets, F


def p_of_theta(F, th):
    a = F @ th; a = a - a.max()
    p = np.exp(a)
    return p / p.sum()


def fisher_F(F, p):
    m = F.T @ p
    return F.T @ np.diag(p) @ F - np.outer(m, m)


def theta_of_p(F, p):
    """The log-linear coefficients (6.105): solve F theta + c = log p exactly."""
    sol = np.linalg.solve(np.hstack([np.ones((len(p), 1)), F]), np.log(p))
    return sol[1:]


def logZ(F, th):
    a = F @ th; m = a.max()
    return m + math.log(float(np.exp(a - m).sum()))


def p_of_mixed(F, first, eta1, th2, th_init=None):
    """The distribution with eta_A = eta1 for A in `first` and theta_B = th2 for the other coordinates: damped Newton on the convex function
    log Z(theta) - theta_A . eta1 of theta_A."""
    n = F.shape[1]; rest = [i for i in range(n) if i not in first]
    th = np.zeros(n) if th_init is None else th_init.copy()
    th[rest] = th2
    for _ in range(200):
        p = p_of_theta(F, th); r = (F.T @ p)[first] - eta1
        if np.abs(r).max() < 1e-15:
            break
        step = np.linalg.solve(fisher_F(F, p)[np.ix_(first, first)], r)
        f0 = logZ(F, th) - th[first] @ eta1; lam = 1.0
        while lam > 1e-12:
            t2 = th.copy(); t2[first] = th[first] - lam * step
            if logZ(F, t2) - t2[first] @ eta1 <= f0 + 1e-14:
                break
            lam /= 2
        th = t2
    return p_of_theta(F, th), th


def mixed_metric(F, p, first):
    """The Fisher metric in the mixed chart (eta_first ; theta_rest): J^T G J with J = d theta / d xi by finite differences."""
    n = F.shape[1]; rest = [i for i in range(n) if i not in first]
    th0 = theta_of_p(F, p); eta0 = F.T @ p; G = fisher_F(F, p)
    xi0 = np.concatenate([eta0[first], th0[rest]]); J = np.zeros((n, n)); h = 1e-5
    for j in range(n):
        d = np.zeros(n); d[j] = h; cols = []
        for s_ in (1, -1):
            xi = xi0 + s_ * d
            cols.append(p_of_mixed(F, first, xi[:len(first)], xi[len(first):], th_init=th0)[1])
        J[:, j] = (cols[0] - cols[1]) / (2 * h)
    return J.T @ G @ J


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def check_mixed():
    head("8. Dual foliations and mixed coordinates (§6.8)")
    X, subsets, F = design(2)
    p = np.array([0.35, 0.15, 0.2, 0.3])                       # p(x1, x2) for 00, 01, 10, 11
    th = theta_of_p(F, p); eta = F.T @ p; G = fisher_F(F, p)
    print(f"two neurons, p(00, 01, 10, 11) = (0.35, 0.15, 0.20, 0.30): eta = ({eta[0]:.3f}, {eta[1]:.3f}, {eta[2]:.3f}), theta = ({th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f}); "
          f"theta12 = log(p11 p00/(p10 p01)) = {math.log(p[3] * p[0] / (p[2] * p[1])):.4f}")
    Gm = mixed_metric(F, p, [0, 1])
    G11 = G[:2, :2]; G12 = G[:2, 2:]; G22 = G[2:, 2:]
    Ginv = np.linalg.inv(G)
    schur = G22 - G12.T @ np.linalg.inv(G11) @ G12
    print(f"metric in the mixed chart (eta1, eta2; theta12): off-diagonal block {sci(maxabs(Gm[:2, 2]))} (orthogonal, (6.89)); eta block [[{Gm[0, 0]:.4f}, {Gm[0, 1]:.4f}], [{Gm[1, 0]:.4f}, {Gm[1, 1]:.4f}]], theta12 entry {Gm[2, 2]:.4f}")
    print(f"      the eta block is the inverse of the 2x2 block of G(theta): [[{np.linalg.inv(G11)[0, 0]:.4f}, {np.linalg.inv(G11)[0, 1]:.4f}], [{np.linalg.inv(G11)[1, 0]:.4f}, {np.linalg.inv(G11)[1, 1]:.4f}]] (error {sci(maxabs(Gm[:2, :2] - np.linalg.inv(G11)))}); "
          f"the theta12 entry is the Schur complement {schur[0, 0]:.4f} (error {sci(abs(Gm[2, 2] - schur[0, 0]))})")
    print(f"      read literally, (6.90) would put the (eta1, eta2) entries of G^-1, [[{Ginv[0, 0]:.4f}, {Ginv[0, 1]:.4f}], [{Ginv[1, 0]:.4f}, {Ginv[1, 1]:.4f}]], and the theta12 entry of G, {G[2, 2]:.4f}, on the diagonal: wrong unless G_12 = 0")
    STORE["mixed_blocks"] = {"true": (np.linalg.inv(G11), schur[0, 0]), "printed": (Ginv[:2, :2], G[2, 2])}
    # the covariance v = eta12 - eta1 eta2 as the third coordinate
    xi = np.array([eta[0], eta[1], eta[2] - eta[0] * eta[1]]); Jv = np.zeros((3, 3)); h = 1e-5

    def p_of_eta(eta_t):
        t_ = np.zeros(3)
        for _ in range(80):
            pp = p_of_theta(F, t_); r = F.T @ pp - eta_t
            if np.abs(r).max() < 1e-15:
                break
            t_ = t_ - np.linalg.solve(fisher_F(F, pp), r)
        return t_

    for j in range(3):
        d = np.zeros(3); d[j] = h; vals = []
        for s_ in (1, -1):
            x_ = xi + s_ * d
            vals.append(p_of_eta(np.array([x_[0], x_[1], x_[2] + x_[0] * x_[1]])))
        Jv[:, j] = (vals[0] - vals[1]) / (2 * h)
    Gv = Jv.T @ G @ Jv
    c1 = Gv[0, 2] / math.sqrt(Gv[0, 0] * Gv[2, 2]); c2 = Gv[1, 2] / math.sqrt(Gv[1, 1] * Gv[2, 2])
    print(f"in the chart (eta1, eta2, v) with v = Cov[x1, x2] = {xi[2]:.4f}: cos(d/dv, d/deta1) = {c1:.4f}, cos(d/dv, d/deta2) = {c2:.4f}: v is not orthogonal to the firing rates (the mixed chart gives {sci(abs(Gm[0, 2]))}, {sci(abs(Gm[1, 2]))})")
    STORE["cov_cos"] = (c1, c2); STORE["mixed_G"] = (Gm, Gv)
    # Theorem 6.12 and the orientation of the decomposition
    q = np.array([0.1, 0.3, 0.4, 0.2]); thq = theta_of_p(F, q); etaq = F.T @ q
    r, _ = p_of_mixed(F, [0, 1], eta[:2], thq[2:])               # marginals of p, interaction of q: the book's R_PQ, "r" of (6.104)
    r2, _ = p_of_mixed(F, [0, 1], etaq[:2], th[2:])              # marginals of q, interaction of p: R_QP
    print(f"(6.104) KL[p:q] = {kl(p, q):.10f}; with r = (marginals of p, interaction of q): KL[p:r] + KL[r:q] = {kl(p, r) + kl(r, q):.10f} (KL[p:r] = {kl(p, r):.6f} from the interaction, KL[r:q] = {kl(r, q):.6f} from the rates)")
    print(f"      with r' = (marginals of q, interaction of p): KL[p:r'] + KL[r':q] = {kl(p, r2) + kl(r2, q):.10f}, not KL[p:q]; instead KL[q:p] = {kl(q, p):.10f} = KL[q:r'] + KL[r':p] = {kl(q, r2) + kl(r2, p):.10f}")
    D = lambda a_, b_: kl(b_, a_)                                # the canonical divergence in the orientation of (6.68): D[P:Q] = KL[Q:P]
    print(f"      in the orientation of (6.68), D[P:Q] = KL[Q:P]: (6.96) as printed, D[P:Q] = D[P:R_PQ] + D[R_PQ:Q] with R_PQ = r, reads {D(p, q):.10f} against {D(p, r) + D(r, q):.10f} (fails); "
          f"with R_QP = r' it reads {D(p, r2) + D(r2, q):.10f} (holds)")
    STORE["decomp_two"] = (kl(p, q), kl(p, r) + kl(r, q), kl(p, r2) + kl(r2, q))
    # when q is independent, KL[p:r] is the mutual information
    qi = np.outer([0.5, 0.5], [0.4, 0.6]).ravel()
    ri, _ = p_of_mixed(F, [0, 1], eta[:2], theta_of_p(F, qi)[2:])
    px1 = np.array([p[0] + p[1], p[2] + p[3]]); px2 = np.array([p[0] + p[2], p[1] + p[3]])
    mi = kl(p, np.outer(px1, px2).ravel())
    print(f"      for an independent q (theta12 = 0), r = p1 x p2 and KL[p:r] = mutual information I(x1:x2) = {mi:.6f} (r - p1 x p2: {sci(maxabs(ri - np.outer(px1, px2).ravel()))}, KL[p:r] = {kl(p, ri):.6f})")
    # random pairs for the figure: sum for r against sum for r'
    rng = np.random.default_rng(7); pts = []
    for _ in range(14):
        a_ = rng.dirichlet(np.ones(4) * 2); b_ = rng.dirichlet(np.ones(4) * 2)
        ta, tb = theta_of_p(F, a_), theta_of_p(F, b_); ea, eb = F.T @ a_, F.T @ b_
        rr, _ = p_of_mixed(F, [0, 1], ea[:2], tb[2:]); rr2, _ = p_of_mixed(F, [0, 1], eb[:2], ta[2:])
        pts.append((kl(a_, b_), kl(a_, rr) + kl(rr, b_), kl(a_, rr2) + kl(rr2, b_)))
    pts = np.array(pts)
    print(f"      14 random pairs: KL[p:r] + KL[r:q] - KL[p:q] has largest absolute value {np.abs(pts[:, 1] - pts[:, 0]).max():.1e}, whereas KL[p:r'] + KL[r':q] - KL[p:q] ranges over [{(pts[:, 2] - pts[:, 0]).min():.4f}, {(pts[:, 2] - pts[:, 0]).max():.4f}]")
    STORE["decomp_scatter"] = pts
    # three neurons
    X3, subs3, F3 = design(3); rng = np.random.default_rng(2); p3 = rng.dirichlet(np.ones(8) * 3)
    th3 = theta_of_p(F3, p3); idx = {s_: k for k, s_ in enumerate(subs3)}
    P = {tuple(x): p3[k] for k, x in enumerate(X3)}
    t123 = math.log(P[(1, 1, 1)] * P[(1, 0, 0)] * P[(0, 1, 0)] * P[(0, 0, 1)] / (P[(1, 1, 0)] * P[(1, 0, 1)] * P[(0, 1, 1)] * P[(0, 0, 0)]))
    t12 = math.log(P[(1, 1, 0)] * P[(0, 0, 0)] / (P[(1, 0, 0)] * P[(0, 1, 0)]))
    print(f"three neurons, a random p: (6.115) theta123 = {t123:.10f} against the log-linear coefficient {th3[idx[(0, 1, 2)]]:.10f}; (6.116) theta12 = {t12:.10f} against {th3[idx[(0, 1)]]:.10f}")
    offs = []; dec = []
    q3 = rng.dirichlet(np.ones(8) * 3); thq3 = theta_of_p(F3, q3); eta3 = F3.T @ p3
    for k in range(1, 7):
        first = list(range(k))
        Gm3 = mixed_metric(F3, p3, first)
        offs.append(maxabs(Gm3[np.ix_(first, [i for i in range(7) if i not in first])]))
        r3, _ = p_of_mixed(F3, first, eta3[first], thq3[[i for i in range(7) if i not in first]])
        dec.append((kl(p3, q3), kl(p3, r3) + kl(r3, q3)))
    print("      Theorem 6.11 for the cuts k = 1..6 of the 7 coordinates (theta ordered neuron, pair, triple; first k replaced by eta): largest off-diagonal entry of the metric: " + ", ".join(sci(o) for o in offs))
    print(f"      Theorem 6.12 for the same cuts, KL[p:q] = {dec[0][0]:.10f}; KL[p:r_k] + KL[r_k:q] = " + ", ".join(f"{d_[1]:.10f}" for d_ in dec))
    k_trip = mixed_metric(F3, p3, [0, 1, 2, 3, 4, 5])
    G3 = fisher_F(F3, p3)
    print(f"      in the plain theta chart the coordinates are not orthogonal: G(theta12, theta123) = {G3[idx[(0, 1)], idx[(0, 1, 2)]]:.4f}; in the mixed chart (eta1, eta2, eta3, eta12, eta13, eta23; theta123) the theta123 direction is orthogonal to all six eta directions ({sci(maxabs(k_trip[:6, 6]))})")


# ------------------------------------------------------------------ 9. system complexity and integrated information

X4, SUBS4, F4 = design(4)                    # variables (x1, x2, y1, y2)
VN = ["x1", "x2", "y1", "y2"]


def col4(vs):
    return SUBS4.index(tuple(sorted(VN.index(v) for v in vs)))


S_COLS = [col4(c) for c in (["x1"], ["x2"], ["y1"], ["y2"], ["x1", "x2"], ["y1", "y2"], ["x1", "y1"], ["x2", "y2"])]            # the split model M_S of (6.118)
SP_COLS = [col4(c) for c in (["x1"], ["x2"], ["y1"], ["y2"], ["x1", "x2"], ["x1", "y1"], ["x2", "y2"])]                           # Ay's M_S': no y1 y2 term
NAMES4 = ["".join(VN[i] for i in s_) for s_ in SUBS4]


def m_project(cols, p):
    """m-projection of p onto the exponential family with the sufficient statistics F4[:, cols]: match the expectations (damped Newton)."""
    Fm = F4[:, cols]; eta = Fm.T @ p; th = np.zeros(len(cols))

    def f(t):
        return logZ(Fm, t) - t @ eta

    for _ in range(200):
        q = p_of_theta(Fm, th); r = Fm.T @ q - eta
        if np.abs(r).max() < 1e-15:
            break
        step = np.linalg.solve(fisher_F(Fm, q), r); lam = 1.0
        while lam > 1e-12:
            t2 = th - lam * step
            if f(t2) <= f(th) + 1e-14:
                break
            lam /= 2
        th = t2
    return p_of_theta(Fm, th)


def ent(p):
    return -float(np.sum(p * np.log(p)))


def tab4(p):
    return p.reshape(2, 2, 2, 2)                  # [x1, x2, y1, y2]


def cond_entropy(p, given, target):
    """H(target | given); given and target are tuples of axes of the 4-way table."""
    t = tab4(p); keep = sorted(set(given) | set(target))
    other = tuple(a for a in range(4) if a not in keep)
    pt = t.sum(axis=other) if other else t
    pg = pt.sum(axis=tuple(keep.index(a) for a in target))
    return ent(pt.ravel()) - ent(pg.ravel())


def cond_mi(p, a, b, c):
    """I(a : b | c) = H(a|c) + H(b|c) - H(a,b|c)."""
    return cond_entropy(p, c, a) + cond_entropy(p, c, b) - cond_entropy(p, c, tuple(sorted(a + b)))


def mutual_info(p):
    t = tab4(p); px = t.sum((2, 3)); py = t.sum((0, 1))
    return kl(p, (px[:, :, None, None] * py[None, None, :, :]).ravel())


# --- Gaussian channel y = A x + eps, x ~ N(0, I), eps ~ N(0, V)

def gauss_channel_cov(A, V):
    return np.block([[np.eye(2), A.T], [A, A @ A.T + V]])


EDGES_S = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (2, 3), (0, 2), (1, 3)]                  # free precision entries: x1x1, x2x2, y1y1, y2y2, x1x2, y1y2, x1y1, x2y2
EDGES_SP = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (0, 2), (1, 3)]


def precision_from(th, edges):
    R = np.zeros((4, 4))
    for t_, (i, j) in zip(th, edges):
        R[i, j] = t_; R[j, i] = t_
    return R


def cov_select(S, edges):
    """Gaussian m-projection: covariance matching S on the diagonal and the edges, precision zero elsewhere (Newton on the precision entries)."""
    th = np.array([1.0 if i == j else 0.0 for (i, j) in edges])

    def resid(t):
        Sh = np.linalg.inv(precision_from(t, edges))
        return np.array([Sh[i, j] - S[i, j] for (i, j) in edges]), Sh

    for _ in range(100):
        r, Sh = resid(th)
        if np.abs(r).max() < 1e-14:
            break
        J = np.zeros((len(th), len(th))); h = 1e-7
        for k in range(len(th)):
            d = np.zeros(len(th)); d[k] = h
            J[:, k] = (resid(th + d)[0] - resid(th - d)[0]) / (2 * h)
        step = np.linalg.solve(J, r); lam = 1.0
        while True:
            new = th - lam * step
            try:
                np.linalg.cholesky(precision_from(new, edges)); break
            except np.linalg.LinAlgError:
                lam /= 2
        th = new
    return Sh


def logdet(M):
    return np.linalg.slogdet(M)[1]


def gaussian_gi(A, V):
    """GI, GI', GI'' (curved model: A^ diagonal, V^ free) and the mutual information of the Gaussian channel with x ~ N(0, I)."""
    S = gauss_channel_cov(A, V)
    Sh = cov_select(S, EDGES_S); Shp = cov_select(S, EDGES_SP)
    GI = 0.5 * (logdet(Sh) - logdet(S)); GIp = 0.5 * (logdet(Shp) - logdet(S))
    I = 0.5 * (logdet(S[:2, :2]) + logdet(S[2:, 2:]) - logdet(S))

    A00, A01, A10, A11 = float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1])
    V00, V01, V11 = float(V[0, 0]), float(V[0, 1]), float(V[1, 1]); ldV = math.log(V00 * V11 - V01 * V01)

    def f2(a0, a1):
        d00, d11 = A00 - a0, A11 - a1
        w00 = V00 + d00 * d00 + A01 * A01; w01 = V01 + d00 * A10 + A01 * d11; w11 = V11 + A10 * A10 + d11 * d11
        return 0.5 * (math.log(w00 * w11 - w01 * w01) - ldV)

    f = lambda a: f2(a[0], a[1])
    L = 3.0 + float(np.abs(A).max()); gx = np.linspace(-L, L, 61)
    grid = sorted((f2(u, w), u, w) for u in gx for w in gx)[:6]                    # a coarse global search, then Newton from the best points
    best = None
    for _, u, w in grid + [(0.0, A00, A11)]:
        a = np.array([u, w])
        for _ in range(60):
            g = np.zeros(2); H = np.zeros((2, 2)); h = 1e-5
            for i in range(2):
                e = np.zeros(2); e[i] = h
                g[i] = (f(a + e) - f(a - e)) / (2 * h)
                for j in range(2):
                    e2 = np.zeros(2); e2[j] = h
                    H[i, j] = (f(a + e + e2) - f(a + e - e2) - f(a - e + e2) + f(a - e - e2)) / (4 * h * h)
            if np.abs(g).max() < 1e-13:
                break
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            lam = 1.0
            while lam > 1e-8 and f(a - lam * step) > f(a):
                lam /= 2
            if f(a - lam * step) > f(a):
                break
            a = a - lam * step
        if best is None or f(a) < f(best):
            best = a.copy()
    a = best
    return {"GI": GI, "GIp": GIp, "GIpp": f(a), "I": I, "Ahat": Sh[2:, :2] @ np.linalg.inv(Sh[:2, :2]), "Ahatp": Shp[2:, :2] @ np.linalg.inv(Shp[:2, :2]), "a_diag": a, "Sh": Sh, "Shp": Shp}


def channel_binary(eps, nu, delta):
    """The combined channel of Example 2 with a uniform input: C1(eps) with probability 1 - delta, C2 with probability delta."""
    bsc = lambda y, x_: (1 - nu) if y == x_ else nu
    p = np.zeros(16)
    for k, (x1, x2, y1, y2) in enumerate(itertools.product((0, 1), repeat=4)):
        c1 = 1.0
        for yi, xi, xj in ((y1, x1, x2), (y2, x2, x1)):
            c1 *= (1 - eps) * bsc(yi, xi) + eps * bsc(yi, xj)
        c2 = 0.5 if y1 == y2 else 0.0
        p[k] = 0.25 * ((1 - delta) * c1 + delta * c2)
    return p


def check_gi():
    head("9. System complexity and integrated information (§6.9)")
    rank = lambda cols: np.linalg.matrix_rank(np.hstack([np.ones((16, 1)), F4[:, cols]])) - 1
    print(f"dimensions of the binary models: full M_F {F4.shape[1]} (15 log-linear coefficients), split M_S {rank(S_COLS)} (the eight terms of (6.118); the book says ten), M_S' {rank(SP_COLS)}")
    # one random distribution in detail
    rng = np.random.default_rng(0); p = rng.dirichlet(np.ones(16) * 2)
    q = m_project(S_COLS, p); qp = m_project(SP_COLS, p)
    GI, GIp = kl(p, q), kl(p, qp)
    HYX = cond_entropy(p, (0, 1), (2, 3)); HYhX = cond_entropy(q, (0, 1), (2, 3)); HYhpX = cond_entropy(qp, (0, 1), (2, 3))
    print(f"a random p(x, y) on 16 states: GI = KL[p:q^] = {GI:.8f}, GI' = KL[p:q^'] = {GIp:.8f}, mutual information I(X:Y) = {mutual_info(p):.8f}")
    print(f"(6.137) GI = H[q^] - H[p] = {ent(q) - ent(p):.8f}, GI' = H[q^'] - H[p] = {ent(qp) - ent(p):.8f}; (6.140)-(6.141) H(Y^|X) - H(Y|X) = {HYhX - HYX:.8f}, {HYhpX - HYX:.8f}")
    sumH = cond_entropy(p, (0,), (2,)) + cond_entropy(p, (1,), (3,))
    I12 = cond_mi(q, (2,), (3,), (0, 1))
    Ix = cond_mi(q, (2,), (1,), (0,)) + cond_mi(q, (3,), (0,), (1,))
    print(f"(6.143) sum_i H(Y_i|X_i) - H(Y|X) = {sumH - HYX:.8f} = GI' ; (6.142) sum_i H(Y_i|X_i) - H(Y|X) - I(Y1^:Y2^|X) = {sumH - HYX - I12:.8f}, which is not GI = {GI:.8f} (difference {sumH - HYX - I12 - GI:.2e})")
    print(f"      the missing terms are I(Y1^:X2|X1) + I(Y2^:X1|X2) = {Ix:.8f}: GI = sum_i H(Y_i|X_i) - H(Y|X) - I(Y1^:Y2^|X) - [I(Y1^:X2|X1) + I(Y2^:X1|X2)] = {sumH - HYX - I12 - Ix:.8f}")
    print(f"(6.144)-(6.146): GI' - GI = {GIp - GI:.8f}; KL[q^:q^'] = {kl(q, qp):.8f}; H(Y^'|X) - H(Y^|X) = {HYhpX - HYhX:.8f} (6.145): so GI' = GI + KL[q^:q^'], GI' >= GI, the reverse of what (6.144), (6.146), (6.147) print")
    t, tq, tqp = tab4(p), tab4(q), tab4(qp)
    cy = lambda t_: (lambda m_: m_ / m_.sum(1, keepdims=True))(t_.sum((1, 3)))
    print(f"(6.130) q^_X = p_X to {sci(maxabs(tq.sum((2, 3)) - t.sum((2, 3))))}, q^_Y = p_Y to {sci(maxabs(tq.sum((0, 1)) - t.sum((0, 1))))}; (6.131) q^(y1|x1) = p(y1|x1) to {sci(maxabs(cy(tq) - cy(t)))}")
    print(f"(6.132) q^'_X = p_X to {sci(maxabs(tqp.sum((2, 3)) - t.sum((2, 3))))} (the book prints p_Y(x)); (6.133) q^'(y1) = p(y1) to {sci(maxabs(tqp.sum((0, 1, 3)) - t.sum((0, 1, 3))))}; "
          f"but q^'_Y differs from p_Y by up to {maxabs(tqp.sum((0, 1)) - t.sum((0, 1))):.4f}; (6.134) {sci(maxabs(cy(tqp) - cy(t)))}")
    # statistics over many random distributions
    rng = np.random.default_rng(11); n = 300; cnt_rev = 0; cnt_142 = 0; worst_corr = 0.0; cnt_post = 0; cnt_postp = 0; worst_142 = 0.0; scat = []
    for _ in range(n):
        p = rng.dirichlet(np.ones(16) * 1.0 + rng.random() * 2)
        q = m_project(S_COLS, p); qp = m_project(SP_COLS, p)
        GI, GIp, I = kl(p, q), kl(p, qp), mutual_info(p)
        cnt_rev += GI > GIp + 1e-12
        sumH = cond_entropy(p, (0,), (2,)) + cond_entropy(p, (1,), (3,)); HYX = cond_entropy(p, (0, 1), (2, 3))
        e142 = abs(sumH - HYX - cond_mi(q, (2,), (3,), (0, 1)) - GI); worst_142 = max(worst_142, e142); cnt_142 += e142 > 1e-9
        Ix = cond_mi(q, (2,), (1,), (0,)) + cond_mi(q, (3,), (0,), (1,))
        worst_corr = max(worst_corr, abs(sumH - HYX - cond_mi(q, (2,), (3,), (0, 1)) - Ix - GI))
        cnt_post += GI <= I + 1e-12; cnt_postp += GIp <= I + 1e-12; scat.append((GI, GIp, I))
    print(f"over {n} random distributions: GI > GI' never happens ({cnt_rev} cases); (6.142) is wrong by more than 1e-9 in {cnt_142} cases (largest error {worst_142:.4f}), the corrected identity holds to {sci(worst_corr)}; GI <= I in {cnt_post} of {n}, GI' <= I in {cnt_postp} of {n}")
    STORE["gi_stats"] = (n, cnt_rev, cnt_142, worst_142, worst_corr, cnt_post, cnt_postp); STORE["gi_scatter"] = np.array(scat)
    # (6.152): X and Y independent but Y1 and Y2 correlated
    py = np.array([[0.4, 0.1], [0.1, 0.4]]); px = np.full((2, 2), 0.25)
    pind = (px[:, :, None, None] * py[None, None, :, :]).ravel()
    q = m_project(S_COLS, pind); qp = m_project(SP_COLS, pind)
    mi_y = kl(py.ravel(), np.outer(py.sum(1), py.sum(0)).ravel())
    print(f"(6.152) X independent of Y, Y1 and Y2 correlated (p_Y = [[0.4, 0.1], [0.1, 0.4]]): I(X:Y) = {mutual_info(pind):.2e}, GI = {kl(pind, q):.2e}, GI' = {kl(pind, qp):.6f} = I(Y1:Y2) = {mi_y:.6f} > 0 = I(X:Y)")
    # the Gaussian channel
    print("Gaussian channel y = A x + eps, x ~ N(0, I) (Example 1):")
    cases = [("A = diag(0.8, 0.6), eps correlated (0.6), no cross-talk", np.diag([0.8, 0.6]), np.array([[1.0, 0.6], [0.6, 1.0]])),
             ("A = [[0.8, 0.5], [0.3, 0.6]], eps correlated (0.4)", np.array([[0.8, 0.5], [0.3, 0.6]]), np.array([[1.0, 0.4], [0.4, 1.0]])),
             ("A = 0 (X independent of Y), eps correlated (0.6)", np.zeros((2, 2)), np.array([[1.0, 0.6], [0.6, 1.0]]))]
    for name, A, V in cases:
        r = gaussian_gi(A, V)
        print(f"   {name}: I = {r['I']:.4f}, GI = {r['GI']:.4f}, GI' = {r['GIp']:.4f}, GI'' = {r['GIpp']:.4f}; A^ of q^ has off-diagonal entries ({r['Ahat'][0, 1]:.4f}, {r['Ahat'][1, 0]:.4f}); "
              f"A^ of q^' off-diagonal ({r['Ahatp'][0, 1]:.1e}, {r['Ahatp'][1, 0]:.1e}); GI' - GI = {r['GIp'] - r['GI']:.6f} = (1/2) log(det S^'/det S^) = {0.5 * (logdet(r['Shp']) - logdet(r['Sh'])):.6f}")
        if name.startswith("A = diag"):
            STORE["gauss_case1"] = r
    # a strongly cross-coupled channel: the curved model's best diagonal gain is far from the true gain
    Ax = np.array([[0.8, 0.9], [-0.9, 0.6]]); Vx = np.array([[1.0, 0.95], [0.95, 1.0]])
    rx = gaussian_gi(Ax, Vx)

    def fx(a_):
        Dm = Ax - np.diag(a_)
        return 0.5 * (math.log(np.linalg.det(Vx + Dm @ Dm.T)) - math.log(np.linalg.det(Vx)))

    a0 = np.array([0.8, 0.6]); Hx = np.zeros((2, 2)); hh = 1e-4
    for i in range(2):
        for j in range(2):
            e = np.zeros(2); e[i] = hh; e2 = np.zeros(2); e2[j] = hh
            Hx[i, j] = (fx(a0 + e + e2) - fx(a0 + e - e2) - fx(a0 - e + e2) + fx(a0 - e - e2)) / (4 * hh * hh)
    print(f"   A = [[0.8, 0.9], [-0.9, 0.6]], noise correlation 0.95: I = {rx['I']:.4f}, GI = {rx['GI']:.4f}, GI' = {rx['GIp']:.4f}, GI'' = {rx['GIpp']:.4f} at the diagonal gain ({rx['a_diag'][0]:.4f}, {rx['a_diag'][1]:.4f}), "
          f"far from the true gain (0.8, 0.6), where the criterion is {fx(a0):.4f} with an indefinite Hessian (eigenvalues {np.linalg.eigvalsh(Hx)[0]:.3f}, {np.linalg.eigvalsh(Hx)[1]:.3f}): the minimisation is not convex, so I search a grid before polishing with Newton")
    # random Gaussian channels: the hierarchy of the three measures
    rng = np.random.default_rng(5); nG = 100; cnt = np.zeros(5, int)
    for _ in range(nG):
        A_ = rng.normal(size=(2, 2)) * 0.8; Bm_ = rng.normal(size=(2, 2)); V_ = Bm_ @ Bm_.T + 0.3 * np.eye(2)
        r_ = gaussian_gi(A_, V_)
        cnt += [r_["GIp"] >= r_["GI"] - 1e-9, r_["GIpp"] <= r_["GIp"] + 1e-9, r_["GI"] <= r_["I"] + 1e-9, r_["GIpp"] <= r_["I"] + 1e-9, r_["GIp"] <= r_["I"] + 1e-9]
    print(f"{nG} random Gaussian channels (A normal, V random positive definite): GI' >= GI in {cnt[0]}, GI'' <= GI' in {cnt[1]} (M_S' lies inside the curved model), GI <= I in {cnt[2]}, GI'' <= I in {cnt[3]} (6.162), GI' <= I in {cnt[4]}")
    STORE["gauss_stats"] = (nG, *cnt)
    # GI versus the noise correlation for a diagonal A
    rhos = np.linspace(0.0, 0.95, 20); curves = []
    for rho in rhos:
        r = gaussian_gi(np.diag([0.8, 0.6]), np.array([[1.0, rho], [rho, 1.0]]))
        curves.append((rho, r["I"], r["GI"], r["GIp"], r["GIpp"]))
    STORE["gi_curves"] = np.array(curves)
    r = gaussian_gi(np.diag([0.8, 0.6]), np.array([[1.0, 0.6], [0.6, 1.0]]))
    dev = max(abs(c[3] + 0.5 * math.log(1 - c[0] ** 2)) for c in curves)
    dev2 = max(abs(gaussian_gi(np.diag(dg_), np.array([[1.0, rr], [rr, 1.0]]))["GIp"] + 0.5 * math.log(1 - rr ** 2)) for dg_, rr in (((-0.5, 1.2), 0.3), ((2.0, 0.1), 0.8), ((0.3, 0.3), 0.5)))
    print(f"      for A = diag(0.8, 0.6), GI' = -(1/2) log(1 - rho^2) at each of the 20 values rho = 0, 0.05, ..., 0.95 to {sci(dev)} (rho = 0.6: {-0.5 * math.log(1 - 0.36):.6f}, computed {r['GIp']:.6f}), and for three other diagonal A to {sci(dev2)}; GI'' is 0 up to {max(abs(c[4]) for c in curves):.1e}")
    # binary channel, Example 2
    print("binary channel of Example 2 (uniform input), p(y|x) = (1 - delta) C1(eps, nu) + delta C2:")
    rows = []
    for (eps, nu, delta) in ((0.0, 0.1, 0.0), (0.0, 0.1, 0.3), (0.2, 0.1, 0.0), (0.2, 0.1, 0.3), (0.0, 0.1, 0.999)):
        pc = channel_binary(eps, nu, delta); th = theta_of_p(F4, pc)
        q = m_project(S_COLS, pc); qp = m_project(SP_COLS, pc)
        cross = {NAMES4[k]: round(float(th[k]), 3) for k in range(15) if NAMES4[k] in ("x1y2", "x2y1", "x1x2y1", "x1x2y2", "x1y1y2", "x2y1y2", "x1x2y1y2") and abs(th[k]) > 1e-9}
        ci = cond_mi(pc, (1,), (2,), (0,)) + cond_mi(pc, (0,), (3,), (1,))
        print(f"   eps = {eps:.1f}, nu = {nu:.1f}, delta = {delta:.3f}: I = {mutual_info(pc):.6f}, GI = {kl(pc, q):.6f}, GI' = {kl(pc, qp):.6f}; I(X2:Y1|X1) + I(X1:Y2|X2) = {ci:.1e}; log-linear terms outside M_S: {cross if cross else 'none'}")
        rows.append((eps, nu, delta, mutual_info(pc), kl(pc, q), kl(pc, qp), ci))
    STORE["binary_channel"] = rows
    print(f"      (ln 2 = {math.log(2):.6f}: for delta -> 1 the channel ignores x and sends perfectly correlated outputs, so I -> 0 and GI' -> ln 2)")


# ------------------------------------------------------------------ 10. input-output analysis

def margins3(A):
    return A.sum(1), A.sum(0), float(A.sum())


def L_tilde(A):
    """(6.174)-(6.175): replace the last row and column by the sums."""
    n = A.shape[0]; L = np.log(A); out = np.zeros_like(L)
    for i in range(n - 1):
        for j in range(n - 1):
            out[i, j] = L[i, j] - L[i, n - 1] - L[n - 1, j] + L[n - 1, n - 1]
        out[i, n - 1] = L[i, n - 1] - L[n - 1, n - 1]
    for j in range(n - 1):
        out[n - 1, j] = L[n - 1, j] - L[n - 1, n - 1]
    out[n - 1, n - 1] = L[n - 1, n - 1]
    return out


def A_tilde(A):
    n = A.shape[0]; r, c, T = margins3(A); out = A.copy()
    for i in range(n - 1):
        out[i, n - 1] = r[i]
    for j in range(n - 1):
        out[n - 1, j] = c[j]
    out[n - 1, n - 1] = T
    return out


def ras(K, r, c, iters=2000):
    """Iterative proportional fitting: alternately scale rows and columns of K to the margins r, c."""
    A_ = K.copy()
    for _ in range(iters):
        A_ = A_ * (r / A_.sum(1))[:, None]
        A_ = A_ * (c / A_.sum(0))[None, :]
    return A_


def D_io(A_, B_):
    """(6.172): D[A:B] = sum { B log(B/A) - B + A }."""
    return float(np.sum(B_ * np.log(B_ / A_) - B_ + A_))


def check_io():
    head("10. Input-output analysis in economics (§6.10)")
    A = np.array([[20.0, 12.0, 6.0], [10.0, 30.0, 14.0], [5.0, 18.0, 25.0]])
    n = 3; r, c, T = margins3(A)
    print(f"a made-up 3-industry table A (rows sell, columns buy): row sums (gross products) {r}, column sums (gross consumption) {c}, total {T:.0f}")
    L = np.log(A); Lt = L_tilde(A); At = A_tilde(A)
    print(f"(6.173) sum A_ij L_ij = {(A * L).sum():.8f} and sum A~_ij L~_ij = {(At * Lt).sum():.8f}; (6.174) interaction block L~_ij (i, j < 3) = {Lt[:2, :2].round(4).tolist()}")
    # D as the Bregman divergence of psi(L) = sum exp L
    B = np.array([[22.0, 9.0, 8.0], [14.0, 26.0, 11.0], [4.0, 20.0, 32.0]])
    bregman = float(np.exp(np.log(A)).sum() + (B * np.log(B) - B).sum() - (np.log(A) * B).sum())
    print(f"(6.171)-(6.172): psi(L) + phi(B) - L.B = {bregman:.8f} and D[A:B] = sum B log(B/A) - B + A = {D_io(A, B):.8f}; for tables with equal totals this is the KL divergence KL[B/total : A/total] times the total")
    # RAS invariance and (6.178)-(6.180)
    mu = np.array([1.1, 0.9, 1.0]); lam = np.array([1.2, 1.0, 0.8]); Abar = A * mu[:, None] * lam[None, :]
    print(f"(6.180) A_ij -> mu_i lambda_j A_ij leaves L~ unchanged: {sci(maxabs(L_tilde(Abar)[:2, :2] - Lt[:2, :2]))}")
    print(f"      but the row sums change by the factors {np.round(Abar.sum(1) / r, 4).tolist()}, not by mu = {mu.tolist()}, and the column sums by {np.round(Abar.sum(0) / c, 4).tolist()}, not lambda = {lam.tolist()}: "
          f"mu_i and lambda_j are not the ratios of the margins in (6.178)-(6.179)")
    S = lambda A_: np.log(A_ / np.outer(A_.sum(1), A_.sum(0)))
    print(f"(6.177) S_ij = log(A_ij/(A_i. A_.j)) is not RAS invariant: it changes by up to {maxabs(S(Abar) - S(A)):.4f} under the same transformation")
    # the mixed chart (A_1., A_2., A_.1, A_.2, A_..; L~_ij): numerical metric
    def table_of_xi(xi):
        tot = xi[4]; rr = np.append(xi[:2], tot - xi[:2].sum()); cc = np.append(xi[2:4], tot - xi[2:4].sum())
        K = np.ones((3, 3)); K[:2, :2] = np.exp(xi[5:].reshape(2, 2))
        return ras(K, rr, cc, 400)

    xi0 = np.concatenate([r[:2], c[:2], [T], Lt[:2, :2].ravel()])
    print(f"      the map (A_1., A_2., A_.1, A_.2, A_..; L~) -> table reproduces A: {sci(maxabs(table_of_xi(xi0) - A))}")
    J = np.zeros((9, 9)); h = 1e-6
    for k in range(9):
        d = np.zeros(9); d[k] = h
        J[:, k] = (np.log(table_of_xi(xi0 + d)).ravel() - np.log(table_of_xi(xi0 - d)).ravel()) / (2 * h)
    Gx = J.T @ np.diag(A.ravel()) @ J
    print(f"      the metric g = diag(A_ij) of the L chart, written in the mixed chart (margins; L~): off-diagonal block between the 5 margin coordinates and the 4 interaction coordinates {sci(maxabs(Gx[:5, 5:]))} (orthogonal)")
    rng = np.random.default_rng(0)
    a = rng.normal(size=3); b = rng.normal(size=3); dL = a[:, None] + b[None, :]
    dA = rng.normal(size=(3, 3)); dA = dA - dA.mean(1, keepdims=True) - dA.mean(0, keepdims=True) + dA.mean()
    print(f"      directly: an e-direction along a RAS orbit dL_ij = a_i + b_j and an m-direction dA with zero row and column sums have pairing sum dL dA = {(dL * dA).sum():.1e}")
    # decomposition and its orientation
    rA, cA, TA = margins3(A)
    Bs = B * (TA / B.sum()); rB, cB, TB = margins3(Bs)
    R_AB = ras(Bs, rA, cA); R_BA = ras(A, rB, cB)
    print(f"decomposition (B rescaled to the total of A): D[A:B] = {D_io(A, Bs):.8f}; with R = (margins of B, interaction of A) = RAS(A -> margins of B): D[A:R] + D[R:B] = {D_io(A, R_BA) + D_io(R_BA, Bs):.8f}; "
          f"with R' = (margins of A, interaction of B): {D_io(A, R_AB) + D_io(R_AB, Bs):.8f}")
    print(f"      and D[B:A] = {D_io(Bs, A):.8f} = D[B:R'] + D[R':A] = {D_io(Bs, R_AB) + D_io(R_AB, A):.8f}: for the divergence (6.172) the intermediate table takes the margins of the SECOND argument and the interaction of the first")
    STORE["io_decomp"] = (D_io(A, Bs), D_io(A, R_BA) + D_io(R_BA, Bs), D_io(A, R_AB) + D_io(R_AB, Bs))
    worst = 1e9; pyth = 0.0
    for _ in range(2000):
        Z = rng.normal(size=(3, 3)) * 0.5; Z = Z - Z.mean(1, keepdims=True) - Z.mean(0, keepdims=True) + Z.mean()
        X_ = R_BA + 2.0 * Z
        if X_.min() <= 0:
            continue
        worst = min(worst, D_io(A, X_) - D_io(A, R_BA)); pyth = max(pyth, abs(D_io(A, X_) - D_io(A, R_BA) - D_io(R_BA, X_)))
    print(f"RAS is the projection: for 2000 random tables X with the margins of B, D[A:X] - D[A:RAS] >= {worst:.4f} > 0, and D[A:X] = D[A:RAS] + D[RAS:X] to {sci(pyth)} (Pythagoras)")
    # symmetrised interaction (6.176): average of (6.174) over the deleted industry k; the terms k = i and k = j are identically zero
    def Lstar(A_):
        L_ = np.log(A_); m = A_.shape[0]; out = np.zeros_like(L_)
        for k in range(m):
            out += L_ - L_[:, [k]] - L_[[k], :] + L_[k, k]
        return out / m
    perm = [2, 0, 1]
    Ls = Lstar(A); dc = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
    print(f"(6.176) averaging (6.174) over the deleted industry k (the terms k = i, j vanish identically): RAS invariant to {sci(maxabs(Lstar(Abar) - Ls))}, equivariant under relabelling the industries to {sci(maxabs(Lstar(A[np.ix_(perm, perm)]) - Ls[np.ix_(perm, perm)]))}; "
          f"it equals the double-centred log table up to the constant mean(diag L) - mean(L) = {np.mean(np.diag(L)) - L.mean():.4f}: {sci(maxabs(Ls - dc - (np.mean(np.diag(L)) - L.mean())))}")
    # the e-geodesic in the interaction part: interpolating two tables
    print("e-geodesic interpolation of the interaction part between two years, margins known: the RAS table with the geometric-mean interaction (L~ halfway) and the margins (A_i., A_.j) of the middle year:")
    A0, A1 = A, B * (TA / B.sum()); Lh = 0.5 * (L_tilde(A0) + L_tilde(A1))
    Kh = np.ones((3, 3)); Kh[:2, :2] = np.exp(Lh[:2, :2])
    mid = ras(Kh, 0.5 * (A0.sum(1) + A1.sum(1)), 0.5 * (A0.sum(0) + A1.sum(0)))
    print(f"      L~ of the interpolated table = halfway between L~ of the two years: {sci(maxabs(L_tilde(mid)[:2, :2] - Lh[:2, :2]))}; its margins are the averages of the two years' margins: {sci(maxabs(mid.sum(1) - 0.5 * (A0.sum(1) + A1.sum(1))))}")


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



def trajectory(acc, x0, v0, T=1.0, n=240):
    """Points of the geodesic x'' = acc(x, x') from (x0, v0), RK4."""
    x1, x2 = float(x0[0]), float(x0[1]); v1, v2 = float(v0[0]), float(v0[1]); dt = T / n; pts = [(x1, x2)]
    for _ in range(n):
        a1, a2 = acc(x1, x2, v1, v2)
        y1, y2, u1, u2 = x1 + dt / 2 * v1, x2 + dt / 2 * v2, v1 + dt / 2 * a1, v2 + dt / 2 * a2
        b1, b2 = acc(y1, y2, u1, u2)
        y1, y2, w1, w2 = x1 + dt / 2 * u1, x2 + dt / 2 * u2, v1 + dt / 2 * b1, v2 + dt / 2 * b2
        c1, c2 = acc(y1, y2, w1, w2)
        y1, y2, z1, z2 = x1 + dt * w1, x2 + dt * w2, v1 + dt * c1, v2 + dt * c2
        d1, d2 = acc(y1, y2, z1, z2)
        x1 += dt / 6 * (v1 + 2 * u1 + 2 * w1 + z1); x2 += dt / 6 * (v2 + 2 * u2 + 2 * w2 + z2)
        v1, v2 = v1 + dt / 6 * (a1 + 2 * b1 + 2 * c1 + d1), v2 + dt / 6 * (a2 + 2 * b2 + 2 * c2 + d2)
        pts.append((x1, x2))
    return np.array(pts)


def tri_xy(p1, p2):
    """Simplex S_2 as an equilateral triangle: outcome 1 bottom left, 2 bottom right, 3 top."""
    p3 = 1 - p1 - p2
    return p2 + 0.5 * p3, 0.8660254 * p3


def fig_transport(path):
    st = STORE["transport"]; b = []
    W, H = 800, 405
    allv = np.concatenate([st["ip"], st["ip_same"], st["ip_lc"]])
    lo, hi = math.floor(allv.min() * 20) / 20, math.ceil(allv.max() * 20) / 20 + 0.06
    P1 = Panel(b, 60, 56, 320, 250, (0.1, 1.9), (1.15, 2.9))
    P1.frame([0.5, 1.0, 1.5], [1.5, 2.0, 2.5], "μ", "σ", "A and B carried round a loop", True)
    P1.line(st["curve"][:, 0], st["curve"][:, 1], "thin s0")
    k = 0.2
    for i, (pt, A, B) in enumerate(zip(st["stations"], st["A"], st["B"])):
        if i == 8:
            continue
        for V, c in ((A, 1), (B, 2)):
            arrow(b, P1.X(pt[0]), P1.Y(pt[1]), P1.X(pt[0] + k * V[0]), P1.Y(pt[1] + k * V[1]), str(c), 1.7, 6)
        b.append(f'<circle class="f0" cx="{P1.X(pt[0]):.1f}" cy="{P1.Y(pt[1]):.1f}" r="2.2"/>')
    legend_col(b, 66, 76, [("s1", "A, by the 0.6-connection"), ("s2", "B, by the (−0.6)-connection")])
    P2 = Panel(b, 450, 56, 320, 250, (0, 1), (lo, hi))
    P2.frame([0, 0.25, 0.5, 0.75, 1], [round(v, 2) for v in np.arange(lo, hi + 1e-9, 0.05)], "t along the loop", "⟨A, B⟩ in the Fisher metric", "The inner product of the pair", True)
    t = np.linspace(0, 1, len(st["ip"]))
    P2.line(t, st["ip"], "ln s1"); P2.line(t, st["ip_same"], "ln s2"); P2.line(t, st["ip_lc"], "dash s3")
    legend_col(b, 456, 76, [("s1", "dual pair: constant"), ("s2", "same connection twice: drifts"), ("dash s3", "Levi-Civita twice: constant")])
    note(b, 60, 372, ["Left: the vectors are not parallel to their starting copies when they return (the pair is curved: R ≠ 0), but their inner product never moves.", f"Right: ⟨A, B⟩ = 0.0700 at both ends, drift 5.5·10⁻¹⁵; with one connection for both vectors it wanders by {np.abs(st['ip_same'] - st['ip_same'][0]).max():.4f} (the connection is not metric)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Dual connections preserve the inner product",
        "Left: in the half-plane of Gaussians with coordinates mu and sigma, a vector A is carried round an ellipse by the 0.6-connection and a vector B by the minus 0.6 connection; the arrows at eight stations show they turn and stretch differently. Right: the Fisher inner product of A and B along the loop is constant for the dual pair and for Levi-Civita twice, and drifts when the same non-metric connection is used for both.", b))


def fig_alpha(path):
    b = []
    W, H = 800, 440
    ox, oy, sx = 70, 40, 300
    top = (ox + 0.5 * sx, oy); bl = (ox, oy + 0.866 * sx); br = (ox + sx, oy + 0.866 * sx)
    b.append(f'<polygon class="thin s0" points="{bl[0]},{bl[1]:.1f} {br[0]},{br[1]:.1f} {top[0]},{top[1]:.1f}"/>')
    b.append(f'<text class="hd" x="{ox}" y="{oy - 14}">α-geodesics from one point with one velocity</text>')
    b.append(f'<text class="lab" x="{bl[0] - 4}" y="{bl[1] + 16}" text-anchor="end">p₁ = 1</text><text class="lab" x="{br[0] + 4}" y="{br[1] + 16}">p₂ = 1</text><text class="lab" x="{top[0]}" y="{top[1] - 6}" text-anchor="middle">p₃ = 1</text>')
    p0 = np.array([0.15, 0.5]); v0 = np.array([0.44, -0.11])
    cols = [(-1.0, "s1", "ln", "α = −1, m-geodesic (straight line)"), (-0.5, "s1", "thin", "α = −0.5"), (0.0, "s3", "ln", "α = 0, Levi-Civita (great circle)"), (0.5, "s2", "thin", "α = +0.5"), (1.0, "s2", "ln", "α = +1, e-geodesic")]
    STORE["alpha_geo_ends"] = []
    for al, c, kind, lab in cols:
        tr = trajectory(acc_s2(al), p0, v0, 1.0, 240)
        pts = []
        for q in tr:
            if not (q[0] > 0.004 and q[1] > 0.004 and 1 - q[0] - q[1] > 0.004):
                break
            xy = tri_xy(q[0], q[1]); pts.append((ox + sx * xy[0], oy + sx * (0.866 - xy[1])))
        STORE["alpha_geo_ends"].append((al, tuple(tr[-1])))
        poly(b, pts, ("ln " if kind == "ln" else "thin ") + c, "stroke-width:2.2" if kind == "ln" else "stroke-width:1.4")
    xy0 = tri_xy(*p0); b.append(f'<circle class="f4 ring" cx="{ox + sx * xy0[0]:.1f}" cy="{oy + sx * (0.866 - xy0[1]):.1f}" r="4.8"/>')
    legend_col(b, ox + 6, oy + 0.866 * sx + 36, [(("ln " if kind == "ln" else "thin ") + c, lab) for al, c, kind, lab in cols], 15)
    P2 = Panel(b, 470, 56, 300, 250, (-1.6, 1.6), (-0.58, 0.72))
    P2.frame([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], [-0.5, -0.25, 0, 0.25, 0.5], "α", "curvature coefficient K⁽ᵅ⁾", "Curvature of the α-connection", True)
    al = np.linspace(-1.6, 1.6, 100)
    P2.line([-1.6, 1.6], [0, 0], "dash s0")
    P2.line(al, (1 - al ** 2) * -0.5, "ln s2"); P2.line(al, (1 - al ** 2) * 0.25, "ln s1")
    for name, cl in (("Gaussians at (1, 2)", "f2"), ("simplex S_2 at (0.3, 0.25)", "f1")):
        K0, pts = STORE["alpha_curv"][name]
        for a_, K in pts:
            P2.dot(a_, K, cl, 3.4)
    legend_col(b, 520, 80, [("s1", "simplex S₂: ¼(1 − α²)"), ("s2", "Gaussians: −½(1 − α²)")])
    note(b, 470, 372, ["Dots: curvature computed from the connection", "coefficients; lines: (1 − α²) times the Levi-Civita value.", "Flat exactly at α = ±1, nowhere else."], "sm", 15)
    b.append(f'<text class="sm" x="{ox}" y="{oy + 0.866 * sx + 36 + 5 * 15 + 12}">All five start at (0.15, 0.50, 0.35) with velocity (0.44, −0.11) in the (p₁, p₂) chart.</text>')
    open(path, "w", encoding="utf-8").write(svg(W, H, "α-geodesics and the curvature of α-geometry",
        "Left: the probability triangle of three outcomes with five geodesics leaving the same point with the same velocity, one for each of alpha equal to minus one, minus one half, zero, one half and one; the m-geodesic is straight and the e-geodesic bends furthest. Right: the curvature coefficient of the alpha-connection against alpha is minus one half times one minus alpha squared for the Gaussians and one quarter times one minus alpha squared for the simplex, with computed values on the curves; both vanish at alpha equal to plus or minus one.", b))


def fig_cubic(path):
    cu = STORE["cubic"]; b = []
    W, H = 800, 405
    P1 = Panel(b, 70, 56, 320, 250, (0, 0.6), (0.09, 0.22))
    P1.frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.12, 0.14, 0.16, 0.18, 0.2], "s", "D(p, p + s d) / s²", "Same metric, different cubic term", True)
    sty = {"KL[p:q]": "ln s1", "D_alpha, alpha = -0.5": "thin s1", "D_alpha, alpha = 0": "ln s3", "half the squared distance": "dash s0", "D_alpha, alpha = +0.5": "thin s2", "KL[q:p]": "ln s2"}
    for name, al, c3, pr, vals in cu["curves"]:
        P1.line(np.concatenate([[0], cu["s"]]), np.concatenate([[cu["quad"]], vals]), sty[name])
    P1.dot(0, cu["quad"], "f4", 4.5)
    P1.text(0.03, 0.2125, "all start at ½ g(d,d) = 0.205", "sm", "start", 0, 0)
    legend_col(b, 76, 215, [("ln s1", "KL[p:q]  (α = −1)"), ("thin s1", "α = −0.5"), ("ln s3", "α = 0 (dashed: ½d²)"), ("thin s2", "α = +0.5"), ("ln s2", "KL[q:p]  (α = +1)")], 16)
    P2 = Panel(b, 470, 56, 300, 250, (-1.25, 1.25), (-0.175, 0.005))
    P2.frame([-1, -0.5, 0, 0.5, 1], [-0.16, -0.12, -0.08, -0.04, 0], "α", "cubic coefficient c₃", "c₃ = ½Γ⁰(d,d,d) + (α/12) T(d,d,d)", True)
    P2.line([-1.2, 1.2], [cu["c0"] - 1.2 * cu["slope_T"], cu["c0"] + 1.2 * cu["slope_T"]], "ln s0")
    for name, al, c3, pr, vals in cu["curves"]:
        P2.dot(al, c3, "f1" if al < 0 else ("f3" if al == 0 else "f2"), 4.2)
    note(b, 70, 372, ["Left: every divergence with the Fisher metric starts at the same value; the slope at s = 0 is the cubic term, and it differs.", "Right: that slope is a straight line in α, determined by T: this is what (6.34) and (6.38) say, and why α-geometry is the whole invariant family."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "One metric, a family of cubic terms",
        "Left: six divergences between nearby Gaussians, divided by s squared, along the ray p plus s times d: all start at one half of the Fisher metric of d, then fan out with different slopes. Right: the slope at zero, the cubic Taylor coefficient, against alpha: it lies on a straight line of slope T(d,d,d) over twelve, with the two Kullback-Leibler divergences at alpha equal to minus one and plus one and the half squared distance at zero.", b))


def fig_canonical(path):
    b = []
    W, H = 800, 420
    a3, b3, d3, da, dma, dh = STORE["canon_a05"]
    labels = [("D̃", a3), ("D̃*", b3), ("D, (6.80)", d3), ("D_{0.5}[p:q]", da), ("D_{−0.5}[p:q]", dma), ("½ d²", dh)]
    P1 = Panel(b, 130, 56, 270, 250, (0, 6), (0.10, 0.185))
    P1.frame([], [0.11, 0.13, 0.15, 0.17], "", "divergence between p = (1, 2) and q = (1.8, 2.7)", "Divergences of the α = 0.5 pair", True)
    for k, (lab, v) in enumerate(labels):
        cl = "fillB" if k < 3 else ("fillO" if k < 5 else "fillG")
        b.append(f'<rect class="{cl}" x="{P1.X(k + 0.15):.1f}" y="{P1.Y(v):.1f}" width="{P1.X(0.7) - P1.X(0):.1f}" height="{P1.y0 + P1.h - P1.Y(v):.1f}"/>')
        b.append(f'<text class="sm" x="{P1.X(k + 0.5):.1f}" y="{P1.Y(v) - 4:.1f}" text-anchor="middle">{v:.4f}</text>')
        b.append(f'<text class="sm" x="{P1.X(k + 0.5):.1f}" y="{P1.y0 + P1.h + 14}" text-anchor="middle">{T(lab)}</text>')
    rows = STORE["projection_angles"]
    groups = [("Bregman, α = ±1, Gaussians", "Gaussians", lambda r: r[1].startswith("alpha = +1") or r[1].startswith("alpha = -1")),
              ("α-divergence on S₂", "S_2", lambda r: r[1].startswith("D_alpha")),
              ("(6.80) on S₂", "S_2", lambda r: r[1].startswith("canonical")),
              ("(6.80) on Gaussians", "Gaussians", lambda r: r[1].startswith("canonical")),
              ("α-divergence on Gaussians", "Gaussians", lambda r: r[1].startswith("D_alpha"))]
    P2 = Panel(b, 520, 56, 250, 250, (-8, 1.2), (0, 5))
    P2.frame([-7, -5, -3, -1, 1], [], "angle in the condition (6.81), degrees", "", "Does the normal point along the geodesic?", False, {-7: "10⁻⁷", -5: "10⁻⁵", -3: "10⁻³", -1: "10⁻¹", 1: "10"})
    for k, (lab, man, sel) in enumerate(groups):
        v = max(max(r[2]) for r in rows if r[0] == man and sel(r)); lv = math.log10(max(v, 1e-8))
        y = k + 0.5
        b.append(f'<rect class="{"fillG" if lv < -4 else "fillO"}" x="{P2.X(-8):.1f}" y="{P2.Y(y) - 9:.1f}" width="{P2.X(lv) - P2.X(-8):.1f}" height="22"/>')
        b.append(f'<text class="sm" x="{P2.X(-8) + 4:.1f}" y="{P2.Y(y) - 14:.1f}">{T(lab)}</text>')
        val = f"{v:.2f}°" if v >= 0.1 else f"{v:.1e}°"
        b.append(f'<text class="v" x="{min(P2.X(lv) + 4, P2.X(1.2) - 50):.1f}" y="{P2.Y(y) + 6:.1f}">{val}</text>')
    note(b, 60, 372, ["Left: D̃ and D̃* agree to seven digits; the α-divergences differ from them and from ½d², though all share the geometry to third order.", "Right: largest angle over the test points. Orthogonal to numerical precision: Bregman and the α-divergence on S₂.", "Visibly not: the α-divergence on the Gaussians."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The canonical divergence of a general dual pair",
        "Left: bars for the values of the primal and dual pieces of the canonical divergence, their mean, the two alpha divergences with alpha plus and minus one half, and half the squared Fisher-Rao distance, between the same two Gaussians for the alpha equals one half structure; the first three are almost equal. Right: on a logarithmic scale, the largest angle in degrees between the normal of a divergence ball and the geodesic direction, for five cases: it is below one millionth of a degree for Bregman divergences and for the alpha divergence on the simplex, about a thousandth to a hundredth for the canonical divergence, and up to about one degree for the alpha divergence on the Gaussians.", b))


def fig_mixed(path):
    b = []
    W, H = 800, 405
    Gm, Gv = STORE["mixed_G"]

    def matrix(x0, y0, M, title, labs, cs=70, ch=32):
        b.append(f'<text class="hd" x="{x0}" y="{y0 - 12}">{T(title)}</text>')
        mx = np.abs(M).max()
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                zero = abs(v) < 1e-6
                cl = "fillO" if (i != j and not zero) else "fillB"
                b.append(f'<rect class="thin s0" x="{x0 + j * cs}" y="{y0 + i * ch}" width="{cs}" height="{ch}"/>')
                if not zero:
                    b.append(f'<rect class="{cl}" style="opacity:{0.12 + 0.3 * min(1, abs(v) / mx * (6 if i != j else 1)):.2f}" x="{x0 + j * cs + 1}" y="{y0 + i * ch + 1}" width="{cs - 2}" height="{ch - 2}"/>')
                b.append(f'<text class="v" x="{x0 + j * cs + cs / 2}" y="{y0 + i * ch + 20}" text-anchor="middle">{("0" if zero else f"{v:.3f}").replace("-", "−")}</text>')
        for k, lb in enumerate(labs):
            b.append(f'<text class="sm" x="{x0 + k * cs + cs / 2}" y="{y0 + 3 * ch + 14}" text-anchor="middle">{lb}</text>')
    matrix(50, 76, Gv, "Fisher metric in (η₁, η₂, v)", ["η₁", "η₂", "v"])
    matrix(50, 214, Gm, "Fisher metric in (η₁, η₂; θ¹²)", ["η₁", "η₂", "θ¹²"])
    pts = STORE["decomp_scatter"]; top = math.ceil(pts.max() * 10) / 10
    P2 = Panel(b, 450, 56, 320, 250, (0, top), (0, top))
    P2.frame([round(v, 1) for v in np.arange(0.2, top + 1e-9, 0.2)], [round(v, 1) for v in np.arange(0.2, top + 1e-9, 0.2)], "KL[p : q]", "sum of the two pieces", "The decomposition (6.104) and its orientation", True)
    P2.line([0, top], [0, top], "dash s0")
    for a_, s1_, s2_ in pts:
        P2.dot(a_, s1_, "f1", 3.8); P2.dot(a_, s2_, "f2", 3.8)
    legend_col(b, 460, 76, [("ln s1", "r = (rates of p, interaction of q)"), ("ln s2", "r′ = (rates of q, interaction of p)")])
    note(b, 50, 372, ["Left: with v = Cov[x₁, x₂] as third coordinate the axes are not orthogonal; with θ¹² they are (blocks: (G₁₁)⁻¹ and a Schur complement).", "Right: 14 random pairs of two-neuron distributions; blue points lie on the diagonal (exact Pythagoras), orange points do not."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Mixed coordinates split rates from interaction",
        "Left: two three by three matrices of the Fisher metric of two binary neurons; in the coordinates eta one, eta two and the covariance v there are non-zero off-diagonal entries, while in the mixed coordinates eta one, eta two and theta one two the off-diagonal entries vanish. Right: for fourteen random pairs of distributions, the sum of the divergences through the intermediate distribution with the rates of p and the interaction of q equals the divergence exactly (blue points on the diagonal), whereas the intermediate with the rates of q and the interaction of p does not (orange points scattered).", b))


def fig_gi(path):
    b = []
    W, H = 800, 405
    cv = STORE["gi_curves"]; sc = STORE["gi_scatter"]
    P1 = Panel(b, 70, 56, 320, 250, (0, 1), (0, 1.0))
    P1.frame([0, 0.2, 0.4, 0.6, 0.8], [0, 0.2, 0.4, 0.6, 0.8, 1.0], "noise correlation ρ", "nats", "A = diag(0.8, 0.6): no cross-talk at all", True)
    P1.line(cv[:, 0], cv[:, 1], "dash s0"); P1.line(cv[:, 0], cv[:, 3], "ln s2"); P1.line(cv[:, 0], cv[:, 2], "ln s1"); P1.line(cv[:, 0], cv[:, 4], "ln s3")
    legend_col(b, 80, 80, [("dash s0", "I(X:Y), mutual information"), ("ln s2", "GI′ (Ay): −½ log(1 − ρ²)"), ("ln s1", "GI (split model M_S)"), ("ln s3", "GI″ (curved model): 0")])
    P2 = Panel(b, 470, 56, 300, 250, (0, 0.3), (0, 0.3))
    P2.frame([0.05, 0.1, 0.15, 0.2, 0.25], [0.05, 0.1, 0.15, 0.2, 0.25], "GI", "GI′", "300 random binary distributions", True)
    P2.line([0, 0.3], [0, 0.3], "dash s0")
    for gi, gip, I in sc:
        if gi < 0.3 and gip < 0.3:
            P2.dot(gi, gip, "f2" if gip > I else "f1", 3.0)
    legend_col(b, 584, 258, [("ln s1", "GI′ ≤ I(X:Y)"), ("ln s2", "GI′ > I(X:Y)")])
    n, cnt_rev, cnt_142, w142, wc, cp, cpp = STORE["gi_stats"]
    note(b, 70, 372, ["Left: GI and GI′ grow with the noise correlation although the channel has no cross-talk; only the curved model GI″ stays 0.", f"Right: every point lies above the diagonal, GI′ ≥ GI (the opposite of (6.146)); {n - cpp} of {n} points have GI′ > I(X:Y)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Geometric integrated information, GI and GI prime",
        "Left: for a Gaussian channel with diagonal gain and correlated noise, the Ay measure GI prime grows as minus one half log of one minus rho squared, the split-model measure GI grows more slowly, the mutual information is larger than both, and the measure from the curved model GI double prime stays at zero. Right: for 300 random binary two by two channels, GI prime is never below GI; orange points are those where GI prime exceeds the mutual information.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_transport(out / "dual-transport.svg")
    fig_alpha(out / "alpha-geometry.svg")
    fig_cubic(out / "cubic-terms.svg")
    fig_canonical(out / "canonical-divergence.svg")
    fig_mixed(out / "mixed-coordinates.svg")
    fig_gi(out / "integrated-information.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_dual_connections()
    check_submanifold()
    check_eguchi()
    check_fdiv()
    check_alpha()
    check_dually_flat()
    check_canonical_flat()
    check_canonical_general()
    check_mixed()
    check_gi()
    check_io()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
