#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 4, checked by hand.

alpha-geometry, Tsallis q-entropy and positive-definite matrices. Every number quoted in the notes comes from here.

Running examples. (i) S_2, the simplex of distributions on three outcomes, with p = (0.6, 0.3, 0.1) and q = (0.1, 0.3, 0.6) and the
chart xi = (p_1, p_2) in which the mixture connection is flat; (ii) R^3_+, three positive numbers, where every alpha-divergence is a
Bregman divergence; (iii) a toy population (1.3, 6.6, 21.1, 6.8) sharing nine parliamentary seats; (iv) a four-outcome distribution
for the Tsallis escort geometry; (v) 2x2 positive-definite matrices P and Q.

Conventions of the book. D_alpha[p:q] = 4/(1-alpha^2) sum { (1-alpha)/2 p + (1+alpha)/2 q - p^((1-alpha)/2) q^((1+alpha)/2) } (3.96);
alpha = -1 is KL[p:q], alpha = +1 its dual; the alpha-representation is h_alpha(p) = p^((1-alpha)/2) (log p for alpha = 1); a divergence
D[xi:xi'] induces g_ij = -d_i d'_j D, Gamma_ij,k = -d_i d_j d'_k D and Gamma*_ij,k = -d'_i d'_j d_k D (6.22-6.24), and the alpha-connection
is Gamma^(alpha) = Gamma^(0) - (alpha/2) T (6.38), flat in the alpha-representation (e-flat for alpha = 1, m-flat for alpha = -1).

Checked here, in the order the notes use them:

  1. invariant and flat divergence (4.1): alpha-divergence as a Bregman divergence (the scale factor of (4.6), the range of convexity of
     (4.3), chi-squared as alpha = 3); the converse of Theorem 4.2 by a rank test of the mixed derivative; why S_n is not flat for
     alpha != +-1: the curvature (1-alpha^2)/4 of the alpha-connection of S_2, the connection induced by D_alpha, and a rank test that
     shows D_alpha is not a Bregman divergence of S_2 in any coordinates (Theorem 4.1);
  2. alpha-geometry (4.2): Pythagorean and projection theorems in R^n_+ (4.3, 4.4); the alpha-geodesic of S_n as an ODE solution (4.27)
     and its parameter; Kurose's theorem (4.29) and the projection theorem of S_n (4.6); apportionment as D_alpha minimisation; the
     alpha-mean (Theorem 4.7); optimality of alpha-integration (Theorem 4.9) and the experts' slip;
  3. Tsallis q-entropy (4.3): q-logarithm, the sign of (4.77) and alpha = 2q-1; S_n as a q-family; the escort geometry: psi_q, eta,
     phi_q, the Bregman divergence (4.95), the conformal factor of (4.117), non-invariance of the escort divergence, the q-max-entropy
     theorem (4.12), the chi-deformed family (4.107-4.114) and Theorem 4.14, the q-Gaussian;
  4. (u,v)-divergence (4.4): definition, (alpha,beta) and alpha special cases, the metric (4.130-4.131), the beta-divergence
     (4.139-4.140), flat but not invariant;
  5. positive-definite matrices (4.5): (4.151) is twice a Gaussian KL; invariance under congruence and under rotation; Lemma (4.158),
     Theorems 4.17 and 4.18, (u,v) for matrices, the (alpha,beta)-log-det family: metric (4.186) and the connections, which depend only on
     alpha - beta;
  6. miscellaneous divergences (4.6): gamma-divergence (invariance, KL limit, the matrix version (4.189), a robust-estimation
     experiment); Zhang's (4.190) induces the alpha*beta-geometry; Furuichi's (4.192) is not a divergence; Jensen-Shannon and
     Burbea-Rao; the (F,G)-connection (4.202) and Theorem 4.20.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Fixed seeds; the whole script takes about eight seconds.

Run:  python3 alpha_geometry.py            (checks)
      python3 alpha_geometry.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
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


def sci(x):
    return f"{x:.1e}"


# ------------------------------------------------------------------ the alpha-divergence and alpha-representations

def D_alpha(m, n, a):
    """alpha-divergence (3.96) between positive vectors (rows of arrays are summed over the last axis)."""
    m = np.asarray(m, float)
    n = np.asarray(n, float)
    if abs(a - 1) < 1e-12:
        return np.sum(m - n + n * np.log(n / m), axis=-1)
    if abs(a + 1) < 1e-12:
        return np.sum(n - m + m * np.log(m / n), axis=-1)
    return 4 / (1 - a * a) * np.sum((1 - a) / 2 * m + (1 + a) / 2 * n - m ** ((1 - a) / 2) * n ** ((1 + a) / 2), axis=-1)


def h_rep(p, a):
    """alpha-representation (4.1), (4.50): p^((1-a)/2), and log p for a = 1."""
    p = np.asarray(p, float)
    return np.log(p) if abs(a - 1) < 1e-12 else p ** ((1 - a) / 2)


def h_inv(t, a):
    t = np.asarray(t, float)
    return np.exp(t) if abs(a - 1) < 1e-12 else t ** (2 / (1 - a))


def h_inv_prime(t, a):
    t = np.asarray(t, float)
    return np.exp(t) if abs(a - 1) < 1e-12 else 2 / (1 - a) * t ** ((1 + a) / (1 - a))


def normalised_path(p, q, a, s):
    """the alpha-geodesic of S_n by normalisation (4.27): returns the point and its tangent d/ds."""
    tp, tq = h_rep(p, a), h_rep(q, a)
    t = (1 - s) * tp + s * tq
    m = h_inv(t, a)
    dm = h_inv_prime(t, a) * (tq - tp)
    tot = m.sum()
    r = m / tot
    return r, (dm - r * dm.sum()) / tot


# ------------------------------------------------------------------ S_2 in the chart xi = (p_1, p_2)

A2 = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])           # a_i(x) = d p_x / d xi_i


def pvec(xi):
    return np.array([1 - xi[0] - xi[1], xi[0], xi[1]])


def s2_geometry(xi, al):
    """Fisher metric g, cubic tensor T and Gamma^(alpha)_{ij,k} = -(1+alpha)/2 T_ijk (mixture chart, where Gamma^(0) = -T/2)."""
    p = pvec(xi)
    g = np.einsum("ix,jx,x->ij", A2, A2, 1 / p)
    T = np.einsum("ix,jx,kx,x->ijk", A2, A2, A2, 1 / p ** 2)
    return p, g, T, -(1 + al) / 2 * T


def s2_curvature(xi, al):
    """R_{ijk}^l of Gamma^(alpha) by (5.66) with exact derivatives, K = R_{1221}/det g and the residual of R_ijkl = K(g_jk g_il - g_ik g_jl)."""
    p, g, T, Gl = s2_geometry(xi, al)
    gi = np.linalg.inv(g)
    dg = -T                                                              # d_k g_ij = -T_ijk
    dT = np.einsum("ix,jx,kx,mx,x->ijkm", A2, A2, A2, A2, -2 / p ** 3)    # d_m T_ijk
    dGl = -(1 + al) / 2 * dT
    dgi = -np.einsum("ln,nph,pm->lmh", gi, dg, gi)
    Gam = np.einsum("lk,ijk->ijl", gi, Gl)
    dGam = np.einsum("lmh,ijm->ijlh", dgi, Gl) + np.einsum("lm,ijmh->ijlh", gi, dGl)
    R = np.zeros((2, 2, 2, 2))
    for i, j, k, l in itertools.product(range(2), repeat=4):
        R[i, j, k, l] = dGam[j, k, l, i] - dGam[i, k, l, j] + sum(Gam[i, m, l] * Gam[j, k, m] - Gam[j, m, l] * Gam[i, k, m] for m in range(2))
    Rl = np.einsum("ijkm,ml->ijkl", R, g)
    K = Rl[0, 1, 1, 0] / np.linalg.det(g)
    pred = K * (np.einsum("jk,il->ijkl", g, g) - np.einsum("ik,jl->ijkl", g, g))
    return K, np.abs(Rl - pred).max()


W5 = {-2: 1 / 12, -1: -8 / 12, 1: 8 / 12, 2: -1 / 12}


def induced_structure(Dfun, x0, dim, h):
    """g_ij = -d_i d'_j D, Gamma_ij,k = -d_i d_j d'_k D, Gamma*_ij,k = -d'_i d'_j d_k D at the diagonal point x0 (book 6.22-6.24), by nested 5-point stencils."""
    x0 = np.asarray(x0, float)

    def d5(vars_):
        tot = 0.0
        for ks in itertools.product(W5.keys(), repeat=len(vars_)):
            x, y = x0.copy(), x0.copy()
            w = 1.0
            for (which, i), k in zip(vars_, ks):
                w *= W5[k]
                if which == 0:
                    x[i] += k * h
                else:
                    y[i] += k * h
            tot += w * Dfun(x, y)
        return tot / h ** len(vars_)

    g = np.zeros((dim, dim))
    G = np.zeros((dim, dim, dim))
    Gs = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            g[i, j] = -d5([(0, i), (1, j)])
            for k in range(dim):
                G[i, j, k] = -d5([(0, i), (0, j), (1, k)])
                Gs[i, j, k] = -d5([(1, i), (1, j), (0, k)])
    return g, G, Gs


# ------------------------------------------------------------------ 1. invariant and flat divergence

def check_bregman():
    head("1. Invariant and flat divergence (section 4.1): alpha-divergence as a Bregman divergence of R^n_+ (Theorem 4.2)")
    rng = np.random.default_rng(1)
    m1 = rng.uniform(0.2, 2, 4)
    m2 = rng.uniform(0.2, 2, 4)
    print("   (4.1)-(4.6) as printed: theta = m^((1-a)/2), psi_a = (1-a)/2 sum m, eta = m^((1+a)/2), D = psi_a(theta_1) + psi_-a(eta_2) - theta_1.eta_2, against D_alpha of (3.96):")
    rows = []
    for a in (-3.0, -0.5, 0.0, 0.5, 1.5, 3.0):
        D = float(D_alpha(m1, m2, a))
        B = (1 - a) / 2 * m1.sum() + (1 + a) / 2 * m2.sum() - float(np.sum(m1 ** ((1 - a) / 2) * m2 ** ((1 + a) / 2)))
        rows.append((a, D, B))
        print(f"      alpha = {a:+.1f}: D_alpha = {D:.6f}, (4.6) = {B:.6f}, ratio {B / D:+.6f}  (1-alpha^2)/4 = {(1 - a * a) / 4:+.6f}")
    print("   so (4.6) is (1-alpha^2)/4 times D_alpha: the constants 2/(1-alpha) and 2/(1+alpha) of (4.137) are needed to get (3.96); for |alpha| > 1 the factor is negative,")
    print("   the printed 'divergence' is <= 0, and psi_alpha of (4.3) is concave, not convex:")
    for a in (-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0):
        f = lambda t, a=a: (1 - a) / 2 * t ** (2 / (1 - a))
        th, h = 0.7, 1e-4
        d2 = (f(th + h) - 2 * f(th) + f(th - h)) / h ** 2
        print(f"      alpha = {a:+.1f}: psi_alpha''(0.7) = {d2:+.4f}", end="")
        print("   convex" if d2 > 0 else "   concave")
    pe = np.array([0.6, 0.3, 0.1])
    qe = np.array([0.1, 0.3, 0.6])
    print("   a plain example, p = (0.6, 0.3, 0.1), q = (0.1, 0.3, 0.6): D_alpha[p:q] = " + ", ".join(f"{a:+.0f}: {float(D_alpha(pe, qe, a)):.5f}" for a in (-3.0, -1.0, 0.0, 1.0, 3.0)) + f"; D_alpha[q:p] = D_-alpha[p:q] (3.40): D_1[q:p] = {float(D_alpha(qe, pe, 1.0)):.5f} = D_-1[p:q]")
    print("   (4.3) says convex 'for alpha > -1': true for -1 < alpha < 1 only. With the constants of (4.137) the Hessian of psi_{u,v} in theta is v'/u' = m^alpha > 0 for every alpha (4.123).")
    # chi^2 as alpha = 3
    mm = rng.uniform(0.3, 3, 5)
    nn = rng.uniform(0.3, 3, 5)
    chi2 = float(np.sum((nn - mm) ** 2 / (2 * mm)))
    print(f"   chi-squared (3.37) is alpha = 3: D_3[m:n] = {float(D_alpha(mm, nn, 3.0)):.12f} = sum (n-m)^2/(2m) = {chi2:.12f}; affine coordinates theta = -1/m, eta = m^2/2 (from (4.137)), psi = m/2 per component")
    # standard f of (4.17)
    for a in (-0.5, 0.5, 2.0):
        f = lambda u, a=a: 4 / (1 - a * a) * ((1 - a) / 2 + (1 + a) / 2 * u - u ** ((1 + a) / 2))
        h = 1e-4
        f1 = (f(1 + h) - f(1 - h)) / (2 * h)
        f2 = (f(1 + h) - 2 * f(1) + f(1 - h)) / h ** 2
        print(f"   (4.17), alpha = {a:+.1f}: f(1) = {f(1.0):.1e}, f'(1) = {f1:.1e}, f''(1) = {f2:.6f}  (a standard convex function, f''(1) = 1)")
    STORE["bregman_rows"] = rows


def check_converse():
    head("1b. The converse in Theorem 4.2: a decomposable Bregman divergence has a mixed derivative of the form -theta'(m) eta'(n), a rank-one kernel")
    ms = np.linspace(0.5, 3, 25)
    ns = np.linspace(0.3, 3.5, 25)

    def fpp_alpha(a):
        return lambda u: 4 / (1 - a * a) * (((1 + a) / 2) * ((1 - a) / 2) * u ** ((a - 3) / 2))

    def fpp_num(f):
        h = 1e-4
        return lambda u: (f(u + h) - 2 * f(u) + f(u - h)) / h ** 2

    fJS = lambda u: 0.5 * (np.log(2 / (1 + u)) + u * np.log(2 * u / (1 + u)))
    fTri = lambda u: (u - 1) ** 2 / (u + 1)
    fs = [("alpha = 0 (Hellinger)", fpp_alpha(0.0)), ("alpha = 3 (chi-squared)", fpp_alpha(3.0)), ("KL, f = -log u", lambda u: 1 / u ** 2),
          ("dual KL, f = u log u", lambda u: 1 / u), ("Jensen-Shannon", fpp_num(fJS)), ("triangular (u-1)^2/(u+1)", fpp_num(fTri))]
    print("   kernel K(m,n) = d_m d_n [m f(n/m)] = -(n/m^2) f''(n/m) on a 25 x 25 grid; singular values s2/s1 and s3/s1 (0 means rank one):")
    out = {}
    for name, fpp in fs:
        K = np.array([[n / m ** 2 * fpp(n / m) for n in ns] for m in ms])
        s = np.linalg.svd(K, compute_uv=False)
        out[name] = (s[1] / s[0], s[2] / s[0])
        print(f"      {name:26s} s2/s1 = {s[1] / s[0]:.2e}   s3/s1 = {s[2] / s[0]:.2e}")
    print("   the alpha-family (incl. both KL's and chi-squared) is rank one to 1e-16; Jensen-Shannon and the triangular discrimination, also f-divergences, are not: not Bregman in any coordinates")
    # (4.10) literally
    fstd = lambda u: 4 / (1 - 0.0 ** 2) * (0.5 + 0.5 * u - np.sqrt(u))           # standard f of (4.17), alpha = 0
    fns = lambda u: -np.sqrt(u)                                                   # f = -u^((1+alpha)/2), alpha = 0, before standardising
    Mst = np.array([[m * fstd(n / m) for n in ns] for m in ms])
    Mns = np.array([[m * fns(n / m) for n in ns] for m in ms])
    s1 = np.linalg.svd(Mst, compute_uv=False)
    s2 = np.linalg.svd(Mns, compute_uv=False)
    print(f"   (4.10) m f(n/m) = k(m) k*(n) holds for the non-standard f = -u^((1+alpha)/2) (kernel -sqrt(mn), singular values s2/s1 = {s2[1] / s2[0]:.1e}) but not for the standard f of (4.17):")
    print(f"   m f(n/m) = 4 (m/2 + n/2 - sqrt(mn)) has s2/s1 = {s1[1] / s1[0]:.2e} (alpha = 0); the terms depending on m alone or on n alone are what the standardisation adds, so the argument has to run on the")
    print("   mixed derivative as above; and (4.13), h = log f', needs f' > 0 while f = -u^((1+alpha)/2) has f' < 0 (log |f'| is what is meant)")
    STORE["rank1"] = out


def check_simplex():
    head("1c. Why S_n is not flat for alpha != +-1: curvature of the alpha-connection of S_2, the connection induced by D_alpha, a rank test")
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    print(f"   the alpha-geodesic (4.22) between p = {p.tolist()} and q = {q.tolist()} computed in R^3_+ and its total mass at t = 1/2:")
    STORE["mass_half"] = {}
    for a in (-1.0, 0.0, 1.0, 3.0):
        m = h_inv(0.5 * h_rep(p, a) + 0.5 * h_rep(q, a), a)
        STORE["mass_half"][a] = float(m.sum())
        print(f"      alpha = {a:+.1f}: total mass {m.sum():.6f}   (only alpha = -1, the mixture, stays on the simplex)")
    print("   in the alpha-representation theta the simplex is sum theta_i^(2/(1-alpha)) = 1: for alpha = 0 the sphere sum theta^2 = 1 (curved), for alpha = -1 a plane (linear), and for alpha = +1 it is")
    print("   linear in the dual coordinates eta = m; (4.20) writes the sum to m where n is meant. A curved hypersurface can still have a flat induced connection (the cylinder of Chapter 5), so I compute the curvature:")
    Ks = []
    for xi in ([0.3, 0.2], [0.1, 0.5], [0.45, 0.45]):
        row = []
        for a in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
            K, err = s2_curvature(xi, a)
            row.append((a, K, err))
        Ks.append(row)
        print(f"      xi = {xi}: " + ", ".join(f"K({a:+.1f}) = {K:+.6f}" for a, K, _ in row) + f"   [largest tensor residual {max(e for _, _, e in row):.1e}]")
    print("   so R^(alpha)_ijkl = K (g_jk g_il - g_ik g_jl) with K = (1-alpha^2)/4 at every point tried: a constant, zero exactly for alpha = +-1, negative for |alpha| > 1; (alpha = 0: the sphere of radius 2, K = 1/4)")
    STORE["K_curve"] = [(a, K) for a, K, _ in Ks[0]]
    # connection induced by D_alpha
    xi = [0.3, 0.25]
    print("   the connections induced by D_alpha through (6.23)-(6.24) at xi = (0.3, 0.25), nested 5-point stencils (step 0.004), against the formulas of the book (6.38); errors relative to max|T|:")
    for a in (-0.5, 0.0, 0.5, 2.0):
        g, G, Gs = induced_structure(lambda x, y, a=a: float(D_alpha(pvec(x), pvec(y), a)), xi, 2, 0.004)
        _, g0, T, Gl = s2_geometry(xi, a)
        _, _, _, Glm = s2_geometry(xi, -a)
        sc = np.abs(T).max()
        print(f"      alpha = {a:+.1f}: |g_D - Fisher| = {np.abs(g - g0).max():.1e}, |Gamma_D - Gamma^(alpha)|/|T| = {np.abs(G - Gl).max() / sc:.1e}, |Gamma*_D - Gamma^(-alpha)|/|T| = {np.abs(Gs - Glm).max() / sc:.1e}, |(Gamma* - Gamma) - alpha T|/|T| = {np.abs(Gs - G - a * T).max() / sc:.1e}")
    print("   (exactly, d_i d_j d'_k D = ((1+alpha)/2) T_ijk by differentiating p^a q^b, so Gamma_D = -((1+alpha)/2) T = Gamma^(0) - (alpha/2) T with Gamma^(0) = -T/2 in this chart)")
    # rank test
    rng = np.random.default_rng(21)
    pts = [rng.dirichlet([3, 3, 3]) for _ in range(14)]

    def mixed(pp, qq, a, Am):
        """d_{xi_i} d_{xi'_j} D_alpha[pp:qq] in the chart of the matrix Am (a_i(x) = d p_x / d xi_i): -sum_x p^(a-1) q^(b-1) a_i(x) a_j(x) (4ab/(1-alpha^2) = 1)."""
        if abs(a + 1) < 1e-12:
            w = 1 / qq
        elif abs(a - 1) < 1e-12:
            w = 1 / pp
        else:
            aa, bb = (1 - a) / 2, (1 + a) / 2
            w = pp ** (aa - 1) * qq ** (bb - 1) * (4 * aa * bb / (1 - a * a))
        return -np.einsum("ix,jx,x->ij", Am, Am, w)

    def stacked(a, plist, Am):
        k, n = len(plist), Am.shape[0]
        M = np.zeros((n * k, n * k))
        for i, pp in enumerate(plist):
            for j, qq in enumerate(plist):
                M[n * i:n * i + n, n * j:n * j + n] = mixed(pp, qq, a, Am)
        return M

    print("   rank test of the Bregman structure: for D = psi(theta(x)) + phi(eta(y)) - theta(x).eta(y) in any coordinates, d_{xi_i} d_{xi'_j} D = -sum_k theta^k_,i(x) eta_k,j(y) has rank <= n = 2")
    print("   as a kernel; stacking it over 14 x 14 points of S_2 and taking singular values:")
    for a in (-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0):
        sv = np.linalg.svd(stacked(a, pts, A2), compute_uv=False)
        print(f"      alpha = {a:+.1f}: s1, s2, s3, s4 = {sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2e}, {sv[3]:.1e}   s3/s1 = {sv[2] / sv[0]:.2e}")
    curve = []
    for a in np.linspace(-2, 2, 81):
        sv = np.linalg.svd(stacked(float(a), pts, A2), compute_uv=False)
        curve.append((float(a), sv[2] / sv[0]))
    STORE["rank_curve"] = curve
    print("   rank 2 exactly for alpha = +-1 and rank 3 for every other alpha: no alpha != +-1 gives a Bregman divergence of S_2 in any coordinates, which is Theorem 4.1 (for n = 2, without decomposability)")
    xs1 = [np.array([1 - t, t]) for t in np.linspace(0.1, 0.9, 14)]
    A1 = np.array([[-1.0, 1.0]])
    print("   the same test on S_1 (one coordinate: rank <= 1 for a Bregman divergence), singular values s2/s1 of the 14 x 14 kernel:")
    print("      " + ", ".join(f"alpha = {a:+.1f}: {np.linalg.svd(stacked(a, xs1, A1), compute_uv=False)[1] / np.linalg.svd(stacked(a, xs1, A1), compute_uv=False)[0]:.1e}" for a in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0)))
    print("   so on S_1 too, only KL and its dual are Bregman: the Remark's 'an invariant Bregman divergence is the KL-divergence for any n', n = 1 included (what is special for n = 1 is the f-divergence characterisation)")


def check_alpha_rn():
    head("2. alpha-geometry in R^n_+ (section 4.2.1): Pythagorean theorem (4.25), projection theorem (4.26)")
    rng = np.random.default_rng(5)
    print("   coordinates with the constants of (4.137): theta = 2/(1-a) m^((1-a)/2), eta = 2/(1+a) m^((1+a)/2) (log m for a = 1 and m for a = -1 are the limits):")

    def theta_eta(m, a):
        if abs(a - 1) < 1e-12:
            return np.log(m), m
        if abs(a + 1) < 1e-12:
            return m, np.log(m)
        return 2 / (1 - a) * m ** ((1 - a) / 2), 2 / (1 + a) * m ** ((1 + a) / 2)

    def m_from_eta(e, a):
        if abs(a - 1) < 1e-12:
            return e
        if abs(a + 1) < 1e-12:
            return np.exp(e)
        return ((1 + a) / 2 * e) ** (2 / (1 + a))

    print("   the alpha-geodesic m -> n (linear in theta) is orthogonal to the -alpha-geodesic n -> k (linear in eta) when (theta_n - theta_m).(eta_n - eta_k) = 0; 400 random triples, 3 components each:")
    for a in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
        worst = 0.0
        for _ in range(400):
            m = rng.uniform(0.3, 3, 3)
            n = rng.uniform(0.3, 3, 3)
            th_m, _ = theta_eta(m, a)
            th_n, et_n = theta_eta(n, a)
            dth = th_n - th_m
            v = rng.normal(size=3)
            v -= dth * (v @ dth) / (dth @ dth)               # eta_k - eta_n orthogonal to the theta-direction
            et_k = et_n + 0.15 * v / np.abs(v).max() * np.abs(et_n).min()
            if np.any((et_k <= 0) & (abs(a) < 1)) or (abs(a + 1) > 1e-12 and np.any(et_k <= 0)):
                continue
            k = m_from_eta(et_k, a)
            if not np.all(np.isfinite(k)) or np.any(k <= 0):
                continue
            lhs = float(D_alpha(m, k, a))
            rhs = float(D_alpha(m, n, a) + D_alpha(n, k, a))
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-12))
        print(f"      alpha = {a:+.1f}: largest relative error |D[m:k] - D[m:n] - D[n:k]|/D[m:k] = {worst:.1e}")
    # a second route to Theorem 4.3: orthogonality in the Riemannian sense, with the metric diag(1/m) (Theorem 3.4) and the tangents of the explicit curves (4.22), (4.24)
    print("   second route (Riemannian orthogonality): g(u, w) = sum u_i w_i / n_i = 0 for the tangent u at n of the alpha-geodesic m -> n of (4.22) and the tangent w at n of the -alpha-geodesic (4.24) that ends at k;")
    print("   m, n random, w a random g-orthogonal direction (scaled to keep k positive), 400 triples per alpha, the largest relative error of (4.25) and the largest |cos| of the angle between u and w:")
    rng2 = np.random.default_rng(55)
    for a in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
        worst, wcos = 0.0, 0.0
        for _ in range(400):
            m = rng2.uniform(0.3, 3, 3)
            n = rng2.uniform(0.3, 3, 3)
            tm, tn = h_rep(m, a), h_rep(n, a)
            u = h_inv_prime(tn, a) * (tn - tm)                         # tangent at n of the alpha-geodesic from m (the end t = 1 of (4.22))
            z = rng2.normal(size=3)
            w = z - u * np.sum(u * z / n) / np.sum(u * u / n)          # g-orthogonal to u
            wcos = max(wcos, abs(np.sum(u * w / n)) / math.sqrt(np.sum(u * u / n) * np.sum(w * w / n)))
            en = h_rep(n, -a)                                          # (4.24): the -alpha-geodesic is a straight line in the -alpha-representation
            den = 1 / n if abs(a + 1) < 1e-12 else (1 + a) / 2 * n ** ((1 + a) / 2 - 1)
            step = den * w
            sc = 0.4 * np.min(en / np.abs(step))
            k = h_inv(en + sc * step, -a)
            lhs = float(D_alpha(m, k, a))
            rhs = float(D_alpha(m, n, a) + D_alpha(n, k, a))
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-12))
        print(f"      alpha = {a:+.1f}: largest relative error = {worst:.1e}; largest |cos| = {wcos:.1e}")
    # projection
    a = 0.5
    m = np.array([1.2, 0.7, 2.1])
    eta0 = theta_eta(np.array([0.9, 1.1, 0.8]), a)[1]
    b = np.array([0.5, -0.2, 0.1])
    c = np.array([-0.1, 0.4, 0.3])

    def k_of(st):
        return m_from_eta(eta0 + st[0] * b + st[1] * c, a)

    D = lambda st: float(D_alpha(m, k_of(st), a))

    def gh(f, x, h=1e-4):
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            g[i] = (f(x + e) - f(x - e)) / (2 * h)
            for j in range(2):
                e2 = np.zeros(2)
                e2[j] = h
                H[i, j] = (f(x + e + e2) - f(x + e - e2) - f(x - e + e2) + f(x - e - e2)) / (4 * h * h)
        return g, H

    def newton(x):
        for _ in range(80):
            g, H = gh(D, x)
            x = x - np.linalg.solve(H, g)
            if np.abs(g).max() < 1e-12:
                break
        return x

    st = newton(np.zeros(2))
    k = k_of(st)
    th_m, _ = theta_eta(m, a)
    th_k, _ = theta_eta(k, a)
    starts = [newton(RNG.normal(size=2) * 0.5) for _ in range(20)]
    print(f"   projection (4.26) for alpha = 0.5 onto the plane eta = eta0 + s b + t c of R^3_+ (a (-alpha)-flat submanifold): minimiser (s,t) = ({st[0]:.6f}, {st[1]:.6f}), D = {D(st):.8f};")
    print(f"      the alpha-geodesic from m to the minimiser is orthogonal to the plane: (theta_k - theta_m).b = {(th_k - th_m) @ b:.1e}, .c = {(th_k - th_m) @ c:.1e}; 20 random Newton starts all reach it: {bool(np.allclose(np.array(starts), st, atol=1e-6))}")


def check_geodesic_sn():
    head("2b. The alpha-geodesic of S_n (4.27): the normalised path against solutions of the geodesic equation of Gamma^(alpha)")
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.1, 0.3, 0.6])

    def Gamma_up(xi, al):
        _, g, T, Gl = s2_geometry(xi, al)
        return np.einsum("lk,ijk->ijl", np.linalg.inv(g), Gl)

    def integrate(xi0, v0, al, Tend, nsteps):
        h = Tend / nsteps

        def f(y):
            G = Gamma_up(y[:2], al)
            return np.concatenate([y[2:], -np.einsum("ijl,i,j->l", G, y[2:], y[2:])])

        y = np.concatenate([xi0, v0])
        out = [y.copy()]
        for _ in range(nsteps):
            k1 = f(y)
            k2 = f(y + h / 2 * k1)
            k3 = f(y + h / 2 * k2)
            k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(y.copy())
        return np.array(out)

    print("   claim of (4.27): the curve p_i^((1-a)/2)(t) = c(t) {(1-t) p_i^.. + t q_i^..} is the alpha-geodesic of S_n. Derivation: lowering the geodesic equation of Gamma^(alpha) gives")
    print("   sum_x p^alpha (d_l L)(d^2 L/d tau^2) = 0 for L = p^((1-a)/2), i.e. the acceleration of L is parallel to L; so the curve lies in the plane through the origin spanned by L(p) and L(q),")
    print("   and its affine parameter satisfies d tau / d s = c(s)^2 up to normalisation, with c(s) the normalising constant of (4.27). Test on p = (0.6, 0.3, 0.1), q = (0.1, 0.3, 0.6): I integrate the geodesic")
    print("   equation (Runge-Kutta, 1500 steps) from p with the initial velocity of the path, to the predicted arrival time, and compare:")
    rows = []
    for al in (-1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
        r0, v0 = normalised_path(p, q, al, 0.0)
        ss = np.linspace(0, 1, 4001)
        if abs(al - 1) < 1e-12:
            cs = np.ones_like(ss)
        else:
            cs = np.array([h_inv((1 - s) * h_rep(p, al) + s * h_rep(q, al), al).sum() ** (-(1 - al) / 2) for s in ss])
        cum = np.concatenate([[0], np.cumsum((cs[1:] ** 2 + cs[:-1] ** 2) / 2 * (ss[1] - ss[0]))])
        tau_of_s = cum / cum[-1]
        Tend = cum[-1] / cs[0] ** 2
        nst = 1500
        tr = integrate(p[1:].copy(), v0[1:], al, Tend, nst)
        end_err = np.abs(pvec(tr[-1][:2]) - q).max()
        if abs(al - 1) < 1e-12:
            plane = "plane test not applicable (alpha = 1: the normalisation is a shift of log p)"
        else:
            dets = [abs(np.linalg.det(np.stack([h_rep(p, al), h_rep(q, al), h_rep(pvec(y[:2]), al)]))) for y in tr[::150]]
            plane = f"max |det[L(p), L(q), L(r)]| along the solution = {max(dets):.1e}"
        fr = []
        for s in (0.25, 0.5, 0.75):
            target, _ = normalised_path(p, q, al, s)
            d = np.array([np.abs(pvec(y[:2]) - target).max() for y in tr])
            fr.append((s, int(d.argmin()) / nst, tau_of_s[int(s * 4000)]))
        dev = np.abs(tau_of_s - ss).max()
        rows.append((al, fr, dev))
        print(f"      alpha = {al:+.1f}: reaches q to {end_err:.1e}; {plane};")
        print("         fraction of the affine time at s = 1/4, 1/2, 3/4: ODE " + ", ".join(f"{a:.4f}" for _, a, _ in fr) + " ; predicted (integral of c^2) " + ", ".join(f"{b:.4f}" for _, _, b in fr) + f" ; max |tau(s) - s| = {dev:.4f}")
    STORE["tau_rows"] = rows
    print("   so the normalised curve (4.27) is the geodesic of Gamma^(alpha) as a set (all that Theorems 4.5 and 4.6 use), and its parameter t is the affine parameter only for alpha = +-1;")
    print("   for the others the two differ by up to the max |tau(s) - s| printed above (the book does not say so)")


def check_kurose():
    head("2c. Kurose's Pythagorean theorem (4.29) and the projection theorem (4.6) in S_2")
    rng = np.random.default_rng(3)

    def eta_path(q, w, a, s):
        """-alpha path from q with initial tangent w (sum w = 0), normalised."""
        if abs(a + 1) < 1e-12:
            dEta, e0 = 1 / q, np.log(q)
        elif abs(a - 1) < 1e-12:
            dEta, e0 = np.ones(3), q.copy()
        else:
            dEta, e0 = (1 + a) / 2 * q ** ((a - 1) / 2), q ** ((1 + a) / 2)
        e = e0 + s * (w * dEta)
        m = np.exp(e) if abs(a + 1) < 1e-12 else (e if abs(a - 1) < 1e-12 else e ** (2 / (1 + a)))
        return m / m.sum()

    print("   200 random triples p, q (Dirichlet) and r on the -alpha-geodesic from q whose tangent is Fisher-orthogonal to the alpha-geodesic p -> q at q (tangents computed exactly):")
    print("      alpha     max |D[p:r] - (4.29)|    max |D[p:r] - D[p:q] - D[q:r]|   max |correction|/D[p:r]   max plain error/D[p:r]")
    rows = []
    for a in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        err, plain, relc, relp = 0.0, 0.0, 0.0, 0.0
        pts = []
        cnt = 0
        while cnt < 200:
            p = rng.dirichlet([2, 2, 2])
            q = rng.dirichlet([2, 2, 2])
            _, u = normalised_path(p, q, a, 1.0)
            w = np.cross(np.ones(3), u / q)
            w = w / np.abs(w).max() * 0.3 * q.min()
            r = eta_path(q, w, a, rng.uniform(0.3, 1.0))
            if not np.all(np.isfinite(r)) or np.any(r <= 0):
                continue
            Dpr, Dpq, Dqr = (float(D_alpha(x, y, a)) for x, y in ((p, r), (p, q), (q, r)))
            if not np.isfinite(Dpr):
                continue
            form = Dpq + Dqr - (1 - a * a) / 4 * Dpq * Dqr
            err = max(err, abs(Dpr - form))
            plain = max(plain, abs(Dpr - Dpq - Dqr))
            relc = max(relc, abs((1 - a * a) / 4 * Dpq * Dqr) / Dpr)
            relp = max(relp, abs(Dpr - Dpq - Dqr) / Dpr)
            pts.append((Dpr, Dpq + Dqr, form, Dpq, Dqr))
            cnt += 1
        rows.append((a, pts))
        print(f"      {a:+.1f}      {err:.2e}                {plain:.2e}                  {relc:.4f}                  {relp:.4f}")
    STORE["kurose"] = rows
    print("   (4.29) holds to rounding for every alpha tried, with the correction (1-alpha^2)/4 D[p:q] D[q:r] (negative for |alpha| > 1) and nothing else; without it the identity fails")
    print("   by several percent of D[p:r] for alpha != +-1 (last column) and is exact for alpha = +-1")
    # one worked triple (the default state of the interactive page): p, q fixed, r at s = 0.6 along the orthogonal -alpha-geodesic
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    print("   one worked triple, p = (0.6, 0.3, 0.1), q = (0.1, 0.3, 0.6), r at s = 0.6 on the orthogonal -alpha-geodesic from q (tangent scaled to 0.9 min q):")
    print("      alpha     D[p:q]     D[q:r]     D[p:r]    D[p:q]+D[q:r]   (1-a^2)/4 D[p:q] D[q:r]   (4.29)")
    ex = []
    for a in (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0):
        _, u = normalised_path(p, q, a, 1.0)
        w = np.cross(np.ones(3), u / q)
        w = w / np.abs(w).max() * 0.9 * q.min()
        r = eta_path(q, w, a, 0.6)
        Dpq, Dqr, Dpr = (float(D_alpha(x, y, a)) for x, y in ((p, q), (q, r), (p, r)))
        ex.append((a, Dpq, Dqr, Dpr))
        print(f"      {a:+.1f}    {Dpq:.6f}   {Dqr:.6f}   {Dpr:.6f}   {Dpq + Dqr:.6f}        {(1 - a * a) / 4 * Dpq * Dqr:+.6f}               {Dpq + Dqr - (1 - a * a) / 4 * Dpq * Dqr:.6f}")
    STORE["kurose_example"] = ex
    # alpha = 0 is the sphere
    p = rng.dirichlet([2, 2, 2])
    q = rng.dirichlet([2, 2, 2])
    cosphi = float(np.sum(np.sqrt(p * q)))
    print(f"   alpha = 0: D_0[p:q] = {float(D_alpha(p, q, 0.0)):.10f} = 4 (1 - cos phi) = {4 * (1 - cosphi):.10f} with phi the angle between 2 sqrt p and 2 sqrt q on the sphere of radius 2; (4.29) with factor 1/4 is the spherical law cos c = cos a cos b")
    # projection theorem 4.6 on a curve in S_2
    a = 0.5
    p = np.array([0.5, 0.3, 0.2])
    X0 = np.array([0.25, 0.35, 0.4])
    Xd = np.array([0.4, -0.3, -0.1])                          # a straight segment in the simplex (an m-geodesic) as the submanifold M

    def Mpt(t):
        return X0 + t * Xd

    f = lambda t: float(D_alpha(p, Mpt(t), a))
    ts = np.linspace(-0.3, 0.6, 18001)
    vals = np.array([f(t) for t in ts])
    t0 = ts[vals.argmin()]
    for _ in range(60):                                      # polish by Newton on the derivative
        h = 1e-5
        g1 = (f(t0 + h) - f(t0 - h)) / (2 * h)
        g2 = (f(t0 + h) - 2 * f(t0) + f(t0 - h)) / h ** 2
        t0 -= g1 / g2
    qh = Mpt(t0)
    _, u = normalised_path(p, qh, a, 1.0)
    cosine = lambda uu, ww, qq: float(np.sum(uu * ww / qq) / math.sqrt(np.sum(uu * uu / qq) * np.sum(ww * ww / qq)))
    print(f"   Theorem 4.6: minimiser of D_0.5[p:q] over the segment M: t = {t0:.6f}; the tangent of M at the minimiser has Fisher cosine {cosine(u, Xd, qh):.1e} with the tangent of the alpha-geodesic from p, i.e. it is the alpha-projection")
    # a general divergence is not like that (closing Remarks of the chapter): Jensen-Shannon
    fJ = lambda t: JS_of(p, Mpt(t))
    t1 = ts[np.array([fJ(t) for t in ts]).argmin()]
    for _ in range(60):
        h = 1e-5
        g1 = (fJ(t1 + h) - fJ(t1 - h)) / (2 * h)
        g2 = (fJ(t1 + h) - 2 * fJ(t1) + fJ(t1 - h)) / h ** 2
        t1 -= g1 / g2
    qj = Mpt(t1)
    _, uj = normalised_path(p, qj, 0.0, 1.0)
    print(f"   for the Jensen-Shannon divergence (geometry alpha = 0, self-dual) the minimiser over the same segment is t = {t1:.6f} and the cosine between the tangent of M and the Levi-Civita (alpha = 0) geodesic from p is {cosine(uj, Xd, qj):.4f}: not orthogonal,")
    print("   so for a general divergence the minimiser is not the geodesic projection, while for D_alpha it is (the Remark at the end of the chapter)")


def check_apportionment():
    head("2d. Apportionment (4.2.4): which divisor method minimises D_alpha[p:q] over integer seat numbers q_i = n_i/n")
    a_list = (-3.0, -1.0, 0.0, 1.0, 3.0)

    def log_d(k, a):
        """log of the divisor d_alpha(k) = |(k+1)^b - k^b|^(-1/a), a = (1-alpha)/2, b = (1+alpha)/2 (identric/logarithmic means at alpha = 1, -1)."""
        k = np.asarray(k, float)
        if abs(a - 1) < 1e-12:
            return (k + 1) * np.log(k + 1) - np.where(k > 0, k * np.log(np.maximum(k, 1e-300)), 0.0)
        if abs(a + 1) < 1e-12:
            return np.where(k > 0, -np.log(np.log1p(1 / np.maximum(k, 1e-300))), -np.inf)
        aa, bb = (1 - a) / 2, (1 + a) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            if bb > 0:
                core = bb * np.log(k + 1) + np.log1p(-(k / (k + 1)) ** bb)
            else:                                             # b < 0: k^b - (k+1)^b > 0, infinite at k = 0
                core = np.where(k > 0, bb * np.log(np.maximum(k, 1e-300)) + np.log1p(-(k / (k + 1)) ** (-bb)), np.inf)
        return -core / aa

    def greedy(pop, n, a):
        k = np.zeros(len(pop), int)
        for _ in range(n):
            j = int(np.argmax(np.log(pop) - log_d(k, a)))
            k[j] += 1
        return tuple(int(v) for v in k)

    def divisor_method(pop_, n_, d):
        k = np.zeros(len(pop_), int)
        for _ in range(n_):
            k[int(np.argmax(pop_ / d(k)))] += 1
        return tuple(int(v) for v in k)

    def compositions(s, n):
        out = []
        for comp in itertools.product(range(n + 1), repeat=s - 1):
            last = n - sum(comp)
            if last >= 0:
                out.append(comp + (last,))
        return np.array(out, float)

    def brute(pop, n, a, C):
        p = pop / pop.sum()
        q = C / n
        zero = q == 0
        qs = np.where(zero, 1.0, q)
        with np.errstate(all="ignore"):
            if abs(a - 1) < 1e-12:
                terms = p - q + np.where(zero, 0.0, q * np.log(qs / p))
            elif abs(a + 1) < 1e-12:
                terms = np.where(zero, np.inf, q - p + p * np.log(p / qs))
            else:
                qb = np.where(zero, 0.0, qs ** ((1 + a) / 2))
                terms = 4 / (1 - a * a) * ((1 - a) / 2 * p + (1 + a) / 2 * q - p ** ((1 - a) / 2) * qb)
                if a < -1:
                    terms = np.where(zero, np.inf, terms)
            tot = terms.sum(axis=1)
        best = tot.min()
        idx = np.flatnonzero(tot <= best + 1e-13 * max(1.0, abs(best)))
        return [tuple(int(v) for v in C[i]) for i in idx]

    print("   the greedy rule 'next seat to argmax p_i / d(n_i)' is exact for a separable convex cost; for D_alpha the divisor is d_alpha(k) = |(k+1)^b - k^b|^(-1/a), a = (1-alpha)/2, b = (1+alpha)/2:")
    print("      alpha = 3:  d(k) = 2k+1, Webster / Sainte-Lague (d = k + 1/2);   alpha = -3: d(k) = sqrt(k(k+1)), Hill-Huntington;   alpha = -1: d(k) = 1/log(1+1/k), Theil-Schrage;")
    print("      alpha -> +inf: d -> k+1, Jefferson / d'Hondt;   alpha -> -inf: d -> k, Adams;   d(0) = 0 for alpha <= -1 (every state gets a seat); Dean (harmonic mean of k, k+1) is not of this form")
    ks = np.arange(0, 6)
    d3 = np.exp(log_d(ks, 3.0))
    dm3 = np.exp(log_d(ks, -3.0))
    dm1 = np.exp(log_d(ks, -1.0))
    print(f"      numerically: d_3(k) / 2 - (k + 1/2) = {np.abs(d3 / 2 - (ks + 0.5)).max():.1e} for k = 0..5; d_-3(k) - sqrt(k(k+1)) = {np.abs(dm3 - np.sqrt(ks * (ks + 1.0))).max():.1e}; d_-1(k) - 1/log(1+1/k) (k >= 1) = {np.abs(dm1[1:] - 1 / np.log1p(1 / ks[1:])).max():.1e}")
    print(f"      d_60(k)/(k+1) at k = 1, 3, 8: {np.exp(log_d(np.array([1.0, 3.0, 8.0]), 60.0)) / np.array([2.0, 4.0, 9.0])}   d_-60(k)/k: {np.exp(log_d(np.array([1.0, 3.0, 8.0]), -60.0)) / np.array([1.0, 3.0, 8.0])}")
    rng = np.random.default_rng(7)
    C = compositions(4, 9)
    agree, total = 0, 0
    for _ in range(300):
        pop = np.round(rng.uniform(1, 30, size=4), 1)
        for a in (-3.0, -1.0, 0.0, 1.0, 3.0):
            total += 1
            agree += greedy(pop, 9, a) in brute(pop, 9, a, C)
    print(f"   greedy with d_alpha against brute-force enumeration of all {len(C)} integer allocations (4 states, 9 seats): agreement in {agree} of {total} random populations x alpha in -3, -1, 0, 1, 3")
    pop = np.array([1.3, 6.6, 21.1, 6.8])
    n = 9
    classical = {"Jefferson (d = k+1)": lambda k: k + 1.0, "Webster (d = k+1/2)": lambda k: k + 0.5, "Hill-Huntington (d = sqrt(k(k+1)))": lambda k: np.sqrt(np.maximum(k, 1e-9) * (k + 1.0)),
                 "Theil-Schrage (log mean)": lambda k: np.where(k == 0, 1e-9, 1 / np.log1p(1.0 / np.maximum(k, 1e-9))), "Adams (d = k)": lambda k: np.maximum(k, 1e-9) * 1.0,
                 "Dean (harmonic mean)": lambda k: np.where(k == 0, 1e-9, k * (k + 1.0) / (k + 0.5))}
    print("   the classical rules on the toy population, from their own divisor sequences (no D_alpha involved): " + "; ".join(f"{nm}: {divisor_method(pop, n, d)}" for nm, d in classical.items()))
    print(f"   divisor sequence of alpha = 3, d_3(k) for k = 0..5: {np.round(np.exp(log_d(np.arange(6), 3.0)), 6).tolist()}")
    print(f"   toy population {pop.tolist()}, {n} seats; the seats of the four states:")
    table = []
    for lab, a in (("Adams, alpha -> -inf (-60)", -60.0), ("Hill-Huntington, alpha = -3", -3.0), ("Theil-Schrage = KL[p:q], alpha = -1", -1.0), ("alpha = 0 (Hellinger)", 0.0),
                   ("alpha = 1 (KL[q:p])", 1.0), ("Webster, alpha = 3 (chi-squared)", 3.0), ("Jefferson, alpha -> +inf (60)", 60.0)):
        g = greedy(pop, n, a)
        b = brute(pop, n, a, C)
        table.append((lab, a, g))
        print(f"      {lab:38s}: {g}   brute force: {b}")
    STORE["apportion"] = table
    xs = np.linspace(-3.2, 3.2, 321)
    grid = [float(8 * math.sinh(x)) for x in xs]
    allocs = [greedy(pop, n, a) for a in grid]
    segs = []
    start = grid[0]
    for i in range(1, len(grid)):
        if allocs[i] != allocs[i - 1]:
            lo, hi = grid[i - 1], grid[i]
            for _ in range(40):
                mid = (lo + hi) / 2
                if greedy(pop, n, mid) == allocs[i - 1]:
                    lo = mid
                else:
                    hi = mid
            segs.append((start, (lo + hi) / 2, allocs[i - 1]))
            start = (lo + hi) / 2
    segs.append((start, grid[-1], allocs[-1]))
    STORE["apportion_segments"] = segs
    print(f"   the optimal allocation as alpha runs over [{grid[0]:.0f}, {grid[-1]:.0f}] (bisection to 1e-10 at each change): " + "; ".join(f"{s0:+.4f} .. {s1:+.4f}: {al}" for s0, s1, al in segs))
    print("   Adams, Hill, Theil-Schrage, Webster and Jefferson are the points alpha = -inf, -3, -1, 3, +inf of this one-parameter family (the others show how far each rule extends)")


def check_alpha_mean():
    head("2e. The alpha-mean (4.2.5): special cases, monotonicity (4.47), scale-free characterisation (Theorem 4.7)")

    def mean(x, y, a):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if a == math.inf:
            return np.minimum(x, y)
        if a == -math.inf:
            return np.maximum(x, y)
        if abs(a - 1) < 1e-12:
            return np.sqrt(x * y)
        e = (1 - a) / 2
        return (0.5 * (x ** e + y ** e)) ** (1 / e)

    vals = {a: float(mean(1, 4, a)) for a in (-math.inf, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, math.inf)}
    STORE["amean"] = vals
    print("   m_alpha(1, 4) (the numbers behind Fig. 4.1): " + ", ".join(f"{a}: {v:.4f}" for a, v in vals.items()))
    print(f"   alpha = -1 arithmetic 2.5, alpha = 0: (1/4)(1+2)^2 = {9 / 4:.4f} (also (1/2)((a+b)/2 + sqrt(ab)) = {0.5 * (2.5 + 2):.4f}), alpha = 1 geometric 2, alpha = 3 harmonic 2/(1+1/4) = {2 / 1.25:.4f}")
    al = np.linspace(-6, 6, 241)
    ms = [float(mean(1.0, 4.0, a)) for a in al]
    print(f"   (4.47) m_alpha(a, b) decreases with alpha: strictly decreasing on a grid of 241 values in [-6, 6]: {bool(np.all(np.diff(ms) < 0))}; range of m_alpha(1,4) over the grid {ms[-1]:.4f} to {ms[0]:.4f}")
    STORE["amean_curve"] = [(float(a), v) for a, v in zip(al, ms)]
    c = 3.7
    print("   scale-free (4.34): m(cx, cy) - c m(x,y) for c = 3.7, x = 1, y = 4: " + ", ".join(f"alpha = {a}: {float(mean(c, 4 * c, a) - c * mean(1, 4, a)):.1e}" for a in (-3.0, 0.0, 0.5, 3.0)))
    h = lambda x: x + x ** 2
    hinv = lambda y: (-1 + np.sqrt(1 + 4 * y)) / 2
    print(f"   a quasi-arithmetic mean with h(x) = x + x^2 is not scale-free: m(cx,cy) - c m(x,y) = {float(hinv((h(3.7) + h(14.8)) / 2) - 3.7 * hinv((h(1.0) + h(4.0)) / 2)):.4f}; for h = log x (alpha = 1) the family has h(0) = -inf and for alpha > 1 h is decreasing,")
    print("   so the hypothesis 'increasing, h(0) = 0' before (4.34) excludes the alpha >= 1 members of Theorem 4.7; the proof itself (4.36)-(4.46) uses neither")


def check_integration():
    head("2f. alpha-integration (4.2.6-4.2.8): optimality (Theorem 4.9), the exponent in (4.65), mixture and product of experts")
    rng = np.random.default_rng(8)
    P = np.array([[0.10, 0.25, 0.10, 0.55], [0.35, 0.15, 0.45, 0.05], [0.05, 0.45, 0.20, 0.30]])
    w = np.array([0.5, 0.3, 0.2])
    print(f"   three distributions on four outcomes, weights {w.tolist()}:")
    for i, row in enumerate(P):
        print(f"      p_{i + 1} = {row.tolist()}")

    def integ(P, w, a):
        m = h_inv(np.tensordot(w, h_rep(P, a), axes=1), a)
        return m / m.sum()

    def risk(P, w, q, a):
        return sum(wi * float(D_alpha(pi, q, a)) for wi, pi in zip(w, P))

    print("   (4.65): d D_alpha[p:q]/dq = (2/(1-a)) (1 - (p/q)^((1-a)/2)); the printed integrand has q^(-(1+a)/2) instead of q^(-(1-a)/2), which agrees with (4.66) and with the derivative only at alpha = 0:")
    for a in (-0.5, 0.0, 0.5):
        p0, q0, h = 0.3, 0.5, 1e-6
        num = (float(D_alpha(np.array([p0]), np.array([q0 + h]), a)) - float(D_alpha(np.array([p0]), np.array([q0 - h]), a))) / (2 * h)
        ok = 2 / (1 - a) * (1 - (p0 / q0) ** ((1 - a) / 2))
        pr = 2 / (1 - a) * (1 - p0 ** ((1 - a) / 2) * q0 ** (-(1 + a) / 2))
        print(f"      alpha = {a:+.1f}: numerical derivative {num:.6f}, (2/(1-a))(1 - (p/q)^((1-a)/2)) = {ok:.6f}, with the printed exponent {pr:.6f}")
    grid = (-3.0, -1.0, 0.0, 1.0, 3.0)
    H = np.zeros((len(grid), len(grid)))
    print("   risk R_alpha[q] = sum w_i D_alpha[p_i : q] at the alpha'-integration, excess over the alpha-integration (rows alpha, columns alpha'), and the worst of 2000 random perturbations:")
    print("      alpha    " + "".join(f"alpha'={g:+.0f}   " for g in grid) + "  min perturbation gain")
    for i, a in enumerate(grid):
        q = integ(P, w, a)
        R0 = risk(P, w, q, a)
        for j, g in enumerate(grid):
            H[i, j] = risk(P, w, integ(P, w, g), a) - R0
        worst = 0.0
        for _ in range(2000):
            d = rng.normal(size=4)
            d -= d.mean()
            q2 = q * np.exp(0.2 * d)
            q2 /= q2.sum()
            worst = min(worst, risk(P, w, q2, a) - R0)
        print(f"      {a:+.0f}      " + "".join(f"{H[i, j]:9.5f}   " for j in range(len(grid))) + f"  {worst:.1e}")
    STORE["integration"] = (grid, H)
    print("   every off-diagonal excess is positive and no perturbation lowers the risk: Theorem 4.9; D_alpha[p:q] is convex in q for every alpha (q^((1+a)/2) with the sign of 4/(1-a^2) works out), so the stationary point of (4.66) is the global minimum")
    chk = []
    for a in (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0):
        p0_, q0_, h = 0.3, 0.5, 1e-4
        f = lambda x, a=a: float(D_alpha(np.array([p0_]), np.array([x]), a))
        chk.append(((f(q0_ + h) - 2 * f(q0_) + f(q0_ - h)) / h ** 2, p0_ ** ((1 - a) / 2) * q0_ ** ((a - 3) / 2)))
    print("   convexity of q -> D_alpha[p:q]: its second derivative is p^((1-a)/2) q^((a-3)/2) > 0 for every alpha; numerical second derivative against that at p = 0.3, q = 0.5, alpha = -3, -1.5, -0.5, 0, 0.5, 1.5, 3: "
          + ", ".join(f"{x:.4f} / {y:.4f}" for x, y in chk))
    q0 = integ(P, w, 0.0)
    print(f"   the alpha = 0 integration of the three distributions: q = {np.round(q0, 4).tolist()}, R_0[q] = {risk(P, w, q0, 0.0):.6f}")
    qm = integ(P, w, -1.0)
    qe = integ(P, w, 1.0)
    print(f"   the alpha = -1 integration (mixture) {np.round(qm, 4).tolist()} and the alpha = +1 integration (normalised geometric mean) {np.round(qe, 4).tolist()}")
    print(f"   alpha = -1: integration = the ordinary mixture sum w_i p_i (difference {np.abs(qm - np.tensordot(w, P, axes=1)).max():.1e}); alpha = +1: normalised geometric mean prod p_i^w_i (difference {np.abs(qe - (np.prod(P ** w[:, None], axis=0) / np.prod(P ** w[:, None], axis=0).sum())).max():.1e})")
    print("   so by the book's own convention ((4.56)-(4.57)) the alpha = -1 machine is the mixture of experts and the alpha = +1 machine the product of experts; the sentence after Theorem 4.10 has them the other way round")
    big = integ(P, w, -40.0)
    small = integ(P, w, 40.0)
    mx = P.max(axis=0) / P.max(axis=0).sum()
    mn = P.min(axis=0) / P.min(axis=0).sum()
    print(f"   alpha = -inf (4.53): max_i p_i normalised {np.round(mx, 4).tolist()}, integration at alpha = -40 {np.round(big, 4).tolist()}; alpha = +inf (4.54): min {np.round(mn, 4).tolist()}, alpha = 40 {np.round(small, 4).tolist()}")


# ------------------------------------------------------------------ 3. Tsallis q-entropy

def log_q(u, q):
    return np.log(u) if abs(q - 1) < 1e-12 else (np.asarray(u, float) ** (1 - q) - 1) / (1 - q)


def exp_q(u, q):
    u = np.asarray(u, float)
    return np.exp(u) if abs(q - 1) < 1e-12 else np.maximum(1 + (1 - q) * u, 0) ** (1 / (1 - q))


def check_tsallis_basic():
    head("3. Tsallis q-entropy (section 4.3.1): q-logarithm, the sign of (4.77), alpha = 2q - 1")
    u = np.linspace(0.05, 6, 400)
    print(f"   exp_q(log_q u) - u: largest over q = 0.3, 0.7, 1.5, 2.5 and 400 values of u: {max(np.abs(exp_q(log_q(u, q), q) - u).max() for q in (0.3, 0.7, 1.5, 2.5)):.1e}; q -> 1: log_q(2.5) at q = 1 + 1e-7 is {float(log_q(2.5, 1 + 1e-7)):.7f} against log 2.5 = {math.log(2.5):.7f}")
    rng = np.random.default_rng(0)
    worst = {}
    for q in (0.3, 0.8, 1.5, 2.5):
        # H_q = (sum p^q - 1)/(1-q) has Hessian diag(-q p^(q-2)) on the simplex tangent space
        eig = []
        for _ in range(200):
            p = rng.dirichlet(np.ones(4))
            H = -q * np.diag(p ** (q - 2))
            Z = np.linalg.qr(np.vstack([np.ones(4), np.eye(4)[:3]]).T)[0][:, 1:]        # basis of {sum dp = 0}
            eig.append(np.linalg.eigvalsh(Z.T @ H @ Z).max())
        worst[q] = max(eig)
    print("   (4.76) concavity of H_q: largest eigenvalue of the Hessian restricted to {sum dp = 0} over 200 random p: " + ", ".join(f"q = {q}: {v:.3f}" for q, v in worst.items()) + " (negative for every q > 0, not only 0 < q <= 1)")
    p = rng.dirichlet(np.ones(4))
    r = rng.dirichlet(np.ones(4))
    print("   (4.77): E_p[log_q(r/p)], the right-hand side (1/(1-q))(1 - sum p^q r^(1-q)), and the alpha-divergences with alpha = 2q - 1 (4.77 says 'the same as' D_alpha[p:r]):")
    for q in (0.3, 0.5, 0.8, 1.5):
        E = float(np.sum(p * log_q(r / p, q)))
        S = float((1 - np.sum(p ** q * r ** (1 - q))) / (1 - q))
        al = 2 * q - 1
        print(f"      q = {q}: E_p[log_q(r/p)] = {E:+.6f}, (1/(1-q))(1 - sum p^q r^(1-q)) = {S:+.6f}, D_(2q-1)[p:r] = {float(D_alpha(p, r, al)):.6f}, D_(2q-1)[r:p] = {float(D_alpha(r, p, al)):.6f}, q D_(2q-1)[r:p] = {q * float(D_alpha(r, p, al)):.6f}, q D_(1-2q)[p:r] = {q * float(D_alpha(p, r, 1 - 2 * q)):.6f}")
    print("   so the first equality of (4.77) has the wrong sign (D_q = -E_p[log_q(r/p)], the expectation of a log-ratio is minus a divergence), and D_q[p:r] = q D_(2q-1)[r:p] = q D_(1-2q)[p:r]: the q-divergence")
    print("   is the alpha-divergence with alpha = 2q - 1 only with the two arguments exchanged (alpha -> -alpha) and a factor q; the dual pair at q = 0, 1 is unaffected, and h_alpha = u^(1-q) does match log_q")


def q_family_tools(q):
    def theta_of_p(p):
        return (p[1:] ** (1 - q) - p[0] ** (1 - q)) / (1 - q)

    def p_of_theta(th):
        lo = max(0.0, np.max(-(1 - q) * th)) + 1e-300
        hi = 2.0 + np.max(np.abs((1 - q) * th))
        g = lambda x: x ** (1 / (1 - q)) + np.sum(((1 - q) * th + x) ** (1 / (1 - q))) - 1
        for _ in range(200):
            mid = (lo + hi) / 2
            if g(mid) > 0:
                hi = mid
            else:
                lo = mid
        x = (lo + hi) / 2
        return np.concatenate([[x ** (1 / (1 - q))], ((1 - q) * th + x) ** (1 / (1 - q))])

    def psi(th):
        return float(-log_q(p_of_theta(th)[0], q))

    return theta_of_p, p_of_theta, psi


def num_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def num_hess(f, x, h=1e-3):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h
            ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def check_q_family():
    head("3b. S_n as a q-family (Theorem 4.11) and the escort geometry (4.3.3): psi_q, eta, phi_q, the Bregman divergence (4.95), the conformal factor (4.117)")
    rng = np.random.default_rng(2)
    q = 0.6
    p = np.array([0.4, 0.3, 0.2, 0.1])
    theta_of_p, p_of_theta, psi = q_family_tools(q)
    th = theta_of_p(p)
    pr = p_of_theta(th)
    ps = psi(th)
    lq = log_q(pr, q)
    print(f"   q = {q}, p = {p.tolist()}: theta = {np.round(th, 6).tolist()} (4.84), psi_q = -log_q p_0 = {ps:.8f} = {float(-log_q(p[0], q)):.8f} (4.85); p recovered from theta to {np.abs(pr - p).max():.1e};")
    print(f"      log_q p_x - theta.x + psi_q for the four outcomes: {np.round(lq - np.concatenate([[0.0], th]) + ps, 12).tolist()} (zero: (4.83) holds)")
    hq = float(np.sum(p ** q))
    eta = num_grad(lambda t: psi(t), th)
    print(f"   eta = grad psi_q (numerical) = {np.round(eta, 8).tolist()} against p_i^q / h_q = {np.round(p[1:] ** q / hq, 8).tolist()} (4.93), h_q = {hq:.6f}; the escort distribution (4.96)-(4.98) is {np.round(p ** q / hq, 6).tolist()}, sum {float(np.sum(p ** q / hq)):.1f}")
    phi = float(th @ eta - ps)
    et_full = p ** q / hq
    phi_explicit = (float(np.sum(et_full ** (1 / q)) ** q) - 1) / (1 - q)
    print(f"   phi_q = theta.eta - psi_q = {phi:.8f}; (4.94): (1/(1-q))(1/h_q - 1) = {(1 / hq - 1) / (1 - q):.8f} (exact, no extra scale or constant); as a function of eta, (1/(1-q))((sum eta_i^(1/q))^q - 1) = {phi_explicit:.8f}")
    r = np.array([0.15, 0.35, 0.25, 0.25])
    thr = theta_of_p(r)
    etr = num_grad(lambda t: psi(t), thr)
    hr = float(np.sum(r ** q))
    Dt = psi(th) + float(thr @ etr - psi(thr)) - float(th @ etr)
    Dt_closed = (1 - float(np.sum(p ** (1 - q) * r ** q))) / ((1 - q) * hr)
    print(f"   Bregman divergence psi_q(theta_p) + phi_q(eta_r) - theta_p.eta_r = {Dt:.8f}; (4.95) (1/((1-q) h_q[r])) (1 - sum p^(1-q) r^q) = {Dt_closed:.8f}; q D_(2q-1)[p:r]/h_q[r] = {q * float(D_alpha(p, r, 2 * q - 1)) / hr:.8f}")
    # metric
    n = 3
    J = np.zeros((n + 1, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1e-6
        J[:, i] = (p_of_theta(th + e) - p_of_theta(th - e)) / 2e-6
    gF = J.T @ np.diag(1 / p) @ J
    Hs = num_hess(lambda t: psi(t), th)
    ratio = Hs / gF
    print(f"   Hessian of psi_q (the q-metric (4.92)) divided entrywise by the Fisher metric in the same theta chart: all entries {np.round(ratio.min(), 6)} to {np.round(ratio.max(), 6)};")
    print(f"      q/h_q = {q / hq:.6f}, 1/h_q = {1 / hq:.6f}: with (4.91) as printed the conformal factor is sigma = q/h_q, not 1/h_q of (4.117); eigenvalues of g_F^-1 Hess psi_q: {np.round(np.linalg.eigvals(np.linalg.solve(gF, Hs)).real, 6).tolist()}")
    u1, u2 = rng.normal(size=3), rng.normal(size=3)
    c1 = (u1 @ gF @ u2) / math.sqrt((u1 @ gF @ u1) * (u2 @ gF @ u2))
    c2 = (u1 @ Hs @ u2) / math.sqrt((u1 @ Hs @ u1) * (u2 @ Hs @ u2))
    print(f"   angles are kept (cosine of two random tangent vectors): Fisher {c1:.8f}, q-metric {c2:.8f}; squared lengths scale by {float((u1 @ Hs @ u1) / (u1 @ gF @ u1)):.6f}")
    STORE["q_sigma"] = (q, hq)
    # Pythagorean theorem for the escort divergence: (theta_Q - theta_P).(eta_Q - eta_R) = 0
    worst = 0.0
    cnt = 0
    while cnt < 100:
        P = rng.dirichlet(np.ones(4) * 2)
        Q = rng.dirichlet(np.ones(4) * 2)
        thP, thQ = theta_of_p(P), theta_of_p(Q)
        etQ = (Q ** q / np.sum(Q ** q))[1:]
        v = rng.normal(size=3)
        d = thQ - thP
        v -= d * (v @ d) / (d @ d)
        etR = etQ + 0.05 * v / np.abs(v).max() * etQ.min()
        e0 = 1 - etR.sum()
        if e0 <= 0 or np.any(etR <= 0):
            continue
        full = np.concatenate([[e0], etR])
        R = full ** (1 / q) / np.sum(full ** (1 / q))
        Dt = lambda a, b: (1 - float(np.sum(a ** (1 - q) * b ** q))) / ((1 - q) * float(np.sum(b ** q)))
        worst = max(worst, abs(Dt(P, R) - Dt(P, Q) - Dt(Q, R)))
        cnt += 1
    print(f"   Pythagorean theorem for the escort divergence (a theta-geodesic orthogonal to an eta-geodesic), 100 random triples on S_3: largest |D~[P:R] - D~[P:Q] - D~[Q:R]| = {worst:.1e}")
    # non-invariance
    rng2 = np.random.default_rng(5)
    Pa = rng2.dirichlet(np.ones(4), size=20000)
    Ra = rng2.dirichlet(np.ones(4), size=20000)
    mg = lambda X: np.stack([X[:, 0] + X[:, 1], X[:, 2], X[:, 3]], axis=1)
    Pc, Rc = mg(Pa), mg(Ra)
    print("   information monotonicity D[Tp:Tr] <= D[p:r] under merging two outcomes, 20000 random pairs on S_3 (largest D[Tp:Tr] - D[p:r], positive = violated):")
    best = {}
    mono = {}
    for qq in (0.5, 0.8, 1.5):
        Dt = lambda X, Y, qq=qq: (1 - np.sum(X ** (1 - qq) * Y ** qq, axis=1)) / ((1 - qq) * np.sum(Y ** qq, axis=1))
        Dq = lambda X, Y, qq=qq: (1 - np.sum(X ** qq * Y ** (1 - qq), axis=1)) / (1 - qq)
        v = Dt(Pc, Rc) - Dt(Pa, Ra)
        k = int(v.argmax())
        wa = float(np.max(D_alpha(Pc, Rc, 2 * qq - 1) - D_alpha(Pa, Ra, 2 * qq - 1)))
        wd = float(np.max(Dq(Pc, Rc) - Dq(Pa, Ra)))
        best[qq] = (float(v[k]), (Pa[k], Ra[k]))
        mono[qq] = (float(v[k]), wa, wd)
        print(f"      q = {qq}: escort divergence (4.95): {v[k]:+.4f};  alpha-divergence with alpha = 2q-1: {wa:+.1e};  q-divergence (4.77): {wd:+.1e}")
    # a clean counterexample with round numbers (search a small grid)
    Dt5 = lambda x, y, qq=0.5: (1 - np.sum(x ** (1 - qq) * y ** qq)) / ((1 - qq) * np.sum(y ** qq))
    cand = None
    for x in ([0.02, 0.02, 0.1, 0.86], [0.05, 0.05, 0.1, 0.8], [0.03, 0.02, 0.05, 0.9]):
        for y in ([0.5, 0.45, 0.04, 0.01], [0.49, 0.49, 0.01, 0.01], [0.45, 0.45, 0.05, 0.05]):
            x, y = np.array(x), np.array(y)
            xc, yc = mg(x[None, :])[0], mg(y[None, :])[0]
            gap = Dt5(xc, yc) - Dt5(x, y)
            if cand is None or gap > cand[0]:
                cand = (gap, x, y, xc, yc)
    gap, x, y, xc, yc = cand
    print(f"   an explicit violation at q = 0.5: p = {np.round(x, 4).tolist()}, r = {np.round(y, 4).tolist()}: D~[p:r] = {Dt5(x, y):.4f}; merging outcomes 0 and 1 gives p' = {np.round(xc, 4).tolist()}, r' = {np.round(yc, 4).tolist()} and D~[p':r'] = {Dt5(xc, yc):.4f} (larger)")
    print("   so the flat escort structure is not invariant, as the book says; for q = 1.5 I found no violation in this sample (not a proof of monotonicity there)")
    STORE["escort_violation"] = best
    STORE["escort_mono"] = mono


def check_qmaxent():
    head("3c. The q-max-entropy theorem (4.12): the maximiser of H_q under escort-expectation constraints is a q-exponential family")
    rng = np.random.default_rng(5)
    q = 0.7
    c = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 1.0, 0.0, 2.0]])
    p_true = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    esc = lambda p: p ** q / np.sum(p ** q)
    a_t = c @ esc(p_true)
    Hq = lambda p: (np.sum(p ** q) - 1) / (1 - q)

    def p_of(theta):
        z = theta @ c
        lo, hi = np.max(z) - 1 / (1 - q) + 1e-12, np.max(z) + 50
        f = lambda psi: np.sum(np.maximum(1 + (1 - q) * (z - psi), 0) ** (1 / (1 - q))) - 1
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        psi = (lo + hi) / 2
        return np.maximum(1 + (1 - q) * (z - psi), 0) ** (1 / (1 - q)), psi

    resid = lambda th: c @ esc(p_of(th)[0]) - a_t
    theta = np.zeros(2)
    for _ in range(50):
        r = resid(theta)
        if np.abs(r).max() < 1e-13:
            break
        Jm = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-7
            Jm[:, j] = (resid(theta + e) - resid(theta - e)) / 2e-7
        theta = theta - np.linalg.solve(Jm, r)
    p_hat, psi = p_of(theta)
    print(f"   q = {q}, five outcomes, two random variables c_1, c_2, constraints E~_q[c] = a = {np.round(a_t, 6).tolist()}; the q-exponential p = exp_q(theta.c - psi) with theta = {np.round(theta, 6).tolist()}, psi = {psi:.6f}")
    print(f"      p_hat = {np.round(p_hat, 6).tolist()}, sum {p_hat.sum():.12f}, escort expectations {np.round(c @ esc(p_hat), 10).tolist()}; H_q(p_hat) = {Hq(p_hat):.6f} against H_q(uniform) = {Hq(np.ones(5) / 5):.6f}")
    gap = 0.0
    cnt = 0
    for _ in range(1500):
        pp = np.abs(p_hat + 0.1 * rng.normal(size=5)) + 1e-9
        x = pp / pp.sum()
        for _it in range(30):
            g1 = np.array([np.sum(x) - 1, *(np.sum(x ** q * (c[k] - a_t[k])) for k in range(2))])
            if np.abs(g1).max() < 1e-12:
                break
            Jc = np.array([np.ones(5), *(q * x ** (q - 1) * (c[k] - a_t[k]) for k in range(2))])
            x = np.maximum(x + np.linalg.lstsq(Jc, -g1, rcond=None)[0], 1e-12)
        if np.abs(np.array([np.sum(x) - 1, *(np.sum(x ** q * (c[k] - a_t[k])) for k in range(2))])).max() > 1e-9:
            continue
        cnt += 1
        gap = max(gap, Hq(x) - Hq(p_hat))
    print(f"   {cnt} random feasible points near p_hat (same constraints): largest H_q(x) - H_q(p_hat) = {gap:.1e} (never positive): consistent with Theorem 4.12 (the constraint set is not convex for q < 1, so this is evidence, not a proof)")
    lq = log_q(p_hat, q)
    print(f"   log_q p_x - log_q p_0 = {np.round(lq - lq[0], 8).tolist()} = theta.(c_x - c_0) = {np.round(theta @ (c - c[:, [0]]), 8).tolist()}: the maximiser is on the straight line (in theta) from the uniform distribution, a q-geodesic")


def chi_families():
    class Chi:
        def __init__(self, name, chi, u, v):
            self.name, self.chi, self.u, self.v = name, chi, u, v
            self.du = lambda s: 1 / chi(s)

    def power(q):
        return Chi(f"power chi = s^{q}", lambda s: s ** q, lambda s: (s ** (1 - q) - 1) / (1 - q), lambda y: (1 + (1 - q) * y) ** (1 / (1 - q)))

    def kappa(k):
        return Chi(f"kappa-exponential, kappa = {k}", lambda s: 2 * s / (s ** k + s ** (-k)), lambda s: (s ** k - s ** (-k)) / (2 * k), lambda y: (np.sqrt(1 + k * k * y * y) + k * y) ** (1 / k))

    def sq():
        return Chi("chi = s + s^2", lambda s: s + s * s, lambda s: np.log(2 * s / (1 + s)),
                   lambda y: np.where(np.asarray(y) < np.log(2) - 1e-12, np.exp(np.minimum(y, 0.69)) / (2 - np.exp(np.minimum(y, 0.69))), np.inf))

    return power(0.6), kappa(0.5), sq()


def check_chi():
    head("3d. The chi-deformed family (4.3.4) and conformal flatness (Theorem 4.14)")
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.15, 0.35, 0.25, 0.25])
    print("   u = log_chi (4.103), v = exp_chi = u^-1 (4.105), theta_i = u(p_i) - u(p_0), psi = -u(p_0), p = exp_chi(theta.x - psi); the book writes theta^i = u(p_i) - psi_chi (4.107): with (4.108) that")
    print("   is u(p_i) + u(p_0), the q case (4.84) has u(p_i) - u(p_0) = u(p_i) + psi_chi. For three choices of chi (a power, the Kaniadakis kappa-logarithm, and chi = s + s^2):")
    eigs_store = {}
    for ch in chi_families():
        def p_of_theta(th, ch=ch):
            lo, hi = 1e-14, 1.0
            for _ in range(100):
                mid = (lo + hi) / 2
                if mid + np.sum(ch.v(ch.u(mid) + th)) - 1 > 0:
                    hi = mid
                else:
                    lo = mid
            p0 = (lo + hi) / 2
            return np.concatenate([[p0], ch.v(ch.u(p0) + th)])

        psi = lambda th, ch=ch: -float(ch.u(p_of_theta(th)[0]))
        th = ch.u(p[1:]) - ch.u(p[0])
        eta = num_grad(psi, th)
        hchi = float(np.sum(ch.chi(p)))
        n = 3
        J = np.zeros((n + 1, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1e-6
            J[:, i] = (p_of_theta(th + e) - p_of_theta(th - e)) / 2e-6
        gF = J.T @ np.diag(1 / p) @ J
        H = num_hess(psi, th)
        ev = np.sort(np.linalg.eigvals(np.linalg.solve(gF, H)).real)
        eigs_store[ch.name] = ev
        print(f"   {ch.name}: p recovered to {np.abs(p_of_theta(th) - p).max():.1e}; eta = grad psi = {np.round(eta, 6).tolist()} = chi(p_i)/h_chi = {np.round(ch.chi(p[1:]) / hchi, 6).tolist()};")
        print(f"      Hessian of psi positive definite: {bool(np.all(np.linalg.eigvalsh(H) > 0))}; eigenvalues of g_F^-1 Hess psi: {np.round(ev, 5).tolist()}" + ("  (all equal: conformal)" if ev.max() - ev.min() < 1e-4 else "  (not equal: not conformal)"))
        # in the theta chart both metrics have the shape M[w]_jk = w_j d_jk - w_j eta_k - eta_j w_k + eta_j eta_k sum_x w_x (x = 0..n)
        etx = ch.chi(p[1:]) / hchi
        dchi = (ch.chi(p + 1e-6) - ch.chi(p - 1e-6)) / 2e-6

        def Mm(w, etx=etx):
            return np.diag(w[1:]) - np.outer(w[1:], etx) - np.outer(etx, w[1:]) + w.sum() * np.outer(etx, etx)
        resH = np.abs(H - Mm(dchi * ch.chi(p)) / hchi).max() / np.abs(H).max()
        resg = np.abs(gF - Mm(ch.chi(p) ** 2 / p)).max() / np.abs(gF).max()
        print(f"      structure: Hess psi = M[chi' chi]/h_chi (relative residual {resH:.1e}) and g_F = M[chi^2/p] (relative residual {resg:.1e}), M[w]_jk = w_j d_jk - w_j eta_k - eta_j w_k + eta_j eta_k sum_x w_x")
        phi = float(th @ eta - psi(th))
        phi_ok = float(np.sum(ch.u(p) / ch.du(p)) / hchi)
        thq = ch.u(q[1:]) - ch.u(q[0])
        etq = num_grad(psi, thq)
        Dpq = psi(th) + float(thq @ etq - psi(thq)) - float(th @ etq)
        Dqp = psi(thq) + phi - float(thq @ eta)
        hq_ = float(np.sum(ch.chi(q)))
        ok = float(np.sum((ch.u(q) - ch.u(p)) / ch.du(q)) / hq_)
        prn = float(np.sum((ch.u(p) - ch.u(q)) / ch.du(p)) / hchi)
        print(f"      phi_chi = theta.eta - psi = {phi:.8f} = (1/h) sum u(p_i)/u'(p_i) = {phi_ok:.8f}  (this is (4.113) with v read as u);  D[p:q] = {Dpq:.8f} = (1/h_chi(q)) sum (u(q_i) - u(p_i))/u'(q_i) = {ok:.8f};")
        print(f"      D[q:p] = {Dqp:.8f} = what (4.114) gives with its v' read as u' ((1/h_chi(p)) sum (u(p_i) - u(q_i))/u'(p_i) = {prn:.8f}): the printed (4.114) has the two arguments exchanged")
        v_prime_at_u = float(np.sum(ch.chi(ch.v(ch.u(p)))) / hchi)
        pw = (ch.v(p + 1e-6) - ch.v(p - 1e-6)) / 2e-6
        print(f"      read literally with v = exp_chi as defined in (4.105), (4.113) would give {float(np.sum(ch.v(p) / pw) / hchi):.5f}, not phi_chi: u and v are interchanged in (4.109)-(4.113) (u'' < 0 here, so (4.109) as printed would not be positive)")
    STORE["chi_eigs"] = eigs_store
    # conformality scatter
    rng = np.random.default_rng(12)
    pts = []
    pw, kp = chi_families()[0], chi_families()[1]
    for ch, tag in ((pw, "power"), (kp, "kappa")):
        for _ in range(25):
            pp = rng.dirichlet(np.ones(4) * 1.5)

            def p_of_theta(th, ch=ch):
                lo, hi = 1e-14, 1.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if mid + np.sum(ch.v(ch.u(mid) + th)) - 1 > 0:
                        hi = mid
                    else:
                        lo = mid
                p0 = (lo + hi) / 2
                return np.concatenate([[p0], ch.v(ch.u(p0) + th)])

            psi = lambda th, ch=ch: -float(ch.u(p_of_theta(th)[0]))
            th = ch.u(pp[1:]) - ch.u(pp[0])
            J = np.zeros((4, 3))
            for i in range(3):
                e = np.zeros(3)
                e[i] = 1e-6
                J[:, i] = (p_of_theta(th + e) - p_of_theta(th - e)) / 2e-6
            gF = J.T @ np.diag(1 / pp) @ J
            ev = np.sort(np.linalg.eigvals(np.linalg.solve(gF, num_hess(psi, th))).real)
            pts.append((tag, ev[0], ev[-1]))
    STORE["conformal_scatter"] = pts
    spread = {tag: max((e1 - e0) / e1 for t, e0, e1 in pts if t == tag) for tag in ("power", "kappa")}
    print(f"   25 random points of S_3 each: largest relative spread (max - min)/max of the eigenvalues of g_F^-1 Hess psi: power chi {spread['power']:.1e}, kappa-exponential {spread['kappa']:.3f}")
    print("   Theorem 4.14 (only the q-escort geometry is conformal to the Fisher metric) is consistent: v v''/v'^2 = chi'(v) v / chi(v) must be constant, i.e. chi a power; the printed factor (4.117) should be q/h_q")


def check_qgauss():
    head("3e. The q-Gaussian (4.81)-(4.82): normalisation and support")
    xg, wg = np.polynomial.legendre.leggauss(800)

    def total(c, q, mu, sg):
        """integral of exp_q(-(x-mu)^2/(2 sg^2) - c) over the real line: on the compact support for q < 1, by x = sinh t otherwise."""
        f = lambda x: exp_q(-(x - mu) ** 2 / (2 * sg ** 2) - c, q)
        if q < 1:
            r2 = 2 * sg ** 2 * (1 / (1 - q) - c)
            if r2 <= 0:
                return 0.0
            r = math.sqrt(r2)
            return r * float(np.sum(wg * f(mu + r * xg)))
        t = 12 * xg
        return 12 * float(np.sum(wg * f(mu + np.sinh(t)) * np.cosh(t)))

    mu, sg = 0.3, 1.0
    print(f"   log_q p = -(x-mu)^2/(2 sigma^2) = theta.x - mu^2/(2 sigma^2) with theta = (mu/sigma^2, -1/(2 sigma^2)) does not satisfy the normalisation (4.80); the psi_q that does (p = exp_q(-(x-mu)^2/(2 sigma^2) - c), psi_q = c + mu^2/(2 sigma^2); mu = {mu}, sigma = {sg}):")
    rows = []
    for q in (0.5, 1.0, 1.5):
        lo, hi = -5.0, (1 / (1 - q) - 1e-9 if q < 1 else 8.0)
        for _ in range(100):
            mid = (lo + hi) / 2
            if total(mid, q, mu, sg) > 1:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2
        extra = ""
        if q < 1:
            extra = f"; compact support |x - mu| < {math.sqrt(2 * sg ** 2 * (1 / (1 - q) - c)):.4f}"
        if q > 1:
            extra = f"; power-law tail: density at x = 20 is {float(exp_q(-(20 - mu) ** 2 / 2 - c, q)):.2e} against the Gaussian's {math.exp(-(20 - mu) ** 2 / 2) / math.sqrt(2 * math.pi):.1e}"
        rows.append((q, c))
        print(f"      q = {q}: psi_q(theta) = c + mu^2/(2 sigma^2) = {c + mu ** 2 / (2 * sg ** 2):.6f} (c = {c:.6f}) against {mu ** 2 / (2 * sg ** 2):.6f} in the printed middle expression{extra}")
    A = (15 / 32) ** 0.4
    print(f"   check of q = 1/2 by hand: exp_(1/2)(u) = (1 + u/2)^2, so p = (A - y^2/4)^2 with A = 1 - c/2 and integral (32/15) A^(5/2) = 1, giving A = (15/32)^(2/5) = {A:.6f}, c = 2(1 - A) = {2 * (1 - A):.6f}, support radius 2 sqrt(A) = {2 * math.sqrt(A):.4f}")
    print(f"   at q = 1 it is the Gaussian: c = log sqrt(2 pi sigma^2) = {math.log(math.sqrt(2 * math.pi)):.6f}. So x is confined to a finite interval only for q < 1 (the book says it of the q-Gaussian without qualification);")
    print("   for q > 1, the case of Tsallis physics, the tails are power laws")
    STORE["qgauss"] = rows


# ------------------------------------------------------------------ 4. (u,v)-divergence

def check_uv():
    head("4. (u,v)-divergence (section 4.4): Definition 4.1, special cases, the metric (4.130)-(4.131), the beta-divergence (4.139)-(4.140)")
    xg, wg = np.polynomial.legendre.leggauss(200)

    def integ0(f, m):
        """int_0^m f(s) ds with s = m t^4 (smooths the endpoint singularities)."""
        t = (xg + 1) / 2
        return float(m * np.sum(wg / 2 * f(m * t ** 4) * 4 * t ** 3))

    def D_uv(m, mp, u, du, v, dv):
        return integ0(lambda s: v(s) * du(s), m) + integ0(lambda s: u(s) * dv(s), mp) - u(m) * v(mp)

    m, mp = 1.7, 0.6
    print(f"   (alpha,beta)-divergence (4.132)-(4.133) at m = {m}, m' = {mp}: quadrature of (4.127) against the closed form (4.133):")
    for al, be in ((1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (0.3, 0.3)):
        u = lambda s, al=al: s ** al / al
        du = lambda s, al=al: s ** (al - 1)
        v = lambda s, be=be: s ** be / be
        dv = lambda s, be=be: s ** (be - 1)
        D = D_uv(m, mp, u, du, v, dv)
        closed = (al * m ** (al + be) + be * mp ** (al + be) - (al + be) * m ** al * mp ** be) / (al * be * (al + be))
        print(f"      (alpha,beta) = ({al}, {be}): {D:.10f} against {closed:.10f}")
    print("   alpha-divergence as (u,v) = (2/(1-a) m^((1-a)/2), 2/(1+a) m^((1+a)/2)) (4.137)-(4.138):")
    for a in (-0.5, 0.0, 0.5, 2.0):
        u = lambda s, a=a: 2 / (1 - a) * s ** ((1 - a) / 2)
        du = lambda s, a=a: s ** (-(1 + a) / 2)
        v = lambda s, a=a: 2 / (1 + a) * s ** ((1 + a) / 2)
        dv = lambda s, a=a: s ** ((a - 1) / 2)
        print(f"      alpha = {a:+.1f}: {D_uv(m, mp, u, du, v, dv):.10f} against D_alpha (3.96) {float(D_alpha(np.array([m]), np.array([mp]), a)):.10f}")
    # (4.124) boundary and (4.172)
    print("   (4.124) needs int_0^m (v u)' = u(m) v(m) - u(0) v(0) with u(0) v(0) = 0; for (4.172), u = m, v = -1/m, the integral int_0^m v u' ds diverges (the potentials are fixed up to constants by duality instead):")
    for eps in (1e-3, 1e-6, 1e-9):
        print(f"      int_eps^1.7 (-1/s) ds with eps = {eps:.0e}: {-math.log(1.7 / eps):.4f}")
    # beta-divergence
    print("   beta-divergence (4.139)-(4.140): (4.127) with u = m and v = m^(1+b)/b as printed, with v = m^b/b, the printed (4.140), and the standard form (1/(b(b+1))) [m^(b+1) + b m'^(b+1) - (b+1) m m'^b]:")
    for be in (0.5, 1.0, 2.0):
        u = lambda s: s
        du = lambda s: np.ones_like(s)
        v1 = lambda s, be=be: s ** (1 + be) / be
        dv1 = lambda s, be=be: (1 + be) / be * s ** be
        v2 = lambda s, be=be: s ** be / be
        dv2 = lambda s, be=be: s ** (be - 1)
        D1 = D_uv(m, mp, u, du, v1, dv1)
        D2 = D_uv(m, mp, u, du, v2, dv2)
        printed = (m ** (be + 1) + (be + 1) * mp - mp ** (be + 1) - (be + 1) * m * mp ** be) / (be * (be + 1))
        std = (m ** (be + 1) + be * mp ** (be + 1) - (be + 1) * m * mp ** be) / (be * (be + 1))
        diag = (m ** (be + 1) + (be + 1) * m - m ** (be + 1) - (be + 1) * m * m ** be) / (be * (be + 1))
        print(f"      beta = {be}: v = m^(1+b)/b: {D1:.6f}; v = m^b/b: {D2:.6f}; printed (4.140): {printed:.6f}; standard: {std:.6f}; printed (4.140) at m' = m = {m}: {diag:+.4f} (not zero)")
    print("   so the exponent of v in (4.139) should be beta, and the bracket of (4.140) (and of (4.180) for matrices) should read m^(b+1) + b m'^(b+1) - (b+1) m m'^b; as printed it is not a divergence (it is negative on the diagonal)")
    # metric
    print("   metric (4.130)-(4.131): second derivative of D_{u,v}[m : m'] in m' at m' = m, in the m chart, against u'v' and v'/u':")
    for al, be in ((0.5, 2.0), (1.0, 1.0)):
        u = lambda s, al=al: s ** al / al
        du = lambda s, al=al: s ** (al - 1)
        v = lambda s, be=be: s ** be / be
        dv = lambda s, be=be: s ** (be - 1)
        mm, h = 1.7, 1e-3
        f = lambda x: D_uv(mm, x, u, du, v, dv)
        d2 = (f(mm + h) - 2 * f(mm) + f(mm - h)) / h ** 2
        print(f"      (alpha,beta) = ({al}, {be}) at m = {mm}: Hessian {d2:.6f}; u'v' = {du(mm) * dv(mm):.6f}; v'/u' = {dv(mm) / du(mm):.6f}")
    for a in (0.0, 0.5, 3.0):
        mm, h = 1.7, 1e-3
        f = lambda x: float(D_alpha(np.array([mm]), np.array([x]), a))
        d2 = (f(mm + h) - 2 * f(mm) + f(mm - h)) / h ** 2
        print(f"      alpha-divergence, alpha = {a}: Hessian {d2:.6f} = 1/m = {1 / mm:.6f} (Theorem 3.4); (4.130) would give m^alpha = {mm ** a:.6f}")
    print("   (4.130) is the metric in the theta chart (the Hessian of psi_{u,v} in theta, (4.123)); in the chart m of the line above it is u'(m) v'(m), and the Euclidean coordinate is r(m) = int sqrt(u'v') dm")
    print("   (2 sqrt m for the alpha-divergence, the xi = 2 sqrt m of (3.92)), not int sqrt(v'/u') of (4.131)")
    # flat but not invariant
    rng = np.random.default_rng(14)
    A_ = rng.dirichlet(np.ones(4), size=20000)
    B_ = rng.dirichlet(np.ones(4), size=20000)
    mg = lambda X: np.stack([X[:, 0] + X[:, 1], X[:, 2], X[:, 3]], axis=1)
    Ac, Bc = mg(A_), mg(B_)
    Db = lambda X, Y, be: np.sum(X ** (be + 1) + be * Y ** (be + 1) - (be + 1) * X * Y ** be, axis=1) / (be * (be + 1))
    wb = {be: float(np.max(Db(Ac, Bc, be) - Db(A_, B_, be))) for be in (0.5, 2.0)}
    wa = float(np.max(D_alpha(Ac, Bc, 0.5) - D_alpha(A_, B_, 0.5)))
    wf = float(np.max(0.5 * np.sum((Ac - Bc) ** 2, axis=1) - 0.5 * np.sum((A_ - B_) ** 2, axis=1)))
    print("   flat is not invariant: the beta-divergence (u linear in m, so S_n stays flat: sum m = 1 is linear in theta = m) and the Euclidean divergence violate information monotonicity under")
    print(f"   merging two outcomes (largest D[Tp:Tr] - D[p:r] over 20000 random pairs of S_3: beta = 0.5: {wb[0.5]:+.4f}, beta = 2: {wb[2.0]:+.4f}, Euclidean (beta = 1): {wf:+.4f}; alpha = 0.5: {wa:+.1e}); by Theorem 4.1 they cannot be invariant")
    STORE["monotone_beta"] = (wb, wf, wa)
    # criterion (4.145)
    print("   criterion (4.145): eta = v(u^-1(theta)) must be a gradient, i.e. its Jacobian symmetric: for a decomposable pair the Jacobian is diagonal; for the coupled pair")
    print("   u(m) = m, v(m) = (m_1 + m_2^2/2, m_2) it is [[1, m_2], [0, 1]] (not symmetric): no convex psi, so no dually flat structure")


# ------------------------------------------------------------------ 5. positive-definite matrices

def sym_fun(P, f):
    w, V = np.linalg.eigh(P)
    return (V * f(w)) @ V.T


def rand_pd(rng, n):
    A = rng.normal(size=(n, n))
    return A @ A.T / n + 0.3 * np.eye(n)


def D_stein(P, Q):
    """(4.151): tr(P Q^-1) - log det(P Q^-1) - n."""
    n = len(P)
    M = P @ np.linalg.inv(Q)
    return float(np.trace(M) - np.log(np.linalg.det(M)) - n)


def kl_gauss_quad(P, Q, order=24):
    """KL[N(0,P) || N(0,Q)] by tensor Gauss-Hermite quadrature of log p/q under p (no formula for KL used)."""
    n = len(P)
    x, w = np.polynomial.hermite_e.hermegauss(order)
    L = np.linalg.cholesky(P)
    pts = np.array(list(itertools.product(range(order), repeat=n)))
    wt = np.prod(w[pts], axis=1) / (2 * math.pi) ** (n / 2)
    X = x[pts] @ L.T
    Pi, Qi = np.linalg.inv(P), np.linalg.inv(Q)
    logp = -0.5 * np.einsum("ni,ij,nj->n", X, Pi, X) - 0.5 * np.log(np.linalg.det(2 * math.pi * P))
    logq = -0.5 * np.einsum("ni,ij,nj->n", X, Qi, X) - 0.5 * np.log(np.linalg.det(2 * math.pi * Q))
    return float(np.sum(wt * (logp - logq)))


def D_fro(P, Q):
    return float(0.5 * np.sum((P - Q) ** 2))


def D_vn(P, Q):
    lg = lambda M: sym_fun(M, np.log)
    return float(np.trace(P @ lg(P) - P @ lg(Q) - P + Q))


def D_alpha_mat(P, Q, a):
    """(4.179)."""
    Pa = sym_fun(P, lambda w: w ** ((1 - a) / 2))
    Qb = sym_fun(Q, lambda w: w ** ((1 + a) / 2))
    return float(4 / (1 - a * a) * np.trace(-Pa @ Qb + (1 - a) / 2 * P + (1 + a) / 2 * Q))


def D_ab_mat(P, Q, al, be):
    """(4.176)."""
    return float(np.trace(al / (al + be) * sym_fun(P, lambda w: w ** (al + be)) + be / (al + be) * sym_fun(Q, lambda w: w ** (al + be))
                          - sym_fun(P, lambda w: w ** al) @ sym_fun(Q, lambda w: w ** be)))


def pq_eigs(P, Q):
    w, V = np.linalg.eigh(Q)
    Qih = (V / np.sqrt(w)) @ V.T
    return np.linalg.eigvalsh(Qih @ P @ Qih)


def D_logdet(P, Q, al, be):
    """(4.182)-(4.185) through the eigenvalues of P Q^-1."""
    lam = pq_eigs(P, Q)
    if abs(al) < 1e-12 and abs(be) < 1e-12:
        return float(0.5 * np.sum(np.log(lam) ** 2))
    if abs(be) < 1e-12:
        return float((np.sum(lam ** (-al) + al * np.log(lam)) - len(lam)) / al ** 2)
    if abs(al) < 1e-12:
        return float((np.sum(lam ** be - be * np.log(lam)) - len(lam)) / be ** 2)
    return float(np.sum(np.log((al * lam ** be + be * lam ** (-al)) / (al + be))) / (al * be))


def check_matrices():
    head("5. Positive-definite matrices (section 4.5): (4.150)-(4.155), invariance, Lemma (4.158), Theorem 4.17, (u,v) for matrices")
    rng = np.random.default_rng(11)
    xg, wg = np.polynomial.legendre.leggauss(400)
    grow = ", ".join(f"{float(np.sum(wg * np.exp((L * xg) ** 2 / 2)) * L):.3g} (L = {L})" for L in (5, 10, 20))
    print("   (4.150) as printed has exp{+(1/2) x^T P^-1 x - ...}: in one dimension int_-L^L exp(+x^2/2) dx (P = 1) is " + grow + ";")
    print(f"      the density needs exp{{-(1/2) x^T P^-1 x}}, whose integral is sqrt(2 pi) = {math.sqrt(2 * math.pi):.6f}: a sign slip.")
    P = np.array([[1.4, 0.3], [0.3, 0.9]])
    Q = np.array([[0.8, -0.2], [-0.2, 1.3]])
    kl_pq = kl_gauss_quad(P, Q)
    kl_qp = kl_gauss_quad(Q, P)
    print(f"   P = {P.tolist()}, Q = {Q.tolist()}: (4.151) D[P:Q] = {D_stein(P, Q):.12f}; KL[N(0,P)||N(0,Q)] by Gauss-Hermite quadrature = {kl_pq:.12f}; twice that = {2 * kl_pq:.12f}; D[Q:P] = {D_stein(Q, P):.12f} = 2 KL[N(0,Q)||N(0,P)] = {2 * kl_qp:.12f}")

    def breg(f, gradf, X1, X2):
        return float(f(X1) - f(X2) - np.trace(gradf(X2) @ (X1 - X2)))

    psi = lambda M: -np.log(np.linalg.det(M))
    gpsi = lambda M: -np.linalg.inv(M)
    print(f"   as a Bregman divergence: psi = -log det of the m-coordinate P gives D_psi[P:Q] = {breg(psi, gpsi, P, Q):.12f} = (4.151); psi = -log det of the e-coordinate Theta = P^-1 gives {breg(psi, gpsi, np.linalg.inv(Q), np.linalg.inv(P)):.12f} for D_psi[Theta_Q : Theta_P],")
    print(f"   and {breg(psi, gpsi, np.linalg.inv(P), np.linalg.inv(Q)):.12f} = D[Q:P] for the other order: (4.152) is the potential of the divergence with the arguments reversed, and (4.151) is twice the Gaussian KL, not the KL.")
    # invariance
    L = rng.normal(size=(3, 3)) + 2 * np.eye(3)
    P3, Q3 = rand_pd(rng, 3), rand_pd(rng, 3)
    O = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c = 3.0
    divs = [("KL / Stein (4.151)", D_stein), ("Frobenius (4.163)", D_fro), ("von Neumann (4.166)", D_vn), ("alpha = 0 (4.179)", lambda a, b: D_alpha_mat(a, b, 0.0)),
            ("(alpha,beta) = (1.3, 0.7) (4.176)", lambda a, b: D_ab_mat(a, b, 1.3, 0.7)), ("log-det (0.5, 1.5) (4.182)", lambda a, b: D_logdet(a, b, 0.5, 1.5))]
    print("   ratio D[L^T P L : L^T Q L] / D[P:Q] for a random invertible L (3x3), for a rotation O, and D[cP:cQ]/D[P:Q] for c = 3 (random positive-definite P, Q):")
    rows = []
    for name, D in divs:
        d0 = D(P3, Q3)
        rl = D(L.T @ P3 @ L, L.T @ Q3 @ L) / d0
        ro = D(O.T @ P3 @ O, O.T @ Q3 @ O) / d0
        rc = D(c * P3, c * Q3) / d0
        rows.append((name, rl, ro, rc))
        print(f"      {name:36s} L: {rl:12.6f}   O: {ro:.10f}   c: {rc:.6f}")
    STORE["invariance_rows"] = rows
    L2 = np.array([[2.0, 0.5], [-0.3, 1.5]])
    print("   the same for the 2x2 matrices P, Q above and L = [[2, 0.5], [-0.3, 1.5]] (the default state of the interactive page): D[P:Q] -> D[L^T P L : L^T Q L]")
    for name, D in divs:
        print(f"      {name:36s} {D(P, Q):.6f} -> {D(L2.T @ P @ L2, L2.T @ Q @ L2):.6f}")
    cs = np.array([0.25, 0.5, 1, 2, 4, 8])
    STORE["scale_curves"] = {name: [D(c_ * P3, c_ * Q3) / D(P3, Q3) for c_ in cs] for name, D in divs}
    STORE["scale_cs"] = cs
    print("   only the log-det divergences (functions of the eigenvalues of P Q^-1) are invariant under congruence; every other (u,v)-divergence is invariant under rotations only and changes under the scaling P -> cP,")
    print("   so Theorem 4.16 does not extend to them. (4.154): x~ = L x has covariance L P L^T, the transformation P~ = L^T P L of (4.153) belongs to x~ = L^T x (harmless: Gl(n) is closed under transposition).")
    # Lemma (4.158)
    f = lambda w: w ** 3 / 3 + np.log(w)
    fp = lambda w: w ** 2 + 1 / w
    P2 = rand_pd(rng, 2)
    trf = lambda M: float(np.sum(f(np.linalg.eigvalsh(M))))
    G = np.zeros((2, 2))
    h = 1e-6
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2))
            E[i, j] += 1
            if i != j:
                E[j, i] += 1
            d = (trf(P2 + h * E) - trf(P2 - h * E)) / (2 * h)
            G[i, j] = d / 2 if i != j else d
    print(f"   Lemma (4.158): numerical gradient of tr f(P) for f = lambda^3/3 + log lambda (symmetric perturbations) differs from f'(P) by {np.abs(G - sym_fun(P2, fp)).max():.1e}")
    # Theorem 4.17 examples and the normalisation of g
    Pn, Qn = rand_pd(rng, 3), rand_pd(rng, 3)
    f = lambda w: w * np.log(w) - w
    direct = float(np.trace(sym_fun(Pn, f) - sym_fun(Qn, f) - (Pn - Qn) @ sym_fun(Qn, np.log)))
    print(f"   examples (4.162)-(4.166): tr[f(P) - f(Q) - (P-Q) f'(Q)] for f = lambda log lambda - lambda is {direct:.10f} = (4.166) {D_vn(Pn, Qn):.10f}; f = lambda^2/2 gives {D_fro(Pn, Qn):.10f} = (1/2)||P-Q||^2")
    psi_f = float(np.trace(sym_fun(Pn, f)))
    g_book = lambda w: np.exp(w) - 1          # g' = (f')^-1 = exp with g(0) = 0 as in the text after (4.158)
    D_book = psi_f + float(np.trace(sym_fun(sym_fun(Qn, np.log), g_book))) - float(np.trace(Pn @ sym_fun(Qn, np.log)))
    print(f"   the dual potential of (4.159)-(4.160), with g' = (f')^-1 and g(0) = 0, gives for f = lambda log lambda - lambda: g = e^u - 1 and (4.161) = {D_book:.6f}, which is (4.166) minus n = 3 ({D_vn(Pn, Qn) - 3:.6f});")
    print("   the normalisation of g has to be Legendre duality, g(f'(lambda)) = lambda f'(lambda) - f(lambda), i.e. g = e^u here: 'f(0) = 0 and g(0) = 0' cannot both hold when f'(0) = -infinity")
    # (u,v) matrix divergences: nonnegativity, Pythagoras
    worst, cnt = np.inf, 0
    for _ in range(600):
        A_, B_ = rand_pd(rng, 3), rand_pd(rng, 3)
        worst = min(worst, D_alpha_mat(A_, B_, 0.4))
        cnt += 1
    print(f"   (4.170) with the alpha-pair, alpha = 0.4: smallest D_alpha[P:Q] over {cnt} random pairs of 3x3 matrices = {worst:.4f} (positive); D[P:P] = {D_alpha_mat(Pn, Pn, 0.4):.1e}")
    a = 0.4
    Th = lambda M: 2 / (1 - a) * sym_fun(M, lambda w: w ** ((1 - a) / 2))
    He = lambda M: 2 / (1 + a) * sym_fun(M, lambda w: w ** ((1 + a) / 2))
    He_inv = lambda H: sym_fun(H, lambda w: (((1 + a) / 2) * w) ** (2 / (1 + a)))
    errs = []
    cnt = 0
    while cnt < 100:
        Pm, Qm = rand_pd(rng, 3), rand_pd(rng, 3)
        d = Th(Qm) - Th(Pm)
        V = rng.normal(size=(3, 3))
        V = (V + V.T) / 2
        V -= d * np.trace(d @ V) / np.trace(d @ d)
        HR = He(Qm) + 0.1 * V / np.abs(V).max() * 0.3
        if np.linalg.eigvalsh(HR).min() <= 0.05:
            continue
        Rm = He_inv(HR)
        errs.append(abs(D_alpha_mat(Pm, Rm, a) - D_alpha_mat(Pm, Qm, a) - D_alpha_mat(Qm, Rm, a)))
        cnt += 1
    print(f"   Pythagorean theorem for (u,v) matrices: the direction Theta = u(P) from P to Q orthogonal (trace pairing) to the direction H = v(.) from Q to R; 100 random triples: largest |D[P:R] - D[P:Q] - D[Q:R]| = {max(errs):.1e}")
    al, be = 0.7, 1.3
    Pd, Qd = np.diag([1.3, 2.1]), np.diag([0.8, 1.5])
    sc = sum((al * m ** (al + be) + be * n ** (al + be) - (al + be) * m ** al * n ** be) / (al * be * (al + be)) for m, n in zip(np.diag(Pd), np.diag(Qd)))
    print(f"   (4.176) against the trace of the scalar (4.133) on diagonal matrices, (alpha, beta) = ({al}, {be}): {D_ab_mat(Pd, Qd, al, be):.8f} against {sc:.8f}: ratio {D_ab_mat(Pd, Qd, al, be) / sc:.4f} = alpha beta = {al * be:.4f}")
    print("   ((4.176) uses Theta = P^alpha, H = P^beta, without the 1/alpha and 1/beta of (4.132), though it says it follows from (4.132))")


def numerical_riemann(g_fun, gamma_low_fun, x0, h=1e-5):
    """R_{ijk}^l by (5.66) from Gamma_{ij,k}(x) and g(x) with central differences."""
    def gam_up(x):
        g = g_fun(x)
        return np.einsum("lk,ijk->ijl", np.linalg.inv(g), gamma_low_fun(x))

    G0 = gam_up(x0)
    dG = np.zeros((3, 3, 3, 3))
    for m in range(3):
        e = np.zeros(3)
        e[m] = h
        dG[:, :, :, m] = (gam_up(x0 + e) - gam_up(x0 - e)) / (2 * h)       # dG[j,k,l,m] = d_m Gamma_{jk}^l
    R = np.zeros((3, 3, 3, 3))
    for i, j, k, l in itertools.product(range(3), repeat=4):
        R[i, j, k, l] = dG[j, k, l, i] - dG[i, k, l, j] + sum(G0[i, m, l] * G0[j, k, m] - G0[j, m, l] * G0[i, k, m] for m in range(3))
    return R


def check_logdet():
    head("5b. The (alpha,beta)-log-det divergences (4.5.3): invariance, limits, metric (4.186) and the connections they induce")
    P = np.array([[1.4, 0.3], [0.3, 0.9]])
    Q = np.array([[0.8, -0.2], [-0.2, 1.3]])
    L = np.array([[2.0, 0.5], [-0.3, 1.5]])
    print("   D_(alpha,beta)^(log-det)[P:Q] depends on P, Q only through the eigenvalues lambda of P Q^-1, which are invariant under congruence (4.181):")
    print(f"      eigenvalues of P Q^-1: {np.round(pq_eigs(P, Q), 8).tolist()}; of (L^T P L)(L^T Q L)^-1 for L = [[2, 0.5], [-0.3, 1.5]]: {np.round(pq_eigs(L.T @ P @ L, L.T @ Q @ L), 8).tolist()}; D_(0.5,1.5) before and after: {D_logdet(P, Q, 0.5, 1.5):.10f}, {D_logdet(L.T @ P @ L, L.T @ Q @ L, 0.5, 1.5):.10f}")
    e = 1e-5
    print(f"   limits (4.184)-(4.185): D_(1,0) = {D_logdet(P, Q, 1.0, 0.0):.8f} (the limit beta -> 0 evaluated at (1, {e}): {D_logdet(P, Q, 1.0, e):.8f}); D_(0,0) = {D_logdet(P, Q, 0.0, 0.0):.8f} = (1/2) sum (log lambda)^2 = {0.5 * np.sum(np.log(pq_eigs(P, Q)) ** 2):.8f}")
    print(f"   duality: D_(alpha,beta)[P:Q] = D_(beta,alpha)[Q:P]: (0.5, 1.5): {D_logdet(P, Q, 0.5, 1.5):.10f} = {D_logdet(Q, P, 1.5, 0.5):.10f}; alpha = beta is symmetric: D_(1,1)[P:Q] = {D_logdet(P, Q, 1.0, 1.0):.10f} = D_(1,1)[Q:P] = {D_logdet(Q, P, 1.0, 1.0):.10f}")
    to_mat = lambda x: np.array([[x[0], x[1]], [x[1], x[2]]])
    x0 = np.array([P[0, 0], P[0, 1], P[1, 1]])
    E = [np.array([[1, 0], [0, 0.0]]), np.array([[0, 1], [1, 0.0]]), np.array([[0, 0], [0, 1.0]])]
    Pi = np.linalg.inv(P)
    gpred = np.array([[np.trace(Pi @ E[i] @ Pi @ E[j]) for j in range(3)] for i in range(3)])
    res = {}
    pairs = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (0.5, 1.5), (0.3, 0.8)]
    for key in pairs:
        res[key] = induced_structure(lambda x, y, key=key: D_logdet(to_mat(x), to_mat(y), key[0], key[1]), x0, 3, 0.01)
    T10 = res[(1.0, 0.0)][2] - res[(1.0, 0.0)][1]
    print("   metric and cubic tensor induced through (6.22)-(6.25) at P (chart = the three entries of P, nested 5-point stencils, step 0.01):")
    print("      (alpha,beta)    |g - tr(P^-1 dP P^-1 dP)|   |g - (1/2) tr(...)|   T = k T^(1,0): k    residual    alpha - beta")
    trows = []
    for key in pairs:
        g, G, Gs = res[key]
        T = Gs - G
        k = float(np.sum(T * T10) / np.sum(T10 * T10))
        trows.append((key, k))
        print(f"      {str(key):14s}  {np.abs(g - gpred).max():.1e}                      {np.abs(g - 0.5 * gpred).max():.2e}              {k:+.6f}        {np.abs(T - k * T10).max():.1e}    {key[0] - key[1]:+.1f}")
    STORE["logdet_k"] = trows
    print("   the induced metric is tr(P^-1 dP P^-1 dP), the same for every (alpha,beta): twice the (1/2) tr(...) printed in (4.186), in the convention g = -d_i d'_j D of Chapter 1 and (6.22).")
    print("   (The printed value is the Fisher metric of N(0,P), whose KL has quadratic form (1/4) tr(...); D_(0,0) = (1/2) sum (log lambda)^2 is half the squared affine-invariant distance, which has metric tr(...).)")
    print("   And the cubic tensor, i.e. the dual pair of connections, is (alpha - beta) times that of (1,0): the connections depend on alpha and beta only through alpha - beta; this is visible in the expansion")
    print("   (1/(ab)) log((a e^(bt) + b e^(-at))/(a+b)) = t^2/2 + ((b-a)/6) t^3 + O(t^4) in t = log lambda, whose cubic coefficient is the only third-order datum.")
    g10, G10, Gs10 = res[(1.0, 0.0)]
    g01, G01, Gs01 = res[(0.0, 1.0)]
    print(f"   which chart is flat: for (1,0) Gamma* vanishes in the P chart ({np.abs(Gs10).max():.1e}, against {np.abs(G10).max():.2f} for Gamma) and for (0,1) Gamma vanishes ({np.abs(G01).max():.1e}); for (2,1) |Gamma*| = {np.abs(res[(2.0, 1.0)][2]).max():.1e}: the same flat pair as (1,0);")
    print(f"   for (0,0), Gamma = Gamma* (largest difference {np.abs(res[(0.0, 0.0)][1] - res[(0.0, 0.0)][2]).max():.1e}): the Levi-Civita connection of the affine-invariant metric")

    def g_fun(x):
        Pi_ = np.linalg.inv(to_mat(x))
        return np.array([[np.trace(Pi_ @ E[i] @ Pi_ @ E[j]) for j in range(3)] for i in range(3)])

    def dg_fun(x):
        Pi_ = np.linalg.inv(to_mat(x))
        d = np.zeros((3, 3, 3))
        for i, j, k in itertools.product(range(3), repeat=3):
            d[i, j, k] = -np.trace(Pi_ @ E[k] @ Pi_ @ E[i] @ Pi_ @ E[j]) - np.trace(Pi_ @ E[i] @ Pi_ @ E[k] @ Pi_ @ E[j])
        return d

    def gam_lc(x):
        d = dg_fun(x)                                                         # d[i,j,k] = d_k g_ij
        return 0.5 * (np.einsum("jki->ijk", d) + np.einsum("ikj->ijk", d) - d)

    gk = res[(0.3, 0.8)][1]
    kap = 0.3 - 0.8
    print(f"   Gamma^(kappa), kappa = alpha - beta, is (1 + kappa) times the Levi-Civita symbol in the P chart: for (0.3, 0.8) |Gamma_D - (1 + kappa) Gamma^LC| = {np.abs(gk - (1 + kap) * gam_lc(x0)).max():.1e} (|Gamma^LC| = {np.abs(gam_lc(x0)).max():.3f})")
    print("   curvature of Gamma^(kappa) from (5.66) (central differences of the exact symbols): max |R|, and its ratio to |1 - kappa^2| max |R^LC|:")
    RL = numerical_riemann(g_fun, gam_lc, x0)
    for kap in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        R = numerical_riemann(g_fun, lambda x, kap=kap: (1 + kap) * gam_lc(x), x0)
        if abs(1 - kap ** 2) > 1e-12:
            print(f"      kappa = {kap:+.1f}: max |R| = {np.abs(R).max():.4e}   ratio {np.abs(R).max() / (abs(1 - kap ** 2) * np.abs(RL).max()):.6f}")
        else:
            print(f"      kappa = {kap:+.1f}: max |R| = {np.abs(R).max():.1e} (flat)")
    xI = np.array([1.0, 0.0, 1.0])
    RI = numerical_riemann(g_fun, gam_lc, xI)
    gI = g_fun(xI)

    def sect(X, Y):
        RXY = np.einsum("ijkl,i,j,k->l", RI, X, Y, Y)
        return float((RXY @ gI @ X) / ((X @ gI @ X) * (Y @ gI @ Y) - (X @ gI @ Y) ** 2))

    print(f"   Levi-Civita sectional curvature at P = I: plane (E_11, E_22) of commuting directions {sect(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])):+.6f}; plane (E_11 - E_22, E_12 + E_21) of non-commuting directions {sect(np.array([1.0, 0, -1.0]), np.array([0, 1.0, 0])):+.6f}")
    print("   so D_(0,0) is not flat (it is Euclidean in the commuting directions and hyperbolic in the rotation directions), D_(alpha,beta) is flat exactly when |alpha - beta| = 1, and R^(kappa) = (1 - kappa^2) R^LC,")
    print("   the law 4s(1-s) of Chapter 5 with s = (1 - kappa)/2")
    # same flat geometry, different divergence: the minimiser is the geodesic projection only for the Bregman one
    Q0 = np.array([[0.8, -0.2], [-0.2, 1.3]])
    Q1 = np.array([[0.3, 0.2], [0.2, -0.4]])
    Qt = lambda t: Q0 + t * Q1

    def minimise(al, be):
        f = lambda t: D_logdet(P, Qt(t), al, be)
        ts = np.linspace(-1.0, 1.5, 5001)
        vals = np.array([f(t) if np.linalg.eigvalsh(Qt(t)).min() > 0.05 else np.inf for t in ts])
        t0 = ts[vals.argmin()]
        for _ in range(60):
            h = 1e-5
            t0 -= (f(t0 + h) - f(t0 - h)) / (2 * h) / ((f(t0 + h) - 2 * f(t0) + f(t0 - h)) / h ** 2)
        return t0

    def cos_orth(t):
        Q = Qt(t)
        Qi = np.linalg.inv(Q)
        dTh = np.linalg.inv(Q) - np.linalg.inv(P)                  # end tangent of the straight line in Theta = P^-1 from P to Q
        dP = -Q @ dTh @ Q
        gq = lambda X, Y: np.trace(Qi @ X @ Qi @ Y)
        return float(gq(dP, Q1) / math.sqrt(gq(dP, dP) * gq(Q1, Q1)))

    print("   minimising D_(alpha,beta)[P : Q(t)] over the segment Q(t) = Q + t E, Q as above and E = [[0.3, 0.2], [0.2, -0.4]]: cosine (affine-invariant metric) between the segment and the Theta-straight geodesic from P to the minimiser,")
    print("   the geodesic of the flat structure shared by (1,0) and every (alpha, beta) with alpha - beta = 1:")
    print("      " + ", ".join(f"({al}, {be}): t = {minimise(al, be):.5f}, cos = {cos_orth(minimise(al, be)):+.1e}" for al, be in ((1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0))))
    print("   so (1,0), a Bregman divergence, minimises exactly along the orthogonal geodesic (the projection theorem), while the others induce the same dual pair (g, Gamma, Gamma*) yet their minimisers are not")
    print("   the projections: the remark closing the chapter, that for a general divergence the minimiser is not the geodesic projection, in a sharper form (same geometry, different divergence)")
    d21 = lambda lam: np.log((2 * lam + lam ** (-2.0)) / 3) / 2
    h = 1e-3
    sec = lambda lam: (d21(lam + h) - 2 * d21(lam) + d21(lam - h)) / h ** 2
    print(f"   D_(2,1) induces the flat Gaussian structure but is not a Bregman divergence in P: its per-eigenvalue function d(lambda) = (1/2) log((2 lambda + lambda^-2)/3) has d'' = {sec(0.5):+.4f}, {sec(1.0):+.4f}, {sec(3.0):+.4f}, {sec(10.0):+.4f}")
    print("   at lambda = 0.5, 1, 3, 10: concave for large lambda, whereas a Bregman divergence in P is convex in P")


# ------------------------------------------------------------------ 6. miscellaneous divergences

def check_gamma():
    head("6. Miscellaneous divergences (section 4.6): gamma-divergence (4.187)-(4.189) and a robustness experiment")
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))

    def Dg(p, q, g):
        return float(np.log(np.sum(p ** g) * np.sum(q ** g) ** (g - 1) / np.sum(p * q ** (g - 1)) ** g) / (g * (g - 1)))

    print("   projective invariance (4.188): D_gamma[c1 p : c2 q] - D_gamma[p:q] for c1 = 2.5, c2 = 0.3, and D_gamma[p:p]: " + "; ".join(f"gamma = {g}: {Dg(2.5 * p, 0.3 * q, g) - Dg(p, q, g):.0e}, {Dg(p, p, g):.0e}" for g in (0.5, 1.5, 2.0, 3.0)))
    print(f"   gamma -> 1: D_gamma[p:q] at gamma = 1.001, 1.0001: {Dg(p, q, 1.001):.8f}, {Dg(p, q, 1.0001):.8f}; KL[p:q] = {float(np.sum(p * np.log(p / q))):.8f}")

    def mpow(M, a):
        return sym_fun(M, lambda w: w ** a)

    def Dg_mat(P, Q, g, printed):
        num = np.trace(mpow(P, g)) * ((np.trace(Q) ** g) if printed else np.trace(mpow(Q, g))) ** (g - 1)
        return float(np.log(num / np.trace(P @ mpow(Q, g - 1)) ** g) / (g * (g - 1)))

    Pm, Qm = rand_pd(rng, 3), rand_pd(rng, 3)
    print("   matrix version (4.189): D[P:P] as printed ((tr Q)^gamma raised to gamma - 1) and with tr(Q^gamma), and the corrected D[P:Q] before and after the scaling (1.7 P, 0.4 Q):")
    for g in (1.5, 2.0, 3.0):
        print(f"      gamma = {g}: printed D[P:P] = {Dg_mat(Pm, Pm, g, True):+.6f}, corrected D[P:P] = {Dg_mat(Pm, Pm, g, False):+.1e}, corrected D[P:Q] = {Dg_mat(Pm, Qm, g, False):.6f}, scaled {Dg_mat(1.7 * Pm, 0.4 * Qm, g, False):.6f}")
    print("   the matrix analogue of sum q_i^gamma is tr(Q^gamma); with (tr Q)^gamma as printed the divergence is not zero at P = Q")
    print("   robustness: n = 4000 draws from 0.9 N(0,1) + 0.1 N(m_out,1); the MLE (sample mean and standard deviation) against the gamma-estimator, gamma = 1.5, which minimises D_gamma[empirical : N(mu, s^2)]:")
    print("   weights w_i = f(x_i)^(gamma-1), mu = sum w x / sum w, s^2 = gamma sum w (x-mu)^2 / sum w (the estimating equations; fixed-point iteration started at the median and the MAD):")
    rr = np.random.default_rng(30)
    g = 1.5
    rows = []
    for mout in (4.0, 8.0, 16.0, 32.0):
        n = 4000
        z = rr.normal(size=n)
        out = rr.random(n) < 0.1
        x = np.where(out, mout + rr.normal(size=n), z)
        mu, s = float(np.median(x)), 1.4826 * float(np.median(np.abs(x - np.median(x))))
        for _ in range(200):
            w = np.exp(-(g - 1) * (x - mu) ** 2 / (2 * s * s)) / s ** (g - 1)
            mu_new = float(np.sum(w * x) / np.sum(w))
            s_new = math.sqrt(g * float(np.sum(w * (x - mu_new) ** 2) / np.sum(w)))
            done = abs(mu_new - mu) + abs(s_new - s) < 1e-12
            mu, s = mu_new, s_new
            if done:
                break
        rows.append((mout, float(x.mean()), float(x.std()), mu, s))
        print(f"      m_out = {mout:5.1f}: MLE mean {x.mean():+.4f}, sd {x.std():.4f};  gamma-estimate mean {mu:+.4f}, sd {s:.4f}")
    STORE["robust"] = rows
    print("   the MLE mean moves like 0.1 m_out without bound while the gamma-estimate stays near (0, 1) however far the outliers are: the robustness claimed in 4.6.1 (no efficiency comparison was made here)")


def JS_of(p, q):
    m = (p + q) / 2
    Hh = lambda a: float(-np.sum(a * np.log(a)))
    return Hh(m) - (Hh(p) + Hh(q)) / 2


def check_misc():
    head("6b. Zhang's (alpha,beta)-divergence (4.190), Furuichi's (4.192), Jensen-Shannon and Burbea-Rao (4.193)-(4.196), the (F,G)-connection (4.202), Theorem 4.20")
    xi = [0.3, 0.25]

    def D_zhang(p, q, al, be):
        a1, a2 = (1 - al) / 2, (1 + al) / 2
        s = (1 - be) / 2
        A_ = a1 * p + a2 * q
        M = (a1 * p ** s + a2 * q ** s) ** (1 / s)
        return 4 / (1 - al ** 2) * 2 / (1 + be) * float(np.sum(A_ - M))

    _, g0, T, Gl0 = s2_geometry(xi, 0.0)
    print("   Zhang's (4.190) at xi = (0.3, 0.25) on S_2: the induced metric, and the cubic tensor T^D = Gamma* - Gamma written as k T (T the Fisher cubic tensor):")
    rows = []
    for al, be in ((0.5, 0.7), (0.5, -0.5), (-0.3, 0.2), (0.8, 0.8), (0.5, 0.0), (0.0, 0.5), (0.5, 1 - 1e-6)):
        g, G, Gs = induced_structure(lambda x, y: D_zhang(pvec(x), pvec(y), al, be), xi, 2, 0.004)
        TD = Gs - G
        k = float(np.sum(TD * T) / np.sum(T * T))
        rows.append((al, be, k))
        print(f"      (alpha,beta) = ({al:+.1f}, {be:+.6f}): |g - Fisher| = {np.abs(g - g0).max():.1e}, k = {k:+.5f}, alpha beta = {al * be:+.5f}, |T^D - k T|/|T| = {np.abs(TD - k * T).max() / np.abs(T).max():.1e}")
    STORE["zhang"] = rows
    print("   so (4.190) induces the Fisher metric and the (alpha beta)-connection, not the alpha-connection: 'exactly the same as the alpha-geometry' holds at beta = 1, where (4.190) becomes (3.96)")
    rng = np.random.default_rng(31)

    def D_fur(p, q, al, be):
        return float(np.sum(p ** al * q ** (1 - al) - p ** be * q ** (1 - be)) / (al - be))

    print("   Furuichi's (4.192): fraction of 2000 random pairs of distributions on four outcomes with D < 0 (and the most negative value):")
    rows = []
    for al, be in ((0.8, 0.2), (0.9, 0.1), (0.5, 0.0), (0.3, -0.5), (1.0, 0.5), (2.0, 1.0)):
        neg = 0
        mn = 0.0
        for _ in range(2000):
            p = rng.dirichlet(np.ones(4))
            q = rng.dirichlet(np.ones(4))
            d = D_fur(p, q, al, be)
            if d < -1e-12:
                neg += 1
                mn = min(mn, d)
        rows.append((al, be, neg / 2000))
        print(f"      (alpha,beta) = ({al}, {be}): {neg / 2000:.3f} ({mn:.4f})")
    STORE["furuichi"] = rows
    print("   with c(t) = sum p^t q^(1-t) (log-convex, c(0) = c(1) = 1, below 1 in between, above 1 outside) D = (c(alpha) - c(beta))/(alpha - beta) is a difference quotient of c. For alpha > beta it is >= 0 when alpha >= 1 and")
    print("   beta >= 0 ((1, 0.5), and (2, 1), which is chi-squared), it is <= 0 for alpha <= 0 or for 0 = beta < alpha < 1, and it takes both signs for 0 < beta < alpha < 1; as printed it is not a divergence")
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))
    KL = lambda a, b: float(np.sum(a * np.log(a / b)))
    H = lambda a: float(-np.sum(a * np.log(a)))
    m = (p + q) / 2
    JS = H(m) - (H(p) + H(q)) / 2
    print(f"   Jensen-Shannon (4.194) = {JS:.12f}, (4.195) = {0.5 * (KL(p, m) + KL(q, m)):.12f}, symmetric: difference {abs(JS_of(p, q) - JS_of(q, p)):.1e}, bounded by log 2 = {math.log(2):.6f}")
    fJS = lambda u: 0.5 * (np.log(2 / (1 + u)) + u * np.log(2 * u / (1 + u)))
    h_ = 1e-3
    f2 = (fJS(1 + h_) - 2 * fJS(1.0) + fJS(1 - h_)) / h_ ** 2
    f3 = (fJS(1 + 2 * h_) - 2 * fJS(1 + h_) + 2 * fJS(1 - h_) - fJS(1 - 2 * h_)) / (2 * h_ ** 3)
    gj, Gj, Gsj = induced_structure(lambda x, y: JS_of(pvec(x), pvec(y)), xi, 2, 0.004)
    print(f"   Jensen-Shannon is an f-divergence, f(u) = (1/2)[log(2/(1+u)) + u log(2u/(1+u))]: sum p f(q/p) = {float(np.sum(p * fJS(q / p))):.12f}, f''(1) = {f2:.5f}, f'''(1) = {f3:.5f}; on S_2 the induced metric is {np.round((gj / g0).ravel(), 5).tolist()}")
    print(f"      times Fisher (entrywise ratios) and the cubic tensor is {np.abs(Gsj - Gj).max():.1e}: symmetric, hence self-dual, the Levi-Civita (alpha = 0) geometry of (1/4) Fisher; (6.36), alpha = 2 f''' + 3, needs f''(1) = 1: here 2 f''' + 3 f'' = {2 * f3 + 3 * f2:.1e}")
    F = lambda a: float(np.sum(a * np.log(a)))
    BR = lambda a, b, w: w * F(a) + (1 - w) * F(b) - F(w * a + (1 - w) * b)
    Breg = lambda a, b: float(F(a) - F(b) - np.sum((np.log(b) + 1) * (a - b)))
    print("   the skew Burbea-Rao divergence (4.196) with F = sum p log p ('alpha' is a mixture weight w in (0,1) here, not Amari's alpha): D^w_F[p:q]/(1-w) -> B_F[q:p] as w -> 1 and D^w_F[p:q]/w -> B_F[p:q] as w -> 0:")
    print("      " + ", ".join(f"w = {w}: {BR(p, q, w) / (1 - w):.5f}" for w in (0.5, 0.9, 0.99, 0.999)) + f"  against B_F[q:p] = {Breg(q, p):.5f};  " + ", ".join(f"w = {w}: {BR(p, q, w) / w:.5f}" for w in (0.5, 0.1, 0.01, 0.001)) + f"  against B_F[p:q] = {Breg(p, q):.5f}")
    print("   the (F,G)-connection (4.201)-(4.202) on S_2 with G(u) = u^0.3, F(u) = u^0.4 and H fixed by F'H' = G/u (Theorem 4.20); duality means d_k g_ij = Gamma^F_ki,j + Gamma^H_kj,i:")
    Gf = lambda u: u ** 0.3
    sF = 0.4
    Fp = lambda u: sF * u ** (sF - 1)
    Fpp = lambda u: sF * (sF - 1) * u ** (sF - 2)
    Hp = lambda u: Gf(u) / (u * Fp(u))
    Hpp = lambda u: (0.3 - sF) * u ** (0.3 - sF - 1) / sF

    def FG(xi_, Fp_, Fpp_, G_, printed):
        p = pvec(xi_)
        dl = A2 / p
        ddl = -np.einsum("ix,jx->ijx", A2, A2) / p ** 2
        g = np.einsum("x,ix,jx->ij", p * G_(p), dl, dl)
        coef = 1 + (Fpp_(p) / Fp_(p) if printed else p * Fpp_(p) / Fp_(p))
        Gam = np.einsum("x,ijx,kx->ijk", p * G_(p), ddl + coef * np.einsum("ix,jx->ijx", dl, dl), dl)
        return g, Gam

    xi_ = np.array(xi)
    h = 1e-5
    dg = np.zeros((2, 2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        dg[:, :, k] = (FG(xi_ + e, Fp, Fpp, Gf, False)[0] - FG(xi_ - e, Fp, Fpp, Gf, False)[0]) / (2 * h)
    for printed in (False, True):
        _, GF = FG(xi_, Fp, Fpp, Gf, printed)
        _, GH = FG(xi_, Hp, Hpp, Gf, printed)
        res = max(abs(dg[i, j, k] - GF[k, i, j] - GH[k, j, i]) for i, j, k in itertools.product(range(2), repeat=3))
        lab = "F''/F' as printed" if printed else "p F''/F' (corrected)"
        print(f"      coefficient {lab:22s}: largest duality residual {res:.2e}")
    for a in (0.5, -0.5):
        ss = (1 - a) / 2
        F1 = lambda u, ss=ss: ss * u ** (ss - 1)
        F2 = lambda u, ss=ss: ss * (ss - 1) * u ** (ss - 2)
        _, GA = FG(xi_, F1, F2, lambda u: np.ones_like(u), False)
        _, GP = FG(xi_, F1, F2, lambda u: np.ones_like(u), True)
        _, _, _, Gl = s2_geometry(xi, a)
        print(f"      G = 1, F = p^((1-a)/2), a = {a}: |Gamma^(F,G) - Gamma^(alpha)| = {np.abs(GA - Gl).max():.1e} with the corrected coefficient, {np.abs(GP - Gl).max():.2f} as printed")
    print("   (4.202) also prints d_j d_j l where d_i d_j l is meant; with the factor p the pair is dual exactly as Theorem 4.20 says and G = 1 gives the alpha-connection")


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
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{T(xlab)}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{T(ylab)}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{T(title)}</text>')

    def line(self, xs, ys, cls):
        keep = [(x, y) for x, y in zip(xs, ys) if self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9]
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in keep)
        self.b.append(f'<polyline class="{cls}" points="{pts}"/>')

    def dot(self, x, y, cls, r=4.5):
        if not (self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9):
            return
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{T(s)}</text>')


def T(s):
    """Tiny text markup for the figures: _x or _{xy} is a subscript, ^x or ^{xy} a superscript; <, > and & are escaped."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = re.split(r"([_^])(?:\{([^}]*)\}|(.))", s)
    out, cur, i = [], 0.0, 0
    # re.split with three groups yields: text, kind, braced, single, text, ...
    while i < len(parts):
        txt = parts[i]
        if txt:
            if cur != 0.0:
                dxs = ' dx="3"' if txt.startswith(" ") else ""
                out.append(f'<tspan{dxs} dy="{-cur:g}">{txt.lstrip(" ")}</tspan>'); cur = 0.0
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





def fig_alpha_geodesics(path):
    b = []
    W, H = 800, 480
    V = [np.array([225.0, 85.0]), np.array([75.0, 335.0]), np.array([375.0, 335.0])]       # outcomes 0, 1, 2

    def xy(p):
        return p[0] * V[0] + p[1] * V[1] + p[2] * V[2]

    b.append(f'<text class="hd" x="60" y="52">{T("The α-geodesic of S_2 from p to q")}</text>')
    for t in (0.25, 0.5, 0.75):
        for k in range(3):
            o = [j for j in range(3) if j != k]
            a = np.zeros(3)
            a[k] = t
            a[o[0]] = 1 - t
            c = np.zeros(3)
            c[k] = t
            c[o[1]] = 1 - t
            poly(b, [tuple(xy(a)), tuple(xy(c))], "gd")
    poly(b, [tuple(v) for v in V] + [tuple(V[0])], "ax")
    b.append(f'<text class="sm" x="{V[0][0]:.0f}" y="{V[0][1] - 9:.0f}" text-anchor="middle">{T("p_0 = 1")}</text>')
    b.append(f'<text class="sm" x="{V[1][0]:.0f}" y="{V[1][1] + 16:.0f}" text-anchor="middle">{T("p_1 = 1")}</text>')
    b.append(f'<text class="sm" x="{V[2][0]:.0f}" y="{V[2][1] + 16:.0f}" text-anchor="middle">{T("p_2 = 1")}</text>')
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    for al, cls in ((-1.0, "s1"), (0.0, "s3"), (1.0, "s2"), (3.0, "s4")):
        pts = [tuple(xy(normalised_path(p, q, al, s)[0])) for s in np.linspace(0, 1, 101)]
        poly(b, pts, f"ln {cls}")
    for pt, name, dx, dy in ((p, "p", -14, -4), (q, "q", 10, 14)):
        x, y = xy(pt)
        b.append(f'<circle class="f0 ring" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        b.append(f'<text class="v" x="{x + dx:.1f}" y="{y + dy:.1f}">{name}</text>')
    legend(b, 60, 385, [("s1", "α = −1: straight (mixture)"), ("s3", "α = 0"), ("s2", "α = +1: exponential"), ("s4", "α = 3")], col=190)
    P2 = Panel(b, 480, 85, 290, 235, (-3.2, 3.2), (-2.35, 0.4))
    P2.frame([-3, -2, -1, 0, 1, 2, 3], [-2, -1.5, -1, -0.5, 0], "α", "curvature K of the α-connection", "Curvature of the α-connection of S_2", True)
    al = np.linspace(-3.2, 3.2, 200)
    P2.line(al, (1 - al ** 2) / 4, "ln s1")
    P2.line([-3.2, 3.2], [0, 0], "dash s0")
    for a, K in STORE["K_curve"]:
        P2.dot(a, K, "f2", 4.0)
    P2.text(-1.0, 0.0, "flat", "sm", "middle", 0, -9)
    P2.text(1.0, 0.0, "flat", "sm", "middle", 0, -9)
    P2.text(0.1, -2.0, "K = (1−α²)/4", "sm", "start")
    m = STORE["mass_half"]
    note(b, 60, 432, ["Left: the normalised curves (4.27) for p = (0.6, 0.3, 0.1), q = (0.1, 0.3, 0.6). The un-normalised curves (4.22) of R^{3}_{+}",
                      f"have total mass {m[-1.0]:.3f}, {m[0.0]:.3f}, {m[1.0]:.3f}, {m[3.0]:.3f} at t = ½ for α = −1, 0, 1, 3: only the mixture stays on the simplex.",
                      "Right: the curvature measured through (5.66) (dots, at one point) against (1−α²)/4."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "α-geodesics and curvature on the three-outcome simplex",
        "Left: the equilateral triangle of distributions on three outcomes with the normalised α-geodesics from p = (0.6, 0.3, 0.1) to q = (0.1, 0.3, 0.6) for α = −1 (a straight line), 0, 1 and 3, bending progressively. "
        "Right: the curvature of the α-connection of the simplex, measured numerically at one point, lies on the parabola (1 minus alpha squared) over 4 and is zero only at alpha = plus or minus 1.", b))


def fig_kurose(path):
    b = []
    W, H = 800, 415
    rows = STORE["kurose"]
    P1 = Panel(b, 70, 60, 310, 240, (-0.045, 0.045), (-0.045, 0.045))
    P1.frame([-0.04, -0.02, 0, 0.02, 0.04], [-0.04, -0.02, 0, 0.02, 0.04], "predicted correction (1−α²)/4 · D[p:q] D[q:r] / D[p:r]", "(D[p:r] − D[p:q] − D[q:r]) / D[p:r]", "Plain Pythagoras: measured miss", True)
    P1.line([-0.045, 0.045], [0.045, -0.045], "dash s0")
    for a, pts in rows:
        if abs(abs(a) - 1) < 1e-9:
            continue
        cls = "f1" if abs(a) < 1 else "f2"
        for Dpr, plain, form, Dpq, Dqr in pts[::2]:
            x = (1 - a * a) / 4 * Dpq * Dqr / Dpr
            y = (Dpr - plain) / Dpr
            P1.dot(x, y, cls, 2.0)
    legend_col(b, 262, 82, [("s1", "|α| < 1  (K > 0)"), ("s2", "|α| > 1  (K < 0)"), ("dash s0", "y = −x")])
    P2 = Panel(b, 470, 60, 300, 240, (-3.3, 3.3), (-22, 0.5))
    P2.frame([-3, -2, -1, 0, 1, 2, 3], [-16, -12, -8, -4, 0], "α", "log₁₀ of the largest error over 200 triples", "Error of the plain sum and of (4.29)", True)
    al = [a for a, _ in rows]
    e_plain = [max(abs(Dpr - plain) for Dpr, plain, form, _, _ in pts) for _, pts in rows]
    e_form = [max(abs(Dpr - form) for Dpr, plain, form, _, _ in pts) for _, pts in rows]
    lg = lambda v: math.log10(max(v, 1e-16))
    P2.line(al, [lg(v) for v in e_plain], "ln s2")
    P2.line(al, [lg(v) for v in e_form], "ln s1")
    for a, v, w in zip(al, e_plain, e_form):
        P2.dot(a, lg(v), "f2", 3.8)
        P2.dot(a, lg(w), "f1", 3.8)
    legend_col(b, 520, 262, [("s2", "D[p:r] − D[p:q] − D[q:r]"), ("s1", "D[p:r] − (4.29)")])
    note(b, 70, 372, ["Left: every random triple (the α-geodesic p→q orthogonal to the −α-geodesic q→r) lies on the line y = −x.",
                      "Right: with the correction the error is rounding (10⁻¹⁵); without it the miss reaches about 10⁻¹ for α = 3.",
                      "At α = ±1 both are exact (the floor of the plot is 10⁻¹⁶)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Kurose's Pythagorean theorem (4.29), checked",
        "Left: for random triples on the three-outcome simplex with orthogonal geodesics, the relative miss of the plain Pythagorean identity equals minus the predicted correction (one minus alpha squared) over 4 times D[p:q] D[q:r] / D[p:r], so all points lie on the line y = minus x. "
        "Right: the largest error of the plain sum and of the corrected formula against alpha, on a log scale: the corrected formula is exact to rounding, the plain sum is off except at alpha = plus or minus 1.", b))


def fig_means_apportion(path):
    b = []
    W, H = 800, 425
    P1 = Panel(b, 70, 70, 300, 230, (-6, 6), (0.5, 4.5))
    P1.frame([-6, -3, 0, 3, 6], [1, 2, 3, 4], "α", "m_α(1, 4)", "The α-mean of 1 and 4 (cf. Fig. 4.1)", True)
    P1.line([-6, 6], [4, 4], "dash s0")
    P1.line([-6, 6], [1, 1], "dash s0")
    cur = STORE["amean_curve"]
    P1.line([a for a, _ in cur], [v for _, v in cur], "ln s1")
    am = STORE["amean"]
    for a, lab, dx, dy in ((-3.0, "2.92", 6, -8), (-1.0, "2.5 arithmetic", 6, -8), (0.0, "2.25", 6, -8), (1.0, "2 geometric", 6, -8), (3.0, "1.6 harmonic", 6, -8)):
        P1.dot(a, am[a], "f2", 4.0)
        P1.text(a, am[a], lab, "sm", "start", dx, dy)
    P1.text(-5.9, 4.0, "max (α → −∞)", "sm", "start", 0, -5)
    P1.text(5.9, 1.0, "min (α → +∞)", "sm", "end", 0, -5)
    x0, y0, w, h = 450, 88, 320, 200
    segs = STORE["apportion_segments"]
    g = lambda a: math.asinh(a / 3)
    lo, hi = g(segs[0][0]), g(segs[-1][1])
    X = lambda a: x0 + (g(a) - lo) / (hi - lo) * w
    b.append(f'<text class="hd" x="{x0}" y="{y0 - 38}">Optimal seats as α varies</text>')
    rowh = h / 4
    pops = [1.3, 6.6, 21.1, 6.8]
    for i, pop in enumerate(pops):
        b.append(f'<text class="sm" x="{x0 - 8}" y="{y0 + rowh * i + rowh / 2 + 4:.1f}" text-anchor="end">pop {pop}</text>')
    for s0, s1, al in segs:
        for i, nseat in enumerate(al):
            op = 0.10 + 0.13 * nseat
            b.append(f'<rect class="f1" style="opacity:{op:.2f}" x="{X(s0):.1f}" y="{y0 + rowh * i:.1f}" width="{X(s1) - X(s0):.1f}" height="{rowh - 2:.1f}"/>')
            b.append(f'<text class="v" x="{(X(s0) + X(s1)) / 2:.1f}" y="{y0 + rowh * i + rowh / 2 + 4:.1f}" text-anchor="middle">{nseat}</text>')
    for a in (-60, -10, -3, 0, 3, 10, 60):
        b.append(f'<line class="ax" x1="{X(a):.1f}" y1="{y0 + h}" x2="{X(a):.1f}" y2="{y0 + h + 4}"/>')
        b.append(f'<text class="sm" x="{X(a):.1f}" y="{y0 + h + 17}" text-anchor="middle">{a}</text>')
    b.append(f'<text class="lab" x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">α  (axis: asinh(α/3))</text>')
    for a, lab, yy in ((-3, "Hill", 24), (-1, "T–S", 10), (3, "Webster", 10)):
        b.append(f'<line class="s2" style="stroke-width:1;stroke-dasharray:3 3" x1="{X(a):.1f}" y1="{y0 - yy + 2}" x2="{X(a):.1f}" y2="{y0 - 1}"/>')
        b.append(f'<text class="sm" x="{X(a):.1f}" y="{y0 - yy}" text-anchor="middle">{lab}</text>')
    b.append(f'<text class="sm" x="{X(segs[0][0]) + 2:.1f}" y="{y0 + h + 52}" text-anchor="start">← Adams (α → −∞)</text>')
    b.append(f'<text class="sm" x="{X(segs[-1][1]):.1f}" y="{y0 + h + 52}" text-anchor="end">Jefferson (α → +∞) →</text>')
    note(b, 70, 392, ["Left: the α-means run from the maximum to the minimum. Right: nine seats for four states of populations 1.3, 6.6,",
                      "21.1, 6.8; the integer allocation minimising D_α[p:q]. Hill, T–S (Theil–Schrage) and Webster are the rules at α = −3, −1, 3."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "α-means and seat apportionment",
        "Left: the α-mean of 1 and 4 decreases from 4 (the maximum) at alpha minus infinity through 2.5 (arithmetic), 2.25, 2 (geometric) and 1.6 (harmonic) to 1 (the minimum) at alpha plus infinity. "
        "Right: for nine seats and four states with populations 1.3, 6.6, 21.1 and 6.8, the integer allocation that minimises the alpha-divergence changes from (1,2,4,2) for alpha below minus 4.2 to (1,1,5,2) up to 0.27, to (0,2,5,2) up to 34, and to (0,1,6,2) beyond.", b))


def fig_tsallis(path):
    b = []
    W, H = 800, 415
    P1 = Panel(b, 70, 60, 310, 240, (-7, 7), (-7, 2.2))
    P1.frame([-6, -3, 0, 3, 6], [-6, -4, -2, 0], "x", "log₁₀ p(x)", "q-Gaussians with log_q p = −x²/2 − c", True)
    xs = np.linspace(-7, 7, 700)
    qg = dict(STORE["qgauss"])
    for q, cls in ((0.5, "s2"), (1.0, "s0"), (1.5, "s1")):
        dens = exp_q(-xs ** 2 / 2 - qg[q], q)
        keep = dens > 1e-7
        P1.line(xs[keep], np.log10(dens[keep]), f"ln {cls}")
    legend(b, 80, 80, [("s2", "q = 0.5: compact support"), ("s0", "q = 1: Gaussian"), ("s1", "q = 1.5: power-law tail")], col=160)
    P2 = Panel(b, 470, 60, 300, 240, (0, 26), (-10, 0.5))
    P2.frame([1, 5, 10, 15, 20, 25], [-8, -6, -4, -2, 0], "random point of S_3", "log₁₀ of (λmax − λmin) / λmax", "Is the q-metric conformal to Fisher?", True)
    for tag, cls in (("power", "f1"), ("kappa", "f2")):
        k = 0
        for t, e0, e1 in STORE["conformal_scatter"]:
            if t != tag:
                continue
            k += 1
            P2.dot(k, math.log10(max((e1 - e0) / e1, 1e-8)), cls, 3.6)
    legend_col(b, 500, 262, [("s1", "power χ = s^q (the q-escort)"), ("s2", "κ-exponential χ")])
    note(b, 70, 375, ["Left: exp_q of a quadratic has finite support for q < 1 and a power-law tail for q > 1.",
                      "Right: eigenvalues of g_F⁻¹ Hess ψ_χ (escort metric against Fisher): equal for a power χ (the 10⁻⁶ spread is",
                      "finite-difference noise), unequal for the κ-exponential: only the q-escort geometry is conformal (Theorem 4.14)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Tsallis q-Gaussians and conformal flatness of the q-escort",
        "Left: logarithm of the q-Gaussian density for q = 0.5 (compact support), q = 1 (Gaussian, a parabola on this scale) and q = 1.5 (power-law tail). "
        "Right: relative spread of the eigenvalues of the ratio of the escort metric to the Fisher metric at 25 random points of the simplex: about 1e-6 (noise) for a power function chi, up to 0.2 for the kappa-exponential.", b))


def fig_matrices(path):
    b = []
    W, H = 800, 415
    P1 = Panel(b, 70, 60, 310, 240, (-2.2, 3.2), (-4.6, 6.6))
    P1.frame([-2, -1, 0, 1, 2, 3], [-4, -2, 0, 2, 4, 6], "log₂ c  (P → cP, Q → cQ)", "log₂ of D[cP:cQ] / D[P:Q]", "Scaling: only the log-det family is invariant", True)
    cs = np.log2(STORE["scale_cs"])
    groups = [("s1", ("KL / Stein (4.151)", "log-det (0.5, 1.5) (4.182)")), ("s3", ("von Neumann (4.166)", "alpha = 0 (4.179)")), ("s2", ("Frobenius (4.163)", "(alpha,beta) = (1.3, 0.7) (4.176)"))]
    for cls, names in groups:
        for k, name in enumerate(names):
            pts = [(P1.X(x), P1.Y(math.log2(v))) for x, v in zip(cs, STORE["scale_curves"][name])]
            poly(b, pts, f"ln {cls}", "stroke-width:7;opacity:.30" if k == 0 else "stroke-width:2")
            for x, v in zip(cs, STORE["scale_curves"][name]):
                if k == 1:
                    P1.dot(x, math.log2(v), "f" + cls[1], 3.0)
    legend_col(b, 84, 82, [("s1", "KL (Stein) and log-det: slope 0"), ("s3", "von Neumann and α = 0: slope 1"), ("s2", "Frobenius and (α,β): slope 2")])
    P2 = Panel(b, 470, 60, 300, 240, (-1.7, 1.7), (-0.36, 0.36))
    P2.frame([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], "t = log λ", "(d(e^t) − t²/2) / t³", "Third-order term of the log-det family", True)
    t = np.linspace(-1.6, 1.6, 160)
    t = t[np.abs(t) > 1e-9]

    def dfun(al, be, tt):
        if abs(al) < 1e-12 and abs(be) < 1e-12:
            return tt ** 2 / 2
        if abs(be) < 1e-12:
            return (np.exp(-al * tt) + al * tt - 1) / al ** 2
        if abs(al) < 1e-12:
            return (np.exp(be * tt) - be * tt - 1) / be ** 2
        return np.log((al * np.exp(be * tt) + be * np.exp(-al * tt)) / (al + be)) / (al * be)

    for (al, be), cls in (((1.0, 0.0), "s1"), ((2.0, 1.0), "s4"), ((0.0, 1.0), "s2"), ((0.0, 0.0), "s3"), ((1.0, 1.0), "s0")):
        P2.line(t, (dfun(al, be, t) - t ** 2 / 2) / t ** 3, f"ln {cls}")
    P2.dot(0, -1 / 6, "f1", 4.0)
    P2.dot(0, 1 / 6, "f2", 4.0)
    P2.text(-0.08, -1 / 6, "−1/6", "sm", "end", 0, 4)
    P2.text(-0.08, 1 / 6, "+1/6", "sm", "end", 0, -5)
    legend(b, 478, 82, [("s1", "(α,β) = (1, 0)"), ("s4", "(2, 1)"), ("s2", "(0, 1)"), ("s3", "(0, 0)"), ("s0", "(1, 1)")], col=100, lh=16)
    note(b, 70, 375, ["Left: D[cP:cQ]/D[P:Q] for random positive-definite P, Q: Stein and the log-det divergences do not change,",
                      "the other (u,v)-divergences scale like c or c² (and change under congruence).",
                      "Right: the log-det divergence is t²/2 + ((β−α)/6) t³ + … in t = log λ: (1,0) and (2,1) agree at t = 0 only."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Invariance under scaling and the third-order term of the log-det family",
        "Left: logarithm base 2 of D[cP:cQ] over D[P:Q] against log2 of c for six matrix divergences: lines of slope 0 for the Stein divergence and the log-det divergence, slope 1 for the von Neumann and alpha = 0 divergences and slope 2 for the Frobenius and (alpha,beta) divergences. "
        "Right: the third-order coefficient of the per-eigenvalue function of the (alpha,beta)-log-det divergence tends to (beta minus alpha) over 6 as t tends to 0, equal to minus one sixth for (1,0) and (2,1), plus one sixth for (0,1) and zero for (0,0) and (1,1).", b))


def fig_flat_invariant(path):
    b = []
    W, H = 800, 415
    P1 = Panel(b, 70, 60, 310, 240, (-2, 2), (-0.012, 0.056))
    P1.frame([-2, -1, 0, 1, 2], [0, 0.02, 0.04], "α", "s_3 / s_1", "Is D_α Bregman on S_2? third singular value", True)
    cur = STORE["rank_curve"]
    P1.line([a for a, _ in cur], [v for _, v in cur], "ln s1")
    P1.dot(-1, 0, "f2", 4.5)
    P1.dot(1, 0, "f2", 4.5)
    P1.text(-1, 0.0, "rank 2: KL", "sm", "middle", 0, 19)
    P1.text(1, 0.0, "rank 2: dual KL", "sm", "middle", 0, 19)
    P1.text(0, 0.047, "rank 3 for every other α", "sm", "middle", 0, -8)
    mono = STORE["escort_mono"]
    wb, wf, wa = STORE["monotone_beta"]
    cats = [("D_α, α = 0.5", wa), ("escort, q = 1.5", mono[1.5][0]), ("β-divergence, β = 2", wb[2.0]), ("Euclidean (β = 1)", wf), ("β-divergence, β = 0.5", wb[0.5]), ("escort, q = 0.8", mono[0.8][0]), ("escort, q = 0.5", mono[0.5][0])]
    x0, y0, w, h = 570, 78, 200, 210
    xr = (-0.02, 0.3)
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w
    b.append(f'<text class="hd" x="470" y="{y0 - 18}">Largest D[Tp:Tr] − D[p:r] (T merges two outcomes)</text>')
    rowh = h / len(cats)
    for i, (lab, v) in enumerate(cats):
        yy = y0 + rowh * i
        b.append(f'<text class="sm" x="{x0 - 8}" y="{yy + rowh / 2 + 4:.1f}" text-anchor="end">{T(lab)}</text>')
        vv = max(v, 0.0)
        cls = "f1" if v <= 1e-6 else "f2"
        b.append(f'<rect class="{cls}" style="opacity:0.85" x="{X(0):.1f}" y="{yy + 3:.1f}" width="{max(X(vv) - X(0), 1.5):.1f}" height="{rowh - 6:.1f}"/>')
        b.append(f'<text class="v" x="{X(vv) + 5:.1f}" y="{yy + rowh / 2 + 4:.1f}">{"≤ 0" if v <= 1e-6 else f"+{v:.3f}"}</text>')
    b.append(f'<line class="ax" x1="{X(0):.1f}" y1="{y0 - 2}" x2="{X(0):.1f}" y2="{y0 + h + 2}"/>')
    for v in (0, 0.1, 0.2, 0.3):
        b.append(f'<text class="sm" x="{X(v):.1f}" y="{y0 + h + 16}" text-anchor="middle">{v:g}</text>')
    note(b, 70, 375, ["Left: the mixed derivative of D_α stacked over 14 × 14 points of S_2 has rank 2 for α = ±1 (as any Bregman",
                      "divergence must, in any coordinates) and rank 3 otherwise. Right: invariance means D[Tp:Tr] ≤ D[p:r];",
                      "the other flat divergences (Euclidean, β, q-escort) violate it, by up to about 0.27 here."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Flat versus invariant on the simplex",
        "Left: the ratio of the third to the first singular value of the stacked mixed derivative of the alpha-divergence on the three-outcome simplex against alpha: it vanishes only at alpha equal to minus one and plus one, where the divergence is Bregman (rank 2). "
        "Right: the largest violation of information monotonicity under merging two outcomes among 20000 random pairs: none for the alpha-divergence, and about 0.2 to 0.27 for the Euclidean divergence, the beta-divergences and the q-escort divergence at q = 0.5 and 0.8 (no violation found for the escort divergence at q = 1.5).", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_alpha_geodesics(out / "alpha-geodesics.svg")
    fig_kurose(out / "kurose-pythagoras.svg")
    fig_means_apportion(out / "means-apportionment.svg")
    fig_tsallis(out / "tsallis.svg")
    fig_matrices(out / "matrices.svg")
    fig_flat_invariant(out / "flat-vs-invariant.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


def main():
    check_bregman()
    check_converse()
    check_simplex()
    check_alpha_rn()
    check_geodesic_sn()
    check_kurose()
    check_apportionment()
    check_alpha_mean()
    check_integration()
    check_tsallis_basic()
    check_q_family()
    check_qmaxent()
    check_chi()
    check_qgauss()
    check_uv()
    check_matrices()
    check_logdet()
    check_gamma()
    check_misc()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
