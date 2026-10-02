#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 5, checked by hand.

Elements of differential geometry. Every number quoted in the notes comes from here. The three running examples are
the plane in polar coordinates (flat, but with non-zero Christoffel symbols), the unit sphere (positively curved), and the
Gaussian manifold of Chapters 1 and 2 with its Fisher metric (negatively curved), on which three different affine
connections can be written down: the e-connection (flat in theta), the m-connection (flat in eta) and Levi-Civita.

Convention, as in the book: Gamma_{ij}^k are the components of nabla_{e_i} e_j = Gamma_{ij}^k e_k, so the first index is the
direction of differentiation. In the code Gamma[i, j, k] stores Gamma_{ij}^k, and R[i, j, k, l] stores R_{ijk}^l.

Checked here, in the order the notes use them:

  1. tangent vectors as velocities, derivative operators and score functions; the transformation laws of basis vectors and components;
  2. the metric: transformation law, the Fisher information as an inner product of scores, Euclidean versus curved;
  3. connections: Christoffel symbols of the three examples, three ways (analytic, from the metric, by projection / the transformation law);
     the Gaussian e-, m- and Levi-Civita connections and the identity Levi-Civita = (e + m)/2;
  4. tensors and non-tensors: the cubic tensor, the Hessian, the Christoffel symbols themselves;
  5. covariant derivative of a constant field in polar coordinates;
  6. geodesics: great circles, the same start and velocity under three connections, lengths, shortest path;
  7. parallel transport and holonomy on the sphere, in the plane, on the Gaussian manifold;
  8. curvature: the Riemann tensor, Gauss curvature of each example, the round-the-world loop (including the factor 1/2 in (5.65)/(5.69)),
     and non-commutativity of the covariant derivative (5.74);
  9. the Levi-Civita connection: metric compatibility, uniqueness formula (5.85), the minimal-length property, and duality of e and m (a look ahead);
 10. submanifolds: induced metric, induced connection, embedding curvature; the cylinder (flat inside, curved outside), the sphere, a curve of Gaussians.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only.

Run:  python3 diffgeo.py            (checks)
      python3 diffgeo.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def head(s):
    print("\n" + s)


# ------------------------------------------------------------------ the three examples

def metric_sphere(x):
    return np.diag([1.0, math.sin(x[0]) ** 2])                       # x = (theta, phi)


def metric_polar(x):
    return np.diag([1.0, x[0] ** 2])                                  # x = (r, theta)


def metric_gauss(x):
    return np.diag([1 / x[1] ** 2, 2 / x[1] ** 2])                    # x = (mu, sigma): the Fisher metric


def gamma_sphere(x):
    th = x[0]; G = np.zeros((2, 2, 2))
    G[1, 1, 0] = -math.sin(th) * math.cos(th)                           # Gamma_{phi phi}^theta
    G[0, 1, 1] = G[1, 0, 1] = math.cos(th) / math.sin(th)               # Gamma_{theta phi}^phi
    return G


def gamma_polar(x):
    r = x[0]; G = np.zeros((2, 2, 2))
    G[1, 1, 0] = -r                                                     # Gamma_{theta theta}^r
    G[0, 1, 1] = G[1, 0, 1] = 1 / r                                     # Gamma_{r theta}^theta
    return G


def gamma_lc_gauss(x):
    s = x[1]; G = np.zeros((2, 2, 2))
    G[0, 1, 0] = G[1, 0, 0] = -1 / s                                    # Gamma_{mu sigma}^mu
    G[0, 0, 1] = 1 / (2 * s)                                            # Gamma_{mu mu}^sigma
    G[1, 1, 1] = -1 / s                                                 # Gamma_{sigma sigma}^sigma
    return G


def gamma_e_gauss(x):
    s = x[1]; G = np.zeros((2, 2, 2))
    G[0, 1, 0] = G[1, 0, 0] = -2 / s
    G[1, 1, 1] = -3 / s
    return G


def gamma_m_gauss(x):
    s = x[1]; G = np.zeros((2, 2, 2))
    G[0, 0, 1] = 1 / s
    G[1, 1, 1] = 1 / s
    return G


def christoffel_from_metric(metric, x, h=1e-5):
    """Gamma_{ij}^k = (1/2) g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij}), derivatives by central differences."""
    n = len(x); gi = np.linalg.inv(metric(x)); dg = np.zeros((n, n, n))
    for l in range(n):
        e = np.zeros(n); e[l] = h
        dg[l] = (metric(x + e) - metric(x - e)) / (2 * h)               # dg[l, i, j] = d_l g_ij
    G = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                G[i, j, k] = 0.5 * sum(gi[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j]) for l in range(n))
    return G


def riemann(gamma_fn, x, h=1e-5):
    """R_{ijk}^l = d_i Gamma_{jk}^l - d_j Gamma_{ik}^l + Gamma_{im}^l Gamma_{jk}^m - Gamma_{jm}^l Gamma_{ik}^m  (5.66)."""
    n = len(x); G = gamma_fn(x); dG = np.zeros((n, n, n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        dG[i] = (gamma_fn(x + e) - gamma_fn(x - e)) / (2 * h)
    R = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    R[i, j, k, l] = dG[i][j, k, l] - dG[j][i, k, l] + sum(G[i, m, l] * G[j, k, m] - G[j, m, l] * G[i, k, m] for m in range(n))
    return R


def gauss_curvature(R, g):
    """Gauss curvature of a surface from R(e_1, e_2) e_2 = R_{122}^m e_m:  K = g_{1m} R_{122}^m / det g."""
    return float(sum(g[0, m] * R[0, 1, 1, m] for m in range(2)) / np.linalg.det(g))


def geodesic(gamma_fn, x0, v0, T, n=4000):
    """RK4 for  x'' = -Gamma_{ij}^k x'^i x'^j  (5.54); returns the states (x, x') at n + 1 equally spaced times in [0, T]."""
    def rhs(s):
        x, v = s[:2], s[2:]
        G = gamma_fn(x)
        return np.concatenate([v, -np.einsum("ijk,i,j->k", G, v, v)])
    s = np.concatenate([x0, v0]).astype(float); out = [s.copy()]; dt = T / n
    for _ in range(n):
        k1 = rhs(s); k2 = rhs(s + dt / 2 * k1); k3 = rhs(s + dt / 2 * k2); k4 = rhs(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(s.copy())
    return np.array(out)


def transport(gamma_fn, waypoints, A0, steps=400, record=False):
    """Parallel transport (5.58)  dA^i/dt = -Gamma_{jk}^i xdot^j A^k  along straight coordinate segments between the waypoints.
    Returns the final vector, or with record=True the list of (point, vector) along the whole path."""
    A = np.array(A0, float); path = [(np.array(waypoints[0], float), A.copy())]
    for p, q in zip(waypoints[:-1], waypoints[1:]):
        p, q = np.array(p, float), np.array(q, float); xd = q - p
        def rhs(t, A_):
            G = gamma_fn(p + t * xd)
            return -np.einsum("jki,j,k->i", G, xd, A_)
        dt = 1.0 / steps; t = 0.0
        for _ in range(steps):
            k1 = rhs(t, A); k2 = rhs(t + dt / 2, A + dt / 2 * k1); k3 = rhs(t + dt / 2, A + dt / 2 * k2); k4 = rhs(t + dt, A + dt * k3)
            A = A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
            if record: path.append((p + t * xd, A.copy()))
    return path if record else A


def angle_between(g, A, B):
    """Signed angle from A to B in the metric g (positive from the first coordinate axis towards the second)."""
    c = A @ g @ B; s = math.sqrt(np.linalg.det(g)) * (A[0] * B[1] - A[1] * B[0])
    return math.atan2(s, c)



# ------------------------------------------------------------------ numerical calculus and the Gaussian charts

def jac(f, x, h=1e-6):
    """J[a, i] = d f^a / d x^i by central differences."""
    x = np.asarray(x, float); n = len(x); J = None
    for i in range(n):
        e = np.zeros(n); e[i] = h
        col = (np.asarray(f(x + e), float) - np.asarray(f(x - e), float)) / (2 * h)
        if J is None: J = np.zeros((len(col), n))
        J[:, i] = col
    return J


def hess_map(f, x, h=1e-4):
    """H[a, i, j] = d^2 f^a / d x^i d x^j by central differences."""
    x = np.asarray(x, float); n = len(x); H = None
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
            v = (np.atleast_1d(f(x + ei + ej)) - np.atleast_1d(f(x + ei - ej)) - np.atleast_1d(f(x - ei + ej)) + np.atleast_1d(f(x - ei - ej))) / (4 * h * h)
            if H is None: H = np.zeros((len(v), n, n))
            H[:, i, j] = v
    return H


def theta_of(xi):                     # (mu, sigma) -> natural parameters
    m, s = xi
    return np.array([m / s ** 2, -1 / (2 * s ** 2)])


def xi_of_theta(th):
    s2 = -1 / (2 * th[1])
    return np.array([th[0] * s2, math.sqrt(s2)])


def eta_of(xi):                       # (mu, sigma) -> expectation parameters (mean of x, mean of x^2)
    m, s = xi
    return np.array([m, m * m + s * s])


def xi_of_eta(et):
    return np.array([et[0], math.sqrt(et[1] - et[0] ** 2)])


def polar_to_cart(x):
    return np.array([x[0] * math.cos(x[1]), x[0] * math.sin(x[1])])


def sphere_to_cart(u):                # unit sphere, u = (theta, phi)
    th, ph = u
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def psi_gauss(th):
    return -th[0] ** 2 / (4 * th[1]) + 0.5 * math.log(math.pi / (-th[1]))


def psi_third(th):
    """The third derivatives of psi for the Gaussian family, by hand: psi_{ijk}."""
    t1, t2 = th
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 0.0
    P[0, 0, 1] = P[0, 1, 0] = P[1, 0, 0] = 1 / (2 * t2 ** 2)
    P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = -t1 / t2 ** 3
    P[1, 1, 1] = 3 * t1 ** 2 / (2 * t2 ** 4) - 1 / t2 ** 3
    return P


ZN, ZW = np.polynomial.hermite_e.hermegauss(60)
ZW = ZW / math.sqrt(2 * math.pi)


def expect(f, m, s):
    """E[f(x)] for x ~ N(m, s^2): Gauss-Hermite, exact for polynomials of degree below 120."""
    return float(np.sum(ZW * f(m + s * ZN)))


def scores(x, m, s):
    return np.array([(x - m) / s ** 2, ((x - m) ** 2 - s ** 2) / s ** 3])       # d_mu log p, d_sigma log p


def affine_chart_gamma(chart, xi):
    """Gamma_{ij}^m for the connection whose straight lines are the coordinate lines of the chart a(xi):
    Gamma_{ij}^m = (d^2 a^A / d xi^i d xi^j)(d xi^m / d a^A).  Returned as G[i, j, m]."""
    D = jac(chart, xi); H = hess_map(chart, xi)
    return np.einsum("Aij,mA->ijm", H, np.linalg.inv(D))


def induced_gamma(embed, u, h=1e-4):
    """Gamma_{ab}^c of the induced connection (5.97)-(5.100) of a surface in a Cartesian ambient space (Gamma = 0 there):
    Gamma_{abc} = <d_a d_b x, d_c x>, then raised with the induced metric (5.94).  Returns (G[a, b, c], g)."""
    B = jac(embed, u).T                     # B[a, i] = d x^i / d u^a
    H = hess_map(embed, u, h)               # H[i, a, b]
    g = B @ B.T
    low = np.einsum("iab,ci->abc", H, B)
    return np.einsum("abd,dc->abc", low, np.linalg.inv(g)), g


def lower(G, g):
    """Gamma_{ijk} = Gamma_{ij}^m g_{mk}  (5.22): the first index is the direction of differentiation."""
    return np.einsum("ijm,mk->ijk", G, g)


def maxabs(a):
    return float(np.max(np.abs(a)))


def transform_gamma_lower(Gl_old, g_old, E, Hn):
    """(5.37): Gamma_{klm}(new) = J_k^i J_l^j J_m^p Gamma_{ijp}(old) + (d_k J_l^j) J_m^p g_{jp}(old),
    with E[i, k] = J_k^i = d xi^i / d zeta^k and Hn[j, k, l] = d^2 xi^j / d zeta^k d zeta^l.  Returns (full, tensor part only)."""
    tens = np.einsum("ia,jb,pc,ijp->abc", E, E, E, Gl_old)
    inhom = np.einsum("jab,pc,jp->abc", Hn, E, g_old)
    return tens + inhom, tens


def christoffel_lower_from_metric(metric, x, h=1e-5):
    """Gamma_{ijk} = (1/2)(d_i g_{jk} + d_j g_{ik} - d_k g_{ij})  (5.85), by central differences of the metric."""
    n = len(x); dg = np.zeros((n, n, n))
    for l in range(n):
        e = np.zeros(n); e[l] = h
        dg[l] = (metric(x + e) - metric(x - e)) / (2 * h)               # dg[l, i, j] = d_l g_ij
    G = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                G[i, j, k] = 0.5 * (dg[i][j, k] + dg[j][i, k] - dg[k][i, j])
    return G


def xi_of_theta_derivs(th):
    """Analytic E[i, k] = d xi^i / d theta^k and Hn[j, k, l] = d^2 xi^j / d theta^k d theta^l for xi = (mu, sigma)."""
    t1, t2 = th
    E = np.array([[-1 / (2 * t2), t1 / (2 * t2 ** 2)], [0.0, (-2 * t2) ** -1.5]])
    Hn = np.zeros((2, 2, 2))
    Hn[0, 0, 1] = Hn[0, 1, 0] = 1 / (2 * t2 ** 2)
    Hn[0, 1, 1] = -t1 / t2 ** 3
    Hn[1, 1, 1] = 3 * (-2 * t2) ** -2.5
    return E, Hn


def metric_theta(th):
    """The Fisher metric of the Gaussians in the natural parameters: the Hessian of psi."""
    t1, t2 = th
    return np.array([[-1 / (2 * t2), t1 / (2 * t2 ** 2)], [t1 / (2 * t2 ** 2), -t1 ** 2 / (2 * t2 ** 3) + 1 / (2 * t2 ** 2)]])


def rd(a, k=4):
    """Round for printing, without negative zeros."""
    return (np.round(np.asarray(a, float), k) + 0.0).tolist()


# ------------------------------------------------------------------ 1. tangent vectors

def check_tangent():
    head("1. Tangent vectors (section 5.1): operators, scores, and the transformation laws (5.6)-(5.10)")
    xi0 = np.array([1.0, 2.0]); m, s = xi0
    v = np.array([1.0, 0.5])
    f = lambda x: x[0] ** 2 + x[1] ** 3
    fd = (f(xi0 + 1e-6 * v) - f(xi0 - 1e-6 * v)) / 2e-6
    print(f"   a tangent vector as a derivative operator (5.4)-(5.5): A = (1, 0.5) at (mu, sigma) = (1, 2), f = mu^2 + sigma^3:"
          f" the difference quotient along the curve is {fd:.6f}, A^i d_i f = {v[0] * 2 * m + v[1] * 3 * s ** 2:.6f}")
    th0 = theta_of(xi0)
    E = jac(xi_of_theta, th0)               # E[i, k] = d xi^i / d theta^k  = J_k^i
    D = jac(theta_of, xi0)                  # D[k, i] = d theta^k / d xi^i  = J_i^k
    print(f"   Jacobians at (1, 2): J_k^i = d xi^i/d theta^k = {E.round(6).tolist()}   J_i^k = d theta^k/d xi^i = {D.round(6).tolist()}   product = identity: {maxabs(E @ D - np.eye(2)) < 1e-6}")
    xs = np.linspace(-4, 6, 41)
    s_xi = scores(xs, m, s)                                 # d_mu log p, d_sigma log p
    s_th = np.array([xs - m, xs ** 2 - (m * m + s * s)])    # d_theta log p = x - eta
    law = E.T @ s_xi                                        # e_k = J_k^i e_i  (5.8)
    print(f"   scores are tangent vectors (5.10): the basis law e_k = J_k^i e_i turns the (mu, sigma) scores into the theta scores x - eta_1, x^2 - eta_2:"
          f" largest error on 41 points {maxabs(law - s_th):.1e}")
    print(f"   explicitly at (1, 2): d_theta1 log p = 4 d_mu log p = x - 1;   d_theta2 log p = 8 d_mu log p + 8 d_sigma log p = x^2 - 5")
    A = v.copy(); Ath = D @ A                               # contravariant components (5.28)
    print(f"   components move the other way (5.28): A = (1, 0.5) in (mu, sigma) is A = ({Ath[0]:.4f}, {Ath[1]:.4f}) in theta;"
          f" the random variable A^i d_i log p is the same either way: {maxabs(A @ s_xi - Ath @ s_th):.1e}")
    mean = [expect(lambda x, i=i: scores(x, m, s)[i], m, s) for i in range(2)]
    var = expect(lambda x: (A @ scores(x, m, s)) ** 2, m, s)
    g = metric_gauss(xi0)
    print(f"   E[score] = ({mean[0]:.1e}, {mean[1]:.1e});  E[(A^i d_i log p)^2] = {var:.6f} = A^T g A = {A @ g @ A:.6f}  (the squared length of A in the Fisher metric)")


# ------------------------------------------------------------------ 2. the metric

def check_metric():
    head("2. The Riemannian metric (section 5.2): transformation law (5.12), Fisher information (5.13)")
    xi0 = np.array([1.0, 2.0]); m, s = xi0
    g = metric_gauss(xi0)
    sc = lambda x: scores(x, m, s)
    G = np.array([[expect(lambda x, i=i, j=j: sc(x)[i] * sc(x)[j], m, s) for j in range(2)] for i in range(2)])
    print(f"   Fisher information E[d_i log p d_j log p] at (1, 2) by quadrature = {G.round(6).tolist()};  the closed form diag(1/sigma^2, 2/sigma^2) = {g.round(6).tolist()}")
    E = jac(xi_of_theta, theta_of(xi0))
    gth = E.T @ g @ E
    cov = np.array([[s ** 2, 2 * m * s ** 2], [2 * m * s ** 2, 2 * s ** 2 * (s ** 2 + 2 * m * m)]])
    Hpsi = hess_map(psi_gauss, theta_of(xi0))[0]
    print(f"   the law g_kl = J_k^i J_l^j g_ij (5.12) gives g in theta: {gth.round(6).tolist()};  Cov[(x, x^2)] = {cov.round(6).tolist()};  Hessian of psi = {Hpsi.round(5).tolist()}   (all equal: {maxabs(gth - cov) < 1e-6 and maxabs(gth - Hpsi) < 1e-4})")
    A = np.array([1.0, 0.5]); D = jac(theta_of, xi0); Ath = D @ A
    print(f"   the length of A = (1, 0.5) does not depend on the chart (5.14): {A @ g @ A:.6f} in (mu, sigma),  {Ath @ gth @ Ath:.6f} in theta")
    x0 = np.array([2.0, 0.7]); Jc = jac(polar_to_cart, x0)
    print(f"   the plane in polar coordinates, at (r, theta) = (2, 0.7): J^T J = {(Jc.T @ Jc).round(6).tolist()} = diag(1, r^2) = {metric_polar(x0).tolist()}"
          f"   (it is Euclidean: g = delta in the Cartesian chart, (5.15))")
    print(f"   the Gaussian manifold has g = diag(1/sigma^2, 2/sigma^2); no chart gives g = delta, because its Gauss curvature is -1/2 (checked in section 8)")


# ------------------------------------------------------------------ 3. connections

def check_connections():
    head("3. Affine connections (sections 5.3, 5.4): Christoffel symbols of the three examples, three ways")
    x = np.array([1.0, 2.0]); m, s = x
    # polar: analytic, from the metric, and from the Cartesian chart through (5.37)
    xp = np.array([2.0, 0.7])
    Gc = affine_chart_gamma(polar_to_cart, xp)
    Gm = christoffel_lower_from_metric(metric_polar, xp)
    rest = Gc.copy(); rest[1, 1, 0] = rest[0, 1, 1] = rest[1, 0, 1] = 0.0; others = maxabs(rest)
    print(f"   plane in polar coordinates at (r, theta) = (2, 0.7): Gamma_(theta theta)^r = {Gc[1, 1, 0]:.6f} = -r;  Gamma_(r theta)^theta = Gamma_(theta r)^theta = {Gc[0, 1, 1]:.6f} = 1/r;"
          f" all other entries {others:.1e}")
    print(f"      from the metric by (5.85): largest difference {maxabs(lower(Gc, metric_polar(xp)) - Gm):.1e};  analytic: {maxabs(Gc - gamma_polar(xp)):.1e}")
    # the frame turning: the exact change of the polar basis over a finite step against the first-order prediction (5.19)-(5.20)
    r0, t0, dt_ = 2.0, math.radians(40.0), 0.4
    e_r = lambda t: np.array([math.cos(t), math.sin(t)])
    e_t = lambda r, t: np.array([-r * math.sin(t), r * math.cos(t)])
    Bm = np.stack([e_r(t0), e_t(r0, t0)], axis=1)
    d_er = np.linalg.solve(Bm, e_r(t0 + dt_) - e_r(t0)); d_et = np.linalg.solve(Bm, e_t(r0, t0 + dt_) - e_t(r0, t0))
    print(f"   the polar frame at (r, theta) = (2, 40 deg), step d theta = 0.4, d r = 0: exact change of e_r in the basis (e_r, e_theta) at P = ({d_er[0]:.4f}, {d_er[1]:.4f}),"
          f" first order Gamma_(ki)^j d xi^k = (0, d theta/r) = (0, {dt_ / r0:.4f});  exact change of e_theta = ({d_et[0]:.4f}, {d_et[1]:.4f}), first order (-r d theta, 0) = ({-r0 * dt_:.4f}, 0)")
    # sphere
    xs = np.array([1.0, 0.4])
    Gi, gi = induced_gamma(sphere_to_cart, xs)
    Gm = christoffel_lower_from_metric(metric_sphere, xs)
    print(f"   unit sphere at (theta, phi) = (1, 0.4): Gamma_(phi phi)^theta = {Gi[1, 1, 0]:.6f} = -sin cos = {-math.sin(1) * math.cos(1):.6f};  Gamma_(theta phi)^phi = {Gi[0, 1, 1]:.6f} = cot theta = {1 / math.tan(1):.6f}")
    print(f"      the induced connection (5.100) from the embedding in R^3 equals the Levi-Civita symbol (5.85) of the induced metric: {maxabs(lower(Gi, gi) - Gm):.1e};  analytic {maxabs(Gi - gamma_sphere(xs)):.1e}")
    # Gaussian: e from the theta chart, m from the eta chart, Levi-Civita from the Fisher metric
    Ge = affine_chart_gamma(theta_of, x); Gmm = affine_chart_gamma(eta_of, x)
    g = metric_gauss(x)
    Gl = christoffel_lower_from_metric(metric_gauss, x)
    Glc = np.einsum("ijk,km->ijm", Gl, np.linalg.inv(g))
    print(f"   Gaussians at (mu, sigma) = (1, 2), Gamma_(ij)^k:")
    names = ["mu", "sigma"]
    for lab, G, ref in (("e (theta lines straight)", Ge, gamma_e_gauss(x)), ("m (eta lines straight)", Gmm, gamma_m_gauss(x)), ("Levi-Civita", Glc, gamma_lc_gauss(x))):
        ent = [f"Gamma_({names[i]} {names[j]})^{names[k]} = {G[i, j, k]:+.4f}" for i in range(2) for j in range(i, 2) for k in range(2) if abs(G[i, j, k]) > 1e-6]
        print(f"      {lab:26s} " + ";  ".join(ent) + f"      [analytic: {maxabs(G - ref):.1e}]")
    print(f"   the closed forms: e: Gamma_(mu sigma)^mu = -2/sigma, Gamma_(sigma sigma)^sigma = -3/sigma;  m: Gamma_(mu mu)^sigma = 1/sigma, Gamma_(sigma sigma)^sigma = 1/sigma;"
          f"  Levi-Civita: -1/sigma, 1/(2 sigma), -1/sigma")
    print(f"   Levi-Civita is the average of the e- and m-connections: (e + m)/2 - LC = {maxabs((Ge + Gmm) / 2 - Glc):.1e}")
    print(f"   all three are symmetric in the two lower indices (5.84): {maxabs(Ge - Ge.transpose(1, 0, 2)):.0e}, {maxabs(Gmm - Gmm.transpose(1, 0, 2)):.0e}, {maxabs(Glc - Glc.transpose(1, 0, 2)):.0e}")
    # (5.82) metric condition: d_k g_ij = Gamma_kij + Gamma_kji
    dg = np.zeros((2, 2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = 1e-6
        dg[k] = (metric_gauss(x + e) - metric_gauss(x - e)) / 2e-6
    res = {}
    for lab, G in (("e", Ge), ("m", Gmm), ("LC", Glc)):
        L = lower(G, g)
        res[lab] = maxabs(dg - (L + L.transpose(0, 2, 1)))
    print(f"   metric condition (5.82) d_k g_ij = Gamma_kij + Gamma_kji: residual  e: {res['e']:.3f}   m: {res['m']:.3f}   Levi-Civita: {res['LC']:.1e}   (only Levi-Civita preserves lengths)")
    Le, Lm = lower(Ge, g), lower(Gmm, g)
    print(f"   the pair (e, m) satisfies the dual version d_k g_ij = Gamma^e_kij + Gamma^m_kji: residual {maxabs(dg - (Le + Lm.transpose(0, 2, 1))):.1e}   (the identity of Chapter 6)")


# ------------------------------------------------------------------ 4. tensors and non-tensors

def check_tensors():
    head("4. Tensors and non-tensors (section 5.4): the cubic tensor, the Hessian, Gamma, and the difference of two connections")
    xi0 = np.array([1.0, 2.0]); m, s = xi0; th0 = theta_of(xi0)
    sc = lambda x: scores(x, m, s)
    T = np.array([[[expect(lambda x, i=i, j=j, k=k: sc(x)[i] * sc(x)[j] * sc(x)[k], m, s) for k in range(2)] for j in range(2)] for i in range(2)])
    print(f"   cubic tensor T_ijk = E[d_i l d_j l d_k l] (5.33) at (1, 2): T_(mu mu mu) = {T[0, 0, 0]:+.4f}, T_(mu mu sigma) = {T[0, 0, 1]:.4f} = 2/sigma^3 = {2 / s ** 3:.4f},"
          f" T_(mu sigma sigma) = {T[0, 1, 1]:+.4f}, T_(sigma sigma sigma) = {T[1, 1, 1]:.4f} = 8/sigma^3 = {8 / s ** 3:.4f};  symmetric: {maxabs(T - T.transpose(1, 0, 2)) + maxabs(T - T.transpose(0, 2, 1)):.0e}")
    E, Hn = xi_of_theta_derivs(th0)
    Tth = np.einsum("ia,jb,kc,ijk->abc", E, E, E, T)
    mom = np.array([[[expect(lambda x, i=i, j=j, k=k: (np.array([x, x * x])[i] - eta_of(xi0)[i]) * (np.array([x, x * x])[j] - eta_of(xi0)[j]) * (np.array([x, x * x])[k] - eta_of(xi0)[k]), m, s)
                      for k in range(2)] for j in range(2)] for i in range(2)])
    P3 = psi_third(th0)
    print(f"   it is a tensor (5.30): transforming with J three times gives, in theta, {Tth[0, 0, 0]:.3f}, {Tth[0, 0, 1]:.3f}, {Tth[0, 1, 1]:.3f}, {Tth[1, 1, 1]:.3f};"
          f" the third central moments of (x, x^2) and the third derivatives of psi give the same numbers (differences {maxabs(Tth - mom):.1e}, {maxabs(Tth - P3):.1e})")
    # Hessian of a scalar function: not a tensor, except at a critical point (5.34)-(5.35)
    f = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + 0.5 * x[0] * x[1]
    for lab, pt in (("a point where grad f != 0", np.array([1.5, 2.5])), ("the critical point of f", None)):
        if pt is None:
            # critical point of f: solve grad f = 0
            A_ = np.array([[2.0, 0.5], [0.5, 2.0]]); pt = np.linalg.solve(A_, np.array([2.0, 4.0]))
        th = theta_of(pt)
        H_xi = hess_map(lambda y: np.array([f(y)]), pt)[0]
        H_th = hess_map(lambda t_: np.array([f(xi_of_theta(t_))]), th)[0]                    # honest second derivatives in the new chart
        Ep, Hp = xi_of_theta_derivs(th)
        law = Ep.T @ H_xi @ Ep                                                              # what a tensor would do
        grad = jac(lambda y: np.array([f(y)]), pt)[0]
        corr = np.einsum("jkl,j->kl", Hp, grad)                                             # (d_k J_l^j) d_j f
        print(f"   f(mu, sigma) = (mu-1)^2 + (sigma-2)^2 + mu sigma/2 at {lab} {np.round(pt, 4).tolist()}: honest f_kl in theta = {rd(H_th, 3)};"
              f" tensor law gives {rd(law, 3)}; the extra term (d_k J_l^j) d_j f = {rd(corr, 3)} (5.35);  |grad f| = {np.linalg.norm(grad):.4f}")
    # Gamma is not a tensor (5.37)
    g = metric_gauss(xi0)
    Glc = np.einsum("ijk,km->ijm", christoffel_lower_from_metric(metric_gauss, xi0), np.linalg.inv(g))
    Gl = lower(Glc, g)
    full, tens = transform_gamma_lower(Gl, g, E, Hn)
    direct = christoffel_lower_from_metric(metric_theta, th0)
    print(f"   Levi-Civita symbol of the Gaussian manifold in theta, computed directly from the metric g = Hessian of psi by (5.85):"
          f" Gamma_(111) = {direct[0, 0, 0]:.4f}, Gamma_(112) = {direct[0, 0, 1]:.4f}, Gamma_(122) = {direct[0, 1, 1]:.4f}, Gamma_(222) = {direct[1, 1, 1]:.4f}")
    print(f"      by the law (5.37) from the (mu, sigma) symbol: {full[0, 0, 0]:.4f}, {full[0, 0, 1]:.4f}, {full[0, 1, 1]:.4f}, {full[1, 1, 1]:.4f}  (difference {maxabs(full - direct):.1e});"
          f" the tensor-like part alone gives {tens[0, 0, 0]:.4f}, {tens[0, 0, 1]:.4f}, {tens[0, 1, 1]:.4f}, {tens[1, 1, 1]:.4f}  (off by up to {maxabs(tens - direct):.2f})")
    print(f"      and it equals half the cubic tensor, Gamma_(ijk) = T_(ijk)/2 = (1/2) psi_(ijk): difference {maxabs(direct - 0.5 * P3):.1e}   (the Hessian-metric identity)")
    # difference of two connections is a tensor
    Ge = gamma_e_gauss(xi0); Gmm = gamma_m_gauss(xi0)
    diff = lower(Gmm, g) - lower(Ge, g)
    print(f"   the difference of two connections is a tensor: Gamma^m_(ijk) - Gamma^e_(ijk) at (1, 2) = {diff[0, 0, 1]:.4f} (mu mu sigma), {diff[1, 1, 1]:.4f} (sigma sigma sigma), all others {maxabs(np.delete(diff.reshape(-1), [1, 2, 4, 7])):.1e};"
          f" this is T_(ijk): difference {maxabs(diff - T):.1e}")
    fm, _ = transform_gamma_lower(lower(Gmm, g), g, E, Hn); fe, _ = transform_gamma_lower(lower(Ge, g), g, E, Hn)
    print(f"      in the theta chart the e-symbol is {maxabs(fe):.1e} (that is the definition of e: flat in theta) and the m-symbol equals T^theta = psi_(ijk): difference {maxabs(fm - P3):.1e}")


# ------------------------------------------------------------------ 5. covariant derivative

def check_covariant_derivative():
    head("5. The covariant derivative (section 5.5): a constant vector field in polar coordinates")
    x = np.array([2.0, 0.7]); r, th = x
    # X = d/dx (Cartesian constant): X^r = cos th, X^th = -sin th / r
    X = lambda p: np.array([math.cos(p[1]), -math.sin(p[1]) / p[0]])
    dX = jac(X, x)                                       # dX[k, i] = d_i X^k
    G = gamma_polar(x)
    cov = dX.T + np.einsum("ijk,j->ik", G, X(x))          # nabla_i X^k = d_i X^k + Gamma_ij^k X^j   (5.48); [i, k]
    print(f"   X = d/dx has polar components (cos theta, -sin theta / r) = ({X(x)[0]:.4f}, {X(x)[1]:.4f}) at (r, theta) = (2, 0.7)")
    print(f"      partial derivatives d_i X^k [rows i = r, theta; columns k = r, theta]: {rd(dX.T, 4)}  (not zero)")
    print(f"      covariant derivatives nabla_i X^k = d_i X^k + Gamma_ij^k X^j:  {rd(cov, 8)}  (zero: the field does not change intrinsically)")
    Y = lambda p: np.array([1.0, 0.0])                    # constant polar components: the radial unit-coordinate field e_r
    cy = jac(Y, x).T + np.einsum("ijk,j->ik", G, Y(x))
    print(f"   the opposite example, X = e_r with constant polar components (1, 0): the partials are 0 but nabla_theta X^theta = Gamma_(theta r)^theta = {cy[1, 1]:.4f} = 1/r;"
          f" the field turns: nabla_theta e_r = (1/r) e_theta, whose Euclidean length is {cy[1, 1] * x[0]:.4f} per radian")
    # (5.48) is a tensor: Y = x d/dy in the Cartesian chart
    Yc = lambda c: np.array([0.0, c[0]])                  # components in Cartesian
    # polar components of the same field: Y^r = r cos sin, Y^th = cos^2
    Yp = lambda p: np.array([p[0] * math.cos(p[1]) * math.sin(p[1]), math.cos(p[1]) ** 2])
    c = polar_to_cart(x)
    nabla_cart = jac(Yc, c).T                              # [i, k] = d_i Y^k, Gamma = 0 in Cartesian
    nabla_pol = jac(Yp, x).T + np.einsum("ijk,j->ik", G, Yp(x))
    Jp = jac(lambda cc: np.array([math.hypot(*cc), math.atan2(cc[1], cc[0])]), c)      # d zeta^kappa / d x^i : [kappa, i]
    Jc = jac(polar_to_cart, x)                            # d x^i / d zeta^kappa : [i, kappa]
    law = np.einsum("ik,ia,bk->ab", nabla_cart, Jc, Jp)    # (1,1) tensor: T_a^b = J_a^i J_k^b T_i^k
    part = jac(Yp, x).T
    law_part = np.einsum("ik,ia,bk->ab", jac(Yc, c).T, Jc, Jp)
    print(f"   it is a tensor: for Y = x d/dy, nabla_i Y^k in polar coordinates {rd(nabla_pol, 5)};  the tensor law applied to the Cartesian nabla Y gives {rd(law, 5)};"
          f"  difference {maxabs(nabla_pol - law):.1e}.  The bare partial derivatives d_i Y^k do not obey the law: they differ by {maxabs(part - law_part):.2f}")


# ------------------------------------------------------------------ 6. geodesics

def fr_distance(a, b):
    """Fisher-Rao distance between N(mu1, s1^2) and N(mu2, s2^2): sqrt(2) arccosh(1 + (dmu^2/2 + ds^2)/(2 s1 s2))."""
    return math.sqrt(2) * math.acosh(1 + ((a[0] - b[0]) ** 2 / 2 + (a[1] - b[1]) ** 2) / (2 * a[1] * b[1]))


def speed(g, v):
    return math.sqrt(v @ g @ v)


def curve_length(metric, pts):
    """Length of the polyline through pts, with the metric taken at segment midpoints."""
    tot = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        d = q - p; tot += math.sqrt(d @ metric((p + q) / 2) @ d)
    return tot


def check_geodesics():
    head("6. Geodesics (section 5.6): straight lines, three connections, speed, reparametrisation, straight versus shortest")
    # plane in polar coordinates: a straight line
    x0 = np.array([1.0, 0.0]); v0 = np.array([0.5, 1.0])                  # Cartesian velocity (0.5 cos 0 - 0, 0.5 sin 0 + 1) = (0.5, 1)
    st = geodesic(gamma_polar, x0, v0, 2.0, 2000)
    cart = np.array([polar_to_cart(p[:2]) for p in st])
    ts = np.linspace(0, 2, 2001)
    line = np.stack([1 + 0.5 * ts, 1.0 * ts], axis=1)
    print(f"   plane in polar coordinates: start (r, theta) = (1, 0) with polar velocity (0.5, 1); the solution of (5.54) is the straight line (1 + t/2, t):"
          f" largest deviation over t in [0, 2] is {maxabs(cart - line):.1e}")
    r_end = st[-1, :2]
    print(f"      at t = 2 it is at (r, theta) = ({r_end[0]:.6f}, {r_end[1]:.6f}) = (sqrt(2^2 + 2^2) = {math.hypot(2, 2):.6f}, atan2(2, 2) = {math.atan2(2, 2):.6f})")
    # reparametrisation (5.55): xi(t) = gamma(t^3)
    def gam(sv):                                                        # the straight line above, as a polar curve, parameter s
        return np.array([math.hypot(1 + 0.5 * sv, sv), math.atan2(sv, 1 + 0.5 * sv)])
    xi_t = lambda t: gam(t ** 3)
    t0 = 1.3; h = 1e-4
    xd = (xi_t(t0 + h) - xi_t(t0 - h)) / (2 * h)
    xdd = (xi_t(t0 + h) - 2 * xi_t(t0) + xi_t(t0 - h)) / (h * h)
    resid = xdd + np.einsum("ijk,i,j->k", gamma_polar(xi_t(t0)), xd, xd)
    print(f"   reparametrisation (5.55): xi(t) = gamma(t^3) is the same line run at a different pace;  xi'' + Gamma xi' xi' = ({resid[0]:.5f}, {resid[1]:.5f}),"
          f"  c(t) xi' with c = 2/t = {2 / t0:.5f} gives ({2 / t0 * xd[0]:.5f}, {2 / t0 * xd[1]:.5f})")
    # sphere: a tilted great circle; speed conserved; a long arc is a geodesic but not shortest
    xs0 = np.array([math.pi / 2, 0.0]); vs0 = np.array([0.6, 0.8])
    L = 3 * math.pi / 2
    sg = geodesic(gamma_sphere, xs0, vs0, L, 6000)
    sp = np.array([speed(metric_sphere(p[:2]), p[2:]) for p in sg])
    P = sphere_to_cart(sg[0, :2]); Q = sphere_to_cart(sg[-1, :2])
    nrm = np.cross(P, sphere_to_cart(sg[len(sg) // 2, :2]))
    plane = max(abs(np.dot(nrm / np.linalg.norm(nrm), sphere_to_cart(p[:2]))) for p in sg)
    ang = math.acos(max(-1.0, min(1.0, float(P @ Q))))
    print(f"   unit sphere: from the equator at (theta, phi) = (pi/2, 0) with velocity (0.6, 0.8): speed g(xi', xi')^(1/2) stays {sp.min():.9f} to {sp.max():.9f};"
          f" the points stay in one plane through the centre (largest distance {plane:.1e}): a great circle")
    print(f"      run for length 3 pi/2 = {L:.4f}, it ends at angular distance arccos(P.Q) = {ang:.4f} = pi/2 from the start: a geodesic, yet the other way round the circle is {L / 3:.4f}, three times shorter")
    # Gaussians: the same start and velocity under three connections
    x = np.array([1.0, 2.0]); v = np.array([1.0, 0.5])
    th0 = theta_of(x); et0 = eta_of(x)
    Dth = jac(theta_of, x); Det = jac(eta_of, x)
    thd = Dth @ v; etd = Det @ v
    out = {}
    for lab, fn, aff in (("e", gamma_e_gauss, lambda t: xi_of_theta(th0 + t * thd)), ("m", gamma_m_gauss, lambda t: xi_of_eta(et0 + t * etd)), ("Levi-Civita", gamma_lc_gauss, None)):
        st = geodesic(fn, x, v, 1.0, 4000)
        out[lab] = st
        sp = np.array([speed(metric_gauss(p[:2]), p[2:]) for p in st])
        extra = ""
        if aff is not None:
            dev = max(maxabs(st[k, :2] - aff(k / 4000)) for k in range(0, 4001, 100))
            extra = f";  it is the straight line of its flat chart, deviation {dev:.1e}"
        print(f"   Gaussians, start (mu, sigma) = (1, 2), velocity (1, 0.5), connection {lab:12s}: endpoint t = 1 at ({st[-1, 0]:.4f}, {st[-1, 1]:.4f}); speed {sp[0]:.4f} -> {sp[-1]:.4f}{extra}")
    print(f"      theta(0) = ({th0[0]:.4f}, {th0[1]:.4f}), eta(0) = ({et0[0]:.4f}, {et0[1]:.4f}); theta velocity = ({thd[0]:.4f}, {thd[1]:.4f}), eta velocity = ({etd[0]:.4f}, {etd[1]:.4f}):  e endpoint theta = {np.round(th0 + thd, 4).tolist()} -> (mu, sigma) = {np.round(xi_of_theta(th0 + thd), 4).tolist()};"
          f"  m endpoint eta = {np.round(et0 + etd, 4).tolist()} -> (mu, sigma) = {np.round(xi_of_eta(et0 + etd), 4).tolist()}")
    # straight versus shortest between two points
    A = np.array([0.0, 1.0]); B = np.array([3.0, 2.0])
    d = fr_distance(A, B)
    thA, thB = theta_of(A), theta_of(B); etA, etB = eta_of(A), eta_of(B)
    ts = np.linspace(0, 1, 4001)
    e_curve = np.array([xi_of_theta(thA + t * (thB - thA)) for t in ts])
    m_curve = np.array([xi_of_eta(etA + t * (etB - etA)) for t in ts])
    l_line = np.array([A + t * (B - A) for t in ts])
    Le = curve_length(metric_gauss, e_curve); Lm = curve_length(metric_gauss, m_curve); Ll = curve_length(metric_gauss, l_line)
    print(f"   between N(0, 1) and N(3, 4): the Fisher-Rao distance is {d:.6f};  the e-geodesic is {Le:.6f} long ({100 * (Le / d - 1):.2f}% longer),"
          f" the m-geodesic {Lm:.6f} ({100 * (Lm / d - 1):.2f}% longer), the straight line in (mu, sigma) {Ll:.6f} ({100 * (Ll / d - 1):.2f}% longer)")
    print(f"      straight for the e-connection means straight in theta; it is not the shortest: the Levi-Civita geodesic is the shortest (section 9)")


# ------------------------------------------------------------------ 7. parallel transport

def rect(a1, b1, a2, b2):
    """The loop P -> Q -> R -> S -> P of the coordinate rectangle [a1, b1] x [a2, b2]."""
    return [(a1, a2), (b1, a2), (b1, b2), (a1, b2), (a1, a2)]


def check_transport():
    head("7. Parallel transport (section 5.7): path dependence and holonomy")
    # sphere: coordinate rectangle
    th1, th2, ph1, ph2 = 0.8, 1.6, 0.2, 1.4
    area = (math.cos(th1) - math.cos(th2)) * (ph2 - ph1)
    loop = rect(th1, th2, ph1, ph2)
    g0 = metric_sphere(np.array(loop[0], float)); A0 = np.array([1.0, 0.0])
    A1 = transport(gamma_sphere, loop, A0, 300)
    ang = angle_between(g0, A0, A1)
    print(f"   unit sphere, the coordinate rectangle theta in [{th1}, {th2}], phi in [{ph1}, {ph2}] (area {area:.6f}), vector starting along e_theta:"
          f" after P->Q->R->S->P it has turned by {ang:.6f} rad (counter-clockwise); K x area = {area:.6f};  length {speed(g0, A0):.6f} -> {speed(g0, A1):.6f}")
    # path dependence: PQR versus PSR
    P, Q, R, S = [np.array(p, float) for p in loop[:4]]
    viaQ = transport(gamma_sphere, [P, Q, R], A0, 300); viaS = transport(gamma_sphere, [P, S, R], A0, 300)
    gR = metric_sphere(R)
    print(f"      path dependence (5.59): from P to R via Q gives components {np.round(viaQ, 6).tolist()}, via S gives {np.round(viaS, 6).tolist()};"
          f" the two arrive {angle_between(gR, viaS, viaQ):.6f} rad apart")
    # latitude circle
    th0 = math.pi / 4
    A1 = transport(gamma_sphere, [(th0, 0.0), (th0, 2 * math.pi)], np.array([1.0, 0.0]), 2000)
    g1 = metric_sphere(np.array([th0, 0.0]))
    cap = 2 * math.pi * (1 - math.cos(th0))
    print(f"   a full turn round the circle of colatitude pi/4 (it encloses a cap of area 2 pi (1 - cos theta) = {cap:.6f}): the vector comes back rotated by "
          f"{angle_between(g1, np.array([1.0, 0.0]), A1) % (2 * math.pi):.6f} rad (mod 2 pi); in the moving frame it turned by -2 pi cos theta = {-2 * math.pi * math.cos(th0):.6f}")
    # polar plane: no holonomy
    r0 = 2.0
    pth = transport(gamma_polar, [(r0, 0.0), (r0, math.pi), (r0, 2 * math.pi)], np.array([1.0, 0.0]), 1500, record=True)
    mid = [(p, a) for p, a in pth if abs(p[1] - math.pi) < 1e-9][0]
    Cm = jac(polar_to_cart, mid[0]) @ mid[1]; Cs = jac(polar_to_cart, np.array([r0, 0.0])) @ np.array([1.0, 0.0])
    Ce = jac(polar_to_cart, pth[-1][0]) @ pth[-1][1]
    print(f"   plane in polar coordinates, round the circle r = 2: the polar components change on the way (at theta = pi: {np.round(mid[1], 4).tolist()}),"
          f" the Cartesian vector does not: start {np.round(Cs, 6).tolist()}, halfway {np.round(Cm, 6).tolist()}, end {np.round(Ce, 6).tolist()};  net rotation {angle_between(metric_polar(np.array([r0, 0.0])), np.array([1.0, 0.0]), pth[-1][1]):.1e}")
    # Gaussians, three connections
    m1, m2, s1, s2 = 0.0, 1.5, 1.0, 2.0
    area_g = math.sqrt(2) * (m2 - m1) * (1 / s1 - 1 / s2)
    loop = rect(m1, m2, s1, s2); gg = metric_gauss(np.array(loop[0], float)); A0 = np.array([1.0, 0.0])
    print(f"   Gaussians, the rectangle mu in [{m1}, {m2}], sigma in [{s1}, {s2}]: area in the Fisher metric = sqrt(2) (mu2 - mu1)(1/sigma1 - 1/sigma2) = {area_g:.6f}")
    for lab, fn in (("Levi-Civita", gamma_lc_gauss), ("e", gamma_e_gauss), ("m", gamma_m_gauss)):
        path = transport(fn, loop, A0, 400, record=True)
        lens = [speed(metric_gauss(p), a) for p, a in path]
        Af = path[-1][1]
        print(f"      {lab:12s}: final vector {np.round(Af, 6).tolist()}, rotation {angle_between(gg, A0, Af):+.6f} rad;  length along the way {min(lens):.4f} to {max(lens):.4f}, at the end {lens[-1]:.4f}"
              + (f";  K x area = {-0.5 * area_g:+.6f}" if lab == "Levi-Civita" else ""))
    # in a flat manifold the transport is path independent and keeps the affine components constant
    P, Q, R, S = [np.array(p, float) for p in loop[:4]]
    a = transport(gamma_e_gauss, [P, Q, R], A0, 400); b = transport(gamma_e_gauss, [P, S, R], A0, 400)
    th_P = jac(theta_of, P) @ A0; th_R = jac(theta_of, R) @ a
    print(f"   e-connection: P -> R via Q and via S give the same vector (difference {maxabs(a - b):.1e}); its theta components are constant: {np.round(th_P, 6).tolist()} -> {np.round(th_R, 6).tolist()}")
    a = transport(gamma_lc_gauss, [P, Q, R], A0, 400); b = transport(gamma_lc_gauss, [P, S, R], A0, 400)
    print(f"   Levi-Civita: the two routes differ by {angle_between(metric_gauss(R), b, a):+.6f} rad (the same K x area)")


# ------------------------------------------------------------------ 8. curvature

def gamma_stereo(x):
    """The unit sphere in stereographic coordinates (u, v): g = 4 delta / (1 + u^2 + v^2)^2, conformally flat."""
    rho2 = x[0] ** 2 + x[1] ** 2; dw = -2 * x / (1 + rho2)                    # d_j omega
    G = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                G[i, j, k] = (k == i) * dw[j] + (k == j) * dw[i] - (i == j) * dw[k]
    return G


def thph_to_uv(u):
    th, ph = u
    c = 1 / math.tan(th / 2)
    return np.array([c * math.cos(ph), c * math.sin(ph)])


def lowered_R(R, g):
    return np.einsum("ijkm,ml->ijkl", R, g)                                    # R_{ijkl} = R_{ijk}^m g_{ml}


def check_curvature():
    head("8. Riemann-Christoffel curvature (section 5.8): the tensor, the loop, the commutator")
    examples = [("plane in polar coordinates", gamma_polar, np.array([2.0, 0.7]), metric_polar),
                ("unit sphere (theta, phi)", gamma_sphere, np.array([1.0, 0.4]), metric_sphere),
                ("Gaussians, e-connection", gamma_e_gauss, np.array([1.0, 2.0]), metric_gauss),
                ("Gaussians, m-connection", gamma_m_gauss, np.array([1.0, 2.0]), metric_gauss),
                ("Gaussians, Levi-Civita", gamma_lc_gauss, np.array([1.0, 2.0]), metric_gauss)]
    for lab, fn, x, met in examples:
        R = riemann(fn, x); g = met(x)
        K = gauss_curvature(R, g)
        print(f"   {lab:28s} at {np.round(x, 2).tolist()}: largest |R_ijk^l| = {maxabs(R):.1e}"
              + (f";  R_(122)^m g_(m1)/det g = K = {K:+.6f}" if maxabs(R) > 1e-6 else "  (flat)"))
    # K at other points
    Ks = [gauss_curvature(riemann(gamma_lc_gauss, np.array(p)), metric_gauss(np.array(p))) for p in ((0.0, 0.5), (3.0, 1.0), (-2.0, 7.0))]
    Kp = [gauss_curvature(riemann(gamma_sphere, np.array(p)), metric_sphere(np.array(p))) for p in ((0.5, 0.0), (1.5, 2.0), (2.5, 4.0))]
    print(f"      K of the Gaussian manifold at (0, 0.5), (3, 1), (-2, 7): {Ks[0]:+.6f}, {Ks[1]:+.6f}, {Ks[2]:+.6f};  of the sphere at three points: {Kp[0]:+.6f}, {Kp[1]:+.6f}, {Kp[2]:+.6f}")
    # in two dimensions R is determined by K
    for lab, fn, x, met in examples[1:2] + examples[4:5]:
        R = riemann(fn, x); g = met(x); K = gauss_curvature(R, g); L = lowered_R(R, g)
        pred = np.array([[[[K * (g[j, k] * g[i, l] - g[i, k] * g[j, l]) for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)])
        print(f"      {lab}: R_ijkl = K (g_jk g_il - g_ik g_jl): largest difference {maxabs(L - pred):.1e};  antisymmetric in the last pair (metric connection): {maxabs(L + L.transpose(0, 1, 3, 2)):.1e}")
    # (5.64) as printed
    th = 1.0
    full = riemann(gamma_sphere, np.array([th, 0.4]))[0, 1, 1, 0]
    dgam_only = math.sin(th) ** 2 - math.cos(th) ** 2
    print(f"   the printed (5.64) repeats the last term of (5.63); then Gamma Gamma cancels in A_21 - A_12 and only d Gamma survives: R_(theta phi phi)^theta would be"
          f" sin^2 - cos^2 = {dgam_only:+.6f} at theta = 1 instead of sin^2 theta = {full:+.6f}")
    # R is a tensor: sphere, (theta, phi) -> stereographic (u, v)
    x0 = np.array([1.0, 0.4]); z0 = thph_to_uv(x0)
    Dn = jac(thph_to_uv, x0)                    # D[d, l] = d zeta^d / d xi^l
    En = np.linalg.inv(Dn)                      # E[i, a] = d xi^i / d zeta^a
    Rold = riemann(gamma_sphere, x0); Rnew = riemann(gamma_stereo, z0)
    law = np.einsum("ia,jb,kc,dl,ijkl->abcd", En, En, En, Dn, Rold)
    print(f"   R is a tensor: R in the stereographic chart (u, v) computed from its own Gamma by (5.66) differs from the tensor law applied to R in (theta, phi) by {maxabs(Rnew - law):.1e}"
          f" (entries of size {maxabs(Rnew):.2f})")
    # the loop experiment (5.60)-(5.70)
    print("   the loop experiment, d1 xi = eps (1, 0.3), d2 xi = eps (-0.2, 1) at the base point, A = (0.3, 0.8):  A_21 - A_12 versus R_jkl^i A^l d1^j d2^k (mine) and the printed (5.65), twice that")
    for lab, fn, x in (("unit sphere", gamma_sphere, np.array([1.0, 0.4])), ("Gaussians, Levi-Civita", gamma_lc_gauss, np.array([1.0, 2.0])), ("Gaussians, e-connection", gamma_e_gauss, np.array([1.0, 2.0]))):
        R = riemann(fn, x); A = np.array([0.3, 0.8]); u1 = np.array([1.0, 0.3]); u2 = np.array([-0.2, 1.0])
        pred = np.einsum("jkli,l,j,k->i", R, A, u1, u2)
        rows = []
        for eps in (0.2, 0.1, 0.05, 0.02):
            P = x; Q = x + eps * u1; Rr = x + eps * u1 + eps * u2; S = x + eps * u2
            a12 = transport(fn, [P, Q, Rr], A, 80); a21 = transport(fn, [P, S, Rr], A, 80)
            loop = transport(fn, [P, Q, Rr, S, P], A, 80)
            D = a21 - a12
            q = eps ** 2 * pred
            if np.linalg.norm(q) > 1e-9:
                rows.append(f"eps {eps}: |A21-A12|/|R A d1 d2| = {np.linalg.norm(D) / np.linalg.norm(q):.4f}, loop change / (-R A d1 d2) = {np.dot(loop - A, -q) / np.dot(q, q):.4f}")
            else:
                rows.append(f"eps {eps}: |A21-A12| = {np.linalg.norm(D):.1e}, |loop change| = {np.linalg.norm(loop - A):.1e}")
        print(f"      {lab}: " + ";  ".join(rows))
    print(f"      so A_21 - A_12 = R_jkl^i A^l d1^j d2^k = (1/2) R_jkl^i A^l df^jk: the printed (5.65), (5.69), (5.70) carry a factor 2 too much unless df^jk is read as half the printed (5.67);"
          f" and going round P->Q->R->S->P changes A by A_12 - A_21, the negative")
    # non-commutativity of the covariant derivative (5.72)-(5.74)
    def X(x_):
        return np.array([1 + 2 * x_[0] - x_[1] + x_[0] * x_[1], 0.5 + x_[0] + x_[1] ** 2])
    def dX(x_):                                       # dX[j, l] = d_j X^l
        return np.array([[2 + x_[1], 1.0], [-1 + x_[0], 2 * x_[1]]])
    def noncomm(fn, x):
        def Y(j, y):
            return dX(y)[j] + np.einsum("k,kl->l", X(y), np.array([[fn(y)[j, k, l] for l in range(2)] for k in range(2)]))
        def nabla_i_of_Yj(i, j):
            h = 1e-5; e = np.zeros(2); e[i] = h
            dY = (Y(j, x + e) - Y(j, x - e)) / (2 * h)
            return dY + np.einsum("m,ml->l", Y(j, x), np.array([[fn(x)[i, m, l] for l in range(2)] for m in range(2)]))
        C = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                C[i, j] = nabla_i_of_Yj(i, j) - nabla_i_of_Yj(j, i)
        return C
    for lab, fn, x in (("unit sphere", gamma_sphere, np.array([1.0, 0.4])), ("Gaussians, Levi-Civita", gamma_lc_gauss, np.array([1.0, 2.0])), ("Gaussians, e-connection", gamma_e_gauss, np.array([1.0, 2.0])), ("plane in polar coordinates", gamma_polar, np.array([2.0, 0.7]))):
        C = noncomm(fn, x); R = riemann(fn, x); rhs = np.einsum("ijkl,k->ijl", R, X(x))
        print(f"   {lab:28s}: (nabla_i nabla_j - nabla_j nabla_i) X for X = (1 + 2 th - ph + th ph, 0.5 + th + ph^2) at {np.round(x, 2).tolist()}: (i,j) = (1,2) gives {rd(C[0, 1], 5)};"
              f"  R_12k^l X^k = {rd(rhs[0, 1], 5)};  largest difference {maxabs(C - rhs):.1e}")


# ------------------------------------------------------------------ 9. flat manifolds

def check_flat():
    head("9. Flat manifolds (section 5.8.3): R = 0 gives affine coordinates, parallel frames and path independence")
    x = np.array([1.0, 2.0]); y = np.array([2.5, 1.2])
    for lab, fn, chart in (("e-connection", gamma_e_gauss, theta_of), ("m-connection", gamma_m_gauss, eta_of)):
        Rx = riemann(fn, x); Ry = riemann(fn, y)
        # the frame d xi / d a at x, transported to y along two routes, versus the frame d xi / d a at y
        Ex = np.linalg.inv(jac(chart, x)); Ey = np.linalg.inv(jac(chart, y))
        mid1 = np.array([y[0], x[1]]); mid2 = np.array([x[0], y[1]])
        T1 = np.stack([transport(fn, [x, mid1, y], Ex[:, k], 600) for k in range(2)], axis=1)
        T2 = np.stack([transport(fn, [x, mid2, y], Ex[:, k], 600) for k in range(2)], axis=1)
        print(f"   {lab}: largest |R| at two points {maxabs(Rx):.1e}, {maxabs(Ry):.1e}; the frame d xi/d(affine coordinate) at (1, 2), parallel-transported to (2.5, 1.2) along two routes,"
              f" agrees with the coordinate frame there: differences {maxabs(T1 - Ey):.1e} and {maxabs(T2 - Ey):.1e}")
    fn = gamma_lc_gauss
    Ex = np.eye(2)
    T1 = transport(fn, [x, np.array([y[0], x[1]]), y], np.array([1.0, 0.0]), 600); T2 = transport(fn, [x, np.array([x[0], y[1]]), y], np.array([1.0, 0.0]), 600)
    print(f"   Levi-Civita (curved): the same vector arrives as {np.round(T1, 4).tolist()} or {np.round(T2, 4).tolist()} depending on the route; no frame is parallel everywhere")
    # exponential map of a flat connection is the affine chart
    v = np.array([1.0, 0.5]); thd = jac(theta_of, x) @ v
    st = geodesic(gamma_e_gauss, x, v, 1.0, 4000)
    thr = np.array([theta_of(p[:2]) for p in st])
    lin = np.array([theta_of(x) + t * thd for t in np.linspace(0, 1, 4001)])
    print(f"   geodesics of the e-connection are straight lines in theta: the solution of (5.54) from (1, 2) with velocity (1, 0.5), mapped to theta, deviates from theta_0 + t theta_dot by {maxabs(thr - lin):.1e}")
    # a one-parameter family of connections joining e and m
    xs = np.array([1.0, 2.0]); gs = metric_gauss(xs)
    RL = riemann(gamma_lc_gauss, xs)
    dgs = np.zeros((2, 2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = 1e-6
        dgs[k] = (metric_gauss(xs + e) - metric_gauss(xs - e)) / 2e-6
    rows = []
    for sv in (0.0, 0.25, 0.5, 0.75, 1.0):
        fn = lambda y, sv=sv: (1 - sv) * gamma_e_gauss(y) + sv * gamma_m_gauss(y)
        Rs = riemann(fn, xs); Ls = lower(fn(xs), gs)
        defect = maxabs(dgs - (Ls + Ls.transpose(0, 2, 1)))
        rows.append(f"s = {sv}: R_s = {4 * sv * (1 - sv):.4f} x R_LC (difference {maxabs(Rs - 4 * sv * (1 - sv) * RL):.0e}), metric defect {defect:.4f}")
    print(f"   the connections (1 - s) Gamma^e + s Gamma^m join e (s = 0) to m (s = 1) and pass through Levi-Civita at s = 1/2: " + ";  ".join(rows))
    print(f"      so the curvature is 4 s (1 - s) times the Levi-Civita curvature (it vanishes only at the ends) and the metric condition (5.82) fails by |1 - 2 s| times its size at e: no connection on this manifold is both flat and metric")
    # the Euclidean claim: flat for the metric means g = delta in some chart, which needs the Levi-Civita curvature to vanish
    print(f"   so 'flat' needs a connection: the Gaussian manifold is flat for e and for m (R = 0) but not for Levi-Civita (K = -1/2), hence not Euclidean (section 2)")


# ------------------------------------------------------------------ 10. Levi-Civita

def metric_condition_system(dg, n, symmetric):
    """Unknowns Gamma_{kij} (first index = direction); equations d_k g_ij = Gamma_kij + Gamma_kji for i <= j."""
    if symmetric:
        idx = {(i, j, k): None for i in range(n) for j in range(n) for k in range(n)}
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        unk = [(i, j, k) for (i, j) in pairs for k in range(n)]
        pos = {u: a for a, u in enumerate(unk)}
        def col(i, j, k):
            return pos[(min(i, j), max(i, j), k)]
    else:
        unk = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
        pos = {u: a for a, u in enumerate(unk)}
        def col(i, j, k):
            return pos[(i, j, k)]
    rows, rhs = [], []
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                row = np.zeros(len(unk))
                row[col(k, i, j)] += 1; row[col(k, j, i)] += 1
                rows.append(row); rhs.append(dg[k][i, j])
    return np.array(rows), np.array(rhs), unk


def newton_polyline(A, B, N=48, iters=30):
    """Minimise the discrete energy N sum (dmu^2 + 2 dsigma^2)/s^2 of a polyline from A to B in the Fisher metric of the Gaussians, by Newton's method."""
    def energy_grad(z):
        P = np.vstack([A, z.reshape(-1, 2), B])
        d = np.diff(P, axis=0); s = (P[:-1, 1] + P[1:, 1]) / 2
        w = (d[:, 0] ** 2 + 2 * d[:, 1] ** 2)
        E = N * np.sum(w / s ** 2)
        gP = np.zeros_like(P)
        gP[:-1, 0] += -2 * N * d[:, 0] / s ** 2; gP[1:, 0] += 2 * N * d[:, 0] / s ** 2
        gP[:-1, 1] += -4 * N * d[:, 1] / s ** 2 - N * w / s ** 3; gP[1:, 1] += 4 * N * d[:, 1] / s ** 2 - N * w / s ** 3
        return E, gP[1:-1].reshape(-1)
    z = np.array([A + t * (B - A) for t in np.linspace(0, 1, N + 1)[1:-1]]).reshape(-1)
    for _ in range(iters):
        E, g = energy_grad(z)
        H = np.zeros((len(z), len(z))); h = 1e-6
        for a in range(len(z)):
            e = np.zeros(len(z)); e[a] = h
            H[:, a] = (energy_grad(z + e)[1] - energy_grad(z - e)[1]) / (2 * h)
        H = (H + H.T) / 2
        step = np.linalg.solve(H, -g)
        t = 1.0
        while energy_grad(z + t * step)[0] > E + 1e-4 * t * (g @ step) and t > 1e-6:
            t /= 2
        z = z + t * step
        if np.linalg.norm(step) * t < 1e-13: break
    return np.vstack([A, z.reshape(-1, 2), B]), energy_grad(z)[0]


def check_levi_civita():
    head("10. The Levi-Civita connection (section 5.9): Theorem 5.1 and Theorem 5.2")
    x = np.array([1.0, 2.0])
    dg = np.zeros((2, 2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = 1e-6
        dg[k] = (metric_gauss(x + e) - metric_gauss(x - e)) / 2e-6
    Msym, rhs, unk = metric_condition_system(dg, 2, True)
    sol = np.linalg.solve(Msym, rhs)
    ref = christoffel_lower_from_metric(metric_gauss, x)
    got = np.zeros((2, 2, 2))
    for a, (i, j, k) in enumerate(unk):
        got[i, j, k] = got[j, i, k] = sol[a]
    # the unknowns (i, j, k) hold Gamma_{ijk} with the symmetry in the first two indices: here the system was written for Gamma_{kij}, whose symmetry is in (k, i)
    print(f"   Theorem 5.1 as linear algebra, n = 2: unknown Gamma_(ijk) with the symmetry (5.84): {len(unk)} unknowns, {Msym.shape[0]} equations from (5.82), rank {np.linalg.matrix_rank(Msym)}")
    for n in (2, 3, 4):
        rng = np.random.default_rng(n)
        dgn = np.zeros((n, n, n))
        for k in range(n):
            A_ = rng.normal(size=(n, n)); dgn[k] = A_ + A_.T
        Mfull, _, unk_f = metric_condition_system(dgn, n, False)
        Ms, _, unk_s = metric_condition_system(dgn, n, True)
        print(f"      n = {n}: with symmetry {len(unk_s)} unknowns, rank {np.linalg.matrix_rank(Ms)} (unique); without it {len(unk_f)} unknowns, rank {np.linalg.matrix_rank(Mfull)}, so {len(unk_f) - np.linalg.matrix_rank(Mfull)} free parameters"
              f" = n^2 (n-1)/2 = {n * n * (n - 1) // 2} (the torsion)")
    # exact solution check using (5.85): Gamma_{kij} with (k, i) symmetric? no: (5.85) is symmetric in its first two indices (i, j), the direction and the vector
    print(f"   (5.85) satisfies (5.82): {maxabs(dg - (ref + ref.transpose(0, 2, 1))):.1e};  and (5.85) is symmetric in the first two indices: {maxabs(ref - ref.transpose(1, 0, 2)):.1e}")
    # Theorem 5.2: the shortest curve is the geodesic of (5.54) with (5.85)
    A = np.array([0.0, 1.0]); B = np.array([3.0, 2.0])
    d = fr_distance(A, B)
    print(f"   Theorem 5.2: between N(0, 1) and N(3, 4) the Fisher-Rao distance (closed form) is {d:.6f}")
    for N in (12, 24, 48):
        P, E = newton_polyline(A, B, N)
        print(f"      shortest polyline, {N} segments, found by minimising the discrete energy with Newton's method: length {curve_length(metric_gauss, P):.6f}, sqrt(energy) {math.sqrt(E):.6f}")
    P, E = newton_polyline(A, B, 48)
    v0 = 48 * (P[1] - P[0])
    for it in range(8):
        st = geodesic(gamma_lc_gauss, A, v0, 1.0, 960)
        miss = st[-1, :2] - B
        if np.linalg.norm(miss) < 1e-12: break
        Jm = np.zeros((2, 2))
        for a in range(2):
            dv = np.zeros(2); dv[a] = 1e-6
            Jm[:, a] = (geodesic(gamma_lc_gauss, A, v0 + dv, 1.0, 960)[-1, :2] - B - miss) / 1e-6
        v0 = v0 - np.linalg.solve(Jm, miss)
    st = geodesic(gamma_lc_gauss, A, v0, 1.0, 960)
    sp = speed(metric_gauss(A), v0)
    dev = max(np.linalg.norm(P[k] - st[20 * k, :2]) for k in range(49))
    print(f"      the solution of (5.54) with the Levi-Civita symbol from A with initial velocity {np.round(v0, 6).tolist()} hits B (miss {np.linalg.norm(st[-1, :2] - B):.1e});"
          f" its speed is constant, {sp:.6f}, equal to the closed-form distance {d:.6f} because the time interval is 1;  at t = k/48 it lies within {dev:.1e} of the k-th vertex of the energy minimiser")
    print(f"   the e-geodesic, m-geodesic and straight line between the same points are longer (section 6); only the Levi-Civita geodesic is both straight and shortest")


# ------------------------------------------------------------------ 11. submanifolds

def cylinder(Rr):
    return lambda u: np.array([Rr * math.cos(u[0]), Rr * math.sin(u[0]), u[1]])


def sphere_r(rho):
    return lambda u: rho * np.array([math.sin(u[0]) * math.cos(u[1]), math.sin(u[0]) * math.sin(u[1]), math.cos(u[0])])


def check_submanifolds():
    head("11. Submanifolds and embedding curvature (section 5.10): the cylinder, the sphere, curves of Gaussians")
    # cylinder radius 2
    Rr = 2.0; u = np.array([0.8, 0.3]); emb = cylinder(Rr)
    B = jac(emb, u).T
    g = B @ B.T
    Gi, _ = induced_gamma(emb, u)
    H3 = hess_map(emb, u)                                   # d_a d_b x^i
    n = np.array([math.cos(u[0]), math.sin(u[0]), 0.0])
    H = np.einsum("iab,i->ab", H3, n)
    print(f"   cylinder of radius {Rr}, coordinates (phi, z): induced metric (5.94) B B^T = {np.round(g, 6).tolist()};  induced connection (5.100): largest entry {maxabs(Gi):.1e} (it vanishes)")
    print(f"      so its Riemann-Christoffel curvature is 0 and Euclidean geometry holds inside;  the embedding curvature along the outward normal, H_ab = <d_a e_b, n> = {np.round(H, 6).tolist()};"
          f" normal curvature in the phi direction H_(phi phi)/g_(phi phi) = {H[0, 0] / g[0, 0]:+.4f} = -1/R = {-1 / Rr:+.4f}; along z: {H[1, 1] / g[1, 1]:+.4f}")
    # sphere radius 2
    rho = 2.0; us = np.array([1.0, 0.4]); embs = sphere_r(rho)
    Bs = jac(embs, us).T; gs = Bs @ Bs.T
    H3s = hess_map(embs, us); ns = embs(us) / rho
    Hs = np.einsum("iab,i->ab", H3s, ns)
    Rs = riemann(gamma_sphere, us); Ks = gauss_curvature(Rs, gs)
    print(f"   sphere of radius {rho}: g = {np.round(gs, 5).tolist()}, H_ab = {np.round(Hs, 5).tolist()} = -g_ab/rho: {maxabs(Hs + gs / rho):.1e};  intrinsic K from (5.66) = {Ks:.6f} = 1/rho^2 = {1 / rho ** 2:.6f};  det H/det g = {np.linalg.det(Hs) / np.linalg.det(gs):.6f}")
    print(f"      for a surface in Euclidean space, K = det H / det g links the two curvatures (Gauss's equation); for the cylinder det H = {np.linalg.det(H):.1e}, so K = 0 although H is not 0")
    # helix on the cylinder: a geodesic of the induced connection, with purely normal acceleration
    om, c = 1.5, 0.7
    t = 0.9
    pos = lambda tt: emb(np.array([om * tt, c * tt]))
    h = 1e-4
    vel = (pos(t + h) - pos(t - h)) / (2 * h); acc = (pos(t + h) - 2 * pos(t) + pos(t - h)) / (h * h)
    nn = np.array([math.cos(om * t), math.sin(om * t), 0.0])
    tang = acc - (acc @ nn) * nn
    print(f"   the helix (phi, z) = (1.5 t, 0.7 t): in R^3 its acceleration {np.round(acc, 4).tolist()} is purely normal (tangential part {np.linalg.norm(tang):.1e}), length {np.linalg.norm(acc):.4f} = R omega^2 = {Rr * om * om:.4f};"
          f" in the unrolled coordinates (R phi, z) it is the straight line z = (c/(R omega)) s, slope {c / (Rr * om):.4f}")
    # curves of Gaussians: embedding curvature depends on the ambient connection
    x = np.array([1.0, 2.0]); v = np.array([1.0, 0.5]); g = metric_gauss(x)
    curves = {"theta-line (e-geodesic)": gamma_e_gauss, "eta-line (m-geodesic)": gamma_m_gauss, "Fisher-Rao geodesic": gamma_lc_gauss}
    conns = {"e": gamma_e_gauss, "m": gamma_m_gauss, "Levi-Civita": gamma_lc_gauss}
    print(f"   one-dimensional submanifolds of the Gaussian manifold, at (1, 2) with velocity (1, 0.5): normal part of the acceleration nabla_c' c', measured in the Fisher metric per unit speed squared"
          f" (rows: the curve; columns: the ambient connection used to differentiate)")
    print(f"      {'':28s}" + "".join(f"{k:>14s}" for k in conns))
    for lab, own in curves.items():
        row = []
        for cn, cg in conns.items():
            # c'' = -Gamma_own c' c';  a = c'' + Gamma_conn c' c' = (Gamma_conn - Gamma_own) c' c'
            a = np.einsum("ijk,i,j->k", cg(x) - own(x), v, v)
            perp = a - (a @ g @ v) / (v @ g @ v) * v
            row.append(math.sqrt(perp @ g @ perp) / (v @ g @ v))
        print(f"      {lab:28s}" + "".join(f"{r:14.6f}" for r in row))
    print(f"   so a curve is 'straight' (zero embedding curvature) exactly for the connection that defines it: the theta-line for e, the eta-line for m, the Fisher-Rao geodesic for Levi-Civita")




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
  .sk{stroke:#1a1a19} .fk{fill:#1a1a19}
  .dot{stroke-width:1.2;fill:none;stroke-dasharray:1.5 3.5;stroke-linecap:round}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .fillS{fill:#3987e5}
    .ring{stroke:#161615}
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


def fig_polar_frames(path):
    b = []
    W, H = 780, 420
    P1 = Panel(b, 40, 56, 300, 300, (-3.1, 3.1), (-3.1, 3.1))
    b.append(f'<text class="hd" x="40" y="26">{T("The polar frame turns; the constant field does not")}</text>')
    for r in (1, 2, 3):
        pts = [(P1.X(r * math.cos(t)), P1.Y(r * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 181)]
        poly(b, pts, "ax" if r == 2 else "gd", "stroke-width:1.4" if r == 2 else "")
    for k in range(12):
        t = k * math.pi / 6
        b.append(f'<line class="gd" x1="{P1.X(0):.1f}" y1="{P1.Y(0):.1f}" x2="{P1.X(3.05 * math.cos(t)):.1f}" y2="{P1.Y(3.05 * math.sin(t)):.1f}"/>')
    for k in range(8):
        t = k * math.pi / 4; r = 2.0
        px, py = r * math.cos(t), r * math.sin(t)
        er = (math.cos(t), math.sin(t)); et = (-math.sin(t), math.cos(t)); L = 0.62
        arrow(b, P1.X(px), P1.Y(py), P1.X(px + L * er[0]), P1.Y(py + L * er[1]), "1", 2.0, 7)
        arrow(b, P1.X(px), P1.Y(py), P1.X(px + L * et[0]), P1.Y(py + L * et[1]), "2", 2.0, 7)
        arrow(b, P1.X(px), P1.Y(py), P1.X(px + L), P1.Y(py), "0", 1.4, 6, ink=True)
        b.append(f'<circle class="fk" cx="{P1.X(px):.1f}" cy="{P1.Y(py):.1f}" r="2.2"/>')
    note(b, 40, 380, ["circles r = 1, 2, 3.  Blue: e_r.  Orange: unit vector along e_θ.", "Black: the constant field ∂/∂x, the same arrow everywhere."])
    P2 = Panel(b, 430, 56, 320, 260, (0, 2 * math.pi), (-1.4, 2.6))
    P2.frame([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi], [-1, 0, 1], "θ (position on the circle r = 2)", "", T("The covariant derivative of X along e_θ, radial part"), True,
             {0: "0", math.pi / 2: "π/2", math.pi: "π", 3 * math.pi / 2: "3π/2", 2 * math.pi: "2π"})
    ts = np.linspace(0, 2 * math.pi, 241)
    P2.line(ts, np.cos(ts), "thin s0")
    P2.line(ts, -np.sin(ts), "ln s1")
    P2.line(ts, np.sin(ts), "ln s2")
    P2.line(ts, np.zeros_like(ts), "ln sk")
    legend(b, 440, 78, [("s1", "∂_θ X^r = −sin θ"), ("s2", "Γ_{θθ}^r X^θ = +sin θ"), ("sk", "sum = ∇_θ X^r = 0"), ("s0", "X^r = cos θ itself")], 160, 17)
    note(b, 430, 376, ["At r = 2: Γ_{θθ}^r = −r and X^θ = −sin θ / r, so the correction", "is +sin θ and cancels the partial derivative exactly."])
    open(path, "w", encoding="utf-8").write(svg(W, H, "The polar frame turns while a constant vector field has zero covariant derivative",
        "Left: the plane with polar coordinate curves; at eight points of the circle r = 2 the polar basis vectors e_r and e_theta turn, while the constant field d/dx stays parallel. Right: the partial derivative of the radial component of that field along the circle, minus sine theta, plus the connection term, plus sine theta, add to zero.", b))


def fig_geodesics_gauss(path):
    b = []
    W, H = 780, 392
    x0 = np.array([1.0, 2.0]); v = np.array([1.0, 0.5])
    th0 = theta_of(x0); et0 = eta_of(x0); thd = jac(theta_of, x0) @ v; etd = jac(eta_of, x0) @ v
    ts = np.linspace(0, 1.4, 281)
    e_c = np.array([xi_of_theta(th0 + t * thd) for t in ts]); m_c = np.array([xi_of_eta(et0 + t * etd) for t in ts])
    st = geodesic(gamma_lc_gauss, x0, v, 1.4, 1400); l_c = st[::5, :2]
    P1 = Panel(b, 56, 56, 320, 250, (0, 4.4), (0.8, 3.6))
    P1.frame([0, 1, 2, 3, 4], [1, 2, 3], "μ", "σ", "Same start, same velocity, three connections", True)
    for cur, cls in ((e_c, "ln s1"), (m_c, "ln s2"), (l_c, "ln s3")):
        P1.line(cur[:, 0], cur[:, 1], cls)
    P1.dot(1, 2, "f0", 5)
    arrow(b, P1.X(1), P1.Y(2), P1.X(1.6), P1.Y(2.3), "0", 1.6, 7, ink=True)
    P1.text(0.1, 1.72, "velocity (1, ½)", "sm")
    for cur, cls, lab, dxy in ((e_c, "f1", "e", (7, 4)), (m_c, "f2", "m", (8, 13)), (l_c, "f3", "Levi-Civita", (8, -9))):
        k = int(round(1.0 / 1.4 * (len(cur) - 1)))
        P1.dot(cur[k, 0], cur[k, 1], cls, 4.2)
        P1.text(cur[k, 0], cur[k, 1], lab, "v", "start", *dxy)
    note(b, 56, 358, ["dots mark t = 1:  e at (3.00, 2.83),  m at (2.00, 2.24),  Levi-Civita at (2.23, 2.39)"])
    P2 = Panel(b, 450, 56, 290, 250, (0, 1), (0, 1.8))
    P2.frame([0, 0.5, 1], [0, 0.5, 1, 1.5], "t", "speed √(g(ξ̇, ξ̇))", "Only Levi-Civita keeps the speed", True)
    sp = {}
    for lab, fn in (("e", gamma_e_gauss), ("m", gamma_m_gauss), ("lc", gamma_lc_gauss)):
        sts = geodesic(fn, x0, v, 1.0, 200)
        sp[lab] = [speed(metric_gauss(q[:2]), q[2:]) for q in sts]
    tt = np.linspace(0, 1, 201)
    P2.line(tt, sp["e"], "ln s1"); P2.line(tt, sp["m"], "ln s2"); P2.line(tt, sp["lc"], "ln s3")
    P2.text(1.0, sp["e"][-1], "e", "v", "start", 6, 4); P2.text(1.0, sp["m"][-1], "m", "v", "start", 6, 4); P2.text(1.0, sp["lc"][-1], "LC", "v", "start", 6, 4)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Three geodesics from one point with one velocity",
        "On the Gaussian half-plane the geodesics of the e-connection, the m-connection and the Levi-Civita connection leave (mu, sigma) = (1, 2) with the same velocity (1, 0.5) and arrive at different points at t = 1. The speed in the Fisher metric grows along the e-geodesic, falls along the m-geodesic and is constant along the Levi-Civita geodesic.", b))


def cam(c):
    c = np.asarray(c, float); c = c / np.linalg.norm(c)
    up = np.array([0.0, 0.0, 1.0]); u = up - (up @ c) * c; u /= np.linalg.norm(u)
    r = np.cross(-c, u)
    return c, r, u


def fig_holonomy(path):
    b = []
    W, H = 800, 430
    b.append(f'<text class="hd" x="30" y="26">{T("Parallel transport round a coordinate rectangle: K × area")}</text>')
    cx, cy, Rp = 200, 205, 142
    th1, th2, ph1, ph2 = 0.8, 1.6, 0.2, 1.4
    c, r_, u_ = cam(sphere_to_cart(np.array([1.25, 0.55])))
    def scr(p3):
        return cx + Rp * float(p3 @ r_), cy - Rp * float(p3 @ u_)
    def vis(p3):
        return float(p3 @ c) > 0.02
    b.append(f'<circle class="ax" cx="{cx}" cy="{cy}" r="{Rp}"/>')
    def curve(fn, params):
        seg = []
        for q in params:
            p3 = sphere_to_cart(fn(q))
            if vis(p3): seg.append(scr(p3))
            else:
                if len(seg) > 1: poly(b, seg, "gd")
                seg = []
        if len(seg) > 1: poly(b, seg, "gd")
    for ph in np.arange(0, 2 * math.pi, math.pi / 6):
        curve(lambda th, ph=ph: np.array([th, ph]), np.linspace(0.02, math.pi - 0.02, 90))
    for th in np.arange(math.pi / 6, math.pi, math.pi / 6):
        curve(lambda ph, th=th: np.array([th, ph]), np.linspace(0, 2 * math.pi, 181))
    loop = rect(th1, th2, ph1, ph2)
    dense = []
    for p_, q_ in zip(loop[:-1], loop[1:]):
        for t in np.linspace(0, 1, 40, endpoint=False):
            dense.append(np.array(p_) + t * (np.array(q_) - np.array(p_)))
    dense.append(np.array(loop[-1], float))
    poly(b, [scr(sphere_to_cart(d_)) for d_ in dense], "ln sk")
    rec = transport(gamma_sphere, loop, np.array([1.0, 0.0]), 200, record=True)
    def tangent3(pt, A):
        th, ph = pt
        eth = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
        eph = np.array([-math.sin(th) * math.sin(ph), math.sin(th) * math.cos(ph), 0.0])
        return A[0] * eth + A[1] * eph
    n_rec = len(rec) - 1
    for k in range(1, 8):
        pt, A = rec[k * n_rec // 8]
        p3 = sphere_to_cart(pt); v3 = tangent3(pt, A) * 0.34
        x1, y1 = scr(p3); x2, y2 = scr(p3 + v3)
        arrow(b, x1, y1, x2, y2, "3", 1.8, 6)
    pt, A = rec[0]; p3 = sphere_to_cart(pt); v3 = tangent3(pt, A) * 0.34
    x1, y1 = scr(p3); arrow(b, x1, y1, *scr(p3 + v3), "1", 3.0, 8)
    pt, A = rec[-1]; v3 = tangent3(pt, A) * 0.34
    arrow(b, x1, y1, *scr(p3 + v3), "2", 3.0, 8)
    b.append(f'<circle class="fk" cx="{x1:.1f}" cy="{y1:.1f}" r="3"/>')
    note(b, 30, cy + Rp + 34, ["Unit sphere, θ ∈ [0.8, 1.6], φ ∈ [0.2, 1.4], area 0.871.", "Blue: the start. Orange: after P→Q→R→S→P,", "turned by +0.871 rad = K × area, with K = 1."])
    P2 = Panel(b, 470, 54, 300, 283, (-0.35, 1.45), (0.55, 2.25))
    P2.frame([0, 0.5, 1], [1, 1.5, 2], "x = μ/√2  (angles are true in these coordinates)", "σ", "", True)
    loopg = rect(0.0, 1.5, 1.0, 2.0)
    poly(b, [(P2.X(q[0] / math.sqrt(2)), P2.Y(q[1])) for q in loopg], "ln sk")
    recg = transport(gamma_lc_gauss, loopg, np.array([1.0, 0.0]), 200, record=True)
    def garrow(pt, A, k, sw, head):
        L = speed(metric_gauss(pt), A)
        dxy = np.array([A[0] / math.sqrt(2), A[1]]); dxy = dxy / np.linalg.norm(dxy) * (0.20 * pt[1] * L)
        x_, y_ = pt[0] / math.sqrt(2), pt[1]
        arrow(b, P2.X(x_), P2.Y(y_), P2.X(x_ + dxy[0]), P2.Y(y_ + dxy[1]), k, sw, head)
    n_rec = len(recg) - 1
    for k in range(1, 8):
        pt, A = recg[k * n_rec // 8]; garrow(pt, A, "3", 1.8, 6)
    garrow(recg[0][0], recg[0][1], "1", 3.0, 8); garrow(recg[-1][0], recg[-1][1], "2", 3.0, 8)
    P2.dot(0, 1, "f0", 3.5)
    note(b, 470, 54 + 283 + 52, ["Gaussians, μ ∈ [0, 1.5], σ ∈ [1, 2], area 1.061.", "Turned by −0.530 rad = K × area, with K = −1/2.", "Arrows have equal Fisher length, so they shrink toward small σ."])
    open(path, "w", encoding="utf-8").write(svg(W, H, "Holonomy on the sphere and on the Gaussian half-plane",
        "A vector parallel-transported round a coordinate rectangle with the Levi-Civita connection comes back rotated by the Gauss curvature times the enclosed area: counter-clockwise by 0.871 radians on the unit sphere, clockwise by 0.530 radians on the Gaussian manifold where K is minus one half.", b))


def fig_cylinder(path):
    b = []
    W, H = 800, 410
    b.append(f'<text class="hd" x="30" y="26">{T("A cylinder is flat inside and curved outside")}</text>')
    Rr = 1.0; om = 2.0; cz = 0.8; phi0 = -2.1; t1 = 2.55
    c, r_, u_ = cam(np.array([0.78, -0.60, 0.50]))
    cx, cy, sc = 200, 190, 92
    def scr(p3):
        return cx + sc * float(p3 @ r_), cy - sc * float(p3 @ u_)
    def pos(phi, z):
        return np.array([Rr * math.cos(phi), Rr * math.sin(phi), z - 1.2])
    for k in range(24):
        ph = k * math.pi / 12
        nrm = np.array([math.cos(ph), math.sin(ph), 0.0])
        poly(b, [scr(pos(ph, 0.0)), scr(pos(ph, 2.4))], "gd", "" if nrm @ c > 0 else "opacity:.35")
    for z in (0.0, 2.4):
        poly(b, [scr(pos(ph, z)) for ph in np.linspace(0, 2 * math.pi, 145)], "ax")
    ts = np.linspace(0, t1, 300)
    seg, flag = [], None
    for t in ts:
        ph = phi0 + om * t; p3 = pos(ph, 0.2 + cz * t)
        v_ = (np.array([math.cos(ph), math.sin(ph), 0.0]) @ c) > 0
        if flag is None: flag = v_
        if v_ != flag:
            poly(b, seg, "ln s1" if flag else "dash s1"); seg = [seg[-1]]; flag = v_
        seg.append(scr(p3))
    poly(b, seg, "ln s1" if flag else "dash s1")
    poly(b, [scr(np.array([0.0, 0.0, -1.2])), scr(np.array([0.0, 0.0, 1.2]))], "dot sk")
    t = 0.75; ph = phi0 + om * t; p3 = pos(ph, 0.2 + cz * t)
    poly(b, [scr(p3), scr(np.array([0.0, 0.0, p3[2]]))], "dot s2")
    vel = np.array([-Rr * om * math.sin(ph), Rr * om * math.cos(ph), cz]); vel = vel / np.linalg.norm(vel) * 0.55
    acc = -np.array([math.cos(ph), math.sin(ph), 0.0]) * 0.62
    x1, y1 = scr(p3)
    arrow(b, x1, y1, *scr(p3 + vel), "1", 2.4, 8)
    arrow(b, x1, y1, *scr(p3 + acc), "2", 2.4, 8)
    b.append(f'<circle class="fk" cx="{x1:.1f}" cy="{y1:.1f}" r="3"/>')
    note(b, 30, 346, ["In R³ the helix (blue, dashed behind) has an acceleration (orange)", "pointing at the axis, normal to the surface:", "H_{φφ} / g_{φφ} = −1/R, and nothing along the sheet."])
    P2 = Panel(b, 450, 54, 320, 215, (0, 6.6), (0, 2.5))
    P2.frame([0, 2, 4, 6], [0, 1, 2], "s = R φ  (the sheet, unrolled)", "z", "", False)
    for k in range(0, 24):
        s_ = k * math.pi / 12 * Rr
        b.append(f'<line class="gd" x1="{P2.X(s_):.1f}" y1="{P2.Y(0):.1f}" x2="{P2.X(s_):.1f}" y2="{P2.Y(2.4):.1f}"/>')
    for z in (0.0, 2.4):
        b.append(f'<line class="ax" x1="{P2.X(0):.1f}" y1="{P2.Y(z):.1f}" x2="{P2.X(2 * math.pi):.1f}" y2="{P2.Y(z):.1f}"/>')
    s_end = Rr * om * t1
    P2.line([0, s_end], [0.2, 0.2 + cz * t1], "ln s1")
    s_t = Rr * om * t; z_t = 0.2 + cz * t
    P2.dot(s_t, z_t, "f0", 3.5)
    arrow(b, P2.X(s_t), P2.Y(z_t), P2.X(s_t + 0.7 * 0.93), P2.Y(z_t + 0.7 * 0.37), "1", 2.4, 8)
    note(b, 450, 54 + 215 + 52, ["Unrolled, the helix is the straight line z = (c / Rω) s:", "induced Γ = 0 and R = 0, so the sheet is Euclidean", "(K = det H / det g = 0 although H ≠ 0)."])
    open(path, "w", encoding="utf-8").write(svg(W, H, "A cylinder: zero intrinsic curvature, non-zero embedding curvature",
        "Left: a cylinder in three-dimensional space with a helix whose acceleration points at the axis, perpendicular to the surface. Right: the cylinder unrolled into a flat sheet, where the same helix is a straight line.", b))


def loop_ratios():
    out = {}
    eps_list = [0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02, 0.01]
    for lab, fn, x in (("sphere", gamma_sphere, np.array([1.0, 0.4])), ("gauss", gamma_lc_gauss, np.array([1.0, 2.0]))):
        R = riemann(fn, x); A = np.array([0.3, 0.8]); u1 = np.array([1.0, 0.3]); u2 = np.array([-0.2, 1.0])
        pred = np.einsum("jkli,l,j,k->i", R, A, u1, u2); rows = []
        for eps in eps_list:
            Q = x + eps * u1; Rr = x + eps * u1 + eps * u2; S = x + eps * u2
            a12 = transport(fn, [x, Q, Rr], A, 60); a21 = transport(fn, [x, S, Rr], A, 60)
            rows.append(np.linalg.norm(a21 - a12) / (eps ** 2 * np.linalg.norm(pred)))
        out[lab] = rows
    return eps_list, out


def fig_loop_factor(path):
    b = []
    W, H = 780, 392
    eps_list, out = loop_ratios()
    P1 = Panel(b, 56, 56, 320, 250, (0, 0.21), (0, 2.3))
    P1.frame([0, 0.05, 0.1, 0.15, 0.2], [0, 0.5, 1, 1.5, 2], "ε (size of the loop)", "|A₂₁ − A₁₂| / |R A d₁ξ d₂ξ|", "The loop formula has no factor 2", True)
    P1.line([0, 0.21], [1, 1], "dash s0"); P1.line([0, 0.21], [2, 2], "dash s2")
    P1.text(0.005, 0.86, "R A d₁ξ d₂ξ  (derived)", "sm"); P1.text(0.005, 2.08, "printed (5.65): twice that", "sm")
    P1.line(eps_list, out["sphere"], "ln s1"); P1.line(eps_list, out["gauss"], "ln s3")
    for e_, a_ in zip(eps_list, out["sphere"]): P1.dot(e_, a_, "f1", 3.2)
    for e_, a_ in zip(eps_list, out["gauss"]): P1.dot(e_, a_, "f3", 3.2)
    P1.text(0.2, 1.2, "unit sphere", "sm", "end"); P1.text(0.2, 0.72, "Gaussians (Levi-Civita)", "sm", "end")
    note(b, 56, 56 + 250 + 56, ["Measured on the sphere and on the Gaussian manifold:", "the ratio tends to 1 as the loop shrinks."])
    P2 = Panel(b, 450, 56, 290, 250, (0, math.pi), (-0.6, 1.1))
    P2.frame([0, 1, 2, 3], [-0.5, 0, 0.5, 1], "θ on the unit sphere", T("R_{θφφ}^θ"), "The printed (5.64) loses the ΓΓ terms", True)
    ths = np.linspace(0.02, math.pi - 0.02, 200)
    P2.line(ths, np.sin(ths) ** 2, "ln s1"); P2.line(ths, np.sin(ths) ** 2 - np.cos(ths) ** 2, "dash s2")
    P2.dot(1.0, math.sin(1.0) ** 2, "f1", 4); P2.dot(1.0, math.sin(1.0) ** 2 - math.cos(1.0) ** 2, "f2", 4)
    P2.text(2.45, 0.6, "sin²θ from (5.66)", "sm", "start"); P2.text(2.0, -0.38, "∂Γ only: sin²θ − cos²θ", "sm", "start")
    note(b, 450, 56 + 250 + 56, ["At θ = 1: 0.708 with the quadratic terms, 0.416 without;", "only the first gives the sphere's K = 1."])
    open(path, "w", encoding="utf-8").write(svg(W, H, "Checking the loop formula of Section 5.8",
        "Left: the measured difference of the two parallel transports round an infinitesimal quadrilateral, divided by the curvature term R A d1xi d2xi, tends to 1 as the loop shrinks, on the sphere and on the Gaussian manifold; the printed formula would give 2. Right: the sphere's curvature component with and without the quadratic connection terms that the printed (5.64) cancels.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_polar_frames(out / "polar-frames.svg")
    fig_geodesics_gauss(out / "geodesics-gaussian.svg")
    fig_holonomy(out / "holonomy.svg")
    fig_cylinder(out / "cylinder.svg")
    fig_loop_factor(out / "loop-factor.svg")
    print("\nwrote", ", ".join(sorted(p.name for p in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_tangent()
    check_metric()
    check_connections()
    check_tensors()
    check_covariant_derivative()
    check_geodesics()
    check_transport()
    check_curvature()
    check_flat()
    check_levi_civita()
    check_submanifolds()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
