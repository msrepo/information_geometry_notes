#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 11, checked by hand.

Machine learning: five loosely connected topics. Every number quoted in the notes comes from here.

Running examples, all small: six patterns in the positive quadrant with the generalised KL divergence (clustering); two Poisson laws
(Chernoff information); a 14-point linearly separable set and a noisy-circle problem (support vector machines); an Ising model on a
four-cycle and on the 7-node graph of Fig. 11.8 (belief propagation, mean field, CCCP); a 20-point toy set with decision stumps
(boosting); a binary RBM with 3 visible and 2 hidden units, where every distribution is enumerated, and a Gaussian RBM
(restricted Boltzmann machines). The book's equation numbers are named in the printout.

Convention as in the book (and the earlier chapters): theta natural, eta expectation parameters; D_phi[x : eta] = phi(x) - phi(eta) - grad phi(eta).(x - eta)
with the pattern x = eta in the first slot and the centre in the second.

Checked here, in the order the notes use them:

  Section 11.1, clustering
   1. Theorem 11.1: the phi-centre is the mean for five Bregman divergences, by Nelder-Mead; the other slot gives the theta-mean; the sign of (11.10);
      the converse (squared Hellinger, L1, fourth power) and the test 'D(x:z) - D(x:y) is affine in x'; (11.12)-(11.13) by quadrature;
   2. k-means (11.14)-(11.16): decrease of (11.15), termination, and an exhaustive search showing local optima;
   3. Theorem 11.2 and (11.34): the bisector of two cells is flat in eta and orthogonal to the e-geodesic (the printed version has e and m exchanged);
   4. soft k-means (11.27)-(11.32): the M-step (11.32) lacks a division by pi_h; the base measure of (11.24)-(11.26);
   5. total Bregman divergence (11.37)-(11.43): the perpendicular distance, influence functions of the two possible centres;
   6. Chernoff information (11.44)-(11.57): exact Bayes error for two Poisson laws, the signs and the lambda <-> 1 - lambda in (11.52)-(11.57);
  Section 11.2, support vector machines
   7. the linear SVM (11.58)-(11.73): SMO against a brute-force scan of directions, duality, the Lagrangian (11.69); the circle problem (11.78) and (11.84);
   8. kernels: positivity (11.80), the Mercer expansion (11.83) (wrong factor), Fourier eigenfunctions (11.86), polynomial kernel dimension (11.87);
   9. the induced metric (11.89)-(11.91), the conformal change (11.95)-(11.98);
  10. the conformal change on a noisy-circle toy problem, and the margin/radius heuristic;
  Section 11.3, stochastic reasoning
  11. mean field and the m-projection (11.107)-(11.126), Theorem 11.6, saddles but no maxima;
  12. belief propagation: the geometric algorithm (11.133)-(11.136) against message passing (11.140)-(11.143), Theorem 11.7, Theorem 11.8
      (the first half is false), accuracy on loops, the graph of Fig. 11.8;
  13. CCCP (11.144)-(11.149): the simplified version oscillates, Yuille's lowers the Bethe free energy, the stability threshold atanh(1/3);
  Section 11.4, boosting
  14. (11.153)-(11.175): AdaBoost on a toy set, the neglected constant c', normalised against unnormalised projection, the Pythagorean identity of the weights;
  Section 11.5, Bayesian inference and deep learning
  15. Bayesian duality (11.176)-(11.184) with Tweedie's formula, conjugate priors (11.185)-(11.188);
  16. the RBM: the marginal is not exponential, (11.205), (11.209), Fisher information against Hessian, natural gradient and its invariance;
  17. contrastive divergence: Theorems 11.11 and 11.12 by enumeration;
  18. the Gaussian RBM: (11.226)-(11.234) against the exact learning equation, the factor W missing in (11.229)-(11.230), the equilibria, CD_k.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Fixed seeds throughout; the whole script takes about twelve seconds.

Run:  python3 machine_learning.py            (checks)
      python3 machine_learning.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np

CHUNK = 1_000_000
STORE = {}                         # numbers computed by the checks, reused by the figures


def head(s):
    print("\n" + s)


def sub(s):
    print("  " + s)


# ------------------------------------------------------------------ small numerical tools (numpy only)

def nelder_mead(f, x0, step=0.4, tol=1e-14, maxit=6000):
    """Plain Nelder-Mead simplex minimiser; good enough for the smooth low-dimensional problems below."""
    x0 = np.asarray(x0, float); n = len(x0)
    S = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    F = np.array([f(s) for s in S])
    for _ in range(maxit):
        o = np.argsort(F); S, F = S[o], F[o]
        if abs(F[-1] - F[0]) < tol * (1 + abs(F[0])) and np.max(np.abs(S[1:] - S[0])) < 1e-9:
            break
        c = S[:-1].mean(0)
        xr = c + (c - S[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - S[-1]); fe = f(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:
            S[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (S[-1] - c) if fr >= F[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, F[-1]):
                S[-1], F[-1] = xc, fc
            else:
                S[1:] = S[0] + 0.5 * (S[1:] - S[0]); F[1:] = [f(s) for s in S[1:]]
    o = np.argsort(F)
    return S[o[0]], F[o[0]]


def bisect(f, a, b, tol=1e-14):
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b); fm = f(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
        if abs(b - a) < tol:
            break
    return 0.5 * (a + b)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------------ Bregman divergences in the eta (pattern) coordinates

class Breg:
    """D[x:y] = phi(x) - phi(y) - grad phi(y).(x - y) for a convex phi on the pattern space; theta = grad phi(eta), eta = grad psi(theta)."""

    def __init__(self, name, phi, grad, ginv, hess, to_free, from_free):
        self.name, self.phi, self.grad, self.ginv, self.hess = name, phi, grad, ginv, hess
        self.to_free, self.from_free = to_free, from_free      # a chart in which the domain is all of R^d (for the simplex search)

    def D(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return self.phi(x) - self.phi(y) - np.sum(self.grad(y) * (x - y), axis=-1)


def _sum(a):
    return np.sum(a, axis=-1)


def make_breg():
    ident = lambda v: v
    G = {}
    G["Euclid"] = Breg("squared Euclidean (phi = |x|^2/2)", lambda x: 0.5 * _sum(x * x), lambda x: x, lambda t: t,
                       lambda x: np.eye(x.shape[-1]), ident, ident)
    G["KL"] = Breg("generalised KL (phi = sum x log x - x)", lambda x: _sum(x * np.log(x) - x), lambda x: np.log(x), lambda t: np.exp(t),
                   lambda x: np.diag(1 / x), np.log, np.exp)
    G["IS"] = Breg("Itakura-Saito (phi = -sum log x)", lambda x: -_sum(np.log(x)), lambda x: -1 / x, lambda t: -1 / t,
                   lambda x: np.diag(1 / x ** 2), np.log, np.exp)
    A = np.array([[2.0, 0.6], [0.6, 1.0]])
    G["Mahalanobis"] = Breg("Mahalanobis (phi = x'Ax/2)", lambda x: 0.5 * np.einsum("...i,ij,...j->...", x, A, x), lambda x: x @ A, lambda t: t @ np.linalg.inv(A),
                            lambda x: A, ident, ident)
    G["logistic"] = Breg("logistic (phi = sum x log x + (1-x) log (1-x))", lambda x: _sum(x * np.log(x) + (1 - x) * np.log(1 - x)),
                         lambda x: np.log(x / (1 - x)), lambda t: 1 / (1 + np.exp(-t)), lambda x: np.diag(1 / (x * (1 - x))),
                         lambda v: np.log(v / (1 - v)), lambda u: 1 / (1 + np.exp(-u)))
    return G


BREG = make_breg()


def argmin_free(g, cost, x0):
    """Minimise cost(eta) over the domain of g by Nelder-Mead in the free chart; returns eta and the cost."""
    u0 = g.to_free(np.asarray(x0, float))
    u, fu = nelder_mead(lambda u: cost(g.from_free(u)), u0)
    return g.from_free(u), fu


# ------------------------------------------------------------------ 1. clustering with a divergence (section 11.1)

def affinity_defect(D, d, rg):
    """D(x:z) - D(x:y) is an affine function of x for every y, z exactly when D is a Bregman divergence (it is the bisector test).
    Returns the largest second difference of that function along random lines, relative to its size."""
    worst = 0.0
    for _ in range(40):
        y = rg.uniform(0.5, 3.0, d); z = rg.uniform(0.5, 3.0, d)
        x0 = rg.uniform(0.5, 3.0, d); dv = rg.normal(size=d) * 0.1
        xs = [x0 + k * dv for k in (-1, 0, 1)]
        f = [D(x, z) - D(x, y) for x in xs]
        sec = f[0] - 2 * f[1] + f[2]
        worst = max(worst, abs(sec) / (abs(f[1]) + abs(f[0]) + 1e-12))
    return worst


def check_centres():
    head("1. Clustering with a divergence (section 11.1).  Theorem 11.1: the phi-centre of a cluster is the arithmetic mean")
    rg = np.random.default_rng(1101)
    Xp = np.round(0.3 + rg.gamma(4.0, 0.6, size=(6, 2)), 4)    # six patterns in the positive quadrant (rounded to 4 decimals: the page uses the same numbers)
    Xu = rg.uniform(0.1, 0.9, size=(6, 2))                  # six patterns in the unit square (for the logistic divergence)
    sub("Six patterns in R^2; the minimiser of the average divergence (11.7), centre in the second slot, is found by Nelder-Mead in a")
    sub("chart where the domain is free, and compared with the arithmetic mean (11.9).  Also the reversed slot, whose minimiser should be the theta-mean:")
    print("      divergence                                   |argmin - mean|   cost at mean   | reversed slot: |argmin - mean|   |argmin - theta-mean|")
    rows = {}
    for key in ("Euclid", "KL", "IS", "Mahalanobis", "logistic"):
        g = BREG[key]; X = Xu if key == "logistic" else Xp
        mean = X.mean(0)
        c, fc = argmin_free(g, lambda e: np.mean(g.D(X, e)), mean * 0.8 + 0.1 * X.max(0))
        theta_mean = g.ginv(np.mean(g.grad(X), axis=0))
        cr, _ = argmin_free(g, lambda e: np.mean(g.D(e, X)), mean * 0.8 + 0.1 * X.max(0))
        print(f"      {g.name:<46s} {np.abs(c - mean).max():.1e}        {np.mean(g.D(X, mean)):.5f}      {np.abs(cr - mean).max():.4f}               {np.abs(cr - theta_mean).max():.1e}")
        rows[key] = (np.abs(c - mean).max(), np.abs(cr - mean).max(), np.abs(cr - theta_mean).max())
    STORE["centre_rows"] = rows
    sub("so the mean is the centre for every Bregman divergence, and for the squared Euclidean one the two slots agree; in the other slot the centre moves to the mean in theta.")
    sub("The patterns used: " + str(np.round(Xp, 4).tolist()))
    gK = BREG["KL"]; mK = Xp.mean(0); tmK = gK.ginv(np.mean(gK.grad(Xp), axis=0))
    sub(f"Generalised KL, the two candidates: mean {np.round(mK, 4).tolist()} and theta-mean {np.round(tmK, 4).tolist()}; average D[x_i : c] (centre in the 2nd slot) {np.mean(gK.D(Xp, mK)):.5f} at the mean, {np.mean(gK.D(Xp, tmK)):.5f} at the theta-mean;")
    sub(f"      average D[c : x_i] (centre in the 1st slot) {np.mean(gK.D(mK, Xp)):.5f} at the mean, {np.mean(gK.D(tmK, Xp)):.5f} at the theta-mean.")
    # (11.12)-(11.13): the centre of a distribution is its expectation, by quadrature (generalised KL, gamma density)
    xq = np.linspace(1e-6, 60, 600001); dxq = xq[1] - xq[0]
    shape_, scale_ = 4.0, 0.6
    pdf = xq ** (shape_ - 1) * np.exp(-xq / scale_) / (math.gamma(shape_) * scale_ ** shape_)
    Dq = lambda e: np.sum((xq * np.log(xq / e) - xq + e) * pdf) * dxq
    ebest, _ = nelder_mead(lambda u: float(Dq(math.exp(u[0]))), [0.3])
    sub(f"(11.12)-(11.13): for the gamma density (shape 4, scale 0.6) and the generalised KL divergence, the minimiser of the average divergence (quadrature) is {math.exp(ebest[0]):.5f}; the expectation is {shape_ * scale_:.5f}.")
    # the sign of (11.10)
    g = BREG["KL"]; e0 = np.array([1.7, 2.2]); h = 1e-6
    fd = np.array([(np.mean(g.D(Xp, e0 + h * np.eye(2)[i])) - np.mean(g.D(Xp, e0 - h * np.eye(2)[i]))) / (2 * h) for i in range(2)])
    formula = np.mean([g.hess(e0) @ (x - e0) for x in Xp], axis=0)       # (1/k) sum G^{-1}(eta)(x_i - eta), G^{-1} = Hessian of phi (11.11)
    print(f"   Derivative (11.10) at eta = (1.7, 2.2), generalised KL: finite difference {np.round(fd, 6).tolist()}, (1/k) sum G^-1 (x_i - eta) = {np.round(formula, 6).tolist()}:")
    print(f"      the gradient is the NEGATIVE of the printed right-hand side (error with the minus sign {np.abs(fd + formula).max():.1e}, with the printed plus sign {np.abs(fd - formula).max():.2f});")
    print("      the zero, hence the conclusion (11.9), is unaffected.")
    STORE["sign_err"] = (np.abs(fd + formula).max(), np.abs(fd - formula).max())
    # which divergences have the mean as the centre: the converse
    rg2 = np.random.default_rng(5)
    hel = lambda x, y: np.sum((np.sqrt(x) - np.sqrt(y)) ** 2, axis=-1)
    quart = lambda x, y: np.sum((x - y) ** 4, axis=-1)
    kl = lambda x, y: BREG["KL"].D(x, y)
    print("   The converse.  The test 'D(x:z) - D(x:y) is affine in x' characterises Bregman divergences (it is the bisector test of section 11.1.4);")
    print("      largest relative second difference along random lines:")
    for nm, D in (("squared Euclidean", BREG["Euclid"].D), ("generalised KL", kl), ("Itakura-Saito", BREG["IS"].D), ("squared Hellinger", hel), ("fourth power |x-y|^4", quart)):
        print(f"         {nm:<24s} {affinity_defect(D, 2, rg2):.1e}")
    x1 = np.array([1.0, 4.0, 9.0, 25.0])
    mean1 = x1.mean(); hc = (np.sqrt(x1).mean()) ** 2
    cost = lambda c: float(np.sum((np.sqrt(x1) - np.sqrt(c)) ** 2))
    c_num, _ = nelder_mead(lambda u: cost(np.exp(u[0])), [math.log(mean1)])
    print(f"   Squared Hellinger distance on x = (1, 4, 9, 25): the arithmetic mean is {mean1:.4f} with cost {cost(mean1):.4f}; the minimiser is {math.exp(c_num[0]):.4f} = (mean of sqrt x)^2 = {hc:.4f}, cost {cost(hc):.4f}.")
    med = np.median(x1); l1 = lambda c: float(np.sum(np.abs(x1 - c)))
    print(f"   L1 distance: the minimiser of sum |x_i - c| is the median {med:.1f} (cost {l1(med):.1f}), not the mean {mean1:.2f} (cost {l1(mean1):.2f}).")
    STORE["hellinger"] = (mean1, cost(mean1), hc, cost(hc))
    # k-means with the wrong update: start from the optimal centre of a given assignment and renew with the arithmetic mean
    sub(f"With squared Hellinger distance the 'renewal step' by the arithmetic mean RAISES the cost ({cost(hc):.4f} -> {cost(mean1):.4f}) from the optimal centre of the same cluster:")
    sub("the monotone decrease of k-means is a property of Bregman divergences, because Theorem 11.1 holds only for them.")


def lloyd(X, g, C0, maxit=200):
    """k-means with a Bregman divergence (the algorithm of section 11.1.3, centre in the second slot).  Returns costs after each full step, final centres, labels."""
    C = np.array(C0, float); m = len(C)
    costs = []; labels = None
    for it in range(maxit):
        Dm = np.stack([g.D(X, C[j]) for j in range(m)], axis=1)          # (N, m): D[x_i : eta_j]
        new = Dm.argmin(1)
        costs.append(float(Dm[np.arange(len(X)), new].sum()))
        if labels is not None and (new == labels).all():
            break
        labels = new
        for j in range(m):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    return costs, C, labels, it


def check_kmeans():
    head("2. k-means with a Bregman divergence (11.14)-(11.16): decrease of the objective (11.15), finite termination, no optimality")
    rg = np.random.default_rng(1102)
    centres = np.array([[1.0, 4.0], [4.0, 1.2], [6.0, 6.0]])
    X = np.vstack([rg.gamma(7.0, c / 7.0, size=(30, 2)) for c in centres])          # 3 clusters x 30 patterns, spread ~ 38 percent
    init = rg.choice(len(X), 3, replace=False)
    sub(f"N = {len(X)} patterns in the positive quadrant (three gamma clusters), the same three initial patterns for all divergences:")
    print("      divergence                                   steps   cost at step 1 -> final     largest increase between steps   cluster sizes")
    out = {}
    for key in ("Euclid", "KL", "IS"):
        g = BREG[key]
        costs, C, lab, it = lloyd(X, g, X[init], maxit=100)
        inc = max(np.diff(costs).max(), 0.0)
        sizes = np.bincount(lab, minlength=3).tolist()
        print(f"      {g.name:<46s} {len(costs) - 1:<7d} {costs[0]:9.3f} -> {costs[-1]:9.3f}          {inc:.1e}                          {sizes}")
        out[key] = costs
    STORE["kmeans_costs"] = out
    # brute force on a small set: exact optimum over all 2^(N-1) partitions, and the outcome of k-means from every pair of initial patterns
    rg = np.random.default_rng(7)
    Xs = np.sort(0.2 + rg.gamma(2.0, 1.0, size=(10, 1)), axis=0)
    N = len(Xs); res = {}
    for key in ("Euclid", "KL"):
        g = BREG[key]
        best = (np.inf, None)
        for mask in range(1, 2 ** (N - 1)):
            lab = np.array([(mask >> i) & 1 for i in range(N)])
            if lab.all() or (~lab.astype(bool)).all():
                continue
            c = sum(g.D(Xs[lab == j], Xs[lab == j].mean(0)).sum() for j in (0, 1))
            if c < best[0]:
                best = (c, lab.copy())
        finals = {}
        for a, b in itertools.combinations(range(N), 2):
            costs, C, lab, it = lloyd(Xs, g, Xs[[a, b]])
            finals[(a, b)] = costs[-1]
        vals = np.array(list(finals.values()))
        nbad = int((vals > best[0] + 1e-9).sum())
        bad = max(finals, key=lambda k: finals[k])
        distinct = sorted(set(round(float(v), 5) for v in vals if v > best[0] + 1e-9))
        res[key] = (best[0], len(vals), nbad, finals[bad], bad, distinct)
        print(f"   {g.name}: 10 one-dimensional patterns, 2 clusters.  Exhaustive search over all {2 ** (N - 1) - 1} partitions: optimal cost {best[0]:.5f}.")
        print(f"      k-means from each of the {len(vals)} pairs of initial patterns: {len(vals) - nbad} reach the optimum; the others stop at one of the worse local optima with costs {distinct}")
        print(f"      (the worst, {finals[bad] / best[0]:.2f} times the optimum, from the initial patterns {bad}).")
    STORE["kmeans_brute"] = res
    sub("Every run terminated (finite number of steps) with a non-increasing objective, as the book says, and 'no guarantee that it is optimal' is literally true.")


def eq_point_on_line(g, A, B, kind):
    """Point of the straight line between A and B (kind 'eta': straight in eta; 'theta': straight in theta) at which the divergences to the two ends are equal:
    D[P:A] = D[P:B] as in (11.20).  Returns (P_eta, parameter)."""
    if kind == "eta":
        pt = lambda t: (1 - t) * A + t * B
    else:
        tA, tB = g.grad(A), g.grad(B)
        pt = lambda t: g.ginv((1 - t) * tA + t * tB)
    f = lambda t: float(g.D(pt(t), A) - g.D(pt(t), B))
    t = bisect(f, 1e-9, 1 - 1e-9)
    return pt(t), t


def check_voronoi():
    head("3. Decision boundaries (Theorem 11.2, (11.17)-(11.23)): the bisector of D_phi[x:eta_1] = D_phi[x:eta_2]")
    g = BREG["KL"]
    e1, e2 = np.array([1.0, 3.0]), np.array([4.0, 1.5])
    th1, th2 = g.grad(e1), g.grad(e2)
    sub("Generalised KL on the positive quadrant, centres eta_1 = (1, 3), eta_2 = (4, 1.5), patterns x = eta, theta = log x.")
    # boundary points by bisection along vertical lines
    pts = []
    for x1 in (1.5, 2.0, 2.6, 3.4, 4.5):
        f = lambda x2: float(g.D(np.array([x1, x2]), e1) - g.D(np.array([x1, x2]), e2))
        x2 = bisect(f, 0.05, 30.0)
        pts.append((x1, x2))
    pts = np.array(pts)
    def line_resid(P):
        A = np.vstack([P[:, 0], np.ones(len(P))]).T
        coef, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
        return np.abs(A @ coef - P[:, 1]).max()
    r_eta = line_resid(pts); r_th = line_resid(np.log(pts))
    sub(f"Five boundary points found by bisection on D1 - D2 = 0: best straight line through them in the eta (pattern) chart leaves residual {r_eta:.1e};")
    sub(f"in the theta = log x chart the same points leave residual {r_th:.3f}.  The boundary is a hyperplane in eta (= x) and a curve in theta; the book says the opposite (p. 235).")
    STORE["vor_pts"] = pts
    # corrected construction: e-geodesic (straight in theta) and the eta-hyperplane orthogonal to it
    P12, t12 = eq_point_on_line(g, e1, e2, "theta")
    nrm = th2 - th1
    on_plane = []
    for tau in (-0.9, -0.3, 0.4, 1.1):
        v = np.array([-nrm[1], nrm[0]]) / np.linalg.norm(nrm)      # direction in eta-space annihilated by the normal (theta_2 - theta_1)
        # point of the plane through P12 with normal nrm (in eta): x = P12 + tau v
        x = P12 + tau * v
        if (x > 0).all():
            on_plane.append(x)
    d_c = [float(g.D(x, e1) - g.D(x, e2)) for x in on_plane]
    pyth = [abs(float(g.D(x, e1) - g.D(x, P12) - g.D(P12, e1))) for x in on_plane] + [abs(float(g.D(x, e2) - g.D(x, P12) - g.D(P12, e2))) for x in on_plane]
    sub(f"Corrected statement: the equidistant point on the theta-straight line (the e-geodesic) is eta_12 = {np.round(P12, 4).tolist()} (parameter t = {t12:.4f}); the hyperplane through it")
    sub(f"with normal theta_2 - theta_1 (flat in eta) has D1 - D2 = {max(abs(v) for v in d_c):.1e} at {len(on_plane)} test points, and the Pythagorean relation (11.21) holds to {max(pyth):.1e}.")
    # printed statement: the eta-straight line (dual geodesic), the theta-hyperplane orthogonal to it
    Q12, s12 = eq_point_on_line(g, e1, e2, "eta")
    thQ = g.grad(Q12)
    nrm2 = e2 - e1                                                # normal of the theta-hyperplane, a vector paired with theta
    v2 = np.array([-nrm2[1], nrm2[0]]) / np.linalg.norm(nrm2)
    bad = []
    for tau in (-0.5, -0.25, 0.25, 0.5):
        x = g.ginv(thQ + tau * v2)
        bad.append((float(g.D(x, e1) - g.D(x, e2)), float(g.D(x, e1) - g.D(x, Q12) - g.D(Q12, e1))))
    sub(f"As printed: the midpoint on the eta-straight line is {np.round(Q12, 4).tolist()}, and on the theta-hyperplane through it orthogonal to that line the divergences differ by")
    sub(f"{[round(b[0], 3) for b in bad]} (should be 0), and (11.21) fails by {[round(b[1], 3) for b in bad]}.")
    STORE["vor_printed"] = (Q12, P12, [b[0] for b in bad])
    STORE["vor_geo"] = (e1, e2, P12, Q12)
    # the printed theorem is right for the exchanged divergence D[eta_i : x] (centres in the first slot)
    pr = []
    for x1 in (1.5, 2.0, 2.6, 3.4, 4.5):
        f = lambda x2: float(g.D(e1, np.array([x1, x2])) - g.D(e2, np.array([x1, x2])))
        if f(0.05) * f(30.0) < 0:
            pr.append((x1, bisect(f, 0.05, 30.0)))
    pr = np.array(pr)
    fR = lambda t: float(g.D(e1, (1 - t) * e1 + t * e2) - g.D(e2, (1 - t) * e1 + t * e2))
    sR = bisect(fR, 1e-9, 1 - 1e-9)
    QR = (1 - sR) * e1 + sR * e2
    thQR = g.grad(QR)
    nR = e2 - e1; vR = np.array([-nR[1], nR[0]]) / np.linalg.norm(nR)
    xs_ = [g.ginv(thQR + tau * vR) for tau in (-0.5, -0.25, 0.25, 0.5)]
    dR = [float(g.D(e1, x_) - g.D(e2, x_)) for x_ in xs_]
    pyR = [abs(float(g.D(e1, x_) - g.D(e1, QR) - g.D(QR, x_))) for x_ in xs_] + [abs(float(g.D(e2, x_) - g.D(e2, QR) - g.D(QR, x_))) for x_ in xs_]
    sub(f"With the arguments exchanged, D[eta_i : x] (centres in the FIRST slot), the book's statement is the right one: boundary points leave residual {line_resid(np.log(pr)):.1e} on a line in the theta chart and {line_resid(pr):.3f} in the eta chart;")
    sub(f"      the midpoint of the eta-line defined by D[eta_1 : eta_12] = D[eta_2 : eta_12] is {np.round(QR, 4).tolist()}, and on the theta-hyperplane through it orthogonal to that line D1 - D2 = {max(abs(v_) for v_ in dR):.1e} and the Pythagorean relation holds to {max(pyR):.1e}.")
    sub("      But for that divergence the cluster centre is the mean in theta, not the mean (section 1); the printed (11.20)-(11.22) take the pattern in the first slot, while 'flat in theta' needs it in the second.")
    STORE["vor_swapped"] = (line_resid(np.log(pr)), line_resid(pr), QR, max(abs(v_) for v_ in dR), max(pyR))
    # the same statement with priors (Theorem 11.4, (11.34))
    sub("Theorem 11.4 / (11.34): with prior weights the posterior boundary is pi_1 exp(-D1) = pi_2 exp(-D2), i.e. D1 - D2 = log(pi_1/pi_2), not pi_1 D1 = pi_2 D2.")
    gE = BREG["Euclid"]
    pi1, pi2 = 0.8, 0.2
    c1, c2 = np.array([0.0]), np.array([4.0])
    post = lambda x: pi1 * math.exp(-float(gE.D(np.array([x]), c1))) - pi2 * math.exp(-float(gE.D(np.array([x]), c2)))
    xb = bisect(post, 0.0, 4.0)
    printed = lambda x: pi1 * float(gE.D(np.array([x]), c1)) - pi2 * float(gE.D(np.array([x]), c2))
    xp = bisect(printed, 0.0, 3.9)
    sub(f"One dimension, unit-variance Gaussians (phi = x^2/2) at 0 and 4, priors 0.8 and 0.2: the posteriors are equal at x = {xb:.4f} (= 2 + log(4)/4 = {2 + math.log(4) / 4:.4f});")
    sub(f"the printed condition (11.34) gives x = {xp:.4f} (the other root is {-4.0:.1f}).")
    STORE["prior_boundary"] = (xb, xp)


def check_soft_kmeans():
    head("4. Soft k-means (11.27)-(11.32): the M-step (11.32) and the base measure of (11.24)-(11.26)")
    rg = np.random.default_rng(1104)
    x = np.concatenate([rg.normal(-2.5, 1.0, 120), rg.normal(3.0, 1.0, 60)])
    N = len(x)
    def loglik(eta, pi):
        return float(np.sum(np.log(sum(pi[h] * np.exp(-0.5 * (x - eta[h]) ** 2) / math.sqrt(2 * math.pi) for h in range(2)))))
    def run(correct, iters=300):
        eta = np.array([-1.0, 1.0]); pi = np.array([0.5, 0.5]); ll = []
        for _ in range(iters):
            w = np.stack([pi[h] * np.exp(-0.5 * (x - eta[h]) ** 2) for h in range(2)], axis=1)
            w /= w.sum(1, keepdims=True)                                # (11.30): p(C_h | x_i), unit-variance Gaussians (phi = x^2/2)
            pi = w.mean(0)                                             # (11.31)
            eta = (w * x[:, None]).sum(0) / (N * pi) if correct else (w * x[:, None]).sum(0) / N      # (11.32) as printed: no division by pi_h
            ll.append(loglik(eta, pi))
        return eta, pi, np.array(ll), w
    eta_c, pi_c, ll_c, w_c = run(True)
    eta_p, pi_p, ll_p, w_p = run(False)
    grad = lambda eta, pi, w: np.array([np.sum(w[:, h] * (x - eta[h])) for h in range(2)])
    sub("N = 180 points on a line from two unit-variance Gaussians (true centres -2.5 and 3, weights 2/3 and 1/3); the pattern geometry is phi = x^2/2.")
    sub(f"Correct M-step (weighted mean, sum_i p x_i / sum_i p):  centres {np.round(eta_c, 4).tolist()}, weights {np.round(pi_c, 4).tolist()}, log-likelihood {ll_c[-1]:.4f}; it never decreased (largest drop {max(0, -np.diff(ll_c).min()):.1e}).")
    sub(f"M-step as printed in (11.32) (1/N sum_i x_i p):  centres {np.round(eta_p, 4).tolist()} = pi_h x (weighted mean) = {np.round(pi_p * (w_p * x[:, None]).sum(0) / (w_p.sum(0)), 4).tolist()}, log-likelihood {ll_p[-1]:.4f}.")
    sub(f"      At that point the likelihood gradient with respect to the centres is {np.round(grad(eta_p, pi_p, w_p), 3).tolist()} (zero would be a stationary point); at the correct fixed point it is {np.round(grad(eta_c, pi_c, w_c), 8).tolist()}.")
    STORE["soft"] = (eta_c, pi_c, ll_c[-1], eta_p, pi_p, ll_p[-1])
    # base measure of (11.24)-(11.26)
    xs = np.linspace(1e-9, 400.0, 4_000_001)
    eta = 2.0
    phi = lambda v: -1.0 - np.log(v)                       # Itakura-Saito: dual of the exponential distribution
    D = lambda v: phi(v) - phi(eta) - (-1.0 / eta) * (v - eta)
    f_is = np.exp(phi(xs)) * np.exp(-D(xs))
    I_is = float(np.sum(f_is) * (xs[1] - xs[0]))
    xs2 = np.linspace(-60.0, 60.0, 600_001)
    D2 = 0.5 * (xs2 - eta) ** 2
    f_g = np.exp(-D2)
    I_g = float(np.sum(f_g) * (xs2[1] - xs2[0]))
    f_g_print_60 = math.exp(60.0 * eta - 0.5 * eta ** 2)                  # e^phi e^(-D) at x = 60 for phi = x^2/2
    sub(f"(11.24) literally, e^phi e^(-D_phi): Itakura-Saito phi = -1 - log x, centre 2: the integral over x > 0 is {I_is:.5f} (it is the exponential density with mean 2, base measure dx).")
    sub(f"For phi = x^2/2 (Gaussian) e^phi e^(-D) = exp(x eta - eta^2/2) grows without bound (value {f_g_print_60:.1e} at x = 60), while e^(-D) alone integrates to {I_g:.5f} = sqrt(2 pi) = {math.sqrt(2 * math.pi):.5f};")
    sub("the right general statement (Banerjee et al.) is exp(-D_phi) times a base density b(x), with b = exp(phi) only for families whose own base measure is dx.")


def check_tbd():
    head("5. Total Bregman divergence and robust centres (11.35)-(11.43)")
    g = BREG["KL"]
    ep, e = np.array([4.0, 1.5]), np.array([1.0, 2.5])      # eta' and eta
    Dv = float(g.D(ep, e))
    # distance from the lifted point (eta', phi(eta')) to the tangent hyperplane at eta, by explicit minimisation
    n_e = g.grad(e)
    def dist2(y):
        z = g.phi(e) + n_e @ (y - e)
        return float(np.sum((y - ep) ** 2) + (z - g.phi(ep)) ** 2)
    y, f2 = nelder_mead(dist2, ep.copy())
    d_true = math.sqrt(f2)
    w_e = math.sqrt(1 + np.sum(g.grad(e) ** 2)); w_ep = math.sqrt(1 + np.sum(g.grad(ep) ** 2))
    sub(f"Generalised KL, eta' = (4, 1.5), eta = (1, 2.5): D[eta':eta] = {Dv:.5f}.  Perpendicular distance from (eta', phi(eta')) to the tangent hyperplane at eta, by minimisation: {d_true:.5f};")
    sub(f"      D/sqrt(1 + |grad phi(eta)|^2) = {Dv / w_e:.5f} (gradient at the point of tangency);  the printed (11.37)-(11.38), D/sqrt(1 + |grad phi(eta')|^2) = {Dv / w_ep:.5f}.")
    STORE["tbd"] = (Dv, d_true, Dv / w_e, Dv / w_ep)
    # influence functions
    sub("Influence of one outlier x* on the centre of k = 50 patterns (c = 1), z = k (new centre - old centre), one dimension:")
    k = 50
    rg = np.random.default_rng(11)
    for key, lab in (("Euclid", "phi = x^2/2"), ("KL", "phi = x log x - x")):
        gg = BREG[key]
        pts = (0.5 + rg.gamma(9.0, 0.25, size=(k, 1))) if key == "KL" else rg.normal(2.0, 0.5, size=(k, 1))
        w = lambda v: np.sqrt(1 + np.sum(gg.grad(np.atleast_2d(v)) ** 2, axis=-1))
        wi = w(pts)
        mean_old_R = (pts / wi[:, None]).sum(0) / (1 / wi).sum()                      # centre in the 2nd slot, weights at the data: weighted mean
        mean_old_L = gg.ginv((gg.grad(pts) / wi[:, None]).sum(0) / (1 / wi).sum())     # centre in the 1st slot: weighted theta-mean
        zs = {}
        for xs_ in (10.0, 100.0, 1e4, 1e6):
            xo = np.array([xs_]); wo = float(w(xo)[0])
            newR = ((pts / wi[:, None]).sum(0) + xo / wo) / ((1 / wi).sum() + 1 / wo)
            newL = gg.ginv(((gg.grad(pts) / wi[:, None]).sum(0) + gg.grad(xo) / wo) / ((1 / wi).sum() + 1 / wo))
            zs[xs_] = (k * float((newR - mean_old_R)[0]), k * float((newL - mean_old_L)[0]))
        print(f"      {lab:<20s} outlier x* = " + "  ".join(f"{xs_:>8.0e}" for xs_ in zs))
        print(f"      {'':<20s} z, centre as in (11.40)     " + "  ".join(f"{v[0]:8.2f}" for v in zs.values()))
        print(f"      {'':<20s} z, centre in 1st slot       " + "  ".join(f"{v[1]:8.2f}" for v in zs.values()))
        STORE[f"infl_{key}"] = zs
    sub("The centre defined by the printed objective (11.40) (patterns in the first slot, weights 1/w(x_i)) is robust for the Euclidean phi but its influence grows like x*/log x* for the KL phi;")
    sub("the influence function (11.41)-(11.43) belongs to the centre in the FIRST slot, which stays bounded for both.")
    # the sign of (11.41), numerically, 1-d KL
    gg = BREG["KL"]
    pts = 0.5 + rg.gamma(9.0, 0.25, size=(k, 1))
    wi = np.sqrt(1 + np.sum(gg.grad(pts) ** 2, axis=-1))
    eb = gg.ginv((gg.grad(pts) / wi[:, None]).sum(0) / (1 / wi).sum())
    xo = np.array([8.0]); wo = float(np.sqrt(1 + np.sum(gg.grad(xo) ** 2)))
    new = gg.ginv(((gg.grad(pts) / wi[:, None]).sum(0) + gg.grad(xo) / wo) / ((1 / wi).sum() + 1 / wo))
    z_num = k * float((new - eb)[0])
    # the closed forms used above, checked by minimising the two objectives numerically (KL, one dimension)
    allp = np.vstack([pts, xo[None, :]]); allw = np.append(wi, wo)
    cR, _ = nelder_mead(lambda u: float(np.sum(gg.D(allp, np.array([math.exp(u[0])])) / allw)), [math.log(2.5)])
    cL, _ = nelder_mead(lambda u: float(np.sum(gg.D(np.array([math.exp(u[0])]), allp) / allw)), [math.log(2.5)])
    wmean = float((allp[:, 0] / allw).sum() / (1 / allw).sum())
    sub(f"Closed forms checked by minimising the objectives numerically (outlier 8): second slot {math.exp(cR[0]):.6f} = weighted mean {wmean:.6f}; first slot {math.exp(cL[0]):.6f} = theta-weighted mean {float(new[0]):.6f}.")
    Gm = (1 / k) * (1 / wi).sum() * gg.hess(eb)                                      # (11.42), N = k
    z_pr = float(((1 / wo) * np.linalg.inv(Gm) @ (gg.grad(eb) - gg.grad(xo)))[0])  # (11.41) as printed
    sub(f"KL, centre {eb[0]:.4f}, outlier x* = 8: exact z = {z_num:.4f}; (11.41) as printed, (1/w) G^-1 {{grad phi(centre) - grad phi(x*)}} = {z_pr:.4f}  (opposite sign, magnitude within {abs(abs(z_pr) / z_num - 1) * 100:.1f} percent: the formula is first order in 1/k).")
    STORE["z_sign"] = (z_num, z_pr)


def check_chernoff():
    head("6. Error probability and Chernoff information (11.44)-(11.57)")
    # two Poisson distributions as p_i = exp(theta_i x - psi(theta_i))/x!, psi = e^theta; N observations; the sufficient statistic S is Poisson(N lam)
    lam1, lam2 = 2.0, 5.0
    th1, th2 = math.log(lam1), math.log(lam2)
    psi = math.exp
    # (11.49)-(11.50): theta_t = (1 - t) th1 + t th2; log-integral of p1^(1-t) p2^t
    logI = lambda t: psi((1 - t) * th1 + t * th2) - (1 - t) * lam1 - t * lam2        # log sum_x p1^(1-t) p2^t
    ts = np.linspace(0, 1, 100001)
    vals = np.array([logI(t) for t in ts[::100]])
    tstar = bisect(lambda t: (th2 - th1) * math.exp((1 - t) * th1 + t * th2) - (lam2 - lam1), 1e-9, 1 - 1e-9)   # d/dt log I = 0
    C = -logI(tstar)
    kl = lambda ta, tb: math.exp(ta) * (ta - tb) - math.exp(ta) + math.exp(tb)            # KL[p_ta : p_tb] = (ta - tb) eta - psi(ta) + psi(tb)
    ttheta = (1 - tstar) * th1 + tstar * th2
    sub(f"Poisson(2) against Poisson(5).  The e-geodesic (11.50) is theta_t = (1-t) log 2 + t log 5; the equidistant point (11.51) is at t* = {tstar:.5f}, theta* = {ttheta:.5f}:")
    sub(f"KL[theta*:theta_1] = {kl(ttheta, th1):.6f}, KL[theta*:theta_2] = {kl(ttheta, th2):.6f}, and the Chernoff information -min_t log sum p1^(1-t) p2^t = {C:.6f}.")
    sub(f"The integral (11.52) at its minimum is {math.exp(-C):.5f}: the printed (11.55), KL = psi(lambda*) with psi the integral (11.52), would give {math.exp(-C):.5f}; the correct relation is KL = -log psi(lambda*) = {C:.6f}.")
    # exact Bayes error for N observations, equal priors
    Ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    def log_pois(s, mu):
        return s * math.log(mu) - mu - np.array([math.lgamma(v + 1.0) for v in s])
    perr = {}
    for pri in ((0.5, 0.5), (0.9, 0.1)):
        for N in Ns:
            smax = int(5 * N * lam2 + 50)
            s = np.arange(0, smax + 1)
            lg = np.array([math.lgamma(v + 1.0) for v in s])
            a = math.log(pri[0]) + s * math.log(N * lam1) - N * lam1 - lg
            b = math.log(pri[1]) + s * math.log(N * lam2) - N * lam2 - lg
            m = np.minimum(a, b)
            mx = m.max()
            perr[(pri, N)] = mx + math.log(np.exp(m - mx).sum())
    print("      N      -log P_err / N (equal priors)   (priors 0.9, 0.1)    N^(1/2) e^(NC) P_err   (Chernoff bound P_err <= e^(-NC)/2 holds: ratio to it)")
    for N in Ns:
        e1 = perr[((0.5, 0.5), N)]; e2 = perr[((0.9, 0.1), N)]
        print(f"      {N:<6d} {-e1 / N:<31.6f} {-e2 / N:<21.6f} {math.sqrt(N) * math.exp(e1 + N * C):<22.5f} {math.exp(e1 + N * C) * 2:.5f}")
    STORE["chernoff"] = (C, tstar, Ns, [perr[((0.5, 0.5), N)] for N in Ns], [perr[((0.9, 0.1), N)] for N in Ns], lam1, lam2)
    pref = math.sqrt(Ns[-1]) * math.exp(perr[((0.5, 0.5), Ns[-1])] + Ns[-1] * C)
    sub(f"Both exponents tend to C = {C:.6f}: the prior only changes the prefactor (the Remark after (11.57)); P_err itself is about {pref:.2f} N^(-1/2) e^(-NC),")
    sub("so (11.54) is an equality of exponents, and the true error is below the bound e^(-NC)/2 for every N (last column).")
    # alpha-divergence: lambda* from (11.56)-(11.57) with the book's (3.39)
    x = np.arange(0, 80)
    p1 = np.exp(x * math.log(lam1) - lam1 - np.array([math.lgamma(v + 1.0) for v in x]))
    p2 = np.exp(x * math.log(lam2) - lam2 - np.array([math.lgamma(v + 1.0) for v in x]))
    D_alpha = lambda a: 4 / (1 - a * a) * (1 - np.sum(p1 ** ((1 - a) / 2) * p2 ** ((1 + a) / 2)))
    obj = lambda a: -(1 - a * a) / 4 * D_alpha(a)
    al, _ = nelder_mead(lambda u: obj(math.tanh(u[0])), [0.1])
    astar = math.tanh(al[0])
    sub(f"Book's alpha-divergence (3.39): the maximiser of ((1 - alpha^2)/4) D_alpha[p1:p2] is alpha* = {astar:.5f}; min over lambda of sum p1^lambda p2^(1-lambda) = {1 - (1 - astar ** 2) / 4 * D_alpha(astar):.5f} (11.56) = {math.exp(-C):.5f} above;")
    sub(f"      the minimiser of (11.52), lambda = (1 - alpha*)/2 = {(1 - astar) / 2:.5f}, is {1 - tstar:.5f} = 1 - t*; the e-geodesic midpoint t* = (1 + alpha*)/2 = {(1 + astar) / 2:.5f} is what (11.57) gives.")
    STORE["alpha_star"] = astar


# ------------------------------------------------------------------ 2. geometry of the support vector machine (section 11.2)

def smo(K, y, C=1e6, tol=1e-10, maxit=200000):
    """Dual of the soft-margin SVM, min 1/2 a'Qa - 1'a  s.t. 0 <= a <= C, y'a = 0, Q = (y y') * K, by SMO with the second-order working-set rule of LIBSVM.
    Returns (alpha, b, iterations)."""
    n = len(y); Q = (y[:, None] * y[None, :]) * K
    a = np.zeros(n); G = -np.ones(n)
    kd = np.diag(K)
    for it in range(maxit):
        yG = -y * G
        up = ((y > 0) & (a < C - 1e-12)) | ((y < 0) & (a > 1e-12))
        low = ((y > 0) & (a > 1e-12)) | ((y < 0) & (a < C - 1e-12))
        iu = np.where(up)[0]
        i = iu[np.argmax(yG[iu])]
        il = np.where(low & (yG < yG[i]))[0]
        if len(il) == 0 or yG[i] - yG[low].min() < tol:
            break
        bit = yG[i] - yG[il]
        ait = np.maximum(kd[i] + kd[il] - 2 * K[i, il], 1e-12)
        j = il[np.argmin(-bit * bit / ait)]
        kappa = max(kd[i] + kd[j] - 2 * K[i, j], 1e-12)
        t = (yG[i] - yG[j]) / kappa                        # a_i += y_i t, a_j -= y_j t keeps y'a fixed
        lo_i, hi_i = ((-a[i], C - a[i]) if y[i] > 0 else (a[i] - C, a[i]))
        lo_j, hi_j = ((a[j] - C, a[j]) if y[j] > 0 else (-a[j], C - a[j]))
        t = min(max(t, max(lo_i, lo_j)), min(hi_i, hi_j))
        di, dj = y[i] * t, -y[j] * t
        a[i] += di; a[j] += dj
        G += Q[:, i] * di + Q[:, j] * dj
    # bias from the free support vectors, or the midpoint of the admissible interval when there are none
    f0 = (a * y) @ K
    free = (a > 1e-9 * max(C, 1.0)) & (a < C - 1e-9 * max(C, 1.0))
    if free.any():
        b = float(np.mean(y[free] - f0[free]))
    else:
        up_v = np.max(np.where(y < 0, -1 - f0, -np.inf)); lo_v = np.min(np.where(y > 0, 1 - f0, np.inf))
        b = 0.5 * (up_v + lo_v)
    return a, b, it


def toy_linear(rg, n_each=6, gap=1.2):
    """Two linearly separable clouds in the plane."""
    while True:
        Xp = rg.normal([1.6, 1.4], 0.9, size=(n_each, 2)); Xm = rg.normal([-1.2, -1.0], 0.9, size=(n_each, 2))
        X = np.round(np.vstack([Xp, Xm]), 5); y = np.r_[np.ones(n_each), -np.ones(n_each)]
        d = np.linspace(0, math.pi, 4001)
        best = max((np.min(X[y > 0] @ np.array([math.cos(t), math.sin(t)])) - np.max(X[y < 0] @ np.array([math.cos(t), math.sin(t)]))) / 2 for t in d)
        if best > 0.25:
            return X, y


def check_svm_linear():
    head("7. The linear support vector machine (11.58)-(11.73)")
    rg = np.random.default_rng(1105)
    X, y = toy_linear(rg, n_each=7)
    K = X @ X.T
    sub("points (x_1, x_2, y): " + str([[round(float(v), 5) for v in X[i]] + [int(y[i])] for i in range(len(y))]))
    a, b, it = smo(K, y, C=1e6)
    w = (a * y) @ X
    f = X @ w + b
    sv = a > 1e-7
    margin = 1 / np.linalg.norm(w)
    # brute-force primal: scan the direction n of the hyperplane normal; best margin for a given direction is half the gap between projected classes
    ang = np.linspace(0, math.pi, 200001)
    N_ = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    proj = X @ N_.T
    gap = proj[y > 0].min(0) - proj[y < 0].max(0)
    ib = int(np.argmax(gap)); rho = gap[ib] / 2
    sub(f"{len(y)} separable points in the plane.  SMO solution of the dual (11.71)-(11.72): {sv.sum()} support vectors, sum_i alpha_i y_i = {np.sum(a * y):.1e}, w = sum alpha_i y_i x_i = {np.round(w, 5).tolist()}, b = {b:.5f}.")
    sub(f"y_i f(x_i) - 1 on the support vectors: {np.abs(y[sv] * f[sv] - 1).max():.1e}; smallest y_i f(x_i) over all points: {(y * f).min():.8f} (>= 1, (11.68)); alpha_i = 0 off the support vectors (smallest non-SV y f = {(y * f)[~sv].min():.4f}).")
    sub(f"Margin 1/|w| = {margin:.6f}.  Independent route: scan 200001 directions of the hyperplane normal, take half the gap between the projected classes: best margin {rho:.6f} at angle {math.degrees(ang[ib]):.3f} deg;")
    sub(f"direction of w: {math.degrees(math.atan2(w[1], w[0])) % 180:.3f} deg.  (11.65) verified: the nearest points are at distance {np.min(y * f / np.linalg.norm(w)):.6f} from the hyperplane.")
    primal = 0.5 * w @ w; dual = a.sum() - 0.5 * (a * y) @ K @ (a * y)
    sub(f"Strong duality: primal (11.67) |w|^2/2 = {primal:.8f}, dual (11.71) = {dual:.8f}; also sum alpha_i = |w|^2 ({a.sum():.8f} vs {w @ w:.8f}).")
    L_print = 0.5 * w @ w - np.sum(a * y * (X @ w + b))
    L_full = L_print + a.sum()
    sub(f"Lagrangian at the optimum: as printed in (11.69), |w|^2/2 - sum alpha_i y_i (w.x_i + b) = {L_print:.8f} (= -|w|^2/2); with the constraint's right-hand side, + sum alpha_i, it is {L_full:.8f} = the optimum.")
    STORE["svm_lin"] = (X, y, w, b, a, margin, rho)
    # the circle problem (11.78): not separable in the plane, separable after the embedding; the kernel trick (11.84)
    rg2 = np.random.default_rng(1178)
    Xc, yc = noisy_circle(rg2, 24, 0.0, R0=1.1)
    ang = np.linspace(0, 2 * math.pi, 20001)
    proj = Xc @ np.stack([np.cos(ang), np.sin(ang)])
    best_gap = float(np.max(proj[yc > 0].min(0) - proj[yc < 0].max(0)))
    Z = np.stack([Xc[:, 0], Xc[:, 1], Xc[:, 0] ** 2 + Xc[:, 1] ** 2], axis=1)
    Kz = Z @ Z.T
    az, bz, itz = smo(Kz, yc, C=1e8, tol=1e-9)
    wz = (az * yc) @ Z
    f_ker = Kz @ (az * yc) + bz; f_lin = Z @ wz + bz
    sub(f"Circle problem (11.78), 24 noise-free points: in the plane no direction separates the classes (best gap over 20001 directions {best_gap:.3f} < 0); after z = (x_1, x_2, x_1^2 + x_2^2) the hard-margin SVM separates them,")
    sub(f"      margin {1 / np.linalg.norm(wz):.5f}, w = {np.round(wz, 4).tolist()}; the output through the kernel, sum_i alpha_i y_i K(x_i, x) + b, equals w.z(x) + b to {np.abs(f_ker - f_lin).max():.1e} (11.84), and min y_i f(x_i) = {(yc * f_lin).min():.6f}.")


def gauss_K(A, B, s2):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / s2)


def check_kernel():
    head("8. Kernels: positivity (11.80), the Mercer expansion (11.82)-(11.83), Gaussian and polynomial kernels (11.85)-(11.87)")
    rg = np.random.default_rng(1108)
    # (11.78) embedding of the plane into R^3
    emb = lambda X: np.stack([X[:, 0], X[:, 1], X[:, 0] ** 2 + X[:, 1] ** 2], axis=1)
    X5 = rg.normal(size=(5, 2)); Z = emb(X5); Kc = Z @ Z.T
    ev = np.linalg.eigvalsh(Kc)
    c = np.linalg.eigh(Kc)[1][:, 0]
    sub(f"(11.80): the embedding (11.78) into R^3 has three linearly independent coordinate functions, yet for 5 distinct points the Gram matrix has rank {np.linalg.matrix_rank(Kc, tol=1e-10)} < 5,")
    sub(f"smallest eigenvalue {ev[0]:.1e}, and the combination c = (smallest eigenvector) gives sum c_i c_j K(x_i, x_j) = {c @ Kc @ c:.1e}, not > 0.  Positive SEMI-definite is all one gets in general.")
    Xg = rg.uniform(-2, 2, size=(8, 2))
    Kg = gauss_K(Xg, Xg, 1.0)
    Xd = np.vstack([Xg, Xg[:1]]); Kd = gauss_K(Xd, Xd, 1.0)
    sub(f"Gaussian kernel (infinite-dimensional feature space), 8 distinct points: smallest Gram eigenvalue {np.linalg.eigvalsh(Kg)[0]:.4f} > 0; repeating one point: {np.linalg.eigvalsh(Kd)[0]:.1e}.")
    # polynomial kernel dimension (11.87)
    out = []
    for (n, p) in ((2, 2), (2, 3), (3, 2), (3, 3)):
        Xp = rg.normal(size=(80, n)); Kp = (Xp @ Xp.T + 1) ** p
        r = int(np.linalg.matrix_rank(Kp, tol=1e-8 * np.linalg.norm(Kp, 2)))
        out.append((n, p, r, math.comb(n + p, p)))
    sub("(11.87): rank of the Gram matrix of (x.x' + 1)^p on 80 random points = number of independent monomials of degree <= p:   " + "   ".join(f"n={n}, p={p}: {r} (= C(n+p, p) = {c_})" for n, p, r, c_ in out))
    # Mercer expansion on a grid
    M = 120; h = 1.0 / M
    xg = (np.arange(M) + 0.5) * h
    s2 = 0.3 ** 2
    Kx = np.exp(-(xg[:, None] - xg[None, :]) ** 2 / s2)
    lam_K, U = np.linalg.eigh(Kx); o = np.argsort(lam_K)[::-1]; lam_K, U = lam_K[o], U[:, o]
    lam = h * lam_K                                     # eigenvalues of the integral operator (11.81)
    kf = U / math.sqrt(h)                               # orthonormal eigenfunctions, int k_i^2 dx = 1
    r = 8
    Kr_ok = (kf[:, :r] * lam[:r]) @ kf[:, :r].T
    Kr_book = (kf[:, :r] / lam[:r]) @ kf[:, :r].T       # z_i = k_i / sqrt(lambda_i), as printed in (11.83), z.z'
    full_ok = (kf * lam) @ kf.T
    err_ok = np.abs(full_ok - Kx).max()
    sub(f"Gaussian kernel on a grid of {M} points in [0,1], sigma^2 = 0.09: eigenvalues of the operator (11.81): {np.round(lam[:5], 5).tolist()} ...")
    sub(f"(11.82) with all terms reproduces K to {err_ok:.1e}.  Embedding functions z_i = sqrt(lambda_i) k_i reproduce K (z.z' = sum lambda_i k_i k_i), truncated to {r} terms with max error {np.abs(Kr_ok - Kx).max():.1e};")
    sub(f"the printed (11.83), z_i = k_i / sqrt(lambda_i), gives z.z' = sum k_i k_i / lambda_i whose largest entry is {np.abs(Kr_book).max():.1e} against max K = 1: it is wrong, the factor is sqrt(lambda_i), not its inverse.")
    STORE["mercer"] = (lam, np.abs(Kr_ok - Kx).max(), np.abs(Kr_book).max())
    # Gaussian kernel: Fourier eigenfunctions (11.86), by FFT on a circle; metric from the spectrum
    Lc, Mc, sg = 40.0, 2048, 1.0
    hc = Lc / Mc; xc = (np.arange(Mc) - Mc // 2) * hc
    row = np.exp(-(xc ** 2) / sg ** 2)
    row = np.fft.ifftshift(row)
    lam_fft = np.real(np.fft.fft(row)) * hc                  # eigenvalues of the circulant operator, eigenfunctions e^{-i w x}
    om = 2 * np.pi * np.fft.fftfreq(Mc, d=hc)
    lam_th = sg * math.sqrt(math.pi) * np.exp(-sg ** 2 * om ** 2 / 4)
    sub(f"Gaussian kernel on a circle of length {Lc:.0f} (sigma = 1): the circulant eigenvalues, from the FFT, match sigma sqrt(pi) exp(-sigma^2 w^2/4) to {np.abs(lam_fft - lam_th).max():.1e}, with eigenfunctions e^(-i w x) (11.86).")
    for K_cut in (6.0, 10.0, 14.0):
        m = np.abs(om) <= K_cut
        g_ok = np.sum(lam_th[m] * om[m] ** 2) / Lc
        g_bk = np.sum(om[m] ** 2 / lam_th[m]) / Lc
        print(f"      metric g = sum_w lambda(w) w^2 / L over |w| <= {K_cut:>4.0f}: {g_ok:.6f}  (2/sigma^2 = 2);   with the printed z = k/sqrt(lambda): {g_bk:.3e}")
    STORE["spectrum"] = (Lc, sg)


def num_mixed(Kf, x, h=1e-4):
    """Matrix of d^2 K(x, x') / dx_i dx'_j at x' = x by central differences."""
    n = len(x); E = np.eye(n) * h; G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = (Kf(x + E[i], x + E[j]) - Kf(x + E[i], x - E[j]) - Kf(x - E[i], x + E[j]) + Kf(x - E[i], x - E[j])) / (4 * h * h)
    return G


def check_metric():
    head("9. The metric induced by a kernel (11.88)-(11.98) and its conformal change")
    x = np.array([0.7, -0.4])
    # polynomial kernel of order 2 with its explicit feature map
    phi = lambda v: np.array([1.0, math.sqrt(2) * v[0], math.sqrt(2) * v[1], v[0] ** 2, v[1] ** 2, math.sqrt(2) * v[0] * v[1]])
    Kp = lambda u, v: (u @ v + 1.0) ** 2
    u = np.array([0.3, 0.9]); v = np.array([-0.5, 0.2])
    Jm = np.stack([(phi(x + 1e-6 * e) - phi(x - 1e-6 * e)) / 2e-6 for e in np.eye(2)], axis=1)    # d phi / d x_i
    g1 = Jm.T @ Jm
    g2 = num_mixed(Kp, x)
    g3 = 2 * (x @ x + 1) * np.eye(2) + 2 * np.outer(x, x)             # closed form of d^2/dx dx' (x.x'+1)^2 at x' = x
    sub(f"Polynomial kernel (x.x'+1)^2 at x = (0.7, -0.4): K = phi.phi' to {abs(Kp(u, v) - phi(u) @ phi(v)):.1e}; metric from the Jacobian of the 6-dimensional embedding (11.89): {np.round(g1, 5).tolist()};")
    sub(f"      from the mixed second derivative of the kernel (11.90): {np.round(g2, 5).tolist()}; closed form 2(|x|^2 + 1) I + 2 x x^T: {np.round(g3, 5).tolist()}  (differences {np.abs(g1 - g2).max():.1e}, {np.abs(g1 - g3).max():.1e}).")
    s2 = 0.8
    Kg = lambda u_, v_: math.exp(-np.sum((u_ - v_) ** 2) / s2)
    gg = [num_mixed(Kg, p) for p in (np.array([0.7, -0.4]), np.array([-1.3, 2.0]), np.array([0.0, 0.0]))]
    sub(f"Gaussian kernel exp(-|x-x'|^2/sigma^2), sigma^2 = 0.8: the induced metric at three points is {np.round(gg[0], 4).tolist()}, {np.round(gg[1], 4).tolist()}, {np.round(gg[2], 4).tolist()}: (2/sigma^2) I = {2 / s2:.4f} I,")
    sub("      the same everywhere: the feature manifold of the Gaussian kernel is intrinsically flat (volume element constant), embedded curved in a Hilbert space.")
    # the circle embedding (11.78)
    Kc = lambda u_, v_: u_ @ v_ + (u_ @ u_) * (v_ @ v_)
    gc = num_mixed(Kc, x)
    sub(f"Embedding (11.78) of the plane, z = (x_1, x_2, x_1^2 + x_2^2): metric {np.round(gc, 5).tolist()} = I + 4 x x^T = {np.round(np.eye(2) + 4 * np.outer(x, x), 5).tolist()}; volume element sqrt(1 + 4|x|^2) = {math.sqrt(1 + 4 * (x @ x)):.5f}, sqrt(det g) = {math.sqrt(np.linalg.det(gc)):.5f} (11.91).")
    STORE["metric_poly"] = (g1, g3)
    # conformal change (11.92)-(11.98)
    svs = np.array([[0.2, 0.5], [-0.6, -0.3], [1.0, 0.1]]); kap = np.array([1.0, 1.5, 0.7])
    sig = lambda p: float(np.sum(np.exp(-kap * np.sqrt(np.sum((p - svs) ** 2, axis=1)))))              # (11.93)
    dsig = lambda p: np.array([(sig(p + 1e-6 * e) - sig(p - 1e-6 * e)) / 2e-6 for e in np.eye(2)])
    Kti = lambda Kf: (lambda u_, v_: sig(u_) * sig(v_) * Kf(u_, v_))
    for nm, Kf, g0, Ki in (("Gaussian", Kg, np.array([[2 / s2, 0], [0, 2 / s2]]), lambda p: np.zeros(2)),
                           ("polynomial (x.x'+1)^2", Kp, g3, lambda p: 2 * (p @ p + 1) * p)):
        p = x
        gt_num = num_mixed(Kti(Kf), p)
        s0 = sig(p); ds = dsig(p); Kpp = Kf(p, p); Kif = Ki(p)
        g95 = s0 ** 2 * g0 + np.outer(ds, ds) * Kpp + s0 * (np.outer(ds, Kif) + np.outer(Kif, ds))
        g98 = s0 ** 2 * g0 + np.outer(ds, ds) * Kpp
        sub(f"{nm} kernel, sigma(x) = sum_i exp(-kappa_i |x - x_i*|): K_i(x,x) = {np.round(Kif, 5).tolist()};  finite-difference metric of K~ = sigma sigma' K vs (11.95): {np.abs(gt_num - g95).max():.1e}; vs the short form (11.98): {np.abs(gt_num - g98).max():.1e}.")
    STORE["conf_check"] = True
    # magnification of the volume element near a support vector
    for p in (np.array([0.2, 0.5]), np.array([0.2, 2.5])):
        s0 = sig(p); ds = dsig(p)
        gt = s0 ** 2 * (2 / s2) * np.eye(2) + np.outer(ds, ds)
        sub(f"Volume element of K~ at x = {p.tolist()}: sqrt(det g~) = {math.sqrt(np.linalg.det(gt)):.4f} against {2 / s2:.4f} for K (ratio {math.sqrt(np.linalg.det(gt)) / (2 / s2):.3f}); sigma = {s0:.3f}, |grad sigma| = {np.linalg.norm(ds):.3f}.")


# ------------------------------------------------------------------ 3. stochastic reasoning: belief propagation and CCCP (section 11.3)

def spin_states(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)


class Ising:
    """q(x) = exp{ h.x + sum_r w_r x_i x_j - psi } on x in {-1, +1}^n (pairwise cliques r = (i, j))."""

    def __init__(self, n, h, edges, w):
        self.n, self.h, self.edges, self.w = n, np.asarray(h, float), list(edges), np.asarray(w, float)
        self.L = len(self.edges)
        self.X = spin_states(n)
        self.E_pair = [self.X[:, i] * self.X[:, j] for (i, j) in self.edges]
        e = self.X @ self.h + sum(self.w[r] * self.E_pair[r] for r in range(self.L))
        self.q = np.exp(e - e.max()); self.q /= self.q.sum()
        self.eta = self.q @ self.X
        self.deg = np.zeros(n)
        for (i, j) in self.edges:
            self.deg[i] += 1; self.deg[j] += 1
        self.nb = {i: [] for i in range(n)}
        for r, (i, j) in enumerate(self.edges):
            self.nb[i].append((j, r)); self.nb[j].append((i, r))

    def p_r(self, r, theta_r):
        """p_r(x; theta_r) in M_r, (11.127): exp{(h + theta_r).x + c_r(x) - psi_r}."""
        e = self.X @ (self.h + theta_r) + self.w[r] * self.E_pair[r]
        p = np.exp(e - e.max()); return p / p.sum()

    def p_0(self, theta_0):
        e = self.X @ (self.h + theta_0)
        p = np.exp(e - e.max()); return p / p.sum()


def inv_two_spin(mi, mj, w):
    """Fields (a_i, a_j) and correlation c of the 2-spin model exp(a_i s + a_j t + w s t) whose node means are (m_i, m_j): the inverse m-projection onto one clique."""
    E = math.exp(4 * w)
    A = 1 - E; B = 2 * (1 + E); Cc = (1 - E) - (mi + mj) ** 2 + E * (mi - mj) ** 2
    if abs(A) < 1e-14:
        c = -Cc / B
    else:
        disc = B * B - 4 * A * Cc
        cands = [(-B + sg * math.sqrt(disc)) / (2 * A) for sg in (1, -1)]
        c = [x for x in cands if abs(x) <= 1 + 1e-12 and min(1 + mi + mj + x, 1 - mi - mj + x, 1 + mi - mj - x, 1 - mi + mj - x) >= -1e-12][0]
    P = [(1 + mi + mj + c) / 4, (1 - mi - mj + c) / 4, (1 + mi - mj - c) / 4, (1 - mi + mj - c) / 4]   # ++, --, +-, -+
    ai = 0.25 * math.log(P[0] * P[2] / (P[3] * P[1])); aj = 0.25 * math.log(P[0] * P[3] / (P[2] * P[1]))
    return ai, aj, c


def bp_geometric(M, iters):
    """The algorithm (11.133)-(11.136), literally: m-project p_r(theta_r^t) to M_0, beliefs xi_r = theta~_0r - theta_r, theta_0 = sum xi_r, theta_r = theta_0 - xi_r."""
    th0 = np.zeros(M.n); thr = [np.zeros(M.n) for _ in range(M.L)]
    hist = []; xis_hist = []
    for t in range(iters):
        xis = []
        for r in range(M.L):
            eta_r = M.p_r(r, thr[r]) @ M.X                    # m-projection to M_0 keeps E[x] (Theorem 11.6)
            xis.append(np.arctanh(eta_r) - M.h - thr[r])      # theta~_0r - theta_r  (M_0 has natural parameter h + theta_0 in actual terms)
        th0 = sum(xis)
        thr = [th0 - xis[r] for r in range(M.L)]
        hist.append(np.tanh(M.h + th0)); xis_hist.append(xis)
    return np.array(hist), th0, thr, xis_hist


def bp_messages(M, iters, damp=0.0):
    """Conventional BP (11.140)-(11.141) for binary pairwise models, messages in log-odds u_{k->i} = (1/2) log m_ki(+1)/m_ki(-1)."""
    u = {}
    for (i, j) in M.edges:
        u[(i, j)] = 0.0; u[(j, i)] = 0.0
    hist = []; u_hist = []
    for t in range(iters):
        new = {}
        for r, (i, j) in enumerate(M.edges):
            for (a, b) in ((i, j), (j, i)):
                f = M.h[a] + sum(u[(k, a)] for (k, rr) in M.nb[a] if k != b)
                new[(a, b)] = math.atanh(math.tanh(M.w[r]) * math.tanh(f))
        u = {k: damp * u[k] + (1 - damp) * new[k] for k in u}
        beta = np.array([M.h[i] + sum(u[(k, i)] for (k, rr) in M.nb[i]) for i in range(M.n)])
        hist.append(np.tanh(beta)); u_hist.append(dict(u))
    return np.array(hist), u_hist


def mean_field(M, iters=4000, damp=0.5):
    """Damped iteration of (11.125), eta_i = tanh(sum_j w_ij eta_j + h_i), from eta = tanh(h)."""
    W = np.zeros((M.n, M.n))
    for r, (i, j) in enumerate(M.edges):
        W[i, j] += M.w[r]; W[j, i] += M.w[r]
    eta = np.tanh(M.h)
    for _ in range(iters):
        new = np.tanh(W @ eta + M.h)
        if np.abs(new - eta).max() < 1e-14:
            eta = new; break
        eta = damp * eta + (1 - damp) * new
    return eta, W


def kl_pq(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def S1(m):
    b = np.array([(1 + m) / 2, (1 - m) / 2]); b = b[b > 0]
    return -float((b * np.log(b)).sum())


def S2(mi, mj, c):
    P = np.array([(1 + mi + mj + c) / 4, (1 + mi - mj - c) / 4, (1 - mi + mj - c) / 4, (1 - mi - mj + c) / 4])
    P = P[P > 1e-300]
    return -float((P * np.log(P)).sum())


def bethe(M, m, c):
    """Bethe free energy of consistent pair beliefs: F = -h.m - sum w c - sum_r S2 + sum_i (d_i - 1) S1."""
    F = -M.h @ m - sum(M.w[r] * c[r] for r in range(M.L))
    F -= sum(S2(m[i], m[j], c[r]) for r, (i, j) in enumerate(M.edges))
    F += sum((M.deg[i] - 1) * S1(m[i]) for i in range(M.n))
    return F


def node_fields(M, Theta):
    """For node parameters Theta (means m = tanh Theta): sum over incident cliques of the node fields a_{r,i} of the inverse m-projections, and the pair correlations."""
    m = np.tanh(Theta); s = np.zeros(M.n); cs = np.zeros(M.L)
    for r, (i, j) in enumerate(M.edges):
        ai, aj, c = inv_two_spin(m[i], m[j], M.w[r])
        s[i] += ai; s[j] += aj; cs[r] = c
    return s, cs


def cccp_simplified(M, iters):
    """(11.144)-(11.145): inverse m-projection of p_0(theta_0^t) to every M_r, then theta_0^{t+1} = sum_r (theta_0^t - theta_r^t).  In actual node parameters Theta = h + theta_0."""
    Theta = M.h.copy(); out = []
    for t in range(iters):
        s, cs = node_fields(M, Theta)
        m = np.tanh(Theta)
        out.append((m.copy(), bethe(M, m, cs)))
        Theta = M.h + M.deg * Theta - s
    return out, Theta


def cccp_yuille(M, iters, tol=1e-13):
    """(11.148)-(11.149): theta_0^{t+1} = L theta_0^t - sum_r theta_r^{t+1}, p_r(theta_r^{t+1}) projecting onto p_0(theta_0^{t+1}); inner equation solved by damped Newton."""
    Theta = M.h.copy(); out = []; n = M.n
    for t in range(iters):
        s, cs = node_fields(M, Theta)
        m = np.tanh(Theta)
        out.append((m.copy(), bethe(M, m, cs)))
        rhs = M.h + M.deg * Theta
        R = lambda Z: Z + node_fields(M, Z)[0] - rhs
        X = Theta.copy()
        for it in range(100):
            r0 = R(X)
            if np.abs(r0).max() < tol:
                break
            J = np.zeros((n, n))
            for k in range(n):
                e = np.zeros(n); e[k] = 1e-6
                J[:, k] = (R(X + e) - R(X - e)) / 2e-6
            dX = -np.linalg.solve(J, r0)
            lam = 1.0
            while lam > 1e-6:
                Xn = X + lam * dX
                try:
                    if np.abs(R(Xn)).max() < np.abs(r0).max() * (1 - 1e-4 * lam):
                        break
                except (ValueError, ZeroDivisionError):
                    pass
                lam /= 2
            X = Xn
        Theta = X
    return out, Theta


def make_cycle(wv):
    return Ising(4, [0.3, -0.2, 0.1, 0.25], [(0, 1), (1, 2), (2, 3), (3, 0)], [wv] * 4)


def check_mean_field():
    head("11. Mean field and the m-projection (11.107)-(11.126)")
    rg = np.random.default_rng(1110)
    # Theorem 11.6: the m-projection to independent distributions keeps E[x]; mean field is the e-projection
    n = 4
    X = spin_states(n)
    e = X @ np.array([0.4, -0.3, 0.2, 0.1]) + 0.9 * X[:, 0] * X[:, 1] + 0.7 * X[:, 1] * X[:, 2] * X[:, 3] - 0.8 * X[:, 2] * X[:, 3]
    q = np.exp(e); q /= q.sum()
    th_m, kl_m = nelder_mead(lambda th: kl_pq(q, np.exp(X @ th - np.log(np.exp(X @ th).sum()))), np.zeros(n))
    th_e, kl_e = nelder_mead(lambda th: kl_pq(np.exp(X @ th - np.log(np.exp(X @ th).sum())), q), np.zeros(n))
    pm = np.exp(X @ th_m); pm /= pm.sum(); pe = np.exp(X @ th_e); pe /= pe.sum()
    sub("A random 4-spin q with a three-body term; independent distributions M_0 = {exp(theta.x - psi)}:")
    sub(f"m-projection (minimise KL[q:p], Nelder-Mead): E_p[x] = {np.round(pm @ X, 6).tolist()} against E_q[x] = {np.round(q @ X, 6).tolist()} (Theorem 11.6, difference {np.abs(pm @ X - q @ X).max():.1e}), KL[q:p*] = {kl_m:.6f};")
    sub(f"e-projection (minimise KL[p:q]), the mean-field answer: E_p[x] = {np.round(pe @ X, 6).tolist()}, KL[p~*:q] = {kl_e:.6f}, and its KL[q:p~*] = {kl_pq(q, pe):.6f} >= {kl_m:.6f}.")
    # 2-spin ferromagnet: three solutions of (11.125)
    W = np.array([[0, 2.0], [2.0, 0]])
    hess = lambda eta, W_: np.diag(1 / (1 - eta ** 2)) - W_
    m_star = bisect(lambda m_: m_ - math.tanh(2 * m_), 0.5, 1.5)
    for eta_ in (np.array([m_star, m_star]), np.array([0.0, 0.0])):
        ev = np.linalg.eigvalsh(hess(eta_, W))
        sub(f"Two spins, w = 2, h = 0: solution eta = {np.round(eta_, 4).tolist()} of (11.125), Hessian of KL[p:q] has eigenvalues {np.round(ev, 4).tolist()} -> {'minimum' if (ev > 0).all() else 'saddle'}  (the mirror image -eta is the third solution).")
    # random instances: classification of all solutions found by Newton from many starts
    counts = {"min": 0, "saddle": 0, "max": 0}; multi = 0
    for inst in range(40):
        n = int(rg.integers(2, 6))
        Wr = rg.normal(size=(n, n)) * 1.2; Wr = (Wr + Wr.T) / 2; np.fill_diagonal(Wr, 0)
        hr = rg.normal(size=n) * 0.3
        sols = []
        for _ in range(15):
            eta = np.tanh(rg.normal(size=n) * 2)
            for it in range(40):
                gdir = np.arctanh(eta) - hr - Wr @ eta
                if np.abs(gdir).max() < 1e-13:
                    break
                d = -np.linalg.solve(hess(eta, Wr), gdir)
                lam = 1.0
                while lam > 1e-9:
                    en = eta + lam * d
                    if np.all(np.abs(en) < 1) and np.abs(np.arctanh(en) - hr - Wr @ en).max() < np.abs(gdir).max():
                        break
                    lam /= 2
                eta = en
            if np.abs(np.arctanh(eta) - hr - Wr @ eta).max() < 1e-10 and not any(np.abs(eta - s_).max() < 1e-6 for s_ in sols):
                sols.append(eta)
        multi += len(sols) > 1
        for s_ in sols:
            ev = np.linalg.eigvalsh(hess(s_, Wr))
            counts["min" if (ev > 0).all() else ("max" if (ev < 0).all() else "saddle")] += 1
    sub(f"40 random Boltzmann machines with 2-5 spins: solutions of (11.125) found from random starts: {counts['min']} minima, {counts['saddle']} saddles, {counts['max']} maxima; {multi} instances have more than one solution.")
    sub("A maximum cannot occur: the Hessian diag(1/(1 - eta_i^2)) - W has a positive trace (W has zero diagonal), so it is never negative definite; the book says 'a maximum or a saddle point'.")
    STORE["mf_counts"] = (counts, multi)


def check_bp():
    head("12. Belief propagation (11.127)-(11.143), Theorems 11.7 and 11.8")
    M = make_cycle(0.4)
    iters = 60
    hg, th0, thr, xis = bp_geometric(M, iters)
    hm, uh = bp_messages(M, iters)
    d_mean = np.abs(hg - hm).max()
    d_xi = 0.0
    for t in range(iters):
        for r, (i, j) in enumerate(M.edges):
            d_xi = max(d_xi, abs(xis[t][r][i] - uh[t][(j, i)]), abs(xis[t][r][j] - uh[t][(i, j)]))
    sub(f"Four spins on a cycle, fields h = (0.3, -0.2, 0.1, 0.25), couplings 0.4.  The geometric algorithm (11.133)-(11.136) (e/m-projections by enumeration of the 16 states) and the usual message passing")
    sub(f"(11.140)-(11.141) run in lockstep: the beliefs E[x_i] agree to {d_mean:.1e} at every step, and the belief xi_r^t of clique r on node i equals the log-odds message u_(j->i) to {d_xi:.1e},")
    sub("which is the correspondence (11.142)-(11.143): theta_0^i = sum over neighbours of the messages, theta_r^i = the same sum without the neighbour on r.")
    STORE["bp_lock"] = (d_mean, d_xi)
    # conditions at the fixed point
    p0 = M.p_0(th0); prs = [M.p_r(r, thr[r]) for r in range(M.L)]
    m_cond = max(np.abs(pr @ M.X - p0 @ M.X).max() for pr in prs)
    e_cond = np.abs((M.L - 1) * th0 - sum(thr)).max()
    res = np.log(M.q) - sum(np.log(pr) for pr in prs) + (M.L - 1) * np.log(p0)
    sub(f"Fixed point after {iters} steps: m-condition max_r |E_r[x] - E_0[x]| = {m_cond:.1e}, e-condition |(L-1) theta_0 - sum theta_r| = {e_cond:.1e} (Theorem 11.7);")
    sub(f"q lies in E*: log q - sum_r log p_r + (L-1) log p_0 is constant over all 16 states (spread {res.max() - res.min():.1e}) (Corollary).  BP beliefs {np.round(p0 @ M.X, 5).tolist()}; exact marginals {np.round(M.eta, 5).tolist()};")
    sub(f"      the loop makes BP wrong: largest error {np.abs(p0 @ M.X - M.eta).max():.4f}.")
    # tree: Theorem 11.8
    T = Ising(3, [0.2, -0.1, 0.3], [(0, 1), (1, 2)], [0.9, 0.7])
    hg_t, th0_t, thr_t, _ = bp_geometric(T, 80)
    p0t = T.p_0(th0_t); prt = [T.p_r(r, thr_t[r]) for r in range(T.L)]
    A = np.stack([pr - p0t for pr in prt], axis=1)
    tt, *_ = np.linalg.lstsq(A, T.q - p0t, rcond=None)
    resid = np.abs(A @ tt - (T.q - p0t)).max()
    cov = float(T.q @ (T.X[:, 0] * T.X[:, 2]) - T.eta[0] * T.eta[2])
    cov_M = [float(pr @ (T.X[:, 0] * T.X[:, 2]) - (pr @ T.X[:, 0]) * (pr @ T.X[:, 2])) for pr in prt + [p0t]]
    sub(f"Chain x_0 - x_1 - x_2 (a tree), h = (0.2, -0.1, 0.3), couplings 0.9, 0.7: BP marginals {np.round(hg_t[-1], 5).tolist()} = exact {np.round(T.eta, 5).tolist()} (difference {np.abs(hg_t[-1] - T.eta).max():.1e}): the answer is exact, as Theorem 11.8 says.")
    sub(f"But 'M* includes q' is false: the best affine combination t_1 p_1 + t_2 p_2 + (1 - t_1 - t_2) p_0 misses q by {resid:.4f} in sup norm (|q - p_0| = {np.abs(T.q - p0t).max():.4f});")
    sub(f"      every p_r and p_0 has Cov(x_0, x_2) = {max(abs(c_) for c_ in cov_M):.1e} (the end spins are independent in each member), so every point of M* does, while q has Cov(x_0, x_2) = {cov:.4f}.")
    ok = []
    for r, (i, j) in enumerate(T.edges):
        ok.append((abs(T.q @ (T.X[:, i] * T.X[:, j]) - prt[r] @ (T.X[:, i] * T.X[:, j])), np.abs(T.q @ T.X - prt[r] @ T.X).max()))
    sub(f"What does hold on a tree: p_r* is the m-projection of q onto M_r (it matches E_q[x] and E_q[x_i x_j] of its own clique): differences {max(o[0] for o in ok):.1e}, {max(o[1] for o in ok):.1e}.")
    STORE["tree"] = (resid, cov, np.abs(T.q - p0t).max())
    res_ = []
    for edges_, w_ in (([(0, 1)], [0.9]), ([(0, 1), (2, 3)], [0.9, 0.7])):
        Mx = Ising(4, [0.2, -0.1, 0.3, 0.1], edges_, w_)
        _, th0x, thrx, _ = bp_geometric(Mx, 200)
        p0x = Mx.p_0(th0x); prx = [Mx.p_r(r, thrx[r]) for r in range(Mx.L)]
        Ax = np.stack([pr - p0x for pr in prx], axis=1)
        tx, *_ = np.linalg.lstsq(Ax, Mx.q - p0x, rcond=None)
        res_.append(np.abs(Ax @ tx - (Mx.q - p0x)).max())
    sub(f"Fewer edges: for a single edge q lies in M* (residual {res_[0]:.1e}); for two disjoint edges it does not (residual {res_[1]:.4f}): the product of the two correlations is a four-spin term that no combination of p_1, p_2, p_0 contains.")
    sub("      So 'M* includes q' holds only for one interacting clique.")
    STORE["mstar_small"] = res_
    # the same equivalence and conditions on the graph of Fig. 11.8 (7 spins, 12 edges, random couplings)
    edges118 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (2, 5), (4, 5), (3, 5), (3, 6)]
    rg7 = np.random.default_rng(1180)
    G7 = Ising(7, rg7.normal(0, 0.3, 7), edges118, rg7.normal(0, 0.3, 12))
    hg7, th07, thr7, xis7 = bp_geometric(G7, 40)
    hm7, uh7 = bp_messages(G7, 40)
    p07 = G7.p_0(th07); pr7 = [G7.p_r(r, thr7[r]) for r in range(G7.L)]
    res7 = np.log(G7.q) - sum(np.log(pr) for pr in pr7) + (G7.L - 1) * np.log(p07)
    sub(f"Graph of Fig. 11.8 with random couplings N(0, 0.3^2): geometric BP and message passing agree to {np.abs(hg7 - hm7).max():.1e} at all 40 steps; m-condition {max(np.abs(pr @ G7.X - p07 @ G7.X).max() for pr in pr7):.1e},")
    sub(f"      e-condition {np.abs((G7.L - 1) * th07 - sum(thr7)).max():.1e}, spread of log q - sum log p_r + (L-1) log p_0 over the 128 states {res7.max() - res7.min():.1e}; BP error {np.abs(p07 @ G7.X - G7.eta).max():.4f}.")
    # accuracy of BP and mean field on a loop
    sub("Accuracy on the 4-cycle (fields as above) against exact marginals, largest error over the four spins:")
    print("      coupling   BP        mean field (11.125)   BP iterations to 1e-12")
    sweep = []
    for wv in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        Mc = make_cycle(wv)
        hb, _ = bp_messages(Mc, 3000)
        d = np.abs(np.diff(hb, axis=0)).max(1)
        nit = int(np.argmax(d < 1e-12)) + 1 if (d < 1e-12).any() else -1
        eb = np.abs(hb[-1] - Mc.eta).max()
        em = np.abs(mean_field(Mc)[0] - Mc.eta).max()
        sweep.append((wv, eb, em, nit))
        print(f"      {wv:<10.1f} {eb:<9.4f} {em:<21.4f} {nit}")
    STORE["bp_sweep"] = sweep
    # Fig. 11.8: seven nodes, twelve edges, random couplings
    edges118 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (2, 5), (4, 5), (3, 5), (3, 6)]
    sub("The graph of Fig. 11.8 (7 spins, 12 edges: K4 on {1,2,3,4}, wheel around x_3, pendant x_7), random couplings N(0, s^2), fields N(0, 0.3^2), 30 draws each:")
    print("      s      BP mean |error|    mean field mean |error|    BP closer in")
    rg2 = np.random.default_rng(118)
    tab = []
    for sc in (0.1, 0.2, 0.3, 0.5):
        eb, em, cnt = [], [], 0
        for _ in range(30):
            Mg = Ising(7, rg2.normal(0, 0.3, 7), edges118, rg2.normal(0, sc, 12))
            hb, _ = bp_messages(Mg, 400, damp=0.3)
            b_err = np.abs(hb[-1] - Mg.eta).mean(); m_err = np.abs(mean_field(Mg)[0] - Mg.eta).mean()
            eb.append(b_err); em.append(m_err); cnt += b_err < m_err
        tab.append((sc, np.mean(eb), np.mean(em), cnt))
        print(f"      {sc:<6.1f} {np.mean(eb):<18.4f} {np.mean(em):<26.4f} {cnt} of 30")
    STORE["bp118"] = tab
    # (11.101): the decomposition over maximal cliques is not unique either
    x3 = spin_states(3)
    f12 = lambda a_, b_: np.exp(0.9 * a_ * b_ + 0.2 * a_); f23 = lambda a_, b_: np.exp(0.7 * a_ * b_ - 0.3 * b_); gg = lambda b_: np.exp(0.8 * b_ + 0.1)
    p_a = f12(x3[:, 0], x3[:, 1]) * f23(x3[:, 1], x3[:, 2])
    p_b = (f12(x3[:, 0], x3[:, 1]) * gg(x3[:, 1])) * (f23(x3[:, 1], x3[:, 2]) / gg(x3[:, 1]))
    sub(f"(11.101): 'unique when only maximal cliques are used' is not so: multiplying the factor of clique (x_0, x_1) by g(x_1) and dividing that of (x_1, x_2) by g(x_1) changes both factors and not their product (difference {np.abs(p_a - p_b).max():.1e}).")


def check_cccp():
    head("13. CCCP (11.144)-(11.149) against BP on the 4-cycle")
    print("      coupling   BP: converged by   simplified CCCP (11.144)-(11.145)                           Yuille's CCCP (11.148)-(11.149)")
    print("                                    steps with F up   F after 400 steps    |m - BP|              steps with F up   F after 60 steps   |m - BP| after 60")
    rows = []
    for wv in (0.3, 0.5, 0.8, 1.2):
        Mc = make_cycle(wv)
        hb, _ = bp_messages(Mc, 3000)
        d = np.abs(np.diff(hb, axis=0)).max(1); nit = int(np.argmax(d < 1e-12)) + 1
        o1, Th1 = cccp_simplified(Mc, 400)
        F1 = np.array([o[1] for o in o1]); up1 = int((np.diff(F1) > 1e-12).sum())
        o2, Th2 = cccp_yuille(Mc, 60)
        F2 = np.array([o[1] for o in o2]); up2 = int((np.diff(F2) > 1e-12).sum())
        e1 = np.abs(o1[-1][0] - hb[-1]).max(); e2 = np.abs(o2[-1][0] - hb[-1]).max()
        rows.append((wv, nit, up1, F1[-1], e1, up2, F2[-1], e2))
        print(f"      {wv:<10.1f} {nit:<18d} {up1:<17d} {F1[-1]:<18.4f} {e1:<21.2e} {up2:<17d} {F2[-1]:<19.4f} {e2:.1e}")
    STORE["cccp_rows"] = rows
    # traces for the figure at w = 0.8
    Mc = make_cycle(0.8)
    o1, _ = cccp_simplified(Mc, 40); o2, _ = cccp_yuille(Mc, 40); hb, _ = bp_messages(Mc, 40)
    STORE["cccp_trace"] = ([o[0][0] for o in o1], [o[1] for o in o1], [o[0][0] for o in o2], [o[1] for o in o2], list(hb[:, 0]))
    o1b, _ = cccp_simplified(Mc, 400); hbb, _ = bp_messages(Mc, 3000)
    last = [o[0][0] for o in o1b[300:]]
    sub(f"At w = 0.8 the belief of spin 0 under the simplified CCCP alternates between {min(last):.4f} and {max(last):.4f} over steps 300-399; BP settles at {hbb[-1][0]:.4f} (exact marginal {Mc.eta[0]:.4f}).")
    sub(f"      All four spins at w = 0.8: exact marginals {np.round(Mc.eta, 4).tolist()}, BP fixed point {np.round(hbb[-1], 4).tolist()}.")
    sub("Yuille's version (inner equation solved exactly) lowers the Bethe free energy at every step for every coupling and creeps to the BP fixed point; the simplified version, which")
    sub("is the same iteration with the inner loop stopped after one step, converges only for weak coupling and otherwise oscillates with the free energy going up and down.")
    # linear stability at the symmetric point h = 0
    print("      Linear stability at h = 0 (fixed point Theta = 0) of the 4-cycle, spectral radius of the one-step map:")
    print("      coupling   tanh w    simplified CCCP    2t/(1-t)     BP (messages)")
    stab = []
    for wv in (0.30, 0.34, 0.35, 0.40, 0.60):
        Mz = Ising(4, np.zeros(4), [(0, 1), (1, 2), (2, 3), (3, 0)], [wv] * 4)
        Fmap = lambda Th: Mz.h + Mz.deg * Th - node_fields(Mz, Th)[0]
        J = np.zeros((4, 4))
        for k in range(4):
            e = np.zeros(4); e[k] = 1e-5
            J[:, k] = (Fmap(e) - Fmap(-e)) / 2e-5
        rho = np.abs(np.linalg.eigvals(J)).max()
        t = math.tanh(wv)
        # BP: message map u'_{j->i} = f(u_{k->j}) for the 8 directed messages, linearised: d u'/d u = t on the two incoming-to-j terms
        idx = {}
        for (i, j) in Mz.edges:
            idx[(i, j)] = len(idx); idx[(j, i)] = len(idx)
        JB = np.zeros((len(idx), len(idx)))
        for (a, b), ia in idx.items():
            for (k, rr) in Mz.nb[a]:
                if k != b:
                    JB[ia, idx[(k, a)]] = t
        rb = np.abs(np.linalg.eigvals(JB)).max()
        stab.append((wv, t, rho, 2 * t / (1 - t), rb))
        print(f"      {wv:<10.2f} {t:<9.4f} {rho:<18.4f} {2 * t / (1 - t):<12.4f} {rb:.4f}")
    STORE["cccp_stab"] = stab
    sub(f"The alternating mode of the simplified iteration multiplies by -2t/(1-t), t = tanh w: unstable beyond tanh w = 1/3, w = atanh(1/3) = {math.atanh(1 / 3):.5f}; the BP map on this cycle has radius tanh w < 1 for every w.")
    # BP fixed points are stationary points of the Bethe free energy
    Mc = make_cycle(0.5)
    hg, th0_, thr_, _ = bp_geometric(Mc, 300)
    m_b = np.tanh(Mc.h + th0_)
    c_b = np.array([float(Mc.p_r(r, thr_[r]) @ Mc.E_pair[r]) for r in range(Mc.L)])
    Fv = lambda z: bethe(Mc, z[:4], z[4:])
    z0 = np.r_[m_b, c_b]
    gr = np.array([(Fv(z0 + 1e-6 * e) - Fv(z0 - 1e-6 * e)) / 2e-6 for e in np.eye(8)])
    sub(f"Bethe free energy at the BP fixed point (w = 0.5): F = {Fv(z0):.6f}, finite-difference gradient in (m, c) {np.abs(gr).max():.1e}: BP fixed points are stationary points, as the book quotes from Yedidia et al.")
    STORE["bethe_grad"] = np.abs(gr).max()


# ------------------------------------------------------------------ 4. information geometry of boosting (section 11.4)

def toy_boost(seed=1114, N=20):
    rg = np.random.default_rng(seed)
    X = np.round(rg.uniform(0, 1, size=(N, 2)), 4)
    y = np.where(X[:, 0] + X[:, 1] - 1 + 0.12 * rg.normal(size=N) > 0, 1.0, -1.0)
    return X, y


def stump_pool(X):
    """All decision stumps h(x) = s * sign(x_k - thr) with thresholds between sorted values; returns the (pool size, N) matrix of outputs and descriptions."""
    H, desc = [], []
    for k in range(X.shape[1]):
        v = np.sort(X[:, k]); thr = 0.5 * (v[1:] + v[:-1])
        for t_ in thr:
            for s_ in (1.0, -1.0):
                H.append(s_ * np.where(X[:, k] > t_, 1.0, -1.0)); desc.append((k, float(t_), s_))
    return np.array(H), desc


def logistic_alpha(yF, yh, a0=0.0):
    """Minimise sum_i log(1 + exp(-y_i (F_i + a h_i))) over a by Newton (the exact, normalised m-projection onto E_{t+1} of (11.164))."""
    a = a0
    for _ in range(50):
        z = yF + a * yh
        s = 1 / (1 + np.exp(z))                    # derivative weights
        g = -np.sum(yh * s); H = np.sum(yh * yh * s * (1 - s))
        step = g / max(H, 1e-12)
        a -= step
        if abs(step) < 1e-13:
            break
    return a


def check_boosting():
    head("14. Boosting (11.150)-(11.175)")
    X, y = toy_boost(); N = len(y)
    H, desc = stump_pool(X)
    sub("points (x_1, x_2, y): " + str([[round(float(v), 4) for v in X[i]] + [int(y[i])] for i in range(N)]))
    sub(f"Toy problem: {N} points in the unit square, label sign(x_1 + x_2 - 1 + noise); {len(H)} decision stumps as weak machines; AdaBoost with exact weights W_t (no resampling).")
    # (11.153)-(11.157): the constant c' is not constant
    Fv = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    cp = 1 / (1 + np.exp(-Fv))                  # c' = c exp(y* F / 2) with y* = +1; c = 1/(2 cosh(F/2))
    c_chk = np.exp(Fv / 2) / (2 * np.cosh(Fv / 2))
    sub(f"(11.155): c' = c exp(y* F/2) = 1/(1 + exp(-y* F)): for y* F = {Fv.tolist()} it is {np.round(cp, 4).tolist()} (direct: {np.round(c_chk, 4).tolist()}). It depends on x, so neglecting it in (11.157)")
    sub(f"      replaces the model's error probability c' e^(-y*F) = 1/(1 + e^(y*F)) by e^(-y*F): the ratio is 1 + e^(-y*F) = {np.round(1 + np.exp(-Fv), 3).tolist()}, not a common constant.")
    F = np.zeros(N); rounds = []; prodZ = 1.0
    W = np.ones(N) / N
    for t in range(1, 11):
        werr = ((H != y[None, :]) * W[None, :]).sum(1)         # weighted error of each stump
        k = int(np.argmax(werr <= werr.min() + 1e-12)); eps = float(werr[k]); h = H[k]        # first stump within 1e-12 of the minimum, so that ties are broken the same way everywhere
        alpha = 0.5 * math.log((1 - eps) / eps)                # (11.174)
        # exact minimiser of the exponential loss along the line, by bisection on (11.170)
        d_loss = lambda a: float(np.sum(W * y * h * np.exp(-a * y * h)))
        a_num = bisect(d_loss, 0.0, 6.0) if d_loss(0.0) > 0 else 0.0
        Wn = W * np.exp(-alpha * y * h); Zt = Wn.sum(); Wn /= Zt            # (11.175)
        a_ml = logistic_alpha(y * F, y * h)
        rounds.append(dict(t=t, desc=desc[k], eps=eps, alpha=alpha, a_num=a_num, Z=Zt, edge_next=float(np.sum(Wn * y * h)), a_ml=a_ml, W=W.copy(), Wn=Wn.copy(), h=h.copy()))
        F = F + alpha * h; prodZ *= Zt
        rounds[-1].update(err=float(np.mean(np.sign(F) != y)), prodZ=prodZ, expl=float(np.mean(np.exp(-y * F))), KLnext=kl_pq(Wn, W))
        W = Wn
    print("      t  stump (feature, threshold, sign)   eps_t     alpha_t (11.174)  numeric argmin  Z_t = 2 sqrt(eps(1-eps))   prod Z   train err   mean exp(-yF)   E_(W_t+1)[y h_t]   exact (logistic) alpha")
    for r in rounds:
        print(f"      {r['t']:<2d} ({r['desc'][0]}, {r['desc'][1]:.3f}, {r['desc'][2]:+.0f})            {r['eps']:.5f}   {r['alpha']:.6f}          {r['a_num']:.6f}        {r['Z']:.6f} = {2 * math.sqrt(r['eps'] * (1 - r['eps'])):.6f}      {r['prodZ']:.5f}   {r['err']:.3f}       {r['expl']:.5f}         {r['edge_next']:+.1e}          {r['a_ml']:.5f}")
    STORE["boost"] = rounds
    r1 = rounds[0]
    sub(f"Round 1 (F = 0): the exponential-loss line search gives alpha = (1/2) log((1-eps)/eps) = {r1['alpha']:.6f} (numerical argmin {r1['a_num']:.6f}); the exact m-projection onto the normalised family (11.164),")
    sub(f"      i.e. maximum likelihood of q(y|x) = c exp(yF/2), gives alpha = log((1-eps)/eps) = {r1['a_ml']:.6f}: exactly twice.  (11.169) comes from projecting onto the UNNORMALISED family (11.165); the text just before speaks of the m-projection onto the (normalised) exponential family E_(t+1), which gives this larger weight.")
    sub(f"      Later rounds differ too: AdaBoost alpha_t {[round(r['alpha'], 3) for r in rounds[1:5]]} against maximum likelihood {[round(float(r['a_ml']), 3) for r in rounds[1:5]]}.")
    # (11.168) -> (11.169): generalised KL of the empirical distribution to the unnormalised family
    r = rounds[2]; Ft = np.zeros(N)
    for q_ in rounds[:2]:
        Ft += q_["alpha"] * q_["h"]
    pemp = np.ones(N) / N
    def gen_kl(al):
        Fn = Ft + al * r["h"]
        # q~(y|x_i) = exp((y - y_i) F/2): equal to 1 at y = y_i (so log q~(y_i|x_i) = 0) and exp(-y_i F) at y = -y_i
        return float(np.sum(pemp * np.log(pemp) - pemp) + np.sum(1 + np.exp(-y * Fn)))
    d_kl = gen_kl(0.37) - gen_kl(1.0)
    d_ex = float(np.sum(np.exp(-y * (Ft + 0.37 * r["h"]))) - np.sum(np.exp(-y * (Ft + 1.0 * r["h"]))))
    sub(f"(11.168): the generalised KL[p_emp : q~] of the empirical distribution to q~(y|x) = exp((y - y*) F/2) differs between alpha = 0.37 and alpha = 1.0 by {d_kl:.6f}; the exponential loss (11.169) differs by {d_ex:.6f}:")
    sub("      the first sum in (11.168) vanishes identically (y_i = y*(x_i)), the rest is sum_i exp(-y_i F(x_i)) plus a constant, so the m-projection onto the unnormalised family is exactly the exponential-loss line search.")
    # (11.170)-(11.175) consequences; projection viewpoint
    rr = rounds[3]
    rg = np.random.default_rng(5)
    errs = []
    while len(errs) < 5:
        Wf = rg.dirichlet(np.ones(N)); Wo = rg.dirichlet(np.ones(N) * 0.5)
        e1 = float(Wf @ (y * rr["h"])); e2 = float(Wo @ (y * rr["h"]))
        if e1 * e2 < 0:
            lam = e1 / (e1 - e2); Wh = (1 - lam) * Wf + lam * Wo           # a point of the hyperplane {E_W[y h] = 0}
            errs.append(abs(kl_pq(Wh, rr["W"]) - kl_pq(Wh, rr["Wn"]) - kl_pq(rr["Wn"], rr["W"])))
    sub(f"Projection viewpoint (not in the book): W_(t+1) = W_t e^(-alpha y h)/Z is the e-projection (minimiser of KL[W:W_t]) onto the m-flat set {{W: E_W[y h_t] = 0}}; the Pythagorean identity")
    sub(f"      KL[W:W_t] = KL[W:W_(t+1)] + KL[W_(t+1):W_t] holds to {max(errs):.1e} for {len(errs)} random W in that set (round 4).")
    # training-error bound
    last = rounds[-1]
    sub(f"Training error after {last['t']} rounds: {last['err']:.3f}; the bound prod Z_t = {last['prodZ']:.5f} equals the mean exponential loss {last['expl']:.5f} and bounds the error from above at every round.")


# ------------------------------------------------------------------ 5. Bayesian inference and deep learning (section 11.5)

def check_bayes():
    head("15. Bayesian duality and conjugate priors (11.176)-(11.188)")
    # a non-conjugate prior: x | theta ~ N(theta, 1), Laplace prior pi(theta) = exp(-|theta|)/2 on a grid
    th = np.linspace(-14, 14, 280001); dth = th[1] - th[0]
    pi = 0.5 * np.exp(-np.abs(th))
    psi_bar = 0.5 * th ** 2 - np.log(pi)                                      # (11.179), psi = theta^2/2 up to a constant
    kbar = lambda x: 0.5 * x * x                                              # underlying measure exp(-k_bar) dx; the constant sqrt(2 pi) is absorbed in psi
    def k_of(x):
        return math.log(float(np.sum(np.exp(th * x - psi_bar)) * dth))        # the log-partition function of the posterior family in theta, natural parameter x: (11.180)
    def post_mean(x):
        w = np.exp(th * x - psi_bar); return float(np.sum(th * w) / np.sum(w))
    def log_px(x):
        lik = np.exp(-0.5 * (x - th) ** 2) / math.sqrt(2 * math.pi)
        return math.log(float(np.sum(lik * pi) * dth))
    xs = (-1.5, 0.4, 2.2)
    h_ = 1e-4
    rows = []
    for x in xs:
        dk = (k_of(x + h_) - k_of(x - h_)) / (2 * h_)
        tw = x + (log_px(x + h_) - log_px(x - h_)) / (2 * h_)               # Tweedie: k'(x) = kbar'(x) + d/dx log p(x), kbar' = x
        rows.append((x, post_mean(x), dk, tw, k_of(x) - (kbar(x) + log_px(x))))
    sub("x | theta ~ N(theta, 1) (k_bar = x^2/2, psi = theta^2/2), Laplace prior on theta; the posterior is an exponential family in theta with natural parameter x and log-partition function k(x) (11.180)-(11.181):")
    print("      x        E_x[theta] = theta* (11.184)   dk/dx (finite diff.)   x + d log p(x)/dx (Tweedie)   k(x) - k_bar(x) - log p(x)")
    for x, a, b, c, d in rows:
        print(f"      {x:<8.2f} {a:<29.8f} {b:<21.8f} {c:<28.8f} {d:.8f}")
    cst = [r_[4] for r_ in rows]
    sub(f"So the m-affine coordinate of the posterior family is the gradient of k, and (11.181) holds with the same constant at every x (spread {max(cst) - min(cst):.1e}; the constant is log sqrt(2 pi) = {0.5 * math.log(2 * math.pi):.6f},")
    sub("the Gaussian normalisation I dropped from k_bar).  The equality k' = x + (log p)' is Tweedie's formula.")
    STORE["bayes_rows"] = rows
    # Poisson with a Gamma prior on lambda = e^theta: conjugate form (11.185)
    a0, b0, N, xbar = 2.0, 4.0, 10, 3.0
    chi = lambda a, b: math.lgamma(a) - a * math.log(b)                        # log integral of exp(a theta - b e^theta) dtheta
    th2 = np.linspace(-12, 8, 400001); d2 = th2[1] - th2[0]
    printed = np.exp(th2 * (a0 + N * xbar) - (N + b0) * np.exp(th2) - chi(a0, b0))          # (11.186) as printed
    correct = np.exp(th2 * (a0 + N * xbar) - (N + b0) * np.exp(th2) - chi(a0 + N * xbar, b0 + N))
    I_pr = float(np.sum(printed) * d2); I_co = float(np.sum(correct) * d2)
    mode_th = th2[int(np.argmax(correct))]
    sub(f"Poisson likelihood with natural parameter theta = log lambda, conjugate prior exp(alpha theta - beta e^theta - chi), alpha = {a0:.0f}, beta = {b0:.0f}, N = {N}, xbar = {xbar:.0f}:")
    sub(f"(11.186) with the prior's normaliser chi(alpha, beta): integrates to {I_pr:.3e}, not 1; with chi(alpha + N xbar, beta + N) (the update (11.188)) it integrates to {I_co:.6f}.")
    sub(f"The posterior mode has e^theta = {math.exp(mode_th):.4f} = (alpha + N xbar)/(N + beta) = {(a0 + N * xbar) / (N + b0):.4f}: the prior acts as beta = {b0:.0f} extra observations of value alpha/beta = {a0 / b0:.2f};")
    sub(f"      'shifting the observed point from xbar to xbar + alpha/N' would give {xbar + a0 / N:.4f}, which is right only to first order and only when beta is small.")
    STORE["conj"] = (I_pr, I_co, (a0 + N * xbar) / (N + b0), xbar + a0 / N)
    # sequential updating (11.188): batches on a grid
    lam = np.exp(th2)
    lik = lambda xs_: np.exp(th2 * np.sum(xs_) - len(xs_) * lam)             # Poisson likelihood (without the 1/x! factors), natural parameter theta
    prior = np.exp(a0 * th2 - b0 * lam - chi(a0, b0))
    b1 = np.array([2.0, 3.0, 2.0, 3.0]); b2 = np.array([4.0, 3.0, 2.0, 4.0, 3.0, 4.0])
    p1 = prior * lik(b1); p1 /= p1.sum() * d2
    p12 = p1 * lik(b2); p12 /= p12.sum() * d2
    pall = prior * lik(np.concatenate([b1, b2])); pall /= pall.sum() * d2
    sub(f"Two batches (N = 4 then N = 6) against all ten observations at once, on a grid in theta: the posteriors agree to {np.abs(p12 - pall).max():.1e}; the hyperparameters move by (alpha, beta) -> (alpha + N xbar, beta + N) = ({a0 + b1.sum() + b2.sum():.0f}, {b0 + 10:.0f}) (11.188).")


def rbm_states(n_v=3, n_h=2):
    V = np.array(list(itertools.product([0.0, 1.0], repeat=n_v))); Hh = np.array(list(itertools.product([0.0, 1.0], repeat=n_h)))
    return V, Hh


class RBM:
    """Binary restricted Boltzmann machine, p(v, h; W) = exp{h' W v - psi(W)} (biases 0, as in (11.192) after the simplification of the text)."""

    def __init__(self, W):
        self.W = np.asarray(W, float); self.nh, self.nv = self.W.shape
        self.V, self.Hh = rbm_states(self.nv, self.nh)
        E = np.einsum("hi,ij,vj->vh", self.Hh, self.W, self.V)       # (v, h) energies h'Wv
        P = np.exp(E - E.max()); self.P = P / P.sum()
        self.pV = self.P.sum(1); self.pH = self.P.sum(0)
        self.feat = np.einsum("hi,vj->vhij", self.Hh, self.V)        # f_ij(v, h) = h_i v_j

    def exp_f(self, P):
        return np.einsum("vh,vhij->ij", P, self.feat)

    def grad_kl(self, q):
        """d KL[q : p_V(W)] / dW = -<h v^T>_{q(v) p(h|v)} + <h v^T>_p  (11.209)."""
        cond = self.P / self.pV[:, None]
        return -self.exp_f(q[:, None] * cond) + self.exp_f(self.P)

    def kl_marg(self, q):
        return kl_pq(q, self.pV)


def check_rbm():
    head("16. The restricted Boltzmann machine: Theorems 11.9 and 11.10, learning rule (11.206), natural gradient")
    rg = np.random.default_rng(1115)
    R = RBM(rg.normal(0, 0.9, size=(2, 3)))
    q = rg.dirichlet(np.ones(8) * 1.5)
    nV, nH = 8, 4
    # marginals are not exponential: log p_V is not affine in W
    W1 = R.W; W2 = rg.normal(0, 0.9, size=R.W.shape)
    lp = lambda W_: np.log(RBM(W_).pV)
    lj = lambda W_: np.log(RBM(W_).P)
    # log p(v,h) = h'Wv - psi(W) is affine in W up to the constant psi: remove it by subtracting the entry (0, 0) before taking the second difference
    d_joint = np.abs((lj(W1) - lj(W1)[0, 0]) - 2 * (lj(0.5 * (W1 + W2)) - lj(0.5 * (W1 + W2))[0, 0]) + (lj(W2) - lj(W2)[0, 0])).max()
    d_marg = np.abs((lp(W1) - lp(W1)[0]) - 2 * (lp(0.5 * (W1 + W2)) - lp(0.5 * (W1 + W2))[0]) + (lp(W2) - lp(W2)[0])).max()
    sub(f"Random RBM, 3 visible and 2 hidden binary units.  p(v,h) is an exponential family in W: log-probability ratios are affine in W (second difference {d_joint:.1e});")
    sub(f"the marginal p_V is not: log p_V(v) = sum_j log(1 + exp((Wv)_j)) - psi has second difference {d_marg:.3f} (11.193).")
    # chain rule (11.205)
    r_cond = rg.dirichlet(np.ones(nH), size=nV)                                  # r(h|v)
    lhs = kl_pq((q[:, None] * r_cond).ravel(), R.P.ravel())
    rhs = kl_pq(q, R.pV) + float(sum(q[v] * kl_pq(r_cond[v], R.P[v] / R.pV[v]) for v in range(nV)))
    sub(f"(11.205): KL[q(v) r(h|v) : p(v,h)] = {lhs:.10f} and KL[q:p_V] + E_q KL[r : p(h|v)] = {rhs:.10f}; the minimum over r is at r = p(h|v) with value KL[q:p_V] = {kl_pq(q, R.pV):.6f}.")
    # (11.209): gradient
    g = R.grad_kl(q)
    fd = np.zeros_like(R.W)
    for i in range(R.nh):
        for j in range(R.nv):
            E = np.zeros_like(R.W); E[i, j] = 1e-6
            fd[i, j] = (RBM(R.W + E).kl_marg(q) - RBM(R.W - E).kl_marg(q)) / 2e-6
    sub(f"(11.209)-(11.210): the gradient of KL[q:p_V] is -<h v'>_(q p(h|v)) + <h v'>_p; finite differences agree to {np.abs(fd - g).max():.1e}, so the learning rule (11.206) is gradient descent.")
    # Fisher information versus Hessian; natural gradient remark
    cond = R.P / R.pV[:, None]
    f = R.feat.reshape(nV, nH, -1)
    mean_p = np.einsum("vh,vhk->k", R.P, f)
    G_joint = np.einsum("vh,vhk,vhl->kl", R.P, f, f) - np.outer(mean_p, mean_p)
    mean_c = np.einsum("vh,vhk->vk", cond, f)
    cov_c = np.einsum("vh,vhk,vhl->vkl", cond, f, f) - np.einsum("vk,vl->vkl", mean_c, mean_c)
    score = mean_c - mean_p[None, :]
    G_V = np.einsum("v,vk,vl->kl", R.pV, score, score)
    H_q = G_joint - np.einsum("v,vkl->kl", q, cov_c)
    Hnum = np.zeros((6, 6))
    for k in range(6):
        E = np.zeros(6); E[k] = 1e-5
        Hnum[:, k] = (RBM(R.W + E.reshape(R.W.shape)).grad_kl(q) - RBM(R.W - E.reshape(R.W.shape)).grad_kl(q)).ravel() / 2e-5
    sub(f"Hessian of KL[q:p_V] in W: finite differences vs G_joint - E_q Cov(h v' | v): {np.abs(Hnum - H_q).max():.1e}.")
    # realisable target: the Hessian is the Fisher information of the marginal
    Hreal = G_joint - np.einsum("v,vkl->kl", R.pV, cov_c)
    sub(f"If q = p_V(W) (realisable), the Hessian equals the Fisher information of the marginal: |G_V - (G_joint - E Cov)| = {np.abs(G_V - Hreal).max():.1e}; the joint's Fisher matrix G_joint is larger: smallest eigenvalue of G_joint - G_V = {np.linalg.eigvalsh(G_joint - G_V)[0]:.4f} >= 0.")
    sub(f"      eigenvalues of G_V: {np.round(np.linalg.eigvalsh(G_V), 5).tolist()}.")
    # gradient descent vs natural gradient on a realisable target
    Wstar = rg.normal(0, 0.9, size=(2, 3)); Rs = RBM(Wstar); qs = Rs.pV
    def fisher_marginal(Rc):
        condc = Rc.P / Rc.pV[:, None]; fc = Rc.feat.reshape(nV, nH, -1)
        mp_ = np.einsum("vh,vhk->k", Rc.P, fc); mc_ = np.einsum("vh,vhk->vk", condc, fc)
        return np.einsum("v,vk,vl->kl", Rc.pV, mc_ - mp_[None, :], mc_ - mp_[None, :])
    GVt = fisher_marginal(Rs)
    lam_max = np.linalg.eigvalsh(GVt)[-1]; lam_min = np.linalg.eigvalsh(GVt)[0]
    def run(W0, natural, step, iters=20000, tol=1e-12):
        W_ = W0.copy(); hist = []
        for it in range(iters):
            Rc = RBM(W_); klv = Rc.kl_marg(qs); hist.append(klv)
            if klv < tol:
                break
            gg = Rc.grad_kl(qs).ravel()
            d = np.linalg.solve(fisher_marginal(Rc) + 1e-9 * np.eye(6), gg) if natural else gg / lam_max
            W_ = W_ - step * d.reshape(W_.shape)
        return it, hist
    W0 = Wstar + 0.25 * rg.normal(size=Wstar.shape)
    it_gd, h_gd = run(W0, False, 1.0); it_n1, h_n1 = run(W0, True, 1.0, iters=2000); it_n5, h_n5 = run(W0, True, 0.5, iters=2000)
    sub(f"Realisable target q = p_V(W*), start at W* + noise (KL = {h_gd[0]:.4f}); G_V at W* has condition number {lam_max / lam_min:.0f}.")
    sub(f"      plain gradient descent, step 1/lambda_max(G_V) = {1 / lam_max:.3f}: {it_gd} steps to KL < 1e-12; natural gradient (G_V + 1e-9 I)^-1 grad: step 1 reaches {h_n1[-1]:.1e} after {it_n1} steps (it overshoots from this start), step 0.5 needs {it_n5}.")
    sub("      Close to W* the Hessian equals G_V and a unit natural-gradient step is Newton's method; far from it they differ (above) and the step must be damped.")
    STORE["ng"] = (h_gd, h_n1, h_n5, it_gd, it_n1, it_n5, lam_max / lam_min)
    # invariance of the natural gradient direction under a change of parametrisation W = f(u)
    f_map = lambda u_: (u_ + 0.4 * u_ ** 3 + 0.3 * np.roll(u_, 1))             # an invertible smooth map R^6 -> R^6 near u0 (weights W = f(u))
    u_pt = np.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.4])
    Jm = np.zeros((6, 6))
    for k_ in range(6):
        e_ = np.zeros(6); e_[k_] = 1e-6
        Jm[:, k_] = (f_map(u_pt + e_) - f_map(u_pt - e_)) / 2e-6
    Rn = RBM(f_map(u_pt).reshape(2, 3)); gW2 = Rn.grad_kl(qs).ravel(); GW2 = fisher_marginal(Rn) + 1e-9 * np.eye(6)
    gu = Jm.T @ gW2; Gu = Jm.T @ GW2 @ Jm
    nat_u = np.linalg.solve(Gu, gu); nat_W = np.linalg.solve(GW2, gW2)
    sub(f"Invariance: change the weights to u with W = f(u) (a nonlinear invertible map with Jacobian J).  The natural-gradient step in u, G_u^-1 grad_u KL, mapped back by J equals the step in W to {np.abs(Jm @ nat_u - nat_W).max():.1e};")
    sub(f"      the plain gradient step in u mapped back, J grad_u KL = J J' grad_W KL, differs from grad_W KL by {np.abs(Jm @ gu - gW2).max():.3f} (relative to {np.abs(gW2).max():.3f}).")
    STORE["ng_inv"] = (np.abs(Jm @ nat_u - nat_W).max(), np.abs(Jm @ gu - gW2).max())


def check_cd():
    head("17. Contrastive divergence: Theorems 11.11 and 11.12")
    rg = np.random.default_rng(1116)
    R = RBM(rg.normal(0, 0.9, size=(2, 3)))
    q = rg.dirichlet(np.ones(8) * 1.5)
    nV, nH = 8, 4
    cond_h = R.P / R.pV[:, None]                      # p(h|v), rows v
    cond_v = (R.P / R.pH[None, :])                    # p(v|h), columns h
    p = R.P.copy()
    p0 = q[:, None] * cond_h
    J = 80
    ps = [p0]; pts = [None]
    for j in range(1, J + 1):
        pH_prev = ps[-1].sum(0)                       # marginal over v of p_{j-1}: p_{H,j-1}(h)
        pt = cond_v * pH_prev[None, :]                # p~_j(v,h) = p_{H,j-1}(h) p(v|h)       (11.214)
        pV_t = pt.sum(1)
        pj = pV_t[:, None] * cond_h                   # p_j(v,h) = p~_{V,j}(v) p(h|v)          (11.213)
        pts.append(pt); ps.append(pj)
    KL = lambda a, b: kl_pq(a.ravel(), b.ravel())
    total = KL(p0, p)
    s1 = sum(KL(ps[j], pts[j + 1]) for j in range(0, J))
    s2 = sum(KL(pts[j], ps[j]) for j in range(1, J + 1))
    rem = KL(ps[J], p)
    sub(f"Random RBM (3 visible, 2 hidden), data q(v) random.  CD chain p_0 = q(v)p(h|v), p~_j = p_H,j-1(h) p(v|h), p_j = p~_V,j(v) p(h|v) up to j = {J}.")
    sub(f"(11.222): KL[p_0:p] = {total:.10f}; sum_(j=0..{J - 1}) KL[p_j:p~_(j+1)] + sum_(j=1..{J}) KL[p~_j:p_j] + KL[p_{J}:p] = {s1 + s2 + rem:.10f} (remainder KL[p_{J}:p] = {max(rem, 0.0):.1e}).")
    kls = [KL(ps[j], p) for j in range(0, 12)]
    sub("KL[p_j:p] for j = 0..8: " + ", ".join(f"{max(v, 0.0):.2e}" for v in kls[:9]) + " (decreasing, tends to 0).")
    # Theorem 11.11: m-projection property, checked by minimisation over the free distribution
    def proj_cost(logits):
        r = np.exp(logits - logits.max()); r /= r.sum()
        return KL(ps[0], r[None, :] * cond_v)        # KL[p_0 : r~(h) p(v|h)]
    sol, fv = nelder_mead(proj_cost, np.zeros(nH), step=0.5)
    r_best = np.exp(sol - sol.max()); r_best /= r_best.sum()
    sub(f"Theorem 11.11, first step: minimising KL[p_0 : r~(h) p(v|h)] over r~ numerically gives r~ = {np.round(r_best, 6).tolist()}; p_H,0(h) = {np.round(ps[0].sum(0), 6).tolist()}.")
    def proj_cost2(logits):
        r = np.exp(logits - logits.max()); r /= r.sum()
        return KL(pts[1], r[:, None] * cond_h)       # KL[p~_1 : r(v) p(h|v)]
    sol2, _ = nelder_mead(proj_cost2, np.zeros(nV), step=0.5)
    r2 = np.exp(sol2 - sol2.max()); r2 /= r2.sum()
    sub(f"      second step: minimiser r(v) = {np.round(r2[:4], 5).tolist()}... against p~_V,1(v) = {np.round(pts[1].sum(1)[:4], 5).tolist()}... (max difference {np.abs(r2 - pts[1].sum(1)).max():.1e}).")
    STORE["cd"] = (kls, total, s1, s2, rem, [KL(ps[j], pts[j + 1]) for j in range(0, 10)], [KL(pts[j], ps[j]) for j in range(1, 11)])
    # the CD_k direction approximates the likelihood direction
    d_exact = R.exp_f(p0) - R.exp_f(p)                         # <h v'>_q - <h v'>_p, the rule (11.206) with the exact average
    sub("The exact direction of (11.206) is <h v'>_q - <h v'>_p = minus the gradient (11.209).  The CD_k direction uses p_k instead of p; max |CD_k - exact| over the 6 weights:")
    errs = [(k, float(np.abs((R.exp_f(ps[0]) - R.exp_f(ps[k])) - d_exact).max())) for k in (1, 2, 3, 5, 10, 20)]
    sub("      " + "  ".join(f"k={k}: {e:.2e}" for k, e in errs) + f"   (exact direction {np.round(d_exact.ravel(), 4).tolist()})")
    STORE["cd_errs"] = errs


def gauss_rbm_rhs(W, C, sv2, k=None):
    """Right-hand side of the learning equation for the Gaussian RBM, written with the factor W that the printed (11.229)-(11.230) lack:
    ML (k=None): W C/sv2 - W (I - W'W)^-1 ;  CD_k: W C/sv2 - W [ (W'W)^k C (W'W)^k / sv2 + sum_{i<2k} (W'W)^i ]."""
    n = C.shape[0]; A = W.T @ W
    if k is None:
        return W @ C / sv2 - W @ np.linalg.inv(np.eye(n) - A)
    Ak = np.linalg.matrix_power(A, k)
    S = sum(np.linalg.matrix_power(A, i) for i in range(2 * k))
    return W @ C / sv2 - W @ (Ak @ C @ Ak / sv2 + S)


def integrate_rbm(W0, C, sv2, k=None, T=120.0, dt=0.02, stride=25):
    W = W0.copy(); f = lambda W_: gauss_rbm_rhs(W_, C, sv2, k)
    traj = [np.linalg.svd(W, compute_uv=False)]
    for step in range(int(T / dt)):
        k1 = f(W); k2 = f(W + 0.5 * dt * k1); k3 = f(W + 0.5 * dt * k2); k4 = f(W + dt * k3)
        W = W + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % stride == 0:
            traj.append(np.linalg.svd(W, compute_uv=False))
    return W, np.array(traj)


def check_gauss_rbm():
    head("18. The Gaussian RBM (11.223)-(11.234)")
    rg = np.random.default_rng(1117)
    n, m = 3, 2
    sv, sh = 0.9, 1.3
    W = rg.normal(0, 0.3, size=(m, n))
    Pm = np.block([[np.eye(n) / sv ** 2, -W.T / (sv * sh)], [-W / (sv * sh), np.eye(m) / sh ** 2]])     # precision of (v, h) from (11.223)
    Cov = np.linalg.inv(Pm)
    marg = sv ** 2 * np.linalg.inv(np.eye(n) - W.T @ W)
    Chv = Cov[n:, :n]
    cond_mean = -np.linalg.solve(Pm[n:, n:], Pm[n:, :n])           # E[h|v] = cond_mean v
    sub(f"Gaussian RBM, n = {n}, m = {m}, sigma_v = {sv}, sigma_h = {sh}.  Marginal covariance of v from the joint precision matrix vs sigma_v^2 (I - W'W)^-1 (11.226): {np.abs(Cov[:n, :n] - marg).max():.1e};")
    sub(f"      E[h|v] = {np.round(cond_mean, 4).tolist()} v vs (sigma_h/sigma_v) W v: {np.abs(cond_mean - sh / sv * W).max():.1e} (11.224; the printed sigma_n is sigma_h); the cross covariance <h v'>_p / (sigma_v sigma_h) = W (I - W'W)^-1: {np.abs(Chv / (sv * sh) - W @ np.linalg.inv(np.eye(n) - W.T @ W)).max():.1e}")
    sub("      so (11.227)-(11.228) are the averages of h v'/(sigma_v sigma_h), the derivative of the exponent with respect to W, not of h v' itself.")
    # equilibria of the corrected learning equation and the printed formula
    sv2 = 0.8
    lam = np.array([3.0, 1.6, 0.4]); O = np.linalg.qr(rg.normal(size=(3, 3)))[0]; C = O.T @ np.diag(lam) @ O
    Lt = np.sqrt(np.maximum(1 - sv2 / lam, 0.0))
    U = np.linalg.qr(rg.normal(size=(3, 3)))[0]
    Wstar = U @ np.diag(Lt) @ O
    r_ok = np.abs(gauss_rbm_rhs(Wstar, C, sv2)).max()
    Lt_book = np.sqrt(np.maximum(1 - 1 / lam, 0.0))
    Wbook = U @ np.diag(Lt_book) @ O
    r_bk = np.abs(gauss_rbm_rhs(Wbook, C, sv2)).max()
    printed_rhs = Wstar @ C / sv2 - np.linalg.inv(np.eye(3) - Wstar.T @ Wstar)      # (11.229) as printed (no W in the second term), square case m = n
    sub(f"Data covariance eigenvalues {lam.tolist()}, sigma_v^2 = {sv2}: r = {int((lam > sv2).sum())} eigenvalues exceed sigma_v^2.  Equilibrium W = U diag(sqrt(1 - sigma_v^2/lambda_i), ..., 0) O of the learning equation:")
    sub(f"      right-hand side (with the factor W restored) = {r_ok:.1e} at this W; at the printed (11.233), sqrt(1 - 1/lambda_i), it is {r_bk:.2f} (the printed form is right only for sigma_v = 1).")
    sub(f"      The printed (11.229), (1/sigma_v^2) W C - (I - W'W)^-1, evaluated at the correct equilibrium of the m = n = 3 machine is {np.abs(printed_rhs).max():.2f}, not 0 (for m != n its two terms do not even have the same shape).")
    # integrate ML and CD_k flows: n=4, m=2 (r = 3 > m)
    lam4 = np.array([4.0, 2.5, 1.5, 0.5]); O4 = np.linalg.qr(rg.normal(size=(4, 4)))[0]; C4 = O4.T @ np.diag(lam4) @ O4
    W0 = rg.normal(0, 0.05, size=(2, 4))
    pred = np.sqrt(1 - sv2 / lam4[:2])
    sub(f"n = 4, m = 2 hidden units, eigenvalues {lam4.tolist()}, sigma_v^2 = {sv2} (r = 3 > m): integrating the flow from a small random W, the singular values of W end at")
    flows = {}
    for lab, kk in (("ML (11.229)", None), ("CD_1", 1), ("CD_3", 3)):
        Wf, traj = integrate_rbm(W0, C4, sv2, kk, T=140.0)
        flows[lab] = traj
        sv_ = np.linalg.svd(Wf, compute_uv=False)
        model_cov = sv2 * np.linalg.inv(np.eye(4) - Wf.T @ Wf)
        ev = np.sort(np.linalg.eigvalsh(model_cov))[::-1]
        print(f"      {lab:<12s} {np.round(sv_, 5).tolist()}   (predicted sqrt(1 - sigma_v^2/lambda_i), i = 1, 2: {np.round(pred, 5).tolist()});  eigenvalues of the model covariance {np.round(ev, 4).tolist()} (11.234)")
    STORE["gflow"] = (flows, pred, lam4, sv2, W0, C4)
    sub("The retained components have the data variance lambda_i, all others the floor sigma_v^2 = 0.8 (including lambda_3 = 1.5, which the 2 hidden units cannot carry), and ML, CD_1, CD_3 reach the same W (11.232).")
    # the CD_k right-hand side: Gibbs chain by sampling
    Wm = rg.normal(0, 0.3, size=(2, 3)); Cm = O.T @ np.diag([1.5, 0.9, 0.5]) @ O
    Lc = np.linalg.cholesky(Cm)
    Rr = 400_000
    k = 2
    v = (Lc @ rg.normal(size=(3, Rr))).T
    for _ in range(k):
        h = (sh / sv) * (v @ Wm.T) + sh * rg.normal(size=(Rr, 2))
        v = (sv / sh) * (h @ Wm) + sv * rg.normal(size=(Rr, 3))
    h = (sh / sv) * (v @ Wm.T) + sh * rg.normal(size=(Rr, 2))
    mc = (h.T @ v) / Rr / (sv * sh)
    A_ = Wm.T @ Wm; Ak = np.linalg.matrix_power(A_, k)
    Ck = Ak @ Cm @ Ak + sv ** 2 * sum(np.linalg.matrix_power(A_, i) for i in range(2 * k))
    exact = Wm @ Ck / sv ** 2
    se = np.sqrt(np.mean((h[:, :, None] * v[:, None, :] / (sv * sh) - mc[None]) ** 2, axis=0) / Rr).max()
    sub(f"CD_2 Gibbs chain simulated ({Rr} chains): <h v'>/(sigma_v sigma_h) after 2 steps = {np.round(mc.ravel(), 3).tolist()} against W C_2/sigma_v^2 with C_k = (W'W)^k C (W'W)^k + sigma_v^2 sum_(i<2k) (W'W)^i: {np.round(exact.ravel(), 3).tolist()}")
    sub(f"      largest difference {np.abs(mc - exact).max():.4f}, largest standard error {se:.4f}; this is (11.230) with the factor W restored in the sum.")
    STORE["cdk_mc"] = (np.abs(mc - exact).max(), se)
    # equilibria are not unique: subsets of the retained directions; stability
    lam2 = np.array([3.0, 2.0]); C2 = np.diag(lam2); s1 = 1.0
    a = np.sqrt(1 - s1 / lam2)
    cands = {"both": np.diag(a), "only 1": np.diag([a[0], 0.0]), "only 2": np.diag([0.0, a[1]]), "none": np.zeros((2, 2))}
    sub("Equilibria are not unique (n = m = 2, lambda = (3, 2), sigma_v = 1): any subset of the directions with lambda_i > sigma_v^2 can be kept; only keeping all of them is stable (largest eigenvalue 0 = rotations).")
    stab = {}
    for nm, Wc in cands.items():
        res = np.abs(gauss_rbm_rhs(Wc, C2, s1)).max()
        J = np.zeros((4, 4))
        for kx in range(4):
            E = np.zeros(4); E[kx] = 1e-6
            J[:, kx] = ((gauss_rbm_rhs(Wc + E.reshape(2, 2), C2, s1) - gauss_rbm_rhs(Wc - E.reshape(2, 2), C2, s1)) / 2e-6).ravel()
        mx = np.linalg.eigvals(J).real.max()
        stab[nm] = (res, mx)
        print(f"      keep {nm:<7s}: |rhs| = {res:.1e}, largest Jacobian eigenvalue {mx:+.3f} -> {'stable (the zero eigenvalues are the rotations U of (11.232))' if mx < 1e-6 else 'unstable'}")
    STORE["gstab"] = stab
    # relaxation rates, scalar machine
    lam1 = 2.0; w_star = math.sqrt(1 - 1 / lam1)
    rates = {}
    for lab, kk in (("ML", None), ("CD_1", 1), ("CD_2", 2), ("CD_5", 5)):
        f_ = lambda w_: float(gauss_rbm_rhs(np.array([[w_]]), np.array([[lam1]]), 1.0, kk)[0, 0])
        rates[lab] = (f_(w_star + 1e-6) - f_(w_star - 1e-6)) / 2e-6
    sub("Scalar machine (n = m = 1, sigma_v = 1, lambda = 2, W* = sqrt(1/2)): linearised relaxation rate d(rhs)/dw at W*: " + ", ".join(f"{k_}: {v_:.4f}" for k_, v_ in rates.items()) + " (all stable, CD_k slower than ML).")
    STORE["rates"] = rates


def noisy_circle(rg, n, noise=0.35, R0=1.1):
    """Points uniform in a disc of radius 2, labelled by whether the true radius is below R0, then moved by Gaussian noise (so the classes overlap)."""
    r = 2.0 * np.sqrt(rg.uniform(size=n)); t = rg.uniform(0, 2 * math.pi, n)
    xt = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    y = np.where(r < R0, 1.0, -1.0)
    return xt + rg.normal(0, noise, size=xt.shape), y


def check_conformal_svm():
    head("10. Does the conformal change of the kernel help?  (11.92)-(11.93) and the remark 'improved by up to ten percent'")
    N, s2, C, noise, reps, Ntest = 40, 1.0, 100.0, 0.35, 30, 1500
    kappas = (1.0, 3.0, 10.0)
    sub(f"Noisy-circle toy problem (label = true radius < 1.1, positions moved by N(0, {noise}^2)), N = {N} training points, Gaussian kernel sigma^2 = {s2}, C = {C:.0f}, {Ntest} fresh test points per run,")
    sub(f"{reps} paired runs.  Stage 1: train with K, take the support vectors x_i*.  Stage 2: retrain with K~ = sigma(x) sigma(x') K, sigma(x) = sum_i exp(-kappa |x - x_i*|) (11.93), the same sigma at test time.")
    diffs = {k: [] for k in kappas}; base = []; geom = {}
    for rep in range(reps):
        rg = np.random.default_rng(7000 + rep)
        X, y = noisy_circle(rg, N, noise); Xt, yt = noisy_circle(rg, Ntest, noise)
        K = gauss_K(X, X, s2); Kt = gauss_K(Xt, X, s2)
        a, b, it = smo(K, y, C=C, tol=1e-6, maxit=30000)
        e0 = float(np.mean(np.sign(Kt @ (a * y) + b) != yt)); base.append(e0)
        sv = a > 1e-7
        for kap in kappas:
            sig = lambda P: np.exp(-kap * np.sqrt(((P[:, None, :] - X[sv][None, :, :]) ** 2).sum(-1))).sum(1)
            sg, sgt = sig(X), sig(Xt)
            K2 = sg[:, None] * sg[None, :] * K
            a2, b2, it2 = smo(K2, y, C=C, tol=1e-6, maxit=30000)
            e2 = float(np.mean(np.sign((sgt[:, None] * sg[None, :] * Kt) @ (a2 * y) + b2) != yt))
            diffs[kap].append(e2 - e0)
            if rep == 0:
                m1 = 1 / math.sqrt((a * y) @ K @ (a * y)); m2 = 1 / math.sqrt((a2 * y) @ K2 @ (a2 * y))
                geom[kap] = (int(sv.sum()), int((a2 > 1e-7).sum()), m1, m2, math.sqrt(np.mean(np.diag(K2))), float(sg.max() / sg.min()))
    base = np.array(base)
    print(f"      test error with the plain Gaussian kernel: {base.mean():.4f} (mean over {reps} runs)")
    print("      kappa   mean (K~ - K) test error ± s.e.   runs improved / worse    relative change   support vectors in run 0 (K -> K~)")
    rows = []
    for kap in kappas:
        d = np.array(diffs[kap]); se = d.std(ddof=1) / math.sqrt(len(d))
        g = geom[kap]
        rows.append((kap, d.mean(), se, int((d < 0).sum()), int((d > 0).sum())))
        print(f"      {kap:<7.1f} {d.mean():+.4f} ± {se:.4f}               {int((d < 0).sum())} / {int((d > 0).sum())}                  {d.mean() / base.mean() * 100:+.1f} percent       {g[0]} -> {g[1]}")
    STORE["conf_exp"] = (base.mean(), rows)
    # the margin heuristic, noise-free case: margin and enclosing-ball radius of the feature vectors
    def enclosing_radius(K_, iters=2000):
        n_ = len(K_); c = np.zeros(n_); c[0] = 1.0
        for t in range(1, iters):
            d2 = np.diag(K_) - 2 * K_ @ c + c @ K_ @ c
            e = np.zeros(n_); e[int(np.argmax(d2))] = 1.0
            c = c + (e - c) / (t + 1)
        return math.sqrt((np.diag(K_) - 2 * K_ @ c + c @ K_ @ c).max())
    sub("Margin heuristic on separable data (noise 0, 24 points, hard margin, 8 runs; R = radius of the smallest ball enclosing the feature vectors by 2000 Badoiu-Clarkson steps, a few percent accurate):")
    print("      kappa   margin rho: K -> K~     radius R: K -> K~     (R/rho)^2: K -> K~")
    mrows = []
    acc = {k: [] for k in (0.3, 1.0, 3.0)}; K_acc = []
    for rep in range(8):
        rg = np.random.default_rng(8000 + rep)
        X, y = noisy_circle(rg, 24, 0.0)
        K = gauss_K(X, X, s2)
        a, b, it = smo(K, y, C=1e8, tol=1e-8, maxit=100000)
        sv = a > 1e-7
        rho = 1 / math.sqrt((a * y) @ K @ (a * y)); R1 = enclosing_radius(K)
        K_acc.append((rho, R1))
        for kap in acc:
            sg = np.exp(-kap * np.sqrt(((X[:, None, :] - X[sv][None, :, :]) ** 2).sum(-1))).sum(1)
            K2 = sg[:, None] * sg[None, :] * K
            a2, b2, it2 = smo(K2, y, C=1e8, tol=1e-8, maxit=100000)
            rho2 = 1 / math.sqrt((a2 * y) @ K2 @ (a2 * y))
            acc[kap].append((rho2, enclosing_radius(K2)))
    Ka = np.array(K_acc)
    for kap, v in acc.items():
        v = np.array(v)
        mrows.append((kap, Ka[:, 0].mean(), v[:, 0].mean(), Ka[:, 1].mean(), v[:, 1].mean(), ((Ka[:, 1] / Ka[:, 0]) ** 2).mean(), ((v[:, 1] / v[:, 0]) ** 2).mean()))
        print(f"      {kap:<7.1f} {Ka[:, 0].mean():.3f} -> {v[:, 0].mean():.3f}         {Ka[:, 1].mean():.3f} -> {v[:, 1].mean():.3f}       {((Ka[:, 1] / Ka[:, 0]) ** 2).mean():.1f} -> {((v[:, 1] / v[:, 0]) ** 2).mean():.1f}")
    STORE["margin_rows"] = mrows
    sub("Margin and radius both change by the same order of magnitude: " + "; ".join(f"kappa {r_[0]:.1f}: margin x{r_[2] / r_[1]:.2f}, radius x{r_[4] / r_[3]:.2f}" for r_ in mrows) + ".")
    sub("The scale-free (R/rho)^2, which is what generalisation bounds use, improves for kappa <= 1 and gets much worse for kappa = 3: the enlargement of the margin is mostly a rescaling of the feature space.")
    best = min(rows, key=lambda r_: r_[1])
    sub(f"In this toy the effect depends strongly on kappa: kappa = {best[0]:.0f} gives the best result, {best[1]:+.4f} ± {best[2]:.4f} ({best[1] / base.mean() * 100:+.1f} percent, {abs(best[1]) / best[2]:.1f} standard errors); a smaller kappa does not help and a larger one hurts badly")
    sub("(sigma vanishes away from the support vectors and the other patterns collapse towards the origin of the feature space).  I cannot confirm 'up to ten percent'; the gain I see is small and fragile.")



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


def clip_halfplane(poly_, nrm, c):
    """Sutherland-Hodgman: keep the part of a convex polygon with nrm.x <= c."""
    out = []
    n_ = len(poly_)
    for k in range(n_):
        P, Q = np.array(poly_[k]), np.array(poly_[(k + 1) % n_])
        fp, fq = nrm @ P - c, nrm @ Q - c
        if fp <= 0:
            out.append(tuple(P))
        if fp * fq < 0:
            t = fp / (fp - fq); out.append(tuple(P + t * (Q - P)))
    return out


def kl_cells(centres, box):
    """Voronoi cells of the generalised KL divergence D[x:eta_h] (centre in the second slot) as convex polygons in the eta plane."""
    g = BREG["KL"]
    th = [g.grad(c) for c in centres]
    cells = []
    for i, ci in enumerate(centres):
        pg = [(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])]
        for j, cj in enumerate(centres):
            if i != j:
                c = float(g.phi(ci) - g.phi(cj) - th[i] @ ci + th[j] @ cj)
                pg = clip_halfplane(pg, th[j] - th[i], c)
        cells.append(pg)
    return cells


def densify(pg, n=24):
    pts = []
    for k in range(len(pg)):
        P, Q = np.array(pg[k]), np.array(pg[(k + 1) % len(pg)])
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(P + t * (Q - P))
    return np.array(pts)


def fig_voronoi(path):
    g = BREG["KL"]
    cen = [np.array([1.0, 3.0]), np.array([4.0, 1.5]), np.array([2.5, 5.2]), np.array([5.3, 4.6])]
    box = (0.25, 6.25, 0.25, 6.25)
    cells = kl_cells(cen, box)
    e1, e2 = cen[0], cen[1]
    th1, th2 = g.grad(e1), g.grad(e2)
    P12, t12 = eq_point_on_line(g, e1, e2, "theta")
    Q12, s12 = eq_point_on_line(g, e1, e2, "eta")
    ts = np.linspace(0, 1, 120)
    egeo = np.array([g.ginv((1 - t) * th1 + t * th2) for t in ts])
    mline = np.array([(1 - t) * e1 + t * e2 for t in ts])
    nrm2 = e2 - e1; v2 = np.array([-nrm2[1], nrm2[0]]) / np.linalg.norm(nrm2)
    thQ = g.grad(Q12)
    printed = np.array([g.ginv(thQ + tau * v2) for tau in np.linspace(-1.1, 1.1, 120)])
    # true bisector of centres 1 and 2: x_2 as a function of x_1, kept where it lies on the boundary of both cells
    bis = []
    for x1 in np.linspace(box[0], box[1], 400):
        f = lambda x2: float(g.D(np.array([x1, x2]), e1) - g.D(np.array([x1, x2]), e2))
        if f(box[2]) * f(box[3]) < 0:
            x2 = bisect(f, box[2], box[3])
            x = np.array([x1, x2])
            d = [float(g.D(x, c)) for c in cen]
            if d[0] <= min(d) + 1e-9:
                bis.append(x)
    bis = np.array(bis)
    b = []
    W, H = 800, 540
    PA = Panel(b, 60, 56, 320, 300, (0.25, 6.25), (0.25, 6.25))
    PA.frame([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], "x₁ = η₁", "x₂ = η₂", "Pattern plane (η coordinates)", False)
    lo, hi = math.log(0.25), math.log(6.25)
    PB = Panel(b, 450, 56, 320, 300, (lo, hi), (lo, hi))
    PB.frame([-1, 0, 1], [-1, 0, 1], "θ₁ = log x₁", "θ₂ = log x₂", "Dual coordinates (θ)", False)
    for k, pg in enumerate(cells):
        pts = np.array(pg)
        cc_ = ("f3", "f4", "f0", "f2")[k]
        b.append(f'<polygon class="{cc_}" style="fill-opacity:.12;stroke:none" points="' + " ".join(f"{PA.X(x):.1f},{PA.Y(y):.1f}" for x, y in pts) + '"/>')
        dp = np.log(densify(pg))
        b.append(f'<polygon class="{cc_}" style="fill-opacity:.12;stroke:none" points="' + " ".join(f"{PB.X(x):.1f},{PB.Y(y):.1f}" for x, y in dp) + '"/>')
        b.append(f'<polygon class="thin s0" points="' + " ".join(f"{PA.X(x):.1f},{PA.Y(y):.1f}" for x, y in pts) + '"/>')
        b.append(f'<polygon class="thin s0" points="' + " ".join(f"{PB.X(x):.1f},{PB.Y(y):.1f}" for x, y in dp) + '"/>')
    PA.line(bis[:, 0], bis[:, 1], "ln sk")
    PB.line(np.log(bis[:, 0]), np.log(bis[:, 1]), "ln sk")
    PA.line(egeo[:, 0], egeo[:, 1], "dash s1"); PB.line(np.log(egeo[:, 0]), np.log(egeo[:, 1]), "dash s1")
    PA.line(mline[:, 0], mline[:, 1], "dot s0"); PB.line(np.log(mline[:, 0]), np.log(mline[:, 1]), "dot s0")
    PA.line(printed[:, 0], printed[:, 1], "dash s2"); PB.line(np.log(printed[:, 0]), np.log(printed[:, 1]), "dash s2")
    for k, c in enumerate(cen):
        cc_ = ("f3", "f4", "f0", "f2")[k]
        PA.dot(c[0], c[1], cc_, 5.0); PB.dot(math.log(c[0]), math.log(c[1]), cc_, 5.0)
    PA.dot(P12[0], P12[1], "f1", 4.0); PB.dot(math.log(P12[0]), math.log(P12[1]), "f1", 4.0)
    PA.dot(Q12[0], Q12[1], "f0", 4.0); PB.dot(math.log(Q12[0]), math.log(Q12[1]), "f0", 4.0)
    legend_col(b, 60, 416, [("ln sk", "bisector D[x:η_1] = D[x:η_2]: straight in η"), ("dash s1", "e-geodesic (straight in θ), equidistant point η_{12} (blue dot)")])
    legend_col(b, 440, 416, [("dot s0", "η-straight line and its 'midpoint' (grey dot)"), ("dash s2", "θ-hyperplane through that point, as printed in (11.19)-(11.20)")])
    note(b, 60, 472, ["Generalised KL divergence, four centres, cells tinted. Left: every cell is a convex polygon in the pattern coordinates, so its boundaries are straight",
                      "and meet the θ-straight line orthogonally. Right: the same cells in θ = log x. The true bisector (black) is curved; the construction printed in",
                      "the book (orange) is straight here and is the right set for the exchanged divergence D[η_h : x], not for D[x : η_h] (Section 1.3 of the notes)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Bregman Voronoi cells and the printed Theorem 11.2",
        "Two panels of the same four-cell Voronoi diagram for the generalised KL divergence with the centre in the second slot. In the pattern coordinates eta the cell boundaries are straight lines, in the dual coordinates theta they are curves. The geodesic between two centres that is straight in theta meets the boundary of the two cells orthogonally at the equidistant point; the construction printed in the book (a boundary straight in theta, orthogonal to the eta-straight line) gives a different curve.", b))


def fig_chernoff(path):
    C, tstar, Ns, lp_eq, lp_90, lam1, lam2 = STORE["chernoff"]
    th1, th2 = math.log(lam1), math.log(lam2)
    kl = lambda ta, tb: math.exp(ta) * (ta - tb) - math.exp(ta) + math.exp(tb)
    b = []
    W, H = 800, 430
    P1 = Panel(b, 70, 56, 320, 250, (math.log10(4), math.log10(2600)), (0.3, 0.9))
    tk = {math.log10(v): str(v) for v in (5, 10, 20, 50, 100, 200, 500, 1000, 2000)}
    P1.frame(list(tk), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "N (observations)", "−log P_err / N", "Exact Bayes error", True, tk)
    P1.line([math.log10(4), math.log10(2600)], [C, C], "dash s0")
    xs = [math.log10(N) for N in Ns]
    P1.line(xs, [-v / N for v, N in zip(lp_eq, Ns)], "ln s1"); P1.line(xs, [-v / N for v, N in zip(lp_90, Ns)], "ln s2")
    for x, v, N in zip(xs, lp_eq, Ns):
        P1.dot(x, -v / N, "f1", 3.4)
    for x, v, N in zip(xs, lp_90, Ns):
        P1.dot(x, -v / N, "f2", 3.4)
    P1.text(math.log10(4.5), C - 0.012, f"Chernoff information C = {C:.4f}", "sm", "start", 0, 12)
    legend_col(b, 150, 80, [("s1", "equal priors"), ("s2", "priors 0.9 / 0.1")])
    P2 = Panel(b, 450, 56, 320, 250, (0, 1), (0, 1.75))
    P2.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1.0, 1.5], "t along the e-geodesic  θ_t = (1−t) θ₁ + t θ₂", "divergence", "The equidistant point", True)
    ts = np.linspace(0, 1, 201)
    k1 = [kl((1 - t) * th1 + t * th2, th1) for t in ts]; k2 = [kl((1 - t) * th1 + t * th2, th2) for t in ts]
    cc = [(1 - t) * a + t * c for t, a, c in zip(ts, k1, k2)]
    P2.line(ts, k1, "ln s1"); P2.line(ts, k2, "ln s2"); P2.line(ts, cc, "ln s3")
    P2.line([tstar, tstar], [0, 1.75], "dash s0"); P2.line([0, 1], [C, C], "dash s0")
    P2.dot(tstar, C, "f4", 5.0)
    P2.text(tstar + 0.02, 0.1, f"t* = {tstar:.4f}", "sm", "start")
    legend_col(b, 466, 74, [("s1", "KL[θ_t : θ₁]"), ("s2", "KL[θ_t : θ₂]"), ("s3", "−log Σ p_1^{1−t} p_2^t, their mix")])
    note(b, 70, 376, ["Left: for two Poisson laws (rates 2 and 5) the exponent of the exact error tends to C for either prior; only the prefactor differs.",
                      "Right: the two KL curves cross at t*, where −log Σ p₁^{1−t}p₂^t, a convex combination of them, is largest and equal to both (Section 1.6 of the notes)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Chernoff information of two Poisson distributions",
        "Left: minus the logarithm of the exact Bayes error divided by N against N for Poisson(2) versus Poisson(5), for equal priors and for priors 0.9 and 0.1; both tend to the Chernoff information 0.3397 from above. Right: along the e-geodesic between the two distributions the KL divergence to each end crosses at t star = 0.538, where the Chernoff function minus log of the sum of p1 to the 1 minus t times p2 to the t reaches its maximum 0.3397.", b))


def fig_svm(path):
    N, s2, C, noise, Ntest = 40, 1.0, 100.0, 0.35, 1500
    rg = np.random.default_rng(7000)
    X, y = noisy_circle(rg, N, noise)
    K = gauss_K(X, X, s2)
    a, bb, it = smo(K, y, C=C, tol=1e-6, maxit=30000)
    sv = a > 1e-7
    kap = 3.0
    sig = lambda P: np.exp(-kap * np.sqrt(((P[:, None, :] - X[sv][None, :, :]) ** 2).sum(-1))).sum(1)
    sg = sig(X)
    K2 = sg[:, None] * sg[None, :] * K
    a2, b2, it2 = smo(K2, y, C=C, tol=1e-6, maxit=30000)
    f1 = lambda P: gauss_K(P, X, s2) @ (a * y) + bb
    f2 = lambda P: sig(P) * ((gauss_K(P, X, s2) * sg[None, :]) @ (a2 * y)) + b2
    def boundary(f):
        pts = []
        rs = np.linspace(0.02, 3.0, 150)
        for th_ in np.linspace(0, 2 * math.pi, 181):
            P = np.stack([rs * math.cos(th_), rs * math.sin(th_)], axis=1)
            v = f(P)
            idx = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
            if len(idx):
                i = idx[0]; r = rs[i] + (rs[i + 1] - rs[i]) * v[i] / (v[i] - v[i + 1])
                pts.append((r * math.cos(th_), r * math.sin(th_)))
        return np.array(pts)
    B1, B2 = boundary(f1), boundary(f2)
    b = []
    W, H = 800, 495
    P1 = Panel(b, 60, 56, 320, 260, (-2.4, 2.4), (-2.4, 2.4))
    P1.frame([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], "x₁", "x₂", "Noisy circle, N = 40", True)
    P1.line(B1[:, 0], B1[:, 1], "ln s1"); P1.line(B2[:, 0], B2[:, 1], "ln s2")
    circ = np.array([(1.1 * math.cos(t), 1.1 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 100)])
    P1.line(circ[:, 0], circ[:, 1], "dash s0")
    for i in range(N):
        cls = "f1" if y[i] > 0 else "f4"
        b.append(f'<circle class="{cls} ring" cx="{P1.X(X[i, 0]):.1f}" cy="{P1.Y(X[i, 1]):.1f}" r="{4.2 if sv[i] else 2.8}" style="{"stroke-width:2.2" if sv[i] else ""}"/>')
    legend_col(b, 60, 372, [("s1", "f = 0 for kernel K"), ("s2", "f = 0 for K~ (κ = 3)"), ("dash s0", "true radius 1.1")], 16)
    rows = STORE["conf_exp"][1]
    P2 = Panel(b, 450, 56, 320, 260, (0, 3), (-0.03, 0.09))
    P2.frame([0.5, 1.5, 2.5], [-0.02, 0, 0.02, 0.04, 0.06, 0.08], "κ", "test error of K~ minus that of K", "30 paired runs", True, {0.5: "1", 1.5: "3", 2.5: "10"})
    P2.line([0, 3], [0, 0], "dash s0")
    for k, (kp, m_, se, nb, nw) in enumerate(rows):
        x = k + 0.5
        b.append(f'<line class="thin s0" x1="{P2.X(x):.1f}" y1="{P2.Y(m_ - 2 * se):.1f}" x2="{P2.X(x):.1f}" y2="{P2.Y(m_ + 2 * se):.1f}"/>')
        P2.dot(x, m_, "f2", 4.6)
        P2.text(x + 0.08, m_, f"{m_:+.3f}", "sm", "start", 0, 4)
    note(b, 60, 440, ["Left: decision boundary of the SVM before (blue) and after the conformal change (orange); blue dots are class +1, yellow dots class -1, support vectors",
                      "are the larger dots. Right: paired change in the test error (dots: mean, bars: two standard errors). The sign depends on κ: a moderate κ helps a",
                      "little, a large one hurts (Section 2.4 of the notes)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Conformal change of a kernel on a noisy-circle problem",
        "Left: the decision boundaries f = 0 of a Gaussian-kernel SVM trained on forty noisy-circle points, before and after the conformal modification of the kernel with kappa equal to 3; the support vectors are circled and the dashed circle is the true class boundary. Right: the paired change in test error over thirty runs for kappa equal to 1, 3 and 10, with two-standard-error bars: about plus 0.012, minus 0.007 and plus 0.058.", b))


def fig_bp(path):
    sweep = STORE["bp_sweep"]
    tr = STORE["cccp_trace"]
    Mc = make_cycle(0.8)
    b = []
    W, H = 800, 480
    P1 = Panel(b, 70, 56, 320, 250, (0, 1.3), (0, 0.65))
    P1.frame([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "coupling w on each edge of the cycle", "largest error in E[x_i]", "BP and mean field against exact", True)
    P1.line([r[0] for r in sweep], [r[1] for r in sweep], "ln s1"); P1.line([r[0] for r in sweep], [r[2] for r in sweep], "ln s2")
    for r in sweep:
        P1.dot(r[0], r[1], "f1", 3.8); P1.dot(r[0], r[2], "f2", 3.8)
    legend_col(b, 80, 80, [("s1", "belief propagation"), ("s2", "mean field (11.125)")])
    its = np.arange(40)
    P2 = Panel(b, 450, 56, 320, 250, (0, 39), (-1.05, 1.05))
    P2.frame([0, 10, 20, 30], [-1, -0.5, 0, 0.5, 1], "iteration t", "belief E[x₀]", "w = 0.8: three iterations, one fixed point", True)
    P2.line(its, tr[0][:40], "ln s2"); P2.line(its, tr[2][:40], "ln s3"); P2.line(its, tr[4][:40], "ln s1")
    P2.line([0, 39], [Mc.eta[0], Mc.eta[0]], "dash s0")
    P2.text(39, Mc.eta[0] - 0.02, f"exact marginal {Mc.eta[0]:.4f}", "sm", "end", 0, 26)
    legend_col(b, 450, 372, [("s2", "simplified CCCP (11.144)-(11.145)"), ("s3", "Yuille's CCCP (11.148)-(11.149)"), ("s1", "belief propagation")], 16)
    note(b, 70, 440, ["Left: on a loop BP is wrong, but much less so than mean field, and both degrade with the coupling. Right: the simplified CCCP flips sign at",
                      "every step and never settles; Yuille's version lowers the Bethe free energy at every step and creeps towards the BP fixed point."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Belief propagation, mean field and CCCP on a four-cycle",
        "Left: the largest error of the belief propagation marginals and of the mean-field marginals against the exact ones on a four-spin cycle, for couplings from 0.1 to 1.2. Right: the belief of spin 0 over forty iterations at coupling 0.8 for the simplified CCCP, which oscillates between about plus and minus 0.9, for Yuille's CCCP, which creeps monotonically towards the BP fixed point near 0.52, and for BP itself; the exact marginal is 0.39.", b))


def fig_boost(path):
    R = STORE["boost"]
    b = []
    W, H = 800, 430
    ts = [r["t"] for r in R]
    P1 = Panel(b, 70, 56, 320, 250, (0.5, 10.5), (0, 2.4))
    P1.frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 0.5, 1.0, 1.5, 2.0], "round t", "weight α_t", "AdaBoost weight against the exact projection", True)
    P1.line(ts, [r["alpha"] for r in R], "ln s1"); P1.line(ts, [r["a_ml"] for r in R], "ln s2")
    for r in R:
        P1.dot(r["t"], r["alpha"], "f1", 3.8); P1.dot(r["t"], r["a_ml"], "f2", 3.8)
    legend_col(b, 160, 82, [("s2", "maximum likelihood, normalised (11.164)"), ("s1", "AdaBoost (11.174)")])
    P2 = Panel(b, 450, 56, 320, 250, (0.5, 10.5), (0, 1.0))
    P2.frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 0.25, 0.5, 0.75, 1.0], "round t", "", "Training error and the product of the Z_t", True)
    P2.line(ts, [r["prodZ"] for r in R], "ln s3"); P2.line(ts, [r["err"] for r in R], "ln s2")
    P2.line(ts, [2 * math.sqrt(r["eps"] * (1 - r["eps"])) for r in R], "dash s0")
    for r in R:
        P2.dot(r["t"], r["prodZ"], "f3", 3.8); P2.dot(r["t"], r["err"], "f2", 3.8)
    legend_col(b, 560, 160, [("s3", "mean exp(−yF) = Π Z_s"), ("s2", "training error"), ("dash s0", "Z_t = 2√(ε_t(1−ε_t))")])
    note(b, 70, 376, ["Left: at round 1 the exact maximum-likelihood weight is exactly twice AdaBoost's, which is the projection onto the unnormalised family (11.165).",
                      "Right: the training error stays below the product bound, which equals the mean exponential loss (Section 4 of the notes)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "AdaBoost on a twenty-point toy problem",
        "Left: the weight alpha of each of ten boosting rounds as computed by AdaBoost, one half log of one minus epsilon over epsilon, against the weight given by exact maximum likelihood for the normalised model: at the first round the latter is exactly twice as large. Right: the training error and the product of the normalisers Z_t, which equals the mean exponential loss, over the ten rounds; the error is below the product at every round and reaches zero at round five.", b))


def fig_rbm(path):
    kls, total, s1, s2, rem, t1, t2 = STORE["cd"]
    flows, pred, lam4, sv2, _W0, _C4 = STORE["gflow"]
    b = []
    W, H = 800, 430
    P1 = Panel(b, 70, 56, 320, 250, (0.5, 8.5), (-16, 0.5))
    lg = lambda v: math.log10(max(v, 1e-17))
    P1.frame([1, 2, 3, 4, 5, 6, 7, 8], [-15, -12, -9, -6, -3, 0], "step j", "log₁₀ of the term", "The Pythagorean decomposition (11.222)", True)
    P1.line(list(range(1, 9)), [lg(v) for v in t1[:8]], "ln s1"); P1.line(list(range(1, 9)), [lg(v) for v in t2[:8]], "ln s2")
    for j in range(8):
        P1.dot(j + 1, lg(t1[j]), "f1", 3.6); P1.dot(j + 1, lg(t2[j]), "f2", 3.6)
    legend_col(b, 210, 100, [("s1", "KL[p_j : p~_{j+1}]"), ("s2", "KL[p~_j : p_j]")])
    P1.text(1.2, lg(total) - 0.4, f"total KL[p_0 : p] = {total:.4f}", "sm", "start")
    P2 = Panel(b, 450, 56, 320, 250, (0, 6), (0, 1.0))
    P2.frame([0, 1, 2, 3, 4, 5, 6], [0, 0.25, 0.5, 0.75, 1.0], "time", "singular values of W", "Gaussian RBM, n = 4, m = 2", True)
    cols = {"ML (11.229)": "s1", "CD_1": "s2", "CD_3": "s3"}
    W0_, C4_ = STORE["gflow"][4], STORE["gflow"][5]
    for lab, kk in (("ML (11.229)", None), ("CD_1", 1), ("CD_3", 3)):
        _, traj = integrate_rbm(W0_, C4_, sv2, kk, T=6.0, dt=0.02, stride=2)
        tt = np.arange(len(traj)) * 2 * 0.02
        for k in range(2):
            P2.line(tt, traj[:, k], f"ln {cols[lab]}" if k == 0 else f"thin {cols[lab]}")
    for v in pred:
        P2.line([0, 6], [v, v], "dash s0")
    legend_col(b, 560, 258, [("s1", "ML"), ("s2", "CD_1"), ("s3", "CD_3")])
    note(b, 70, 376, ["Left: for a small RBM the first term is most of the total and the later ones fall off geometrically. Right: maximum likelihood, CD_1 and CD_3",
                      "end at the same W (dashed: sqrt(1 - σ_v²/λ_i), i = 1, 2); the equilibrium keeps the top two principal directions (Section 5.5 of the notes)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Contrastive divergence and the Gaussian RBM",
        "Left: on a small binary RBM, the terms of the Pythagorean decomposition of KL from p0 to p on a log scale: the first term KL of p0 from p tilde 1 is about 0.25, the later terms fall geometrically to rounding level, and their sum is the total KL of 0.2543. Right: the two singular values of the weight matrix of a Gaussian RBM trained by maximum likelihood, CD1 and CD3 flows, all converging to 0.894 and 0.825, the values of the square root of one minus sigma v squared over lambda.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_voronoi(out / "bregman-voronoi.svg")
    fig_chernoff(out / "chernoff.svg")
    fig_svm(out / "svm-conformal.svg")
    fig_bp(out / "bp-cccp.svg")
    fig_boost(out / "boosting.svg")
    fig_rbm(out / "rbm-cd.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_centres()
    check_kmeans()
    check_voronoi()
    check_soft_kmeans()
    check_tbd()
    check_chernoff()
    check_svm_linear()
    check_kernel()
    check_metric()
    check_conformal_svm()
    check_mean_field()
    check_bp()
    check_cccp()
    check_boosting()
    check_bayes()
    check_rbm()
    check_cd()
    check_gauss_rbm()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
