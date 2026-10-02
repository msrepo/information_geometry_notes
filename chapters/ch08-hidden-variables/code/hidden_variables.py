#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 8, checked by hand.

Estimation in the presence of hidden variables: the EM algorithm as alternating projections between a data manifold D and a model manifold M (section 8.1),
the loss of Fisher information under data reduction (8.2) and estimation with a misspecified model (8.3). Every number quoted in the notes comes from here.

Running examples. (a) A two-component Gaussian mixture with unit variances, xi = (w, mu_1, mu_2): a sample of N = 200 points from 0.35 N(-1, 1) + 0.65 N(1.5, 1), and the
whole density by quadrature; its one-parameter symmetric version 0.5 N(-m, 1) + 0.5 N(m, 1). (b) A restricted Boltzmann machine with 4 visible and 2 hidden binary units.
(c) Three binary neurons with the third-order moment discarded. (d) The curved family N(u, u^2) of Chapter 7 in the Gaussian family with statistics (x, x^2), with several
misspecified models q(x; v), and Gaussian neurons with a correlated truth N(r(u), V) against the model N(r(u), I).

Checked here, in the order the notes use them:

  1. the data manifold D and the model manifold M: D is m-flat and is the space of responsibility matrices; the chain rule behind Theorem 8.1; Lemma 8.1 by brute force and
     Pythagoras; (8.17) and (8.22) for one observation and for a sample;
  2. EM is em: the E-step as the e-projection, the M-step as moment matching (the m-projection onto an e-flat model), brute-force Newton against (8.26); the exact bookkeeping of
     Theorem 8.2 (the two half-steps lower the divergence by two explicit Kullback-Leibler divergences); EM as a natural-gradient step in the eta chart;
     what changes when the complete-data model is curved;
  3. what 'converges to an equilibrium' means: stationary points, a saddle that EM does not leave, two local maxima, a stall at the null model, and the singular
     likelihood when the variances are unknown;
  4. the speed of EM is the fraction of missing information, I - G_X^-1 G_Y, with (8.30) checked by quadrature (sample, population, symmetric model);
  5. the restricted Boltzmann machine: (8.10)-(8.11), the factorised posterior (8.13), EM with a Newton M-step and the em algorithm by brute force;
  6. loss of information (8.28)-(8.31): grouping, censoring, x-bar and the sum of squares, with the identity g^X = g^T + g^(X|T) computed three ways;
  7. three neurons: the lost third-order moment, exact enumeration, the mixed-coordinates formula, the em algorithm for the reduced data (its rate is the loss fraction) and the
     estimator that minimises KL[D : M] against the full MLE and against least squares;
  8. misspecified models: the sign in (8.39)-(8.41), Theorem 8.3 (pseudo-true values), Theorem 8.4 (efficiency by geometry, sandwich, score correlation and Monte Carlo);
  9. the neural field of (8.34)-(8.36): efficiency of the decoder that ignores correlation, two neurons in the five-dimensional family and rings of neurons.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Monte Carlo uses fixed seeds and the exact laws of the sufficient statistics; the checks take about five seconds (about ten with --figures).

Run:  python3 hidden_variables.py            (checks)
      python3 hidden_variables.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np

STORE = {}                         # numbers computed by the checks, reused by the figures
SQ2PI = math.sqrt(2 * math.pi)


def head(s):
    print("\n" + s)


def Phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2)))


def phi(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQ2PI


def grad_fd(f, x, e=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros(len(x))
    for i in range(len(x)):
        d = np.zeros(len(x)); d[i] = e
        g[i] = (f(x + d) - f(x - d)) / (2 * e)
    return g


def hess_fd(f, x, e=1e-4):
    x = np.asarray(x, dtype=float)
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            di = np.zeros(n); di[i] = e
            dj = np.zeros(n); dj[j] = e
            H[i, j] = (f(x + di + dj) - f(x + di - dj) - f(x - di + dj) + f(x - di - dj)) / (4 * e * e)
    return H


# ------------------------------------------------------------------ the Gaussian mixture with unit variances, xi = (w, mu)

def lognorm(y, mu):
    return -0.5 * (y[:, None] - mu[None, :]) ** 2 - 0.5 * math.log(2 * math.pi)


def resp(y, w, mu):
    """Responsibilities p(h|y_i, xi) (the E-step (8.24)) and log p_Y(y_i; xi)."""
    a = np.log(w)[None, :] + lognorm(y, mu)
    m = a.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(a - m).sum(1))
    return np.exp(a - lse[:, None]), lse


def loglik(y, w, mu):
    return float(resp(y, w, mu)[1].sum())


def em_step(y, w, mu):
    """One EM step, the closed form (8.26)."""
    R, _ = resp(y, w, mu)
    n = R.sum(0)
    return n / len(y), (R * y[:, None]).sum(0) / n


def score_rows(y, w, mu):
    """Per-observation score d log p_Y(y_i; xi)/d (w_1, mu_1, mu_2) for the two-component model, by Fisher's identity: the posterior mean of the complete-data score
    (1(h=1)/w_1 - 1(h=2)/w_2, 1(h=1)(y - mu_1), 1(h=2)(y - mu_2))."""
    R, _ = resp(y, w, mu)
    return np.stack([R[:, 0] / w[0] - R[:, 1] / w[1], R[:, 0] * (y - mu[0]), R[:, 1] * (y - mu[1])], 1)


def grad_loglik(y, x):
    w, mu = xi_unpack(x)
    return score_rows(y, w, mu).sum(0)


def free_energy(R, y, w, mu):
    """F(R, xi) = (1/N) sum_i sum_k R_ik [log R_ik - log w_k - log phi(y_i - mu_k)]: the divergence of the point R of D from the point xi of M,
    with the constant c of (8.15) dropped (for the empirical distribution of continuous data that constant is infinite)."""
    lp = np.log(w)[None, :] + lognorm(y, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(R > 0, R * np.log(R), 0.0)
    return float((t - R * lp).sum() / len(y))


def kl_rows(P, Q):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0, P * np.log(P / Q), 0.0).sum(1)


def kl_joint(w1, mu1, w2, mu2):
    """KL[p_{xi1} : p_{xi2}] between two complete-data distributions p(y, h; xi) of (8.9)."""
    return float((w1 * (np.log(w1 / w2) + 0.5 * (mu1 - mu2) ** 2)).sum())


def make_sample():
    """N = 200 draws from 0.35 N(-1, 1) + 0.65 N(1.5, 1), rounded to three decimals (the interactive page embeds the same numbers)."""
    rng = np.random.default_rng(20261002)
    N = 200
    h = rng.random(N) < 0.35
    y = np.where(h, rng.normal(-1.0, 1, N), rng.normal(1.5, 1, N))
    return np.round(y, 3)


Y200 = make_sample()
W_TRUE, MU_TRUE = np.array([0.35, 0.65]), np.array([-1.0, 1.5])
W_START, MU_START = np.array([0.5, 0.5]), np.array([-0.5, 0.5])


def xi_pack(w, mu):
    return np.array([w[0], mu[0], mu[1]])


def xi_unpack(x):
    return np.array([x[0], 1 - x[0]]), np.array([x[1], x[2]])


def em_map(y, x):
    w, mu = xi_unpack(x)
    w2, mu2 = em_step(y, w, mu)
    return xi_pack(w2, mu2)


def em_run(y, x, tol=1e-15, maxit=100000):
    traj = [x.copy()]
    for it in range(maxit):
        x2 = em_map(y, x)
        traj.append(x2.copy())
        if np.abs(x2 - x).max() < tol:
            return x2, np.array(traj), it + 1
        x = x2
    return x, np.array(traj), maxit


# ------------------------------------------------------------------ 1. the data manifold D and the model manifold M

def check_manifolds():
    head("1. The data manifold D and the model manifold M (section 8.1.1-8.1.2, Theorem 8.1, Lemma 8.1)")
    y = Y200; N = len(y)
    # D is m-flat and not e-flat: a discrete toy, y in {0,1,2}, h in {0,1}
    qy = np.array([0.5, 0.3, 0.2])
    a1 = np.array([0.1, 0.5, 0.9]); a2 = np.array([0.8, 0.4, 0.2])          # two points of D: q(h=1 | y)
    def joint(a):
        return np.stack([qy * (1 - a), qy * a], 1)
    q1, q2 = joint(a1), joint(a2)
    mm = 0.3 * q1 + 0.7 * q2
    em_ = q1 ** 0.5 * q2 ** 0.5; em_ /= em_.sum()
    print(f"   D is m-flat (8.6)-(8.7): the mixture 0.3 q1 + 0.7 q2 of two points of D has y-marginal {mm.sum(1).tolist()} = q_Y {qy.tolist()} (max deviation {np.abs(mm.sum(1) - qy).max():.1e});"
          f" the e-geodesic point (normalised geometric mean) has y-marginal {np.round(em_.sum(1), 4).tolist()}, deviation {np.abs(em_.sum(1) - qy).max():.4f}: D is not e-flat")
    # D for the sample is the set of responsibility matrices
    print(f"   For the N = {N} sample from 0.35 N(-1,1) + 0.65 N(1.5,1) a point of D is an N x 2 matrix R of rows summing to 1 (the responsibilities); D is a product of {N} intervals, a convex set;"
          f" sample mean {y.mean():.4f}, sample variance {y.var():.4f}; sum {y.sum():.3f}, sum of squares {(y ** 2).sum():.3f} (checksums for the interactive page, which embeds the same 200 numbers)")
    # orthogonality of the e-geodesic to D (discrete toy)
    pm = rng_toy = np.random.default_rng(5).dirichlet(np.ones(6)).reshape(3, 2)
    qs = qy[:, None] * pm / pm.sum(1, keepdims=True)                         # the e-projection of pm onto D
    tangent_e = np.log(qs / pm)                                              # tangent of the e-geodesic (theta-components) from pm to qs
    worst_orth = 0.0
    for yi in range(3):
        dq = np.zeros((3, 2)); dq[yi] = qy[yi] * np.array([1.0, -1.0])        # a tangent vector of D (m-components): moves mass between h = 0, 1 at fixed y
        worst_orth = max(worst_orth, abs((tangent_e * dq).sum()))
    print(f"   The e-geodesic from a model point p to its e-projection q* has tangent log(q*/p) = log(q_Y(y)/p_Y(y)), a function of y alone; its pairing with every tangent vector of D (which only moves mass between h values at a fixed y) is {worst_orth:.1e}: the projection is orthogonal to D")
    # chain rule behind Theorem 8.1
    rr = np.random.default_rng(1)
    worst, worst_prof = 0.0, 0.0
    for _ in range(300):
        a = rr.random(N); R = np.stack([a, 1 - a], 1)
        wv = rr.dirichlet([2, 2]); muv = rr.normal(0, 2, 2)
        post, lse = resp(y, wv, muv)
        worst = max(worst, abs(free_energy(R, y, wv, muv) - (-lse.sum() / N + kl_rows(R, post).sum() / N)))
        worst_prof = max(worst_prof, abs(free_energy(post, y, wv, muv) + lse.sum() / N))
    print(f"   chain rule behind (8.15): F(R, xi) = -loglik(xi)/N + (1/N) sum_i KL[R_i || p(h|y_i; xi)] holds for 300 random (R, xi) to {worst:.1e};"
          f" hence min over D of F is -loglik/N, attained at R = posterior: |F(posterior, xi) + loglik/N| <= {worst_prof:.1e}. This is Theorem 8.1: minimising the divergence between D and M is maximising the likelihood")
    # Lemma 8.1: brute force e-projection by golden-section search, and Pythagoras
    w0, mu0 = W_START, MU_START
    post, _ = resp(y, w0, mu0)
    lp = np.log(w0)[None, :] + lognorm(y, mu0)
    lo = np.full(N, 1e-12); hi = np.full(N, 1 - 1e-12); g = (math.sqrt(5) - 1) / 2
    f = lambda a: a * np.log(a) + (1 - a) * np.log(1 - a) - a * lp[:, 0] - (1 - a) * lp[:, 1]
    x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo); f1 = f(x1); f2 = f(x2)
    for _ in range(100):
        m = f1 < f2
        hi = np.where(m, x2, hi); lo = np.where(m, lo, x1)
        x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo); f1 = f(x1); f2 = f(x2)
    brute = (lo + hi) / 2
    print(f"   Lemma 8.1 / (8.21): the minimiser over D of the divergence to the point xi_0 = (w = 0.5, mu = -0.5, 0.5) of M, found with no formula by golden-section search on each row,"
          f" is the posterior p(h|y_i; xi_0): largest difference {np.abs(brute - post[:, 0]).max():.1e} (the limit of a search on function values)")
    worst = 0.0
    for _ in range(2000):
        a = rr.random(N); R = np.stack([a, 1 - a], 1)
        worst = max(worst, abs(free_energy(R, y, w0, mu0) - free_energy(post, y, w0, mu0) - kl_rows(R, post).sum() / N))
    print(f"      Pythagoras for the e-projection onto the m-flat D: F(R, xi_0) = F(R*, xi_0) + (1/N) sum_i KL[R_i || R*_i] for 2000 random R in D, to {worst:.1e}"
          f" (so R* is the global minimiser, the 'e-projection' of the book)")
    # (8.17) and (8.22): one observation versus a sample
    x_hat, _, _ = em_run(y, xi_pack(W_START, MU_START))
    w_h, mu_h = xi_unpack(x_hat)
    S = score_rows(y, w_h, mu_h)                         # one row per observation
    pY = np.exp(resp(y, w_h, mu_h)[1])
    xr = np.array([0.4, -0.7, 1.9]); wr, mur = xi_unpack(xr)
    print(f"   (8.17) and (8.22) are written for one observed y: the derivative of log p(y,h;xi) carries a factor 1/p_Y(y; xi), a constant for one y that the printed (8.17) leaves out; it cannot be dropped after summing over the sample."
          f" Fisher's identity d loglik/d xi = sum_i E_(h|y_i)[d log p(y_i,h;xi)/d xi] agrees with finite differences of the likelihood at a random xi to {np.abs(grad_loglik(y, xr) - grad_fd(lambda x: loglik(y, *xi_unpack(x)), xr)).max():.1e}.")
    print(f"   At the MLE of the sample the likelihood equation sum_i d log p_Y(y_i)/d xi = 0 holds (largest component {np.abs(S.sum(0)).max():.1e}), whereas the printed form summed without the weights, sum_i d p_Y(y_i)/d xi = {np.round((pY[:, None] * S).sum(0), 4).tolist()}, is not zero:"
          f" it is the gradient of sum_i p_Y(y_i), which is linear in w and is not the likelihood. Per observation the two are proportional, which is all (8.22) needs")
    STORE["mle200"] = x_hat


# ------------------------------------------------------------------ 2. EM is em

def theta_of_xi(x):
    """Natural parameters of the complete-data family: log p(y,h) = base(y) + theta_a 1(h=1) + theta_b y 1(h=1) + theta_c y 1(h=2) - psi."""
    w, mu = xi_unpack(x)
    return np.array([math.log(w[0] / w[1]) - mu[0] ** 2 / 2 + mu[1] ** 2 / 2, mu[0], mu[1]])


def xi_of_theta(th):
    a, b, c = th
    e1, e2 = math.exp(a + b * b / 2), math.exp(c * c / 2)
    w1 = e1 / (e1 + e2)
    return np.array([w1, b, c])


def eta_of_xi(x):
    w, mu = xi_unpack(x)
    return np.array([w[0], w[0] * mu[0], w[1] * mu[1]])


def check_em_is_em():
    head("2. The EM algorithm is the em algorithm (section 8.1.3-8.1.4, Theorem 8.2)")
    y = Y200; N = len(y)
    x0 = xi_pack(W_START, MU_START)
    w0, mu0 = W_START, MU_START
    R, _ = resp(y, w0, mu0)
    w1, mu1 = em_step(y, w0, mu0)
    eta_new = eta_of_xi(xi_pack(w1, mu1))
    eta_q = np.array([R[:, 0].mean(), (R[:, 0] * y).mean(), (R[:, 1] * y).mean()])
    print(f"   The complete-data model M is the exponential family with statistics (1(h=1), y 1(h=1), y 1(h=2)): p = base * exp(theta.F - psi), e-flat, with expectation parameters eta = (w_1, w_1 mu_1, w_2 mu_2).")
    print(f"   M-step = m-projection of R onto M = moment matching: after one step from xi_0, eta(xi_1) = {np.round(eta_new, 6).tolist()} and E_R[F] = {np.round(eta_q, 6).tolist()}"
          f" (difference {np.abs(eta_new - eta_q).max():.1e}); this is exactly (8.26): w_h = mean of the responsibilities, mu_h = their weighted mean of y")
    # brute force m-projection: Newton with numerical derivatives on F(R, .)
    def Fth(th):
        w = 1 / (1 + np.exp(-th[0]))
        return free_energy(R, y, np.array([w, 1 - w]), th[1:3])
    th = np.array([0.0, -0.5, 0.5])
    for _ in range(30):
        gg = grad_fd(Fth, th, 1e-4); H = hess_fd(Fth, th, 1e-4)
        step = np.linalg.solve(H, gg); th = th - step
        if np.abs(step).max() < 1e-10:
            break
    wb = 1 / (1 + np.exp(-th[0]))
    print(f"      the same projection by damped Newton on F(R, .) with numerical derivatives and no use of (8.26): (w_1, mu_1, mu_2) = ({wb:.8f}, {th[1]:.8f}, {th[2]:.8f}) against ({w1[0]:.8f}, {mu1[0]:.8f}, {mu1[1]:.8f}):"
          f" largest difference {max(abs(wb - w1[0]), np.abs(th[1:3] - mu1).max()):.1e}")
    rr = np.random.default_rng(2)
    worst = 0.0
    for _ in range(1000):
        wv = rr.dirichlet([2, 2]); muv = rr.normal(0, 2, 2)
        worst = max(worst, abs(free_energy(R, y, wv, muv) - free_energy(R, y, w1, mu1) - kl_joint(w1, mu1, wv, muv)))
    print(f"      Pythagoras for the m-projection onto the e-flat M: F(R, xi') - F(R, xi_1) = KL[p_xi1 : p_xi'] (complete-data KL) for 1000 random xi', to {worst:.1e}; this is why the M-step is unique and global")
    # Theorem 8.2: the exact bookkeeping
    print("   Theorem 8.2: the two half-steps lower F by exactly (M-step) KL[p_(t+1) : p_t], the KL of the complete-data distributions, and (E-step) (1/N) sum_i KL[p(h|y_i;xi_t) || p(h|y_i;xi_(t+1))];")
    print("   their sum is the gain in loglik/N:   step  loglik (before)  gain/N         M-step part    E-step part    |gain - sum|")
    w, mu = w0.copy(), mu0.copy(); worst = 0.0; table = []
    for t in range(40):
        Rt, lse = resp(y, w, mu); l0 = lse.sum()
        w2, mu2 = em_step(y, w, mu)
        R2, lse2 = resp(y, w2, mu2); l1 = lse2.sum()
        Mp = kl_joint(w2, mu2, w, mu); Ep = kl_rows(Rt, R2).sum() / N
        worst = max(worst, abs((l1 - l0) / N - Mp - Ep))
        table.append((l0, (l1 - l0) / N, Mp, Ep))
        if t in (0, 1, 2, 3, 9, 19, 39):
            print(f"                                       {t + 1:<5d} {l0:<16.6f} {(l1 - l0) / N:<14.4e} {Mp:<14.4e} {Ep:<14.4e} {abs((l1 - l0) / N - Mp - Ep):.1e}")
        w, mu = w2, mu2
    print(f"   the identity holds to {worst:.1e} at every one of the 40 steps, and both parts are positive: KL[D:M] never increases (Theorem 8.2); a step can only stop when xi_(t+1) = xi_t (both parts vanish)")
    STORE["gain_table"] = table
    # a curved complete-data model: weights fixed at 1/2, means (0, m): the exact accounting of the M-step needs M e-flat
    def post1(m_):
        return 1 / (1 + np.exp(-0.5 * y ** 2 + 0.5 * (y - m_) ** 2))
    def ll_curved(m_):
        return float(np.log(0.5 * np.exp(-0.5 * (y - m_) ** 2) / SQ2PI + 0.5 * np.exp(-0.5 * y ** 2) / SQ2PI).sum())
    mc = 1.0; rows_c = []
    for _ in range(4):
        R1 = post1(mc); mn = float((R1 * y).sum() / R1.sum()); R1n = post1(mn)
        Mg = float((R1 * (-0.5 * (y - mn) ** 2 + 0.5 * (y - mc) ** 2)).sum() / N)             # F(R, m) - F(R, m')
        KLc = (mn - mc) ** 2 / 4                                                                 # KL[p_m' : p_m] for the complete data
        Ep_ = kl_rows(np.stack([R1, 1 - R1], 1), np.stack([R1n, 1 - R1n], 1)).sum() / N
        gn = (ll_curved(mn) - ll_curved(mc)) / N
        rows_c.append((Mg / KLc, abs(gn - Mg - Ep_), abs(gn - KLc - Ep_)))
        mc = mn
    print(f"   When M is curved the exact accounting changes: complete-data model with the weights fixed at 1/2 and means (0, m), M-step m' = sum R_i1 y_i / sum R_i1. The M-step gain F(R,m) - F(R,m') is no longer KL[p_m' : p_m] = (m' - m)^2/4"
          f" but (mean responsibility) (m' - m)^2/2: ratio {', '.join(f'{r[0]:.4f}' for r in rows_c)} over the first four steps (above and below 1, so not even a one-sided inequality);"
          f" what survives is gain = (F-gain of the M-step) + (E part) to {max(r[1] for r in rows_c):.1e}, while gain = KL + E part misses by {', '.join(f'{r[2]:.1e}' for r in rows_c)}")
    # EM as natural-gradient step in the eta chart
    def ll_theta(th):
        x = xi_of_theta(th); w_, mu_ = xi_unpack(x)
        return loglik(y, w_, mu_) / N
    worst = 0.0
    x = x0.copy()
    for t in range(8):
        th = theta_of_xi(x)
        g = grad_fd(ll_theta, th, 1e-6)
        x2 = em_map(y, x)
        worst = max(worst, np.abs(eta_of_xi(x2) - eta_of_xi(x) - g).max())
        x = x2
    print(f"   EM as a gradient step: in the eta chart of the complete-data family the update is eta_(t+1) - eta_t = (1/N) d loglik / d theta (Fisher's identity), checked for 8 steps by finite differences of the likelihood in theta: largest difference {worst:.1e};"
          f" since d eta = G_X d theta, this is a natural-gradient step of unit length, with the complete-data Fisher metric G_X, taken along the m-geodesic (a straight line in eta)")


# ------------------------------------------------------------------ 3. what the algorithm converges to

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(200)
GH_W = GH_W / SQ2PI                         # nodes and weights of E[f(Z)], Z ~ N(0,1)


def sample_three_clusters():
    rng = np.random.default_rng(3)
    centers = np.array([-10.0, 0.0, 10.0]); N = 300
    comp = rng.choice(3, size=N, p=[0.4, 0.2, 0.4])
    return rng.normal(centers[comp], 1.0)


def em_free_step(y, w, mu, s2):
    """One EM step for the mixture with unknown means, variances and weights; returns the loglik at the input and the new parameters."""
    with np.errstate(all="ignore"):
        lp = np.log(w)[None, :] - 0.5 * np.log(2 * np.pi * s2)[None, :] - 0.5 * (y[:, None] - mu[None, :]) ** 2 / s2[None, :]
        mx = lp.max(1, keepdims=True); lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
        R = np.exp(lp - lse[:, None]); n = R.sum(0)
        w2 = n / len(y); mu2 = (R * y[:, None]).sum(0) / n
        s22 = (R * (y[:, None] - mu2[None, :]) ** 2).sum(0) / n
    return float(lse.sum()), w2, mu2, s22


def check_limits():
    head("3. What 'converges to an equilibrium' means (Theorem 8.2 and the remark after it)")
    y = Y200; N = len(y)
    x_hat, traj, it = em_run(y, xi_pack(W_START, MU_START))
    w, mu = xi_unpack(x_hat)
    ll_f = lambda x: loglik(y, *xi_unpack(x))
    H = hess_fd(ll_f, x_hat)
    print(f"   Fixed points of EM are stationary points of the likelihood: from (w, mu) = (0.5, -0.5, 0.5) EM stops after {it} steps at (w_1, mu_1, mu_2) = ({x_hat[0]:.6f}, {x_hat[1]:.6f}, {x_hat[2]:.6f}),"
          f" loglik {ll_f(x_hat):.6f}, gradient (Fisher's identity) {np.abs(grad_loglik(y, x_hat)).max():.1e}, Hessian eigenvalues {np.round(np.linalg.eigvalsh(H), 2).tolist()}: all negative, a strict local maximum")
    # the saddle on the diagonal
    ybar = y.mean()
    xs = np.array([0.5, ybar, ybar])
    Hs = hess_fd(ll_f, xs)
    J = np.zeros((3, 3)); e = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = e
        J[:, j] = (em_map(y, xs + d) - em_map(y, xs - d)) / (2 * e)
    ev = np.sort(np.linalg.eigvals(J).real)[::-1]
    print(f"   A stationary point that is not a maximum: start with equal means, mu_1 = mu_2 = sample mean {ybar:.4f}, w = 0.5. The gradient is {np.abs(grad_loglik(y, xs)).max():.1e}, the loglik is {ll_f(xs):.4f} against {ll_f(x_hat):.4f} at the maximum,"
          f" the Hessian eigenvalues are {np.round(np.linalg.eigvalsh(Hs), 2).tolist()} (a saddle), and the Jacobian of the EM map has eigenvalues {np.round(ev, 4).tolist()}: the unstable multiplier {ev[0]:.4f} is the sample variance {y.var():.4f}")
    x = xs.copy(); gap = []
    for t in range(1, 61):
        x = em_map(y, x)
        gap.append(abs(x[1] - x[2]))
    marks = [(t, gap[t - 1]) for t in (1, 10, 20, 30, 35, 40, 45, 50)]
    print("      EM started exactly there stays (|mu_1 - mu_2| after steps " + ", ".join(f"{t}: {g_:.1e}" for t, g_ in marks) + "),"
          f" until rounding error (1e-16) is amplified by {ev[0]:.2f} per step; it then leaves, and ends at loglik {ll_f(em_run(y, x)[0]):.4f}. An 'equilibrium' is not a maximum")
    xp = xs + np.array([0.0, 1e-6, -1e-6]); K = None
    for t in range(1, 300):
        xp = em_map(y, xp)
        if abs(xp[1] - xp[2]) > 0.5:
            K = t; break
    print(f"      from mu_1, mu_2 = sample mean +- 1e-6 (exactly symmetric arithmetic would stay for ever) the gap exceeds 0.5 after {K} steps (the linear estimate 2e-6 x 2.67^{K} = {2e-6 * 2.6667 ** K:.2f}), and EM then ends at loglik {ll_f(em_run(y, xp)[0]):.4f}")
    # two local maxima
    y3 = sample_three_clusters()
    print(f"   Local maxima although every M-step is unique (M is e-flat): N = {len(y3)} points from three clusters at -10, 0, 10 (weights 0.4, 0.2, 0.4), fitted with two unit-variance components, w = (0.5, 0.5):")
    res3 = []
    for mu0 in ((-10.0, 5.0), (-5.0, 10.0)):
        xx, _, it3 = em_run(y3, xi_pack(np.array([0.5, 0.5]), np.array(mu0)), tol=1e-13)
        H3 = hess_fd(lambda x: loglik(y3, *xi_unpack(x)), xx)
        res3.append((xx, loglik(y3, *xi_unpack(xx))))
        print(f"      start mu = {mu0}: {it3} steps, (w_1, mu_1, mu_2) = ({xx[0]:.4f}, {xx[1]:.4f}, {xx[2]:.4f}), loglik {res3[-1][1]:.4f}, gradient {np.abs(grad_loglik(y3, xx)).max():.1e}, Hessian eigenvalues {np.round(np.linalg.eigvalsh(H3), 1).tolist()}")
    print(f"      two strict local maxima, {res3[0][1] - res3[1][1]:.2f} apart in loglik: every M-step is unique, but the divergence is not jointly convex in (R, xi), and the limit depends on the start (the book's 'might exist local minima')")
    STORE["limits3"] = res3
    # stall at the null model
    m = 1.0; it = 0
    F0 = lambda m_: float((GH_W * GH_X * np.tanh(m_ * GH_X)).sum())
    ms = {}
    while m > 0.01 and it < 100000:
        m = F0(m); it += 1
        if it in (10, 100, 1000):
            ms[it] = m
    print(f"   Stalling: the symmetric model 0.5 N(-m, 1) + 0.5 N(m, 1) fitted to data from N(0, 1) (exact expectations, so no sampling): the EM map is F(m) = E[Z tanh(m Z)] = m - m^3 + O(m^5), slope 1 at m = 0, so the iteration crawls (sub-linear):"
          f" starting at m = 1, m_t = {ms[10]:.5f}, {ms[100]:.5f}, {ms[1000]:.5f} after 10, 100, 1000 steps (the prediction 1/sqrt(2t+1) gives {1 / math.sqrt(21):.5f}, {1 / math.sqrt(201):.5f}, {1 / math.sqrt(2001):.5f}),"
          f" and it takes {it} steps to get below 0.01 (prediction 1/(2*0.01^2) = {1 / (2 * 0.01 ** 2):.0f})")
    STORE["stall_steps"] = it
    # one observation: the staircase of the first figure
    yy = 1.8; mm = 0.3; its = []
    for _ in range(8):
        a_ = 1 / (1 + math.exp(-2 * yy * mm)); mm = (2 * a_ - 1) * yy; its.append(mm)
    f_ = lambda m_: m_ - yy * math.tanh(m_ * yy)
    lo_, hi_ = 1.0, 3.0
    for _ in range(200):
        mid = 0.5 * (lo_ + hi_)
        if f_(mid) < 0: lo_ = mid
        else: hi_ = mid
    m_star = 0.5 * (lo_ + hi_)
    print(f"   One observation y = {yy}, the symmetric model: the EM map is m -> y tanh(m y), the staircase of the first figure. From m = 0.3: m_t = {', '.join(f'{v:.5f}' for v in its[:6])}; the fixed point is m* = {m_star:.6f}"
          f" with slope y^2 sech^2(m* y) = {yy * yy / math.cosh(m_star * yy) ** 2:.5f} there, and the stationary point m = 0 has slope y^2 = {yy * yy:.4f} > 1: repelling (for |y| > 1 it is a local minimum of the likelihood in m)")
    STORE["stair"] = (yy, 0.3, its, m_star)
    # unknown variances: the general case the book mentions
    w_, mu_, s2_ = np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([1.0, 1.0])
    for _ in range(400):
        ll_reg, w_, mu_, s2_ = em_free_step(y, w_, mu_, s2_)
    j = int(np.argmax(y))
    ll0, w1_, mu1_, s21_ = em_free_step(y, np.array([0.5, 0.5]), np.array([y[j] + 0.02, 0.0]), np.array([1e-3, 2.0]))
    ll1, w2_, mu2_, s22_ = em_free_step(y, w1_, mu1_, s21_)
    print(f"   Unknown variances ('in a similar way', p. 180): the likelihood of the model with free variances is unbounded, and EM is still monotone, so it can run into the singularity. Regular start (variances 1): EM reaches loglik {ll_reg:.3f}"
          f" at w = {np.round(w_, 3).tolist()}, mu = {np.round(mu_, 3).tolist()}, sigma^2 = {np.round(s2_, 3).tolist()}. A component started on the isolated data point y = {y[j]:.3f} with variance 0.001, 0.02 away:"
          f" loglik {ll0:.3f} at the start, {ll1:.3f} after one step (variance {s21_[0]:.1e}, already above the regular maximum), and the next step sets the variance to {s22_[0]:.1e}: the loglik is then +infinity."
          f" With unit variances, as in the book's example, loglik <= N log(1/sqrt(2 pi)) = {-N / 2 * math.log(2 * math.pi):.1f}, no such trap; the maximum here is {ll_f(x_hat):.3f}")


# ------------------------------------------------------------------ 4. the speed of EM is the fraction of missing information

YG = np.arange(-14.0, 14.0 + 1e-9, 0.02)
DYG = 0.02


def pop_quant(w1, mu1, mu2):
    """Population quantities of the mixture w N(mu1,1) + (1-w) N(mu2,1) by quadrature on the grid YG: marginal density, the score of the marginal (Fisher's identity),
    G_Y = E[score^2], the missing information G_{X|Y} = E_y Cov_{h|y}(complete score), and the complete information G_X (quadrature and closed form)."""
    y = YG
    p1 = w1 * phi(y - mu1); p2 = (1 - w1) * phi(y - mu2); py = p1 + p2
    R1, R2 = p1 / py, p2 / py
    s1 = np.stack([np.full_like(y, 1 / w1), y - mu1, np.zeros_like(y)], 1)               # complete score if h = 1
    s2 = np.stack([np.full_like(y, -1 / (1 - w1)), np.zeros_like(y), y - mu2], 1)        # complete score if h = 2
    sY = R1[:, None] * s1 + R2[:, None] * s2
    GY = (py[:, None, None] * sY[:, :, None] * sY[:, None, :]).sum(0) * DYG
    E2 = R1[:, None, None] * s1[:, :, None] * s1[:, None, :] + R2[:, None, None] * s2[:, :, None] * s2[:, None, :]
    GM = (py[:, None, None] * (E2 - sY[:, :, None] * sY[:, None, :])).sum(0) * DYG
    GXq = (py[:, None, None] * E2).sum(0) * DYG
    GX = np.diag([1 / (w1 * (1 - w1)), w1, 1 - w1])
    return dict(py=py, GX=GX, GXq=GXq, GY=GY, GM=GM)


def pop_em_map(x, true=(0.35, -1.0, 1.5)):
    """The EM map when the 'data' are the whole true density (N -> infinity)."""
    w1, m1, m2 = x
    y = YG
    p1 = w1 * phi(y - m1); p2 = (1 - w1) * phi(y - m2)
    R1 = p1 / (p1 + p2)
    py0 = true[0] * phi(y - true[1]) + (1 - true[0]) * phi(y - true[2])
    n1 = (py0 * R1).sum() * DYG; n2 = (py0 * (1 - R1)).sum() * DYG
    return np.array([n1, (py0 * R1 * y).sum() * DYG / n1, (py0 * (1 - R1) * y).sum() * DYG / n2])


def sym_rate(m):
    """EM contraction at the true value for the symmetric model: E[y^2 sech^2(m y)] for y ~ N(m,1) (= 1 - I_o/I_c, I_c = 1), by quadrature on the grid YG."""
    return float((phi(YG - m) * YG * YG / np.cosh(m * YG) ** 2).sum() * DYG)


def sym_em_pop(mm, m):
    """The EM map m -> E[y tanh(mm y)] of the symmetric model when the data are the whole density 0.5 N(-m,1) + 0.5 N(m,1) (by symmetry, y ~ N(m,1))."""
    return float((phi(YG - m) * YG * np.tanh(mm * YG)).sum() * DYG)


def check_rates():
    head("4. The speed of EM is the fraction of missing information (not in the chapter; it joins sections 8.1 and 8.2)")
    y = Y200; N = len(y)
    x_hat, traj, it = em_run(y, xi_pack(W_START, MU_START))
    w1 = x_hat[0]
    J = np.zeros((3, 3)); e = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = e
        J[:, j] = (em_map(y, x_hat + d) - em_map(y, x_hat - d)) / (2 * e)
    Io_fd = -hess_fd(lambda x: loglik(y, *xi_unpack(x)), x_hat)
    Ic = N * np.diag([1 / (w1 * (1 - w1)), w1, 1 - w1])
    # Louis: I_m = sum_i Cov_{h|y_i}(complete score)
    w, mu = xi_unpack(x_hat)
    R, _ = resp(y, w, mu)
    s1 = np.stack([np.full(N, 1 / w[0]), y - mu[0], np.zeros(N)], 1); s2 = np.stack([np.full(N, -1 / w[1]), np.zeros(N), y - mu[1]], 1)
    sY = R[:, :1] * s1 + R[:, 1:] * s2
    Im = (R[:, 0, None, None] * s1[:, :, None] * s1[:, None, :] + R[:, 1, None, None] * s2[:, :, None] * s2[:, None, :]).sum(0) - sY.T @ sY
    DM = np.eye(3) - np.linalg.solve(Ic, Io_fd)
    ev = np.sort(np.linalg.eigvals(J).real)[::-1]
    print(f"   Sample N = {N}, at the fixed point: the Jacobian of the EM map (finite differences of the closed form (8.26)) has eigenvalues {np.round(ev, 5).tolist()};"
          f" I - I_c^-1 I_o, with the observed information I_o = -Hessian of loglik (finite differences) and the complete information I_c = N diag(1/(w(1-w)), w, 1-w), agrees entrywise to {np.abs(J - DM).max():.1e};"
          f" Louis' formula I_o = I_c - I_m (I_m = sum_i Cov_(h|y_i)[complete score]) agrees with the Hessian to {np.abs(Io_fd - (Ic - Im)).max():.1e}")
    d = np.array([np.linalg.norm(t - x_hat) for t in traj])
    rat = d[1:] / d[:-1]
    print(f"      observed contraction ||xi_(t+1) - xi*|| / ||xi_t - xi*|| at t = 10, 20, 40, 60: {rat[10]:.5f}, {rat[20]:.5f}, {rat[40]:.5f}, {rat[60]:.5f} -> the largest eigenvalue {ev[0]:.5f}")
    rr = np.random.default_rng(9); worst = 0.0
    for _ in range(5):
        w_, mu_ = em_step(y, rr.dirichlet([2, 2]), rr.normal(0, 2, 2))
        worst = max(worst, abs(w_ @ mu_ - y.mean()))
    print(f"      the eigenvalue 0 is the direction of the mixture mean w_1 mu_1 + w_2 mu_2: after one step it equals the sample mean exactly (largest deviation over 5 random starts {worst:.1e}):"
          f" the hidden label carries no missing information about that combination")
    STORE["sample_rates"] = ev
    # the population version
    q = pop_quant(0.35, -1.0, 1.5)
    ev_pop = np.sort(np.linalg.eigvals(np.linalg.solve(q["GX"], q["GM"])).real)[::-1]
    print("   Population version at the true value (0.35, -1, 1.5), quadrature on a grid of step 0.02 (the printed numbers are those of the interactive page's fourth widget):")
    print(f"      complete information G_X = diag(1/(w(1-w)), w, 1-w) = {np.round(np.diag(q['GX']), 4).tolist()}; by quadrature of E[score score^T] {np.round(np.diag(q['GXq']), 4).tolist()} (largest off-diagonal {np.abs(q['GXq'] - np.diag(np.diag(q['GXq']))).max():.1e})")
    print(f"      observed information G_Y = E[(score of p_Y)^2] =\n{np.round(q['GY'], 4)}")
    print(f"      missing information G_(X|Y) = E_y Cov_(h|y)[complete score] =\n{np.round(q['GM'], 4)}")
    print(f"      (8.30): |G_X - G_Y - G_(X|Y)| <= {np.abs(q['GX'] - q['GY'] - q['GM']).max():.1e}; G_(X|Y) is positive semidefinite (eigenvalues {np.round(np.linalg.eigvalsh(q['GM']), 4).tolist()})")
    def logpy(yy, x):
        return np.log(x[0] * phi(yy - x[1]) + (1 - x[0]) * phi(yy - x[2]))
    x0 = np.array([0.35, -1.0, 1.5]); e = 1e-4; Hs = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            di = np.zeros(3); di[i] = e; dj = np.zeros(3); dj[j] = e
            h = (logpy(YG, x0 + di + dj) - logpy(YG, x0 + di - dj) - logpy(YG, x0 - di + dj) + logpy(YG, x0 - di - dj)) / (4 * e * e)
            Hs[i, j] = -(q["py"] * h).sum() * DYG
    print(f"      information equality: -E[Hessian of log p_Y] (finite differences) agrees with E[score^2] to {np.abs(Hs - q['GY']).max():.1e}")
    Jp = np.zeros((3, 3)); e = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = e
        Jp[:, j] = (pop_em_map(x0 + d) - pop_em_map(x0 - d)) / (2 * e)
    evp = np.sort(np.linalg.eigvals(Jp).real)[::-1]
    print(f"      eigenvalues of G_X^-1 G_(X|Y) (the fraction of the information about each direction that is missing): {np.round(ev_pop, 5).tolist()};"
          f" eigenvalues of the Jacobian of the EM map run on the whole density: {np.round(evp, 5).tolist()}; |J - (I - G_X^-1 G_Y)| = {np.abs(Jp - (np.eye(3) - np.linalg.solve(q['GX'], q['GY']))).max():.1e}")
    x = np.array([0.5, -0.5, 0.5]); errs = []
    for t in range(60):
        x = pop_em_map(x); errs.append(np.linalg.norm(x - x0))
    print(f"      EM on the whole density from (0.5, -0.5, 0.5): the observed contraction at steps 30, 40, 50 is {errs[30] / errs[29]:.5f}, {errs[40] / errs[39]:.5f}, {errs[50] / errs[49]:.5f} (the sample of 200 gave {ev[0]:.5f})")
    STORE["pop_rate"] = ev_pop
    # the symmetric model: rate against separation
    print("   The symmetric model 0.5 N(-m, 1) + 0.5 N(m, 1): complete information 1, so rate = 1 - I_o(m) = E[y^2 sech^2(m y)]; EM run on the whole density from m_0 = m + 0.001:")
    print(f"      {'m':>5s} {'rate (formula)':>15s} {'I_o(m) = 1 - rate':>18s} {'measured contraction':>21s} {'steps to error 1e-10':>21s}")
    tab = []
    for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        rate = sym_rate(m)
        mm = m + 0.001; errs = [0.001]
        for t in range(5000):
            mm = sym_em_pop(mm, m)
            errs.append(abs(mm - m))
            if errs[-1] < 1e-10:
                break
        steps = len(errs) - 1
        # contraction measured at the last step whose error is still above 1e-7 (small enough for the linear regime, large enough to be free of rounding)
        meas = float("nan")
        for t in range(len(errs) - 1, 0, -1):
            if errs[t] >= 1e-7:
                meas = errs[t] / errs[t - 1]; break
        tab.append((m, rate, steps, meas))
        mtxt = f"{meas:>21.6f}" if meas == meas else f"{'(exact in 1 step)':>21s}"
        print(f"      {m:>5.2f} {rate:>15.6f} {1 - rate:>18.6f} {mtxt} {steps:>21d}")
    print("      far apart, nothing is missing and EM is exact after one or two steps; as the components merge the rate tends to 1 (at m = 0 it is exactly 1, the stall of section 3)")
    STORE["sym_table"] = tab


# ------------------------------------------------------------------ 5. the Boltzmann machine with hidden units (the book's second example)

RB_NV, RB_NH = 4, 2
RB_Y = np.array(list(itertools.product([0, 1], repeat=RB_NV)), dtype=float)        # 16 visible states
RB_H = np.array(list(itertools.product([0, 1], repeat=RB_NH)), dtype=float)        # 4 hidden states
RB_F = np.einsum("ai,bj->abij", RB_Y, RB_H).reshape(len(RB_Y), len(RB_H), RB_NV * RB_NH)      # the statistics y_i h_j


def rbm_joint(W):
    """p(y, h; W) proportional to exp(- y^T W h) on all 16 x 4 states, W an RB_NV x RB_NH array (the book's (8.11) without the 1/2, see the check)."""
    s = -(RB_F @ W.reshape(-1))
    s = s - s.max()
    p = np.exp(s)
    return p / p.sum()


def rbm_loglik(W, qy):
    return float((qy * np.log(rbm_joint(W).sum(1))).sum())


def rbm_mstep(Q, W0, iters=60):
    """m-projection of a joint Q (16 x 4) onto the Boltzmann-machine family: moment matching E_p[y_i h_j] = E_Q[y_i h_j], a convex problem in theta = -W, solved by damped Newton."""
    target = np.einsum("ab,abk->k", Q, RB_F)
    th = -W0.reshape(-1).copy()
    def obj(t):
        s = RB_F @ t; mx = s.max()
        return mx + math.log(np.exp(s - mx).sum()) - t @ target
    for _ in range(iters):
        s = RB_F @ th; s = s - s.max(); p = np.exp(s); p /= p.sum()
        mean = np.einsum("ab,abk->k", p, RB_F)
        C = np.einsum("ab,abk,abl->kl", p, RB_F, RB_F) - np.outer(mean, mean)
        g = target - mean
        if np.abs(g).max() < 1e-14:
            break
        step = np.linalg.solve(C + 1e-13 * np.eye(len(g)), g)
        t_ = 1.0; f0 = obj(th)
        while obj(th + t_ * step) > f0 + 1e-15 and t_ > 1e-6:
            t_ *= 0.5
        th = th + t_ * step
    return -th.reshape(RB_NV, RB_NH)


def rbm_info(W):
    """G_X (complete-data Fisher information in theta = -W), G_(X|Y) = E_y Cov_(h|y)[y_i h_j], and the visible marginal."""
    p = rbm_joint(W); py = p.sum(1)
    mean = np.einsum("ab,abk->k", p, RB_F)
    GX = np.einsum("ab,abk,abl->kl", p, RB_F, RB_F) - np.outer(mean, mean)
    post = p / py[:, None]
    cm = np.einsum("ab,abk->ak", post, RB_F)
    cc = np.einsum("ab,abk,abl->akl", post, RB_F, RB_F) - cm[:, :, None] * cm[:, None, :]
    GM = np.einsum("a,akl->kl", py, cc)
    return GX, GM, py


def check_rbm():
    head("5. The restricted Boltzmann machine (the book's second example, (8.10)-(8.13)): 4 visible and 2 hidden binary units, no biases")
    rr = np.random.default_rng(3)
    # (8.10) -> (8.11): the factor 1/2
    A = rr.normal(size=(RB_NV, RB_NH))
    Wf = np.zeros((RB_NV + RB_NH, RB_NV + RB_NH)); Wf[:RB_NV, RB_NV:] = A; Wf[RB_NV:, :RB_NV] = A.T       # symmetric, no connections inside a layer
    states = np.array(list(itertools.product([0, 1], repeat=RB_NV + RB_NH)), dtype=float)
    q_form = 0.5 * np.einsum("si,ij,sj->s", states, Wf, states)
    cross = np.einsum("si,ij,sj->s", states[:, :RB_NV], A, states[:, RB_NV:])
    print(f"   (8.10) with x = (y, h) and no connections inside a layer: (1/2) x^T W x = y^T A h exactly (largest difference over the 64 states {np.abs(q_form - cross).max():.1e}), where A is the visible-hidden block of the symmetric W;"
          f" so (8.11), which prints (1/2) y^T W h, agrees with (8.10) only if its W means twice the block. The exponent is -y^T A h")
    # (8.13): the posterior of the hidden units factorises
    W = rr.normal(0, 1.0, (RB_NV, RB_NH))
    p = rbm_joint(W); post = p / p.sum(1, keepdims=True)
    sig = 1 / (1 + np.exp(-(RB_Y @ W)))                      # P(h_j = 1 | y) = sigma(-(y^T W)_j) with the sign of the exponent -y^T W h: sigma(-z) = 1 - sigma(z)
    ph1 = 1 - sig
    prod = np.prod(np.where(RB_H[None, :, :] > 0, ph1[:, None, :], 1 - ph1[:, None, :]), axis=2)
    print(f"   (8.13): p(h|y; W) factorises over the hidden units, p(h_j = 1 | y) = sigma(-(W^T y)_j): largest deviation of the exact posterior from that product {np.abs(post - prod).max():.1e};"
          f" the E-step costs n_h sigmoids, although p_Y is a mixture of 2^(n_h) = {len(RB_H)} exponential-family members (8.12)")
    # EM with an exact M-step by Newton (no closed form), on the exact visible marginal of a true machine
    Wt = rr.normal(0, 1.0, (RB_NV, RB_NH))
    qy = rbm_joint(Wt).sum(1)
    W = Wt + 0.2 * rr.normal(size=Wt.shape)
    ll_prev = rbm_loglik(W, qy); worst_mm = 0.0; min_gain = 1.0
    traj = [W.copy()]
    for t in range(400):
        p = rbm_joint(W); py = p.sum(1)
        Q = qy[:, None] * p / py[:, None]                            # E-step: e-projection onto D
        W = rbm_mstep(Q, W)                                           # M-step: m-projection onto M
        mm = np.einsum("ab,abk->k", rbm_joint(W), RB_F) - np.einsum("ab,abk->k", Q, RB_F)
        worst_mm = max(worst_mm, np.abs(mm).max())
        ll = rbm_loglik(W, qy); min_gain = min(min_gain, ll - ll_prev); ll_prev = ll
        traj.append(W.copy())
    # em by brute force on the same machine: entropic mirror descent for each E-step row, Newton with finite-difference derivatives for the M-step
    def em_brute(W_, steps):
        for _ in range(steps):
            p_ = rbm_joint(W_); lp_ = np.log(p_)
            Q_ = np.zeros_like(p_)
            for a_ in range(len(RB_Y)):                                   # minimise sum_h q_h (log q_h - log p(y,h)) over the simplex, no formula for the minimiser
                q_ = np.full(len(RB_H), 1.0 / len(RB_H))
                for _ in range(80):
                    q_ = q_ ** 0.5 * np.exp(0.5 * lp_[a_]); q_ /= q_.sum()
                Q_[a_] = qy[a_] * q_
            def Fq(t):
                s_ = RB_F @ t; mx = s_.max()
                return mx + math.log(np.exp(s_ - mx).sum()) - float(np.einsum("ab,abk,k->", Q_, RB_F, t))      # = -E_Q[log p_theta] up to a constant
            th = -W_.reshape(-1).copy()
            for _ in range(25):
                gg = grad_fd(Fq, th, 1e-5); HH = hess_fd(Fq, th, 1e-3)
                st = np.linalg.solve(HH, gg); th = th - st
                if np.abs(st).max() < 1e-9:
                    break
            W_ = -th.reshape(RB_NV, RB_NH)
        return W_
    def em_exact(W_, steps):
        for _ in range(steps):
            p_ = rbm_joint(W_); py_ = p_.sum(1)
            W_ = rbm_mstep(qy[:, None] * p_ / py_[:, None], W_)
        return W_
    Ws = traj[0]
    print(f"   The em algorithm with no closed form anywhere (mirror descent for each E-step, Newton with finite differences for the M-step) against the posterior-plus-moment-matching version, from the same start: after 1, 3 and 5 steps the weights differ by"
          f" {np.abs(em_brute(Ws, 1) - em_exact(Ws, 1)).max():.1e}, {np.abs(em_brute(Ws, 3) - em_exact(Ws, 3)).max():.1e}, {np.abs(em_brute(Ws, 5) - em_exact(Ws, 5)).max():.1e}")
    H_true = -(qy * np.log(qy)).sum()
    print(f"   EM with the M-step solved by Newton (moment matching E_p[y_i h_j] = E_Q[y_i h_j], residual after every one of 400 steps <= {worst_mm:.1e}) on the exact visible marginal of a machine W_true:"
          f" the loglik rises at every step (smallest gain {min_gain:.1e}) to {ll_prev:.8f}, against the value {-H_true:.8f} at W_true")
    GX, GM, _ = rbm_info(Wt)
    ev = np.sort(np.linalg.eigvals(np.linalg.solve(GX, GM)).real)[::-1]
    inc = np.array([np.linalg.norm(traj[k + 1] - traj[k]) for k in range(len(traj) - 1)])
    print(f"   the fraction of missing information, eigenvalues of G_X^-1 G_(X|Y) at W_true: {np.round(ev, 4).tolist()}: {int((ev > 0.9).sum())} of the {len(ev)} directions have more than 90 percent of their information missing (the visible units identify the weights only weakly), so EM contracts by {ev[0]:.4f} per step along the slowest;"
          f" the observed ratio of consecutive increments at steps 300-400 is {np.mean(inc[300:399] / inc[299:398]):.4f}; reaching an error of 1e-3 along it takes about ln(1e-3)/ln({ev[0]:.4f}) = {math.log(1e-3) / math.log(ev[0]):.0f} steps")
    STORE["rbm_rate"] = ev


# ------------------------------------------------------------------ 6. loss of information by data reduction

def score_gauss(x, mu, s):
    """Score of N(mu, s^2) with respect to (mu, sigma): ((x - mu)/s^2, ((x - mu)^2 - s^2)/s^3)."""
    return np.stack([(x - mu) / s ** 2, ((x - mu) ** 2 - s ** 2) / s ** 3], -1)


def gl_panels(a, b, panels=8, nq=24):
    """Composite Gauss-Legendre nodes and weights on [a, b]."""
    xg, wg = np.polynomial.legendre.leggauss(nq)
    edges = np.linspace(a, b, panels + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo)); ws.append(0.5 * (hi - lo) * wg)
    return np.concatenate(xs), np.concatenate(ws)


def grouped_info(mu, s, h, K=60):
    """Fisher information of the binned data (bins of width h, edges at multiples of h) about (mu, sigma) of N(mu, s^2), from the multinomial formula; also the bin edges."""
    edges = np.arange(-K, K + 1) * h
    z = (edges - mu) / s
    P = Phi(z[1:]) - Phi(z[:-1])
    dmu = -(phi(z[1:]) - phi(z[:-1])) / s
    dsg = -(z[1:] * phi(z[1:]) - z[:-1] * phi(z[:-1])) / s
    D = np.stack([dmu, dsg], 1)
    m = P > 1e-300
    return (D[m][:, :, None] * D[m][:, None, :] / P[m][:, None, None]).sum(0), P, D, edges


def cond_info_bins(mu, s, edges, nq=30):
    """g^{X|T} = E_T[Cov(score | bin)] by quadrature inside each bin, and the largest deviation of E[score | bin] from d log P_bin (the score of T)."""
    xg, wg = np.polynomial.legendre.leggauss(nq)
    tot = np.zeros((2, 2)); dev = 0.0
    z = (edges - mu) / s
    P = Phi(z[1:]) - Phi(z[:-1])
    dmu = -(phi(z[1:]) - phi(z[:-1])) / s
    dsg = -(z[1:] * phi(z[1:]) - z[:-1] * phi(z[:-1])) / s
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if P[k] < 1e-300:
            continue
        x = 0.5 * (b - a) * xg + 0.5 * (b + a); w = 0.5 * (b - a) * wg * phi((x - mu) / s) / s
        p = w.sum(); sc = score_gauss(x, mu, s); m = (w[:, None] * sc).sum(0) / p
        tot += (w[:, None, None] * (sc - m)[:, :, None] * (sc - m)[:, None, :]).sum(0)
        if P[k] > 1e-6:
            dev = max(dev, np.abs(m - np.array([dmu[k], dsg[k]]) / P[k]).max())
    return tot, dev


def censored_info(mu, s, c):
    """T = min(x, c) for x ~ N(mu, s^2): g^T = int_{x<c} f score score^T + P(x>=c) sbar sbar^T, g^{X|T} = P(x>=c) Cov(score | x >= c), by quadrature."""
    xl, wl = gl_panels(mu - 12 * s, c, 12, 24)
    wl = wl * phi((xl - mu) / s) / s
    sc = score_gauss(xl, mu, s)
    xr, wr = gl_panels(c, mu + 12 * s + max(c - mu, 0), 12, 24)
    wr = wr * phi((xr - mu) / s) / s
    scr = score_gauss(xr, mu, s)
    Pc = wr.sum(); sbar = (wr[:, None] * scr).sum(0) / Pc
    gT = (wl[:, None, None] * sc[:, :, None] * sc[:, None, :]).sum(0) + Pc * np.outer(sbar, sbar)
    gC = (wr[:, None, None] * (scr - sbar)[:, :, None] * (scr - sbar)[:, None, :]).sum(0)
    return gT, gC, Pc


def check_reduction_gauss():
    head("6. Loss of information by data reduction (section 8.2): the identity (8.30) and four reductions")
    mu, s = 0.3, 1.0
    gX = np.diag([1 / s ** 2, 2 / s ** 2])
    print("   (8.28)-(8.30) say that the score of the whole data splits into the score of T plus the score of the data given T, orthogonally: g^X = g^T + g^(X|T), and the loss of information is g^(X|T) = E_t[Cov(score | T = t)].")
    print(f"   Grouping one observation of N({mu}, {s}^2) into bins of width h (T = the bin): g^T from the multinomial formula sum_k (d P_k)(d P_k)^T / P_k, g^(X|T) from the within-bin score covariance; relative losses = diagonal entries of g^(X|T) over those of g^X:")
    print(f"      {'h':>5s} {'g^T_mumu':>9s} {'g^T_sisi':>9s} {'loss mu':>9s} {'h^2/12':>9s} {'loss sigma':>10s} {'h^2/6':>9s} {'|g^X - g^T - g^(X|T)|':>22s} {'E[score|bin] - score of T':>26s}")
    rows = []
    for h in (0.25, 0.5, 1.0, 2.0, 3.0):
        gT, P, D, edges = grouped_info(mu, s, h)
        gC, dev = cond_info_bins(mu, s, edges)
        rows.append((h, gT, gC))
        print(f"      {h:>5.2f} {gT[0, 0]:>9.5f} {gT[1, 1]:>9.5f} {gC[0, 0] / gX[0, 0]:>9.5f} {h * h / 12 / s ** 2:>9.5f} {gC[1, 1] / gX[1, 1]:>10.5f} {h * h / 6 / s ** 2:>9.5f} {np.abs(gX - gT - gC).max():>22.1e} {dev:>26.1e}")
    STORE["group_rows"] = rows
    print("      small bins lose h^2/(12 sigma^2) of the information about mu and twice that about sigma (the conditional score is uniform-like inside a bin); the sum of the two terms is g^X to rounding in every row")
    # censoring
    print(f"   Censoring (T = min(x, c) for x ~ N({mu}, {s}^2)): the atom at c carries the score E[score | x >= c]; the lost part is P(x >= c) Cov(score | x >= c):")
    print(f"      {'c':>5s} {'P(x>=c)':>9s} {'loss mu':>9s} {'loss sigma':>11s} {'|g^X - g^T - g^(X|T)|':>22s}")
    crows = []
    for c in (-1.0, 0.0, 1.0, 2.0, 3.0):
        gT, gC, Pc = censored_info(mu, s, c)
        crows.append((c, Pc, gC[0, 0] / gX[0, 0], gC[1, 1] / gX[1, 1], gT))
        print(f"      {c:>5.1f} {Pc:>9.5f} {gC[0, 0] / gX[0, 0]:>9.5f} {gC[1, 1] / gX[1, 1]:>11.5f} {np.abs(gX - gT - gC).max():>22.1e}")
    STORE["censor_rows"] = crows
    # N = 3 observations: x-bar, S, both (exact, three-dimensional Gauss-Hermite quadrature)
    N = 3
    gx, gw = np.polynomial.hermite_e.hermegauss(14); gw = gw / SQ2PI
    Z = np.array(list(itertools.product(gx, repeat=N))); Wt = np.array([np.prod(c) for c in itertools.product(gw, repeat=N)])
    xs = mu + s * Z
    SX = score_gauss(xs, mu, s).sum(1)
    xb = xs.mean(1); Ssq = ((xs - xb[:, None]) ** 2).sum(1)
    ST_xb = np.stack([N * (xb - mu) / s ** 2, (N * (xb - mu) ** 2 - s ** 2) / s ** 3], 1)
    ST_S = np.stack([np.zeros(len(xb)), Ssq / s ** 3 - (N - 1) / s], 1)
    out = {}
    for name, ST in (("x-bar alone", ST_xb), ("sum of squares alone", ST_S), ("(x-bar, sum of squares)", ST_xb + ST_S)):
        gTm = (Wt[:, None, None] * ST[:, :, None] * ST[:, None, :]).sum(0)
        SXT = SX - ST
        gCm = (Wt[:, None, None] * SXT[:, :, None] * SXT[:, None, :]).sum(0)
        cross = (Wt[:, None, None] * ST[:, :, None] * SXT[:, None, :]).sum(0)
        out[name] = (gTm, gCm, cross)
        print(f"   N = {N} observations of N({mu}, {s}^2), T = {name}: g^T = diag({gTm[0, 0]:.4f}, {gTm[1, 1]:.4f}), g^(X|T) = diag({gCm[0, 0]:.4f}, {gCm[1, 1]:.4f}) (off-diagonal {abs(gCm[0, 1]):.1e}),"
              f" cross term E[score_T score_(X|T)] = {np.abs(cross).max():.1e}, g^X = diag({N / s ** 2:.4f}, {2 * N / s ** 2:.4f}) = g^T + g^(X|T) to {np.abs(np.diag([N / s ** 2, 2 * N / s ** 2]) - gTm - gCm).max():.1e}")
    print(f"      x-bar loses exactly the information of the sum of squares and vice versa (they are independent), 2(N-1)/sigma^2 = {2 * (N - 1) / s ** 2:.4f} for sigma and none for mu: the pair is sufficient, g^(X|T) = 0, the conditional score is identically zero"
          f" (largest |score_X - score_T| over the quadrature nodes {np.abs(SX - ST_xb - ST_S).max():.1e})")


# ------------------------------------------------------------------ 6b. three neurons: losing the third-order correlation

NR_X = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)               # the 8 firing patterns
NR_F = np.stack([NR_X[:, 0], NR_X[:, 1], NR_X[:, 2], NR_X[:, 0] * NR_X[:, 1], NR_X[:, 0] * NR_X[:, 2], NR_X[:, 1] * NR_X[:, 2],
                 NR_X[:, 0] * NR_X[:, 1] * NR_X[:, 2]], 1)                           # X = (x_i, x_i x_j, x_1 x_2 x_3) of (8.32)
NR_TH0 = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
NR_D = (-1.0) ** NR_X.sum(1)                                                          # the parity pattern: zero 1st and 2nd order moments


def nr_p(xi):
    """Firing-pattern distribution p(x; xi) = exp(theta(xi) . X - psi) with theta = (-1,-1,-1, 0,0,0, xi): sparse independent neurons plus a third-order interaction xi."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    th = np.tile(NR_TH0, (len(xi), 1)); th[:, 6] = xi
    s = th @ NR_F.T; s = s - s.max(1, keepdims=True); p = np.exp(s)
    return p / p.sum(1, keepdims=True)


def nr_eproj(q0, P):
    """e-projection of the model points P (R x 8) onto the data manifolds D_R = {q0 + s D: q >= 0} (R x 8): minimise KL[q : p] over s by safeguarded Newton."""
    pos = NR_D > 0
    smin = np.max(np.where(pos[None, :], -q0, -np.inf), 1)
    smax = np.min(np.where(~pos[None, :], q0, np.inf), 1)
    s = 0.5 * (smin + smax)
    for _ in range(40):
        q = np.maximum(q0 + s[:, None] * NR_D[None, :], 1e-300)
        g = (NR_D[None, :] * np.log(q / P)).sum(1); h = (NR_D[None, :] ** 2 / q).sum(1)
        s = np.minimum(np.maximum(s - g / h, smin + 1e-13 * (1 + np.abs(smin))), smax - 1e-13 * (1 + np.abs(smax)))
    return q0 + s[:, None] * NR_D[None, :]


def nr_mstep(target, xi):
    """m-projection onto the family: the xi with E_xi[x1 x2 x3] = target (Newton, vectorised)."""
    xi = xi.copy()
    for _ in range(40):
        pp = nr_p(xi); m1 = pp @ NR_F[:, 6]; m2 = pp @ (NR_F[:, 6] ** 2)
        xi = xi + (target - m1) / (m2 - m1 ** 2)
    return xi


def compositions(N, k):
    if k == 1:
        yield (N,)
        return
    for i in range(N + 1):
        for rest in compositions(N - i, k - 1):
            yield (i,) + rest


def nr_exact_info(N, xi):
    """Exact Fisher information of N patterns (g^X), of the retained statistics T = (n_i, n_ij) (g^T) and the conditional information E_T[Var(score | T)], by enumerating all count vectors."""
    cnts = np.array(list(compositions(N, 8)), dtype=float)
    lg = np.array([math.lgamma(n + 1) for n in range(N + 1)])
    p = nr_p(xi)[0]
    logP = math.lgamma(N + 1) - lg[cnts.astype(int)].sum(1) + cnts @ np.log(p)
    P = np.exp(logP)
    eta6 = p @ NR_F[:, 6]
    S = cnts @ (NR_F[:, 6] - eta6)                         # score of the whole data: N (mean of x1x2x3 - eta_123)
    T = np.round(cnts @ NR_F[:, :6]).astype(int)
    keys, inv = np.unique(T, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    PT = np.bincount(inv, weights=P)
    ST = np.bincount(inv, weights=P * S) / PT              # E[score | T] = score of T
    return float((P * S ** 2).sum()), float((PT * ST ** 2).sum()), float((P * (S - ST[inv]) ** 2).sum()), len(keys), len(cnts)


def check_neurons():
    head("7. Three neurons: the exponential family (8.32) and what is lost by dropping the third-order correlation (end of section 8.2)")
    xi0 = 0.0
    p = nr_p(xi0)[0]
    eta = p @ NR_F; G = (NR_F * p[:, None]).T @ NR_F - np.outer(eta, eta)
    gX = G[6, 6]
    GRR = G[:6, :6]
    schur = G[6, 6] - G[6, :6] @ np.linalg.solve(GRR, G[:6, 6])
    # regression route to the same number: residual variance of x1x2x3 regressed on 1, x_i, x_i x_j with weights p
    A = np.concatenate([np.ones((8, 1)), NR_F[:, :6]], 1)
    sw = np.sqrt(p)
    beta = np.linalg.lstsq(A * sw[:, None], NR_F[:, 6] * sw, rcond=None)[0]
    resid = NR_F[:, 6] - A @ beta
    gT = gX - schur
    print(f"   Model: three binary neurons, p(x; xi) = exp(theta.X - psi) with X = (x_i, x_i x_j, x_1 x_2 x_3), theta = (-1, -1, -1, 0, 0, 0, xi): independent sparse neurons (firing probability {p @ NR_X[:, 0]:.4f}) plus a third-order interaction xi; true value xi = 0.")
    print(f"   The data manifold D of an experimenter who keeps rates and pairwise correlations but loses the third-order moment is the m-flat line q + s*parity, parity(x) = (-1)^(x_1+x_2+x_3) (it has zero first and second moments: largest {np.abs(NR_F[:, :6].T @ NR_D).max():.1e}).")
    print(f"   Full data: g^X = Var(x1 x2 x3) = {gX:.6f}. Retained statistics (asymptotically, a Gaussian vector with covariance G_RR/N): g^T = d eta_R^T G_RR^-1 d eta_R = {gT:.6f}; loss = {schur:.6f} = {schur / gX:.4f} of g^X.")
    print(f"   In the mixed coordinates (eta_R, theta_123) the Fisher metric is block diagonal and the lost information is (d theta_123/d xi)^2 times the Schur complement G_LL - G_LR G_RR^-1 G_RL = {schur:.6f}"
          f" (here d theta_123/d xi = 1); that is the residual variance of x1 x2 x3 regressed on the other six statistics, {(p * resid ** 2).sum():.6f} by weighted least squares; R^2 = {1 - schur / gX:.4f}")
    print("   Exact finite-N check of (8.30) with T = the six retained counts, by enumerating all count vectors (N = 4, 8, 12):")
    print(f"      {'N':>3s} {'count vectors':>14s} {'values of T':>12s} {'g^X':>10s} {'N g^X_1':>10s} {'g^T':>10s} {'g^(X|T)':>10s} {'|g^X - g^T - g^(X|T)|':>22s} {'g^(X|T)/(N loss)':>17s}")
    ex = []
    for N in (4, 8, 12):
        gXf, gTf, gCf, nk, nc = nr_exact_info(N, xi0)
        ex.append((N, gXf, gTf, gCf))
        print(f"      {N:>3d} {nc:>14d} {nk:>12d} {gXf:>10.6f} {N * gX:>10.6f} {gTf:>10.6f} {gCf:>10.6f} {abs(gXf - gTf - gCf):>22.1e} {gCf / (N * schur):>17.3f}")
    print("      (8.30) is exact at every N; the loss per observation approaches the asymptotic value slowly, because at small N the six counts nearly determine the whole table (the ratio in the last column tends to 1)")
    STORE["nr_exact"] = ex
    # em for the reduced data: the population point
    qhat = p[None, :].copy()
    xi = np.array([1.0]); traj = [1.0]
    for _ in range(40):
        q = nr_eproj(qhat, nr_p(xi))
        xi = nr_mstep(q @ NR_F[:, 6], xi); traj.append(float(xi[0]))
    err = np.abs(np.array(traj) - xi0)
    print(f"   The estimator that minimises KL[D : M] is found by the em algorithm (e-projection onto D, then m-projection onto M). Run on the population point eta_R(xi = 0), from xi_0 = 1, the error contracts by"
          f" {err[5] / err[4]:.6f}, {err[10] / err[9]:.6f}, {err[20] / err[19]:.6f} per step, and the loss fraction is {schur / gX:.6f}: the em rate is the fraction of information lost by the reduction (section 4 in a second setting)")
    STORE["nr_contraction"] = (err[20] / err[19], schur / gX)
    # Monte Carlo
    N, R = 20000, 4000
    rng = np.random.default_rng(123)
    cnt = rng.multinomial(N, p, size=R); qh = cnt / N
    xi_full = nr_mstep(qh[:, 7], np.zeros(R))
    xi_em = np.full(R, 0.5)
    for _ in range(100):
        q = nr_eproj(qh, nr_p(xi_em))
        new = nr_mstep(q @ NR_F[:, 6], xi_em)
        done = np.abs(new - xi_em).max() < 1e-12
        xi_em = new
        if done:
            break
    # naive reduced estimator: unweighted least squares on the retained moments (Gauss-Newton)
    etaR = qh @ NR_F[:, :6]
    xi_ls = np.zeros(R)
    for _ in range(50):
        pp = nr_p(xi_ls); mu_R = pp @ NR_F[:, :6]
        d = ((pp * (NR_F[:, 6][None, :] - (pp @ NR_F[:, 6])[:, None])) @ NR_F[:, :6])      # d eta_R / d xi = Cov(x_R, x1x2x3) at the current xi
        xi_ls = xi_ls + ((etaR - mu_R) * d).sum(1) / (d * d).sum(1)
    v_full, v_em, v_ls = N * xi_full.var(), N * xi_em.var(), N * xi_ls.var()
    se = math.sqrt(2.0 / R)
    print(f"   Monte Carlo, N = {N} patterns, {R} data sets (exact multinomial counts), true xi = 0: N * Var of the estimator of xi:")
    print(f"      full-data MLE (a function of the third-order moment too)          {v_full:8.3f}   against 1/g^X = {1 / gX:.3f}")
    print(f"      min KL[D : M] from the retained moments only (em)                 {v_em:8.3f}   against 1/g^T = {1 / gT:.3f}   (relative s.e. of a variance: {se:.3f})")
    print(f"      unweighted least squares on the same retained moments              {v_ls:8.3f}   (not optimal: the weight G_RR^-1 matters)")
    print(f"      the ratio {v_em / v_full:.3f} of the first two variances against g^X/g^T = {gX / gT:.3f}; means {xi_full.mean():+.4f}, {xi_em.mean():+.4f}, {xi_ls.mean():+.4f} (standard error {math.sqrt(v_full / N / R):.4f}, standard deviation of one estimate {math.sqrt(v_full / N):.4f}: a small O(1/N) bias, which does not touch the comparison)")
    STORE["nr_mc"] = (v_full, v_em, v_ls, 1 / gX, 1 / gT)


# ------------------------------------------------------------------ 7. misspecified models

RNG = np.random.default_rng(20261003)
GH80_X, GH80_W = np.polynomial.hermite_e.hermegauss(80)
GH80_W = GH80_W / SQ2PI


def Ep_gauss(f, mean, sd):
    """E[f(x)], x ~ N(mean, sd^2), Gauss-Hermite with 80 nodes."""
    return float((GH80_W * f(mean + sd * GH80_X)).sum())


def ab_samples(N, R, mu, s, rng=None):
    """(xbar, mean of x^2) of R samples of size N from N(mu, s^2): the exact joint law (xbar normal, the sum of squares chi-square)."""
    rng = rng or RNG
    a = rng.normal(mu, s / math.sqrt(N), R)
    chi = rng.chisquare(N - 1, R)
    return a, a * a + s * s * chi / N


# the misspecified models q(x; v) in the Gaussian family with statistics (x, x^2); the truth is N(u, u^2)
class QModel:
    """A curved family q(x; v) = N(m(v), s2(v)) inside the Gaussian exponential family: natural parameters theta_q(v) = (m/s2, -1/(2 s2)), expectation parameters eta_q(v) = (m, m^2 + s2)."""

    def __init__(self, name, m, s2, dm, ds2, d2m=None, d2s2=None):
        self.name, self.m, self.s2, self.dm, self.ds2 = name, m, s2, dm, ds2

    def logq(self, x, v):
        return -0.5 * np.log(2 * np.pi * self.s2(v)) - (x - self.m(v)) ** 2 / (2 * self.s2(v))

    def score(self, x, v, e=1e-5):
        return (self.logq(x, v + e) - self.logq(x, v - e)) / (2 * e)

    def d2log(self, x, v, e=1e-4):
        return (self.logq(x, v + e) - 2 * self.logq(x, v) + self.logq(x, v - e)) / e ** 2

    def theta_d(self, v, e=1e-6):
        th = lambda t: np.array([self.m(t) / self.s2(t), -1 / (2 * self.s2(t))])
        return (th(v + e) - th(v - e)) / (2 * e)

    def eta(self, v):
        return np.array([self.m(v), self.m(v) ** 2 + self.s2(v)])


def q_models(c=1.5):
    return {
        "q1": QModel("N(v, 1): mean only", lambda v: v, lambda v: 1.0 + 0 * v, None, None),
        "q4": QModel("N(0, v^2): scale only", lambda v: 0 * v, lambda v: v * v, None, None),
        "q5": QModel(f"N(v, {c} v^2): wrong coefficient of variation", lambda v: v, lambda v: c * v * v, None, None),
    }


def kl_true_q(u, Q, v):
    """KL[N(u, u^2) : q(. ; v)] in closed form."""
    s2 = Q.s2(v); m = Q.m(v)
    return 0.5 * np.log(s2 / u ** 2) + (u ** 2 + (u - m) ** 2) / (2 * s2) - 0.5


def pseudo_true(u, Q, lo=1e-3, hi=12.0):
    """The root of E_{N(u,u^2)}[d log q(x; v)/dv] = 0 (8.37), by bisection on a quadrature."""
    f = lambda v: Ep_gauss(lambda x: Q.score(x, v), u, u)
    a, b = lo, hi
    fa = f(a)
    for _ in range(100):
        mid = 0.5 * (a + b); fm = f(mid)
        if (fm > 0) == (fa > 0):
            a, fa = mid, fm
        else:
            b = mid
    return 0.5 * (a + b)


def golden(f, a, b, it=120):
    g = (math.sqrt(5) - 1) / 2
    x1 = b - g * (b - a); x2 = a + g * (b - a); f1 = f(x1); f2 = f(x2)
    for _ in range(it):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1; x1 = b - g * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2; x2 = a + g * (b - a); f2 = f(x2)
    return 0.5 * (a + b)


def check_misspecified():
    head("8. Estimation with a misspecified model (section 8.3): Theorems 8.3 and 8.4 on the curved family N(u, u^2) of Chapter 7")
    # sign in (8.39) and (8.41)
    xs = np.linspace(-2, 3, 11)
    p_ = lambda x: phi(x - 1.0); q_ = lambda x: phi(x - 0.5)
    t = 1e-6
    r_ = lambda x, tt: (1 - tt) * q_(x) + tt * p_(x)
    fd = (np.log(r_(xs, t)) - np.log(r_(xs, -t))) / (2 * t)
    print(f"   Proof of Theorem 8.3: for r(t) = (1-t) q + t p the log-derivative at t = 0 is (p - q)/q, not the printed (8.39) {{q - p}}/q: at 11 points the finite difference of log r agrees with (p - q)/q to {np.abs(fd - (p_(xs) - q_(xs)) / q_(xs)).max():.1e}"
          f" and is the negative of the printed one;"
          f" with q = N(0.5, 1), p = N(1, 1): <r', l'_q>_q = integral of (p - q) d log q/dv = {Ep_gauss(lambda x: x - 0.5, 1.0, 1.0):+.4f} = +E_p[d log q/dv], so (8.41) should end with +integral p d_u log q (a sign slip that does not affect the conclusion that the pairing vanishes iff (8.37) holds)")
    models = q_models(1.5)
    # --- consistency: Theorem 8.3
    print("   Theorem 8.3. Pseudo-true value v*(u) of the q-MLE (the limit of the q-MLE when the data come from N(u, u^2)), three ways: the root of E_p[d log q/dv] = 0 (8.37); the minimiser of KL[p : q(v)] over v (the KL-projection of the truth onto M_q);"
          " the average of the q-MLE over 2e6 exact data sets of N = 2000 (xbar, mean x^2 sampled from their exact law):")
    kappa = {"q1": 1.0, "q4": math.sqrt(2.0), "q5": (-1 + math.sqrt(1 + 8 * 1.5)) / (2 * 1.5)}
    est = {"q1": lambda a, b: a, "q4": lambda a, b: np.sqrt(b), "q5": lambda a, b: (-a + np.sqrt(a * a + 4 * 1.5 * b)) / (2 * 1.5)}
    print(f"      {'model':46s} {'u':>4s} {'root of (8.37)':>15s} {'argmin KL[p:q]':>15s} {'q-MLE, Monte Carlo':>22s} {'kappa u':>9s}")
    mc_means = {}
    for key in ("q1", "q4", "q5"):
        Q = models[key]
        for u in (0.6, 1.0, 1.4):
            v1 = pseudo_true(u, Q)
            v2 = golden(lambda v: float(kl_true_q(u, Q, v)), 1e-3, 12.0)
            N, Rr = 2000, 2_000_000
            a, b = ab_samples(N, Rr, u, u)
            vh = est[key](a, b)
            m_, se_ = float(vh.mean()), float(vh.std() / math.sqrt(Rr))
            mc_means[(key, u)] = (m_, se_)
            print(f"      {Q.name:46s} {u:>4.1f} {v1:>15.6f} {v2:>15.6f} {m_:>14.5f} +- {se_:.5f} {kappa[key] * u:>9.5f}")
    print(f"      so q1 (N(v, 1)) is consistent (v* = u); the scale-only model has v* = sqrt(2) u; the wrong coefficient of variation has v* = kappa u with kappa = (sqrt(1 + 8c) - 1)/(2c) = {kappa['q5']:.4f} for c = 1.5. In every case v*(u) = kappa u,"
          " the point where the leaf A_q(v) meets M is u = f(v) = v/kappa, and the remark after Theorem 8.4 is right: relabelling the points of M_q by f makes the estimator consistent (below, the relabelled estimator f(q-MLE) has mean u)")
    print("   (the book's 'f^-1(u)' as the new parameter is ambiguous in direction: the labelling that works is q_new(x; w) = q(x; f^-1(w)), i.e. estimate w = f(v_hat))")
    # --- efficiency: Theorem 8.4, four routes
    print("   Theorem 8.4. Efficiency of the relabelled q-MLE under N(u, u^2) at u = 1 (the Cramer-Rao bound is 1/g_uu = 1/3), four ways:")
    print("      (a) geometry: leaf direction d orthogonal to theta_q'(v*) in the pairing of the eta-plane, g_uv = <eta_p', d>, g_vv = <d, d> in the Fisher metric of S at the true point, loss = g_uv^2/g_vv (8.42), efficiency = 1 - loss/g_uu;"
          " (b) sandwich: A = -E_p[d^2 log q/dv^2], B = E_p[(d log q/dv)^2], variance f'(v*)^2 B/A^2, efficiency = (1/g_uu) / variance;"
          " (c) the squared correlation of the true score and the q-score under p; (d) Monte Carlo, N = 4000, 2e6 data sets")
    print(f"      {'model':46s} {'(a) geometry':>13s} {'(b) sandwich':>13s} {'(c) corr^2':>11s} {'1/g^_uu (a)':>12s} {'N Var (d)':>16s} {'mean of f(q-MLE)':>24s}")
    u = 1.0
    Gm = np.array([[u * u, 2 * u ** 3], [2 * u ** 3, 6 * u ** 4]]); Gi = np.linalg.inv(Gm)
    t_vec = np.array([1.0, 4 * u])
    g_uu = t_vec @ Gi @ t_vec
    rows = {}
    for key in ("q1", "q4", "q5"):
        Q = models[key]
        v = pseudo_true(u, Q)
        thq = Q.theta_d(v)
        d = np.array([-thq[1], thq[0]])                                  # direction with theta_q' . d = 0
        g_uv = t_vec @ Gi @ d; g_vv = d @ Gi @ d
        loss = g_uv ** 2 / g_vv
        eff_a = 1 - loss / g_uu
        A = -Ep_gauss(lambda x: Q.d2log(x, v), u, u); B = Ep_gauss(lambda x: Q.score(x, v) ** 2, u, u)
        dv_du = v / u                                                    # v*(u) = kappa u
        var_b = (1 / dv_du) ** 2 * B / A ** 2
        eff_b = (1 / g_uu) / var_b
        # correlation of scores: true score of N(u, u^2) is d/du log p
        sp = lambda x: (-1 / u + (x - u) / u ** 2 + (x - u) ** 2 / u ** 3)
        cov = Ep_gauss(lambda x: sp(x) * Q.score(x, v), u, u)
        vp = Ep_gauss(lambda x: sp(x) ** 2, u, u); vq = Ep_gauss(lambda x: Q.score(x, v) ** 2, u, u)
        eff_c = cov ** 2 / (vp * vq)
        N, Rr = 4000, 2_000_000
        a, b = ab_samples(N, Rr, u, u)
        uh = est[key](a, b) / kappa[key]
        var_d = N * float(uh.var()); se_d = var_d * math.sqrt(2.0 / Rr)
        mean_d = float(uh.mean()); se_m = float(uh.std() / math.sqrt(Rr))
        rows[key] = (eff_a, eff_b, eff_c, var_d, loss)
        print(f"      {Q.name:46s} {eff_a:>13.5f} {eff_b:>13.5f} {eff_c:>11.5f} {1 / (g_uu - loss):>12.5f} {var_d:>10.5f} +- {se_d:.5f} {mean_d:>12.5f} +- {se_m:.5f}")
    STORE["mis_rows"] = rows
    print(f"      (the Cramer-Rao value 1/g_uu is {1 / g_uu:.5f}; the relabelled estimators f(q-MLE) = v_hat/kappa are consistent: mean 1 at u = 1, up to the O(1/N) bias; the unit tangent of M at u = 1 is (1, 4), g_uu = {g_uu:.4f}; the three models are the leaf families of Chapter 7: vertical leaves = the sample mean (efficiency 1/3), horizontal leaves = sqrt(m_2/2) (8/9); the sloped leaves of N(v, c v^2) are nearly orthogonal to M, hence nearly efficient)")
    # the geometry of the q-MLE: m-projection along straight leaves, orthogonal to M_q; the pairing form of (8.37); Pythagoras and the Hessian
    Q = models["q5"]; c5 = 1.5
    a, b = ab_samples(20, 1500, 1.0, 1.0, np.random.default_rng(4))
    s2 = b - a * a
    est5 = (-a + np.sqrt(a * a + 4 * c5 * b)) / (2 * c5)
    lo = np.full(len(a), 0.05); hi = np.full(len(a), 4.0); g_ = (math.sqrt(5) - 1) / 2
    klf = lambda v: 0.5 * np.log(c5 * v * v / s2) + (s2 + (a - v) ** 2) / (2 * c5 * v * v) - 0.5          # KL[N(a, s2) : N(v, c v^2)]
    x1 = hi - g_ * (hi - lo); x2 = lo + g_ * (hi - lo); f1 = klf(x1); f2 = klf(x2)
    for _ in range(100):
        m_ = f1 < f2
        hi = np.where(m_, x2, hi); lo = np.where(m_, lo, x1)
        x1 = hi - g_ * (hi - lo); x2 = lo + g_ * (hi - lo); f1 = klf(x1); f2 = klf(x2)
    vkl = (lo + hi) / 2
    thq = np.stack([-1 / (c5 * est5 ** 2), 1 / (c5 * est5 ** 3)], 1)                                         # theta_q'(v) for N(v, c v^2)
    etaq = np.stack([est5, (1 + c5) * est5 ** 2], 1)
    resid = np.sum(thq * (np.stack([a, b], 1) - etaq), 1)
    print(f"   The q-MLE is the m-projection of the observed point onto M_q: for 1500 observed points (N = 20, true u = 1) the minimiser over v of KL[p_etabar : q(v)] for q = N(v, 1.5 v^2), found by golden-section search, agrees with the closed form"
          f" (-xbar + sqrt(xbar^2 + 4 c m_2))/(2c) to {np.abs(vkl - est5).max():.1e}, and the leaf equation theta_q'(v_hat).(etabar - eta_q(v_hat)) = 0 holds to {np.abs(resid).max():.1e}: the leaf through q(v) is the straight (m-flat) line orthogonal to theta_q'(v)")
    v5 = pseudo_true(1.0, Q)
    th5 = Q.theta_d(v5); d5 = np.array([-th5[1], th5[0]])
    etaqp = np.array([1.0, 2 * (1 + c5) * v5])                                                                # d eta_q / dv
    Gq = np.array([[Q.s2(v5), 2 * Q.m(v5) * Q.s2(v5)], [2 * Q.m(v5) * Q.s2(v5), 2 * Q.s2(v5) * (Q.s2(v5) + 2 * Q.m(v5) ** 2)]])     # Fisher metric of S at q(v*)
    print(f"      the leaf is orthogonal to M_q at its foot q(v*): <eta_q', d> = eta_q'^T G_q^-1 d = {abs(etaqp @ np.linalg.solve(Gq, d5)):.1e} (Fisher metric of S at q(v*)); and (8.37) in its pairing form E_p[d log q/dv] = theta_q'(v).(eta_p - eta_q(v)) holds for"
          f" 12 (u, v) pairs to {max(abs(Ep_gauss(lambda x: Q.score(x, v_), u_, u_) - Q.theta_d(v_) @ (np.array([u_, 2 * u_ * u_]) - Q.eta(v_))) for u_ in (0.6, 1.0, 1.4) for v_ in (0.5, 0.9, 1.3, 2.0)):.1e}")
    # Pythagoras for e-flat M_q and the Hessian
    print("   Geometry of the misspecified likelihood: when M_q is e-flat (q1 and q4 are straight lines in theta) the foot q* = q(v*) of the m-projection of p satisfies the exact Pythagorean relation KL[p : q(v)] = KL[p : q*] + KL[q* : q(v)] for every v,")
    print("   so the expected log-likelihood is a quadratic bowl whose curvature A is the Fisher information of the model at the foot; for the curved M_q = N(v, c v^2), A = g_q(v*) - theta_q''.(eta_p - eta_q(v*)): the e-curvature of M_q times the displacement of p from the foot:")
    def kl_gauss(m1, s1, m2, s2_):
        return 0.5 * np.log(s2_ / s1) + (s1 + (m1 - m2) ** 2) / (2 * s2_) - 0.5
    for key in ("q1", "q4", "q5"):
        Qk = models[key]; u_ = 1.0
        v_ = pseudo_true(u_, Qk)
        worst = 0.0
        for vv in (0.4, 0.8, 1.3, 2.0, 3.0):
            lhs = kl_gauss(u_, u_ ** 2, Qk.m(vv), Qk.s2(vv))
            rhs = kl_gauss(u_, u_ ** 2, Qk.m(v_), Qk.s2(v_)) + kl_gauss(Qk.m(v_), Qk.s2(v_), Qk.m(vv), Qk.s2(vv))
            worst = max(worst, abs(lhs - rhs))
        A_ = -Ep_gauss(lambda x: Qk.d2log(x, v_), u_, u_)
        gq = float(Qk.theta_d(v_) @ np.array([[Qk.s2(v_), 2 * Qk.m(v_) * Qk.s2(v_)], [2 * Qk.m(v_) * Qk.s2(v_), 2 * Qk.s2(v_) * (Qk.s2(v_) + 2 * Qk.m(v_) ** 2)]]) @ Qk.theta_d(v_))
        e = 1e-4
        th2 = (Qk.theta_d(v_ + e) - Qk.theta_d(v_ - e)) / (2 * e)
        corr = -th2 @ (np.array([u_, 2 * u_ * u_]) - Qk.eta(v_))
        print(f"      {Qk.name:46s} Pythagoras residual {worst:.1e}; A = {A_:.5f}, g_q(v*) = {gq:.5f}, A - g_q = {A_ - gq:+.5f}, -theta_q''.(eta_p - eta_q) = {corr:+.5f}")


# ------------------------------------------------------------------ 7b. the neural field: the correlated truth and the decoder that ignores correlation

GH12_X, GH12_W = np.polynomial.hermite_e.hermegauss(12)
GH12_W = GH12_W / SQ2PI


def gauss2_expect(f, mean, V):
    """E[f(x)] for x ~ N(mean, V) in two dimensions, tensor Gauss-Hermite (exact for polynomials of degree <= 23)."""
    L = np.linalg.cholesky(V)
    Z = np.array(list(itertools.product(GH12_X, GH12_X))); W = np.array([a * b for a, b in itertools.product(GH12_W, GH12_W)])
    Xs = mean[None, :] + Z @ L.T
    return (W[:, None] * f(Xs)).sum(0)


def stat5(Xs):
    return np.stack([Xs[:, 0], Xs[:, 1], Xs[:, 0] ** 2, Xs[:, 0] * Xs[:, 1], Xs[:, 1] ** 2], 1)


def efficiency_formula(rp, V):
    """Efficiency of the decoder that ignores the covariance: |r'|^4 / ((r'^T V r')(r'^T V^-1 r'))."""
    return float((rp @ rp) ** 2 / ((rp @ V @ rp) * (rp @ np.linalg.solve(V, rp))))


def two_neuron_geometry(u, rho):
    """Theorem 8.4 for p = N(r(u), V), q = N(r(u), I), r(u) = (cos u, sin u), inside the five-dimensional Gaussian family with statistics (x1, x2, x1^2, x1 x2, x2^2)."""
    r = np.array([math.cos(u), math.sin(u)]); rp = np.array([-math.sin(u), math.cos(u)])
    V = np.array([[1.0, rho], [rho, 1.0]]); Vi = np.linalg.inv(V)
    eta = gauss2_expect(stat5, r, V)
    G = gauss2_expect(lambda Xs: np.einsum("ni,nj->nij", stat5(Xs), stat5(Xs)).reshape(len(Xs), -1), r, V).reshape(5, 5) - np.outer(eta, eta)
    Gi = np.linalg.inv(G)
    thp = np.concatenate([Vi @ rp, np.zeros(3)])               # d theta_p/du: only the mean block moves (V does not depend on u)
    t = G @ thp                                                 # d eta/du
    psi = np.concatenate([rp, np.zeros(3)])                     # d theta_q/du
    _, _, Vt = np.linalg.svd(psi[None, :]); Bk = Vt[1:].T         # a basis of the leaf's tangent space {d: psi.d = 0}
    Bk = Bk @ np.random.default_rng(0).normal(size=(4, 4))        # an arbitrary basis of it
    g_uu = t @ Gi @ t; g_uk = t @ Gi @ Bk; g_kl = Bk.T @ Gi @ Bk
    loss = g_uk @ np.linalg.solve(g_kl, g_uk)
    Bfull = np.concatenate([t[None, :], Bk.T], 0)
    gin = np.linalg.inv(Bfull @ Gi @ Bfull.T)
    wrong = g_uk @ gin[1:, 1:] @ g_uk
    sp = lambda Xs: (Xs - r) @ (Vi @ rp); sq = lambda Xs: (Xs - r) @ rp
    cov = gauss2_expect(lambda Xs: (sp(Xs) * sq(Xs))[:, None], r, V)[0]
    vp = gauss2_expect(lambda Xs: (sp(Xs) ** 2)[:, None], r, V)[0]; vq = gauss2_expect(lambda Xs: (sq(Xs) ** 2)[:, None], r, V)[0]
    return dict(g_uu=g_uu, loss=loss, eff_geom=1 - loss / g_uu, eff_formula=efficiency_formula(rp, V), eff_corr=cov ** 2 / (vp * vq), wrong=wrong, g_uu_formula=float(rp @ Vi @ rp),
                resid=float(np.abs(t - np.concatenate([rp, [2 * rp[0] * r[0], rp[0] * r[1] + rp[1] * r[0], 2 * rp[1] * r[1]]])).max()))


def ring_field(n, a, u=0.5):
    """n neurons on a ring with preferred positions j/n and Gaussian tuning curves of width a; returns the derivative r'(u) at the stimulus u."""
    pos = np.arange(n) / n
    d = ((pos - u + 0.5) % 1.0) - 0.5
    r = np.exp(-d ** 2 / (2 * a * a))
    return r * d / (a * a)


def ring_cov(n, rho, b=None, kind="uniform"):
    pos = np.arange(n) / n
    if kind == "uniform":
        K = np.ones((n, n))
    else:                                                   # the wrapped Gaussian: a valid covariance on the circle for every b
        dd = np.abs(pos[:, None] - pos[None, :]); dd = np.minimum(dd, 1 - dd)
        K = sum(np.exp(-(dd + k) ** 2 / (2 * b * b)) for k in range(-3, 4))
        K = K / K[0, 0]
    return (1 - rho) * np.eye(n) + rho * K



GH10_X, GH10_W = np.polynomial.hermite_e.hermegauss(10)
GH10_W = GH10_W / SQ2PI


def gauss3_expect(f, mean, V):
    """E[f(x)] for x ~ N(mean, V) in three dimensions, tensor Gauss-Hermite (exact for polynomials of degree <= 19)."""
    L = np.linalg.cholesky(V)
    Z = np.array(list(itertools.product(GH10_X, repeat=3))); W = np.array([np.prod(c) for c in itertools.product(GH10_W, repeat=3)])
    Xs = mean[None, :] + Z @ L.T
    return (W[:, None] * f(Xs)).sum(0)


def three_neuron_two_parameters():
    """Theorem 8.4 with a two-dimensional M: truth N(r(u), V), r(u) in R^3, u in R^2, model N(r(u), I), in the nine-dimensional Gaussian family (leaf of dimension 7).
    The loss matrix from (8.42) (leaf basis, inverse of the block) against g - A B^-1 A from the sandwich A = J^T J, B = J^T V J."""
    iu = [(i, j) for i in range(3) for j in range(i, 3)]
    stat9 = lambda Xs: np.concatenate([Xs, np.stack([Xs[:, i] * Xs[:, j] for i, j in iu], 1)], 1)
    u = np.array([0.4, 0.7])
    r = np.array([u[0] + 0.3 * u[1] ** 2, math.sin(u[1]) + 0.5 * u[0], u[0] * u[1] - 0.2 * u[1]])
    J = np.array([[1.0, 0.6 * u[1]], [0.5, math.cos(u[1])], [u[1], u[0] - 0.2]])
    V = np.array([[1.0, 0.8, 0.1], [0.8, 1.5, 0.9], [0.1, 0.9, 2.0]]); Vi = np.linalg.inv(V)
    eta = gauss3_expect(stat9, r, V)
    G = gauss3_expect(lambda Xs: np.einsum("ni,nj->nij", stat9(Xs), stat9(Xs)).reshape(len(Xs), -1), r, V).reshape(9, 9) - np.outer(eta, eta)
    Gi = np.linalg.inv(G)
    t = [G @ np.concatenate([Vi @ J[:, a], np.zeros(6)]) for a in range(2)]                    # d eta / d u_a
    psi = np.stack([np.concatenate([J[:, a], np.zeros(6)]) for a in range(2)])                  # d theta_q / d u_a
    _, _, Vt = np.linalg.svd(psi); Bk = Vt[2:].T @ np.random.default_rng(1).normal(size=(7, 7))   # an arbitrary basis of the leaf {d : psi_a . d = 0}
    gab = np.array([[t[a] @ Gi @ t[b] for b in range(2)] for a in range(2)])
    gak = np.array([t[a] @ Gi @ Bk for a in range(2)]); gkl = Bk.T @ Gi @ Bk
    loss = gak @ np.linalg.solve(gkl, gak.T)
    A = J.T @ J; B = J.T @ V @ J
    loss2 = gab - A @ np.linalg.solve(B, A)
    gin = np.linalg.inv(np.concatenate([np.array(t), Bk.T], 0) @ Gi @ np.concatenate([np.array(t), Bk.T], 0).T)
    wrong = gak @ gin[2:, 2:] @ gak.T
    fr = np.sort(np.linalg.eigvals(np.linalg.solve(gab, loss)).real)
    return dict(gab=gab, loss=loss, loss2=loss2, diff=float(np.abs(loss - loss2).max()), wrong=wrong, fractions=fr, gdiff=float(np.abs(gab - J.T @ Vi @ J).max()))


def check_neural_field():
    head("9. The neural field of (8.34)-(8.36): the truth N(r(u), V), the model that ignores correlation N(r(u), I)")
    print("   Both are Gaussian with the same mean r(u); V does not depend on u. Then E_p[d log q/du] = r'.(E_p x - r) = 0 for every V: the q-MLE (least squares, argmin |xbar - r(u)|^2) is consistent (Theorem 8.3),")
    print("   the q-score is r'.(x - r), the true score is r'^T V^-1 (x - r), their covariance under p is |r'|^2, the variances are r'^T V r' and r'^T V^-1 r', so by the correlation route and by the sandwich (A = |r'|^2, B = r'^T V r')")
    print("      efficiency = |r'|^4 / ((r'^T V r') (r'^T V^-1 r')) = 1 - (loss of Fisher information)/(Fisher information r'^T V^-1 r'), which is 1 exactly when V r' is parallel to r' (Cauchy-Schwarz) and smaller otherwise.")
    print("   Two neurons, r(u) = (cos u, sin u), V = [[1, rho], [rho, 1]], geometry in the five-dimensional Gaussian family S (leaf = a 4-dimensional m-flat set; g^(kappa lambda) the inverse of the kappa-lambda block of g in an arbitrary basis of the leaf):")
    hdr = "|G theta_p' - d eta/du|"
    print(f"      {'rho':>4s} {'u':>6s} {'g_uu':>8s} {'loss (8.42)':>12s} {'eff (geometry)':>15s} {'eff (formula)':>14s} {'eff (corr^2)':>13s} {'loss, block of full inverse':>28s} {hdr:>24s}")
    rows = []
    for rho in (0.6,):
        for u in (0.0, math.pi / 4, 0.7):
            o = two_neuron_geometry(u, rho)
            rows.append((rho, u, o))
            print(f"      {rho:>4.1f} {u:>6.3f} {o['g_uu']:>8.4f} {o['loss']:>12.6f} {o['eff_geom']:>15.6f} {o['eff_formula']:>14.6f} {o['eff_corr']:>13.6f} {o['wrong']:>28.6f} {o['resid']:>24.1e}")
    STORE["two_neuron"] = rows
    print(f"      at u = 0 the tuning direction r' = (0, 1) is not an eigenvector of V (whose eigenvectors are (1, 1), (1, -1)) and the efficiency is 1 - rho^2 = {1 - 0.36:.2f}; at u = pi/4, r' is proportional to (-1, 1), an eigenvector, and there is no loss;"
          " reading g^(kappa lambda) of (8.42) as the kappa-lambda block of the full inverse metric gives a different number (last column but one), as in Chapter 7")
    # ring
    bvals = (0.02, 0.05, 0.10, 0.20, 0.50)
    print("   n neurons on a ring, tuning width a = 0.1 at stimulus u = 0.5 (r' = derivative of the Gaussian bump), correlation (1 - rho) I + rho K with rho = 0.5; efficiency of the decoder that ignores correlation:")
    print(f"      {'n':>5s} {'uniform K':>10s} " + " ".join(f"{'b = %.2f' % b_:>9s}" for b_ in bvals) + "   (K_ij = wrapped Gaussian of the ring distance, range b)")
    tab = {}
    for n in (25, 50, 100, 200, 400):
        rp = ring_field(n, 0.1)
        row = [efficiency_formula(rp, ring_cov(n, 0.5, kind="uniform"))] + [efficiency_formula(rp, ring_cov(n, 0.5, b_, "gauss")) for b_ in bvals]
        tab[n] = row
        print(f"      {n:>5d} {row[0]:>10.4f} " + " ".join(f"{v:>9.4f}" for v in row[1:]))
    STORE["ring_table"] = tab
    rp = ring_field(100, 0.1)
    pos = np.arange(100) / 100
    rpc = 2 * math.pi * np.sin(2 * math.pi * (pos - 0.5))
    effs = [efficiency_formula(rpc, ring_cov(100, 0.5, b_, "gauss")) for b_ in (0.02, 0.05, 0.1, 0.2)]
    bmin = min(np.linspace(0.05, 1.0, 96), key=lambda b_: efficiency_formula(rp, ring_cov(100, 0.5, b_, "gauss")))
    print(f"      uniform correlation: r' sums to {rp.sum():.1e} (the total response is flat in u), so r' is an eigenvector and the loss is {1 - tab[100][0]:.1e}; short-range correlation (b well below the tuning width) loses little;"
          f" correlation range comparable to the tuning width loses a lot, and more as n grows (b = 0.2: {tab[25][4]:.3f} at n = 25, {tab[400][4]:.3f} at n = 400); for n = 100 the efficiency is smallest, {efficiency_formula(rp, ring_cov(100, 0.5, bmin, 'gauss')):.3f}, at b = {bmin:.2f},"
          f" and returns to 1 as the correlation becomes uniform (b = 0.5: {tab[100][5]:.3f}). Cosine tuning: r' is a single Fourier mode, an eigenvector of every translation-invariant V: loss {1 - min(effs):.1e} for b = 0.02, 0.05, 0.1, 0.2")
    STORE["ring_cos"] = effs
    o3 = three_neuron_two_parameters()
    print(f"   Two parameters (the index structure of (8.42) as a matrix): three neurons, u in R^2, r(u) = (u_1 + 0.3 u_2^2, sin u_2 + 0.5 u_1, u_1 u_2 - 0.2 u_2) at u = (0.4, 0.7), V = [[1, 0.8, 0.1], [0.8, 1.5, 0.9], [0.1, 0.9, 2]], nine-dimensional Gaussian family, leaf of dimension 7:")
    print(f"      g_ab = J^T V^-1 J = {np.round(o3['gab'], 5).tolist()} (error {o3['gdiff']:.1e});  loss from (8.42), g_(a kappa) (g_(kappa lambda))^-1 g_(b lambda) = {np.round(o3['loss'], 5).tolist()};"
          f"  g - A B^-1 A from the sandwich A = J^T J, B = J^T V J = {np.round(o3['loss2'], 5).tolist()}; largest difference {o3['diff']:.1e}")
    print(f"      fractions of the information lost, eigenvalues of g^-1 loss: {np.round(o3['fractions'], 5).tolist()} (only one direction is affected here); reading g^(kappa lambda) as the block of the full inverse gives the loss {np.round(o3['wrong'], 5).tolist()}")
    print("   So 'no loss' cannot hold for a general correlation V: the decoder that ignores V is efficient exactly when V r' is parallel to r', and otherwise loses a fraction set by the spectrum of V against that of r'")


# ------------------------------------------------------------------ figures: themed SVG helpers (the same as in the Chapter 7 script)

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


# ------------------------------------------------------------------ the figures

def clipped_runs(P, xs, ys):
    """Split a polyline into runs of points inside the panel (so that clipping does not draw chords across gaps)."""
    runs, cur = [], []
    for x, y in zip(xs, ys):
        ok = P.xr[0] - 1e-9 <= x <= P.xr[1] + 1e-9 and P.yr[0] - 1e-9 <= y <= P.yr[1] + 1e-9 and np.isfinite(x) and np.isfinite(y)
        if ok:
            cur.append((x, y))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs


def line_runs(P, xs, ys, cls):
    for run in clipped_runs(P, xs, ys):
        if len(run) > 1:
            P.b.append(f'<polyline class="{cls}" points="' + " ".join(f"{P.X(x):.1f},{P.Y(y):.1f}" for x, y in run) + '"/>')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fig_staircase(path):
    b = []
    W, H = 800, 440
    y = 1.8
    P1 = Panel(b, 62, 56, 318, 250, (-0.3, 2.3), (0.0, 1.0))
    P1.frame([0, 0.5, 1, 1.5, 2], [0, 0.25, 0.5, 0.75, 1], "m   (a point of M)", "a   (a point of D)", "One observation y = 1.8: the two projections", True)
    A = lambda a: a * np.log(a) + (1 - a) * np.log(1 - a) + 2 * y * y * a * (1 - a)
    ag = np.linspace(1e-4, 1 - 1e-4, 800)
    for L in (0.03, 0.12, 0.35, 0.8, 1.6, 3.2):
        val = 2 * (L - A(ag))
        ok = val >= 0
        mm = np.where(ok, (2 * ag - 1) * y + np.sqrt(np.where(ok, val, 0.0)), np.nan)
        mn = np.where(ok, (2 * ag - 1) * y - np.sqrt(np.where(ok, val, 0.0)), np.nan)
        line_runs(P1, mm, ag, "con s0"); line_runs(P1, mn, ag, "con s0")
    mg = np.linspace(-0.3, 2.3, 300)
    line_runs(P1, mg, sigmoid(2 * y * mg), "ln s1")                   # the E-step: a = sigma(2 m y), the e-projection of the model point m onto D
    line_runs(P1, mg, 0.5 * (1 + mg / y), "ln s2")                    # the M-step: m = (2a - 1) y, the m-projection of the data point a onto M: a straight line
    m, a = 0.3, 0.5
    pts = [(m, a)]
    traj = [m]
    for _ in range(7):
        a = float(sigmoid(2 * y * m)); pts.append((m, a))
        m = (2 * a - 1) * y; pts.append((m, a)); traj.append(m)
    line_runs(P1, [p[0] for p in pts], [p[1] for p in pts], "ln sk")
    m_star = traj[-1]
    P1.dot(m_star, float(sigmoid(2 * y * m_star)), "f4", 4.5)
    P1.dot(0.0, 0.5, "f0", 4.5)
    P1.dot(0.3, 0.5, "f2", 3.5)
    P1.text(2.27, 0.03, "grey: contours of F(a, m)", "sm", "end")
    legend_col(b, 62 + 150, 56 + 160, [("s1", "E-step: a = σ(2my)"), ("s2", "M-step: m = (2a − 1)y"), ("sk", "em from m = 0.3")])
    # right: cobweb for the symmetric two-component model
    P2 = Panel(b, 452, 56, 318, 250, (0, 3), (0, 3))
    P2.frame([0, 1, 2, 3], [0, 1, 2, 3], "m_t", "m_(t+1) = E[y tanh(m_t y)]", "EM map of 0.5 N(−m, 1) + 0.5 N(m, 1)", True)
    P2.line([0, 3], [0, 3], "dash s0")
    mg = np.linspace(0, 3, 301)
    colors = {0.5: ("s1", "f1"), 1.5: ("s3", "f3")}
    for mt, (sc, fc) in colors.items():
        P2.line(mg, [sym_em_pop(v, mt) for v in mg], "ln " + sc)
        cur = 2.6; pts = [(cur, 0.0)]
        for _ in range(14):
            nxt = sym_em_pop(cur, mt); pts += [(cur, nxt), (nxt, nxt)]; cur = nxt
        line_runs(P2, [p[0] for p in pts], [p[1] for p in pts], "thin " + sc)
        P2.dot(mt, mt, fc, 4.0)
    r1, r2 = sym_rate(0.5), sym_rate(1.5)
    legend_col(b, 452 + 12, 56 + 14, [("s1", f"true m = 0.5: slope {r1:.3f} at the fixed point"), ("s3", f"true m = 1.5: slope {r2:.3f}")])
    note(b, 62, 372, ["Left: F(a, m) is the divergence between the point a of D and the point m of M. The E-step moves vertically to",
                      "the curve a = σ(2my), the M-step horizontally to the straight line m = (2a − 1)y; the staircase reaches the",
                      "minimum (orange dot). The grey dot, m = 0, is stationary but not a minimum: EM started exactly there stays.",
                      "Right: the same algorithm on many observations, as a map m → m'; its slope at the fixed point is the speed."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The em staircase and the cobweb",
        "Left: for a single observation y = 1.8 of the symmetric two-component mixture, contours of the divergence F(a, m) between a point a of the data manifold D and a point m of the model manifold M, the sigmoid curve of the E-step, the straight line of the M-step and the staircase of the em algorithm descending to the minimum near m = 1.79; the point m = 0 is a stationary point that is not a minimum. Right: the EM map of the symmetric mixture for true separations m = 0.5 and 1.5 with cobwebs from m = 2.6; the slope at the fixed point is 0.657 and 0.084.", b))


def mix3_rate(w, delta):
    """Largest eigenvalue of G_X^-1 G_(X|Y) for the mixture w N(0,1) + (1-w) N(delta,1), and the contraction of EM run on the whole density (last step with error above 1e-7)."""
    q = pop_quant(w, 0.0, delta)
    lam = float(np.sort(np.linalg.eigvals(np.linalg.solve(q["GX"], q["GM"])).real)[-1])
    x0 = np.array([w, 0.0, delta]); x = x0 + np.array([1e-3, 1e-3, -1e-3]); errs = [np.linalg.norm(x - x0)]
    for _ in range(3000):
        x = pop_em_map(x, (w, 0.0, delta)); errs.append(np.linalg.norm(x - x0))
        if errs[-1] < 1e-10:
            break
    meas = float("nan")
    for t in range(len(errs) - 1, 0, -1):
        if errs[t] >= 1e-7:
            meas = errs[t] / errs[t - 1]; break
    return lam, meas


def fig_rate(path):
    b = []
    W, H = 800, 440
    P1 = Panel(b, 62, 56, 318, 250, (0, 6), (0, 1))
    P1.frame([0, 1, 2, 3, 4, 5, 6], [0, 0.25, 0.5, 0.75, 1], "separation of the component means (in σ)", "EM contraction per step", "Speed of EM: the fraction of missing information", True)
    ms = np.linspace(0.02, 3.0, 150)
    P1.line(2 * ms, [sym_rate(m) for m in ms], "ln s1")
    ds = np.linspace(0.1, 6.0, 60)
    lam = [mix3_rate(0.35, d)[0] for d in ds]
    P1.line(ds, lam, "ln s2")
    for m, rate, steps, meas in STORE["sym_table"]:
        if meas == meas and 2 * m <= 6:
            P1.dot(2 * m, meas, "f1", 3.6)
    m3 = []
    for d in (0.5, 1.0, 2.0, 3.0, 4.0):
        l_, me_ = mix3_rate(0.35, d)
        m3.append((d, l_, me_)); P1.dot(d, me_, "f2", 3.6)
    STORE["mix3_rates"] = m3
    legend_col(b, 62 + 150, 56 + 20, [("s1", "symmetric model: 1 − I_o(m)"), ("s2", "w = 0.35: λ_{max}(G_X⁻¹ G_{X∣Y})")])
    b.append(f'<text class="sm" x="{62 + 150}" y="{56 + 20 + 46}">dots: EM run on the whole density</text>')
    # right: iterations
    P2 = Panel(b, 452, 56, 318, 250, (0, 6), (0, 3.6))
    P2.frame([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], "separation of the component means (in σ)", "steps to shrink the error by 10⁷", "Steps needed", True, None, {0: "1", 1: "10", 2: "100", 3: "1000"})
    ms = np.linspace(0.05, 3.0, 200)
    P2.line(2 * ms, [math.log10(math.log(1e-7) / math.log(sym_rate(m))) for m in ms], "ln s1")
    for m, rate, steps, meas in STORE["sym_table"]:
        if steps > 1 and 2 * m <= 6:
            P2.dot(2 * m, math.log10(steps), "f1", 3.6)
    P2.text(0.45, 3.3, "m → 0: the rate tends to 1, the stall of section 3", "sm", "start")
    note(b, 62, 372, ["Left: EM contracts the error by a fixed factor per step, and that factor is the largest fraction of the information about the parameter that",
                      "the hidden label withholds, (8.30)-(8.31). It tends to 0 when the components are far apart and to 1 when they merge. The curves come",
                      "from the information matrices, the dots from running EM. Right: the number of steps (logarithmic axis) to reduce the error by 10⁷,",
                      "from the formula (curve) and measured (dots; from a start 0.001 away, to an error below 1e-10; counts are whole steps)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The speed of EM is the fraction of missing information",
        "Left: the contraction factor of EM per step against the separation of the component means, for the symmetric mixture and for a mixture with weights 0.35 and 0.65. The curves are one minus the Fisher information of the marginal model over the complete-data information, and the largest eigenvalue of the inverse complete information times the missing information; dots are measured by running EM. The factor goes from 0 for far apart components to 1 for merging components. Right: the number of steps (logarithmic scale) needed to reduce the error by seven orders of magnitude, from the formula and from runs.", b))


def fig_loss(path):
    b = []
    W, H = 800, 425
    mu, sg = 0.3, 1.0
    gX = np.diag([1 / sg ** 2, 2 / sg ** 2])
    P1 = Panel(b, 62, 56, 318, 250, (0, 3), (0, 1))
    P1.frame([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 0.25, 0.5, 0.75, 1], "bin width h  (σ = 1)", "fraction of the information lost", "Grouping into bins", True)
    hs = np.linspace(0.04, 3.0, 80)
    lm, ls = [], []
    for h in hs:
        gT = grouped_info(mu, sg, h, K=int(9 / h) + 2)[0]
        lm.append(1 - gT[0, 0] / gX[0, 0]); ls.append(1 - gT[1, 1] / gX[1, 1])
    P1.line(hs, lm, "ln s1"); P1.line(hs, ls, "ln s2")
    P1.line(hs, hs ** 2 / 12, "dot s1"); P1.line(hs, hs ** 2 / 6, "dot s2")
    for h, gT, gC in STORE["group_rows"]:
        P1.dot(h, gC[0, 0] / gX[0, 0], "f1", 3.4); P1.dot(h, gC[1, 1] / gX[1, 1], "f2", 3.4)
    legend_col(b, 62 + 14, 56 + 14, [("s1", "mean μ"), ("s2", "standard deviation σ"), ("dot s1", "h²/12: small-bin law for μ"), ("dot s2", "h²/6: small-bin law for σ")])
    P2 = Panel(b, 452, 56, 318, 250, (-2.5, 3.5), (0, 1))
    P2.frame([-2, -1, 0, 1, 2, 3], [0, 0.25, 0.5, 0.75, 1], "censoring point c   (T = min(x, c))", "fraction of the information lost", "Censoring N(0.3, 1) at c", True)
    cs = np.linspace(-2.5, 3.5, 60)
    cm, cs_ = [], []
    for c in cs:
        gT, gC, Pc = censored_info(mu, sg, c)
        cm.append(gC[0, 0] / gX[0, 0]); cs_.append(gC[1, 1] / gX[1, 1])
    P2.line(cs, cm, "ln s1"); P2.line(cs, cs_, "ln s2")
    for c, Pc, lmu, lsg, gT in STORE["censor_rows"]:
        P2.dot(c, lmu, "f1", 3.4); P2.dot(c, lsg, "f2", 3.4)
    legend_col(b, 452 + 150, 56 + 14, [("s1", "mean μ"), ("s2", "standard deviation σ")])
    note(b, 62, 372, ["Each dot is a number printed by the script: the loss is g^{X∣T}, the score variance inside the bin (or beyond the censoring point), as a fraction of g^X.",
                      "Left: small bins lose h²/12σ² of the information about μ and twice that about σ; coarse bins lose less than that law says. Right: censoring far",
                      "to the right loses little; moving the censoring point towards the centre removes the information carried by the tail."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Loss of information by grouping and by censoring",
        "Left: the fraction of the Fisher information about the mean and about the standard deviation of a Gaussian that is lost when the observation is replaced by the index of a bin of width h; for small h the losses are h squared over 12 and h squared over 6 (dotted), for coarse bins they are smaller. Right: the fraction lost when the observation is replaced by the minimum of x and a censoring point c; it is small for c large and grows as c moves towards the centre of the distribution.", b))


def leaf_efficiency(angle):
    """Efficiency of a consistent estimator of u in N(u, u^2) at u = 1 whose leaf leaves the model point at the given angle from the eta_1 axis (a straight leaf, Fisher metric of the Gaussian family)."""
    Gi = np.array([[3.0, -1.0], [-1.0, 0.5]])
    t = np.array([1.0, 4.0]); d = np.array([math.cos(angle), math.sin(angle)])
    c2 = (t @ Gi @ d) ** 2 / ((t @ Gi @ t) * (d @ Gi @ d))
    return 1 - c2


def fig_misspec(path):
    b = []
    W, H = 800, 500
    P1 = Panel(b, 62, 56, 318, 250, (0.3, 1.7), (0.0, 5.6))
    P1.frame([0.5, 1.0, 1.5], [0, 1, 2, 3, 4, 5], "η₁ = E[x]", "η₂ = E[x²]", "Leaves through the true point p = N(1, 1)", True)
    xs = np.linspace(0.3, 1.7, 200)
    P1.line(xs, xs ** 2, "dash s0")
    P1.line(xs, 2 * xs ** 2, "ln sk")
    c = 1.5
    P1.line(xs, (1 + c) * xs ** 2, "thin s3")
    kap = (-1 + math.sqrt(1 + 8 * c)) / (2 * c)
    for v in (0.4, 0.6, 0.8, 1.1, 1.3):
        P1.line(xs, v * xs + c * v * v, "con s3")
    P1.line(xs, xs + 1.0, "dash sk")                                   # the leaf of the MLE (c = 1)
    P1.line([1.0, 1.0], [0.0, 5.6], "ln s2")                             # q1: vertical
    P1.line(xs, np.full_like(xs, 2.0), "ln s4")                          # q4: horizontal
    P1.line(xs, kap * xs + c * kap * kap, "ln s3")                       # q5: through p, slope kappa
    P1.dot(1.0, 2.0, "f4", 5.0)
    P1.dot(kap, (1 + c) * kap * kap, "f3", 4.0)
    legend_col(b, 62 + 6, 56 + 250 + 52, [("s2", "model N(v, 1): vertical leaf through p, efficiency 1/3"), ("s4", "model N(0, v²): horizontal leaf, efficiency 8/9"),
                                         ("s3", f"model N(v, {c} v²): leaf of slope κ = {kap:.3f}, efficiency {STORE['mis_rows']['q5'][0]:.4f}"), ("dash sk", "the MLE of M itself: slope 1, efficiency 1 (dashed)")])
    P1.text(1.04, 2.3, "p", "v", "start")
    # right: efficiency against the angle of the leaf
    P2 = Panel(b, 452, 56, 318, 250, (-90, 90), (0.0, 1.14))
    P2.frame([-90, -45, 0, 45, 90], [0, 0.25, 0.5, 0.75, 1], "angle of the leaf from the η₁ axis (degrees)", "efficiency  ḡ / g_uu", "Efficiency is the squared sine of the angle", True)
    angs = np.linspace(-90, 90, 361)
    P2.line(angs, [leaf_efficiency(math.radians(a_)) for a_ in angs], "ln sk")
    pts = [(90.0, STORE["mis_rows"]["q1"][0], "f2", "N(v, 1)"), (0.0, STORE["mis_rows"]["q4"][0], "f4", "N(0, v²)"),
           (math.degrees(math.atan(kap)), STORE["mis_rows"]["q5"][0], "f3", "N(v, 1.5 v²)"), (45.0, 1.0, "f0", "MLE")]
    for a_, e_, fc, lab in pts:
        P2.dot(a_, e_, fc, 4.8)
    P2.text(88, 0.42, "N(v, 1)", "sm", "end"); P2.text(4, 0.82, "N(0, v²)", "sm", "start")
    P2.text(-88, 1.09, "dots at 41° (N(v, 1.5 v²)) and 45° (the MLE): both close to 1", "sm", "start")
    note(b, 62, 425, ["Left: the data (x̄, mean of x²) are a point of the plane; a misspecified model sends it to its own model curve along straight leaves",
                      "(the q-ancillary families). Each model's leaf through the true point p has its own angle to M, shown against the efficiency on the right.",
                      "Vertical leaf: the sample mean. Horizontal leaf: √(m₂/2). Sloped leaf: the maximum likelihood estimator of the wrong coefficient of",
                      "variation c = 1.5, after relabelling the model by f(v) = v/κ."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Misspecified models: leaves, angles and efficiency",
        "Left: the plane of the observed point (mean of x, mean of x squared) with the true model N(u, u squared) as a black parabola, the misspecified model N(v, 1.5 v squared) in green, its straight leaves, and the three leaves through the true point N(1, 1): the vertical leaf of the model N(v, 1), the horizontal leaf of the model N(0, v squared) and the sloped leaf of N(v, 1.5 v squared), with the dashed leaf of the true maximum likelihood estimator. Right: the efficiency of a consistent estimator as a function of the angle of its leaf, equal to one third for the vertical leaf, 8/9 for the horizontal leaf and 1 for the orthogonal leaf; the three models sit on the curve.", b))


def fig_neural(path):
    b = []
    W, H = 800, 450
    P1 = Panel(b, 62, 56, 318, 250, (0, 180), (0.0, 1.14))
    P1.frame([0, 45, 90, 135, 180], [0, 0.25, 0.5, 0.75, 1], "direction of r′ in the plane of two neurons (degrees)", "efficiency of the decoder that ignores V", "Two neurons, correlation ρ", True)
    ang = np.linspace(0, 180, 361)
    for rho, cls in ((0.3, "s3"), (0.6, "s1"), (0.9, "s2")):
        P1.line(ang, (1 - rho ** 2) / (1 - rho ** 2 * np.sin(np.radians(2 * ang)) ** 2), "ln " + cls)
    for rho, u, o in STORE["two_neuron"]:
        P1.dot((math.degrees(u) + 90) % 180, o["eff_geom"], "f1", 4.5)
    legend(b, 62 + 104, 56 + 214, [("s3", "ρ = 0.3"), ("s1", "ρ = 0.6"), ("s2", "ρ = 0.9")], col=84)
    P1.text(2, 1.09, "eigenvectors of V at 45° and 135°: efficiency 1", "sm", "start")
    P2 = Panel(b, 452, 56, 318, 250, (0, 0.6), (0.0, 1.14))
    P2.frame([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 0.25, 0.5, 0.75, 1], "range b of the correlation (ring circumference 1)", "efficiency", "Ring of n neurons, tuning width 0.1, ρ = 0.5", True)
    bs = np.linspace(0.005, 0.6, 80)
    for n, cls in ((25, "s3"), (100, "s1"), (400, "s2")):
        rp = ring_field(n, 0.1)
        P2.line(bs, [efficiency_formula(rp, ring_cov(n, 0.5, bb, "gauss")) for bb in bs], "ln " + cls)
    P2.line([0, 0.6], [1, 1], "dash s0")
    P2.line([0.1, 0.1], [0, 1.14], "dot s0")
    P2.text(0.104, 0.05, "tuning width", "sm", "start")
    P2.text(0.595, 1.09, "cosine tuning, or b → ∞ (uniform correlation): exactly 1", "sm", "end")
    legend_col(b, 452 + 232, 56 + 150, [("s3", "n = 25"), ("s1", "n = 100"), ("s2", "n = 400")])
    note(b, 62, 372, ["Left: the efficiency is 1 when r′ is an eigenvector of V (here (1, 1) or (1, −1)) and 1 − ρ² when it is along an axis; the dots are the",
                      "geometric computation of Theorem 8.4 in the five-dimensional Gaussian family for ρ = 0.6. Right: short-range correlation costs almost",
                      "nothing, correlation as wide as the tuning curve costs most of the information (more with more neurons), and uniform correlation",
                      "costs nothing again. 'No loss' is a property of special (V, r) pairs, not of the misspecified model as such."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The decoder that ignores correlation",
        "Left: efficiency of the least-squares decoder that ignores a covariance V for two neurons, as a function of the direction of the derivative of the tuning curve, for correlations 0.3, 0.6 and 0.9; it is one when the direction is an eigenvector of V and one minus rho squared along an axis; dots are the geometric computation of Theorem 8.4. Right: efficiency for a ring of n neurons with Gaussian bump tuning of width 0.1 and wrapped-Gaussian correlation of range b, for n = 25, 100 and 400: nearly one for short-range correlation, much smaller for correlation as wide as the tuning curve and decreasing with n, and one again for uniform correlation (large b); cosine tuning gives exactly one.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_staircase(out / "em-staircase.svg")
    fig_rate(out / "em-rate.svg")
    fig_loss(out / "info-loss.svg")
    fig_misspec(out / "misspecified.svg")
    fig_neural(out / "neural-field.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

def main():
    check_manifolds()
    check_em_is_em()
    check_limits()
    check_rates()
    check_rbm()
    check_reduction_gauss()
    check_neurons()
    check_misspecified()
    check_neural_field()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
