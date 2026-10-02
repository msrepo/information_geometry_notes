#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 13, checked by hand.

Signal processing and optimization: six loosely connected topics, each tested on a small example with an independent route.
Every number quoted in the notes comes from here. Equation numbers are the book's.

Running examples: a 6 x 6 covariance with eigenvalues 5, 3, 1.5, 0.8, 0.4, 0.2 (principal components); two independent sources with exact
Gauss-Legendre quadrature (uniform, bimodal, logistic, sech, Laplace) mixed by a 2 x 2 matrix (independent components); a 6 x 40 non-negative
matrix of rank 3 (NMF); Gaussian measurement matrices and a 7-column regression design (sparse recovery, the lasso path); a hexagon with
a linear cost (the central path); the exponential family exp(-theta x^3) of the book's Example 13.1 (the Hyvarinen score).

Checked here, in the order the notes use them:

  1. principal components (13.1): the loss and the index range of (13.13); whitening and the rotation left over; the missing 1/12 in (13.46);
     mutual information of rotated uniform sources; Oja's rule and its exact mean flow; the matrix flows (13.31), (13.34), (13.35), (13.36),
     (13.40), (13.41): stability of the Stiefel manifold, Lemma 13.1, Xu's flow (13.33), Brockett's flow (13.39); (not in the book) PCA as
     an m-projection onto the probabilistic-PCA family;
  2. independent components (13.2): the density (13.52), the entropy (13.66) and the loss (13.62)-(13.64) by integration in y-space, and the
     missing |det W| in (13.88); the
     tangent spaces of S_W and S_I (orthogonality only modulo the scales, and only for zero means), the critical point and its scale;
     the natural gradient (13.73)-(13.87) and the sign of (13.86); equivariance; the Hessian at the separating point and the stability
     conditions kappa_1 kappa_2 > 1, tested on the online rule; the conditions (13.106)-(13.110); Theorem 13.1 to order eta;
  3. non-negative matrix factorisation (13.3): the gradients (13.123), the sign of (13.120), Lee-Seung against exponential gradient
     (a step with rate 1/logmean), monotonicity, positivity, the cone picture and non-uniqueness;
  4. sparse signal processing (13.4): generalised inverse, L0 against L1, a compressed-sensing experiment against m > 2 k log n,
     the exact lasso path, Theorem 13.3 and the KKT conditions (13.151)-(13.155), the projection theorem, the normal cones, the
     logistic loss (13.143), Theorem 13.4 and (13.160)-(13.161) with a counterexample, the Minkowskian gradient flow against the path;
  5. convex programming (13.5): the barrier (13.181) and (13.188)-(13.189), the central path as a straight line in eta (13.195)-(13.197),
     its curvature, the natural-gradient steps (13.193), the equality-constrained form, the barrier -log det X (13.185);
  6. game theory (13.6): Lemma 13.3 and its boundary term, Example 13.1 (signs of (13.248)-(13.253)), the decomposition (13.228)-(13.236)
     including the matrix order in (13.233),
     Theorem 13.5, the discrete score (13.254)-(13.265), the Bregman score (13.214)-(13.220), (not in the book) the saddle point of log loss.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. Fixed seeds; the whole script takes about ten seconds.

Run:  python3 signal_processing.py            (checks)
      python3 signal_processing.py --figures  (checks, then rewrite ../figures/*.svg)
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


def rng_for(seed):
    return np.random.default_rng(seed)


def fro(M):
    return float(np.linalg.norm(M))




def trap(y, x):
    """Trapezoid rule (np.trapz was renamed np.trapezoid in numpy 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ================================================================== 1. principal component analysis (13.1)

PCA_EIGS = np.array([5.0, 3.0, 1.5, 0.8, 0.4, 0.2])


def pca_problem(seed=11, eigs=PCA_EIGS):
    """A covariance matrix V_X = O diag(eigs) O^T with a random orthogonal O (columns o_i)."""
    rng = rng_for(seed)
    O, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return O, O @ np.diag(eigs) @ O.T


def mi_uniform_pair(t):
    """Mutual information (nats) of the two outputs of a rotation by angle t applied to two independent uniform variables: 2 h(trapezoid) - h(square) = 2 ln a + b/a."""
    a, b = max(abs(math.cos(t)), abs(math.sin(t))), min(abs(math.cos(t)), abs(math.sin(t)))
    return 2 * math.log(a) + b / a


def projector(W):
    """Orthogonal projector onto the column span of W (n x k, full column rank)."""
    return W @ np.linalg.solve(W.T @ W, W.T)


def flow_rhs(name, V, D2):
    """Right-hand sides of the matrix flows of section 13.1.3 (M is n x k)."""
    def f(M):
        VM = V @ M
        MtM = M.T @ M
        MtVM = M.T @ VM
        if name == "oja":                # (13.31)
            return VM - M @ MtVM
        if name == "ascent":             # (13.34)
            return -VM + M @ MtVM
        if name == "n35":                # (13.35)
            return VM @ MtM - M @ MtVM
        if name == "n36_printed":        # (13.36) exactly as printed: -V M M^T M - M M^T V M
            return -(VM @ MtM) - M @ MtVM
        if name == "n36":                # (13.36) with the sign of the second term corrected: -(13.35)
            return -(VM @ MtM) + M @ MtVM
        if name == "n40":                # (13.40)
            return VM @ MtM - M @ MtVM + M @ (D2 - MtM)
        if name == "n41":                # (13.41)
            return -(VM @ MtM) + M @ MtVM + M @ (D2 - MtM)
        if name == "xu":                 # (13.33) with D2 playing the role of D (see the notes)
            return VM @ D2 - M @ D2 @ MtVM
        if name == "xu_D":               # (13.33) as printed, with D itself
            Dm = np.sqrt(D2)
            return VM @ Dm - M @ Dm @ MtVM
        raise ValueError(name)
    return f


def rk4(f, M, T, h, record=None, stop=None):
    """Integrate dM/dt = f(M) by classical Runge-Kutta; returns (M_end, t_end, records).  stop(M) -> True ends the run early."""
    n = int(round(T / h)); t = 0.0; rec = []
    for i in range(n):
        if record is not None and i % record == 0:
            rec.append((t, M.copy()))
        k1 = f(M); k2 = f(M + 0.5 * h * k1); k3 = f(M + 0.5 * h * k2); k4 = f(M + h * k3)
        M = M + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); t += h
        if stop is not None and stop(M):
            break
    if record is not None:
        rec.append((t, M.copy()))
    return M, t, rec


def check_pca():
    head("1. Principal component analysis (section 13.1)")
    O, V = pca_problem()
    lam = PCA_EIGS; n = len(lam)
    print(f"   running example: V_X = O diag{tuple(float(x) for x in lam)} O^T in R^{n}, with a fixed random orthogonal O (eigenvalues distinct, as the book assumes under (13.5))")
    # ---- 1a. the loss (13.11) and what minimises / maximises it
    rng = rng_for(21)
    k = 2
    xs = (O * np.sqrt(lam)) @ rng.normal(size=(n, 400_000))          # samples of x ~ N(0, V_X), columns
    P = O[:, :k] @ O[:, :k].T
    mc = 0.5 * np.mean(np.sum((xs - P @ xs) ** 2, axis=0))
    print(f"   loss L = (1/2) E|x - sum_(i<=k) s_i o_i|^2 (13.11), k = {k}: closed form (1/2) sum_(i>k) lambda_i = {0.5 * lam[k:].sum():.4f}; Monte Carlo (4e5 samples) {mc:.4f}")
    Ls = []
    for _ in range(20000):
        Q, _ = np.linalg.qr(rng.normal(size=(n, k)))
        Ls.append(0.5 * np.trace((np.eye(n) - Q @ Q.T) @ V))
    Ls = np.array(Ls)
    print(f"   20000 random {k}-dimensional subspaces: smallest loss found {Ls.min():.4f} >= {0.5 * lam[k:].sum():.4f} (principal subspace), largest {Ls.max():.4f} <= {0.5 * lam[:n - k].sum():.4f} (minor subspace)")
    Pm = O[:, n - k:] @ O[:, n - k:].T
    print(f"   the minor subspace (o_(n-k+1..n)) as the retained part gives L = {0.5 * np.trace((np.eye(n) - Pm) @ V):.4f} = (1/2) sum_(i<=n-k) lambda_i: it maximises L; as printed, (13.13) repeats the sum over i = 1..k of (13.11), which minimises it (index range slip)")
    STORE["pca_loss"] = (Ls.min(), Ls.max(), 0.5 * lam[k:].sum(), 0.5 * lam[:n - k].sum())
    # ---- 1b. whitening (13.14)-(13.19) and the rotation left over, with the book's example (13.44)-(13.46)
    Sx = O @ np.diag(lam ** -0.5) @ O.T                              # V^(-1/2): the whitening transformation
    print(f"   whitening (13.14)-(13.18): covariance of V^(-1/2) x is I to {fro(Sx @ V @ Sx.T - np.eye(n)):.1e}; any further rotation U keeps it (13.19)")
    A = np.array([[1.0, 0.0], [1.0, 2.0]])                           # x1 = s1, x2 = s1 + 2 s2 (13.44)-(13.45)
    VX = A @ A.T / 12                                                # Var(s_i) = 1/12 for U[-1/2, 1/2]
    w, Ev = np.linalg.eigh(VX)
    print(f"   the book's mixture (13.44)-(13.45): Cov(x) = (1/12) [[1,1],[1,5]] = {np.round(VX * 12, 3).tolist()}/12 (printed without the factor 1/12, which is right only for unit-variance sources); eigenvalues {np.round(w, 5).tolist()}, "
          f"principal vector o_1 = {np.round(Ev[:, 1] * np.sign(Ev[1, 1]), 4).tolist()} (angle {math.degrees(math.atan2(Ev[1, 1], Ev[0, 1])):.2f} deg to the x_1 axis)")
    Wh = Ev @ np.diag(w ** -0.5) @ Ev.T                              # symmetric whitening V^(-1/2)
    Rot = Wh @ A * math.sqrt(1 / 12)                                 # whitened x = Rot s_unit, with s_unit = sqrt(12) s of unit variance
    Rp = np.diag(w ** -0.5) @ Ev.T @ A * math.sqrt(1 / 12)           # the PCA components themselves, rescaled to unit variance: also orthogonal * s_unit
    ang = math.degrees(math.atan2(Rp[1, 0], Rp[0, 0]))
    print(f"   after whitening the data are x~ = Rot s_unit with Rot orthogonal (|Rot Rot^T - I| = {fro(Rot @ Rot.T - np.eye(2)):.1e}); the PCA components themselves, rescaled to unit variance, are the sources rotated by "
          f"{ang:.2f} deg (|R R^T - I| = {fro(Rp @ Rp.T - np.eye(2)):.1e}): second-order statistics cannot see this angle, the book's 'indefiniteness of rotation'")
    STORE["rot_angle"] = ang
    # exact dependence of the PCA components: y = (o_1 . x, o_2 . x) uncorrelated but E[y1^2 y2^2] != E[y1^2] E[y2^2]
    c = Ev.T @ A                                                     # y_i = sum_j c_ij s_j
    def mom22_exact(a, b):
        """E[(a.s)^2 (b.s)^2] for two independent U[-1/2, 1/2] variables: E s^2 = 1/12, E s^4 = 1/80 (sum over all index pairings)."""
        s2, s4 = 1 / 12, 1 / 80
        tot = 0.0
        for i in range(2):
            for j in range(2):
                for kk in range(2):
                    for l in range(2):
                        idx = (i, j, kk, l)
                        if len(set(idx)) == 1:
                            e = s4
                        elif sorted([idx.count(0), idx.count(1)]) == [2, 2]:
                            e = s2 * s2
                        else:
                            e = 0.0
                        tot += a[i] * a[j] * b[kk] * b[l] * e
        return tot
    y1, y2 = c[0], c[1]
    v1, v2 = float(y1 @ y1) / 12, float(y2 @ y2) / 12
    cov12 = float(y1 @ y2) / 12
    dep = mom22_exact(y1, y2) - v1 * v2
    corr2 = dep / math.sqrt((mom22_exact(y1, y1) - v1 * v1) * (mom22_exact(y2, y2) - v2 * v2))
    print(f"   PCA components of the book's example: E[y1 y2] = {cov12:.1e} (uncorrelated) but Cov(y1^2, y2^2) = {dep:+.6f}, correlation of the squares {corr2:+.4f} (dependent); undoing the rotation makes both vanish: the sources")
    # ---- 1c. mutual information of two rotated uniform sources, exactly: the contrast of 13.2 on a slice
    def trapezoid_entropy_numeric(a, b):
        """Differential entropy of the sum of independent U(-a/2, a/2) and U(-b/2, b/2), a >= b > 0, by quadrature of the trapezoid density."""
        xs_ = np.linspace(-(a + b) / 2, (a + b) / 2, 200001)
        d = np.where(np.abs(xs_) <= (a - b) / 2, 1 / a, ((a + b) / 2 - np.abs(xs_)) / (a * b))
        d = np.maximum(d, 1e-300)
        return float(trap(-d * np.log(d), xs_))
    th = np.radians(np.arange(0, 91, 5))
    MI = np.array([mi_uniform_pair(t) for t in th])
    chk = max(abs(2 * trapezoid_entropy_numeric(max(abs(math.cos(t)), abs(math.sin(t))), min(abs(math.cos(t)), abs(math.sin(t)))) - mi_uniform_pair(t)) for t in np.radians([7.0, 20.0, 45.0, 63.0]))
    i45 = int(np.argmin(np.abs(np.degrees(th) - 45)))
    print(f"   mutual information I(y1, y2) = h(y1) + h(y2) - h(s) of the source pair rotated by angle t (the marginals are trapezoids, h = ln a + b/(2a) with a = max(|cos t|, |sin t|), b = min; quadrature agrees to {chk:.0e}): 0 at t = 0, 90 deg; {MI[i45]:.5f} at 45 deg (= 1 - ln 2 = {1 - math.log(2):.5f}); table over t:")
    print("      " + "  ".join(f"{math.degrees(t):.0f}:{m:.4f}" for t, m in zip(th[::2], MI[::2])))
    off = ang % 90.0; off = min(off, 90.0 - off)
    mi_pca = mi_uniform_pair(math.radians(ang))
    print(f"   the PCA solution of the book's example sits at the offset {ang:.2f} deg, that is {off:.2f} deg away from the nearest source axis: I(y1, y2) = {mi_pca:.4f} nats, not 0 (the independent components are at 0 or 90 deg)")
    STORE["mi_curve"] = (np.degrees(th), MI); STORE["mi_pca"] = mi_pca; STORE["mi_off"] = off
    # ---- 1d. Oja's rule (13.24) and the exact solution of its mean flow
    rng = rng_for(22)
    w0 = rng.normal(size=n); w0 /= np.linalg.norm(w0)
    flow1 = flow_rhs("oja", V, None)
    print("   k = 1, mean flow w' = V w - (w^T V w) w (the average of (13.24)): in the eigenbasis w_i/w_1 evolves exactly as exp(-(lambda_1 - lambda_i) t); RK4 against this closed form:")
    c0 = O.T @ w0
    worst = 0.0
    for tt in (2.0, 5.0, 10.0):
        M, _, _ = rk4(lambda m: flow1(m.reshape(-1, 1)).reshape(-1), w0.copy(), tt, 0.005)
        c = O.T @ M
        exact = (c0[1:] / c0[0]) * np.exp(-(lam[0] - lam[1:]) * tt)
        worst = max(worst, float(np.max(np.abs(c[1:] / c[0] - exact))))
    print(f"      largest difference in w_i/w_1 over t = 2, 5, 10: {worst:.1e}; |w|^2 goes to 1: {float(np.sum(M ** 2)):.10f} at t = 10 (started at 1)")
    R, T, eta = 500, 10000, 0.002
    W = rng.normal(size=(R, n)) * 0.3
    L = O * np.sqrt(lam)
    for _ in range(T):
        x = rng.normal(size=(R, n)) @ L.T
        y = np.sum(W * x, axis=1)
        W += eta * (y[:, None] * x - (y ** 2)[:, None] * W)
    cosv = np.abs(W @ O[:, 0]) / np.linalg.norm(W, axis=1)
    nw = np.linalg.norm(W, axis=1)
    print(f"   Oja's rule (13.24) with learning rate {eta} (the printed rule has none), {R} independent runs of {T} samples: |w| = {nw.mean():.4f} +- {nw.std():.4f}, |cos(w, o_1)| = {cosv.mean():.4f} (smallest of {R}: {cosv.min():.3f})")
    # ---- 1e. the matrix flows and the stability of the Stiefel manifold
    k = 2
    rng = rng_for(23)
    W0, _ = np.linalg.qr(rng.normal(size=(n, k)))
    Dd = np.array([1.5, 0.8]); D2 = np.diag(Dd ** 2)
    dM = 0.05 * rng.normal(size=(n, k))
    Pp = O[:, :k] @ O[:, :k].T; Pmin = O[:, n - k:] @ O[:, n - k:].T
    names = (("oja", "(13.31) Oja subspace"), ("ascent", "(13.34) 'ascent'"), ("n35", "(13.35)"), ("n36_printed", "(13.36) as printed"), ("n36", "(13.36), sign corrected"), ("n40", "(13.40)"), ("n41", "(13.41)"))
    # exact algebra first: the rate of change of M^T M at random points M, for each flow
    rng2 = rng_for(25)
    print(f"   rate of change of M^T M, |d(M^T M)/dt|_F at 200 random n x k matrices M (n = {n}, k = {k}, entries N(0,1)), largest value (D^2 = diag(1.5^2, 0.8^2) in the last two):")
    rates = {}
    for name, lab in names[:5]:
        f = flow_rhs(name, V, D2); worst = 0.0
        for _ in range(200):
            M = rng2.normal(size=(n, k)); Md = f(M)
            worst = max(worst, fro(Md.T @ M + M.T @ Md))
        rates[name] = worst
        print(f"      {lab:26s} {worst:.2e}")
    print("      so M^T M is conserved by (13.35) and by the sign-corrected (13.36) (Lemma 13.1 (1), to rounding) and not by (13.31) (stable towards I) or by (13.36) as printed")
    STORE["mtm_rates"] = rates
    print(f"   perturbed trajectories (RK4, h = 0.05; T = 24, and T = 120 for (13.40), (13.41) whose rates carry the factor d_i^2): n = {n}, k = {k}, M(0) = a point of S_(n,k) plus a perturbation of size 0.05")
    print(f"      {'flow':26s} {'|M|_F':>9s} {'|M^TM - I|, t = 0':>18s} {'t = T':>10s} {'drift of M^TM':>14s} {'dist. to principal / minor subspace':>38s}")
    res = {}
    for name, lab in names:
        if name in ("n40", "n41"):
            M0 = np.vstack([np.diag(Dd), np.zeros((n - k, k))]) + 0.5 * dM
            ref = D2; T_ = 120.0
        else:
            M0 = W0 + dM
            ref = np.eye(k); T_ = 24.0
        f = flow_rhs(name, V, D2)
        blown = lambda M: (not np.all(np.isfinite(M))) or np.abs(M).max() > 1e6
        M, tend, rec = rk4(f, M0.copy(), T_, 0.05, record=10, stop=blown)
        d0 = fro(M0.T @ M0 - ref)
        if blown(M):
            res[name] = dict(blow=tend, rec=rec, ref=ref, d0=d0)
            print(f"      {lab:26s} {'diverges':>9s} {d0:>18.3f} {'(|M| > 1e6 at t = %.2f)' % tend:>24s}")
            continue
        Pm_ = projector(M) if np.linalg.matrix_rank(M, tol=1e-9) == k else None
        dpr, dmn = (fro(Pm_ - Pp), fro(Pm_ - Pmin)) if Pm_ is not None else (float("nan"), float("nan"))
        res[name] = dict(d0=d0, dT=fro(M.T @ M - ref), drift=fro(M.T @ M - M0.T @ M0), dpr=dpr, dmn=dmn, rec=rec, ref=ref, normM=fro(M))
        sub = ("%.1e / %.1e" % (dpr, dmn)) if Pm_ is not None else "M -> 0 (no subspace)"
        print(f"      {lab:26s} {fro(M):>9.3f} {d0:>18.3f} {res[name]['dT']:>10.2e} {res[name]['drift']:>14.2e} {sub:>38s}")
    STORE["pca_flows"] = res
    Lend = lambda M: 0.5 * float(np.trace((np.eye(n) - projector(M)) @ V))
    print(f"   the loss (13.29) at the end of the runs: L = {Lend(res['oja']['rec'][-1][1]):.4f} for (13.31) and (13.40) {Lend(res['n40']['rec'][-1][1]):.4f} (minimum {0.5 * lam[k:].sum():.4f}), {Lend(res['n41']['rec'][-1][1]):.4f} for (13.41) and {Lend(res['n36']['rec'][-1][1]):.4f} for the corrected (13.36) (maximum {0.5 * lam[:n - k].sum():.4f}): the flows find the minimiser and the maximiser of (13.11)")
    print("   reading: (13.31) is stable (the defect shrinks to 0), (13.34) leaves S_(n,k) (the puzzle the book mentions), (13.35) and the corrected (13.36) are neutral (the defect stays at its initial value, drift = integration error), "
          "(13.36) as printed shrinks M to 0, and (13.40), (13.41) pull M^T M back to D^2 while following the principal and minor subspace respectively")
    print("   k = 1 closed forms (r = |w|^2, a = w^T V w > 0): (13.31) r' = 2a(1 - r) attracts r = 1; (13.34) r' = 2a(r - 1) repels it; (13.35) r' = 0 (neutral, Lemma 13.1), checked at w = 1.2 o_2:")
    for name in ("oja", "ascent", "n35"):
        f = flow_rhs(name, V, None)
        w_ = O[:, 1] * 1.2
        r_ = float(w_ @ w_); a_ = float(w_ @ V @ w_)
        rd = 2 * float(w_ @ f(w_.reshape(-1, 1)).reshape(-1))
        pred = {"oja": 2 * a_ * (1 - r_), "ascent": 2 * a_ * (r_ - 1), "n35": 0.0}[name]
        print(f"      {name:7s} d|w|^2/dt = {rd:+.6f}, closed form {pred:+.6f}")
    # ---- 1f. Lemma 13.1 (D and U invariant) and the link between (13.35) and Xu's algorithm (13.33)
    f35 = flow_rhs("n35", V, D2)
    th_ = 0.7
    U0 = np.array([[math.cos(th_), -math.sin(th_)], [math.sin(th_), math.cos(th_)]])
    M0 = W0 @ np.diag(Dd) @ U0
    M, _, _ = rk4(f35, M0.copy(), 20.0, 0.05)
    _, s0_, Vt0 = np.linalg.svd(M0, full_matrices=False); _, s1_, Vt1 = np.linalg.svd(M, full_matrices=False)
    print(f"   Lemma 13.1 from M(0) = W D U with D = diag{tuple(float(x) for x in Dd)} and a rotation U (angle {th_}): after t = 20 the singular values are {np.round(s1_, 6).tolist()} (D: {np.round(s0_, 6).tolist()}) and "
          f"|U(20) U(0)^T| = {np.round(np.abs(Vt1 @ Vt0.T), 5).tolist()} (identity up to signs): D and U are invariants; M^T M drifts by {fro(M.T @ M - M0.T @ M0):.1e}")
    M0 = np.vstack([np.diag(Dd), np.zeros((n - k, k))])
    Wx, _, _ = rk4(flow_rhs("xu", V, D2), M0 @ np.diag(1 / Dd), 6.0, 0.02)
    Wy, _, _ = rk4(flow_rhs("xu_D", V, D2), M0 @ np.diag(1 / Dd), 6.0, 0.02)
    Mw, _, _ = rk4(f35, M0.copy(), 6.0, 0.02)
    Mfin, _, _ = rk4(f35, M0.copy(), 60.0, 0.05)
    cols = [abs(float(Mfin[:, i] @ O[:, i])) / np.linalg.norm(Mfin[:, i]) for i in range(k)]
    print(f"   M(0) = [D; 0] has U = I and stays so (M^T M = D^2 is diagonal), so M = W D, and the columns end at m_i = d_i o_i: at t = 60 |cos(m_i, o_i)| = {np.round(cols, 6).tolist()}, |m_i| = {np.round(np.linalg.norm(Mfin, axis=0), 6).tolist()};"
          f" rewriting (13.35) in W = M D^(-1) gives W' = V W D^2 - W D^2 W^T V W (weights d_i^2): at t = 6 the solution of (13.35) divided by D differs from it by {fro(Mw @ np.diag(1 / Dd) - Wx):.1e}"
          f" (RK4 error) but from (13.33) as printed (weights d_i) by {fro(Mw @ np.diag(1 / Dd) - Wy):.3f}: the printed (13.33) is that algorithm only with D read as the matrix of squares")
    Wo, _, reco = rk4(flow_rhs("oja", V, None), W0.copy(), 30.0, 0.05, record=4)
    Wxu, _, recx = rk4(flow_rhs("xu", V, D2), W0.copy(), 30.0, 0.05, record=4)
    co = [abs(float(Wo[:, i] @ O[:, i])) / np.linalg.norm(Wo[:, i]) for i in range(k)]
    cx = [abs(float(Wxu[:, i] @ O[:, i])) / np.linalg.norm(Wxu[:, i]) for i in range(k)]
    print(f"   from a random start of the Stiefel manifold, t = 30: Oja's subspace flow (13.31) has |cos(w_i, o_i)| = {np.round(co, 4).tolist()} (the subspace is right, distance {fro(projector(Wo) - Pp):.1e}, the two vectors are a rotation of o_1, o_2); "
          f"the weighted flow has {np.round(cx, 8).tolist()} (the individual eigenvectors, d_1 > d_2)")
    STORE["xu"] = (co, cx)
    STORE["xu_rec"] = (reco, recx)
    # ---- 1g. Brockett flow, k = n
    Vb = V
    Dn = np.diag(np.array([1.3, 1.1, 0.9, 0.7, 0.5, 0.3]))
    Q0, _ = np.linalg.qr(rng_for(24).normal(size=(n, n)))
    M0 = Q0 @ Dn
    f = flow_rhs("n35", Vb, Dn ** 2)
    M, tt, rec = rk4(f, M0.copy(), 6.0, 0.01, record=20)
    Ls = np.array([np.trace(m @ m.T @ Vb) for _, m in rec])
    ts_ = np.array([t for t, _ in rec])
    # d/dt tr(M M^T V) = || [H, N] ||_F^2 with H = W^T V W, N = D^2, W = M D^-1 (U = I)
    ok = []
    for (t, m) in rec[::5]:
        Wm = m @ np.linalg.inv(Dn); H = Wm.T @ Vb @ Wm; N = Dn @ Dn
        rhs = flow_rhs("n35", Vb, Dn ** 2)(m)
        dL = 2 * np.trace(rhs @ m.T @ Vb)
        ok.append(abs(dL - fro(H @ N - N @ H) ** 2))
    print(f"   k = n (Brockett flow, (13.39)): along (13.35) from M(0) = (orthogonal) D, L = tr(M M^T V) rises monotonically: min increment {float(np.min(np.diff(Ls))):.2e} >= 0, from {Ls[0]:.4f} to {Ls[-1]:.4f} (max possible sum_i d_i^2 lambda_i = {float(np.sum(np.sort(np.diag(Dn) ** 2)[::-1] * lam)):.4f});"
          f" and dL/dt = |[H, D^2]|_F^2 with H = W^T V W holds to {max(ok):.1e} (the double-bracket form of Brockett's flow)")
    STORE["brockett"] = (ts_, Ls)
    # ---- 1h. (my addition) PCA as an m-projection onto the probabilistic-PCA family
    ppca_check(V, lam, O)


def kl_gauss0(V, S):
    """KL[N(0, V) || N(0, S)]."""
    n = len(V)
    return 0.5 * (np.trace(np.linalg.solve(S, V)) - n - np.linalg.slogdet(np.linalg.solve(S, V))[1])


def ppca_check(V, lam, O):
    """Minimise KL[N(0,V) || N(0, W W^T + s2 I)] over (W, s2) by gradient descent and compare with the closed form."""
    n = len(lam)
    print("   (not in the book) PCA as an m-projection: the family S_k = {N(0, W W^T + s^2 I)} (probabilistic PCA) and the data Gaussian N(0, V_X);")
    out = []
    for k in range(0, n):
        rng = rng_for(100 + k)
        W = rng.normal(size=(n, k)) * 0.5; s2 = 1.0
        def obj(W, s2):
            return kl_gauss0(V, W @ W.T + s2 * np.eye(n))
        f = obj(W, s2); step = 0.05
        for it in range(1500):
            S = W @ W.T + s2 * np.eye(n); Si = np.linalg.inv(S)
            Gs = 0.5 * (Si - Si @ V @ Si)                            # d KL / d S
            gW = 2 * Gs @ W; gs = np.trace(Gs)
            while True:
                W2 = W - step * gW; s22 = max(s2 - step * gs, 1e-6)
                f2 = obj(W2, s22)
                if f2 <= f:
                    break
                step *= 0.5
            W, s2, f = W2, s22, f2
            step *= 1.2
            if fro(gW) + abs(gs) < 1e-9:
                break
        disc = lam[k:]
        s2_closed = float(np.mean(disc)); kl_closed = 0.5 * (n - k) * (math.log(float(np.mean(disc))) - float(np.mean(np.log(disc))))
        S_star = O[:, :k] @ np.diag(lam[:k]) @ O[:, :k].T + s2_closed * (np.eye(n) - O[:, :k] @ O[:, :k].T)
        out.append((k, f, kl_closed, s2, s2_closed, fro(W @ W.T + s2 * np.eye(n) - S_star)))
    print(f"      {'k':>2s} {'min KL (gradient descent)':>26s} {'(n-k)/2 ln(AM/GM) of the discarded':>38s} {'s^2 found':>10s} {'mean of discarded':>18s} {'|S_found - S*|':>15s}")
    for k, f, kc, s2, s2c, dS in out:
        print(f"      {k:>2d} {f:>26.8f} {kc:>38.8f} {s2:>10.5f} {s2c:>18.5f} {dS:>15.1e}")
    print("      so the PCA subspace is the m-projection of N(0, V_X) onto S_k (the maximum likelihood fit), the noise level is the arithmetic mean of the discarded eigenvalues,"
          " and the information lost is (n-k)/2 times the log of the ratio of arithmetic to geometric mean of the discarded eigenvalues (0 for k = n-1, a single discarded eigenvalue)")
    STORE["ppca"] = out


# ================================================================== 2. independent component analysis (13.2)

def gl_pieces(edges, n):
    """Composite Gauss-Legendre nodes and weights on the intervals between consecutive edges."""
    x0, w0 = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (b - a) * x0 + 0.5 * (b + a)); ws.append(0.5 * (b - a) * w0)
    return np.concatenate(xs), np.concatenate(ws)


class Src:
    """A zero-mean, unit-variance source density, with exact quadrature (nodes x, weights w = quadrature weight x density)."""

    def __init__(self, name, logpdf, phi, edges, sampler, n=24, shift=0.0):
        self.name, self.logpdf, self.phi, self.sampler, self.shift = name, logpdf, phi, sampler, shift
        x, w = gl_pieces(edges, n)
        self.x = x + shift                                   # a shifted copy has mean `shift` (used to break E s = 0 on purpose)
        self.w = w * np.exp(logpdf(x))

    def E(self, f):
        return float(np.sum(self.w * f(self.x)))

    def kurt(self):
        m = self.E(lambda s: s); v = self.E(lambda s: (s - m) ** 2)
        return self.E(lambda s: (s - m) ** 4) / v ** 2 - 3

    def entropy(self):
        return -self.E(lambda s: self.logpdf(s - self.shift))

    def sample(self, size, rng):
        return self.sampler(size, rng) + self.shift


def make_sources():
    beta = math.sqrt(3) / math.pi
    a, b2 = 0.9, 1 - 0.81
    edges_exp = [-25, -10, -4, -1, 0, 1, 4, 10, 25]
    edges_log = [-22, -8, -3, -1, 0, 1, 3, 8, 22]
    S = {}
    S["uniform"] = Src("uniform", lambda x: np.full_like(x, -math.log(2 * math.sqrt(3))), None, [-math.sqrt(3), 0, math.sqrt(3)],
                       lambda n, r: r.uniform(-math.sqrt(3), math.sqrt(3), n), n=40)
    S["laplace"] = Src("Laplace", lambda x: -math.sqrt(2) * np.abs(x) - 0.5 * math.log(2), lambda x: math.sqrt(2) * np.sign(x), edges_exp,
                       lambda n, r: r.laplace(0, 1 / math.sqrt(2), n))
    S["logistic"] = Src("logistic", lambda x: -np.log(4 * beta) - 2 * np.log(np.cosh(x / (2 * beta))), lambda x: np.tanh(x / (2 * beta)) / beta, edges_log,
                        lambda n, r: r.logistic(0, beta, n))
    S["sech"] = Src("sech", lambda x: -np.log(2 * np.cosh(math.pi * x / 2)), lambda x: (math.pi / 2) * np.tanh(math.pi * x / 2), [-24, -8, -3, -1, 0, 1, 3, 8, 24],
                    lambda n, r: (2 / math.pi) * np.log(np.tan(math.pi * r.random(n) / 2)))

    def lp_bi(x):
        g1 = np.exp(-0.5 * (x + a) ** 2 / b2); g2 = np.exp(-0.5 * (x - a) ** 2 / b2)
        return np.log(0.5 * (g1 + g2) / math.sqrt(2 * math.pi * b2))

    def phi_bi(x):
        g1 = np.exp(-0.5 * (x + a) ** 2 / b2); g2 = np.exp(-0.5 * (x - a) ** 2 / b2)
        return ((x + a) * g1 + (x - a) * g2) / (b2 * (g1 + g2))

    S["bimodal"] = Src("bimodal", lp_bi, phi_bi, [-7, -3, -1, 0, 1, 3, 7], lambda n, r: r.choice([-a, a], n) + math.sqrt(b2) * r.normal(size=n))
    S["gauss"] = Src("Gaussian", lambda x: -0.5 * x * x - 0.5 * math.log(2 * math.pi), lambda x: x, [-12, -4, -1, 0, 1, 4, 12], lambda n, r: r.normal(size=n))
    return S


def grid2(s1, s2):
    """Product quadrature for two independent sources: points (2, N) and weights (N,)."""
    X1, X2 = np.meshgrid(s1.x, s2.x, indexing="ij")
    return np.stack([X1.ravel(), X2.ravel()]), np.outer(s1.w, s2.w).ravel()


NONLIN = {                                                       # (phi, Phi) with Phi' = phi: the model density is q ~ exp(-Phi)
    "cubic": (lambda y: y ** 3, lambda y: y ** 4 / 4),
    "tanh": (np.tanh, lambda y: np.log(np.cosh(y))),
}




def loss_C(C, pts, Wt, Phi):
    """L(C) = -log|det C| + E[Phi(y_1) + Phi(y_2)] (up to the constant H(s)): the loss (13.62) as a function of C = W A for the model q ~ exp(-Phi)."""
    y = C @ pts
    return -math.log(abs(np.linalg.det(C))) + float(np.sum(Wt * (Phi(y[0]) + Phi(y[1]))))


def skewed_source():
    """A zero-mean, unit-variance, skewed source: an asymmetric two-component Gaussian mixture, with its exact score phi = -(log k)'."""
    w, m1, m2, s1, s2 = 0.9, -0.2, 1.8, 0.4, 1.0
    mean = w * m1 + (1 - w) * m2
    sd = math.sqrt(w * (s1 ** 2 + m1 ** 2) + (1 - w) * (s2 ** 2 + m2 ** 2) - mean ** 2)
    a1, a2, b1, b2 = (m1 - mean) / sd, (m2 - mean) / sd, s1 / sd, s2 / sd
    def comps(x):
        return (w * np.exp(-0.5 * ((x - a1) / b1) ** 2) / (b1 * math.sqrt(2 * math.pi)), (1 - w) * np.exp(-0.5 * ((x - a2) / b2) ** 2) / (b2 * math.sqrt(2 * math.pi)))
    def logpdf(x):
        g1, g2 = comps(x); return np.log(g1 + g2)
    def phi(x):
        g1, g2 = comps(x)
        return -(g1 * (-(x - a1) / b1 ** 2) + g2 * (-(x - a2) / b2 ** 2)) / (g1 + g2)
    def sampler(n, r):
        z = r.random(n) < w
        return np.where(z, a1 + b1 * r.normal(size=n), a2 + b2 * r.normal(size=n))
    return Src("skewed", logpdf, phi, [-8, -4, -2, -1, -0.4, 0.4, 1, 2, 4, 8, 16, 30], sampler, n=30)


def second_order_moments(src1, src2, phis, scales, standard):
    """O(eta) stationary moments of the constant-gain rule C <- C - eta A(C s) C, A = H^(-1) vec(phi y^T - I) (standard) or vec(phi y^T - I) (plain), near C = I (sources rescaled so that the equilibrium is I):
    Sigma = E[vec X vec X^T]/eta solves J Sigma + Sigma J^T = Gamma, E[X]/eta = -J^(-1) (Q[Sigma]/2 + E[(J X) X]/eta); returns (J, E[X]/eta, Sigma, coefficient of eta in E[y_1 y_2])."""
    pts, Wt = grid2(src1, src2)
    pts = pts * np.array(scales)[:, None]
    vec = lambda M: M.reshape(-1)
    def Fm(C):
        y = C @ pts
        ph = np.stack([phis[0](y[0]), phis[1](y[1])])
        return (ph * Wt) @ y.T - np.eye(2)
    h = 1e-4
    H = np.zeros((4, 4))
    for k in range(4):
        e = np.zeros(4); e[k] = h
        H[:, k] = (vec(Fm(np.eye(2) + e.reshape(2, 2))) - vec(Fm(np.eye(2) - e.reshape(2, 2)))) / (2 * h)
    Hi = np.linalg.inv(H)
    Mm = (lambda X: Hi @ vec(Fm(np.eye(2) + X.reshape(2, 2)))) if standard else (lambda X: vec(Fm(np.eye(2) + X.reshape(2, 2))))
    J = np.zeros((4, 4))
    for k in range(4):
        e = np.zeros(4); e[k] = h
        J[:, k] = (Mm(e) - Mm(-e)) / (2 * h)
    Q = np.zeros((4, 4, 4)); h2 = 2e-3; z4 = np.zeros(4)
    for k in range(4):
        for l in range(k, 4):
            ek = np.zeros(4); ek[k] = h2; el = np.zeros(4); el[l] = h2
            if k == l:
                Q[:, k, k] = (Mm(ek) - 2 * Mm(z4) + Mm(-ek)) / h2 ** 2
            else:
                Q[:, k, l] = Q[:, l, k] = (Mm(ek + el) - Mm(ek - el) - Mm(-ek + el) + Mm(-ek - el)) / (4 * h2 ** 2)
    ph = np.stack([phis[0](pts[0]), phis[1](pts[1])])
    F0 = np.stack([ph[0] * pts[0] - 1, ph[0] * pts[1], ph[1] * pts[0], ph[1] * pts[1] - 1])
    A0 = Hi @ F0 if standard else F0
    Gam = (A0 * Wt) @ A0.T
    L = np.kron(np.eye(4), J) + np.kron(J, np.eye(4))
    Sig = np.linalg.solve(L, Gam.reshape(-1)).reshape(4, 4)
    EJX = np.zeros(4)
    for i in range(2):
        for j in range(2):
            EJX[2 * i + j] = sum(J[2 * i + p, kl] * Sig[kl, 2 * p + j] for p in range(2) for kl in range(4))
    EX = -np.linalg.solve(J, 0.5 * np.einsum("ikl,kl->i", Q, Sig) + EJX)
    V12 = EX[1] + EX[2] + sum(Sig[k, 2 + k] for k in range(2))
    return J, EX, Sig, V12, H


def check_super_efficiency(S):
    print("   Theorem 13.1 ('super efficiency'): V_ij = E[y_i y_j], i != j, of order eta^2 for a fixed gain eta when E[phi_i(s_i)] = 0 (13.106). Order-eta bookkeeping for the rule C <- C - eta A(C s) C near the solution"
          " (A = phi y^T - I, 'plain', or H^(-1) of it, the 'standard' estimating function with J = identity; sources: a skewed one and a logistic one):")
    sk, lg = skewed_source(), S["logistic"]
    cs = []
    for src in (sk, lg):
        lo, hi = 0.02, 40.0
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if src.E(lambda t: np.tanh(mid * t) * mid * t) - 1 > 0: hi = mid
            else: lo = mid
        cs.append(lo)
    print(f"      skewed source: skewness {sk.E(lambda s_: s_ ** 3):.3f}, E[phi_score] = {sk.E(sk.phi):+.1e}, E[tanh(c s)] = {sk.E(lambda s_: np.tanh(cs[0] * s_)):+.4f} (so (13.106) holds for the score and fails for tanh)")
    print(f"      {'phi':8s} {'estimating function':20s} {'E[X_11]/eta':>12s} {'E[X_22]/eta':>12s} {'E[X_12+X_21]/eta':>17s} {'E[y1 y2]/eta':>14s}")
    rows = []
    for lab, phis, scl in (("score", (sk.phi, lg.phi), (1.0, 1.0)), ("tanh", (np.tanh, np.tanh), tuple(cs))):
        for std in (False, True):
            J, EX, Sig, V12, H = second_order_moments(sk, lg, phis, scl, std)
            rows.append((lab, std, EX, V12))
            print(f"      {lab:8s} {'standard (J = I)' if std else 'plain':20s} {EX[0]:>12.4f} {EX[3]:>12.4f} {EX[1] + EX[2]:>17.1e} {V12:>14.1e}")
    Js = second_order_moments(sk, lg, (sk.phi, lg.phi), (1.0, 1.0), True)[0]; Jp = second_order_moments(sk, lg, (sk.phi, lg.phi), (1.0, 1.0), False)[0]
    print(f"      Jacobian J = E[dA/dX]: eigenvalues {np.round(np.sort(np.linalg.eigvals(Jp).real), 3).tolist()} for the plain function, {np.round(np.sort(np.linalg.eigvals(Js).real), 6).tolist()} for the standard one (13.102), (13.103): J~ = identity, all directions converge at the same rate")
    print("      so the O(eta) bias of the diagonal entries (the scales) is large, while the symmetric off-diagonal part, and hence E[y_1 y_2], has no O(eta) term in any of the four cases: the cross-covariance of the noise with the scale components vanishes because E[s_j] = 0, whether or not E[phi_i(s_i)] = 0")
    STORE["superefficiency"] = rows
    # the algorithm: constant gain eta, stationary averages
    print("   direct check, plain rule with phi = tanh on these sources (so (13.106) fails): 1500 chains of the online rule, stationary time averages of C - I and of C C^T, against eta x (the order-eta coefficients above):")
    outs = []
    for eta, Rn, burn, avg in ((0.02, 1500, 400, 800), (0.01, 1500, 800, 1600)):
        rng = rng_for(35)
        phi = np.tanh
        one = np.ones(Rn)
        c11, c12, c21, c22 = one.copy(), 0 * one, 0 * one, one.copy()
        acc = np.zeros((5, Rn))
        for t in range(burn + avg):
            s1 = cs[0] * sk.sample(Rn, rng); s2 = cs[1] * lg.sample(Rn, rng)
            y1 = c11 * s1 + c12 * s2; y2 = c21 * s1 + c22 * s2
            p1, p2 = phi(y1), phi(y2)
            a11 = p1 * y1 - 1; a12 = p1 * y2; a21 = p2 * y1; a22 = p2 * y2 - 1
            c11, c12, c21, c22 = c11 - eta * (a11 * c11 + a12 * c21), c12 - eta * (a11 * c12 + a12 * c22), c21 - eta * (a21 * c11 + a22 * c21), c22 - eta * (a21 * c12 + a22 * c22)
            if t >= burn:
                acc += np.stack([c11 - 1, c22 - 1, c12 + c21, c11 * c21 + c12 * c22, y1 * y2])
        per_chain = acc / avg
        mean = per_chain.mean(axis=1); se = per_chain.std(axis=1) / math.sqrt(Rn)
        pred = rows[2][2]
        outs.append((eta, mean, se))
        print(f"      eta = {eta}: E[X_11] = {mean[0]:+.4f} +- {se[0]:.4f} (predicted {eta * pred[0]:+.4f}), E[X_22] = {mean[1]:+.4f} +- {se[1]:.4f} (predicted {eta * pred[3]:+.4f}); E[X_12 + X_21] = {mean[2]:+.5f} +- {se[2]:.5f};"
              f" E[y_1 y_2] = {mean[3]:+.5f} +- {se[3]:.5f} (time average of (C C^T)_12), {mean[4]:+.5f} +- {se[4]:.5f} (of the realised y_1 y_2)")
    mean = outs[-1][1]; eta = outs[-1][0]
    print("      the diagonal bias follows the O(eta) prediction (relative gap " + ", ".join(f"{abs(o[1][0] / (o[0] * rows[2][2][0]) - 1):.2f}" for o in outs) + " at eta = " + ", ".join(str(o[0]) for o in outs)
          + "), while the cross term E[y_1 y_2] stays at the level of the statistical error, well below eta^2-type size: consistent with the theorem's order, and independent of (13.106)")
    STORE["superefficiency_mc"] = (mean, eta)


def skew_tanh_mean():
    """E[tanh(s)] for s = exponential(1) - 1 (zero mean, skewed): quadrature."""
    x, w = gl_pieces([0, 2, 8, 40], 40)
    return float(np.sum(w * np.exp(-x) * np.tanh(x - 1.0)))


def check_ica():
    head("2. Independent component analysis (section 13.2)")
    S = make_sources()
    print("   sources (zero mean, unit variance), exact Gauss-Legendre quadrature; excess kurtosis: " + ", ".join(f"{k} {S[k].kurt():+.3f}" for k in ("uniform", "bimodal", "gauss", "logistic", "sech", "laplace"))
          + f"   (check: E s^2 = {S['logistic'].E(lambda s: s * s):.12f}, {S['bimodal'].E(lambda s: s * s):.12f})")
    STORE["ica_kurt"] = {k: S[k].kurt() for k in S}
    # ---- 2a. the density of y = W x, (13.52)-(13.53): box probabilities by quadrature of the formula against sampling
    rng = rng_for(31)
    A = np.array([[1.0, 0.6], [-0.4, 1.2]]); Wm = np.array([[0.9, -0.2], [0.35, 0.7]]); C = Wm @ A
    s1, s2 = S["logistic"], S["bimodal"]
    n = 4_000_000
    Ci = np.linalg.inv(C)
    ys = np.concatenate([C @ np.stack([s1.sample(CHUNK * 2, rng), s2.sample(CHUNK * 2, rng)]) for _ in range(n // (CHUNK * 2))], axis=1)
    def pY(y1, y2):
        z = Ci @ np.stack([y1.ravel(), y2.ravel()])
        return np.exp(s1.logpdf(z[0]) + s2.logpdf(z[1])).reshape(y1.shape) / abs(np.linalg.det(C))
    xg, wg = np.polynomial.legendre.leggauss(80)
    out = []
    for (lo1, hi1, lo2, hi2) in ((-0.5, 0.5, 0.0, 1.0), (0.2, 2.0, -1.5, 0.3), (-3.0, -1.0, -2.0, 2.0)):
        u = 0.5 * (hi1 - lo1) * xg + 0.5 * (hi1 + lo1); v = 0.5 * (hi2 - lo2) * xg + 0.5 * (hi2 + lo2)
        U, V_ = np.meshgrid(u, v, indexing="ij")
        q = float(np.sum(np.outer(wg, wg) * pY(U, V_)) * 0.25 * (hi1 - lo1) * (hi2 - lo2))
        inside = (ys[0] > lo1) & (ys[0] < hi1) & (ys[1] > lo2) & (ys[1] < hi2)
        p = float(inside.mean()); se = math.sqrt(p * (1 - p) / n)
        out.append((q, p, se))
    print("   (13.52) p_Y(y) = |WA|^(-1) k(A^(-1) W^(-1) y), logistic and bimodal sources, A = [[1, .6], [-.4, 1.2]], W = [[.9, -.2], [.35, .7]]: probability of three boxes by quadrature of the formula | by 4e6 samples: "
          + "; ".join(f"{q:.5f} | {p:.5f} +- {se:.5f}" for q, p, se in out))
    # ---- 2b. entropy and loss, (13.64)-(13.66), by integrating in y-space on a grid (independent of the s-space formula)
    q1 = q2 = S["sech"]
    g = np.linspace(-12, 12, 1201); dg = g[1] - g[0]
    Y1, Y2 = np.meshgrid(g, g, indexing="ij")
    P = pY(Y1, Y2)
    Hy = -float(np.sum(P * np.log(np.maximum(P, 1e-300))) * dg * dg)
    HS = s1.entropy() + s2.entropy()
    print(f"   (13.66) H(Y) = H(X) + log|det W|: entropy of y from the grid {Hy:.6f}; H(s1) + H(s2) + log|det A| + log|det W| = {HS:.4f} + {math.log(abs(np.linalg.det(A))):.4f} + {math.log(abs(np.linalg.det(Wm))):.4f} = {HS + math.log(abs(np.linalg.det(A))) + math.log(abs(np.linalg.det(Wm))):.6f}")
    gx = np.linspace(-20, 20, 1601); dgx = gx[1] - gx[0]
    X1g, X2g = np.meshgrid(gx, gx, indexing="ij")
    Zg = Wm @ np.stack([X1g.ravel(), X2g.ravel()])
    with np.errstate(divide="ignore"):
        I88 = float(np.sum(np.exp(s1.logpdf(Zg[0]) + s2.logpdf(Zg[1]))) * dgx * dgx)
    detW = abs(float(np.linalg.det(Wm)))
    print(f"   (13.88) as printed, p(x, W, k) = prod_i k_i((W x)_i) without the factor |det W|: its integral over x (grid) is {I88:.6f} = 1/|det W| = {1 / detW:.6f}, not 1; with the factor it is {detW * I88:.6f}")
    Eq = float(np.sum(P * (q1.logpdf(Y1) + q2.logpdf(Y2))) * dg * dg)
    KL_grid = float(np.sum(P * (np.log(np.maximum(P, 1e-300)) - q1.logpdf(Y1) - q2.logpdf(Y2))) * dg * dg)
    pts, Wt = grid2(s1, s2)
    KL_form = -HS - math.log(abs(np.linalg.det(C))) - float(np.sum(Wt * (q1.logpdf((C @ pts)[0]) + q2.logpdf((C @ pts)[1]))))
    print(f"   (13.62)-(13.64): D_KL[p_Y : q] with q the product of two sech densities: y-space integral {KL_grid:.6f} = -H(Y) - E log q = {-Hy - Eq:.6f}; the s-space expression -H(s) - log|det WA| - E log q(WAs) gives {KL_form:.6f}")
    STORE["ica_loss_check"] = (KL_grid, KL_form)
    # ---- 2c. S_W and S_I at the solution: Fisher geometry in relative coordinates
    sr = [S["logistic"], S["bimodal"]]
    pts, Wt = grid2(*sr)
    def score(i, j):                                          # tangent of S_W along dX_ij (C = I): phi_i(s_i) s_j - delta_ij
        return sr[i].phi(pts[i]) * pts[j] - (1.0 if i == j else 0.0)
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Gm = np.array([[float(np.sum(Wt * score(*a_) * score(*b_))) for b_ in dirs] for a_ in dirs])
    # orthonormal basis of T S_I: zero-mean functions of one variable (polynomials up to degree 9 and the scale directions themselves)
    basis = []
    for i in range(2):
        cols = [pts[i] ** d for d in range(1, 10)] + [sr[i].phi(pts[i]) * pts[i] - 1.0, np.tanh(pts[i]), np.tanh(2 * pts[i])]
        cols = [c - float(np.sum(Wt * c)) for c in cols]
        basis += cols
    B = np.array(basis).T                                     # (N, nb)
    sw = np.sqrt(Wt)[:, None]
    Qb, _ = np.linalg.qr(B * sw, mode="reduced")
    cos2 = []
    for d_ in dirs:
        v = score(*d_) * np.sqrt(Wt)
        proj = Qb @ (Qb.T @ v)
        cos2.append(float(proj @ proj / (v @ v)))
    print(f"   Fisher metric of S_W at W = A^(-1) in the relative coordinates (dX_11, dX_12, dX_21, dX_22), logistic and bimodal sources: diag {np.round(np.diag(Gm), 4).tolist()}, the off-diagonal entries between a diagonal and an off-diagonal direction vanish ({np.abs(Gm[[0, 3]][:, [1, 2]]).max():.1e})")
    print(f"   squared cosine between each direction of T S_W and the tangent space of S_I (all zero-mean functions of single variables): {np.round(cos2, 12).tolist()}: the two off-diagonal directions are orthogonal to S_I ({max(cos2[1], cos2[2]):.1e}),"
          " the two scale directions lie inside S_I (1.0): S_W meets S_I orthogonally only modulo the scales, as the book says it neglects")
    # for q = k the Hessian of L at the solution is the Fisher information of S_W: positive definite, so A^(-1) is a minimum ("q close to k")
    dphi_fd = lambda src, x_: (src.phi(x_ + 1e-6) - src.phi(x_ - 1e-6)) / 2e-6
    mq = [src.E(lambda t_, src=src: dphi_fd(src, t_)) for src in sr]; v2q = [src.E(lambda t_: t_ * t_) for src in sr]
    muq = [src.E(lambda t_, src=src: dphi_fd(src, t_) * t_ * t_) + 1 for src in sr]
    Hq = np.array([[muq[0], 0, 0, 0], [0, mq[0] * v2q[1], 1, 0], [0, 1, mq[1] * v2q[0], 0], [0, 0, 0, muq[1]]])
    print(f"   for the matched model q = k the Hessian of L at the solution, [[E[phi' s^2] + 1, ...], pair blocks [[m_1 s_2^2, 1], [1, m_2 s_1^2]]], equals the Fisher Gram matrix of S_W above to {np.abs(Hq - Gm).max():.1e}"
          f" (eigenvalues {np.round(np.linalg.eigvalsh(Hq), 4).tolist()}, all positive): when q = k, A^(-1) is a minimum, as the book says for q close to k")
    # a non-zero mean breaks the orthogonality
    srm = [S["logistic"], Src("logistic+0.5", S["logistic"].logpdf, S["logistic"].phi, [-22, -8, -3, -1, 0, 1, 3, 8, 22], S["logistic"].sampler, shift=0.5)]
    srm[1].logpdf = S["logistic"].logpdf
    ptsm, Wm_ = grid2(*srm)
    # tangent along dX_12 for the model with s_2 shifted: phi_1(s_1) s_2 ; inner product with the single-variable function s_1 (zero mean) is E[phi_1 s_1] E[s_2]
    sc12 = srm[0].phi(ptsm[0]) * ptsm[1]
    ip = float(np.sum(Wm_ * sc12 * ptsm[0]))
    print(f"   if the second source has mean 0.5 instead of 0 (assumption (13.50) violated) the tangent along dX_12 has inner product E[phi_1 s_1] E[s_2] = {ip:.4f} with the S_I direction s_1: no longer orthogonal")
    # ---- 2d. critical point and scale for a mismatched model q: gradient at C = I
    print("   gradient of L at W = A^(-1) for the model q_i: off-diagonal entries E[phi_i(s_i)] E[s_j] vanish, the diagonal entries E[phi_i(s_i) s_i] - 1 vanish only if q is matched in scale:")
    print(f"      {'sources (true k)':26s} {'model q':10s} {'grad diag':>22s} {'scale c with E[phi(c s) c s] = 1':>34s} {'argmin of L along the scale':>28s}")
    rows = []
    for (n1, n2, qn) in (("logistic", "bimodal", "match"), ("logistic", "bimodal", "sech"), ("logistic", "bimodal", "tanh"), ("laplace", "uniform", "cubic")):
        sa, sb = S[n1], S[n2]; ptsl, Wl = grid2(sa, sb)
        if qn == "match":
            phis = [sa.phi, sb.phi]; Phis = [lambda y, s_=sa: -s_.logpdf(y), lambda y, s_=sb: -s_.logpdf(y)]
        elif qn == "sech":
            phis = [S["sech"].phi] * 2; Phis = [lambda y: -S["sech"].logpdf(y)] * 2
        else:
            phis = [NONLIN[qn][0]] * 2; Phis = [NONLIN[qn][1]] * 2
        gd = [float(np.sum(Wl * phis[i](ptsl[i]) * ptsl[i])) - 1.0 for i in range(2)]
        off = [float(np.sum(Wl * phis[0](ptsl[0]) * ptsl[1])), float(np.sum(Wl * phis[1](ptsl[1]) * ptsl[0]))]
        cs, cm = [], []
        for i, src in enumerate((sa, sb)):
            f = lambda c: src.E(lambda s: phis[i](c * s) * c * s) - 1.0
            lo, hi = 0.05, 20.0
            for _ in range(60):
                mid = math.sqrt(lo * hi)
                if f(mid) > 0: hi = mid
                else: lo = mid
            cs.append(lo)
            Lc = lambda c: -math.log(c) + src.E(lambda s: Phis[i](c * s))
            a_, b_ = lo / 3, lo * 3
            gr = (math.sqrt(5) - 1) / 2
            for _ in range(70):
                c1 = b_ - gr * (b_ - a_); c2 = a_ + gr * (b_ - a_)
                if Lc(c1) < Lc(c2): b_ = c2
                else: a_ = c1
            cm.append(0.5 * (a_ + b_))
        rows.append((n1, n2, qn, gd, off, cs, cm))
        print(f"      {n1 + ' + ' + n2:26s} {qn:10s} {np.round(gd, 4).tolist()!s:>22s} {np.round(cs, 4).tolist()!s:>34s} {np.round(cm, 4).tolist()!s:>28s}   (off-diagonal gradient entries {max(abs(off[0]), abs(off[1])):.0e})")
    STORE["ica_scale"] = rows
    print("   so for q != k the point W = A^(-1) is not critical along the diagonal (scale) directions; the critical point is D A^(-1) with the q-dependent scale D, and the minimiser of L along the scale is that scale (columns 4 and 5 agree)")

    # ---- 2e. the natural gradient (13.67)-(13.87): finite differences and the signs
    rng = rng_for(32)
    d = 3
    Wn = np.eye(d) + 0.3 * rng.normal(size=(d, d)); xv = rng.normal(size=d)
    phi, Phi = NONLIN["tanh"]
    def inst(Wq):
        y = Wq @ xv
        return -math.log(abs(np.linalg.det(Wq))) + float(np.sum(Phi(y)))
    h = 1e-6; G_fd = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Wp = Wn.copy(); Wm_ = Wn.copy(); Wp[i, j] += h; Wm_[i, j] -= h
            G_fd[i, j] = (inst(Wp) - inst(Wm_)) / (2 * h)
    y = Wn @ xv
    G_form = -np.linalg.inv(Wn).T + np.outer(phi(y), xv)
    nat_form = -(np.eye(d) - np.outer(phi(y), y)) @ Wn
    print(f"   instantaneous loss l(y, W) = -log|W| - log q(y) (13.67), q_i ~ 1/cosh, a random 3 x 3 W: finite-difference gradient against -W^(-T) + phi(y) x^T (from (13.73)): {fro(G_fd - G_form):.1e}; "
          f"natural gradient (gradient times W^T W, (13.77)-(13.87)) against -(I - phi(y) y^T) W: {fro(G_form @ Wn.T @ Wn - nat_form):.1e}")
    eps = 1e-4
    dl_print = inst(Wn - eps * (np.eye(d) - np.outer(phi(y), y)) @ Wn) - inst(Wn)
    dl_corr = inst(Wn + eps * (np.eye(d) - np.outer(phi(y), y)) @ Wn) - inst(Wn)
    pred = eps * fro(np.eye(d) - np.outer(phi(y), y)) ** 2
    print(f"   one step of size {eps}: with the sign printed in (13.86), Delta X = -eps (I - phi(y) y^T), the loss changes by {dl_print:+.3e} (increase: it is the ascent step); with Delta X = +eps (I - phi(y) y^T) it changes by {dl_corr:+.3e} "
          f"(first order {-pred:+.3e} = -eps |I - phi y^T|_F^2): the descent step is W <- W + eps (I - phi(y) y^T) W")
    STORE["ng_signs"] = (dl_print, dl_corr)

    # ---- 2f. equivariance: trajectories of C = W A do not depend on A for the natural gradient
    su = S["uniform"]
    C0 = np.array([[0.9, 0.25], [-0.2, 0.8]])
    A1 = np.array([[1.0, 0.5], [0.3, 1.0]]); A2 = np.array([[3.0, -2.0], [0.5, 0.4]])
    print(f"   equivariance: uniform sources, cubic phi, eps = 0.01, the same source samples s_t for two mixing matrices (cond(A1) = {np.linalg.cond(A1):.1f}, cond(A2) = {np.linalg.cond(A2):.1f}), W(0) = C(0) A^(-1) with the same C(0):")
    def xerr(C):                                                    # cross-talk index: zero iff C is a scaled permutation
        a2 = C ** 2
        return float(np.sum(a2.sum(1) / a2.max(1) - 1) + np.sum(a2.sum(0) / a2.max(0) - 1))
    T = 3000; eps = 0.01
    rngs = rng_for(33)
    stream = np.stack([su.sample(T, rngs), su.sample(T, rngs)])
    res = {}
    for algo in ("natural", "ordinary"):
        trajs = []
        for A_ in (A1, A2):
            Wc = C0 @ np.linalg.inv(A_); Cs_ = []
            for t in range(T):
                xt = A_ @ stream[:, t]; yt = Wc @ xt
                if algo == "natural":
                    Wc = Wc + eps * (np.eye(2) - np.outer(yt ** 3, yt)) @ Wc
                else:
                    Wc = Wc + eps * (np.linalg.inv(Wc).T - np.outer(yt ** 3, xt))
                Cs_.append(Wc @ A_)
                if not np.all(np.isfinite(Wc)) or np.abs(Wc).max() > 1e8:
                    break
            trajs.append(np.array(Cs_))
        res[algo] = trajs
    n_ = min(len(res["natural"][0]), len(res["natural"][1]))
    dnat = float(np.max(np.abs(res["natural"][0][:n_] - res["natural"][1][:n_])))
    print(f"      natural gradient (13.87): max over t of |C_1(t) - C_2(t)| = {dnat:.1e}; cross-talk index at t = {T}: {xerr(res['natural'][0][-1]):.4f} (A1), {xerr(res['natural'][1][-1]):.4f} (A2): identical dynamics, whatever the mixing")
    ordn = []
    for tr in res["ordinary"]:
        ordn.append((len(tr), xerr(tr[-1]) if np.all(np.isfinite(tr[-1])) else float("nan")))
    n1, n2 = ordn
    print(f"      ordinary gradient (W <- W + eps(W^(-T) - phi(y) x^T)): after {n1[0]} steps the index is {n1[1]:.4f} for A1; for A2 it ran {n2[0]} steps and ended at index {n2[1]:.4g}"
          + (" (diverged)" if n2[0] < T else "") + ": speed and even stability depend on A (dC = eps (C^(-T) - phi(y) s^T) A^T A)")
    STORE["equiv"] = dict(nat=res["natural"], ord=res["ordinary"], T=T)
    # speed: steps to reach index < 0.05 for natural gradient (same for both) vs ordinary gradient
    def first_below(tr, thr=0.05):
        for i, c in enumerate(tr):
            if np.all(np.isfinite(c)) and xerr(c) < thr:
                return i
        return None
    print(f"      steps until cross-talk index < 0.05: natural {first_below(res['natural'][0])} (A1), {first_below(res['natural'][1])} (A2); ordinary {first_below(res['ordinary'][0])} (A1), {first_below(res['ordinary'][1])} (A2)")

    # ---- 2g. stability of the learning fixed point: Hessian of L, closed form, and the algorithm itself
    print("   stability of the separating point under the natural-gradient rule dC = eps (I - phi(y) y^T) C (C = W A, sources rescaled by the equilibrium scales c_i, E[phi(c s) c s] = 1):")
    print("   linearisation: for the pair (i, j) the block of the Hessian of L in the relative coordinates (dX_ij, dX_ji) is [[m_i s_j^2, 1], [1, m_j s_i^2]], with m_i = E[phi'(s_i)], s_i^2 = E[s_i^2], and the diagonal entries are E[phi'(s_i) s_i^2] + 1;"
          " with kappa_i = m_i s_i^2 the point is a minimum iff kappa_i > 0 and kappa_1 kappa_2 > 1")
    dphi = {"cubic": lambda y: 3 * y ** 2, "tanh": lambda y: 1 - np.tanh(y) ** 2}
    cases = [("uniform", "uniform", "cubic"), ("uniform", "bimodal", "cubic"), ("uniform", "logistic", "cubic"), ("uniform", "sech", "cubic"), ("uniform", "laplace", "cubic"),
             ("logistic", "logistic", "cubic"), ("laplace", "laplace", "cubic"), ("logistic", "laplace", "tanh"), ("laplace", "laplace", "tanh"), ("uniform", "uniform", "tanh")]
    print(f"      {'sources':20s} {'phi':6s} {'kappa_1':>8s} {'kappa_2':>8s} {'k1 k2':>7s} {'min eig H':>10s} {'|H_fd - H|':>11s} {'|J + H|':>9s}  verdict")
    stab = []
    for n1, n2, nl in cases:
        a_, b_ = S[n1], S[n2]
        phi, Phi = NONLIN[nl]
        cs = []
        for src in (a_, b_):
            lo, hi = 0.02, 40.0
            for _ in range(60):
                mid = math.sqrt(lo * hi)
                if src.E(lambda t: phi(mid * t) * mid * t) - 1 > 0: hi = mid
                else: lo = mid
            cs.append(lo)
        pts, Wt = grid2(a_, b_)
        pts = pts * np.array(cs)[:, None]                              # rescaled sources: the equilibrium is C = I
        v2 = [float(np.sum(Wt * pts[i] ** 2)) for i in range(2)]
        m = [float(np.sum(Wt * dphi[nl](pts[i]))) for i in range(2)]
        mu = [float(np.sum(Wt * dphi[nl](pts[i]) * pts[i] ** 2)) + 1 for i in range(2)]
        a12, b12 = m[0] * v2[1], m[1] * v2[0]
        Hc = np.array([[mu[0], 0, 0, 0], [0, a12, 1, 0], [0, 1, b12, 0], [0, 0, 0, mu[1]]])
        # the Jacobian of F((I + X)) at X = 0 (the learning rule's mean flow), and the Hessian of the loss by second differences
        def Fm(Cq):
            y = Cq @ pts
            return np.eye(2) - (phi(y) * Wt) @ y.T
        h = 1e-5; J = np.zeros((4, 4))
        for k_ in range(4):
            Xp = np.zeros(4); Xp[k_] = h
            J[:, k_] = ((Fm(np.eye(2) + Xp.reshape(2, 2)) - Fm(np.eye(2) - Xp.reshape(2, 2))) / (2 * h)).ravel()
        Lf = lambda X: loss_C(np.eye(2) + X.reshape(2, 2), pts, Wt, Phi)
        h2 = 2e-3; Hfd = np.zeros((4, 4)); L0 = Lf(np.zeros(4))
        for i in range(4):
            for j in range(i, 4):
                ei = np.zeros(4); ei[i] = h2; ej = np.zeros(4); ej[j] = h2
                if i == j:
                    Hfd[i, i] = (Lf(ei) - 2 * L0 + Lf(-ei)) / h2 ** 2
                else:
                    Hfd[i, j] = Hfd[j, i] = (Lf(ei + ej) - Lf(ei - ej) - Lf(-ei + ej) + Lf(-ei - ej)) / (4 * h2 ** 2)
        ev = np.linalg.eigvalsh(Hc)
        kap = (m[0] * v2[0], m[1] * v2[1])
        verdict = "stable (minimum)" if ev[0] > 1e-9 else ("marginal" if abs(ev[0]) <= 1e-9 else "unstable")
        stab.append(dict(n1=n1, n2=n2, nl=nl, kap=kap, hmin=ev[0], cs=cs, Hc=Hc, k=(a_.kurt(), b_.kurt())))
        print(f"      {n1 + ' + ' + n2:20s} {nl:6s} {kap[0]:8.4f} {kap[1]:8.4f} {kap[0] * kap[1]:7.4f} {ev[0]:+10.4f} {np.abs(Hfd - Hc).max():11.1e} {np.abs(J + Hc).max():9.1e}  {verdict}")
    STORE["stab"] = stab
    print("      (columns: H_fd = Hessian of the loss L(C) by second differences on the quadrature grid; J = Jacobian of the mean update by differences; both agree with the closed form.)"
          " For the cubic phi, kappa = 3/(3 + excess kurtosis), so the condition kappa_1 kappa_2 > 1 reads (3 + k_1)(3 + k_2) < 9, e.g. two uniform sources, or a uniform and a logistic one, but not a uniform and a Laplace one; the sech density (k = 2) with a uniform one sits exactly on the boundary (eigenvalue 0)")
    print("      the unstable direction is the antisymmetric pair (dX_12, dX_21) = (1, -1): a rotation of the two outputs, the direction that second-order statistics leave free (13.19)")
    # the learning rule itself: start along the weakest direction of the pair block; paired runs C = I +- delta v with common samples (noise and all even-order terms cancel)
    print("   the algorithm itself: paired online runs started at C = I + delta v and C = I - delta v with the same source samples (v: the weakest eigenvector of the pair block, delta = 0.02, R = 5000 runs);"
          " the measured linear response [proj(X+) - proj(X-)]/(2 delta) at time t = 2, run with step eps = 0.01 and 0.005 and extrapolated to eps -> 0 (the O(eps) deviation is the noise-induced correction of stochastic approximation), against exp(-h_min t) (the linear extrapolation in eps leaves a residual of about a percent):")
    rngo = rng_for(34)
    out = []
    wanted = (("uniform", "uniform", "cubic"), ("logistic", "laplace", "tanh"), ("uniform", "uniform", "tanh"))
    for rec_ in stab:
        if (rec_["n1"], rec_["n2"], rec_["nl"]) not in wanted:
            continue
        Rn, delta = 5000, 0.02
        a_, b_ = S[rec_["n1"]], S[rec_["n2"]]; cs = rec_["cs"]; phi = NONLIN[rec_["nl"]][0]
        blk = rec_["Hc"][1:3, 1:3]; w_, V_ = np.linalg.eigh(blk); v = V_[:, 0]
        one = np.ones(Rn)
        resps = {}
        for eps, steps in ((0.01, 200), (0.005, 400)):
            Cp = [one, delta * v[0] * one, delta * v[1] * one, one]; Cm = [one, -delta * v[0] * one, -delta * v[1] * one, one]
            def step(C, s1, s2):
                c11, c12, c21, c22 = C
                y1 = c11 * s1 + c12 * s2; y2 = c21 * s1 + c22 * s2
                p1, p2 = phi(y1), phi(y2)
                f11 = 1 - p1 * y1; f12 = -p1 * y2; f21 = -p2 * y1; f22 = 1 - p2 * y2
                return [c11 + eps * (f11 * c11 + f12 * c21), c12 + eps * (f11 * c12 + f12 * c22), c21 + eps * (f21 * c11 + f22 * c21), c22 + eps * (f21 * c12 + f22 * c22)]
            for t in range(steps):
                s1 = cs[0] * a_.sample(Rn, rngo); s2 = cs[1] * b_.sample(Rn, rngo)
                Cp = step(Cp, s1, s2); Cm = step(Cm, s1, s2)
            resp = ((v[0] * Cp[1] + v[1] * Cp[2]) - (v[0] * Cm[1] + v[1] * Cm[2])) / (2 * delta)
            resps[eps] = (float(np.mean(resp)), float(np.std(resp)) / math.sqrt(Rn))
        ext = 2 * resps[0.005][0] - resps[0.01][0]; se = math.sqrt(4 * resps[0.005][1] ** 2 + resps[0.01][1] ** 2)
        pred = math.exp(-rec_["hmin"] * 2.0)
        out.append((rec_["n1"], rec_["n2"], rec_["nl"], rec_["hmin"], pred, resps[0.01][0], resps[0.005][0], ext, se))
        print(f"      {rec_['n1'] + ' + ' + rec_['n2']:20s} {rec_['nl']:6s} h_min = {rec_['hmin']:+.4f}: predicted {pred:.4f}; measured {resps[0.01][0]:.4f} (eps 0.01), {resps[0.005][0]:.4f} (eps 0.005), extrapolated {ext:.4f} +- {se:.4f} ({100 * (ext / pred - 1):+.1f}% from the prediction)")
    # conditions (13.106), (13.109), (13.110)
    print("   the conditions under (13.106): (1) phi_i = -(log k_i)' gives E[phi_i(s_i)] = -integral of k_i' = 0 ((13.109)); (2) k_i even and phi_i odd gives 0 ((13.110)); by quadrature:")
    print("      " + "; ".join(f"{k}: E[phi_score] = {S[k].E(S[k].phi):+.1e}" for k in ("logistic", "sech", "laplace", "bimodal")) + f"; E[tanh(s)] = {S['logistic'].E(np.tanh):+.1e} (logistic, even k, odd phi),"
          f" and for a skewed source (exponential, shifted to mean 0) E[tanh(s)] would be {skew_tanh_mean():+.4f}, so (13.106) fails there")
    STORE["stab_runs"] = out
    check_super_efficiency(S)


# ================================================================== 3. non-negative matrix factorisation (13.3)

def frob_loss(X, A, S):
    return 0.5 * float(np.sum((X - A @ S) ** 2))


def gkl(X, Y):
    """Generalised Kullback-Leibler divergence sum x log(x/y) - x + y."""
    return float(np.sum(X * np.log(X / Y) - X + Y))


def check_nmf():
    head("3. Non-negative matrix factorisation (section 13.3)")
    rng = rng_for(41)
    n, r, T = 6, 3, 40
    A_true = rng.random((n, r)) + 0.05
    S_true = rng.random((r, T)) ** 2 + 0.01
    X = A_true @ S_true
    A0 = rng.random((n, r)) + 0.1; S0 = rng.random((r, T)) + 0.1
    # ---- 3a. the gradients (13.123) and the printed KL (13.120)
    gA = -X @ S0.T + A0 @ S0 @ S0.T; gS = -A0.T @ X + A0.T @ A0 @ S0
    h = 1e-6; fa = np.zeros_like(A0); fs = np.zeros_like(S0)
    for i in range(n):
        for j in range(r):
            Ap = A0.copy(); Am = A0.copy(); Ap[i, j] += h; Am[i, j] -= h
            fa[i, j] = (frob_loss(X, Ap, S0) - frob_loss(X, Am, S0)) / (2 * h)
    for i in range(r):
        for j in range(T):
            Sp = S0.copy(); Sm = S0.copy(); Sp[i, j] += h; Sm[i, j] -= h
            fs[i, j] = (frob_loss(X, A0, Sp) - frob_loss(X, A0, Sm)) / (2 * h)
    print(f"   X = A S with A 6 x 3, S 3 x 40 non-negative (exact, no noise); gradients (13.123) -X S^T + A S S^T and -A^T X + A^T A S against finite differences of L = (1/2)|X - A S|^2: {np.abs(gA - fa).max():.1e}, {np.abs(gS - fs).max():.1e}")
    a_ = np.array([1.0, 2.0, 0.5]); b_ = np.array([1.0, 5.0, 0.5])
    printed = lambda a, b: float(np.sum(a * np.log(a / b) - a - b)); right = lambda a, b: float(np.sum(a * np.log(a / b) - a + b))
    print(f"   the 'KL-divergence' (13.120): with + b replaced by - b as printed it equals {printed(a_, a_):.3f} at b = a (should be 0) and falls without bound as b grows (sum over three entries at b = (1, 5, 0.5): {printed(a_, b_):.3f}, at b = 100 b: "
          f"{printed(a_, 100 * a_):.3f}); the generalised KL sum a log(a/b) - a + b gives {right(a_, a_):.3f} at b = a and {right(a_, b_):.3f}, {right(a_, 100 * a_):.3f} at the other two points (non-negative, 0 only at b = a)")
    # ---- 3b. Lee-Seung against exponential gradient
    def run_ls(iters):
        A, S = A0.copy(), S0.copy(); Ls = [frob_loss(X, A, S)]
        for _ in range(iters):
            A = A * (X @ S.T) / (A @ S @ S.T + 1e-300)
            S = S * (A.T @ X) / (A.T @ A @ S + 1e-300)
            Ls.append(frob_loss(X, A, S))
        return A, S, np.array(Ls)
    def run_eg(eta, iters):
        A, S = A0.copy(), S0.copy(); Ls = [frob_loss(X, A, S)]
        with np.errstate(all="ignore"):
            for _ in range(iters):
                A = A * np.exp(-eta * (-X @ S.T + A @ S @ S.T))
                S = S * np.exp(-eta * (-A.T @ X + A.T @ A @ S))
                Ls.append(frob_loss(X, A, S))
                if not np.isfinite(Ls[-1]) or Ls[-1] > 1e12:
                    break
        return A, S, np.array(Ls)
    iters = 300
    _, _, L_ls = run_ls(iters)
    print(f"   Lee-Seung updates (13.124)-(13.125), {iters} iterations from a common start: L = {L_ls[0]:.4f} -> {L_ls[10]:.4f} (10) -> {L_ls[100]:.4f} (100) -> {L_ls[-1]:.6f}; the loss decreased at every iteration (largest change L_(t+1) - L_t {float(np.max(np.diff(L_ls))):.1e} < 0);"
          f" relative residual |X - A S|/|X| = {math.sqrt(2 * L_ls[-1]) / fro(X):.1e}")
    print(f"      {'eta':>6s} {'iterations with an increase of L':>34s} {'L after 100 iterations':>24s} {'after 300':>12s}")
    eg = {}
    for eta in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        _, _, L_ = run_eg(eta, iters)
        inc = int(np.sum(np.diff(L_) > 1e-12)); eg[eta] = L_
        if len(L_) < iters + 1:
            print(f"      {eta:>6.3f} {inc:>34d} {'diverged after ' + str(len(L_) - 1) + ' iterations':>37s}")
        else:
            print(f"      {eta:>6.3f} {inc:>34d} {L_[100]:>24.5g} {L_[-1]:>12.5g}")
    STORE["nmf_curves"] = dict(ls=L_ls, eg=eg)
    # identity: one Lee-Seung step of A is one exponential-gradient step with entry-wise rate eta_it = (log N - log P)/(N - P)
    Nn = X @ S0.T; Pp_ = A0 @ S0 @ S0.T
    eta_it = (np.log(Nn) - np.log(Pp_)) / (Nn - Pp_)
    A_ls1 = A0 * Nn / Pp_; A_eg1 = A0 * np.exp(-eta_it * (Pp_ - Nn))
    print(f"   (13.124) is not (13.121) with one learning constant: one Lee-Seung step of A equals one exponential-gradient step with the entry-wise rate eta_it = (log N - log P)/(N - P) = 1/logmean(N, P), "
          f"N = (X S^T)_it, P = (A S S^T)_it: difference {np.abs(A_ls1 - A_eg1).max():.1e}; these rates range over [{eta_it.min():.4f}, {eta_it.max():.4f}] at the starting point, so no single eta reproduces it"
          f" (relative gap to EG with eta = 0.01, 0.1: {np.abs(A_ls1 - A0 * np.exp(-0.01 * (Pp_ - Nn))).max() / np.abs(A_ls1 - A0).max():.2f}, {np.abs(A_ls1 - A0 * np.exp(-0.1 * (Pp_ - Nn))).max() / np.abs(A_ls1 - A0).max():.2f} of the step)")
    # plain gradient descent breaks positivity
    etag = 0.05
    A_gd = A0 - etag * (Pp_ - Nn)
    print(f"   plain gradient descent A <- A - {etag} dL/dA from the same point leaves the non-negative orthant: {int(np.sum(A_gd < 0))} of {A_gd.size} entries of A become negative after one step; the exponential-gradient step cannot ({int(np.sum(A_eg1 < 0))}), it is gradient descent on log A (13.122)")
    # generalised KL version: the same structure
    def run_ls_kl(iters):
        A, S = A0.copy(), S0.copy(); Ls = [gkl(X, A @ S)]
        for _ in range(iters):
            A = A * ((X / (A @ S)) @ S.T) / S.sum(axis=1)[None, :]
            S = S * (A.T @ (X / (A @ S))) / A.sum(axis=0)[:, None]
            Ls.append(gkl(X, A @ S))
        return A, S, np.array(Ls)
    _, _, Lk = run_ls_kl(iters)
    print(f"   the same construction for the generalised KL (13.120, corrected): dD/dA = P - N with P = 1 S^T (row sums), N = (X/(A S)) S^T and the multiplicative update A <- A N/P is the exponential-gradient step with rate 1/logmean(N, P) again; "
          f"D = {Lk[0]:.4f} -> {Lk[100]:.5f} (100) -> {Lk[-1]:.2e}, largest change D_(t+1) - D_t {float(np.max(np.diff(Lk))):.1e} < 0")
    # ---- 3c. the cone picture: uniqueness fails unless the data fill the cone
    A2 = np.array([[1.0, 0.2], [0.1, 1.0]])
    def ang(v):
        return math.degrees(math.atan2(v[1], v[0]))
    rng = rng_for(42)
    S_full = np.hstack([rng.random((2, 28)) ** 2, np.array([[1.0, 0.0], [0.0, 1.0]])])               # includes the two boundary rays
    S_int = 0.3 + 0.7 * rng.random((2, 30))                                                           # entries in [0.3, 1]: stays away from the axes
    out = {}
    for lab, Sx in (("data touch both boundary rays", S_full), ("data stay inside the cone", S_int)):
        Xc = A2 @ Sx
        angs = np.degrees(np.arctan2(Xc[1], Xc[0]))
        lo, hi = angs.min(), angs.max()
        a1, a2 = ang(A2[:, 0]), ang(A2[:, 1])
        slack = (lo - a1, a2 - hi)
        # alternative factorisations: pick columns on the rays at angles interpolated between A's columns and the extreme data columns
        ok = []; alts = []
        for f in (0.0, 0.5, 1.0):
            t1 = math.radians(a1 + f * slack[0]); t2 = math.radians(a2 - f * slack[1])
            Ap = np.array([[math.cos(t1), math.cos(t2)], [math.sin(t1), math.sin(t2)]])
            Sp = np.linalg.solve(Ap, Xc)
            ok.append((f, Sp.min(), fro(Ap @ Sp - Xc), (Ap[:, 0] >= 0).all() and (Ap[:, 1] >= 0).all())); alts.append(Ap)
        out[lab] = (slack, ok, Xc, alts, (lo, hi))
        print(f"   cone picture (Fig. 13.7), A = [[1, .2], [.1, 1]] (columns at {a1:.2f} and {a2:.2f} deg): {lab}: the data fill the angles [{lo:.2f}, {hi:.2f}] deg, slack at the two edges {slack[0]:.2f}, {slack[1]:.2f} deg; "
              "alternative factorisations X = A' S' with A' >= 0 whose columns move a fraction f of the slack inwards: " + ", ".join(f"f = {f}: min S' = {mn:+.3f}, residual {rs:.0e}" for f, mn, rs, _ in ok))
    STORE["nmf_cone"] = out
    print("   so the mixing matrix is recovered from the cone of the data only if the data reach its boundary (a 'sparse enough' condition); otherwise every cone between the data cone and the positive quadrant gives an exact non-negative factorisation")


# ================================================================== 4. sparse signal processing (13.4)

def lasso_path(A, x, tol=1e-12, max_steps=1000, record=True):
    """Exact lasso path for psi(th) = 1/2 |x - A th|^2 + lam |th|_1 by homotopy (LARS with the lasso modification).
    th(lam) is piecewise linear; eta = grad psi = A^T(A th - x) is the dual coordinate. Returns the knots (lam, theta, active set, signs, segment direction).
    On a segment with active set A and signs s: eta_A = -lam s_A, d = (A_A^T A_A)^(-1) s_A is d theta_A / d(-lam), and g_j = (G d)_j = d eta_j / d(-lam) for inactive j."""
    m, n = A.shape
    b = A.T @ x
    th = np.zeros(n)
    lam = float(np.max(np.abs(b))); lam_init = lam
    j0 = int(np.argmax(np.abs(b)))
    act = [j0]; sg = {j0: float(np.sign(b[j0]))}
    knots = [dict(lam=lam, theta=th.copy(), active=list(act), signs=dict(sg), event=("start", j0))]
    for it in range(max_steps):
        Aidx = np.array(act); sA = np.array([sg[i] for i in act])
        AA = A[:, Aidx]
        d = np.linalg.solve(AA.T @ AA, sA)
        eta = A.T @ (A @ th - x)
        g = A.T @ (AA @ d)
        isin = np.ones(n, bool); isin[Aidx] = False
        # inactive coordinate j reaches |eta_j| = lam: eta_j + delta g_j = +-(lam - delta)
        den1 = 1 + g; den2 = 1 - g
        with np.errstate(divide="ignore", invalid="ignore"):
            c1 = np.where(isin & (den1 > tol), (lam - eta) / den1, np.inf)
            c2 = np.where(isin & (den2 > tol), (lam + eta) / den2, np.inf)
        c1 = np.where(c1 > tol, c1, np.inf); c2 = np.where(c2 > tol, c2, np.inf)
        if len(act) >= m:                                           # at most m columns can be active: nothing more can enter (the ratio |eta_j|/lam is constant on this segment)
            c1[:] = np.inf; c2[:] = np.inf
        j1 = int(np.argmin(c1)); j2 = int(np.argmin(c2))
        best, ev = lam, ("end", None)
        if c1[j1] < best - 1e-14:
            best, ev = float(c1[j1]), ("enter", j1, -1.0)
        if c2[j2] < best - 1e-14:
            best, ev = float(c2[j2]), ("enter", j2, +1.0)
        thA = th[Aidx]
        with np.errstate(divide="ignore", invalid="ignore"):
            cz = np.where(thA * d < 0, -thA / d, np.inf)
        cz = np.where(cz > tol, cz, np.inf)
        iz = int(np.argmin(cz))
        if cz[iz] < best - 1e-14:
            best, ev = float(cz[iz]), ("leave", int(Aidx[iz]))
        if lam - best < 1e-9 * lam_init:                              # an event at lam ~ 1e-9 lam_max is numerically the end of the path
            best, ev = lam, ("end", None)
        if record:
            knots[-1].update(d=(Aidx.copy(), d.copy(), g.copy(), np.where(isin)[0]))
        th = th.copy(); th[Aidx] += best * d
        lam -= best
        if ev[0] == "end":
            knots.append(dict(lam=0.0, theta=th.copy(), active=list(act), signs=dict(sg), event=ev, dual=AA @ d))
            break
        if ev[0] == "enter":
            act.append(ev[1]); sg[ev[1]] = ev[2]
        else:
            th[ev[1]] = 0.0; act.remove(ev[1]); sg.pop(ev[1])
        knots.append(dict(lam=lam, theta=th.copy(), active=list(act), signs=dict(sg), event=ev))
    return knots


def theta_on_path(knots, lam):
    """theta(lam) by linear interpolation between the knots (lam decreasing along the list)."""
    for k in range(len(knots) - 1):
        l0, l1 = knots[k]["lam"], knots[k + 1]["lam"]
        if l1 - 1e-15 <= lam <= l0 + 1e-15:
            t = (l0 - lam) / (l0 - l1) if l0 > l1 else 0.0
            return (1 - t) * knots[k]["theta"] + t * knots[k + 1]["theta"]
    return knots[-1]["theta"].copy()


def theta_at_l1(knots, c):
    """The point of the path with |theta|_1 = c (the L1 norm decreases monotonically in lam)."""
    lo, hi = 0.0, knots[0]["lam"]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.abs(theta_on_path(knots, mid)).sum() > c:
            lo = mid
        else:
            hi = mid
    return theta_on_path(knots, 0.5 * (lo + hi)), 0.5 * (lo + hi)


def proj_l1_ball(v, c):
    """Euclidean projection onto {|th|_1 <= c} (sorting algorithm)."""
    if np.abs(v).sum() <= c:
        return v.copy()
    u = np.sort(np.abs(v))[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (css - c))[0][-1]
    t = (css[rho] - c) / (rho + 1)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)


def fista_l1ball(G, b, c, iters=4000):
    """Minimise (1/2) th^T G th - b^T th over |th|_1 <= c by accelerated projected gradient: a route independent of the path."""
    L = float(np.linalg.eigvalsh(G)[-1]); th = np.zeros(len(b)); z = th.copy(); t = 1.0
    for _ in range(iters):
        thn = proj_l1_ball(z - (G @ z - b) / L, c)
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        z = thn + ((t - 1) / tn) * (thn - th); th, t = thn, tn
    return th


def bp_solution(A, x):
    """Basis pursuit min |th|_1 s.t. A th = x as the end (lam -> 0) of the lasso path of an under-determined problem; also returns the dual vector y with A^T y in the subdifferential."""
    kn = lasso_path(A, x, record=False)
    return kn[-1]["theta"], kn[-1].get("dual"), len(kn)


def check_sparse():
    head("4. Sparse signal processing (section 13.4)")
    # ---- 4a. generalised inverse, L0, L1 on small cases
    rng = rng_for(51)
    m, n, k = 12, 40, 3
    A = rng.normal(size=(m, n)) / math.sqrt(m)
    th0 = np.zeros(n); sup = rng.choice(n, k, replace=False); th0[sup] = rng.normal(size=k) + np.sign(rng.normal(size=k))
    x = A @ th0
    thp = A.T @ np.linalg.solve(A @ A.T, x)                                   # (13.135)-(13.136)
    thb, y, nk = bp_solution(A, x)
    gap = float(np.abs(thb).sum() - x @ y)
    print(f"   m = {m}, n = {n}, a {k}-sparse truth: the generalised inverse A^T (A A^T)^(-1) x (13.135)-(13.136) has |theta|_2 = {np.linalg.norm(thp):.3f} <= {np.linalg.norm(th0):.3f} (truth) and {int(np.sum(np.abs(thp) > 1e-10))} of {n} non-zero components; "
          f"the minimum-L1 solution (the end of the lasso path, {nk} knots) has {int(np.sum(np.abs(thb) > 1e-10))} non-zeros and error {np.abs(thb - th0).max():.1e}")
    print(f"      optimality certificate of the L1 solution: the vector y = lim r/lam satisfies |A^T y|_inf = {np.abs(A.T @ y).max():.12f} (<= 1, = 1 on the support) and x.y = {x @ y:.10f} = |theta|_1 = {np.abs(thb).sum():.10f} (duality gap {gap:.1e})")
    # exhaustive L0 on a tiny problem, and an instance where L1 and L0 disagree
    rng = rng_for(52)
    n_, m_ = 12, 5
    from itertools import combinations
    cnt_agree, cnt_dis, ex = 0, 0, None
    worst_gap, worst_dual = 0.0, 0.0
    for trial in range(300):
        A_ = rng.normal(size=(m_, n_)); kk = 2
        t0 = np.zeros(n_); sp = rng.choice(n_, kk, replace=False); t0[sp] = rng.normal(size=kk)
        x_ = A_ @ t0
        t1, y1, _ = bp_solution(A_, x_)
        worst_gap = max(worst_gap, float(abs(np.abs(t1).sum() - x_ @ y1))); worst_dual = max(worst_dual, float(np.abs(A_.T @ y1).max() - 1))
        same = np.abs(t1 - t0).max() < 1e-6
        if same: cnt_agree += 1
        else:
            cnt_dis += 1
            if ex is None: ex = (A_, x_, t0, t1)
    A_, x_, t0, t1 = ex
    sols = []
    for kk in (1, 2):
        for S_ in combinations(range(n_), kk):
            AS = A_[:, S_]
            c_, res, rk, _ = np.linalg.lstsq(AS, x_, rcond=None)
            if np.linalg.norm(AS @ c_ - x_) < 1e-9:
                sols.append((kk, S_))
    print(f"   exhaustive L0 on n = {n_}, m = {m_}, 2-sparse truths: all supports of size <= 2 checked, so the sparsest solution is exact; in {cnt_agree} of {cnt_agree + cnt_dis} random instances the L1 solution equals it, in {cnt_dis} it does not "
          f"(e.g. L0 = {int(np.sum(np.abs(t0) > 1e-9))} but the L1 minimiser has {int(np.sum(np.abs(t1) > 1e-9))} non-zeros and |theta|_1 = {np.abs(t1).sum():.3f} < {np.abs(t0).sum():.3f}); the L0 solution is the only one with <= 2 non-zeros: {len(sols) == 1}")
    print(f"      the endpoint of the path is a minimum-L1 solution in all 300 instances: duality gap |theta|_1 - x.y at most {worst_gap:.1e}, and |A^T y|_inf - 1 at most {worst_dual:.1e}")
    STORE["l1_vs_l0"] = (cnt_agree, cnt_dis)
    # ---- 4b. compressed sensing experiment: how many measurements does recovery need?
    print("   Theorem 13.2 / (13.128), (13.139): m > 2 k log n. Experiment: n = 200, Gaussian A (m x n), k-sparse theta (non-zeros N(0,1)), recovery by minimum L1 (exact homotopy), 'success' = error < 1e-6; 30 trials per cell:")
    rng = rng_for(53)
    n = 200; ks = (3, 5, 8); ms = (10, 14, 18, 22, 26, 30, 36, 44, 56, 70, 90); trials = 30
    table = {}
    for k_ in ks:
        row = []
        for m_ in ms:
            ok = 0
            for _ in range(trials):
                A_ = rng.normal(size=(m_, n)) / math.sqrt(m_)
                t0 = np.zeros(n); sp = rng.choice(n, k_, replace=False); t0[sp] = rng.normal(size=k_)
                kn = lasso_path(A_, A_ @ t0, record=False)
                ok += int(np.abs(kn[-1]["theta"] - t0).max() < 1e-6)
            row.append(ok / trials)
        table[k_] = row
    print("      m:      " + " ".join(f"{m_:>5d}" for m_ in ms))
    m50 = {}
    for k_ in ks:
        row = table[k_]
        print(f"      k = {k_:<3d} " + " ".join(f"{p:>5.2f}" for p in row))
        m50[k_] = next((ms[i - 1] + (0.5 - row[i - 1]) / (row[i] - row[i - 1]) * (ms[i] - ms[i - 1]) for i in range(1, len(ms)) if row[i] >= 0.5 > row[i - 1]), float("nan"))
    print(f"      {'k':>3s} {'m at 50% (interpolated)':>24s} {'2 k ln n':>10s} {'2 k ln(n/k)':>12s} {'2 k ln(n/m_50)':>16s} {'m_50/(2 k ln n)':>16s} {'success at m = 2k ln n':>24s}")
    for k_ in ks:
        mb = 2 * k_ * math.log(n)
        idx = min(range(len(ms)), key=lambda i: abs(ms[i] - mb))
        print(f"      {k_:>3d} {m50[k_]:>24.1f} {mb:>10.1f} {2 * k_ * math.log(n / k_):>12.1f} {2 * k_ * math.log(n / m50[k_]):>16.1f} {m50[k_] / mb:>16.2f} {('%.2f at m = %d' % (table[k_][idx], ms[idx])):>24s}")
    STORE["cs"] = dict(n=n, ks=ks, ms=ms, table=table, m50=m50)
    print("      so the printed condition is a comfortable sufficient one at this size (recovery is certain at m = 2 k ln n) but not the threshold: the threshold is a factor 2 to 3 lower, and its growth with k is closer to k ln(n/m) than to k ln n")
    sparse_geometry()


def segment_table(knots):
    """For each segment of a recorded lasso path: (lam_start, lam_end, active, signs, d, g, inactive, monotone), where
    g_j = d eta_j / d(-lam) for the inactive j and the path is monotone on the segment if every active coefficient moves away from 0."""
    out = []
    for k in range(len(knots) - 1):
        kd = knots[k].get("d")
        if kd is None:
            continue
        Aidx, d, g, inact = kd
        sA = np.array([knots[k]["signs"][i] for i in Aidx])
        out.append(dict(l0=knots[k]["lam"], l1=knots[k + 1]["lam"], active=Aidx, s=sA, d=d, g=g[inact], inact=inact, monotone=bool(np.all(sA * d > 0)), gA=g[Aidx]))
    return out


def correlated_design(rng, m, n, mix=0.9):
    Z = rng.normal(size=(m, n))
    Mx = np.eye(n) + mix * rng.normal(size=(n, n))
    A = Z @ Mx.T
    return A / np.linalg.norm(A, axis=0)


def stagewise_path(A, x, eps, c_max, cap=6000):
    """The L1-Minkowskian gradient flow (13.170)-(13.172) with step eps: move only the coordinate with the largest |eta_j| = |dpsi/dtheta_j|, by -eps sign(eta_j)."""
    G = A.T @ A; b = A.T @ x
    n = A.shape[1]
    th = np.zeros(n); eta = -b.copy(); l1 = 0.0
    ths = np.zeros((cap, n)); l1s = np.zeros(cap); t = 0
    while l1 < c_max and t < cap - 1:
        j = int(np.argmax(np.abs(eta))); sj = 1.0 if eta[j] > 0 else -1.0
        l1 += (abs(th[j] - eps * sj) - abs(th[j]))
        th[j] -= eps * sj; eta -= eps * sj * G[:, j]
        t += 1; ths[t] = th; l1s[t] = l1
    return ths[:t + 1], l1s[:t + 1]


def sparse_geometry():
    # ---- 4c. Theorem 13.3, the KKT conditions (13.151)-(13.155), Lemma 13.2, the Pythagorean inequality
    rng = rng_for(54)
    m, n = 40, 7
    Z = rng.normal(size=(m, n)); Mx = np.eye(n) + 0.7 * rng.normal(size=(n, n))
    A = Z @ Mx.T / math.sqrt(m)
    th_true = np.array([2.0, -1.5, 0.0, 0.8, 0.0, 0.0, -0.5]); x = A @ th_true + 0.3 * rng.normal(size=m)
    G = A.T @ A; b = A.T @ x; th_star = np.linalg.solve(G, b)
    kn = lasso_path(A, x)
    print(f"   overdetermined example, m = {m}, n = {n}, psi = (1/2)|x - A theta|^2, G = A^T A (condition number {np.linalg.cond(G):.1f}); unconstrained optimum theta* = G^(-1) A^T x with |theta*|_1 = {np.abs(th_star).sum():.4f}; "
          f"the lasso path (exact homotopy) has {len(kn) - 1} segments: events " + ", ".join(f"{e['event'][0]} {e['event'][1]}" for e in kn[1:-1]) + f"; it ends at theta* (difference {np.abs(kn[-1]['theta'] - th_star).max():.1e})")
    worst_a, worst_i = 0.0, 0.0
    for lam in np.linspace(0.02, 0.98, 97) * kn[0]["lam"]:
        th = theta_on_path(kn, lam); eta = G @ th - b
        act = np.abs(th) > 1e-12
        worst_a = max(worst_a, float(np.max(np.abs(eta[act] + lam * np.sign(th[act]))))) if act.any() else worst_a
        worst_i = max(worst_i, float(np.max(np.abs(eta[~act]) - lam))) if (~act).any() else worst_i
    print(f"   Lemma 13.2 / (13.155): along the path, eta_A + lam s_A = 0 to {worst_a:.1e} on the active set (largest violation over 97 values of lam), and |eta_j| - lam <= {worst_i:.4f} < 0 on the inactive coordinates (sub-gradient condition)")
    c = 0.5 * np.abs(th_star).sum()
    th_c, lam_c = theta_at_l1(kn, c)
    th_fista = fista_l1ball(G, b, c)
    eta_c = G @ th_c - b
    act = np.abs(th_c) > 1e-9
    print(f"   Theorem 13.3: the minimiser of psi on B_c, c = |theta*|_1 / 2 = {c:.4f}: from the path {np.round(th_c, 4).tolist()}; by accelerated projected gradient (an independent algorithm) the difference is {np.abs(th_c - th_fista).max():.1e}; "
          f"active set {np.where(act)[0].tolist()}, lam(c) = {lam_c:.4f}; (13.151): eta_c = {np.round(eta_c, 4).tolist()} = -lam s on the active set and |eta_j| <= lam off it ({np.abs(eta_c[~act]).max() if (~act).any() else 0.0:.4f} <= {lam_c:.4f})")
    # Pythagorean inequality for the projection onto a convex set that is not flat
    D = lambda u, v: 0.5 * (u - v) @ G @ (u - v)          # Bregman divergence of the quadratic psi: D[u : v]
    rng2 = rng_for(55)
    Zs = rng2.laplace(size=(20000, n)); Zs *= (c * rng2.random((20000, 1)) ** (1 / n) / np.abs(Zs).sum(axis=1, keepdims=True))
    slack = (Zs - th_c) @ eta_c
    direct = np.array([D(z_, th_star) - D(z_, th_c) - D(th_c, th_star) for z_ in Zs[:2000]])
    print(f"   projection theorem on the (non-flat) convex set B_c: for 20000 random points x of B_c, D[x:theta*] - D[x:theta_c] - D[theta_c:theta*] = eta_c . (x - theta_c) >= 0, smallest value {slack.min():.2e} (explicit divergences agree: {np.abs(direct - slack[:2000]).max():.1e});"
          " equality would be the Pythagorean theorem (it needs a flat S)")
    # points on the same face: same support, same signs, |x|_1 = c
    sA = np.sign(th_c[act]); w = rng2.dirichlet(np.ones(int(act.sum())), size=2000) * c
    Zf = np.zeros((2000, n)); Zf[:, act] = w * sA
    print(f"   on the face of B_c that contains theta_c (same support and signs, |x|_1 = c) the slack is {np.abs((Zf - th_c) @ eta_c).max():.1e}: the Pythagorean equality holds along the flat piece, as it must (a facet is e-flat)")
    # infinitely many theta* are mapped to a non-regular point: change theta* inside the normal cone
    lam_ = lam_c; Aidx = np.where(act)[0]; Ab = np.where(~act)[0]
    devs = []
    for t in range(20):
        eta_t = np.zeros(n); eta_t[Aidx] = -lam_ * np.sign(th_c[Aidx]); eta_t[Ab] = lam_ * (2 * rng2.random(len(Ab)) - 1)
        th_star_t = th_c - np.linalg.solve(G, eta_t)                       # a different unconstrained optimum with grad psi~(theta_c) = eta_t
        devs.append(np.abs(fista_l1ball(G, G @ th_star_t, c) - th_c).max())
    print(f"   (13.154): 20 different points theta*~ = theta_c - G^(-1) eta~ with eta~_A = -lam s_A and eta~_j drawn from [-lam, lam] on the {len(Ab)} inactive coordinates are all projected onto the same theta_c (largest deviation {max(devs):.1e}):"
          f" a whole {len(Ab)}-dimensional cone of unconstrained optima lands on this non-regular point, against a single ray for a regular boundary point")
    # sparsity as a function of the radius, G = I
    rng3 = rng_for(56)
    Tt = rng3.normal(size=(100000, n)); u = np.sort(np.abs(Tt), axis=1)[:, ::-1]; css = np.cumsum(u, axis=1)
    fr = {}
    for f_ in (0.8, 0.5, 0.2):
        cc = f_ * u.sum(axis=1, keepdims=True)
        rho = (u * np.arange(1, n + 1)[None, :] > css - cc).sum(axis=1)          # number of non-zero components of the projection
        fr[f_] = (float(np.mean(rho)), float(np.mean(rho < n)), float(np.mean(rho == 1)))
    STORE["sparsity_vs_c"] = fr
    print("   (G = I, n = 7, 1e5 Gaussian points theta*) the projection onto B_c with c = f |theta*|_1 has on average " + ", ".join(f"{fr[f_][0]:.2f} non-zeros for f = {f_}" for f_ in fr)
          + "; the fraction of points mapped to a non-regular point (some theta_i = 0): " + ", ".join(f"{fr[f_][1]:.3f}" for f_ in fr) + "; to a vertex (one non-zero): " + ", ".join(f"{fr[f_][2]:.3f}" for f_ in fr))

    # logistic regression (13.142)-(13.144): the printed cumulant function lacks the logarithm
    u = 0.7
    printed = sum(math.exp(y * u - (1 + math.exp(u))) for y in (0, 1)); right = sum(math.exp(y * u - math.log(1 + math.exp(u))) for y in (0, 1))
    rng4 = rng_for(60); Xl = rng4.normal(size=(30, 4)); yl = (rng4.random(30) < 0.5).astype(float)
    th_l = rng4.normal(size=4)
    def logistic_psi(t_):
        z_ = Xl @ t_
        return float(-np.sum(yl * z_) + np.sum(np.log1p(np.exp(z_))))
    pr = 1 / (1 + np.exp(-(Xl @ th_l)))
    Hl = (Xl.T * (pr * (1 - pr))) @ Xl
    print(f"   logistic regression (13.142)-(13.144): the probabilities exp(y u - psi~(u)), y in {{0, 1}}, add up to {printed:.4f} with the printed psi~ = 1 + exp(u) (u = {u}) and to {right:.4f} with psi~ = log(1 + exp(u));"
          f" the loss (13.144) has Hessian sum_i p_i (1 - p_i) x_i x_i^T, smallest eigenvalue {np.linalg.eigvalsh(Hl)[0]:.4f} > 0 for 30 random inputs in R^4 (strictly convex when the inputs span R^n)")
    # ---- 4d. the least-angle theorem 13.4
    rng = rng_for(57)
    nd, viol_designs, viol_seg, tot_seg, maxr = 200, 0, 0, 0, 0.0
    cnt_noneq = 0
    for t in range(nd):
        Ad = correlated_design(rng, 30, 8); xd = rng.normal(size=30)
        kd = lasso_path(Ad, xd)
        segs = segment_table(kd)
        v = [np.max(np.abs(sg_["g"])) if len(sg_["g"]) else 0.0 for sg_ in segs]
        viol_seg += sum(1 for r_ in v if r_ > 1 + 1e-9); tot_seg += len(segs); maxr = max(maxr, max(v) if v else 0.0)
        viol_designs += int(any(r_ > 1 + 1e-9 for r_ in v))
        cnt_noneq += int(any(np.max(np.abs(np.abs(sg_["gA"]) - 1)) > 1e-8 for sg_ in segs))
    print(f"   Theorem 13.4 on {nd} random designs (m = 30, n = 8, unit-norm correlated columns, x pure noise): equal |<theta', e_i>| on the active set holds on every segment (violations of (13.160): {cnt_noneq});"
          f" but the inactive axes are NOT always at larger angles: on {viol_seg} of {tot_seg} segments ({100 * viol_seg / tot_seg:.1f}%) some inactive axis has |<theta', e_j>| > |<theta', e_i>| (largest ratio {maxr:.2f}), in {viol_designs} of {nd} designs")
    STORE["least_angle_stats"] = (nd, viol_designs, viol_seg, tot_seg, maxr)
    # explicit three-column counterexample
    a_ = 0.5 * math.acos(0.9); beta = 0.3
    cols = np.array([[math.cos(a_), math.sin(a_), 0.0], [-math.cos(a_), math.sin(a_), 0.0], [0.0, math.cos(beta), math.sin(beta)]]).T
    Gc = cols.T @ cols
    best = None
    rng = rng_for(58)
    for t in range(1500):
        xc = rng.normal(size=3)
        kc = lasso_path(cols, xc)
        for sg_ in segment_table(kc):
            if set(sg_["active"].tolist()) == {0, 1} and sg_["s"][0] == sg_["s"][1]:
                r_ = float(np.abs(sg_["g"][0]))
                if best is None or r_ > best[0]:
                    best = (r_, xc.copy(), sg_, kc)
    r_, xc, sg_, kc = best
    nu = math.sqrt(float(sg_["s"] @ np.linalg.solve(Gc[np.ix_(sg_["active"], sg_["active"])], sg_["s"])))
    print(f"   explicit counterexample: three unit columns in R^3 with G_12 = {Gc[0, 1]:.2f} (two nearly opposite directions), G_13 = G_23 = {Gc[0, 2]:.4f}; for the data x = {np.round(xc, 3).tolist()} the path has the active set {{1, 2}} with equal signs on lam in "
          f"[{sg_['l1']:.4f}, {sg_['l0']:.4f}]; there |<theta', e_1>| = |<theta', e_2>| = 1 (per unit decrease of lam) but the inactive axis has |<theta', e_3>| = {r_:.4f} > 1 (= G_3A G_AA^(-1) s_A);"
          f" the angles between theta' and the axes are {math.degrees(math.acos(1 / nu)):.1f} deg (active) and {math.degrees(math.acos(min(1.0, r_ / nu))):.1f} deg (inactive): the inactive axis is the closer one, the opposite of (13.161)'s prose")
    STORE["cx"] = dict(r=r_, nu=nu, x=xc, G=Gc, seg=sg_, knots=kc, cols=cols)
    # the angle statement needs equal axis lengths
    Ab = correlated_design(rng, 30, 3) * np.array([1.0, 2.5, 0.4]); xb = rng.normal(size=30)
    kb = lasso_path(Ab, xb); sgb = [s_ for s_ in segment_table(kb) if len(s_["active"]) >= 2][0]
    Gb = Ab.T @ Ab; ia = sgb["active"]
    ip = [abs(float((Gb @ np.bincount(ia, weights=sgb["d"], minlength=3))[i])) for i in ia]
    nrm = [math.sqrt(Gb[i, i]) for i in ia]; nu_b = math.sqrt(float(sgb["d"] @ Gb[np.ix_(ia, ia)] @ sgb["d"]))
    print(f"   (13.160) with unequal column norms (|e_i| = {np.round(nrm, 3).tolist()}): the inner products are equal ({np.round(ip, 6).tolist()}) but the angles are not: arccos(<theta', e_i>/(|theta'| |e_i|)) = {np.round([math.degrees(math.acos(ip[q] / (nu_b * nrm[q]))) for q in range(len(ia))], 2).tolist()} deg;"
          " 'same angle' needs unit-length axes (G_ii = 1, as in the standard LARS normalisation)")

    # ---- 4e. the Minkowskian gradient flow against the lasso path
    rng = rng_for(59)
    nd = 30; eps = 0.004
    rows = []
    for t in range(nd):
        Ad = correlated_design(rng, 40, 6, mix=0.25); xd = rng.normal(size=40)
        ths_ = np.linalg.solve(Ad.T @ Ad, Ad.T @ xd); xd = xd * (1.5 / np.abs(ths_).sum())
        kd = lasso_path(Ad, xd)
        segs = segment_table(kd)
        mono = all(sg_["monotone"] for sg_ in segs)
        leave = any(e["event"][0] == "leave" for e in kd)
        cmax = float(np.abs(kd[-1]["theta"]).sum())
        thf, l1f = stagewise_path(Ad, xd, eps, 0.95 * cmax)
        dev = 0.0
        for c_ in np.linspace(0.05, 0.95, 10) * cmax:
            i = int(np.argmin(np.abs(l1f - c_)))
            dev = max(dev, float(np.abs(thf[i] - theta_at_l1(kd, l1f[i])[0]).max()))
        rows.append((mono, leave, dev))
    groups = {"monotone": [d_ for mo, lv, d_ in rows if mo], "shrinking coefficient, no drop-out": [d_ for mo, lv, d_ in rows if not mo and not lv], "a variable leaves the active set": [d_ for mo, lv, d_ in rows if lv]}
    print(f"   Minkowskian gradient flow (13.170)-(13.172) with step {eps} (a coordinate-wise 'forward stagewise' rule) against the exact lasso path at equal |theta|_1: {nd} random correlated designs (m = 40, n = 6, |theta*|_1 = 1.5);"
          " largest deviation between the two paths (sup norm over 10 values of |theta|_1), by what the lasso path does:")
    for kname, vals in groups.items():
        if vals:
            print(f"      {kname:36s} {len(vals):>3d} designs: median {np.median(vals):.4f}, maximum {max(vals):.4f}, {sum(1 for d_ in vals if d_ > 5 * eps)} above {5 * eps:.3f}")
        else:
            print(f"      {kname:36s}   0 designs")
    STORE["stagewise"] = rows


# ================================================================== 5. optimisation in convex programming (13.5)

HEX_A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0], [1.0, -2.0]])
HEX_B = np.array([0.0, 0.0, -4.0, -3.0, -5.5, -4.0])
HEX_C = np.array([-1.0, -0.7])                                   # minimise c.theta: the vertex (4, 1.5) is optimal, value -5.05


def barrier(theta, A, b):
    """psi = -sum log(A theta - b) (13.181): value, eta = grad psi, g = Hessian, and the slacks s."""
    s = A @ theta - b
    return -float(np.sum(np.log(s))), -A.T @ (1 / s), (A.T * (1 / s ** 2)) @ A, s


def cubic_T(theta, A, b):
    """The cubic tensor psi_ijk = -2 sum_k A_ki A_kj A_kk / s_k^3 of the log barrier."""
    s = A @ theta - b
    return -2 * np.einsum("ki,kj,kl,k->ijl", A, A, A, 1 / s ** 3)


def newton_barrier(A, b, c, t, theta0, tol=1e-13, maxit=100):
    """Damped Newton method for min t c.theta + psi(theta), run to a gradient residual below tol * max(1, t); returns theta and the number of iterations."""
    th = theta0.copy()
    for it in range(maxit):
        f0, eta, G, s = barrier(th, A, b)
        gr = t * c + eta
        if float(np.abs(gr).max()) < tol * max(1.0, t):
            return th, it
        step = -np.linalg.solve(G, gr)
        dec = float(-gr @ step)
        al = 1.0
        while np.any(A @ (th + al * step) - b <= 0):
            al *= 0.5
        f0 = t * c @ th + f0
        while t * c @ (th + al * step) + barrier(th + al * step, A, b)[0] > f0 - 0.25 * al * dec and al > 1e-12:
            al *= 0.5
        th = th + al * step
    return th, maxit




def check_convex():
    head("5. Optimisation in convex programming (section 13.5)")
    A, b, c = HEX_A, HEX_B, HEX_C
    m = len(b)
    verts = []
    for i in range(m):
        for j in range(i + 1, m):
            M = A[[i, j]]
            if abs(np.linalg.det(M)) > 1e-12:
                v = np.linalg.solve(M, b[[i, j]])
                if np.all(A @ v - b >= -1e-9):
                    verts.append(v)
    vals = [float(c @ v) for v in verts]
    pstar = min(vals); vstar = verts[int(np.argmin(vals))]
    print(f"   running example: the LP min c.theta over the hexagon A theta >= b, c = (-1, -0.7), {m} constraints; vertices {[tuple(float(round(x_, 2)) for x_ in v) for v in verts]}, optimum {tuple(float(x_) for x_ in vstar)} with value {pstar:.4f}")
    # ---- 5a. (13.181), (13.186)-(13.189) for the discrete barrier
    th = np.array([1.7, 1.2])
    psi0, eta, G, s = barrier(th, A, b)
    h = 1e-6
    eta_fd = np.array([(barrier(th + h * e, A, b)[0] - barrier(th - h * e, A, b)[0]) / (2 * h) for e in np.eye(2)])
    G_fd = np.array([(barrier(th + h * e, A, b)[1] - barrier(th - h * e, A, b)[1]) / (2 * h) for e in np.eye(2)])
    print(f"   barrier psi = -sum_k log(A_k theta - b_k) (13.181) at theta = (1.7, 1.2): eta = grad psi = -sum_k A_k/s_k = {np.round(eta, 6).tolist()} (finite differences: difference {np.abs(eta - eta_fd).max():.1e}), "
          f"g_ij = sum_k A_ki A_kj / s_k^2 = {np.round(G, 5).tolist()} (finite differences: {np.abs(G - G_fd).max():.1e}); the printed (13.188)-(13.189) are integrals over omega without the weight w(omega) of (13.179) although they are said to treat (13.181), where they are sums over k")
    # ---- 5b. the central path
    th0, _ = newton_barrier(A, b, c, 0.0, np.array([1.7, 1.2]))
    print(f"   analytic centre (t = 0, the minimiser of psi): theta = {np.round(th0, 6).tolist()}, eta(theta) = {np.round(barrier(th0, A, b)[1], 12).tolist()} (the centre is eta = 0, (13.197))")
    ts = [0.5, 1, 2, 5, 10, 30, 100, 300, 1000]
    path = {}
    prev = th0
    for t in ts:
        thc, it = newton_barrier(A, b, c, t, prev); prev = thc
        path[t] = thc
    print("   central path theta*(t) = argmin t c.theta + psi (13.196): dual coordinates, duality gap and the dual vector y_k = 1/(t s_k):")
    print(f"      {'t':>6s} {'theta*(t)':>22s} {'eta + t c (13.197)':>20s} {'c.theta* - p*':>14s} {'m / t':>9s} {'c.theta* - b.y':>15s} {'A^T y - c':>10s}")
    for t in ts:
        thc = path[t]; _, eta, G, s = barrier(thc, A, b); y = 1 / (t * s)
        print(f"      {t:>6g} {('(%.4f, %.4f)' % (thc[0], thc[1])):>22s} {np.abs(eta + t * c).max():>20.1e} {c @ thc - pstar:>14.3e} {m / t:>9.3e} {c @ thc - b @ y:>15.3e} {np.abs(A.T @ y - c).max():>10.1e}")
    print("      so the dual coordinates run along the straight line eta = -t c (an m-geodesic from the centre); the primal path is curved; the gap to the optimum is at most m/t, and c.theta - b.y = m/t exactly (the dual vector y_k = 1/(t s_k) is feasible)")
    STORE["cp_path"] = path; STORE["cp_vals"] = (pstar, vstar, verts)
    # velocity and acceleration: formula against finite differences along the path
    rows = []
    for t in (1.0, 5.0, 30.0):
        thc = path[t] if t in path else None
        _, eta, G, s = barrier(thc, A, b)
        Ginv = np.linalg.inv(G)
        thd = -Ginv @ c
        T = cubic_T(thc, A, b)
        thdd = -Ginv @ np.einsum("ijk,j,k->i", T, thd, thd)
        hh = 1e-2 * t
        tp, _ = newton_barrier(A, b, c, t + hh, thc); tm, _ = newton_barrier(A, b, c, t - hh, thc)
        thd_fd = (tp - tm) / (2 * hh); thdd_fd = (tp - 2 * thc + tm) / hh ** 2
        nrm2 = float(thd @ G @ thd)
        perp2 = float(thdd @ G @ thdd) - float(thd @ G @ thdd) ** 2 / nrm2
        rows.append((t, np.abs(thd - thd_fd).max(), np.abs(thdd - thdd_fd).max(), math.sqrt(perp2) / nrm2, thd, thdd, G))
    print("   velocity and acceleration of the central path: dtheta/dt = -G^(-1) c (from (13.192), (13.194)) and d2theta/dt2 = -G^(-1) T(dtheta/dt, dtheta/dt) (T the cubic tensor of psi) against finite differences of Newton solutions;"
          " e-curvature of the path (the part of theta'' normal to theta' in the metric g, per unit |theta'|_g^2):")
    for t, e1, e2, kap, *_ in rows:
        print(f"      t = {t:>5g}: |theta' - FD| = {e1:.1e}, |theta'' - FD| = {e2:.1e}, curvature {kap:.4f}")
    STORE["cp_curv"] = [(t, kap) for t, _, _, kap, *_ in rows]
    def signed_curv(t):
        thc = newton_barrier(A, b, c, t, th0)[0]
        G_ = barrier(thc, A, b)[2]; Gi = np.linalg.inv(G_)
        thd = -Gi @ c; thdd = -Gi @ np.einsum("ijk,j,k->i", cubic_T(thc, A, b), thd, thd)
        return float(thd[0] * thdd[1] - thd[1] * thdd[0])
    lo_, hi_ = 1.0, 8.0
    f_lo = signed_curv(lo_)
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        if signed_curv(mid) * f_lo > 0: lo_ = mid
        else: hi_ = mid
    t_infl = 0.5 * (lo_ + hi_)
    print(f"      the signed curvature theta' x theta'' changes sign at t = {t_infl:.3f} (signs {np.sign(signed_curv(1.0)):+.0f} at t = 1, {np.sign(signed_curv(8.0)):+.0f} at t = 8): the path has an inflection there, so its curvature has two humps")
    STORE["cp_infl"] = t_infl
    # ---- 5c. natural-gradient (affine scaling) steps from the centre against the central path
    print("   the natural gradient step (13.193), theta <- theta - eps G(theta)^(-1) c, from the analytic centre, against the central path at the same time t = k eps, final time t = 2:")
    print(f"      {'eps':>6s} {'steps':>6s} {'|theta_k - theta*(t)|':>22s} {'|eta_k + t c|':>14s} {'error / eps':>12s}")
    euler = []
    for eps in (0.4, 0.2, 0.1, 0.05, 0.025):
        thk = th0.copy(); K = int(round(2.0 / eps))
        for k in range(K):
            thk = thk - eps * np.linalg.solve(barrier(thk, A, b)[2], c)
        tt = K * eps
        ref, _ = newton_barrier(A, b, c, tt, th0)
        err = float(np.linalg.norm(thk - ref)); eta_err = float(np.abs(barrier(thk, A, b)[1] + tt * c).max())
        euler.append((eps, err, eta_err))
        print(f"      {eps:>6.3f} {K:>6d} {err:>22.3e} {eta_err:>14.3e} {err / eps:>12.4f}")
    STORE["euler"] = euler
    print("      the global error is first order in eps (error/eps is nearly constant): the flow (13.192) started at the centre follows the central path, and so does its Euler discretisation, with an O(eps) lag")
    thn = np.array([1.7, 1.2]); eta0 = barrier(thn, A, b)[1]; eps = 0.005
    for k in range(int(round(2.0 / eps))):
        thn = thn - eps * np.linalg.solve(barrier(thn, A, b)[2], c)
    print(f"   (13.195) from the off-centre start theta(0) = (1.7, 1.2): after t = 2 (eps = {eps}) eta(t) = {np.round(barrier(thn, A, b)[1], 4).tolist()} against eta(0) - t c = {np.round(eta0 - 2.0 * c, 4).tolist()}: the flow is a straight line in eta (an m-geodesic), the central path only when it starts at the centre")
    _, _, G0, _ = barrier(th0, A, b)
    G0i = np.linalg.inv(G0); T0 = cubic_T(th0, A, b); thd0 = -G0i @ c; thdd0 = -G0i @ np.einsum("ijk,j,k->i", T0, thd0, thd0)
    print("   one step from the centre: theta_1 - theta*(eps) against -(1/2) eps^2 theta''(0), the e-acceleration of the m-geodesic: ratio of the two vectors' lengths and the cosine of their angle:")
    onestep = []
    for eps in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        ref, _ = newton_barrier(A, b, c, eps, th0)
        d1 = (th0 - eps * G0i @ c) - ref
        pred = -0.5 * eps ** 2 * thdd0
        onestep.append((eps, float(np.linalg.norm(d1)), float(np.linalg.norm(pred))))
        print(f"      eps = {eps:>5.3f}: |theta_1 - theta*(eps)| = {np.linalg.norm(d1):.3e}, (1/2) eps^2 |theta''(0)| = {np.linalg.norm(pred):.3e}, ratio {np.linalg.norm(d1) / np.linalg.norm(pred):.4f}, cos = {float(d1 @ pred) / (np.linalg.norm(d1) * np.linalg.norm(pred)):.5f}")
    print(f"      the ratio tends to 1 only slowly because the path's acceleration changes fast near the centre (|theta''| = {np.linalg.norm(thdd0):.3f} at t = 0 but {np.linalg.norm(-np.linalg.inv(barrier(path[0.5], A, b)[2]) @ np.einsum('ijk,j,k->i', cubic_T(path[0.5], A, b), -np.linalg.inv(barrier(path[0.5], A, b)[2]) @ c, -np.linalg.inv(barrier(path[0.5], A, b)[2]) @ c)):.3f} at t = 0.5)")
    STORE["onestep"] = onestep
    # ---- 5d. standard (equality) form: the path is straight in the induced dual coordinates, not in the ambient ones
    print("   the same LP in slack form: x = A theta - b in R^6_+ with the affine constraint set {x = A theta - b}; barrier -sum log x_k, ambient dual coordinates eta = -1/x; path x*(t) = A theta*(t) - b:")
    pts = {t: -1 / (A @ newton_barrier(A, b, c, t, th0)[0] - b) for t in (1, 2, 3)}
    e1, e2, e3 = pts[1], pts[2], pts[3]
    line = e1 + (e2 - e1) * 2.0
    amb_def = np.linalg.norm(e3 - line) / np.linalg.norm(e3 - e1)
    ind = {t: A.T @ pts[t] for t in pts}
    ind_def = np.linalg.norm(ind[3] - (ind[1] + (ind[2] - ind[1]) * 2.0))
    print(f"      the ambient points eta(1), eta(2), eta(3) are not collinear (distance of eta(3) from the line through the first two, relative to |eta(3) - eta(1)|: {amb_def:.3f}), "
          f"but the induced coordinates A^T eta (the dual coordinates of the restricted potential psi(A theta - b)) lie on the line -t c to {ind_def:.1e}: the central path of a problem with equality constraints is an m-geodesic of the induced structure of the feasible affine set, not of the ambient one")
    # ---- 5e. the cone programme: -log det X
    rng = rng_for(61)
    Q = rng.normal(size=(3, 3)); X = Q @ Q.T + 0.5 * np.eye(3)
    dX = rng.normal(size=(3, 3)); dX = 0.5 * (dX + dX.T)
    f = lambda Y: -np.linalg.slogdet(Y)[1]
    hh = 1e-4
    hess = (f(X + hh * dX) - 2 * f(X) + f(X - hh * dX)) / hh ** 2
    fisher_form = float(np.trace(np.linalg.solve(X, dX) @ np.linalg.solve(X, dX)))
    def klg(P, Qm):
        return 0.5 * (np.trace(np.linalg.solve(Qm, P)) - 3 - np.linalg.slogdet(np.linalg.solve(Qm, P))[1])
    e_ = 1e-3
    fisher_kl = 2 * klg(X, X + e_ * dX) / e_ ** 2
    Y = Q[:, :] @ Q.T * 0.7 + 0.8 * np.eye(3)
    Dpsi = f(X) - f(Y) - float(np.trace(-np.linalg.inv(Y) @ (X - Y)))             # Bregman divergence of psi = -log det at (X, Y)
    print(f"   barrier -log det X (13.185), a random 3 x 3 positive-definite X and direction dX: second derivative {hess:.6f} = tr(X^(-1) dX X^(-1) dX) = {fisher_form:.6f}; Fisher metric of N(0, X) in the covariance coordinates, 2 KL[N(0,X) : N(0, X + e dX)]/e^2 -> g(dX, dX) = {fisher_kl:.6f} "
          f"(= (1/2) tr(...) = {0.5 * fisher_form:.6f} doubled): the barrier metric is twice the Fisher metric; Bregman divergence D_psi[X : Y] = {Dpsi:.6f} = 2 KL[N(0,X) : N(0,Y)] = {2 * klg(X, Y):.6f}, with X (the e-affine coordinate of the barrier) playing the role of the m-affine (covariance) coordinate of the Gaussian family:"
          " the same geometry as the zero-mean Gaussians with e and m exchanged and a factor 2")


# ================================================================== 6. dual geometry derived from game theory (13.6)

def hyv_efficiency(T, dT, ddT, k, dk, ddk, theta, xs, ws):
    """Hyvarinen estimator of an exponential family p ~ exp(theta . T(x) + k(x)): s_a = T_a'' + (theta . T' + k') T_a' (the theta-derivative of l'' + l'^2/2),
    K_ab = E[T_a' T_b'], V = E[s s^T], G = Cov[T].  Returns (G^-1 K^-1-sandwich comparison): the matrices G^(-1) and K^(-1) V K^(-T), and the efficiency det ratio."""
    th = np.asarray(theta, float)
    Tx = np.stack([f(xs) for f in T]); dTx = np.stack([f(xs) for f in dT]); ddTx = np.stack([f(xs) for f in ddT])
    logp = th @ Tx + k(xs)
    p = np.exp(logp - logp.max()); Zc = float(np.sum(ws * p)); p = p / Zc
    E = lambda f: np.sum(ws * p * f, axis=-1)
    mT = E(Tx)
    G = (Tx - mT[:, None]) * (ws * p) @ (Tx - mT[:, None]).T
    lp1 = th @ dTx + dk(xs)
    sv = ddTx + lp1[None, :] * dTx                                  # (a, nodes)
    K = (dTx * (ws * p)) @ dTx.T
    V = (sv * (ws * p)) @ sv.T
    Ki = np.linalg.inv(K)
    asy = Ki @ V @ Ki.T
    Gi = np.linalg.inv(G)
    return Gi, asy, G, K, V, sv, Tx - mT[:, None], p


def check_game():
    head("6. Dual geometry derived from game theory (section 13.6)")
    # ---- 6a. Lemma 13.3 and the boundary term
    xg, wg = gl_pieces(list(np.linspace(-14, 14, 29)), 40)
    w1, w2, mu1, mu2, s1, s2 = 0.6, 0.4, -1.0, 1.5, 0.8, 1.2
    def gm(x, k):
        g1 = w1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2) / (s1 * math.sqrt(2 * math.pi)); g2 = w2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2) / (s2 * math.sqrt(2 * math.pi))
        d1 = [g1, g1 * (-(x - mu1) / s1 ** 2), g1 * (((x - mu1) / s1 ** 2) ** 2 - 1 / s1 ** 2)]
        d2 = [g2, g2 * (-(x - mu2) / s2 ** 2), g2 * (((x - mu2) / s2 ** 2) ** 2 - 1 / s2 ** 2)]
        return d1[k] + d2[k]
    pp = gm(xg, 0); lp1 = gm(xg, 1) / pp; lp2 = gm(xg, 2) / pp - lp1 ** 2
    mq, sq = 0.3, 1.1
    lq1 = -np.tanh((xg - mq) / (2 * sq)) / sq; lq2 = -0.5 / sq ** 2 / np.cosh((xg - mq) / (2 * sq)) ** 2
    S_q = lq2 + 0.5 * lq1 ** 2; S_p = lp2 + 0.5 * lp1 ** 2
    lhs = float(np.sum(wg * pp * (S_q - S_p))); rhs = float(0.5 * np.sum(wg * pp * (lp1 - lq1) ** 2))
    print(f"   Lemma 13.3: p = 0.6 N(-1, 0.8^2) + 0.4 N(1.5, 1.2^2), q = logistic(0.3, 1.1), Hyvarinen score S = l'' + (1/2) l'^2: E_p[S(x,q) - S(x,p)] = {lhs:.12f} and (1/2) integral p (l_p' - l_q')^2 = {rhs:.12f}; "
          f"the game entropy (13.239) -(1/2) E_p[l'^2] = {-0.5 * float(np.sum(wg * pp * lp1 ** 2)):.6f} equals E_p[S(x,p)] = {float(np.sum(wg * pp * S_p)):.6f}")
    print("   (13.244): q enters only through l_q' and l_q'', which do not change if q is multiplied by a constant: D[p : c q] = D[p : q] and q need not be normalised")
    print(f"   boundary condition: p = Exp(1) on (0, inf), q = Exp(2) (l_p' = -1, l_q' = -2, all l'' = 0): E_p[S(x,q) - S(x,p)] = 2 - 1/2 = {2.0 - 0.5:.1f} but (1/2) E_p[(l_p' - l_q')^2] = {0.5:.1f}: the partial integration needs p l_q' -> 0 at the ends of the support; "
          f"the missing term is -p(0)(l_q'(0) - l_p'(0)) = {1.0:.1f}, and {0.5:.1f} + {1.0:.1f} = {1.5:.1f}: Lemma 13.3 holds only under that hypothesis, which the book does not state; "
          f"for q = Exp(1/2) (l_q' = -1/2) the excess E_p[S(x,q) - S(x,p)] = (1/4 - 1)/2 = {(0.25 - 1) / 2:+.3f} < 0: on the half line the score is not even proper without the hypothesis")
    # ---- 6b. Example 13.1
    th = 1.7
    xq, wq = gl_pieces([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], 60)
    Zf = lambda t: float(np.sum(wq * np.exp(-t * xq ** 3)))
    psi = lambda t: math.log(Zf(t))
    h = 1e-5
    dpsi = (psi(th + h) - psi(th - h)) / (2 * h)
    print(f"   Example 13.1, p = exp(-theta x^3 - psi(theta)) on x > 0, theta = {th}: the normaliser is Z = Gamma(4/3) theta^(-1/3) (quadrature {Zf(th):.10f}, closed form {math.exp(math.lgamma(4 / 3)) * th ** (-1 / 3):.10f}), so psi = -(1/3) log theta + log Gamma(4/3);"
          f" psi'(theta) by differences = {dpsi:.8f} = -1/(3 theta) = {-1 / (3 * th):.8f}: the printed psi = +(1/3) log theta + c (13.248) has the wrong sign, and eta = 1/(3 theta) is E[x^3], not the expectation -1/(3 theta) of the statistic -x^3 that goes with theta")
    x0 = 0.9
    Sx = lambda t, x: (-6 * t * x) + 0.5 * (3 * t * x ** 2) ** 2           # l'' + (1/2) l'^2 with l = -theta x^3 - psi
    s_fd = (Sx(th + h, x0) - Sx(th - h, x0)) / (2 * h)
    print(f"      Hyvarinen score S(x, theta) = l'' + l'^2/2, l' = -3 theta x^2, l'' = -6 theta x; d S/d theta at x = {x0}: finite differences {s_fd:.6f}, -6x + 9 theta x^4 = {-6 * x0 + 9 * th * x0 ** 4:.6f}, "
          f"the printed (13.251) -6x - 9 theta x^4 = {-6 * x0 - 9 * th * x0 ** 4:.6f}")
    mom = lambda j, t: math.exp(math.lgamma((j + 1) / 3) - math.lgamma(1 / 3)) * t ** (-j / 3)       # E[x^j]
    Es_ok = -6 * mom(1, th) + 9 * th * mom(4, th); Es_pr = -6 * mom(1, th) - 9 * th * mom(4, th)
    print(f"      E[s] = {Es_ok:+.1e} with the corrected sign (an estimating function), {Es_pr:+.4f} with the printed sign; the estimator (13.252) theta_hat = (2/3) sum x_i / sum x_i^4 solves the corrected equation (the printed one would give a negative value); the score (13.253) is d_theta log p = -x^3 - psi'(theta), not x^3 - psi'")
    K = 9 * mom(4, th)
    V = 36 * mom(2, th) - 108 * th * mom(5, th) + 81 * th ** 2 * mom(8, th)
    G = 1 / (3 * th ** 2)
    var_h = V / K ** 2; var_m = 1 / G
    print(f"      asymptotic variances (per sample, theta = {th}): Fisher bound 1/G = 3 theta^2 = {var_m:.6f}; Hyvarinen estimator K^(-1) V K^(-1) = {var_h:.6f} (K = E[ds/dtheta] = {K:.6f}, V = E[s^2] = {V:.6f}): efficiency {var_m / var_h:.6f}, the same at every theta (a scale family)")
    Esdl = 6 * mom(4, th) - 2 / th * mom(1, th) - 9 * th * mom(7, th) + 3 * mom(4, th)
    c_ = Esdl / G
    A_ = V / c_ ** 2 - 2 * Esdl / c_ + G
    print(f"      decomposition (13.228)-(13.236): E[s d_theta l] = {Esdl:.6f}, c = E[s dl]/G = {c_:.6f}; with s divided by c, A = E[a^2] = {A_:.6f} and G^(-1) + G^(-1) A G^(-1) = {var_m + var_m * A_ * var_m:.6f} = K^(-1) V K^(-1) = {var_h:.6f} (13.235); "
          f"E[ds/dtheta] = -E[s dl] = {-Esdl:.6f} (so after the division K = -G, (13.230)); information lost, G^(-1) A G^(-1) = {var_m * A_ * var_m:.6f}")
    rng = rng_for(71)
    N, R = 400, 30000
    th_mle = np.empty(R); th_h = np.empty(R)
    for i in range(0, R, 10000):
        y = rng.gamma(1 / 3, 1.0, size=(10000, N))                    # theta x^3 ~ Gamma(1/3)
        x = (y / th) ** (1 / 3)
        th_mle[i:i + 10000] = N / (3 * np.sum(x ** 3, axis=1))
        th_h[i:i + 10000] = (2 / 3) * np.sum(x, axis=1) / np.sum(x ** 4, axis=1)
    ex_var = th ** 2 * (N / 3) ** 2 / ((N / 3 - 1) ** 2 * (N / 3 - 2))          # exact variance of N theta / (3 sum y): inverse-gamma moments
    print(f"      Monte Carlo, N = {N}, {R} replications (exact samples: theta x^3 ~ Gamma(1/3)): N Var(theta_hat) = {N * np.var(th_mle):.4f} (MLE; asymptotic bound {var_m:.4f}, exact finite-N value {N * ex_var:.4f}), "
          f"{N * np.var(th_h):.4f} (Hyvarinen, asymptotic {var_h:.4f}); ratio of the two measured variances {np.var(th_mle) / np.var(th_h):.4f} against the efficiency {var_m / var_h:.4f}; "
          f"means {th_mle.mean():.4f}, {th_h.mean():.4f} (theta = {th}): both asymptotically unbiased, the second uses {100 * var_m / var_h:.1f}% of the information")
    STORE["hyv_mc"] = (th_mle, th_h, th, N, var_m, var_h)
    # ---- 6c. Theorem 13.5: where is the Hyvarinen estimator efficient?
    xs_, ws_ = gl_pieces(list(np.linspace(-12, 12, 25)), 40)
    Gi, asy, G_, K_, V_, sv, sc, pden = hyv_efficiency([lambda x: x, lambda x: x ** 2], [lambda x: np.ones_like(x), lambda x: 2 * x], [lambda x: np.zeros_like(x), lambda x: 2 * np.ones_like(x)],
                                                       lambda x: np.zeros_like(x), lambda x: np.zeros_like(x), lambda x: np.zeros_like(x), [0.3, -0.45], xs_, ws_)
    # s lies in the span of 1 and the two score components: residual a of the projection
    B_ = np.vstack([np.ones_like(xs_), sc]) * np.sqrt(ws_ * pden); Q_, _ = np.linalg.qr(B_.T)
    res = [float(np.linalg.norm((sv[a] * np.sqrt(ws_ * pden)) - Q_ @ (Q_.T @ (sv[a] * np.sqrt(ws_ * pden))))) for a in range(2)]
    print(f"   Theorem 13.5, Gaussian family N(mu, sigma^2) in the natural parameters (T = (x, x^2)): the Hyvarinen function s lies in the span of 1 and the score components (residual norms {res[0]:.1e}, {res[1]:.1e}); the sandwich K^(-1) V K^(-T) = {np.round(asy, 6).tolist()} equals G^(-1) = {np.round(Gi, 6).tolist()} to {np.abs(asy - Gi).max():.1e}: efficient")
    Esl = (sv * (ws_ * pden)) @ sc.T                                  # E[s dl^T]
    a_ok = sv - Esl @ np.linalg.inv(G_) @ sc
    a_pr = sv - np.linalg.inv(G_) @ Esl @ sc
    nrm_a = lambda a_: float(np.sqrt(np.sum((a_ * a_) * (ws_ * pden))))
    print(f"      (13.233) for this two-parameter family: the part of s orthogonal to the score, a = s - E[s dl^T] G^(-1) dl, has norm {nrm_a(a_ok):.1e}; with the matrices in the printed order, a = s - G^(-1) E[s dl^T] dl, the norm is {nrm_a(a_pr):.3f}"
          " (G^(-1) and E[s dl^T] do not commute): the printed order is right only for a scalar parameter")
    print("   one-parameter families p ~ exp(-theta |x|^k): s = T'' + theta T'^2 with T = -|x|^k; efficiency = G K^2 / V (G = 1/(k theta^2)); by quadrature and from the Gamma-function moments:")
    effs = {}
    for kk in (2, 3, 4, 5, 6):
        xs2, ws2 = gl_pieces([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 9], 40)
        xs2 = np.concatenate([-xs2[::-1], xs2]); ws2 = np.concatenate([ws2[::-1], ws2])
        Gi2, asy2, G2, K2, V2, *_ = hyv_efficiency([lambda x, kk=kk: -np.abs(x) ** kk], [lambda x, kk=kk: -kk * np.abs(x) ** (kk - 1) * np.sign(x)], [lambda x, kk=kk: -kk * (kk - 1) * np.abs(x) ** (kk - 2)],
                                                   lambda x: np.zeros_like(x), lambda x: np.zeros_like(x), lambda x: np.zeros_like(x), [1.0], xs2, ws2)
        eff = float((Gi2 / asy2)[0, 0])
        mk = lambda j: math.exp(math.lgamma((j + 1) / kk) - math.lgamma(1 / kk))
        Kc = kk ** 2 * mk(2 * kk - 2)
        Vc = kk ** 2 * (kk - 1) ** 2 * mk(2 * kk - 4) - 2 * kk ** 3 * (kk - 1) * mk(3 * kk - 4) + kk ** 4 * mk(4 * kk - 4)
        effc = (Kc ** 2 / Vc) / (1 / kk)
        effs[kk] = (eff, effc)
    print("      " + "; ".join(f"k = {kk}: {effs[kk][0]:.6f} (closed form {effs[kk][1]:.6f})" for kk in effs) + "   (k = 2 is the Gaussian, k = 3 the book's Example 13.1)")
    near = {}
    for dl in (0.02, 0.1, 0.5, 2.0):
        xs3, ws3 = gl_pieces(list(np.linspace(-8, 8, 33)), 30)
        Gi3, asy3, *_ = hyv_efficiency([lambda x: -x ** 2], [lambda x: -2 * x], [lambda x: -2 * np.ones_like(x)], lambda x, dl=dl: -dl * x ** 4, lambda x, dl=dl: -4 * dl * x ** 3, lambda x, dl=dl: -12 * dl * x ** 2, [1.0], xs3, ws3)
        near[dl] = float((Gi3 / asy3)[0, 0])
    print("   a Gaussian plus a small quartic term, p ~ exp(-theta x^2 - delta x^4): efficiency " + ", ".join(f"{near[d_]:.4f} (delta = {d_})" for d_ in near)
          + ": it falls away from 1 as soon as the family leaves the Gaussian, as the theorem says; the printed proof that this happens 'only' for Gaussians is one sentence")
    STORE["hyv_eff"] = (effs, near)
    # ---- 6d. the discrete case: graph Laplacian score (13.254)-(13.265)
    def cube_C(nbits):
        N_ = 2 ** nbits; C = -np.eye(N_)
        for a in range(N_):
            for bit in range(nbits):
                C[a, a ^ (1 << bit)] = 1.0 / nbits
        return C
    def path_C(N_):
        C = -np.eye(N_)
        for a in range(N_):
            nb = [a2 for a2 in (a - 1, a + 1) if 0 <= a2 < N_]
            for a2 in nb:
                C[a, a2] = 1.0 / len(nb)
        return C
    rng = rng_for(72)
    out = {}
    for name, C in (("3-cube (homogeneous graph)", cube_C(3)), ("path with 6 nodes (|N_x| = 1, 2, ..., 2, 1)", path_C(6))):
        N_ = len(C)
        def Lap(f): return C @ f
        def LapT(f): return C.T @ f
        pdist = rng.random(N_) + 0.2; pdist /= pdist.sum(); qdist = rng.random(N_) + 0.2; qdist /= qdist.sum()
        f_, hh = rng.normal(size=N_), rng.normal(size=N_)
        lemma = abs(float(np.sum(Lap(f_) * hh) - np.sum(f_ * LapT(hh))))
        sym_gap = fro(C - C.T)
        Sfun = lambda pv: (Lap(pv) / pv) ** 2 - 2 * LapT(Lap(pv) / pv)                      # (13.261) with Delta' where the proof needs it
        Sfun_book = lambda pv: (Lap(pv) / pv) ** 2 - 2 * Lap(Lap(pv) / pv)                  # (13.261) literally, Delta' = Delta
        lhs_ = float(np.sum(pdist * (Sfun_book(qdist) - Sfun_book(pdist))))
        lhs2 = float(np.sum(pdist * (Sfun(qdist) - Sfun(pdist))))
        rhs_ = float(np.sum(pdist * (Lap(qdist) / qdist - Lap(pdist) / pdist) ** 2))
        out[name] = (lemma, sym_gap, lhs_, lhs2, rhs_)
        print(f"   {name}: Lemma (13.257) sum_x (Delta f) h - sum_x f (Delta' h) = {lemma:.1e} with Delta' the transpose of C; |C - C^T| = {sym_gap:.2f} (0 iff Delta' = Delta, (13.259)); "
              f"E_p[S(x,q) - S(x,p)] with (13.261) literally (Delta' = Delta): {lhs_:.10f}; with Delta' in the second term: {lhs2:.10f}; E_p[(Delta q/q - Delta p/p)^2] = {rhs_:.10f}")
    STORE["disc"] = out
    C3 = cube_C(3); N3 = 8
    mins = []; zero_gap = []
    for _ in range(2000):
        pv = rng.random(N3) + 0.05; pv /= pv.sum(); qv = rng.random(N3) + 0.05; qv /= qv.sum()
        S_ = lambda zz: (C3 @ zz / zz) ** 2 - 2 * C3 @ (C3 @ zz / zz)
        mins.append(float(np.sum(pv * (S_(qv) - S_(pv)))))
        zero_gap.append(abs(float(np.sum(pv * (S_(5.0 * pv) - S_(pv))))))
    pv = rng.random(N3) + 0.05; pv /= pv.sum(); qv = rng.random(N3) + 0.05; qv /= qv.sum()
    S_ = lambda zz: (C3 @ zz / zz) ** 2 - 2 * C3 @ (C3 @ zz / zz)
    scale_gap = abs(float(np.sum(pv * (S_(3.0 * qv) - S_(3.0 * pv)))) - float(np.sum(pv * (S_(qv) - S_(pv)))))
    sign_gap = float(np.abs(((-C3) @ qv / qv) ** 2 - 2 * (-C3) @ ((-C3) @ qv / qv) - ((C3 @ qv / qv) ** 2 - 2 * C3 @ (C3 @ qv / qv))).max())
    print(f"   on the 3-cube, 2000 random pairs (p, q): the divergence of (13.261) is non-negative (smallest {min(mins):.2e}) and vanishes for q proportional to p (largest value at q = 5p: {max(zero_gap):.1e}), "
          f"S does not depend on the normalisation of p (change {scale_gap:.1e} when both are multiplied by 3), and it is the same for Delta and -Delta (the sign conventions of (13.254) and (13.256) are opposite; difference {sign_gap:.1e})")
    # ---- 6e. Bregman scores and the sign of (13.214)
    rng = rng_for(73)
    p5 = rng.random(5) + 0.1; p5 /= p5.sum(); q5 = rng.random(5) + 0.1; q5 /= q5.sum()
    psis = {"psi(u) = -u log u (concave, as in (13.216))": (lambda u: -u * np.log(u), lambda u: -np.log(u) - 1, lambda u: -1 / u),
            "psi(u) = u^2/2 (strictly convex)": (lambda u: 0.5 * u * u, lambda u: u, lambda u: np.ones_like(u))}
    print("   (13.214)-(13.217) on a five-point alphabet: D_psi[p:q] as printed = sum [psi(q) - psi(p) + (p - q) psi'(q)], game score S(x,q) = psi'(q(x)) + sum_y [psi(q(y)) - q(y) psi'(q(y))] (13.215):")
    for lab, (ps, dps, ddps) in psis.items():
        Dp = float(np.sum(ps(q5) - ps(p5) + (p5 - q5) * dps(q5)))
        Sq = lambda qq: dps(qq) + float(np.sum(ps(qq) - qq * dps(qq)))
        gap = float(np.sum(p5 * (Sq(q5) - Sq(p5))))
        print(f"      {lab}: D_psi[p:q] = {Dp:+.6f} (the score's divergence E_p[S(x,q) - S(x,p)] = {gap:+.6f}); the log score is recovered: S(x, q) {'= -log q(x) to ' + format(float(np.abs(Sq(q5) - (-np.log(q5))).max()), '.1e') if 'concave' in lab else 'is not the log score'}")
    kl_gen = float(np.sum(p5 * np.log(p5 / q5) - p5 + q5))
    print(f"      generalised KL sum p log(p/q) - p + q = {kl_gen:+.6f}; so the printed D_psi is a divergence (non-negative) only for concave psi (the printed sentence says convex); for convex psi it equals minus the Bregman divergence, and the 'score' is improper (E_p S(x,q) is largest at q = p)")
    # estimating function (13.217): derivative of the score along a one-parameter family
    def fam(xi):
        z = np.array([0.0, 0.8 * xi, 0.5, -0.3 * xi ** 2, 1.1]); e = np.exp(z); return e / e.sum()
    xi0 = 0.7; hx = 1e-6
    dps = psis["psi(u) = -u log u (concave, as in (13.216))"]
    Sfull = lambda qq: dps[1](qq) + float(np.sum(dps[0](qq) - qq * dps[1](qq)))        # the game score S(x, q) for every x, as a vector (the sum over y is a constant)
    s_fd = (Sfull(fam(xi0 + hx)) - Sfull(fam(xi0 - hx))) / (2 * hx)
    pf = fam(xi0); dpf = (fam(xi0 + hx) - fam(xi0 - hx)) / (2 * hx)
    s_form = dps[2](pf) * dpf - float(np.sum(pf * dps[2](pf) * dpf))
    print(f"      (13.217)-(13.218): d/dxi of the score along a one-parameter family against psi''(p) d_xi p - c(xi) with c = E[psi''(p) d_xi p]: difference {np.abs(s_fd - s_form).max():.1e}")
    # ---- 6f. (not in the book) the game has a saddle point: log loss against a set of distributions with a given mean
    xs4 = np.arange(4.0); mean = 1.2
    def tilt(lam):
        w = np.exp(lam * xs4); return w / w.sum()
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tilt(mid) @ xs4 < mean: lo = mid
        else: hi = mid
    lam = 0.5 * (lo + hi); pstar = tilt(lam); Hstar = float(-np.sum(pstar * np.log(pstar)))
    verts = []
    for i in range(4):
        for j in range(i + 1, 4):
            if xs4[i] <= mean <= xs4[j]:
                a_ = (xs4[j] - mean) / (xs4[j] - xs4[i]); v = np.zeros(4); v[i] = a_; v[j] = 1 - a_; verts.append(v)
    V_ = np.array(verts)
    print(f"   (not in the book) the game of 13.6.1 with Nature restricted to the m-flat set Gamma = {{p on {{0,1,2,3}} : E[x] = {mean}}} and log loss: the maximum-entropy member is the exponential tilt p* = {np.round(pstar, 5).tolist()} (lambda = {lam:.5f}), H(p*) = {Hstar:.6f};"
          f" the vertices of Gamma (two-point laws) give E_v[-log p*] = {np.round(V_ @ -np.log(pstar), 10).tolist()}, all equal to H(p*): p* is an equaliser (log p* is linear in x)")
    g_ = np.linspace(0.02, 0.96, 95); best = (1e9, None)
    Q1, Q2, Q3 = np.meshgrid(g_, g_, g_, indexing="ij")
    Q4 = 1 - Q1 - Q2 - Q3
    ok = Q4 > 0.005
    Qs = np.stack([Q1[ok], Q2[ok], Q3[ok], Q4[ok]], axis=1)
    worst = np.max(-np.log(Qs) @ V_.T, axis=1)                                        # max over the vertices of E_v[-log q]: the maximum over Gamma of a linear function is at a vertex
    i_ = int(np.argmin(worst))
    q0 = Qs[i_].copy(); g2 = np.linspace(-0.03, 0.03, 31)
    D1, D2, D3 = np.meshgrid(g2, g2, g2, indexing="ij")
    Qf = np.stack([q0[0] + D1.ravel(), q0[1] + D2.ravel(), q0[2] + D3.ravel(), q0[3] - D1.ravel() - D2.ravel() - D3.ravel()], axis=1)
    Qf = Qf[np.all(Qf > 0, axis=1)]
    wf = np.max(-np.log(Qf) @ V_.T, axis=1); jf = int(np.argmin(wf))
    print(f"      brute force over {len(Qs)} distributions q of the statistician on a grid of spacing 0.01 and then 0.002 around the best: min over q of max over Gamma of E_p[-log q] = {wf[jf]:.6f} at q = {np.round(Qf[jf], 3).tolist()}; "
          f"H(p*) = {Hstar:.6f}, p* = {np.round(pstar, 3).tolist()}: minimax value = maximum entropy, saddle point at (p*, p*)")
    worst = wf; i_ = jf; Qs = Qf
    kls = [float(np.sum(v[v > 0] * np.log(v[v > 0] / pstar[v > 0]))) for v in V_]
    print(f"      Pythagoras for log loss: for p in Gamma, KL[p || p*] = H(p*) - H(p): at the vertices {np.round(kls, 6).tolist()} against {np.round([Hstar - float(-np.sum(v[v > 0] * np.log(v[v > 0]))) for v in V_], 6).tolist()}"
          " (the e-projection of the uniform law onto the m-flat set Gamma, Chapter 1)")
    STORE["minimax"] = (pstar, Hstar, float(worst[i_]))


import re


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
  .halo{stroke:#fdfdfc;stroke-width:3.4px;stroke-linejoin:round;paint-order:stroke}
  .dot{stroke-width:1.2;fill:none;stroke-dasharray:1.5 3.5;stroke-linecap:round}
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


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;")


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


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
    s = s.replace("&", "&amp;").replace("<", "&lt;")
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


def cap(b, x, y, text, width=122, lh=15):
    """A caption wrapped to the figure width (the wrapped lines go through the T markup)."""
    import textwrap
    note(b, x, y, textwrap.wrap(text, width), "sm", lh)


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




# ================================================================== figures

def hyv_eff_k(kk):
    """Efficiency of the Hyvarinen estimator for p ~ exp(-theta |x|^k), from Gamma-function moments."""
    mk = lambda j: math.exp(math.lgamma((j + 1) / kk) - math.lgamma(1 / kk))
    Kc = kk ** 2 * mk(2 * kk - 2)
    Vc = kk ** 2 * (kk - 1) ** 2 * mk(2 * kk - 4) - 2 * kk ** 3 * (kk - 1) * mk(3 * kk - 4) + kk ** 4 * mk(4 * kk - 4)
    return (Kc ** 2 / Vc) * kk


def xerr_index(C):
    a2 = C ** 2
    return float(np.sum(a2.sum(1) / a2.max(1) - 1) + np.sum(a2.sum(0) / a2.max(0) - 1))


def fig_pca(path):
    b = []
    W, H = 800, 430
    res = STORE["pca_flows"]
    P1 = Panel(b, 70, 56, 320, 250, (0, 24), (-16.5, 3))
    P1.frame([0, 6, 12, 18, 24], [-16, -12, -8, -4, 0], "t", "log10 of the defect of M^T M", "Is the Stiefel manifold stable?", True)
    spec = (("oja", "ln s1", "(13.31): stable"), ("n35", "ln s3", "(13.35): neutral"), ("n36_printed", "ln s2", "(13.36) as printed"), ("ascent", "ln s4", "(13.34): unstable"), ("n40", "dash s0", "(13.40): back to D^2"))
    for name, cls, lab in spec:
        r = res[name]
        pts = [(t, math.log10(max(fro(M.T @ M - (r["ref"] if name == "n40" else np.eye(2))), 1e-16))) for t, M in r["rec"] if t <= 24.0 and np.all(np.isfinite(M))]
        pts = [(t, min(v, 3.0)) for t, v in pts]
        P1.line([q[0] for q in pts], [q[1] for q in pts], cls)
        if name == "ascent":
            P1.dot(r["blow"], 2.6, "f4", 4.0); P1.text(r["blow"] + 0.6, 2.6, "diverges at t = %.1f" % r["blow"], "sm", "start", 0, 4)
    legend_col(b, 215, 118, [(c_, l_) for _, c_, l_ in spec], 15)
    xs_rec, xu_rec = STORE["xu_rec"]
    P2 = Panel(b, 470, 56, 300, 250, (0, 30), (0, 1.02))
    P2.frame([0, 10, 20, 30], [0, 0.25, 0.5, 0.75, 1], "t", "|cos(w_i, o_i)|", "One vector at a time, or only the subspace", True)
    O = pca_problem()[0]
    for rec, cls in ((xs_rec, "s2"), (xu_rec, "s1")):
        for i in range(2):
            cs_ = [abs(float(M[:, i] @ O[:, i])) / np.linalg.norm(M[:, i]) for _, M in rec]
            P2.line([t for t, _ in rec], cs_, f"ln {cls}" if i == 0 else f"thin {cls}")
    legend_col(b, 560, 130, [("ln s1", "weighted flow (13.33): w_1"), ("thin s1", "weighted flow: w_2"), ("ln s2", "Oja subspace flow (13.31): w_1"), ("thin s2", "Oja subspace flow: w_2")], 15)
    cap(b, 70, 376, "Left: k = 2, n = 6, started off the Stiefel manifold (defect 0.106). (13.35) conserves M^T M, so the defect never decays; (13.40) pulls it back to D^2; "
                    "the printed (13.36) shrinks M to 0 and (13.34) blows up. Right: the weighted flow returns the eigenvectors one by one, Oja's flow only a rotated basis of their span.")
    open(path, "w", encoding="utf-8").write(svg(W, H, "Matrix flows for principal and minor components",
        "Left: logarithm of the distance of M transpose M from its reference value against time for five matrix flows of section 13.1: the Oja subspace flow decays to rounding error, the flow (13.35) keeps the defect at its initial value, the flow (13.40) returns M transpose M to D squared, the printed flow (13.36) shrinks M towards zero and the ascent flow (13.34) diverges at t about 1.2. Right: absolute cosine between the i-th column of W and the i-th eigenvector for Oja's subspace flow, which stays near 0.33, and for the weighted flow of Xu, which tends to 1.", b))


def fig_ica(path):
    b = []
    W, H = 800, 790
    # (a) mutual information against the rotation angle
    P1 = Panel(b, 70, 56, 310, 250, (0, 90), (0, 0.36))
    P1.frame([0, 15, 30, 45, 60, 75, 90], [0, 0.1, 0.2, 0.3], "rotation angle t (degrees)", "I(y_1, y_2) in nats", "Two uniform sources, rotated", True)
    ang = np.linspace(0, 90, 181)
    P1.line(ang, [mi_uniform_pair(math.radians(t)) for t in ang], "ln s1")
    P1.dot(0, 0, "f3", 4.5); P1.dot(90, 0, "f3", 4.5)
    P1.dot(STORE["mi_off"], STORE["mi_pca"], "f2", 5)
    P1.text(STORE["mi_off"] - 3, STORE["mi_pca"] + 0.05, "PCA: %.1f deg off" % STORE["mi_off"], "sm halo", "end")
    P1.text(STORE["mi_off"] - 3, STORE["mi_pca"] + 0.022, "I = %.3f nats" % STORE["mi_pca"], "sm halo", "end")
    P1.text(46, 0.325, "1 - ln 2 at 45 deg", "sm", "start"); P1.text(3, 0.025, "sources: I = 0", "sm", "start")
    # (b) stability map for the cubic nonlinearity
    P2 = Panel(b, 470, 56, 300, 250, (-1.5, 3.5), (-1.5, 3.5))
    P2.frame([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3], "excess kurtosis of source 1", "excess kurtosis of source 2", "Cubic phi: kappa_1 kappa_2 > 1", True)
    k1 = np.linspace(-1.5, 3.0, 200)
    k2 = 9 / (3 + k1) - 3
    poly(b, [(P2.X(-1.5), P2.Y(-1.5)), (P2.X(3.0), P2.Y(-1.5))] + [(P2.X(x), P2.Y(y)) for x, y in zip(k1[::-1], k2[::-1])] + [(P2.X(-1.5), P2.Y(-1.5))], "fillS")
    b[-1] = b[-1].replace('<polyline class="fillS"', '<polygon class="fillS"')
    P2.line(k1, k2, "ln sk")
    names = {"uniform": "U", "bimodal": "B", "logistic": "Lg", "sech": "S", "laplace": "La"}
    for rec in STORE["stab"]:
        if rec["nl"] != "cubic":
            continue
        cls = "f1" if rec["hmin"] > 1e-9 else ("f4" if abs(rec["hmin"]) <= 1e-9 else "f2")
        P2.dot(rec["k"][0], rec["k"][1], cls, 4.4); P2.dot(rec["k"][1], rec["k"][0], cls, 2.6)
        dy = {"U+B": -0.28, "U+U": 0.12}.get(names[rec["n1"]] + "+" + names[rec["n2"]], 0.1)
        P2.text(rec["k"][0] + 0.09, rec["k"][1] + dy, names[rec["n1"]] + "+" + names[rec["n2"]], "sm halo", "start")
    P2.text(-1.4, 0.35, "stable", "sm", "start"); P2.text(1.6, 2.2, "unstable", "sm", "start")
    # (c) equivariance
    P3 = Panel(b, 70, 396, 310, 250, (0, 600), (-3.2, 1.6))
    P3.frame([0, 150, 300, 450, 600], [-3, -2, -1, 0, 1], "learning step", "log10 cross-talk index", "Natural gradient: one curve for any A", True)
    eq = STORE["equiv"]
    def curve(tr, cls, maxn=600):
        xs_, ys_ = [], []
        for i in range(0, min(len(tr), maxn), 4):
            c_ = tr[i]
            if not np.all(np.isfinite(c_)) or np.abs(c_).max() > 1e6:
                break
            xs_.append(i); ys_.append(math.log10(max(xerr_index(c_), 1e-4)))
        P3.line(xs_, ys_, cls)
    curve(eq["ord"][0], "ln s3"); curve(eq["ord"][1], "ln s2"); curve(eq["nat"][0], "ln s1"); curve(eq["nat"][1], "dash s0")
    legend_col(b, 170, 424, [("ln s1", "natural, A_1"), ("dash s0", "natural, A_2 (same curve)"), ("ln s3", "ordinary, A_1"), ("ln s2", "ordinary, A_2: diverges")], 15)
    # (d) predicted against measured linear response
    P4 = Panel(b, 470, 396, 300, 250, (0, 1.8), (0, 1.8))
    P4.frame([0, 0.4, 0.8, 1.2, 1.6], [0, 0.4, 0.8, 1.2, 1.6], "predicted exp(-h_min t)", "measured response", "The learning rule against the Hessian", True)
    P4.line([0, 1.8], [0, 1.8], "dash s0")
    lab = {"uniform": "U", "bimodal": "B", "logistic": "Lg", "laplace": "La"}
    offs = {("uniform", "uniform", "cubic"): (0.06, 0.1), ("uniform", "bimodal", "cubic"): (0.06, -0.1), ("logistic", "laplace", "tanh"): (0.05, -0.09), ("uniform", "uniform", "tanh"): (-0.5, -0.1)}
    for (n1, n2, nl, h_, pred, r1, r2, ext, se) in STORE["stab_runs"]:
        P4.dot(pred, r1, "f4", 3.2); P4.dot(pred, ext, "f1", 4.5)
        dx, dy = offs.get((n1, n2, nl), (0.05, -0.08))
        P4.text(pred + dx, ext + dy, f"{lab[n1]}+{lab[n2]} {nl}", "sm", "start")
    b.append('<circle class="f4 ring" cx="486" cy="420" r="3.2"/><text class="sm" x="494" y="424">eps = 0.01</text><circle class="f1 ring" cx="486" cy="436" r="4.5"/><text class="sm" x="494" y="440">extrapolated to eps = 0</text>')
    cap(b, 70, 706, "(a) Second-order statistics fix the output only up to a rotation; independence picks the angle. (b) Stability of the separating point for phi(y) = y^3: "
                    "the shaded region is (3 + k_1)(3 + k_2) < 9 (U uniform, B bimodal, Lg logistic, S sech, La Laplace; small dots: the same pair with the roles swapped). "
                    "(c) The natural gradient depends on A only through the starting C = W A; the ordinary gradient does not. "
                    "(d) Mean response of paired online runs to a perturbation along the weakest eigenvector of the Hessian (t = 2) against the linearised prediction.")
    open(path, "w", encoding="utf-8").write(svg(W, H, "Independent component analysis: contrast, stability and equivariance",
        "Four panels. (a) Mutual information of the two outputs of a rotation of two independent uniform variables against the rotation angle: zero at 0 and 90 degrees, 1 minus ln 2 at 45 degrees, and 0.2945 at the offset of the principal component solution of the book's example. (b) The plane of the excess kurtoses of two sources with the region where the cubic nonlinearity gives a stable separating point, bounded by the curve (3 + k1)(3 + k2) = 9, and the tested source pairs. (c) Cross-talk index against learning step: the natural gradient gives the same curve for two mixing matrices, the ordinary gradient converges for one and diverges for the other. (d) Measured linear response of the online learning rule against the prediction from the Hessian, for four source and nonlinearity pairs.", b))


def fig_nmf(path):
    b = []
    W, H = 800, 430
    cur = STORE["nmf_curves"]
    P1 = Panel(b, 70, 56, 320, 250, (0, 300), (-2, 2.2))
    P1.frame([0, 100, 200, 300], [-2, -1, 0, 1, 2], "iteration", "log10 L", "Lee-Seung against exponential gradient", True)
    its = np.arange(len(cur["ls"]))
    for eta, cls in ((0.3, "ln s2"), (0.03, "ln s3"), (0.1, "ln s4")):
        L_ = cur["eg"][eta]
        P1.line(np.arange(len(L_)), np.clip(np.log10(np.maximum(L_, 1e-12)), -2, 2.2), cls)
    P1.line(its, np.log10(cur["ls"]), "ln s1")
    legend_col(b, 215, 116, [("ln s1", "Lee-Seung (13.124)"), ("ln s3", "EG, eta = 0.03"), ("ln s4", "EG, eta = 0.1: 6 increases"), ("ln s2", "EG, eta = 0.3: 83 increases")], 15)
    out = STORE["nmf_cone"]["data stay inside the cone"]
    slack, ok, Xc, alts, (lo, hi) = out
    P2 = Panel(b, 470, 56, 300, 250, (0, 1.3), (0, 1.3))
    P2.frame([0, 0.5, 1.0], [0, 0.5, 1.0], "x_1", "x_2", "Data inside the cone: the factor is not unique", True)
    A2 = np.array([[1.0, 0.2], [0.1, 1.0]])
    for j in range(2):
        v = A2[:, j] * 1.25 / max(A2[:, j])
        P2.line([0, v[0]], [0, v[1]], "ln s1")
    for Ap, cls in ((alts[1], "dash s3"), (alts[2], "ln s2")):
        for j in range(2):
            v = Ap[:, j] * 1.2
            P2.line([0, v[0]], [0, v[1]], cls)
    for i in range(Xc.shape[1]):
        P2.dot(Xc[0, i], Xc[1, i], "f0", 2.4)
    legend_col(b, 618, 70, [("ln s1", "true A (cone of A)"), ("dash s3", "A' at f = 0.5"), ("ln s2", "tightest A' (f = 1)")], 15)
    cap(b, 70, 376, "Left: both updates descend; exponential gradient with a fixed step is slow for small eta and not monotone for eta >= 0.1, Lee-Seung is monotone and equals "
                    "exponential gradient with the entry-wise rate 1/logmean(N, P). Right: 30 non-negative data points (grey) drawn from the cone of A but away from its edges: "
                    "every cone between the data and the quadrant gives an exact factorisation.")
    open(path, "w", encoding="utf-8").write(svg(W, H, "Non-negative matrix factorisation: updates and uniqueness",
        "Left: logarithm of the squared error against iteration for the Lee-Seung multiplicative update and for exponential gradient descent with three step sizes: Lee-Seung decreases at every step, exponential gradient with step 0.1 or larger increases at some steps. Right: in the positive quadrant, the cone spanned by the columns of the true mixing matrix, thirty data points that stay inside it, and two alternative non-negative factorisations whose cones are tighter around the data.", b))


def projection_map(G, c, lim, n=100):
    """For a grid of points theta* in the plane, the minimiser of (1/2)(th - th*)^T G (th - th*) over the diamond |th|_1 <= c: returns the grid, the projections and the kind (0 inside, 1 on an edge, 2 at a vertex)."""
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Ts = np.stack([X.ravel(), Y.ravel()], axis=1)
    verts = c * np.array([[1.0, 0], [0, 1.0], [-1.0, 0], [0, -1.0]])
    best = np.full(len(Ts), np.inf); proj = np.zeros_like(Ts); kind = np.zeros(len(Ts), int)
    for i in range(4):
        P0, P1 = verts[i], verts[(i + 1) % 4]
        d = P1 - P0
        t = np.clip(((Ts - P0) @ G @ d) / (d @ G @ d), 0, 1)
        Q = P0[None, :] + t[:, None] * d[None, :]
        val = 0.5 * np.einsum("ni,ij,nj->n", Ts - Q, G, Ts - Q)
        upd = val < best - 1e-14
        best = np.where(upd, val, best); proj = np.where(upd[:, None], Q, proj)
        kind = np.where(upd, np.where((t < 1e-9) | (t > 1 - 1e-9), 2, 1), kind)
    inside = np.abs(Ts).sum(axis=1) <= c
    proj[inside] = Ts[inside]; kind[inside] = 0
    return xs, Ts, proj, kind


def fig_sparse(path):
    b = []
    W, H = 800, 430
    cs = STORE["cs"]
    P1 = Panel(b, 70, 56, 320, 250, (10, 90), (0, 1.03))
    P1.frame([10, 30, 50, 70, 90], [0, 0.25, 0.5, 0.75, 1], "number of measurements m", "fraction of exact recoveries", "n = 200: how many measurements?", True)
    for k_, cls in zip(cs["ks"], ("s1", "s3", "s2")):
        P1.line(cs["ms"], cs["table"][k_], f"ln {cls}")
        for m_, p_ in zip(cs["ms"], cs["table"][k_]):
            P1.dot(m_, p_, "f" + cls[1], 3.2)
        xb = 2 * k_ * math.log(cs["n"])
        P1.line([xb, xb], [0, 1.0], "dot " + cls)
        P1.dot(cs["m50"][k_], 0.5, "f0", 4.6)
    legend_col(b, 255, 250, [("ln s1", "k = 3"), ("ln s3", "k = 5"), ("ln s2", "k = 8")], 15)
    G = np.array([[1.0, 0.8], [0.8, 1.6]])
    lim = 2.4; c = 1.0
    xs, Ts, proj, kind = projection_map(G, c, lim, 90)
    P2 = Panel(b, 470, 56, 300, 250, (-lim, lim), (-lim, lim))
    P2.frame([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], "theta_1", "theta_2", "Where the m-projection lands", True)
    cell = (xs[1] - xs[0])
    cls_of = {0: "fillS", 1: "fillG", 2: "fillO"}
    grid = kind.reshape(len(xs), len(xs))
    for j in range(len(xs)):                      # rows of constant theta_2, run-length merged along theta_1
        i = 0
        while i < len(xs):
            k0 = grid[i, j]; i2 = i
            while i2 + 1 < len(xs) and grid[i2 + 1, j] == k0:
                i2 += 1
            if k0 != 0:
                b.append(f'<rect class="{cls_of[k0]}" style="opacity:0.35" x="{P2.X(xs[i] - cell / 2):.1f}" y="{P2.Y(xs[j] + cell / 2):.1f}" width="{P2.X(xs[i2] + cell / 2) - P2.X(xs[i] - cell / 2):.1f}" height="{P2.Y(xs[j] - cell / 2) - P2.Y(xs[j] + cell / 2):.1f}"/>')
            i = i2 + 1
    P2.line([c, 0, -c, 0, c], [0, c, 0, -c, 0], "ln sk")
    for th_ in ((1.8, 0.2), (-1.2, 1.9), (0.4, -1.6), (-2.0, -0.7), (1.0, 1.3)):
        idx = int(np.argmin(np.abs(Ts[:, 0] - th_[0]) + np.abs(Ts[:, 1] - th_[1])))
        q = proj[idx]
        P2.line([th_[0], q[0]], [th_[1], q[1]], "thin s2"); P2.dot(th_[0], th_[1], "f0", 3.0); P2.dot(q[0], q[1], "f2", 3.6)
    fr = [float(np.mean(kind == k_)) for k_ in (0, 1, 2)]
    STORE["proj_map"] = fr
    for k_, (cl_, tx_) in enumerate((("fillG", "projection on an edge of the ball"), ("fillO", "projection on a vertex"))):
        b.append(f'<rect class="{cl_}" style="opacity:0.5" x="478" y="{70 + 15 * k_}" width="20" height="10"/><text class="sm" x="504" y="{79 + 15 * k_}">{tx_}</text>')
    cap(b, 70, 376, "Left: fraction of 30 random problems recovered exactly (m x 200 Gaussian A, k non-zeros); dotted lines: the printed threshold 2 k ln n; circles: the 50 percent points. "
                    "Right: points theta* of the plane (grey dots) and their m-projections onto the unit L1 ball for the metric G = [[1, .8], [.8, 1.6]]: "
                    "the orange regions are the normal cones of the four vertices, which is why the solution is sparse (white: points inside the ball, projected onto themselves).")
    open(path, "w", encoding="utf-8").write(svg(W, H, "Sparse recovery: the measurement threshold and the m-projection onto the L1 ball",
        "Left: fraction of exactly recovered k-sparse signals of length 200 against the number of measurements for k equal to 3, 5 and 8, with dotted vertical lines at the printed threshold 2 k ln n and circles at the measured 50 percent points, which are two to three times smaller. Right: in the plane, the diamond |theta|_1 <= 1 and the regions of points whose m-projection onto it lies in the interior of an edge or at a vertex, the latter being the normal cones that make the solution sparse.", b))


def fig_convex(path):
    b = []
    W, H = 800, 430
    A, bb, c = HEX_A, HEX_B, HEX_C
    pstar, vstar, verts = STORE["cp_vals"]
    P1 = Panel(b, 55, 56, 230, 230, (-0.4, 4.4), (-0.4, 3.4))
    P1.frame([0, 1, 2, 3, 4], [0, 1, 2, 3], "theta_1", "theta_2", "Central path and Euler steps", True)
    order = [(0, 0), (4, 0), (4, 1.5), (2.5, 3), (2, 3), (0, 2)]
    b.append('<polygon class="fillS" points="' + " ".join(f"{P1.X(x):.1f},{P1.Y(y):.1f}" for x, y in order) + '"/>')
    poly(b, [(P1.X(x), P1.Y(y)) for x, y in order + [order[0]]], "ln sk")
    th0, _ = newton_barrier(A, bb, c, 0.0, np.array([1.7, 1.2]))
    tg = np.concatenate([np.linspace(0, 2, 60), np.linspace(2, 400, 200)]); prev = th0; xs_, ys_ = [], []
    for t in tg:
        prev, _ = newton_barrier(A, bb, c, t, prev); xs_.append(prev[0]); ys_.append(prev[1])
    P1.line(xs_, ys_, "ln s1")
    for eps, cls, fc, rr in ((0.1, "thin s3", "f3", 2.6), (0.4, "ln s2", "f2", 4.0)):
        thk = th0.copy(); pts = [thk.copy()]
        for k in range(int(round(2.0 / eps))):
            thk = thk - eps * np.linalg.solve(barrier(thk, A, bb)[2], c); pts.append(thk.copy())
        P1.line([q[0] for q in pts], [q[1] for q in pts], cls)
        for q in pts:
            P1.dot(q[0], q[1], fc, rr)
    P1.dot(th0[0], th0[1], "f0", 4.2); P1.dot(vstar[0], vstar[1], "f4", 5)
    P1.text(th0[0] - 0.1, th0[1] - 0.4, "centre", "sm", "end"); P1.text(vstar[0] - 0.12, vstar[1] + 0.25, "optimum", "sm", "end")
    legend_col(b, 66, 150, [("ln s1", "central path"), ("ln s2", "Euler, eps = 0.4"), ("thin s3", "Euler, eps = 0.1")], 14)
    # curvature along the path
    P2 = Panel(b, 345, 56, 200, 230, (-1.3, 2.0), (0, 0.4))
    P2.frame([-1, 0, 1, 2], [0, 0.1, 0.2, 0.3, 0.4], "t (log scale)", "e-curvature", "How curved is the path?", True,
             xtl={-1: "0.1", 0: "1", 1: "10", 2: "100"})
    ts_ = np.logspace(-1.3, 2.0, 120); prev = th0; xs2, ys2 = [], []
    for t in ts_:
        prev, _ = newton_barrier(A, bb, c, t, prev)
        G_ = barrier(prev, A, bb)[2]; Gi = np.linalg.inv(G_); T3 = cubic_T(prev, A, bb)
        thd = -Gi @ c; thdd = -Gi @ np.einsum("ijk,j,k->i", T3, thd, thd)
        n2 = float(thd @ G_ @ thd); perp2 = float(thdd @ G_ @ thdd) - float(thd @ G_ @ thdd) ** 2 / n2
        xs2.append(math.log10(t)); ys2.append(math.sqrt(max(perp2, 0)) / n2)
    P2.line(xs2, ys2, "ln s1")
    for t_, kap in STORE["cp_curv"]:
        P2.dot(math.log10(t_), kap, "f2", 3.8)
    P2.text(-0.9, 0.36, "dots: t = 1, 5, 30", "sm", "start")
    # Euler errors
    P3 = Panel(b, 605, 56, 170, 230, (-3, 0), (-7, -0.5))
    P3.frame([-3, -2, -1, 0], [-6, -4, -2], "log10 eps", "log10 error", "Euler against the path", True)
    eu = STORE["euler"]; os_ = STORE["onestep"]
    P3.line([math.log10(e) for e, *_ in eu], [math.log10(er) for _, er, _ in eu], "ln s1")
    for e, er, _ in eu:
        P3.dot(math.log10(e), math.log10(er), "f1", 3.6)
    P3.line([math.log10(e) for e, *_ in os_], [math.log10(d1) for _, d1, _ in os_], "ln s2")
    for e, d1, _ in os_:
        P3.dot(math.log10(e), math.log10(d1), "f2", 3.2)
    P3.text(-2.95, -1.0, "global, t = 2", "sm", "start"); P3.text(-1.55, -6.3, "one step", "sm", "start")
    cap(b, 55, 346, "Left: the hexagon A theta >= b, the central path theta*(t) (curved in theta, the straight line eta = -t c in the dual coordinates), the analytic centre and the "
                    "Euler steps of the natural gradient (13.193). Middle: normal part of the acceleration theta'' per squared speed along the path, the quantity that sets the step size; it vanishes at the "
                    f"inflection t = {STORE['cp_infl']:.2f}. Right: the global error of the steps is first order in eps and the error of one step second order, as the curvature predicts.", 120)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Linear programming: the central path and natural-gradient steps",
        "Left: a hexagon in the theta plane, the central path from its analytic centre to the optimal vertex, and the Euler iterates of the natural gradient flow with step sizes 0.4 and 0.1, which follow the path with a lag. Middle: the e-curvature of the path against t on a logarithmic axis, with two humps and a zero near t equal to 2.7. Right: log-log plot of the Euler error at t equal to 2, which falls like the step size, and of the error of one step from the centre, which falls like its square.", b))


def fig_hyv(path):
    b = []
    W, H = 800, 430
    P1 = Panel(b, 70, 56, 320, 250, (2, 8), (0.3, 1.04))
    P1.frame([2, 3, 4, 5, 6, 7, 8], [0.4, 0.6, 0.8, 1.0], "exponent k in p ~ exp(-theta |x|^k)", "efficiency of the Hyvarinen estimator", "Efficient only at the Gaussian", True)
    kk = np.linspace(2, 8, 120)
    P1.line(kk, [hyv_eff_k(k_) for k_ in kk], "ln s1")
    eff, near = STORE["hyv_eff"]
    for k_ in eff:
        P1.dot(k_, eff[k_][0], "f1", 4.2)
    P1.dot(3, eff[3][0], "f2", 5.4)
    P1.text(3.2, eff[3][0] + 0.03, "Example 13.1: %.4f" % eff[3][0], "sm", "start"); P1.text(2.2, 1.0, "Gaussian: 1", "sm", "start")
    th_mle, th_h, th, N, var_m, var_h = STORE["hyv_mc"]
    P2 = Panel(b, 470, 56, 300, 250, (1.2, 2.2), (0, 3.4))
    P2.frame([1.2, 1.4, 1.6, 1.8, 2.0, 2.2], [0, 1, 2, 3], "estimate of theta", "density", f"N = {N}, {len(th_mle)} replications", True)
    edges = np.linspace(1.2, 2.2, 81)
    for arr, cls in ((th_mle, "s1"), (th_h, "s2")):
        h_, _ = np.histogram(arr, bins=edges, density=True)
        pts = []
        for k_, hv in enumerate(h_):
            pts += [(P2.X(edges[k_]), P2.Y(hv)), (P2.X(edges[k_ + 1]), P2.Y(hv))]
        poly(b, pts, f"ln {cls}")
    P2.line([th, th], [0, 3.4], "dash s0")
    legend_col(b, 480, 76, [("ln s1", "MLE: sd %.4f" % float(np.std(th_mle))), ("ln s2", "Hyvarinen: sd %.4f" % float(np.std(th_h)))], 15)
    cap(b, 70, 376, "Left: efficiency G K^2 / V of the Hyvarinen estimator of theta in the family exp(-theta |x|^k): 1 for k = 2 (Gaussian), 0.8214 for the book's example k = 3, "
                    "lower for larger k. Right: both estimators are centred on theta = 1.7; the Hyvarinen one has the larger spread (variance ratio 0.82).")
    open(path, "w", encoding="utf-8").write(svg(W, H, "The Hyvarinen score loses information outside the Gaussian",
        "Left: efficiency of the Hyvarinen score matching estimator of theta for densities proportional to exp(minus theta times the absolute value of x to the power k), equal to one for k equal to two and 0.8214 for k equal to three, the book's example, decreasing for larger k. Right: histograms of the maximum likelihood estimator and the Hyvarinen estimator of theta in 30000 samples of 400 observations from the example, both centred at 1.7, the second with a wider spread.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_pca(out / "pca-flows.svg")
    fig_ica(out / "ica-contrast-stability.svg")
    fig_nmf(out / "nmf-updates.svg")
    fig_sparse(out / "sparse-recovery.svg")
    fig_convex(out / "central-path.svg")
    fig_hyv(out / "hyvarinen-score.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


def main():
    check_pca()
    check_ica()
    check_nmf()
    check_sparse()
    check_convex()
    check_game()
    print("\nall checks ran")


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
