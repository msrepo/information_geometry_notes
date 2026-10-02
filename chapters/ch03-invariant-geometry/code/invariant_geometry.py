#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 3, checked by hand.

Invariant geometry of the manifold of probability distributions. Every number quoted in the notes comes from here.

The chapter asks which divergences and which Riemannian metric on a space of distributions deserve to be called natural.
Its answer is a criterion (information monotonicity under coarse graining, with equality exactly for sufficient statistics),
a class (the f-divergences), a metric (the Fisher information, unique up to scale) and a picture (the probability simplex is
a piece of a sphere of radius 2 inside the positive measures). The running examples are small and explicit: a three- or
six-outcome distribution on which coarse grainings can be enumerated, two Bernoulli trials (where the sum is sufficient),
Gaussians (where x -> x^2 is sufficient on one line of the family and on no larger one), a 3 x 3 table, and the three-outcome
simplex S_2 with its Fisher-Rao geometry.

Checked here, in the order the notes use them:

  1. the toolbox: standard f (f(1) = f'(1) = 0, f''(1) = 1), the dual f*(u) = u f(1/u), the freedom f + c(u-1) (3.26)-(3.32);
  2. the invariance criterion (3.1)-(3.13): merging cells, sufficiency of a pair versus sufficiency for a model, Bernoulli and
     Gaussian examples including x -> x^2;
  3. information monotonicity (3.11) and Theorem 3.1: random coarse grainings for nine f-divergences; equality exactly for
     equal conditionals; counterexamples (squared Euclid, Itakura-Saito, a non-convex f, p^a f(q/p)); the missing hypothesis of
     strict convexity (total variation, a Huber-type f); Markov kernels; the converse proof; D + D^2;
  4. the examples (3.33)-(3.46): KL, chi^2, the alpha family, Hellinger, the limits alpha -> +-1, the variation distance; which
     f-divergences are Bregman divergences;
  5. properties (3.47)-(3.53): joint convexity, the upper bounds (including a counterexample to (3.48) as printed), infinite
     values, zero forcing and zero avoidance;
  6. KL in detail (3.54)-(3.66): Sanov's lemma with exact multinomial sums, the central limit theorem and (3.56), the large
     deviation theorem against Cramer's transform and the Bahadur-Rao prefactor, and Theorem 3.2 (a factor 2);
  7. the Fisher metric from any standard f-divergence (3.67)-(3.68) and Chentsov's theorem (3.70)-(3.88): invariance under
     random Markov embeddings, competitors, the proof's computations, positive measures, the cubic tensor;
  8. positive measures (3.89)-(3.97): standard f, the sign in (3.95), Theorem 3.4, the sphere of radius 2 (volume, curvature,
     Fisher-Rao distance, Hellinger = half a chord), scale behaviour;
  9. the remark on location-scale families (Hotelling): constant negative curvature;
 10. the default readouts of the widgets of figures/interactive.html (the page computes the same quantities).

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only. All randomness uses fixed seeds, so the printed numbers do not change between runs; the checks take about five seconds.

Run:  python3 invariant_geometry.py            (checks)
      python3 invariant_geometry.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SEED = 20261002
STORE = {}                         # numbers computed by the checks, reused by the figures
FIG_P = np.array([.60, .38, .02])  # the pair of points of S_2 used for Theorem 3.2, the sphere figure and the matching widgets
FIG_Q = np.array([.02, .38, .60])
SQ3 = math.sqrt(3)


def head(s):
    print("\n" + s)


def rng_for(k):
    """A separate generator per check, so that changing one check never changes the numbers of another."""
    return np.random.default_rng(SEED + k)


def trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1])) / 2)


# ------------------------------------------------------------------ the toolbox: f-divergences

def f_alpha(a):
    """The standard alpha function: f(1) = f'(1) = 0, f''(1) = 1. Equals (3.38) plus the linear term +2(u-1)/(1-a), i.e. (3.95)
    with the sign of the linear term that makes it standard (the printed one has the opposite sign, see check 8)."""
    if a == 1:
        return lambda u: u * np.log(u) - (u - 1)
    if a == -1:
        return lambda u: -np.log(u) + (u - 1)
    return lambda u: 4 / (1 - a * a) * (1 - u ** ((1 + a) / 2)) + 2 / (1 - a) * (u - 1)


def f_alpha_book(a):
    """(3.38): the book's first definition, not standard (f'(1) = -2/(1-a))."""
    return lambda u: 4 / (1 - a * a) * (1 - u ** ((1 + a) / 2))


def f_alpha_printed(a):
    """(3.95) exactly as printed: linear term -2(u-1)/(1-a)."""
    if a == 1 or a == -1:
        return f_alpha(a)
    return lambda u: 4 / (1 - a * a) * (1 - u ** ((1 + a) / 2)) - 2 / (1 - a) * (u - 1)


def f_js(u):
    """Jensen-Shannon: D = (1/2) KL[p:m] + (1/2) KL[q:m], m = (p+q)/2.  f''(1) = 1/4, f* = f."""
    return 0.5 * (u * np.log(u) - (1 + u) * np.log((1 + u) / 2))


def f_tv(u):
    """(3.44): |1 - u|, not differentiable at 1."""
    return np.abs(1 - u)


def f_huber(u):
    """C^1 and convex, f(1) = f'(1) = 0, f''(1) = 1, but linear (not strictly convex) for u > 2."""
    return np.where(u <= 2, 0.5 * (u - 1) ** 2, u - 1.5)


def f_quartic(u):
    return (u - 1) ** 4


def f_nonconvex(u):
    """(u-1)^2 / (1 + (u-1)^2): positive, f(1) = f'(1) = 0, f''(1) = 2, bounded, hence not convex."""
    t = (u - 1) ** 2
    return t / (1 + t)


def dual(f):
    """(3.31): f*(u) = u f(1/u)."""
    return lambda u: u * f(1.0 / u)


def Df(f, p, q):
    """(3.14) for the last axis: sum_i p_i f(q_i / p_i); works on arrays of distributions."""
    return np.sum(p * f(q / p), axis=-1)


def num_d1(f, u, h=1e-6):
    return (f(u * (1 + h)) - f(u * (1 - h))) / (2 * u * h)


def num_d2(f, u, h=1e-4):
    return (f(u * (1 + h)) - 2 * f(u) + f(u * (1 - h))) / (u * h) ** 2


# the nine f-divergences used throughout: name -> (f, f''(1) if differentiable)
FUNS = {
    "KL  (alpha=-1)": (f_alpha(-1), 1.0),
    "dual KL  (alpha=+1)": (f_alpha(1), 1.0),
    "chi2  (alpha=3)": (f_alpha(3), 1.0),
    "dual chi2  (alpha=-3)": (f_alpha(-3), 1.0),
    "Hellinger  (alpha=0)": (f_alpha(0), 1.0),
    "alpha=0.5": (f_alpha(0.5), 1.0),
    "alpha=-0.5": (f_alpha(-0.5), 1.0),
    "Jensen-Shannon": (f_js, 0.25),
    "variation  |1-u|": (f_tv, None),
}


def rand_simplex(rng, shape, n, conc=1.0):
    return rng.dirichlet(np.full(n, conc), size=shape)


# ------------------------------------------------------------------ 1. the toolbox

def check_toolbox():
    head("1. The toolbox: standard f, the dual f*, the freedom f + c(u-1)  (3.26)-(3.32)")
    rng = rng_for(1)
    P = rand_simplex(rng, 2000, 5)
    Q = rand_simplex(rng, 2000, 5)
    print("   f(1), f'(1), f''(1) of each f (by finite differences; 0 means below 1e-6), then f*'(1) and f*''(1) for f*(u) = u f(1/u), and the largest gap between")
    print("   D_{f*}[p:q] and D_f[q:p] over 2000 random pairs of 5-outcome distributions (3.32):")
    z = lambda v: "0" if abs(v) < 1e-6 else f"{v:+.4f}"
    rows = []
    for name, (f, _) in FUNS.items():
        if f is f_tv:
            continue
        fs = dual(f)
        gap = float(np.max(np.abs(Df(fs, P, Q) - Df(f, Q, P))))
        rows.append((name, float(f(1.0)), float(num_d1(f, 1.0)), float(num_d2(f, 1.0)), float(num_d1(fs, 1.0)), float(num_d2(fs, 1.0)), gap))
        print(f"   {name:24s} f(1)={z(rows[-1][1])}  f'(1)={z(rows[-1][2])}  f''(1)={rows[-1][3]:.4f}   f*'(1)={z(rows[-1][4])}  f*''(1)={rows[-1][5]:.4f}   max gap {gap:.1e}")
    STORE["toolbox_rows"] = rows
    # the freedom f -> f + c (u - 1), (3.26)-(3.27): only for normalised distributions
    f = f_alpha(0.5)
    c = 0.7
    fb = lambda u: f(u) + c * (u - 1)
    p, q = P[0], Q[0]
    print(f"   (3.27) on S_n: D_(f + c(u-1)) - D_f = {Df(fb, p, q) - Df(f, p, q):+.1e} (c = 0.7);  "
          f"on positive measures m, n with sums {2.1:.1f} and {3.4:.1f}:", end=" ")
    m = 2.1 * p
    n = 3.4 * q
    print(f"{Df(fb, m, n) - Df(f, m, n):+.6f} = c (sum n - sum m) = {c * (n.sum() - m.sum()):+.6f}")
    # the book's alpha function (3.38) is not standard; its dual is f_{-alpha} only modulo the linear term
    a = 0.5
    u = np.linspace(0.2, 4, 400)
    fa_book = f_alpha_book(a)
    d_ = np.max(np.abs(dual(fa_book)(u) - f_alpha_book(-a)(u)))
    d_lin = np.max(np.abs(dual(fa_book)(u) - f_alpha_book(-a)(u) - 4 * (u - 1) / (1 - a * a)))
    d_std = np.max(np.abs(dual(f_alpha(a))(u) - f_alpha(-a)(u)))
    print(f"   alpha = 0.5: the dual of (3.38) differs from the (3.38) function of -alpha by up to {d_:.3f} = 4(u-1)/(1-alpha^2) (remaining gap {d_lin:.1e}),")
    print(f"   so (3.40) is true for the divergences only up to (3.26); for the standard functions the duality is exact (gap {d_std:.1e})")
    STORE["dual_gap_book"] = d_
    # alpha = 3 is chi^2, alpha = -3 is its dual, sum (p-q)^2/(2q) (not listed as such in the book)
    e3 = np.max(np.abs(f_alpha(3)(u) - 0.5 * (u - 1) ** 2))
    em3 = np.max(np.abs(f_alpha(-3)(u) - 0.5 * (u - 1) ** 2 / u))
    print(f"   the standard f_alpha equals (u-1)^2/2 at alpha = 3 (max gap {e3:.1e}) and (u-1)^2/(2u) at alpha = -3 ({em3:.1e}): the chi^2 divergence (3.37) is the alpha = 3 member of the family")
    # a divergence needs f''(1) > 0
    print("   a convex f with f(1) = 0 need not give a divergence: D_f[p : p + eps d] / eps^2 for p = (.5,.3,.2), d = (.1,-.04,-.06):")
    p3 = np.array([.5, .3, .2])
    d = np.array([.1, -.04, -.06])
    for nm, g in (("f = 0", lambda u: 0 * u), ("f = (u-1)^4", f_quartic), ("f = (u-1)^2/2", lambda u: 0.5 * (u - 1) ** 2)):
        vals = [float(Df(g, p3, p3 + eps * d)) / eps ** 2 for eps in (0.1, 0.01, 0.001)]
        print(f"      {nm:14s} " + "  ".join(f"{v:.3e}" for v in vals) + ("   (-> 0: no metric)" if nm != "f = (u-1)^2/2" else f"   (-> (1/2) sum d^2/p = {0.5 * np.sum(d * d / p3):.4e})"))
    STORE["half_chi_fisher"] = 0.5 * float(np.sum(d * d / p3))


# ------------------------------------------------------------------ 2. the invariance criterion

def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def kl_gauss(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 * s1 + (m1 - m2) ** 2) / (2 * s2 * s2) - 0.5


def gpdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def kl_quad(p, q, x):
    """KL by the trapezoid rule on the grid x, for two densities given as arrays on that grid."""
    with np.errstate(all="ignore"):
        return trap(np.where(p > 1e-300, p * np.log(p / np.where(q > 1e-300, q, 1e-300)), 0.0), x)


def check_invariance():
    head("2. The invariance criterion: coarse graining and sufficiency  (3.1)-(3.13)")
    # (a) merging cells
    p = np.array([.40, .30, .20, .10])
    q = np.array([.10, .20, .30, .40])
    pb = np.array([p[:2].sum(), p[2:].sum()])
    qb = np.array([q[:2].sum(), q[2:].sum()])
    print(f"   merge the outcomes {{0,1}} and {{2,3}} of p = {p.tolist()}, q = {q.tolist()}: KL[p:q] = {kl(p, q):.4f}, KL of the merged pair {np.round(pb, 4).tolist()}, {np.round(qb, 4).tolist()} = {kl(pb, qb):.4f}  (information lost: (3.4))")
    Qc = np.array([.35, .65])
    q2 = p * np.repeat(Qc / pb, 2)
    print(f"   same merge, q chosen with q_i/p_i constant on each cell (3.13): q = {np.round(q2, 4).tolist()}: KL[p:q] = {kl(p, q2):.6f}, merged {kl(pb, Qc):.6f}  (equal, gap {abs(kl(p, q2) - kl(pb, Qc)):.1e})")
    STORE["merge_example"] = (kl(p, q), kl(pb, qb), kl(p, q2), kl(pb, Qc))
    # (b) sufficient for a pair or sufficient for the model? two Bernoulli trials, s = x1 + x2
    def bern4(t):
        return np.array([(1 - t) ** 2, t * (1 - t), t * (1 - t), t * t])           # (0,0),(0,1),(1,0),(1,1)
    def binom(t):
        return np.array([(1 - t) ** 2, 2 * t * (1 - t), t * t])
    t1, t2 = 0.3, 0.8
    k4, k3 = kl(bern4(t1), bern4(t2)), kl(binom(t1), binom(t2))
    kb = t1 * math.log(t1 / t2) + (1 - t1) * math.log((1 - t1) / (1 - t2))
    cond1 = bern4(t1)[1:3] / bern4(t1)[1:3].sum()
    cond2 = bern4(t2)[1:3] / bern4(t2)[1:3].sum()
    print(f"   two Bernoulli trials, theta = {t1} and {t2}: KL on the four outcomes {k4:.6f} = KL of the sum s = x1 + x2 {k3:.6f} = 2 kl(theta:theta') {2 * kb:.6f};"
          f" p(x | s = 1) = {cond1.tolist()} and {cond2.tolist()} for both: s is sufficient for the model")
    rng = rng_for(2)
    P = rand_simplex(rng, 2000, 4)
    Q = rand_simplex(rng, 2000, 4)
    mrg = lambda A: np.stack([A[:, 0], A[:, 1] + A[:, 2], A[:, 3]], axis=1)
    gap = np.sum(P * np.log(P / Q), -1) - np.sum(mrg(P) * np.log(mrg(P) / mrg(Q)), -1)
    print(f"   the same map on the full simplex S_3 loses information for generic pairs: over 2000 random pairs the loss KL - KL-merged has minimum {gap.min():.2e} and median {np.median(gap):.4f}; "
          f"equality needs p_1/p_2 = q_1/q_2 for the pair, so on S_3 only a bijection is sufficient")
    STORE["bern_gap_median"] = float(np.median(gap))
    # (c) continuous case: invertible maps keep KL, x -> x^2 is sufficient on mu = 0 but not on the plane
    print("   continuous case, KL[N(m1,s1^2) : N(m2,s2^2)]; the image under y = g(x) computed by quadrature on the y variable:")
    cases = [(1.0, 1.0, 0.0, 2.0), (0.0, 1.0, 0.0, 2.0), (1.0, 1.0, -1.0, 1.0), (2.0, 1.0, 0.0, 1.5)]
    S = np.linspace(1e-9, 40, 400001)
    Y = np.exp(np.linspace(-45, 45, 900001))
    folded = lambda s, m, sg: gpdf(s, m, sg) + gpdf(s, -m, sg)
    lognorm = lambda y, m, s: np.exp(-0.5 * ((np.log(y) - m) / s) ** 2) / (y * s * math.sqrt(2 * math.pi))
    out = []
    for (m1, s1, m2, s2) in cases:
        kx = kl_gauss(m1, s1, m2, s2)
        kaff = kl_gauss(3 * m1 + 1, 3 * s1, 3 * m2 + 1, 3 * s2)                      # y = 3x + 1 is again Gaussian
        kexp = kl_quad(lognorm(Y, m1, s1), lognorm(Y, m2, s2), Y)                    # y = exp(x), invertible
        ksq = kl_quad(folded(S, m1, s1), folded(S, m2, s2), S)                       # y = x^2 (same information as |x|)
        out.append((m1, s1, m2, s2, kx, kaff, kexp, ksq))
        print(f"      (m1,s1;m2,s2) = ({m1:g},{s1:g};{m2:g},{s2:g}):  KL(x) = {kx:.6f}   y=3x+1: {kaff:.6f}   y=exp(x): {kexp:.6f}   y=x^2: {ksq:.6f}")
    STORE["gauss_sq"] = out
    print("   x -> x^2 is sufficient for the scale family {N(0,s^2)} (equality in the second row) but not for the Gaussians N(m,s^2): it keeps nothing of the sign of m (third row: 2 -> 0).")


# ------------------------------------------------------------------ 3. information monotonicity and Theorem 3.1

def random_merges(rng, R, n, m, sufficient=False, conc=1.0):
    """R pairs (p, q) on n outcomes and R random partitions into m non-empty cells; returns P, Q, merged P, merged Q.
    sufficient=True builds q with q_i/p_i constant on every cell, i.e. equal conditional distributions (3.13)."""
    P = rng.dirichlet(np.full(n, conc), R)
    Q = rng.dirichlet(np.full(n, conc), R)
    lab = rng.integers(0, m, (R, n))
    perm = np.argsort(rng.random((R, n)), axis=1)[:, :m]
    lab[np.arange(R)[:, None], perm] = np.arange(m)[None, :]
    onehot = (lab[..., None] == np.arange(m)).astype(float)
    Pb = np.einsum("rn,rnm->rm", P, onehot)
    if sufficient:
        Qc = rng.dirichlet(np.ones(m), R)
        Q = P * np.take_along_axis(Qc / Pb, lab, axis=1)
    Qb = np.einsum("rn,rnm->rm", Q, onehot)
    return P, Q, Pb, Qb


def d_a(a):
    """Decomposable divergence with a different degree of homogeneity: sum_i p_i^a f(q_i/p_i), f = -log u + u - 1."""
    f = f_alpha(-1)
    return lambda P, Q: np.sum(P ** a * f(Q / P), axis=-1)


def check_monotone():
    head("3. Information monotonicity and Theorem 3.1  (3.11), (3.16)-(3.24), (3.25)")
    rng = rng_for(3)
    R = 5000
    trials = [random_merges(rng, R, 6, m) for m in (2, 3, 4, 5)]
    print("   20000 random pairs of 6-outcome distributions, each merged into 2, 3, 4 or 5 cells at random. For every f-divergence (3.14):")
    print("   largest D_merged/D, smallest relative loss (D - D_merged)/D, and the share of trials with no loss at all although the partition is not sufficient:")
    rows = []
    for name, (f, _) in FUNS.items():
        ratio, loss, ties = [], [], 0
        for t in trials:
            D = Df(f, t[0], t[1]); Dm = Df(f, t[2], t[3])
            ratio.append(Dm / D); loss.append((D - Dm) / D); ties += int(np.sum(np.abs(D - Dm) < 1e-13 * D))
        ratio = np.concatenate(ratio); loss = np.concatenate(loss)
        rows.append((name, float(ratio.max()), float(loss.min()), ties / (4 * R)))
        print(f"      {name:24s} max ratio {ratio.max():.6f}   min loss {loss.min():+.2e}   no-loss share {ties / (4 * R):.4f}")
    STORE["mono_rows"] = rows
    # scatter data for the figure
    sc = {}
    for nm in ("KL  (alpha=-1)", "chi2  (alpha=3)", "Hellinger  (alpha=0)"):
        f = FUNS[nm][0]
        sc[nm] = (np.concatenate([Df(f, t[0], t[1]) for t in trials])[:400], np.concatenate([Df(f, t[2], t[3]) for t in trials])[:400])
    sqe = lambda A, B: np.sum((A - B) ** 2, -1)
    sc["squared Euclid"] = (np.concatenate([sqe(t[0], t[1]) for t in trials])[:400], np.concatenate([sqe(t[2], t[3]) for t in trials])[:400])
    STORE["mono_scatter"] = sc
    # equality for equal conditionals, all strictly convex f
    suff = [random_merges(rng, R, 6, m, sufficient=True) for m in (2, 3, 4, 5)]
    worst = {}
    for name, (f, _) in FUNS.items():
        worst[name] = max(float(np.max(np.abs(Df(f, t[0], t[1]) - Df(f, t[2], t[3])))) for t in suff)
    print("   equal conditionals (q_i/p_i constant on cells, (3.13)): largest |D - D_merged| over 20000 trials: " + ", ".join(f"{k.split('  ')[0]} {v:.0e}" for k, v in worst.items()))
    # divergences that are not f-divergences
    print("   decomposable divergences that are not f-divergences, same trials:")
    comp = {
        "squared Euclid (3.46)": lambda A, B: np.sum((A - B) ** 2, -1),
        "Itakura-Saito (Bregman, Burg entropy)": lambda A, B: np.sum(A / B - np.log(A / B) - 1, -1),
        "non-convex f = t/(1+t), t = (u-1)^2": lambda A, B: Df(f_nonconvex, A, B),
        "p^a f(q/p), a = 0.5": d_a(0.5),
        "p^a f(q/p), a = 1.5": d_a(1.5),
        "p^a f(q/p), a = 1 (the KL)": d_a(1.0),
    }
    crow = []
    for name, fn in comp.items():
        viol, mx = 0, 0.0
        for t in trials:
            D = fn(t[0], t[1]); Dm = fn(t[2], t[3]); viol += int(np.sum(Dm > D * (1 + 1e-12))); mx = max(mx, float((Dm / D).max()))
        sratio = np.concatenate([fn(t[2], t[3]) / fn(t[0], t[1]) for t in suff])
        crow.append((name, viol / (4 * R), mx, float(sratio.min()), float(sratio.max())))
        print(f"      {name:40s} violations of D_merged <= D: {viol / (4 * R):.3f}   max D_merged/D {mx:.3f};   at equal conditionals D_merged/D lies in [{sratio.min():.4f}, {sratio.max():.4f}]")
    STORE["comp_rows"] = crow
    xs_cdf = np.linspace(0, 3.2, 129)
    STORE["ratio_cdf"] = {}
    for nm, fn in (("KL divergence", lambda A, B: Df(f_alpha(-1), A, B)), ("squared Euclid", lambda A, B: np.sum((A - B) ** 2, -1)),
                   ("non-convex f", lambda A, B: Df(f_nonconvex, A, B))):
        rr = np.concatenate([fn(t[2], t[3]) / fn(t[0], t[1]) for t in trials])
        STORE["ratio_cdf"][nm] = (xs_cdf, [float(np.mean(rr <= x)) for x in xs_cdf])
    # explicit counterexamples with small numbers
    p = np.array([.05, .05, .90]); q = np.array([.15, .45, .40])
    pm = np.array([.10, .90]); qm = np.array([.60, .40])
    D1, D2 = float(Df(f_nonconvex, p, q)), float(Df(f_nonconvex, pm, qm))
    print(f"   explicit: p = {p.tolist()}, q = {q.tolist()} merged over the first two outcomes to {pm.tolist()}, {qm.tolist()}: the non-convex f gives D = {D1:.4f} before and {D2:.4f} after (information gained)")
    ps = np.array([.40, .40, .20]); qs = np.array([.20, .20, .60])
    print(f"   explicit: squared Euclid for p = {ps.tolist()}, q = {qs.tolist()}: {np.sum((ps - qs) ** 2):.2f} before, {(0.8 - 0.4) ** 2 + (0.2 - 0.6) ** 2:.2f} after merging the first two outcomes, "
          f"although they carry equal conditionals (u_1 = u_2 = 0.5)")
    STORE["nonconvex_ex"] = (D1, D2)
    # strict convexity is needed for 'equality iff sufficient'
    ps = np.array([.10, .10, .80]); qs = np.array([.30, .40, .30])
    pm = np.array([.20, .80]); qm = np.array([.70, .30])
    h1, h2 = float(Df(f_huber, ps, qs)), float(Df(f_huber, pm, qm))
    t1, t2 = float(Df(f_tv, ps, qs)), float(Df(f_tv, pm, qm))
    print(f"   strict convexity: p = {ps.tolist()}, q = {qs.tolist()} (u = 3, 4, 0.375, different conditionals) merged over the first two outcomes: Huber-type f (C^1, f''(1) = 1, linear for u > 2) gives {h1:.6f} before and {h2:.6f} after;"
          f" total variation gives {t1:.4f} and {t2:.4f}: equality although the statistic is not sufficient")
    STORE["huber_ex"] = (h1, h2, t1, t2)
    # Markov kernels (noisy coarse grainings) and embeddings
    Pn = rng.dirichlet(np.ones(6), 800); Qn = rng.dirichlet(np.ones(6), 800)
    ratios = {name: 0.0 for name in FUNS}
    for i in range(800):
        m = int(rng.integers(2, 10))
        K = rng.dirichlet(np.ones(m), 6).T                                 # m x 6, columns sum to 1: a random Markov kernel
        Pk, Qk = K @ Pn[i], K @ Qn[i]
        for name, (f, _) in FUNS.items():
            ratios[name] = max(ratios[name], float(Df(f, Pk, Qk) / Df(f, Pn[i], Qn[i])))
    print("   random Markov kernels (6 outcomes -> 2..9 outcomes, 800 trials): largest D_f[Kp:Kq]/D_f[p:q]: " + ", ".join(f"{k.split('  ')[0]} {v:.4f}" for k, v in ratios.items()))
    # Markov embedding: r_ij supported on cell A_j: equality for every f
    errs = []
    for _ in range(300):
        r = np.zeros((9, 3)); cells = [(0, 2), (2, 6), (6, 9)]
        for j, (a, b) in enumerate(cells):
            w = rng.random(b - a) + 0.1; r[a:b, j] = w / w.sum()
        q0, q1 = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        for name, (f, _) in FUNS.items():
            errs.append(abs(float(Df(f, r @ q0, r @ q1) - Df(f, q0, q1))))
    print(f"   Markov embeddings (3.71)-(3.72), 300 random r: largest |D_f[hq:hq'] - D_f[q:q']| over all nine f: {max(errs):.1e}")
    # the converse proof: additivity (3.22) forces homogeneity of degree one
    print("   the converse (3.20)-(3.24): d(p, u p) = p^a f(u) is additive in p, as (3.22) demands, only for a = 1; residual k(p1,u) + k(p2,u) - k(p1+p2,u) at p1 = 0.2, p2 = 0.3, f(u) = 1:")
    res = {a: 0.2 ** a + 0.3 ** a - 0.5 ** a for a in (0.5, 1.0, 1.5, 2.0)}
    print("      " + "   ".join(f"a={a}: {r:+.4f}" for a, r in res.items()))
    # n = 1: S_1 has two outcomes, merging them leaves a point, so the criterion (3.4) is empty there
    print("   n = 1 (Remark 1): on S_1 the only coarse graining sends everything to one point (D_merged = 0 <= D), so (3.4) is vacuous and any decomposable D passes. "
          "D = (p-q)^2 + ((1-p)-(1-q))^2 = 2 (p-q)^2 is not an f-divergence: its metric is the constant 4, an f-divergence has f''(1)/(p(1-p)):")
    print("      p = 0.1, 0.3, 0.5: f''(1)/(p(1-p)) = " + ", ".join(f"{1 / (p_ * (1 - p_)):.4f}" for p_ in (0.1, 0.3, 0.5)) + " (f''(1) = 1) against 4")
    # the ratio D_merged/D at equal conditionals as a function of a (figure)
    avals = np.linspace(0.4, 1.8, 29)
    STORE["a_curve"] = (avals, [float(np.mean(np.concatenate([d_a(a)(t[2], t[3]) / d_a(a)(t[0], t[1]) for t in suff]))) for a in avals],
                        [float(np.max(np.concatenate([d_a(a)(t[2], t[3]) / d_a(a)(t[0], t[1]) for t in trials]))) for a in avals])
    # D + D^2: invariant but not decomposable, same g and same cubic term
    m = np.array([1.0, 1.4, 0.8]); n = np.array([1.2, 0.9, 1.1])
    Dg = lambda m_, n_: float(np.sum(n_ - m_ + m_ * np.log(m_ / n_)))
    def cross(fun, i=0, j=1, h=1e-4):
        e = np.eye(3)
        return (fun(m + h * e[i], n + h * e[j]) - fun(m + h * e[i], n - h * e[j]) - fun(m - h * e[i], n + h * e[j]) + fun(m - h * e[i], n - h * e[j])) / (4 * h * h)
    c1, c2 = cross(Dg), cross(lambda a_, b_: Dg(a_, b_) + Dg(a_, b_) ** 2)
    print(f"   (3.25) D + D^2: the mixed derivative d^2/dm_0 dn_1 is {c1:+.1e} for the decomposable D and {c2:+.4f} for D + D^2 (not decomposable); D^2 is of order eps^4, so D + D^2 has the same metric and cubic tensor as D")


# ------------------------------------------------------------------ 4. examples in S_n

def D_alpha(p, q, a):
    """The alpha-divergence (3.39), (3.43) written directly in p and q, zeros allowed; +inf where (3.49), (3.50) say so."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    if a == -1:
        if np.any((p > 0) & (q == 0)): return math.inf
        m = p > 0
        return float(np.sum(p[m] * np.log(p[m] / q[m])))
    if a == 1:
        if np.any((q > 0) & (p == 0)): return math.inf
        m = q > 0
        return float(np.sum(q[m] * np.log(q[m] / p[m])))
    if a > 1 and np.any((p == 0) & (q > 0)): return math.inf
    if a < -1 and np.any((p > 0) & (q == 0)): return math.inf
    m = (p > 0) & (q > 0)
    return float(4 / (1 - a * a) * (1 - np.sum(p[m] ** ((1 - a) / 2) * q[m] ** ((1 + a) / 2))))


def chart_eta(e):
    return np.array([1 - e.sum(), *e])


def chart_theta(t):
    w = np.concatenate([[1.0], np.exp(t)])
    return w / w.sum()


def mixed_variation(f, chart, y, xs, h=1e-4):
    """If D[x:y] is a Bregman divergence in the chart x, then d^2 D / dx_i dy_j depends on y only (it is minus the Hessian of the potential at y).
    Returns the largest relative change of that mixed derivative when x moves over the points xs, y fixed."""
    n = len(y)
    def mixed(x):
        M = np.zeros((n, n))
        D = lambda a, b: float(Df(f, chart(a), chart(b)))
        for i in range(n):
            for j in range(n):
                ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
                M[i, j] = (D(x + ei, y + ej) - D(x + ei, y - ej) - D(x - ei, y + ej) + D(x - ei, y - ej)) / (4 * h * h)
        return M
    Ms = [mixed(x) for x in xs]
    ref = Ms[-1]
    return max(float(np.abs(M - ref).max()) for M in Ms[:-1]) / float(np.abs(ref).max())


def check_examples():
    head("4. Examples of f-divergences in S_n  (3.33)-(3.46), and which of them are Bregman")
    rng = rng_for(4)
    P = rand_simplex(rng, 2000, 4)
    Q = rand_simplex(rng, 2000, 4)
    mx = lambda a, b: float(np.max(np.abs(a - b)))
    # KL, dual KL, chi^2
    kl_f = lambda u: -np.log(u)                                              # (3.33), not standard
    print(f"   (3.33)-(3.34): f = -log u gives sum p log(p/q); largest gap over 2000 pairs {mx(Df(kl_f, P, Q), np.sum(P * np.log(P / Q), -1)):.1e}; "
          f"(3.35)-(3.36): f* = u log u gives KL[q:p]: {mx(Df(lambda u: u * np.log(u), P, Q), np.sum(Q * np.log(Q / P), -1)):.1e}")
    # the dual KL is the Bregman divergence of psi = log(1 + sum e^theta) in the logits (ch. 1, 2)
    psi = lambda th: math.log(1 + np.exp(th).sum())
    def breg_psi(p, q):
        tp, tq = np.log(p[1:] / p[0]), np.log(q[1:] / q[0])
        return psi(tp) - psi(tq) - q[1:] @ (tp - tq)
    e = max(abs(breg_psi(P[i], Q[i]) - kl(Q[i], P[i])) for i in range(200))
    print(f"   (3.36) 'coincides with the divergence derived from psi': Bregman divergence of psi between the logits of p and q equals KL[q:p], gap {e:.1e} (200 pairs)")
    print(f"   (3.37) f = (u-1)^2/2 gives (1/2) sum (p-q)^2/p: gap {mx(Df(lambda u: 0.5 * (u - 1) ** 2, P, Q), 0.5 * np.sum((P - Q) ** 2 / P, -1)):.1e}")
    # alpha family
    print("   alpha-divergence: (3.39) written in p, q against sum p f(q/p) with the book's (3.38) and with the standard f; and the duality (3.40) D_alpha[p:q] = D_{-alpha}[q:p]:")
    for a in (-3, -0.5, 0, 0.5, 3):
        d39 = np.array([D_alpha(P[i], Q[i], a) for i in range(300)])
        e1 = mx(d39, Df(f_alpha_book(a), P[:300], Q[:300]))
        e2 = mx(d39, Df(f_alpha(a), P[:300], Q[:300]))
        e3 = mx(Df(f_alpha(a), P[:300], Q[:300]), Df(f_alpha(-a), Q[:300], P[:300]))
        print(f"      alpha = {a:+.1f}: gap to (3.38) {e1:.1e}, to standard f {e2:.1e}, duality {e3:.1e}")
    h = np.array([2 * np.sum((np.sqrt(P[i]) - np.sqrt(Q[i])) ** 2) for i in range(300)])
    print(f"   (3.41) alpha = 0: 2 sum (sqrt p - sqrt q)^2 against D_0: {mx(h, np.array([D_alpha(P[i], Q[i], 0) for i in range(300)])):.1e}")
    # limits
    p = np.array([.5, .3, .2]); q = np.array([.2, .3, .5])
    u0 = 2.0
    print(f"   (3.42)-(3.43) the limit alpha -> 1: D_alpha[p:q] -> KL[q:p] = {kl(q, p):.6f}: " + ", ".join(f"{a}: {D_alpha(p, q, a):.6f}" for a in (0.9, 0.99, 0.999, 0.9999)))
    print(f"      but the function (3.38) itself has no limit: f_alpha(2) = " + ", ".join(f"{a}: {f_alpha_book(a)(u0):.2f}" for a in (0.9, 0.99, 0.999))
          + f"; the standard f_alpha(2) = " + ", ".join(f"{a}: {f_alpha(a)(u0):.4f}" for a in (0.9, 0.99, 0.999)) + f" -> u log u - (u-1) = {u0 * math.log(u0) - (u0 - 1):.4f}")
    STORE["alpha_limit"] = [(a, D_alpha(p, q, a)) for a in (0.9, 0.99, 0.999)]
    # variation distance
    pv = np.array([.5, .3, .2]); dv = np.array([.1, -.04, -.06])
    tv = [float(Df(f_tv, pv, pv + eps * dv)) / eps for eps in (0.1, 0.01, 0.001)]
    sup = max(float(Df(f_tv, P[i], Q[i])) for i in range(2000))
    print(f"   (3.44)-(3.45) f = |1-u| gives D_f = sum |p - q| = {float(Df(f_tv, pv, pv + dv)):.4f} for p = {pv.tolist()}, q = p + d (d = {dv.tolist()}); the printed (1/2) sum |p-q| = {0.5 * np.sum(np.abs(dv)):.4f} is half of it"
          f" (it belongs to f = |1-u|/2). The largest value over 2000 random pairs is {sup:.3f} (the bound is 2).")
    print(f"      and D_f[p : p + eps d] / eps = {tv[0]:.4f}, {tv[1]:.4f}, {tv[2]:.4f} for eps = 0.1, 0.01, 0.001: linear in eps, no quadratic form, no metric")
    STORE["tv_ratio"] = tv
    # (3.46): squared Euclid -- see check 3. Bregman test
    print("   which of these are Bregman divergences? Relative change of d^2 D / dx_i dy_j when x moves (y fixed); a Bregman divergence in a chart has 0 (up to finite-difference noise 1e-8):")
    y_eta = np.array([0.3, 0.25]); y_th = np.array([0.2, -0.4])
    xs_eta = [np.array([0.2, 0.3]), np.array([0.45, 0.2]), np.array([0.1, 0.6]), y_eta]
    xs_th = [np.array([0.0, 0.0]), np.array([0.5, -0.3]), np.array([-0.4, 0.6]), y_th]
    brows = []
    for name, (f, _) in FUNS.items():
        if f is f_tv:
            continue
        ve = mixed_variation(f, chart_eta, y_eta, xs_eta)
        vt = mixed_variation(f, chart_theta, y_th, xs_th)
        brows.append((name, ve, vt))
        print(f"      {name:24s} eta chart (p_1, p_2): {ve:.1e}    theta chart (logits): {vt:.1e}")
    STORE["bregman_rows"] = brows
    print("   only KL is Bregman in the eta chart and only its dual in the theta chart; none of the others is in either. This does not exclude some other chart; the book's Theorem 4.1 says no chart works (its proof is not checked here)")


# ------------------------------------------------------------------ 5. properties of f-divergences

def bound47(name):
    """(3.47): f(0) + f*(0) = lim_{u -> 0} {f(u) + u f(1/u)}, in closed form for the members used here."""
    if name.startswith("Jensen"):
        return math.log(2)
    if name.startswith("variation"):
        return 2.0
    a = {"KL  (alpha=-1)": -1, "dual KL  (alpha=+1)": 1, "chi2  (alpha=3)": 3, "dual chi2  (alpha=-3)": -3,
         "Hellinger  (alpha=0)": 0, "alpha=0.5": 0.5, "alpha=-0.5": -0.5}[name]
    return 4 / (1 - a * a) if abs(a) < 1 else math.inf


def zoom_min(fun, x0, h0, iters=12, k=3):
    """Minimise fun over a small box around x0 by repeatedly re-gridding it with a shrinking step (no gradients needed)."""
    x = np.array(x0, float)
    h = h0
    grid1 = np.arange(-k, k + 1)
    best = fun(x)
    for _ in range(iters):
        pts = np.stack(np.meshgrid(*[grid1] * len(x), indexing="ij"), -1).reshape(-1, len(x)) * h + x
        vals = [fun(pt) for pt in pts]
        i = int(np.argmin(vals))
        if vals[i] < best:
            best, x = vals[i], pts[i]
        h *= 0.45
    return x, best


def check_properties():
    head("5. Properties of f-divergences  (3.47)-(3.53)")
    rng = rng_for(5)
    # (1) joint convexity
    R = 20000
    P1, Q1, P2, Q2 = (rand_simplex(rng, R, 5) for _ in range(4))
    t = rng.random(R)[:, None]
    marg = {}
    for nm, (f, _) in FUNS.items():
        lhs = t[:, 0] * Df(f, P1, Q1) + (1 - t[:, 0]) * Df(f, P2, Q2)
        marg[nm] = float(np.min((lhs - Df(f, t * P1 + (1 - t) * P2, t * Q1 + (1 - t) * Q2)) / lhs))
    print(f"   (1) joint convexity D(t x1 + (1-t) x2) <= t D(x1) + (1-t) D(x2), x = (p, q): smallest relative margin over 20000 random quadruples: "
          f"{min(v for k, v in marg.items() if not k.startswith('variation')):.1e} for the eight smooth f, {marg['variation  |1-u|']:.1e} for the variation distance (zero margin is allowed)")
    STORE["convexity_margin"] = marg
    # (2) the upper bound (3.47)
    print("   (2) (3.47): the limit f(d) + d f(1/d) at d = 1e-6 and 1e-12; D at the nearly disjoint pair p = (1-d, d), q = (d, 1-d), d = 1e-9; the largest D over 100000 random sparse pairs")
    rows = []
    S1 = rng.dirichlet(np.full(4, 0.1), 100000); S2 = rng.dirichlet(np.full(4, 0.1), 100000)
    d9 = 1e-9
    for name, (f, _) in FUNS.items():
        b = bound47(name)
        l6 = float(f(1e-6) + 1e-6 * f(1e6)); l12 = float(f(1e-12) + 1e-12 * f(1e12))
        near = float(Df(f, np.array([1 - d9, d9]), np.array([d9, 1 - d9])))
        with np.errstate(all="ignore"):
            top = float(np.nanmax(Df(f, S1, S2)))
        rows.append((name, b, l6, l12, near, top))
        if math.isinf(b):
            print(f"      {name:24s} the limit grows ({l6:.3g} at 1e-6, {l12:.3g} at 1e-12): no finite bound")
        else:
            print(f"      {name:24s} limit {l6:8.4f}, {l12:8.4f}   D(disjoint) {near:8.4f}   max over random pairs {top:8.4f}   closed form {b:.4f}")
    STORE["bound_rows"] = rows
    # (3.48) as printed, and with the arguments exchanged
    print("   (3.48): the printed bound  D_f[p:q] <= sum (p_i - q_i) f'(p_i/q_i)  against the version with the arguments exchanged, D_f[p:q] <= sum (q_i - p_i) f'(q_i/p_i)")
    p = np.array([.9, .1]); q = np.array([.5, .5])
    fchi = lambda u: 0.5 * (u - 1) ** 2
    d1 = lambda u: u - 1
    print(f"      chi^2 at p = {p.tolist()}, q = {q.tolist()}: D_f[p:q] = {float(Df(fchi, p, q)):.4f}; printed bound {float(np.sum((p - q) * d1(p / q))):.4f} (violated); exchanged bound {float(np.sum((q - p) * d1(q / p))):.4f}")
    STORE["b48_example"] = (float(Df(fchi, p, q)), float(np.sum((p - q) * d1(p / q))), float(np.sum((q - p) * d1(q / p))))
    Pa, Qa = rand_simplex(rng, 50000, 4), rand_simplex(rng, 50000, 4)
    vrows = []
    for name, (f, _) in FUNS.items():
        if f is f_tv:
            continue
        D = Df(f, Pa, Qa)
        pr = np.sum((Pa - Qa) * num_d1(f, Pa / Qa), -1)
        ex = np.sum((Qa - Pa) * num_d1(f, Qa / Pa), -1)
        vrows.append((name, float(np.mean(D > pr * (1 + 1e-7) + 1e-12)), float(np.mean(D > ex * (1 + 1e-7) + 1e-12))))
        print(f"      {name:24s} share of 50000 random pairs violating the printed bound {vrows[-1][1]:.3f},  the exchanged bound {vrows[-1][2]:.3f}")
    STORE["b48_rows"] = vrows
    # (3), (4): infinite values; (5), (6): zero forcing and zero avoidance
    alphas = (-3, -1, -0.5, 0, 0.5, 1, 3)
    pz = np.array([.5, .5, 0.]); qz = np.array([.4, .4, .2])
    print("   (3.49)-(3.50): D_alpha for p = (.5,.5,0), q = (.4,.4,.2) [p = 0 < q at the third outcome] and for the pair exchanged [p > 0 = q]:")
    print("      alpha          " + "".join(f"{a:>9.1f}" for a in alphas))
    print("      p=0<q          " + "".join(f"{D_alpha(pz, qz, a):9.4f}" for a in alphas))
    print("      p>0=q          " + "".join(f"{D_alpha(qz, pz, a):9.4f}" for a in alphas))
    STORE["inf_table"] = [[D_alpha(pz, qz, a) for a in alphas], [D_alpha(qz, pz, a) for a in alphas]]
    # a 3 x 3 table with all its mass on the diagonal, approximated by independent distributions r (x) s
    Pd = np.diag([.5, .3, .2])
    K = 40
    pts = np.array([[i, j, K - i - j] for i in range(K + 1) for j in range(K + 1 - i)], float) / K
    Rr = pts[:, None, :, None] * pts[None, :, None, :]                       # all products r_i s_j on a grid
    def Dgrid(a):
        pp = np.broadcast_to(Pd, Rr.shape)
        with np.errstate(all="ignore"):
            if a == -1:
                bad = ((pp > 0) & (Rr == 0)).any((2, 3))
                t = np.where(pp > 0, pp * np.log(pp / np.where(Rr > 0, Rr, 1)), 0).sum((2, 3))
            elif a == 1:
                bad = ((Rr > 0) & (pp == 0)).any((2, 3))
                t = np.where(Rr > 0, Rr * np.log(np.where(Rr > 0, Rr, 1) / np.where(pp > 0, pp, 1)), 0).sum((2, 3))
            else:
                bad = ((pp == 0) & (Rr > 0)).any((2, 3)) if a > 1 else (((pp > 0) & (Rr == 0)).any((2, 3)) if a < -1 else np.zeros(Rr.shape[:2], bool))
                m = (pp > 0) & (Rr > 0)
                t = 4 / (1 - a * a) * (1 - np.where(m, np.where(pp > 0, pp, 1) ** ((1 - a) / 2) * np.where(Rr > 0, Rr, 1) ** ((1 + a) / 2), 0).sum((2, 3)))
        return np.where(bad, np.inf, t)
    print("   a perfectly dependent 3 x 3 table p = diag(.5,.3,.2) approximated by independent tables r (x) s (grid step 1/40, then refined): the best fit for each alpha")
    trows = []
    for a in (-3, -2, -1, -0.5, 0, 0.5, 1, 3):
        G = Dgrid(a)
        i, j = np.unravel_index(np.argmin(G), G.shape)
        r0, s0 = pts[i], pts[j]
        def obj(z, a=a):
            r = np.array([z[0], z[1], 1 - z[0] - z[1]]); s = np.array([z[2], z[3], 1 - z[2] - z[3]])
            if r.min() < 0 or s.min() < 0:
                return math.inf
            return D_alpha(Pd.ravel(), np.outer(r, s).ravel(), a)
        z, best = zoom_min(obj, [r0[0], r0[1], s0[0], s0[1]], 1 / 80, iters=8, k=1)
        r = np.array([z[0], z[1], 1 - z[0] - z[1]]); s = np.array([z[2], z[3], 1 - z[2] - z[3]])
        q = np.outer(r, s)
        trows.append((a, best, r.round(3).tolist(), float(np.trace(q))))
        print(f"      alpha = {a:+.1f}: min D_alpha = {best:.5f}   r = {np.round(r, 3).tolist()}   s = {np.round(s, 3).tolist()}   mass on the diagonal {np.trace(q):.3f}   "
              + ("(point mass: zero forcing)" if np.trace(q) > 0.999 else "(full support: zero avoiding)"))
    STORE["table_fit"] = trows
    ent = -float(np.sum(np.array([.5, .3, .2]) * np.log([.5, .3, .2])))
    print(f"      alpha = -1: the minimiser is p_X (x) p_Y with value I(X;Y) = H(X) = {ent:.5f}; alpha = +1: D = -log 0.5 = {-math.log(.5):.5f} at the point mass on the largest diagonal cell")
    # the same phenomenon with a smooth model: a bimodal p on 40 points, best discretised Gaussian
    x = np.arange(40.0)
    bump = lambda m, s: (lambda w: w / w.sum())(np.exp(-0.5 * ((x - m) / s) ** 2))
    pb = 0.5 * bump(10, 2) + 0.5 * bump(28, 3)
    MU = np.arange(0, 39.01, 0.25); SG = np.exp(np.linspace(math.log(0.4), math.log(20), 160))
    Qg = np.exp(-0.5 * ((x[None, None, :] - MU[:, None, None]) / SG[None, :, None]) ** 2)
    Qg /= Qg.sum(-1, keepdims=True)
    Qg = np.maximum(Qg, 1e-300)
    def Dg(a):
        if a == -1: return np.sum(pb * np.log(pb / Qg), -1)
        if a == 1: return np.sum(Qg * np.log(Qg / pb), -1)
        return 4 / (1 - a * a) * (1 - np.sum(pb ** ((1 - a) / 2) * Qg ** ((1 + a) / 2), -1))
    print(f"   smooth version: p = half N(10, 2^2) + half N(28, 3^2) on 40 points (mean {pb @ x:.2f}, sd {math.sqrt(pb @ x ** 2 - (pb @ x) ** 2):.2f}); best discretised Gaussian N(mu, sigma^2) for each alpha:")
    srows = []
    for a in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
        G = Dg(a)
        i, j = np.unravel_index(np.argmin(G), G.shape)
        srows.append((a, float(MU[i]), float(SG[j]), float(G[i, j])))
        print(f"      alpha = {a:+.1f}: mu = {MU[i]:5.2f}, sigma = {SG[j]:6.3f}, D = {G[i, j]:.4f}")
    STORE["bimodal_rows"] = srows
    STORE["bimodal_p"] = pb


# ------------------------------------------------------------------ 6. KL divergence: Sanov, CLT, large deviations, Theorem 3.2

def lgam(N):
    return np.array([math.lgamma(k + 1) for k in range(N + 1)])


def log_multinomial3(N, n1, n2, p, lg=None):
    """log P(counts = (N-n1-n2, n1, n2)) for the multinomial (N; p), arrays n1, n2."""
    lg = lgam(N) if lg is None else lg
    n0 = N - n1 - n2
    ok = n0 >= 0
    n0c = np.clip(n0, 0, N)
    lp = lg[N] - lg[n0c] - lg[np.clip(n1, 0, N)] - lg[np.clip(n2, 0, N)] + n0c * math.log(p[0]) + n1 * math.log(p[1]) + n2 * math.log(p[2])
    return np.where(ok, lp, -np.inf)


def gl_nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def check_kl():
    head("6. Properties of the KL divergence: Sanov, the central limit theorem, large deviations, Theorem 3.2  (3.54)-(3.66)")
    p = np.array([.5, .3, .2])
    # Sanov's lemma (3.55): exact multinomial probability of one empirical distribution
    ph = np.array([.2, .3, .5])
    print(f"   (3.55) exact probability of the empirical distribution p_hat = {ph.tolist()} for p = {p.tolist()}; KL[p_hat:p] = {kl(ph, p):.6f}")
    print("      N      -(1/N) log P    exp(-N KL) / P      P / [(2 pi N)^-1 (prod p_hat)^-1/2 exp(-N KL)]")
    srows = []
    for N in (10, 50, 200, 1000, 5000):
        n1, n2 = int(round(N * ph[1])), int(round(N * ph[2]))
        lp = float(log_multinomial3(N, np.array([n1]), np.array([n2]), p)[0])
        stir = -math.log(2 * math.pi * N) - 0.5 * math.log(float(np.prod(ph))) - N * kl(ph, p)
        srows.append((N, -lp / N, math.exp(-N * kl(ph, p) - lp), math.exp(lp - stir)))
        print(f"      {N:5d}  {-lp / N:12.6f}  {srows[-1][2]:14.4f}   {srows[-1][3]:.6f}")
    STORE["sanov_rows"] = srows
    print("      so the exponent is right and the polynomial factor (2 pi N)^-1 (prod p_hat)^-1/2 is what 'exp' in (3.55) hides")
    # (3.56)-(3.57)
    N = 400
    dl = np.array([.03, -.02, -.01])
    phat = p + dl
    Geta = np.diag(1 / p[1:]) + 1 / p[0]                         # Fisher information of S_2 in the eta chart (p_1, p_2)
    quad = 0.5 * N * dl[1:] @ Geta @ dl[1:]
    eps_book = dl / math.sqrt(N)
    print(f"   (3.56): N = {N}, p_hat - p = {dl.tolist()}: N KL[p_hat:p] = {N * kl(phat, p):.5f}; (1/2) eps^T G_eta eps with eps = sqrt(N)(p_hat - p): {quad:.5f};"
          f" with the printed eps = (p_hat - p)/sqrt(N) it would be {0.5 * eps_book[1:] @ Geta @ eps_book[1:]:.2e}")
    STORE["eps_example"] = (N * kl(phat, p), quad, float(0.5 * eps_book[1:] @ Geta @ eps_book[1:]))
    # (3.57) exact covariance of the empirical distribution
    N2 = 12
    n1, n2 = np.meshgrid(np.arange(N2 + 1), np.arange(N2 + 1), indexing="ij")
    lp = log_multinomial3(N2, n1, n2, p)
    w = np.exp(lp)
    ph1, ph2 = n1 / N2, n2 / N2
    m1, m2 = float((w * ph1).sum()), float((w * ph2).sum())
    C = np.array([[(w * (ph1 - m1) ** 2).sum(), (w * (ph1 - m1) * (ph2 - m2)).sum()], [(w * (ph1 - m1) * (ph2 - m2)).sum(), (w * (ph2 - m2) ** 2).sum()]])
    Gth = (np.diag(p) - np.outer(p, p))[1:, 1:]
    print(f"   (3.57) exact covariance of p_hat for N = {N2} (sum over all {int(np.sum(np.isfinite(lp)))} outcomes): N Cov = {np.round(N2 * C, 6).tolist()};  g_ij in theta = diag(p) - p p^T = {np.round(Gth, 6).tolist()}  (gap {np.abs(N2 * C - Gth).max():.1e});"
          f"  the Fisher matrix in the eta chart, diag(1/p_i) + 1/p_0, is the inverse: G_eta (N Cov) = {np.round(Geta @ (N2 * C), 12).tolist()}")
    # local limit theorem at N = 400
    n1o, n2o = int(round(N * phat[1])), int(round(N * phat[2]))
    pmf = math.exp(float(log_multinomial3(N, np.array([n1o]), np.array([n2o]), p)[0]))
    Cg = Gth / N
    dd = dl[1:]
    rho = math.exp(-0.5 * dd @ np.linalg.solve(Cg, dd)) / (2 * math.pi * math.sqrt(np.linalg.det(Cg)))
    print(f"   local limit at N = {N}: P(p_hat = {np.round(phat, 3).tolist()}) = {pmf:.6f}, Gaussian density with covariance g/N times the cell area 1/N^2: {rho / N ** 2:.6f} (ratio {pmf * N ** 2 / rho:.4f})")
    # large deviations (3.58)-(3.59): A = {q : sum_i i q_i >= a}
    x = np.array([0., 1., 2.])
    a = 1.1
    tilt = lambda lam: (lambda w_: w_ / w_.sum())(p * np.exp(lam * x))
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tilt(mid) @ x < a: lo = mid
        else: hi = mid
    lam = (lo + hi) / 2
    qs = tilt(lam)
    I_tilt = kl(qs, p)
    lams = np.linspace(0, 5, 500001)
    I_cramer = float(np.max(a * lams - np.log((p[None, :] * np.exp(lams[:, None] * x[None, :])).sum(1))))
    G1, G2 = np.meshgrid(np.linspace(0, 1, 1601), np.linspace(0, 1, 1601), indexing="ij")
    G0 = 1 - G1 - G2
    with np.errstate(all="ignore"):
        Kg = np.where((G0 >= 0) & (G1 + 2 * G2 >= a - 1e-12),
                      np.where(G0 > 0, G0 * np.log(G0 / p[0]), 0) + np.where(G1 > 0, G1 * np.log(G1 / p[1]), 0) + np.where(G2 > 0, G2 * np.log(G2 / p[2]), 0), np.inf)
    ig = np.unravel_index(np.argmin(Kg), Kg.shape)
    print(f"   (3.58)-(3.59) A = {{q : mean of x >= {a}}}, x = (0,1,2), p = {p.tolist()} (mean {p @ x:.1f}): minimiser of KL[q:p] over A")
    print(f"      by exponential tilting q_i ~ p_i exp(lam x_i) (the e-projection): lam = {lam:.6f}, q* = {np.round(qs, 5).tolist()}, KL = {I_tilt:.8f}")
    print(f"      by Cramer's transform sup_lam (a lam - log E exp(lam x)): {I_cramer:.8f};  by brute force over a 1600 x 1600 grid of A: {Kg[ig]:.6f} at q = {[round(float(G0[ig]), 4), round(float(G1[ig]), 4), round(float(G2[ig]), 4)]}")
    # orthogonality in the Fisher metric
    Z = np.array([1., -2., 1.])                                    # tangent to the boundary {mean = a} inside the simplex
    V = qs * (lam * x - qs @ (lam * x))                           # velocity of the e-geodesic p -> q* at q* (d/dt of q_t ~ p^(1-t) q*^t at t = 1)
    print(f"      Fisher inner product of the boundary direction Z = (1,-2,1) with the e-geodesic's velocity at q*: sum Z_i V_i / q*_i = {np.sum(Z * V / qs):+.1e};"
          f"  the Euclidean cosine of the angle between them is {np.sum(Z * V) / math.sqrt(np.sum(Z * Z) * np.sum(V * V)):+.3f}")
    # Pythagorean inequality for the I-projection onto the (m-convex) set A
    rng = rng_for(6)
    Qr = rng.dirichlet(np.ones(3), 200000)
    Qr = Qr[Qr @ x >= a]
    slack = np.sum(Qr * np.log(Qr / p), -1) - np.sum(Qr * np.log(Qr / qs), -1) - I_tilt
    print(f"      Pythagorean inequality KL[q:p] >= KL[q:q*] + KL[q*:p] for {len(Qr)} random q in A: smallest slack {slack.min():+.2e}")
    # exact probabilities
    print("      exact P(p_hat in A) from the multinomial, Bahadur-Rao = exp(-N I) / ((1 - e^-lam) sqrt(2 pi N var)):")
    print("      N       -(1/N) log P     P exp(N I) sqrt(N)    P / Bahadur-Rao")
    var = qs @ x ** 2 - (qs @ x) ** 2
    lrows = []
    for N in (10, 20, 50, 100, 200, 400, 800, 1600):
        n1, n2 = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        lp = log_multinomial3(N, n1, n2, p)
        lp = np.where(n1 + 2 * n2 >= round(a * N), lp, -np.inf)
        mxx = lp.max()
        lP = mxx + math.log(float(np.exp(lp - mxx).sum()))
        br = math.exp(-N * I_tilt) / ((1 - math.exp(-lam)) * math.sqrt(2 * math.pi * N * var))
        lrows.append((N, -lP / N, math.exp(lP + N * I_tilt) * math.sqrt(N), math.exp(lP) / br))
        print(f"      {N:5d}   {-lP / N:12.6f}     {lrows[-1][2]:12.6f}      {lrows[-1][3]:.6f}")
    STORE["ld_rows"] = lrows
    STORE["ld"] = dict(lam=lam, qs=qs, I=I_tilt, a=a, p=p, var=var)
    # Theorem 3.2
    print("   Theorem 3.2 (3.66): e-geodesic p_t ~ p^(1-t) q^t (3.62) and m-geodesic (1-t) p + t q (3.63); Fisher information along each, integrated over t in [0,1] by Gauss-Legendre (200 nodes)")
    t, wt = gl_nodes(200)
    def integrals(pp, qq):
        d = np.log(qq) - np.log(pp)
        ge, gm = [], []
        for ti in t:
            lw = (1 - ti) * np.log(pp) + ti * np.log(qq)
            pe = np.exp(lw - lw.max()); pe /= pe.sum()
            ge.append(pe @ (d - pe @ d) ** 2)                       # (d/dt log p_t)^2 averaged over p_t = Var_t[log q/p]
            pm = (1 - ti) * pp + ti * qq
            gm.append(np.sum((qq - pp) ** 2 / pm))
        return float(np.dot(wt, ge)), float(np.dot(wt, gm)), np.array(ge), np.array(gm)
    rng = rng_for(7)
    print("      pair (p, q) on 4 outcomes:  int g_e         int g_m         D[p:q] + D[q:p]   (1/2){D + D*}    s^2 (Fisher-Rao distance squared)")
    rows32 = []
    for k in range(5):
        pp, qq = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
        ie, im, _, _ = integrals(pp, qq)
        s = 2 * math.acos(min(1.0, float(np.sum(np.sqrt(pp * qq)))))
        rows32.append((ie, im, kl(pp, qq) + kl(qq, pp), s * s))
        print(f"      random pair {k + 1}:        {ie:.10f}  {im:.10f}  {kl(pp, qq) + kl(qq, pp):.10f}    {(kl(pp, qq) + kl(qq, pp)) / 2:.10f}   {s * s:.6f}")
    STORE["thm32_rows"] = rows32
    # local limit: all four quantities divided by s^2
    p0 = np.array([.5, .3, .2]); d0 = np.array([.1, -.04, -.06])
    print("      near p = (.5,.3,.2): ratios to s^2 as the distance shrinks (eps = 1, 0.3, 0.1, 0.03, 0.01):")
    loc = []
    for eps in (1, 0.3, 0.1, 0.03, 0.01):
        qq = p0 + eps * 0.5 * d0
        ie, im, _, _ = integrals(p0, qq)
        s2 = (2 * math.acos(float(np.sum(np.sqrt(p0 * qq))))) ** 2
        loc.append((eps, math.sqrt(s2), ie / s2, im / s2, (kl(p0, qq) + kl(qq, p0)) / s2, 0.5 * (kl(p0, qq) + kl(qq, p0)) / s2))
        print(f"         eps = {eps:5.2f}: s = {math.sqrt(s2):.4f}   int g_e / s^2 = {ie / s2:.6f}   int g_m / s^2 = {im / s2:.6f}   (D+D*) / s^2 = {(kl(p0, qq) + kl(qq, p0)) / s2:.6f}   (1/2)(D+D*) / s^2 = {0.5 * (kl(p0, qq) + kl(qq, p0)) / s2:.6f}")
    STORE["thm32_local"] = loc
    # the pair used in the figure and in the widgets, and a sweep towards it
    pf, qf = FIG_P, FIG_Q
    ie, im, ge, gm = integrals(pf, qf)
    sweep = []
    for tau in np.linspace(0.02, 1.0, 25):
        qq = pf + tau * (qf - pf)
        a1, a2, _, _ = integrals(pf, qq)
        s2 = (2 * math.acos(float(np.sum(np.sqrt(pf * qq))))) ** 2
        sweep.append((math.sqrt(s2), a1 / s2, a2 / s2, (kl(pf, qq) + kl(qq, pf)) / s2))
    STORE["thm32_sweep"] = sweep
    sf = 2 * math.acos(float(np.sum(np.sqrt(pf * qf))))
    print(f"      figure pair p = {pf.tolist()}, q = {qf.tolist()}: int g_e = {ie:.6f}, int g_m = {im:.6f}, D[p:q] + D[q:p] = {kl(pf, qf) + kl(qf, pf):.6f}, (1/2){{...}} = {0.5 * (kl(pf, qf) + kl(qf, pf)):.6f}, s^2 = {sf * sf:.6f}")
    STORE["thm32_fig"] = dict(p=pf, q=qf, t=t, ge=ge, gm=gm, ie=ie, im=im, s=sf, J=kl(pf, qf) + kl(qf, pf))
    # continuous densities: two Gaussians; g_m has integrable end-point singularities, so the nodes are clustered towards t = 0, 1
    N1, N2 = (0.0, 1.0), (1.0, 1.3)
    X = np.linspace(-40, 42, 200001)
    th1 = np.array([N1[0] / N1[1] ** 2, -0.5 / N1[1] ** 2]); th2 = np.array([N2[0] / N2[1] ** 2, -0.5 / N2[1] ** 2])
    dth = th2 - th1
    u, wu = gl_nodes(120)
    tc, wc = (1 - np.cos(math.pi * u)) / 2, wu * (math.pi / 2) * np.sin(math.pi * u)
    ge_c, gm_c = [], []
    for ti in tc:
        th = (1 - ti) * th1 + ti * th2
        s2 = -0.5 / th[1]; mu = th[0] * s2
        # Var_t of dth . (x, x^2): moments of N(mu, s2)
        m4 = mu ** 4 + 6 * mu ** 2 * s2 + 3 * s2 ** 2; m3 = mu ** 3 + 3 * mu * s2; m2 = mu ** 2 + s2
        ge_c.append(dth[0] ** 2 * s2 + 2 * dth[0] * dth[1] * (m3 - mu * m2) + dth[1] ** 2 * (m4 - m2 ** 2))
        pm = (1 - ti) * gpdf(X, *N1) + ti * gpdf(X, *N2)
        with np.errstate(all="ignore"):
            gm_c.append(trap(np.where(pm > 1e-300, (gpdf(X, *N2) - gpdf(X, *N1)) ** 2 / np.where(pm > 1e-300, pm, 1), 0.0), X))
    Jc = kl_gauss(*N1, *N2) + kl_gauss(*N2, *N1)
    print(f"      continuous: N(0,1) and N(1,1.3^2): int g_e = {float(np.dot(wc, ge_c)):.8f}, int g_m = {float(np.dot(wc, gm_c)):.8f}, KL + KL* = {Jc:.8f} (closed form)")


# ------------------------------------------------------------------ 7. Fisher information from f-divergences; Chentsov's theorem

def fisher_eta_matrix(p):
    """Fisher information of S_n in the eta chart (p_1..p_n): diag(1/p_i) + 1/p_0."""
    return np.diag(1 / p[1:]) + 1 / p[0]


def hessian_second_arg(D, x, h):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
            H[i, j] = (D(x, x + ei + ej) - D(x, x + ei - ej) - D(x, x - ei + ej) + D(x, x - ei - ej)) / (4 * h * h)
    return H


def random_embedding(rng, m=3, maxcell=4):
    """A Markov embedding r (n x m): column j is a distribution supported on its own block of rows (cell A_j)."""
    sizes = rng.integers(1, maxcell + 1, m)
    n = int(sizes.sum())
    r = np.zeros((n, m))
    a = 0
    for j, sz in enumerate(sizes):
        w = rng.random(sz) + 0.05
        r[a:a + sz, j] = w / w.sum()
        a += sz
    return r


def split_image_lengths(r):
    """The image of S_1 under the embedding that splits outcome 0 into (r, 1-r): the segment from the vertex (0,0,1) of S_2 to the point (r, 1-r, 0).
    Returns its Euclidean length in the equilateral picture of the simplex (side 1; vertex 2 on top) and its Fisher length, the integral over q in (0,1)
    of sqrt(sum dp_i^2/p_i); the substitution q = sin^2(theta) removes the end-point singularities."""
    eu = math.sqrt((0.5 - r) ** 2 + 0.75)
    dp = np.array([r, 1 - r, -1.0])
    th = np.linspace(1e-9, math.pi / 2 - 1e-9, 20001)
    q_ = np.sin(th) ** 2
    speed = np.sqrt(dp[0] ** 2 / (r * q_) + dp[1] ** 2 / ((1 - r) * q_) + 1 / np.cos(th) ** 2) * 2 * np.sin(th) * np.cos(th)
    return eu, trap(speed, th)


def check_fisher_chentsov():
    head("7. The Fisher metric from any standard f-divergence, and Chentsov's theorem  (3.67)-(3.88)")
    # discrete: Hessian in the second argument at q = p, S_3 in the eta chart
    p = np.array([.4, .3, .2, .1])
    G = fisher_eta_matrix(p)
    print(f"   (3.67)-(3.68) discrete: Hessian of D_f[p : p'] in p' at p' = p (eta chart of S_3, p = {p.tolist()}) against f''(1) times the Fisher matrix diag(1/p_i) + 1/p_0:")
    hrows = []
    for name, (f, f2) in FUNS.items():
        if f is f_tv:
            continue
        D = lambda x, y, f=f: float(Df(f, chart_eta(x), chart_eta(y)))
        H = hessian_second_arg(D, p[1:], 2e-4)
        err = float(np.abs(H - f2 * G).max() / np.abs(G).max())
        hrows.append((name, f2, err))
        print(f"      {name:24s} f''(1) = {f2:.2f}   relative error of Hessian - f''(1) G: {err:.1e}")
    STORE["hess_rows"] = hrows
    # continuous: Gaussians
    X = np.linspace(-30, 32, 200001)
    xi0 = np.array([1.0, 2.0])
    def Dgauss(f):
        def D(a, b):
            pa, pb = gpdf(X, *a), gpdf(X, *b)
            with np.errstate(all="ignore"):
                return trap(np.where(pa > 1e-300, pa * f(pb / np.where(pa > 1e-300, pa, 1)), 0.0), X)
        return D
    print(f"   continuous: Gaussians at (mu, sigma) = (1, 2), Fisher information diag(1/sigma^2, 2/sigma^2) = diag(0.25, 0.5); Hessian of D_f by quadrature:")
    for name in ("KL  (alpha=-1)", "chi2  (alpha=3)", "Hellinger  (alpha=0)", "alpha=0.5", "Jensen-Shannon"):
        f, f2 = FUNS[name]
        H = hessian_second_arg(Dgauss(f), xi0, 2e-3)
        print(f"      {name:24s} {np.round(np.diag(H), 5).tolist()} (off-diagonal {H[0, 1]:+.1e}), divided by f''(1) = {f2}: {np.round(np.diag(H) / f2, 5).tolist()}")
    # Markov embeddings: invariance of the Fisher metric, failure of competitors
    rng = rng_for(8)
    mets = {
        "Fisher: sum Z^2/p": lambda p_, Z: np.sum(Z * Z / p_),
        "Euclid: sum Z^2": lambda p_, Z: np.sum(Z * Z),
        "sum Z^2/p^0.5": lambda p_, Z: np.sum(Z * Z / p_ ** 0.5),
        "sum Z^2/p^1.5": lambda p_, Z: np.sum(Z * Z / p_ ** 1.5),
        "sum Z^2/p^2 (Hessian of Burg entropy)": lambda p_, Z: np.sum(Z * Z / p_ ** 2),
        "Fisher + 0.7 (sum Z)^2": lambda p_, Z: np.sum(Z * Z / p_) + 0.7 * np.sum(Z) ** 2,
    }
    print("   Markov embeddings h: q -> r q (3.72), 400 random r (3 cells of 1-4 outcomes each); defect = max |g_hq(hZ, hZ) / g_q(Z, Z) - 1| for tangent Z (sum Z = 0) and for arbitrary Z:")
    defect = {k: [0.0, 0.0] for k in mets}
    for _ in range(400):
        r = random_embedding(rng)
        q = rng.dirichlet(np.ones(3))
        Zt = rng.normal(size=3); Zt -= Zt.mean()
        Za = rng.normal(size=3)
        for k, g in mets.items():
            defect[k][0] = max(defect[k][0], abs(g(r @ q, r @ Zt) / g(q, Zt) - 1))
            defect[k][1] = max(defect[k][1], abs(g(r @ q, r @ Za) / g(q, Za) - 1))
    for k, (dt, da) in defect.items():
        print(f"      {k:40s} tangent {dt:9.2e}    arbitrary {da:9.2e}")
    STORE["defect"] = defect
    # monotonicity of the metric under coarse graining, and equality at horizontal vectors
    print("   coarse graining a metric (merging cells): ||f_* Z||^2 / ||Z||^2 over 20000 random (p, Z, partition); 'horizontal' Z has Z_i/p_i constant on each cell:")
    mrow = []
    for a_ in (0.5, 1.0, 1.5, 2.0):
        g = lambda p_, Z, a_=a_: np.sum(Z * Z / p_ ** a_)
        mx_, hor = 0.0, []
        for _ in range(4000):
            n_ = 6; m_ = int(rng.integers(2, 6))
            pp = rng.dirichlet(np.ones(n_)); lab = rng.integers(0, m_, n_)
            lab[rng.permutation(n_)[:m_]] = np.arange(m_)
            O = (lab[:, None] == np.arange(m_)).astype(float)
            Z = rng.normal(size=n_); Z -= Z.mean()
            mx_ = max(mx_, g(pp @ O, Z @ O) / g(pp, Z))
            cell = rng.normal(size=m_); Zh = pp * (cell[lab] - np.sum(pp * cell[lab]))      # horizontal and tangent
            hor.append(g(pp @ O, Zh @ O) / g(pp, Zh))
        mrow.append((a_, mx_, float(np.min(hor)), float(np.max(hor))))
        print(f"      weight p^-{a_}: max ratio {mx_:.4f} (monotone iff <= 1);  ratio at horizontal Z in [{min(hor):.4f}, {max(hor):.4f}] (invariant iff = 1)")
    STORE["metric_mono"] = mrow
    # the proof's computation: q = k/n, r_ij = 1/k_j
    n_, ks = 12, np.array([2, 3, 7])
    q = ks / n_
    r = np.zeros((n_, 3)); a = 0
    for j, k in enumerate(ks):
        r[a:a + k, j] = 1.0 / k; a += k
    pbar = r @ q
    fisher = lambda u, v: float(np.sum(u * v / pbar))
    print(f"   (3.81)-(3.87) the proof: n = {n_}, k = {ks.tolist()}, q = k/n = {np.round(q, 4).tolist()}; r_ij = 1/k_j maps q to the uniform distribution (max |h q - 1/n| = {np.abs(pbar - 1 / n_).max():.1e}); pushed-forward basis vectors e~_j = r[:, j]:")
    print(f"      <e~_1,e~_1> = {fisher(r[:, 0], r[:, 0]):.4f}, <e~_2,e~_2> = {fisher(r[:, 1], r[:, 1]):.4f}, <e~_3,e~_3> = {fisher(r[:, 2], r[:, 2]):.4f} (= n/k_j = {n_ / ks[0]:.4f}, {n_ / ks[1]:.4f}, {n_ / ks[2]:.4f} = 1/q_j); <e~_1,e~_2> = {fisher(r[:, 0], r[:, 1]):.1e}")
    zt = r[:, 0] - r[:, 1]
    print(f"      tangent vector e_1 - e_2: embedded norm^2 {fisher(zt, zt):.4f} = 1/q_1 + 1/q_2 = {1 / q[0] + 1 / q[1]:.4f}: on S_(m-1) this is all one needs; the book's (3.85) uses the non-tangent e_1, where B(n) = 0 is not justified")
    # the book's vector e~_1: how much does splitting a cell change its squared length, for the weight p^-a?
    print("   splitting one outcome into cells with proportions r (the vector e~_1 of (3.84)): ||e~_1||^2 / ||e_1||^2 for the weight p^-a equals sum_i r_i^(2-a):")
    split = {}
    for rr in ((0.5, 0.5), (0.3, 0.7), (0.2, 0.3, 0.5)):
        rr = np.array(rr)
        vals = []
        for a_ in (0.0, 0.5, 1.0, 1.5, 2.0):
            q1 = 0.37                                                # any value: it cancels
            vals.append(float(np.sum(rr * rr / (rr * q1) ** a_) / q1 ** (-a_)))
        split[tuple(rr)] = vals
        print(f"      r = {rr.tolist()}: a = 0, 0.5, 1, 1.5, 2 -> " + ", ".join(f"{v:.4f}" for v in vals))
    STORE["split"] = split
    print("   the image of S_1 under the embedding that splits outcome 0 by (r, 1-r) is a segment of S_2; its Euclidean length in the picture of the simplex (side 1) and its Fisher length:")
    for r_ in (0.2, 0.5, 0.8):
        eu_, fl_ = split_image_lengths(r_)
        print(f"      r = {r_}: Euclid {eu_:.4f}, Fisher {fl_:.4f} (pi = {math.pi:.4f})")
    # cubic tensor
    print("   (3.88) cubic tensor: T_b(p; Z) = sum Z^3 / p^b under the same embeddings (b = 2 is E[score^3]); defect max |T_hq(hZ)/T_q(Z) - 1|:")
    for b in (1.0, 1.5, 2.0, 2.5, 3.0):
        dmax = 0.0
        for _ in range(300):
            r = random_embedding(rng); q = rng.dirichlet(np.ones(3)); Z = rng.normal(size=3)
            dmax = max(dmax, abs(np.sum((r @ Z) ** 3 / (r @ q) ** b) / np.sum(Z ** 3 / q ** b) - 1))
        print(f"      b = {b}: {dmax:.2e}")


# ------------------------------------------------------------------ 8. positive measures, the sphere of radius 2

def D_alpha_pm(m, n, a):
    """(3.96): the alpha-divergence on positive measures."""
    if a == 1:
        return float(np.sum(m - n + n * np.log(n / m)))
    if a == -1:
        return float(np.sum(n - m + m * np.log(m / n)))
    return float(4 / (1 - a * a) * np.sum((1 - a) / 2 * m + (1 + a) / 2 * n - m ** ((1 - a) / 2) * n ** ((1 + a) / 2)))


def christoffel_from_metric(metric, x, h=1e-5):
    """Gamma_{ij}^k = (1/2) g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij}) by central differences (as in the notes on Chapter 5)."""
    n = len(x); gi = np.linalg.inv(metric(x)); dg = np.zeros((n, n, n))
    for l in range(n):
        e = np.zeros(n); e[l] = h
        dg[l] = (metric(x + e) - metric(x - e)) / (2 * h)
    G = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                G[i, j, k] = 0.5 * sum(gi[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j]) for l in range(n))
    return G


def riemann(gamma_fn, x, h=1e-5):
    """R_{ijk}^l = d_i Gamma_{jk}^l - d_j Gamma_{ik}^l + Gamma_{im}^l Gamma_{jk}^m - Gamma_{jm}^l Gamma_{ik}^m, (5.66)."""
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
    """K = g_{1m} R_{122}^m / det g for a surface."""
    return float(sum(g[0, m] * R[0, 1, 1, m] for m in range(2)) / np.linalg.det(g))


def fisher_energy_grad(z, A, B, N):
    """Discrete energy N sum d^T G(mid) d of a polyline A -> z -> B in the Fisher metric of S_2 (eta chart), and its gradient in the interior vertices."""
    P = np.vstack([A, z.reshape(-1, 2), B])
    d = np.diff(P, axis=0)
    mid = (P[:-1] + P[1:]) / 2
    p1, p2 = mid[:, 0], mid[:, 1]
    p0 = 1 - p1 - p2
    w = d[:, 0] ** 2 / p1 + d[:, 1] ** 2 / p2 + (d[:, 0] + d[:, 1]) ** 2 / p0
    E = N * float(np.sum(w))
    Gd = np.stack([d[:, 0] / p1 + (d[:, 0] + d[:, 1]) / p0, d[:, 1] / p2 + (d[:, 0] + d[:, 1]) / p0], 1)      # G(mid) d
    dmid = np.stack([-d[:, 0] ** 2 / p1 ** 2 + (d[:, 0] + d[:, 1]) ** 2 / p0 ** 2, -d[:, 1] ** 2 / p2 ** 2 + (d[:, 0] + d[:, 1]) ** 2 / p0 ** 2], 1)   # d_l (d^T G d)
    gP = np.zeros_like(P)
    gP[:-1] += -2 * N * Gd + 0.5 * N * dmid
    gP[1:] += 2 * N * Gd + 0.5 * N * dmid
    return E, gP[1:-1].reshape(-1)


def newton_polyline_s2(A, B, N, iters=40):
    z = np.array([A + t * (B - A) for t in np.linspace(0, 1, N + 1)[1:-1]]).reshape(-1)
    for _ in range(iters):
        E, g = fisher_energy_grad(z, A, B, N)
        H = np.zeros((len(z), len(z))); h = 1e-6
        for a in range(len(z)):
            e = np.zeros(len(z)); e[a] = h
            H[:, a] = (fisher_energy_grad(z + e, A, B, N)[1] - fisher_energy_grad(z - e, A, B, N)[1]) / (2 * h)
        H = (H + H.T) / 2
        step = np.linalg.solve(H, -g)
        t = 1.0
        while fisher_energy_grad(z + t * step, A, B, N)[0] > E + 1e-4 * t * (g @ step) and t > 1e-8:
            t /= 2
        z = z + t * step
        if np.linalg.norm(step) * t < 1e-12:
            break
    return np.vstack([A, z.reshape(-1, 2), B]), fisher_energy_grad(z, A, B, N)[0]


def check_positive_measures():
    head("8. Positive measures and the sphere of radius 2  (3.89)-(3.97), (3.92)-(3.94)")
    rng = rng_for(9)
    # standard f is needed on R^n_+
    m1, n1 = np.array([1.0]), np.array([0.5])
    print(f"   (3.89) with a non-standard f: m = (1), n = (0.5): f = u log u gives sum n log(n/m) = {float(np.sum(n1 * np.log(n1 / m1))):+.4f} < 0; the standard u log u - (u-1) gives {D_alpha_pm(m1, n1, 1):+.4f} (3.96)")
    M = rng.gamma(2, 1, (20000, 4)); Nn = rng.gamma(2, 1, (20000, 4))
    for nm, f in (("u log u", lambda u: u * np.log(u)), ("-log u", lambda u: -np.log(u)), ("(3.38) at alpha = 0.5", f_alpha_book(0.5)),
                  ("standard u log u - (u-1)", f_alpha(1)), ("standard (3.95) at alpha = 0.5", f_alpha(0.5))):
        D = Df(f, M, Nn)
        print(f"      f = {nm:32s} smallest D over 20000 random pairs of positive measures: {D.min():+.4f}")
    # the sign in (3.95)
    ar = []
    for a in (-3, -0.5, 0, 0.5, 3):
        ar.append((a, float(num_d1(f_alpha_printed(a), 1.0)), float(num_d1(f_alpha(a), 1.0)), -4 / (1 - a)))
    print("   (3.95) as printed has a linear term -2(u-1)/(1-alpha); the standard function needs +2(u-1)/(1-alpha). Slope at u = 1 (f'(1) must be 0):")
    for a, sp, ss, pred in ar:
        print(f"      alpha = {a:+.1f}: printed {sp:+.4f} (= -4/(1-alpha) = {pred:+.4f}),  with the sign corrected {ss:+.1e}")
    STORE["slope_rows"] = ar
    err_std, err_pr, neg = 0.0, 0.0, 0
    cnt = 0
    for i in range(3000):
        a = float(rng.choice([-3, -0.5, 0, 0.5, 3]))
        d96 = D_alpha_pm(M[i], Nn[i], a)
        err_std = max(err_std, abs(d96 - float(Df(f_alpha(a), M[i], Nn[i]))) / max(1, abs(d96)))
        dp = float(Df(f_alpha_printed(a), M[i], Nn[i]))
        err_pr = max(err_pr, abs(d96 - dp) / max(1, abs(d96))); neg += int(dp < -1e-12); cnt += 1
    print(f"   (3.96) against sum m f(n/m) over 3000 random pairs of positive measures, alpha drawn from -3, -0.5, 0, 0.5, 3: with the corrected sign the largest relative gap is {err_std:.1e}; with (3.95) as printed it is {err_pr:.1f},"
          f" and the printed function gives a negative 'divergence' in {neg} of {cnt} cases")
    STORE["pm_neg"] = (neg, cnt)
    # alpha duality and limits on R^n_+ (exact for the standard functions)
    mm, nn = M[0], Nn[0]
    print("   duality on R^n_+ through the standard functions (exact now): sum m f_alpha(n/m) - sum n f_{-alpha}(m/n) = " + ", ".join(f"{a}: {float(Df(f_alpha(a), mm, nn) - Df(f_alpha(-a), nn, mm)):+.1e}" for a in (0.3, 2.5, -0.7))
          + ";  limits alpha -> 1: " + ", ".join(f"{a}: {D_alpha_pm(mm, nn, a):.6f}" for a in (0.99, 0.9999)) + f" -> {D_alpha_pm(mm, nn, 1):.6f}")
    # scale behaviour
    c = 3.7
    print(f"   scale: D_f[c m : c n] = c D_f[m:n]: ratio {D_alpha_pm(c * mm, c * nn, 0.5) / D_alpha_pm(mm, nn, 0.5):.12f} for c = {c} (alpha = 0.5); the metric diag(1/m) scales as 1/c")
    # Theorem 3.4: the metric diag(1/m) from the Taylor expansion, for several f
    m0 = np.array([1.3, 0.7, 2.1]); d0 = np.array([.4, -.2, .3])
    print(f"   Theorem 3.4 / (3.91): D_f[m : m + eps d] / eps^2 -> (1/2) sum d^2/m = {0.5 * np.sum(d0 * d0 / m0):.6f} for m = {m0.tolist()}, d = {d0.tolist()}:")
    for a in (-1, -0.5, 0, 0.5, 1, 3):
        vals = [D_alpha_pm(m0, m0 + eps * d0, a) / eps ** 2 for eps in (0.1, 0.01, 0.001)]
        print(f"      alpha = {a:+.1f}: " + ", ".join(f"{v:.6f}" for v in vals))
    xi = lambda m_: 2 * np.sqrt(m_)
    Jm = np.diag(1 / np.sqrt(m0))                                 # d xi / d m
    gm = np.diag(1 / m0)
    print(f"   (3.92)-(3.93) xi = 2 sqrt(m): the Jacobian d m / d xi = diag(sqrt m) pulls diag(1/m) back to the identity: {np.round(np.linalg.inv(Jm).T @ gm @ np.linalg.inv(Jm), 12).tolist()};  sum xi^2 = 4 sum m: {float(np.sum(xi(m0) ** 2)):.4f} = 4 x {m0.sum():.1f}")
    # the simplex S_2 as a piece of a sphere of radius 2
    print("   S_2 in the chart xi = 2 sqrt(p): a sphere of radius 2 (3.94); checks of that statement made in the p chart, with the Fisher metric:")
    pts = [np.array([.3, .25]), np.array([.1, .6]), np.array([.45, .45])]
    Ks = []
    for x in pts:
        Rm = riemann(lambda y: christoffel_from_metric(fisher_eta_matrix_2, y, h=1e-5), x, h=1e-4)
        Ks.append(gauss_curvature(Rm, fisher_eta_matrix_2(x)))
    print(f"      Gauss curvature of the Fisher metric of S_2 (Christoffel and Riemann by finite differences) at {[x.tolist() for x in pts]}: {', '.join(f'{k:.6f}' for k in Ks)}; a sphere of radius 2 has 1/4")
    STORE["K_sphere"] = Ks
    # volume
    xs = np.random.default_rng(5).dirichlet(np.ones(3), 5)
    detgap = max(abs(math.sqrt(np.linalg.det(fisher_eta_matrix_2(x[1:]))) - 1 / math.sqrt(np.prod(x))) for x in xs)
    t, w = gl_nodes(60)
    vol = float(np.sum(w[:, None] * w[None, :] * 4 * np.sin(t * math.pi / 2)[:, None] * (math.pi / 2) * (math.pi / 2) * np.ones((1, 60))))
    print(f"      Riemannian volume: sqrt(det G) = 1/sqrt(p_0 p_1 p_2) (gap {detgap:.1e}); with p = (s_1^2, s_2^2, 1 - s_1^2 - s_2^2), s = r (cos phi, sin phi), r = sin t, the volume element becomes 4 sin t dt dphi: "
          f"total volume {vol:.10f} = 2 pi = {2 * math.pi:.10f} = 4 pi 2^2 / 8 (an octant of the sphere); Dirichlet: Gamma(1/2)^3 / Gamma(3/2) = {math.gamma(.5) ** 3 / math.gamma(1.5):.10f}")
    STORE["vol"] = vol
    # Fisher-Rao distance: formula, polyline minimiser, lengths of the e- and m-geodesic
    pf, qf = FIG_P, FIG_Q
    s_formula = 2 * math.acos(float(np.sum(np.sqrt(pf * qf))))
    print(f"   Fisher-Rao distance between p = {pf.tolist()} and q = {qf.tolist()}: formula 2 arccos(sum sqrt(p q)) = {s_formula:.6f}")
    polys = {}
    for Nseg in (8, 16, 32):
        Pl, E = newton_polyline_s2(pf[1:], qf[1:], Nseg)
        polys[Nseg] = (Pl, math.sqrt(E))
    print("      by minimising the discrete energy of a polyline in the p chart (Newton): " + ", ".join(f"{N} segments: {v[1]:.6f}" for N, v in polys.items()) + " (the errors shrink like 1/N^2)")
    STORE["polyline"] = polys[32][0]
    fig = STORE["thm32_fig"]
    wq = gl_nodes(200)[1]
    Le, Lm = float(np.sum(wq * np.sqrt(fig["ge"]))), float(np.sum(wq * np.sqrt(fig["gm"])))
    chord = float(np.linalg.norm(2 * np.sqrt(pf) - 2 * np.sqrt(qf)))
    print(f"      lengths in the Fisher metric: Fisher-Rao geodesic (great circle) {s_formula:.6f}, m-geodesic (straight in p) {Lm:.6f} ({100 * (Lm / s_formula - 1):.2f}% longer), e-geodesic {Le:.6f} ({100 * (Le / s_formula - 1):.2f}% longer), chord {chord:.6f}")
    STORE["lengths"] = (s_formula, Lm, Le, chord)
    print(f"      Hellinger and the chord: D_0[p:q] = {D_alpha(pf, qf, 0):.8f} = chord^2 / 2 = {chord ** 2 / 2:.8f} = 4 (1 - cos(s/2)) = {4 * (1 - math.cos(s_formula / 2)):.8f}")
    # warped product
    mq = rng.gamma(2, 1, 3); dm = rng.normal(size=3)
    Mtot = mq.sum(); pq = mq / Mtot
    dM = dm.sum(); dp_ = dm / Mtot - mq * dM / Mtot ** 2
    lhs = float(np.sum(dm * dm / mq))
    rhs = dM ** 2 / Mtot + Mtot * float(np.sum(dp_ * dp_ / pq))
    print(f"   scale behaviour: writing m = M p with M the total mass, sum dm^2/m = dM^2/M + M sum dp^2/p: {lhs:.12f} = {rhs:.12f} (gap {abs(lhs - rhs):.1e}); "
          f"the Fisher metric of the distributions p is multiplied by M and the radius of the sphere grows like 2 sqrt(M)")
    # Bregman on positive measures
    def mixed_pm(Dfun, chart, y, xs):
        n_ = len(y)
        h = 1e-4
        def mixed(x):
            Mx = np.zeros((n_, n_))
            for i in range(n_):
                for j in range(n_):
                    ei = np.zeros(n_); ei[i] = h; ej = np.zeros(n_); ej[j] = h
                    Dd = lambda a_, b_: Dfun(chart(a_), chart(b_))
                    Mx[i, j] = (Dd(x + ei, y + ej) - Dd(x + ei, y - ej) - Dd(x - ei, y + ej) + Dd(x - ei, y - ej)) / (4 * h * h)
            return Mx
        Ms = [mixed(x) for x in xs]
        return max(float(np.abs(Mx - Ms[-1]).max()) for Mx in Ms[:-1]) / float(np.abs(Ms[-1]).max())
    y0 = np.array([1.1, 0.9, 1.3])
    xs = [np.array([1.0, 1.4, 0.8]), np.array([0.6, 1.1, 1.6]), np.array([1.5, 0.7, 0.9]), y0]
    ident = lambda v: v
    out = {}
    out["KL (alpha=-1), m chart"] = mixed_pm(lambda a_, b_: D_alpha_pm(a_, b_, -1), ident, y0, xs)
    out["alpha=0.5, m chart"] = mixed_pm(lambda a_, b_: D_alpha_pm(a_, b_, 0.5), ident, y0, xs)
    out["alpha=0.5, chart m^((1-alpha)/2) = m^0.25"] = mixed_pm(lambda a_, b_: D_alpha_pm(a_, b_, 0.5), lambda th: th ** 4, y0 ** 0.25, [x ** 0.25 for x in xs])
    out["alpha=0 (Hellinger), chart sqrt(m)"] = mixed_pm(lambda a_, b_: D_alpha_pm(a_, b_, 0), lambda th: th ** 2, np.sqrt(y0), [np.sqrt(x) for x in xs])
    print("   on R^3_+ the picture is richer (Theorem 4.2): relative change of the mixed derivative, a Bregman divergence in the chart having 0:")
    for k, v in out.items():
        print(f"      {k:46s} {v:.1e}")
    STORE["pm_bregman"] = out


def fisher_eta_matrix_2(x):
    """Fisher information of S_2 in the chart (p_1, p_2)."""
    return np.diag(1 / x) + 1 / (1 - x.sum())


# ------------------------------------------------------------------ 9. Hotelling's remark: location-scale families

def halfline(fun, panels=80, order=8):
    """Integral of fun over (0, inf) by z = s/(1-s), s in (0,1), composite Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0, 1, panels + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = 0.5 * (hi - lo) * (x + 1) + lo
        z = s / (1 - s)
        with np.errstate(all="ignore"):
            v = np.nan_to_num(fun(z) / (1 - s) ** 2)
        tot += 0.5 * (hi - lo) * float(np.dot(w, v))
    return tot


def check_hotelling():
    head("9. Remark on location-scale families (Hotelling): constant negative curvature")
    fam = {
        "Gaussian": (lambda z: np.exp(-z * z / 2) / math.sqrt(2 * math.pi), lambda z: -z),
        "Laplace": (lambda z: 0.5 * np.exp(-np.abs(z)), lambda z: -np.sign(z)),
        "logistic": (lambda z: np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))) ** 2, lambda z: -np.tanh(z / 2)),
        "Cauchy": (lambda z: 1 / (math.pi * (1 + z * z)), lambda z: -2 * z / (1 + z * z)),
        "Student t, 3 d.f.": (lambda z: (2 / (math.pi * math.sqrt(3))) * (1 + z * z / 3) ** -2, lambda z: -4 * z / (3 + z * z)),
        "Gumbel (minimum)": (lambda z: np.exp(np.minimum(z, 700) - np.exp(np.minimum(z, 700))), lambda z: 1 - np.exp(np.minimum(z, 700))),
    }
    print("   density (1/s) q((x - m)/s): the Fisher matrix is [[a, b], [b, c]] / s^2 with a = E[q'/q]^2, b = E[(q'/q)(1 + z q'/q)], c = E[(1 + z q'/q)^2] (z = standardised variable);")
    print("   Gauss curvature of ds^2 = (a dm^2 + 2 b dm ds + c ds^2)/s^2 by finite differences at three points, and -a/(ac - b^2):")
    rows = []
    for name, (q, dlq) in fam.items():
        def integ(g):
            pos = halfline(lambda z: g(z) * q(z))
            neg = halfline(lambda z: g(-z) * q(-z))
            return pos + neg
        a = integ(lambda z: dlq(z) ** 2)
        b = integ(lambda z: dlq(z) * (1 + z * dlq(z)))
        c = integ(lambda z: (1 + z * dlq(z)) ** 2)
        mass = integ(lambda z: np.ones_like(z))
        metric = lambda x, a=a, b=b, c=c: np.array([[a, b], [b, c]]) / x[1] ** 2
        Ks = []
        for x in (np.array([0.3, 1.4]), np.array([-1.0, 0.6]), np.array([2.5, 3.1])):
            Rm = riemann(lambda y: christoffel_from_metric(metric, y, h=1e-5), x, h=1e-4)
            Ks.append(gauss_curvature(Rm, metric(x)))
        rows.append((name, a, b, c, Ks, -a / (a * c - b * b)))
        print(f"      {name:18s} (mass {mass:.6f})  a = {a:.5f}  b = {b:+.5f}  c = {c:.5f}   K = {Ks[0]:+.5f}, {Ks[1]:+.5f}, {Ks[2]:+.5f}   -a/(ac-b^2) = {-a / (a * c - b * b):+.5f}")
    STORE["hotelling"] = rows
    print(f"   Gaussian: -1/2, the value of Chapter 5; Gumbel: -6/pi^2 = {-6 / math.pi ** 2:.5f}")


# ------------------------------------------------------------------ 10. the default readouts of the interactive page

def check_widgets():
    head("10. Default readouts of the widgets of figures/interactive.html")
    p = np.array([.5, .3, .2]); q = np.array([.2, .2, .6])
    pb, qb = np.array([p[0], p[1] + p[2]]), np.array([q[0], q[1] + q[2]])
    print(f"   widget 1: p = {p.tolist()}, q = {q.tolist()}, outcomes 1 and 2 merged (p~ = {pb.tolist()}, q~ = {qb.tolist()}):")
    sqe = lambda a_, b_: float(np.sum((a_ - b_) ** 2))
    for name, (f, _) in FUNS.items():
        print(f"      {name:24s} D = {float(Df(f, p, q)):.6f}   D merged = {float(Df(f, pb, qb)):.6f}")
    print(f"      {'squared Euclid':24s} D = {sqe(p, q):.6f}   D merged = {sqe(pb, qb):.6f}")
    print(f"      {'non-convex f':24s} D = {float(Df(f_nonconvex, p, q)):.6f}   D merged = {float(Df(f_nonconvex, pb, qb)):.6f}")
    q2 = np.array([q[0], q[1:].sum() * p[1] / p[1:].sum(), q[1:].sum() * p[2] / p[1:].sum()])
    print(f"      equal conditionals q' = {np.round(q2, 4).tolist()}: KL = {kl(p, q2):.6f}, merged {kl(pb, np.array([q2[0], q2[1:].sum()])):.6f}")
    print(f"   widget 2: p, q as above; D_alpha[p:q] and the slope f'(1) of the three versions of the alpha function:")
    for a in (-3, -1, 0, 1, 3):
        if abs(a) == 1:
            print(f"      alpha = {a:+d}: D_alpha[p:q] = {D_alpha(p, q, a):.6f}   (the limit functions (3.42)/(3.95) are standard up to the linear term; no sign question)")
        else:
            print(f"      alpha = {a:+d}: D_alpha[p:q] = {D_alpha(p, q, a):.6f}   standard slope 0, (3.38) slope {-2 / (1 - a):+.4f}, printed (3.95) slope {-4 / (1 - a):+.4f}")
    print("   widget 3: D_f[p : p + eps d] / eps^2 for p = (.5,.3,.2), d = (.1,-.04,-.06), eps = 0.1, 0.01, 0.001:")
    d = np.array([.1, -.04, -.06])
    for name, f in (("KL", f_alpha(-1)), ("chi2", f_alpha(3)), ("Hellinger", f_alpha(0)), ("Jensen-Shannon", f_js), ("variation |1-u|", f_tv), ("(u-1)^4", f_quartic)):
        print(f"      {name:16s} " + "  ".join(f"{float(Df(f, p, p + e * d)) / e ** 2:.6f}" for e in (0.1, 0.01, 0.001)))
    print(f"      (1/2) sum d^2/p = {0.5 * np.sum(d * d / p):.6f}")
    print("   widget 7: split r = (0.3, 0.7) of the first outcome of q = (0.3, 0.7); squared length after / before for g = sum Z^2/p^a + B (sum Z)^2:")
    h = np.array([[0.3, 0.0], [0.7, 0.0], [0.0, 1.0]])                       # columns: where each outcome of S_1 goes
    q = np.array([0.3, 0.7]); pq = h @ q
    e1v = h[:, 0]
    print(f"      the embedded point h q = {np.round(pq, 4).tolist()}; squared length of e_1 before and of its image after: Fisher {1 / q[0]:.4f} -> {float(np.sum(e1v ** 2 / pq)):.4f}, Euclid {1.0:.4f} -> {float(np.sum(e1v ** 2)):.4f}")
    for a, B in ((0.0, 0.0), (0.0, 0.7), (1.0, 0.0), (1.0, 0.7), (2.0, 0.0)):
        gm = lambda pp, Z: float(np.sum(Z * Z / pp ** a) + B * np.sum(Z) ** 2)
        e1, zt = np.array([1.0, 0.0]), np.array([1.0, -1.0])
        print(f"      a = {a:.0f}, B = {B}: split vector e_1 {gm(pq, h @ e1) / gm(q, e1):.4f}   tangent vector e_1 - e_2 {gm(pq, h @ zt) / gm(q, zt):.4f}")


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


def fig_monotonicity(path):
    b = []
    W, H = 800, 410
    P1 = Panel(b, 70, 56, 320, 250, (0, 3.2), (0, 1.02))
    P1.frame([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 0.25, 0.5, 0.75, 1], T("D_{merged} / D"), "share of random merges with ratio ≤ x", "Does coarse graining lose information?", True)
    cdf = STORE["ratio_cdf"]
    P1.line([1, 1], [0, 1.02], "dash s0")
    for nm, cls in (("KL divergence", "ln s1"), ("squared Euclid", "ln s2"), ("non-convex f", "ln s3")):
        xs, ys = cdf[nm]
        P1.line(xs, ys, cls)
    legend_col(b, 70 + 175, 56 + 168, [("s1", "KL (any f-divergence)"), ("s2", "squared Euclid"), ("s3", "non-convex f")], lh=17)
    P2 = Panel(b, 470, 56, 300, 250, (0.4, 1.8), (0.3, 2.3))
    P2.frame([0.5, 0.75, 1, 1.25, 1.5], [0.5, 1, 1.5, 2], "a  (degree of homogeneity)", T("D_{merged} / D"), T("d_a(p,q) = p^a f(q/p)"), True)
    av, mean_suff, max_rand = STORE["a_curve"]
    P2.line([0.4, 1.8], [1, 1], "dash s0")
    P2.line(av, mean_suff, "ln s1")
    P2.line(av, max_rand, "ln s2")
    P2.dot(1.0, 1.0, "f4", 5)
    legend_col(b, 480, 78, [("s1", "equal conditionals (a sufficient merge)"), ("s2", "largest ratio over random merges")], lh=17)
    P2.text(0.43, 0.62, "a &lt; 1: loses information", "sm", "start"); P2.text(1.77, 0.42, "a &gt; 1: invents information", "sm", "end")
    note(b, 70, 376, ["Left: 20000 random merges of 6-outcome distributions; non-convex f is t/(1+t) with t = (u−1)². Every f-divergence stays at or below 1,",
                      "the Euclidean distance exceeds it in 33% of the cases. ",
                      "Right: the same f, weighted by p^a. Only a = 1 (an f-divergence) gives exactly 1 when the merged cells have equal conditionals, and never more."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Information monotonicity and what pins down the f-divergences",
        "Left: cumulative distribution of the ratio of the divergence after merging cells to the divergence before, over 20000 random merges, for the KL divergence (never above 1), the squared Euclidean distance (above 1 in a third of the cases) and a non-convex f. Right: for divergences p to the power a times f of q over p, the ratio at a sufficient merge equals 1 only for a equal to 1; for smaller a it is below 1, for larger a above 1.", b))


def fig_f_functions(path):
    b = []
    W, H = 800, 410
    u = np.linspace(0.12, 3.0, 300)
    P1 = Panel(b, 70, 56, 320, 250, (0, 3), (0, 3))
    P1.frame([0, 1, 2, 3], [0, 1, 2, 3], "u", "standard f_α(u)", "f(1) = f′(1) = 0, f″(1) = 1", True)
    P1.line(u, 0.5 * (u - 1) ** 2, "dash sk")
    for a, cls in ((-3, "ln s4"), (-1, "ln s2"), (0, "ln s3"), (1, "ln s1"), (3, "ln s0")):
        P1.line(u, f_alpha(a)(u), cls)
    legend(b, 205, 74, [("s4", "α = −3"), ("s2", "α = −1 (KL)"), ("s3", "α = 0"), ("s1", "α = +1"), ("s0", "α = 3 (χ²)"), ("dash sk", "(u−1)²/2")], col=100, lh=16)
    P2 = Panel(b, 470, 56, 300, 250, (0, 3), (-7.5, 5))
    P2.frame([0, 1, 2, 3], [-6, -4, -2, 0, 2, 4], "u", "f_0(u)", "α = 0: three versions", True)
    P2.line(u, f_alpha(0)(u), "ln s3")
    P2.line(u, f_alpha_book(0)(u), "ln s1")
    P2.line(u, f_alpha_printed(0)(u), "ln s2")
    for slope, cls in ((0, "dash s3"), (-2, "dash s1"), (-4, "dash s2")):
        P2.line([0.15, 1.9], [slope * (x - 1) for x in (0.15, 1.9)], cls)
    P2.dot(1, 0, "f4", 5)
    P2.text(2.95, 1.35, "standard: slope 0", "sm", "end", 0, -4); P2.text(2.95, -2.7, "(3.38): slope −2", "sm", "end", 0, 12); P2.text(2.95, -6.3, "(3.95) as printed: slope −4", "sm", "end", 0, 0)
    note(b, 70, 376, ["Left: every standard f touches the same parabola at u = 1, so every f-divergence has the same local quadratic form (the Fisher metric).",
                      "Right: the standard function has +2(u−1)/(1−α) added to (3.38); the printed sign makes the slope −4/(1−α) and the function negative."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The standard alpha functions and the sign in (3.95)",
        "Left: the standard alpha functions for alpha equal to minus 3, minus 1, 0, 1 and 3 on the interval 0 to 3, all passing through zero at u equal to 1 with zero slope and unit curvature, tangent to the parabola (u minus 1) squared over 2. Right: for alpha equal to 0, the book's first function (3.38) with slope minus 2 at u equal to 1, the standard function with slope 0 obtained by adding 2(u minus 1), and the formula (3.95) as printed with slope minus 4 and negative values for large u.", b))


# ------------------------------------------------------------------ main

def main():
    check_toolbox()
    check_invariance()
    check_monotone()
    check_examples()
    check_properties()
    check_kl()
    check_fisher_chentsov()
    check_positive_measures()
    check_hotelling()
    check_widgets()
    print("\nall checks ran")


VIEW = (0.35, 0.2)


def tern_map(x0, y0, side):
    """Barycentric (q0, q1, q2) -> pixels: vertex 0 bottom left, vertex 1 bottom right, vertex 2 on top; (x0, y0) is vertex 0."""
    V = np.array([[x0, y0], [x0 + side, y0], [x0 + side / 2, y0 - side * SQ3 / 2]])
    return lambda q: tuple(np.asarray(q, float) @ V)


def kl_contour(p, c, nang=360):
    """Points q of the simplex (barycentric triples) with KL[q:p] = c, along rays from p; a ray that leaves the simplex first is dropped."""
    out, seg = [], []
    for phi in np.linspace(0, 2 * math.pi, nang + 1):
        d = np.array([math.cos(phi), math.sin(phi)])
        sd = d.sum()
        rmax = math.inf
        for k in range(2):
            if d[k] < 0:
                rmax = min(rmax, -p[1 + k] / d[k])
        if sd > 0:
            rmax = min(rmax, p[0] / sd)
        qr = lambda r: np.array([p[0] - r * sd, p[1] + r * d[0], p[2] + r * d[1]])
        def klr(r):
            q = qr(r)
            m = q > 0
            return float(np.sum(q[m] * np.log(q[m] / p[m])))
        if klr(rmax) < c:
            if seg:
                out.append(seg); seg = []
            continue
        lo, hi = 0.0, rmax
        for _ in range(60):
            mid = (lo + hi) / 2
            if klr(mid) < c: lo = mid
            else: hi = mid
        seg.append(qr((lo + hi) / 2))
    if seg:
        out.append(seg)
    return out


def fig_large_deviation(path):
    ld = STORE["ld"]; rows = STORE["ld_rows"]
    p, qs, lam, I, var = ld["p"], ld["qs"], ld["lam"], ld["I"], ld["var"]
    b = []
    W, H = 800, 410
    side = 255
    X0, Y0 = 70 + 32, 56 + 238
    M = tern_map(X0, Y0, side)
    tri = [M((1, 0, 0)), M((0, 1, 0)), M((0, 0, 1))]
    b.append(f'<text class="hd" x="70" y="44">Sanov: the cheapest way to leave p</text>')
    A = [M((0, 0, 1)), M((0, .9, .1)), M((.45, 0, .55))]
    b.append('<polygon class="f1" style="opacity:.22" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in A) + '"/>')
    poly(b, tri + [tri[0]], "thin s0")
    for c, cls in ((0.03, "con s0"), (0.06, "con s0"), (I, "ln s2"), (0.2, "con s0")):
        for seg in kl_contour(p, c):
            if len(seg) > 1:
                poly(b, [M(q) for q in seg], cls)
    # the e-geodesic (exponential tilt) and the m-geodesic p -> q*
    tilt = lambda l: (lambda w: w / w.sum())(p * np.exp(l * np.array([0., 1., 2.])))
    poly(b, [M(tilt(l)) for l in np.linspace(-0.45, 1.05, 80)], "ln s3")
    poly(b, [M(p), M(qs)], "dash s0")
    for q, cls, lab, dx, dy in ((p, "f4", "p", -14, -6), (qs, "f2", "q*", 8, 4)):
        x, y = M(q)
        b.append(f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="5"/><text class="v" x="{x + dx:.1f}" y="{y + dy:.1f}">{lab}</text>')
    for k, lab in enumerate(("x = 0", "x = 1", "x = 2")):
        x, y = tri[k]
        b.append(f'<text class="sm" x="{x + (-8 if k == 0 else 8 if k == 1 else 0):.1f}" y="{y + (16 if k < 2 else -8):.1f}" text-anchor="{"end" if k == 0 else "start" if k == 1 else "middle"}">{lab}</text>')
    xA, yA = M((.2, .1, .7))
    b.append(f'<text class="sm" x="{xA + 14:.1f}" y="{yA - 22:.1f}">A: mean of x ≥ 1.1</text>')
    legend_col(b, 70, 56 + 276, [("s2", "KL[q:p] = I, the level that just touches A"), ("s3", "e-geodesic from p (exponential tilt)")], lh=15)
    P2 = Panel(b, 470, 56, 300, 250, (1, 3.25), (0.1, 0.27))
    P2.frame([1, 1.477, 2, 2.477, 3], [0.1, 0.15, 0.2, 0.25], "N (log scale)", "−(1/N) log P(p̂ ∈ A)", "Exact probabilities (multinomial)", True,
             {1: "10", 1.477: "30", 2: "100", 2.477: "300", 3: "1000"})
    Ns = np.logspace(1, 3.25, 80)
    P2.line([1, 3.25], [I, I], "dash s0")
    P2.line(np.log10(Ns), [I + math.log((1 - math.exp(-lam)) * math.sqrt(2 * math.pi * N * var)) / N for N in Ns], "ln s3")
    for N, rate, _, _ in rows:
        P2.dot(math.log10(N), rate, "f1", 4)
    P2.text(1.05, 0.108, "I = KL[q*:p] = 0.1231", "sm", "start")
    P2.text(1.1, 0.264, "dots: exact; curve: Bahadur–Rao", "sm", "start")
    note(b, 70, 376, ["Left: p = (.5,.3,.2) on the simplex; A is the corner where the sample mean of x = (0,1,2) is at least 1.1. The KL level set through q* touches A there.",
                      "Right: the exponent −(1/N) log P tends to I = KL[q*:p] like I + O(log N / N), with the polynomial factor of Bahadur–Rao (ratio 0.998 at N = 1600)."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Sanov's theorem and the large deviation exponent",
        "Left: the probability simplex of three outcomes with p equal to (0.5, 0.3, 0.2), level curves of the KL divergence from q to p, the shaded corner A where the sample mean of x in {0,1,2} is at least 1.1, the exponential tilt of p (an e-geodesic) reaching the boundary of A at q*, and the straight m-geodesic from p to q*. Right: the exact value of minus one over N times the logarithm of the probability that the empirical distribution lies in A for N from 10 to 1600, tending to the Cramér rate 0.1231, with the Bahadur-Rao refinement curve.", b))


def fig_symmetrised_kl(path):
    fg = STORE["thm32_fig"]; sw = STORE["thm32_sweep"]
    b = []
    W, H = 800, 410
    P1 = Panel(b, 70, 56, 320, 250, (0, 1), (0, 18.5))
    P1.frame([0, 0.25, 0.5, 0.75, 1], [0, 5, 10, 15], "t", "Fisher information along the curve", "g_e(t) and g_m(t) between p and q", True)
    t, ge, gm, J = fg["t"], fg["ge"], fg["gm"], fg["J"]
    pts = [(P1.X(0), P1.Y(0))] + [(P1.X(x), P1.Y(y)) for x, y in zip(t, ge)] + [(P1.X(1), P1.Y(0))]
    b.append('<polygon class="f1" style="opacity:.14" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    P1.line(t, gm, "ln s2"); P1.line(t, ge, "ln s1")
    P1.line([0, 1], [J, J], "dash s0"); P1.line([0, 1], [J / 2, J / 2], "dash s4")
    P1.text(0.5, 8.6, f"area under either curve = {J:.3f} = D[p:q] + D[q:p]", "sm", "middle")
    P1.text(0.5, 6.4, f"half of that, {J / 2:.3f}, is what (3.66) as printed would give", "sm", "middle")
    legend_col(b, 160, 78, [("s2", "g_m(t), m-geodesic"), ("s1", "g_e(t), e-geodesic (area shaded)")], lh=15)
    P2 = Panel(b, 470, 56, 300, 250, (0, 2.0), (0.4, 1.25))
    P2.frame([0, 0.5, 1, 1.5, 2], [0.5, 0.75, 1, 1.25], "Fisher–Rao distance s from p", "ratio to s²", "Shrinking the pair: all ratios tend to 1 or 1/2", True)
    xs = [r[0] for r in sw]
    P2.line([0, 2], [1, 1], "dash s0")
    P2.line(xs, [r[1] for r in sw], "ln s1")
    P2.line(xs, [r[2] for r in sw], "dot s2")
    P2.line(xs, [0.5 * r[3] for r in sw], "ln s4")
    P2.text(0.05, 1.13, "∫g_e / s² = ∫g_m / s² = (D + D*) / s²", "sm", "start")
    P2.text(1.97, 0.46, "½ (D + D*) / s²: the printed left side", "sm", "end", 0, -4)
    P2.text(1.97, 0.98, "s² ≤ ∫g (Cauchy–Schwarz)", "sm", "end", 0, 12)
    note(b, 70, 376, ["p = (.60,.38,.02), q = (.02,.38,.60). The Fisher information integrated along the e-geodesic and along the m-geodesic both equal D[p:q] + D[q:p]:",
                      "the factor ½ in (3.66) is wrong. Since s² ≤ ∫g for any curve of unit parameter length, ½ (D + D*) &lt; s² would be impossible."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Theorem 3.2: the symmetrised KL divergence is the integrated Fisher information",
        "Left: the Fisher information along the e-geodesic (smooth, between 3.3 and 4.2) and along the m-geodesic (large near both ends) between p equal to (0.6, 0.38, 0.02) and q equal to (0.02, 0.38, 0.6); each has area 3.945, which is the sum of the two KL divergences, and the dashed lines mark that mean height and its half. Right: along a path towards q, the ratios of the integrals and of the symmetrised divergence to the squared Fisher-Rao distance stay just above 1, while half the symmetrised divergence stays near one half.", b))


def fig_chentsov(path):
    b = []
    W, H = 800, 410
    side = 215
    X0, Y0 = 70 + 52, 56 + 192
    M = tern_map(X0, Y0, side)
    b.append('<text class="hd" x="70" y="44">Splitting one outcome in two: the image of S₁ inside S₂</text>')
    tri = [M((1, 0, 0)), M((0, 1, 0)), M((0, 0, 1))]
    poly(b, tri + [tri[0]], "thin s0")
    top = np.array(M((0, 0, 1)))
    info = []
    for r, cls, dx in ((0.2, "s1", -4), (0.5, "s3", 0), (0.8, "s2", 4)):
        end = np.array(M((r, 1 - r, 0)))
        poly(b, [tuple(top), tuple(end)], "ln " + cls)
        eu, fl = split_image_lengths(r)
        info.append((r, eu, fl))
        b.append(f'<text class="sm" x="{end[0] + dx:.1f}" y="{end[1] + 15:.1f}" text-anchor="middle">r = {r}</text>')
    STORE["chentsov_lengths"] = info
    b.append(f'<text class="v" x="70" y="{56 + 232}">Euclid length in the picture, r = .2, .5, .8: {info[0][1]:.3f}, {info[1][1]:.3f}, {info[2][1]:.3f}</text>')
    b.append(f'<text class="v" x="70" y="{56 + 249}">Fisher length of each image: {info[0][2]:.4f}, {info[1][2]:.4f}, {info[2][2]:.4f}</text>')
    b.append(f'<text class="sm" x="70" y="{56 + 266}">all equal to π: S₁ is a quarter circle of radius 2 for every r</text>')
    P2 = Panel(b, 470, 56, 300, 250, (0, 2.2), (0, 3.4))
    P2.frame([0, 0.5, 1, 1.5, 2], [0, 1, 2, 3], "a  (the metric is Σ Z²/p^a)", "‖ẽ‖² / ‖e‖² = Σ rᵢ^(2−a)", "Length² of the split vector e₁", True)
    aa = np.linspace(0, 2.2, 111)
    for rr, cls in (((0.5, 0.5), "s1"), ((0.3, 0.7), "s3"), ((0.2, 0.3, 0.5), "s2")):
        P2.line(aa, [sum(x ** (2 - a) for x in rr) for a in aa], "ln " + cls)
    P2.line([0, 2.2], [1, 1], "dash s0"); P2.line([1, 1], [0, 3.4], "dash s0")
    P2.dot(1, 1, "f4", 5)
    P2.text(1.03, 0.2, "a = 1: Fisher", "sm", "start"); P2.text(0.03, 0.2, "a = 0: Euclid", "sm", "start")
    legend_col(b, 480, 78, [("s1", "r = (.5, .5)"), ("s3", "r = (.3, .7)"), ("s2", "r = (.2, .3, .5)")], lh=15)
    note(b, 70, 376, ["Left: the three embeddings h_r of S₁ into S₂ that split the first outcome by (r, 1−r). Right: the Euclidean metric (a = 0) shrinks the split vector,",
                      "a > 1 stretches it; only the Fisher weight a = 1 returns exactly 1 for every split, which is what invariance asks."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "Markov embeddings and the Fisher metric",
        "Left: the triangle of distributions on three outcomes with three segments that are the images of the one-dimensional simplex under the embeddings that split the first outcome into two with proportions 0.2, 0.5 and 0.8; the Euclidean lengths in the picture differ (0.917, 0.866, 0.917) while the Fisher length of each image is pi. Right: the squared length of the split basis vector as a function of the weight exponent a, equal to 1 for every split only at a equal to 1.", b))


def fig_sphere(path):
    b = []
    W, H = 800, 425
    pf, qf = FIG_P, FIG_Q
    s_fr, Lm, Le, chord = STORE["lengths"]
    P1x, P1y, P1w, P1h = 70, 56, 320, 250
    # orbit camera: start from the diagonal view (1,1,1) with vertex 0 on the left, 1 on the right, 2 on top, then turn by yaw and pitch
    ex0, ey0, w0 = np.array([-1, 1, 0]) / math.sqrt(2), np.array([-1, -1, 2]) / math.sqrt(6), np.array([1, 1, 1]) / math.sqrt(3)
    yaw, pitch = VIEW
    wv = math.cos(yaw) * math.cos(pitch) * w0 + math.sin(yaw) * math.cos(pitch) * ex0 + math.sin(pitch) * ey0
    exv = math.cos(yaw) * ex0 - math.sin(yaw) * w0
    eyv = np.cross(wv, exv)
    proj = lambda xi: (xi @ exv, xi @ eyv)
    # bounding box of the octant: sample its three edges
    th = np.linspace(0, math.pi / 2, 200)
    edges = [2 * np.stack([np.cos(th), np.sin(th), 0 * th], -1), 2 * np.stack([0 * th, np.cos(th), np.sin(th)], -1), 2 * np.stack([np.sin(th), 0 * th, np.cos(th)], -1)]
    allp = np.concatenate(edges)
    X, Y = proj(allp)
    sc = min((P1w - 50) / (X.max() - X.min()), (P1h - 30) / (Y.max() - Y.min()))
    cx, cy = P1x + P1w / 2 - sc * (X.max() + X.min()) / 2, P1y + P1h / 2 + sc * (Y.max() + Y.min()) / 2
    px = lambda xi: [(cx + sc * x, cy - sc * y) for x, y in zip(*proj(np.asarray(xi)))]
    b.append(f'<text class="hd" x="{P1x}" y="{P1y - 12}">S₂ as an octant of a sphere of radius 2 (ξ = 2√p)</text>')
    for e in edges:
        poly(b, px(e), "thin s0")
    # curves p_i = const
    for i in range(3):
        for c in (0.25, 0.5, 0.75):
            ph = np.linspace(0, math.pi / 2, 100); rad = 2 * math.sqrt(1 - c)
            xi = np.zeros((100, 3)); xi[:, i] = 2 * math.sqrt(c); xi[:, (i + 1) % 3] = rad * np.cos(ph); xi[:, (i + 2) % 3] = rad * np.sin(ph)
            poly(b, px(xi), "gd")
    xp, xq = 2 * np.sqrt(pf), 2 * np.sqrt(qf)
    tt = np.linspace(0, 1, 100)
    om = s_fr / 2
    gc = np.array([(math.sin((1 - t) * om) * xp + math.sin(t * om) * xq) / math.sin(om) for t in tt])
    mg = np.array([2 * np.sqrt((1 - t) * pf + t * qf) for t in tt])
    eg = np.array([2 * np.sqrt((lambda w: w / w.sum())(pf ** (1 - t) * qf ** t)) for t in tt])
    ch = np.array([(1 - t) * xp + t * xq for t in tt])
    poly(b, px(ch), "dash sk")
    poly(b, px(eg), "ln s3"); poly(b, px(mg), "ln s2"); poly(b, px(gc), "ln s1")
    for xi, lab, dx, dy in ((xp, "p", -14, -6), (xq, "q", 8, -6)):
        x, y = px(xi[None, :])[0]
        b.append(f'<circle class="f4 ring" cx="{x:.1f}" cy="{y:.1f}" r="5"/><text class="v" x="{x + dx:.1f}" y="{y + dy:.1f}">{lab}</text>')
    for k in range(3):
        v = np.zeros(3); v[k] = 2
        x, y = px(v[None, :])[0]
        b.append(f'<text class="sm" x="{x + (-6 if k == 0 else 6 if k == 1 else 0):.1f}" y="{y + (14 if k < 2 else -8):.1f}" text-anchor="{"end" if k == 0 else "start" if k == 1 else "middle"}">outcome {k}</text>')
    legend(b, 70, 56 + 276, [("s1", f"Fisher–Rao (great circle): {s_fr:.3f}"), ("s2", f"m-geodesic: {Lm:.3f} (+{100 * (Lm / s_fr - 1):.1f}%)"), ("s3", f"e-geodesic: {Le:.3f} (+{100 * (Le / s_fr - 1):.1f}%)"), ("dash sk", f"chord (in the ball): {chord:.3f}")], col=190, lh=16)
    P2 = Panel(b, 470, 56, 300, 250, (0, 3.2), (0, 4.6))
    P2.frame([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 1, 2, 3, 4], "Fisher–Rao distance s", "divergence", "Hellinger divergence versus distance", True)
    ss = np.linspace(0, math.pi, 200)
    P2.line([0, 3.2], [4, 4], "dot s0")
    P2.line(ss, 0.5 * ss ** 2, "dash s0")
    P2.line(ss, 4 * (1 - np.cos(ss / 2)), "ln s3")
    P2.dot(s_fr, 4 * (1 - math.cos(s_fr / 2)), "f4", 5)
    legend_col(b, 482, 112, [("s3", "D₀ = 4(1 − cos(s/2)) = chord²/2"), ("dash s0", "s²/2, the local form"), ("dot s0", "bound (3.47): 4")], lh=16)
    P2.text(s_fr + 0.12, 4 * (1 - math.cos(s_fr / 2)) - 0.5, "the pair on the left", "sm", "start")
    P2.text(math.pi, 0.3, "π: disjoint", "sm", "end")
    note(b, 70, 376, ["Left: S₂ as the octant ξ = 2√p of a sphere of radius 2 (thin lines: p_i = .25, .5, .75). Its Gauss curvature is 1/4 and its area 2π,",
                      "both computed in the p chart from the Fisher metric. The Fisher–Rao geodesic is a great circle; the m- and e-geodesics are longer.",
                      "Right: the Hellinger divergence is half the squared chord, 4(1 − cos(s/2)), and saturates at 4 when the supports are disjoint."], "sm", 15)
    open(path, "w", encoding="utf-8").write(svg(W, H, "The probability simplex is a piece of a sphere",
        "Left: the octant of a sphere of radius 2 that is the simplex of three outcomes in the coordinates 2 times the square root of p, with the great circle through p equal to (0.6, 0.38, 0.02) and q equal to (0.02, 0.38, 0.6) (Fisher-Rao length 1.857), the m-geodesic (1.905), the e-geodesic (1.985) and the chord (1.791). Right: the Hellinger divergence 4(1 minus cos(s/2)) against the Fisher-Rao distance s, starting like s squared over 2 and saturating at 4 for disjoint distributions at s equal to pi.", b))


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_monotonicity(out / "monotonicity.svg")
    fig_f_functions(out / "f-functions.svg")
    fig_large_deviation(out / "large-deviation.svg")
    fig_symmetrised_kl(out / "symmetrised-kl.svg")
    fig_chentsov(out / "chentsov.svg")
    fig_sphere(out / "sphere.svg")
    print("\nwrote", ", ".join(sorted(q.name for q in out.glob("*.svg"))))


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        make_figures()
