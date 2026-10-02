---
title: "Chapter 3: invariant geometry of the manifold of probability distributions"
short_title: "Ch. 3 — Invariant geometry"
chapter: 3
category: "Part I"
book_pages: "51–69"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [invariance, information-monotonicity, sufficient-statistic, f-divergence, alpha-divergence, hellinger, fisher-information, chentsov, markov-embedding, sanov, large-deviations, positive-measures, sphere]
status: read
---

## Links

- **[Interactive companion](figures/interactive.html)**: eight widgets, all exact sums (no sampling). (1) Merge two outcomes of a three-outcome distribution and read a divergence before and after, for eight divergences; a button makes the merged outcomes' conditionals equal.
  (2) The $\alpha$ family: the standard function, the book's (3.38) and the printed (3.95), their slopes at $u=1$, the dual function and $D_\alpha$ on positive measures. (3) $D_f[p\Vert p+\varepsilon d]/\varepsilon^2$ against $\varepsilon$: it tends to $\tfrac12 f''(1)\sum d_i^2/p_i$; the variation distance and $(u-1)^4$ do not.
  (4) Zero forcing and zero avoiding: the best Gaussian for a two-humped $p$ as $\alpha$ moves. (5) Large deviations: the exact multinomial probability against the exponent $\mathrm{KL}[q^*\Vert p]$ and the e-projection $q^*$. (6) Theorem 3.2: the Fisher information along the e- and m-geodesics and the two areas.
  (7) Chentsov: the length of a split vector under the weight $p^{-a}$. (8) The simplex as a sphere of radius 2: three curves between two points, the chord, and the effect of the total mass.
- **[Runnable checks](https://github.com/msrepo/information_geometry_notes/tree/main/chapters/ch03-invariant-geometry/code)**:
  `code/invariant_geometry.py` prints every number on this page and regenerates the figures with `python3 code/invariant_geometry.py --figures`; `make verify` runs it in about five seconds (numpy only, fixed seeds, exact sums and quadrature rather than simulation wherever possible).
- The book: Amari, *Information Geometry and Its Applications* (Springer, 2016), DOI [10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8). These notes cover Chapter 3 only; equation numbers such as (3.66) refer to the book.
  No text of the book is reproduced here; everything is restated and re-derived.
- Neighbouring chapters: **[Chapter 2: exponential and mixture families](../ch02-exponential-and-mixture-families/index.html)** (where KL appeared as a Bregman divergence) and **[Chapter 1: the dually flat structure](../ch01-dually-flat-structure/index.html)**; ahead, **[Chapter 4: $\alpha$-geometry](../ch04-alpha-geometry/index.html)** takes up the $\alpha$-divergences of §3.3 and the question of which invariant divergences are also flat, and **[Chapter 6: dual connections](../ch06-dual-connections/index.html)** the connections that the cubic tensor of §3.5 defines. **[Chapter 5](../ch05-elements-of-differential-geometry/index.html)** supplies the curvature routine I reuse for the sphere.
  Background on the annotations site: **[The Fisher information matrix](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/)**.

## In one paragraph

Chapters 1 and 2 built a whole geometry out of one convex function: its gap above a tangent is a divergence, its slope is a second coordinate system, its curvature is a metric. For distributions the function was the log-partition function and the divergence was KL. That raises a fair question: why *that* function and why that divergence? Chapter 3 answers with a requirement that has nothing to do with convexity, **invariance**. Describe the same experiment differently and nothing about how far apart two hypotheses are should change. Relabelling the outcomes changes nothing; summarising them (merging outcomes, "coarse graining") can only throw information away, so a divergence may only go **down**, and it should stay exactly the same when the summary throws away nothing the parameter cares about (a **sufficient statistic**).
Among divergences that are sums over outcomes, this requirement leaves one family, the **$f$-divergences** $\sum_ip_if(q_i/p_i)$ with $f$ convex (Theorem 3.1): KL in both orders, $\chi^2$, Hellinger, the $\alpha$-divergences and the variation distance are members, the squared Euclidean distance is not. Every standard $f$-divergence looks, for nearby distributions, like half the **Fisher information**, and Chentsov's theorem says Fisher information is the *only* Riemannian metric that is invariant. On the way the chapter lists the properties that make KL the workhorse of statistics (it governs large deviations, and its symmetrisation is an integral of the Fisher information), and shows that in the coordinates $2\sqrt{p_i}$ the simplex of distributions is a piece of a sphere of radius 2.
I re-checked all of it on small explicit examples. The main results hold. I found five printed formulas that are wrong: (3.48) (the two arguments exchanged), (3.56) ($\sqrt N$ written as $1/\sqrt N$), (3.95) (the sign of a linear term) and, by a factor $2$, (3.45) and (3.66); several missing hypotheses (strict convexity, $f''(1)>0$, normalisation); and a gap in the proof of Chentsov's theorem (it does not establish uniqueness on positive measures).

## The spine of the argument

1. A geometry of distributions should not depend on how the experiment is described. A merge $y=k(x)$ must not increase the divergence, $\bar D\le D$ (3.4), with equality exactly when $y$ is sufficient: this is the **invariance criterion** (§3.1).
2. A divergence that is a sum over outcomes, $\sum_id(p_i,q_i)$, can satisfy the criterion only if $d(p,q)=pf(q/p)$ with $f$ convex, and every such $f$-divergence does (Theorem 3.1; the forward direction is Jensen's inequality). Adding $c(u-1)$ to $f$ changes nothing on normalised distributions; one normalises to $f(1)=f'(1)=0$, $f''(1)=1$ ("standard"); the dual $f^*(u)=uf(1/u)$ swaps the arguments (§3.2).
3. Examples: KL and its dual, $\chi^2$, the $\alpha$-divergences (Hellinger at $\alpha=0$), the variation distance; the squared Euclidean distance fails (§3.3).
4. Properties: convex in both arguments, bounded above except for some $\alpha$, infinite when supports mismatch (hence zero forcing and zero avoiding approximations). For KL: Sanov's lemma, the central limit theorem, the large deviation theorem (rare events happen along the e-projection), and Theorem 3.2 (§3.4).
5. Locally every standard $f$-divergence is $\tfrac12$ the Fisher quadratic form, and Chentsov's theorem (Campbell's proof, via Markov embeddings of $S_m$ in $S_n$) says Fisher information is the unique invariant metric (§3.5).
6. On positive measures $\mathbb R^n_+$ the same classification works with *standard* $f$; the metric is $\mathrm{diag}(1/m_i)$, flat in the chart $\xi=2\sqrt m$, and $S_n$ is the sphere $\sum\xi_i^2=4$ (§3.6).

## Setup and notation

| Symbol | Meaning | In the examples |
|---|---|---|
| $S_n$ | distributions $p=(p_0,\dots,p_n)$, all $p_i>0$, on the outcomes $\{0,\dots,n\}$ | $p=(.5,.3,.2)$ |
| $y=k(x)$, $\bar p$ | a coarse graining (outcomes merged into cells $X_a$) and the induced distribution $\bar p_a=\sum_{i\in X_a}p_i$ (3.8) | merge outcomes 1 and 2 |
| $D[p:q]$, $\bar D$ | a divergence and its value after coarse graining (3.4) | |
| sufficient | $p(x\mid y)$ does not depend on the parameter (3.12)–(3.13) | the sum of two Bernoulli trials |
| $D_f[p:q]=\sum_ip_if(q_i/p_i)$ | $f$-divergence (3.14); I write $u_i=q_i/p_i$ | |
| standard $f$ | convex, $f(1)=f'(1)=0$, $f''(1)=1$ (3.28), (3.30) | $-\log u+u-1$ |
| $f^*(u)=uf(1/u)$ | dual function: $D_{f^*}[p:q]=D_f[q:p]$ (3.31)–(3.32) | |
| $D_\alpha$ | $\alpha$-divergence from $f_\alpha(u)=\frac4{1-\alpha^2}(1-u^{(1+\alpha)/2})$ (3.38): $\alpha=-1$ is $\mathrm{KL}[p\Vert q]$, $\alpha=0$ Hellinger, $\alpha=3$ $\chi^2$ | |
| Markov embedding $h$ | $q\mapsto p=rq$ with $r_{ij}=\mathrm{Prob}(x=i\mid y=j)$ living on the cell $A_j$ (3.71)–(3.72) | splitting an outcome in two |
| $g_{ij}$, $T_{ijk}$ | Fisher information (3.68) and cubic tensor (3.88) | |
| $\mathbb R^n_+$ | positive measures $m,n$ with free total mass (the book also calls the number of outcomes $n$; I use the pair $m,n$ only for positive measures) | |
| $\xi=2\sqrt m$ | the chart in which $\mathbb R^n_+$ is Euclidean (3.92) | |

**Running examples.** (i) Small distributions on 3 to 6 outcomes, on which every coarse graining can be enumerated; random pairs (fixed seeds) give statistics. The default pair is $p=(.5,.3,.2)$, $q=(.2,.2,.6)$. (ii) Two Bernoulli trials, where the sum is sufficient. (iii) Gaussians, where $x\mapsto x^2$ is sufficient on one line of the family and on nothing larger. (iv) A $3\times3$ table with all its mass on the diagonal, for zero forcing. (v) $S_2$ with the pair $p=(.60,.38,.02)$, $q=(.02,.38,.60)$ for Theorem 3.2 and the sphere. Outcomes are numbered from $0$ as in the book.

## 1. The invariance criterion (§3.1)

**In plain words.** If you describe the same experiment in two ways, the geometry should not notice. There are two kinds of re-description. A *one-to-one* relabelling of the outcomes changes nothing, so the divergence must be unchanged. A *many-to-one* summary $y=k(x)$ (you only learn which cell $x$ fell in) generally loses information about which distribution is true, so the divergence between the two summarised distributions should be **at most** the original, $\bar D\le D$ (3.4), the book's *information monotonicity*. When the summary is *sufficient*, it keeps everything the data say about the parameter, and then nothing should be lost: equality. Sufficient means that, given the summary, the remaining detail $x$ is distributed the same way whichever hypothesis is true (3.12)–(3.13), or in factorised form $p(x,\xi)=\bar p(s,\xi)r(x)$ (3.6). The **invariance criterion** is: a geometric structure is invariant if it satisfies (3.4) with equality exactly for sufficient $y$.

*A small case.* $p=(.4,.3,.2,.1)$, $q=(.1,.2,.3,.4)$. Merge $\{0,1\}$ and $\{2,3\}$: $\mathrm{KL}[p\Vert q]=0.4564$ becomes $0.3389$ for the merged pair $(.7,.3)$, $(.3,.7)$: information lost. Now choose $q$ so that $q_i/p_i$ is constant inside each cell, say $q=(.2,.15,.4333,.2167)$ (mass $.35$ on the first cell, split in the proportions of $p$): $\mathrm{KL}$ is $0.253246$ before and after, equal to $6\times10^{-17}$. Inside each cell $p$ and $q$ now distribute their mass identically, so only the cell totals can tell them apart, and the cell totals survive.

**Checks.**

- *Sufficient for a pair or for a model?* Take two Bernoulli trials $x=(x_1,x_2)$ with success probability $\theta$ and the sum $s=x_1+x_2$. For $\theta=0.3$ against $\theta'=0.8$ the KL divergence is $1.165371$ on the four outcomes, $1.165371$ for the binomial distribution of $s$, and $2\,\mathrm{kl}(\theta\Vert\theta')=1.165371$; and $p(x\mid s=1)=(0.5,0.5)$ for both $\theta$. The sum is sufficient *for this model*. The same map on the full simplex $S_3$ (generic pairs of distributions on the four outcomes) loses information: over 2000 random pairs the loss has minimum $5.5\times10^{-8}$ and median $0.0916$. In fact for the full simplex only a bijection is sufficient (the conditional $p_1/(p_1+p_2)$ is free), so the criterion's demand of equality exactly for sufficient statistics has to be read **for the two distributions being compared**, as (3.13) does (the conditionals of $p$ and $q$ agree), or for the model one works in; read model-wide on $S_n$ it would constrain nothing.
- *Continuous variables.* $\mathrm{KL}[N(m_1,s_1^2)\Vert N(m_2,s_2^2)]$ by quadrature on the transformed variable:

| $(m_1,s_1;m_2,s_2)$ | KL of $x$ | $y=3x+1$ | $y=e^x$ | $y=x^2$ |
|---|---|---|---|---|
| $(1,1;0,2)$ | 0.443147 | 0.443147 | 0.443147 | 0.106316 |
| $(0,1;0,2)$ | 0.318147 | 0.318147 | 0.318147 | 0.318147 |
| $(1,1;-1,1)$ | 2.000000 | 2.000000 | 2.000000 | 0.000000 |
| $(2,1;0,1.5)$ | 1.016576 | 1.016576 | 1.016576 | 0.383856 |

  Invertible maps keep KL. The non-invertible $x\mapsto x^2$ is sufficient for the scale family $\{N(0,s^2)\}$ (second row: equality) and for nothing larger: the third row loses everything, since $x^2$ forgets the sign of the mean. The same map, two different verdicts, depending on which model one has in mind.
- *A notational point.* In (3.6) the factor called $\bar p(s,\xi)$ is the induced marginal only if $r$ sums to one on each fibre; the factorisation can always be normalised that way, so nothing depends on it.

## 2. Information monotonicity under coarse graining (§3.2)

### 2.1 $f$-divergences are monotone (Theorem 3.1, first half)

**In plain words.** Suppose the divergence is a sum over outcomes of a pointwise cost, $D=\sum_id(p_i,q_i)$ (3.9) ("decomposable"). Merging two outcomes replaces $d(p_1,q_1)+d(p_2,q_2)$ by $d(p_1+p_2,q_1+q_2)$. An **$f$-divergence** uses the cost $d(p,q)=p\,f(q/p)$ with $f$ convex and $f(1)=0$ (3.14)–(3.15): $q/p=u$ says how much $q$ inflates the outcome compared with $p$, and the cost is the weighted $f(u)$. Merging replaces $u_1,u_2$ by their weighted average $\bar u=(p_1u_1+p_2u_2)/(p_1+p_2)=(q_1+q_2)/(p_1+p_2)$ (3.17), and *Jensen's inequality* (a convex function lies below its chords, so the average of $f$ is at least $f$ of the average) gives
$$(p_1+p_2)f(\bar u)\le p_1f(u_1)+p_2f(u_2)\qquad(3.16),(3.19).$$
The book proves it for one merge; for a general cell the same weights $p_i/P_a$ do it.

*A worked case.* $p=(.5,.3,.2)$, $q=(.2,.2,.6)$, so $u=(0.4,\,0.667,\,3.0)$; merge outcomes 1 and 2 (cell totals $p\to(.5,.5)$, $q\to(.2,.8)$):

| divergence | before | after merging |
|---|---|---|
| $\mathrm{KL}[p\Vert q]$ | 0.360062 | 0.223144 |
| dual KL, $\mathrm{KL}[q\Vert p]$ | 0.394816 | 0.192745 |
| $\chi^2$ | 0.506667 | 0.180000 |
| Hellinger | 0.369652 | 0.205267 |
| Jensen–Shannon | 0.090566 | 0.050672 |
| variation $\sum\lvert p_i-q_i\rvert$ | 0.800000 | 0.600000 |

The first widget has these numbers and a button that makes $u_1=u_2$.

**Checks.**

- *Random coarse grainings.* 20000 random pairs on 6 outcomes, each merged into 2, 3, 4 or 5 cells at random, for nine $f$-divergences. The largest $\bar D/D$ is $1.000000$ for all nine; the smallest relative loss $(D-\bar D)/D$ is positive for the eight smooth ones ($4.7\times10^{-10}$ for KL, $1.9\times10^{-10}$ dual KL, $3.8\times10^{-11}$ $\chi^2$, $4.2\times10^{-10}$ dual $\chi^2$, $3.3\times10^{-10}$ Hellinger, about $3\times10^{-10}$ for the rest). For the variation distance it is zero: $0.1877$ of the random merges lose nothing although the partition is not sufficient (see 2.2).
- *Equality for equal conditionals.* With $q_i/p_i$ constant on every cell (20000 merges), $\lvert D-\bar D\rvert\le2\times10^{-12}$ for all nine.
- *Noisy summaries.* Any Markov kernel is an embedding (a randomisation that does not depend on the parameter, which loses nothing) followed by a merge, so it is monotone too. 800 random kernels from 6 outcomes to 2–9 outcomes: the largest $D[Kp:Kq]/D[p:q]$ is $0.4352$ (KL), $0.3680$ (dual KL), $0.3675$ ($\chi^2$), $0.4074$ (Hellinger), $0.4328$ (Jensen–Shannon), $0.7572$ (variation). For the book's embeddings (3.71)–(3.72), 300 random $r$: equality to $5.7\times10^{-14}$ for all nine.

### 2.2 What the proof leaves out

- **Strict convexity.** The invariance criterion asks for equality *only* for sufficient $y$. Jensen gives equality when $u_1=u_2$, but for an $f$ that is linear on a stretch it also gives equality when $u_1\ne u_2$ lie in that stretch. The variation distance is the obvious case (excluded by the book as non-differentiable), but a perfectly smooth $f$ does it too: take $f=\tfrac12(u-1)^2$ for $u\le2$ and $u-\tfrac32$ beyond (convex, $C^1$, $f(1)=f'(1)=0$, $f''(1)=1$). For $p=(.1,.1,.8)$, $q=(.3,.4,.3)$ ($u=3,4,0.375$; the first two outcomes have *different* conditionals) merging outcomes 0 and 1 leaves this $f$ at $0.556250$ before and after, and the variation distance at $1.0000$ before and after. So the claim that every $f$-divergence is invariant needs **strictly** convex $f$.
- **$f''(1)>0$.** The remark in §3.2.2 that the divergence criteria follow by a Taylor expansion needs the second derivative at 1 to be positive. $f\equiv0$ and $f=(u-1)^4$ are convex with $f(1)=0$; $D_f[p\Vert p+\varepsilon d]/\varepsilon^2$ for $p=(.5,.3,.2)$, $d=(.1,-.04,-.06)$ is $2.5\times10^{-5},\,2.5\times10^{-7},\,2.5\times10^{-9}$ at $\varepsilon=0.1,0.01,0.001$ for the quartic and $0$ for $f=0$: no quadratic form, hence no metric (compare $0.021667=\tfrac12\sum d^2/p$ for $f=\tfrac12(u-1)^2$).

### 2.3 The converse: decomposable and invariant means $f$-divergence

**In plain words.** Suppose $D=\sum d(p_i,q_i)$ satisfies the criterion. Merge two outcomes with $q_1/p_1=q_2/p_2=u$: nothing is lost, so $d(p_1,up_1)+d(p_2,up_2)=d(p_1+p_2,u(p_1+p_2))$ (3.20). Writing $k(p,u)=d(p,up)$ this says $k(p_1,u)+k(p_2,u)=k(p_1+p_2,u)$ (3.22): $k$ is additive in $p$, hence (with some regularity) linear, $k(p,u)=pf(u)$ (3.23), so $d(p,q)=pf(q/p)$ (3.24). In other words the criterion forces the cost to be **homogeneous of degree one** in $(p,q)$. Convexity of $f$ is then forced by (3.16), since (3.16) *is* convexity.

*Checks.* Take a cost of degree $a$, $d_a(p,q)=p^af(q/p)$, with $f=-\log u+u-1$.

- Additivity (3.22) fails unless $a=1$: the residual $k(p_1,u)+k(p_2,u)-k(p_1+p_2,u)$ at $p_1=0.2$, $p_2=0.3$, $f(u)=1$ is $+0.2878$ ($a=0.5$), $0$ ($a=1$), $-0.0998$ ($a=1.5$), $-0.1200$ ($a=2$).
- The consequences, on the 20000 random merges: for $a=1.5$ the merged value exceeds the original in $0.198$ of the cases (largest ratio $1.970$) and at equal conditionals $\bar D/D$ lies in $[1.0000,2.0568]$ (invents information); for $a=0.5$ it is never exceeded but at equal conditionals $\bar D/D$ lies in $[0.4976,1.0000]$ (loses information a sufficient statistic should keep). Only $a=1$ gives $1$ exactly.
- The other candidates. Squared Euclidean distance (3.46): the merged value is larger in $0.326$ of the merges (largest ratio $2.884$); explicitly $p=(.4,.4,.2)$, $q=(.2,.2,.6)$ has $0.24$ before and $0.32$ after merging the first two outcomes, although they have equal conditionals ($u=0.5$ for both). **Itakura–Saito**, a decomposable Bregman divergence other than KL, is monotone in all 20000 trials but at equal conditionals the ratio drops to $0.2001$: it is not invariant. A non-convex $f=t/(1+t)$, $t=(u-1)^2$, is a perfectly good-looking positive divergence with a metric ($f''(1)=2$), but is not monotone ($0.241$ of the merges increase it, largest ratio $2.185$; explicitly $p=(.05,.05,.9)$, $q=(.15,.45,.4)$ gives $0.3015$ before and $0.3084$ after merging the first two outcomes).

<img src="figures/monotonicity.svg" alt="Left: cumulative distribution of the ratio of a divergence after merging cells to its value before, over 20000 random merges: the KL divergence reaches 100 percent at ratio 1, the squared Euclidean distance and a non-convex f do not. Right: for costs p to the power a times f of q over p, the ratio at a sufficient merge is below 1 for a below 1, above 1 for a above 1, and exactly 1 only for a equal to 1.">

*Two caveats the book does not spell out.* The step from additive to linear needs a regularity assumption (Cauchy's functional equation has wild solutions), which is harmless for the smooth divergences of Chapter 1. And the book's **Remark 1** (the case $n=1$): on $S_1$ the only coarse graining sends everything to one point, so (3.4) says nothing there. Any decomposable $D$ passes; for instance $D=(p-q)^2+((1-p)-(1-q))^2=2(p-q)^2$ has the constant metric $4$, whereas an $f$-divergence on $S_1$ has $f''(1)/(p(1-p))=11.1111,\,4.7619,\,4.0000$ at $p=0.1,0.3,0.5$. This is consistent with the book's remark; I did not look at the cited classification.
**Remark 2** is also right: $D+D^2$ is invariant (monotone, equality exactly when $D$'s is) but not decomposable (the mixed derivative $\partial^2/\partial m_0\partial n_1$ is $0.2026$, against $0$ for a sum over outcomes), and since $D^2=O(\varepsilon^4)$ it has the same metric and cubic tensor as $D$.

### 2.4 The freedom $f\to f+c(u-1)$, scale, standard form, duality

On normalised distributions $\sum_i p_i(q_i/p_i-1)=\sum(q_i-p_i)=0$, so adding $c(u-1)$ to $f$ changes nothing (3.26)–(3.27): the difference is $0.0$ on $S_n$. **It is false on positive measures**: for $m,n$ with total masses $2.1$ and $3.4$ and $c=0.7$ the difference is $+0.910000=c(\sum n-\sum m)$. The statement (3.27) holds on $S_n$ only, which matters in §6. The condition $f(1)=0$ (3.15) is just what makes $D_f[p:p]=0$ (otherwise a distribution would have a nonzero divergence from itself, $D_f[p:p]=f(1)$), and by Jensen it also gives $D_f\ge0$ on $S_n$. Scaling $f$ by $c>0$ scales $D_f$ (3.29); the normalisation $f''(1)=1$ (3.30) fixes the scale so that the local quadratic form is exactly $\tfrac12$ Fisher information (§5).
The **dual** $f^*(u)=uf(1/u)$ satisfies $D_{f^*}[p:q]=D_f[q:p]$ (3.32): largest gap $8.5\times10^{-14}$ over 2000 random pairs for all eight smooth $f$'s; $f^*$ is convex with $f^*(1)=0$, $f^{*\prime}(1)=-f'(1)$ and $f^{*\prime\prime}(1)=f''(1)$, so standardness is preserved (checked numerically for each).

## 3. Examples of $f$-divergence in $S_n$ (§3.3)

**In plain words.** KL is $f=-\log u$: the cost $p\log(p/q)$. Its dual $f^*=u\log u$ gives $\mathrm{KL}[q\Vert p]$, which by Chapter 2 is the Bregman divergence of the cumulant function (gap to the explicit Bregman formula $1.0\times10^{-15}$ on 200 pairs). $f=\tfrac12(u-1)^2$ gives $\tfrac12\sum(p-q)^2/p$ (3.37). The **$\alpha$-family** $f_\alpha(u)=\frac4{1-\alpha^2}(1-u^{(1+\alpha)/2})$ gives
$$D_\alpha[p:q]=\frac4{1-\alpha^2}\Big(1-\sum_ip_i^{\frac{1-\alpha}2}q_i^{\frac{1+\alpha}2}\Big)\qquad(3.39),$$
with $D_\alpha[p:q]=D_{-\alpha}[q:p]$ (3.40), the Hellinger divergence $2\sum(\sqrt{p_i}-\sqrt{q_i})^2$ at $\alpha=0$ (3.41), and KL, in one order or the other, at $\alpha=\mp1$ (3.43). **Checks** (the formula (3.39) written in $p,q$ against $\sum p\,f(q/p)$ for $\alpha=-3,-0.5,0,0.5,3$, 300 pairs): gaps up to $2.3\times10^{-13}$; (3.41) $1.1\times10^{-15}$; the duality (3.40) through the standard functions $2.3\times10^{-13}$.

- **$\chi^2$ is the member $\alpha=3$** (the standard $f_3=\tfrac12(u-1)^2$ to $9\times10^{-16}$), and its dual $\tfrac12\sum(p_i-q_i)^2/q_i$ is $\alpha=-3$; the book lists $\chi^2$ separately.
- **(3.40) and (3.42) hold only up to the linear term (3.26).** The dual of the function (3.38) is not the function (3.38) of $-\alpha$: at $\alpha=0.5$ they differ by $4(u-1)/(1-\alpha^2)$, up to $16.000$ on $u\le4$. The divergences agree on $S_n$, the standard functions agree exactly (gap $3.1\times10^{-15}$). Likewise the limit $\alpha\to1$ (3.42) exists for the divergence ($D_\alpha[p:q]=0.273987,\,0.274793,\,0.274878,\,0.274886\to\mathrm{KL}[q\Vert p]=0.274887$ for $\alpha=0.9,0.99,0.999,0.9999$) but not for the function: $f_\alpha(2)=-19.62,\,-199.61,\,-1999.61$ diverges like $-2/(1-\alpha)$, while the standard function gives $0.3816,\,0.3858,\,0.3862\to u\log u-(u-1)=0.3863$.
- **Variation distance, a factor 2.** $f=\lvert1-u\rvert$ (3.44) gives $\sum\lvert p_i-q_i\rvert$, not the printed (3.45) $\tfrac12\sum\lvert p_i-q_i\rvert$: for $p=(.5,.3,.2)$, $q=p+(.1,-.04,-.06)$ it is $0.2000$, the printed would be $0.1000$ (and belongs to $f=\tfrac12\lvert1-u\rvert$). Its largest value over 2000 random pairs is $1.777$ (the supremum is 2). It is not a divergence in the book's sense: $D_f[p\Vert p+\varepsilon d]/\varepsilon=0.2000$ at $\varepsilon=0.1,0.01,0.001$, linear in $\varepsilon$, no quadratic form.
- **Squared Euclidean distance** (3.46): not an $f$-divergence, not invariant (§2.3).
- *An addition of mine:* Jensen–Shannon, $D=\tfrac12\mathrm{KL}[p\Vert m]+\tfrac12\mathrm{KL}[q\Vert m]$ with $m=(p+q)/2$, is an $f$-divergence with $f(u)=\tfrac12[u\log u-(1+u)\log\tfrac{1+u}2]$, $f^*=f$ and $f''(1)=\tfrac14$; it is bounded by $\log2=0.6931$ and is the example I use for a symmetric, bounded divergence.

<img src="figures/f-functions.svg" alt="Left: the standard alpha functions for alpha equal to minus 3, minus 1, 0, 1 and 3 on the interval 0 to 3, all passing through zero at u equal to 1 with zero slope and unit curvature, tangent to the parabola (u minus 1) squared over 2. Right: for alpha equal to 0, the book's function (3.38) with slope minus 2 at u equal to 1, the standard function with slope 0 obtained by adding 2(u minus 1), and the formula (3.95) as printed with slope minus 4 and negative values for large u.">

**Which $f$-divergences are Bregman divergences?** The book remarks (§3.2.2) that an $f$-divergence is in general not a Bregman divergence. A divergence is Bregman in a chart $x$ iff $\partial^2D[x:y]/\partial x_i\partial y_j$ does not depend on $x$ (it is minus the Hessian of the potential at $y$). On $S_2$ I computed that mixed derivative at four values of $x$ and report its relative change: in the $\eta$ chart $(p_1,p_2)$ it is $4.2\times10^{-10}$ (finite-difference noise: constant) for KL and of order one for everything else; in the logit chart $\theta$ it is $9.4\times10^{-9}$ for the dual KL and between $0.13$ and $1.2$ for everything else (in the $\eta$ chart the others lie between $0.21$ and $4.7$). So among these only KL, in $\eta$, and its dual, in $\theta$, are Bregman. This does not exclude another chart; the book's Theorem 4.1 says none works (for decomposable invariant divergences on $S_n$, $n>1$), and I have not checked its proof here. Chapter 4 returns to it.

## 4. General properties of $f$-divergence and KL (§3.4)

### 4.1 Properties of $f$-divergences

**(1) Convexity.** $D_f[p:q]$ is jointly convex because $(p,q)\mapsto pf(q/p)$ is. Over 20000 random quadruples the smallest relative margin is $8.8\times10^{-5}$ for the eight smooth $f$ (zero margin allowed for the variation distance: $-4\times10^{-16}$).

**(2) Upper bounds.** The maximum of a jointly convex function over $S_n\times S_n$ sits at extreme points, that is at two distinct vertices, where it equals $f(0)+\lim_{u\to0}uf(1/u)$: this is (3.47). Checks: Hellinger $4.0000$, $\alpha=\pm0.5$ give $5.3333=4/(1-\alpha^2)$, Jensen–Shannon $0.6931$, variation $2.0000$; the value at the nearly disjoint pair $p=(1-d,d)$, $q=(d,1-d)$, $d=10^{-9}$ is $3.9997$, $5.3033$, $0.6931$, $2.0000$ and no random pair exceeds it; for KL and $\chi^2$ there is no bound (the limit grows: $13.8\to27.6$ for KL, $5\times10^5\to5\times10^{11}$ for $\chi^2$).
**The second bound (3.48) is wrong as printed.** It claims $D_f[p:q]\le\sum_i(p_i-q_i)f'(p_i/q_i)$. Take $\chi^2$ with $p=(.9,.1)$, $q=(.5,.5)$: $D_f[p:q]=0.8889$ but the right-hand side is $0.6400$. Over 50000 random pairs on 4 outcomes it is violated for $0.263$ of the pairs for $\chi^2$, $0.168$ for the dual $\chi^2$, $0.053$ for KL and $0.016$ for $\alpha=-0.5$ (never for Hellinger, Jensen–Shannon, $\alpha=0.5$ and the dual KL). What is true is the same inequality with the roles exchanged: from $f(1)\ge f(u)+f'(u)(1-u)$ and $f(1)=0$ one gets $f(u)\le f'(u)(u-1)$, hence
$$D_f[p:q]\le\sum_i(q_i-p_i)f'(q_i/p_i),$$
The printed right-hand side, with $p_i/q_i$, is exactly the correct bound for $D_f[q:p]$, not for $D_f[p:q]$. The corrected version is never violated in the same 50000 trials ($1.7778\ge0.8889$ in the example).

**(3), (4) Infinite values.** $D_\alpha[p:q]=\infty$ if $\alpha\ge1$ and $p(x)=0<q(x)$ somewhere, and if $\alpha\le-1$ and $p(x)>0=q(x)$. For $p=(.5,.5,0)$, $q=(.4,.4,.2)$:

| $\alpha$ | $-3$ | $-1$ | $-0.5$ | $0$ | $0.5$ | $1$ | $3$ |
|---|---|---|---|---|---|---|---|
| $D_\alpha[p:q]$ ($p=0<q$ at the last outcome) | 0.1250 | 0.2231 | 0.2894 | 0.4223 | 0.8219 | $\infty$ | $\infty$ |
| $D_\alpha[q:p]$ ($p>0=q$) | $\infty$ | $\infty$ | 0.8219 | 0.4223 | 0.2894 | 0.2231 | 0.1250 |

**(5), (6) Zero forcing and zero avoiding.** If you approximate $p$ by the best $q$ in a family $S$ using $D_\alpha[p:q]$, then for $\alpha\ge1$ no mass may sit where $p$ has none (**zero forcing**: $\hat q=0$ wherever $p=0$), and for $\alpha\le-1$ $q$ must cover everything $p$ covers (**zero avoiding**). Two examples.

- A perfectly dependent $3\times3$ table $p=\mathrm{diag}(.5,.3,.2)$ approximated by independent tables $r\otimes s$ (grid of step $1/40$, then refined): $\alpha=-3,-2,-1,-0.5$ give full-support minimisers with $\min D_\alpha=0.93193,\,0.91665,\,1.02965,\,1.17705$ and mass $0.354,\,0.360,\,0.380,\,0.434$ on the diagonal (at $\alpha=-1$ it is $p_X\otimes p_Y$, $D=I(X;Y)=H(X)=1.02965$); but for $\alpha=0,\,0.5,\,1,\,3$ the best independent table is the **point mass on the biggest diagonal cell**, with $D_\alpha=1.17157,\,0.84855,\,0.69315\ (=-\log0.5),\,0.50000$.
- A smooth version: $p$ half $N(10,2^2)$, half $N(28,3^2)$ on 40 points (mean $19.00$, standard deviation $9.35$), best discretised Gaussian $N(\mu,\sigma^2)$ by grid search. For $\alpha=-3,-2,-1,-0.5,0$ it is wide, covering both humps: $(\mu,\sigma)=(17.50,11.640),\,(18.00,11.640),\,(18.75,11.357),\,(19.50,11.357),\,(20.50,11.081)$. For $\alpha=0.5,1,2,3$ it hides in the broader hump: $(28.00,3.008)$ every time, with $D_\alpha=0.8475,\,0.6926,\,0.5520,\,0.4998$.

Both follow the book. But the switch is **not at $\pm1$**: it happens between $\alpha=0$ and $0.5$ in the smooth example and already at $\alpha=0$ in the table (for $\alpha\in(-1,1)$ all divergences are finite, so the infinities of (3.49)–(3.50) cannot be the reason). The book's statement is a sufficient condition for the forcing, not the location of the transition. The fourth widget lets you slide through it.

### 4.2 KL: Sanov, the central limit theorem, large deviations

**In plain words.** Draw $N$ values from $p$ and let $\hat p$ be the empirical distribution (the proportions of each outcome). How likely is it to look like some other distribution $q$? **Sanov's lemma** (3.55): $\mathrm{Prob}(\hat p)\approx e^{-N\,\mathrm{KL}[\hat p\Vert p]}$, so the exponent is the KL divergence *from the observed to the true* distribution. Expanding KL near $p$ gives the quadratic form of the Fisher information, i.e. the Gaussian of the **central limit theorem** (3.57). And if you ask for the probability that $\hat p$ lands in a whole region $A$, the sum is dominated by the cheapest point: **large deviation theorem** (3.58)–(3.59), $\mathrm{Prob}(\hat p\in A)\approx e^{-N\,\mathrm{KL}[p_A^*\Vert p]}$ with $p_A^*=\arg\min_{q\in A}\mathrm{KL}[q\Vert p]$. That minimiser is where the e-geodesic from $p$ meets $A$ orthogonally: the **e-projection**. Rare events happen in the cheapest way.

**Checks.**

- *Sanov, exactly.* For $p=(.5,.3,.2)$ and $\hat p=(.2,.3,.5)$, $\mathrm{KL}[\hat p\Vert p]=0.274887$; the exact multinomial probability gives $-\frac1N\log P=0.521339,\,0.355131,\,0.301821,\,0.281880,\,0.276608$ for $N=10,50,200,1000,5000$, converging to $0.274887$. The "$\approx$" hides a polynomial factor: $P/[(2\pi N)^{-1}(\prod\hat p_i)^{-1/2}e^{-N\,\mathrm{KL}}]=0.925582,\,0.984568,\,0.996119,\,0.999223,\,0.999844$ (Stirling), while $e^{-N\,\mathrm{KL}}/P$ itself grows like $N$ ($5442$ at $N=5000$).
- *The central limit theorem, and (3.56).* The book sets $\varepsilon=\frac1{\sqrt N}(\hat p-p)$ before expanding $N\,\mathrm{KL}[\hat p\Vert p]$. That is a slip: $\hat p-p$ is of order $1/\sqrt N$, so the quantity that is of order one is $\varepsilon=\sqrt N(\hat p-p)$ (and (3.57) has covariance $g_{ij}/N$, i.e. $\sqrt N(\hat p-p)$ has covariance $g_{ij}$). Numbers at $N=400$, $\hat p-p=(.03,-.02,-.01)$: $N\,\mathrm{KL}=0.72752$; $\tfrac12\varepsilon^{\mathsf T}G\varepsilon=0.72667$ with $\varepsilon=\sqrt N(\hat p-p)$ and $G=\mathrm{diag}(1/p_i)+1/p_0$; with the printed $\varepsilon$ it would be $4.5\times10^{-6}$.
- *The covariance (3.57).* The exact covariance of $\hat p$ for $N=12$ (sum over all 91 outcomes) is $\left[\begin{smallmatrix}0.21&-0.06\\-0.06&0.16\end{smallmatrix}\right]/N=\mathrm{diag}(p)-pp^{\mathsf T}$ to $3.6\times10^{-16}$. That is "$g_{ij}$" in the book's convention (lower index: the metric of the natural parameters $\theta$, the covariance of the statistic); the **Fisher matrix of the chart $p$ is its inverse** $\mathrm{diag}(1/p_i)+1/p_0$ (product $=I$ to the digits shown). So (3.57) is right only with that reading. A local check at $N=400$: $P(\hat p=(.53,.28,.19))=0.001142$ against the Gaussian density with that covariance, times the cell area, $0.001111$ (ratio $1.0285$).
- *Large deviations, three routes to the exponent.* $A=\{q:\text{mean of }x\ge1.1\}$, $x=(0,1,2)$, $p=(.5,.3,.2)$ (mean $0.7$). The e-projection is an exponential tilt $q^*_i\propto p_ie^{\lambda x_i}$ with $\lambda=0.606136$, $q^*=(0.29032,0.31935,0.39032)$, $\mathrm{KL}[q^*\Vert p]=0.12313394$. Cramér's transform $\sup_\lambda(a\lambda-\log\mathbb Ee^{\lambda x})$ gives $0.12313394$ and brute force over a $1600\times1600$ grid of $A$ gives $0.123135$ at $(.2906,.3188,.3906)$. The Fisher inner product of the boundary direction $(1,-2,1)$ with the e-geodesic's velocity at $q^*$ is $0$ (to rounding), while the Euclidean cosine of the angle in the picture of the triangle is $0.082$: orthogonality is a statement about the Fisher metric. The Pythagorean inequality $\mathrm{KL}[q\Vert p]\ge\mathrm{KL}[q\Vert q^*]+\mathrm{KL}[q^*\Vert p]$ holds for all 80902 random $q\in A$ (smallest slack $9.6\times10^{-6}$).
- *The exact probability.* The multinomial sum gives $-\frac1N\log P(\hat p\in A)=0.249505,\,0.200092,\,0.161918,\,0.145750,\,0.136107,\,0.130469,\,0.127230,\,0.125397$ at $N=10,20,50,100,200,400,800,1600$, approaching $I=0.123134$ like $\log N/N$. The polynomial factor is the Bahadur–Rao one, $e^{-NI}/((1-e^{-\lambda})\sqrt{2\pi N\sigma^2})$ with $\sigma^2$ the variance of $x$ under $q^*$: the ratio of the exact probability to it is $0.833859,\,0.895316,\,0.948900,\,0.972132,\,0.985353,\,0.992475,\,0.996184,\,0.998078$. I tested only a convex $A$; for a non-convex region there can be several local e-projections and the theorem as stated (for a closed region with a boundary) would need the cheapest.

<img src="figures/large-deviation.svg" alt="Left: the probability simplex of three outcomes with p equal to (0.5, 0.3, 0.2), level curves of the KL divergence from q to p, the shaded corner A where the sample mean of x in {0,1,2} is at least 1.1, the exponential tilt of p (an e-geodesic) reaching the boundary of A at q star, and the straight m-geodesic from p to q star. Right: the exact value of minus one over N times the logarithm of the probability that the empirical distribution lies in A for N from 10 to 1600, tending to the Cramér rate 0.1231, with the Bahadur-Rao refinement curve.">

### 4.3 Theorem 3.2: the symmetrised KL is an integrated Fisher information

**In plain words.** Join $p$ and $q$ by the e-geodesic $p_t\propto p^{1-t}q^t$ and by the m-geodesic $(1-t)p+tq$, and measure at each $t$ the Fisher information $g(t)$ of the moving point in the direction it is moving. The theorem says the area under $g_e$ and the area under $g_m$ are the same number, a symmetrised KL. The book omits the proof as technical; it takes two lines. Along the e-geodesic $\partial_t\log p_t=\log(q/p)-\text{const}$, so $g_e(t)=\mathrm{Var}_{p_t}[\log(q/p)]=\psi''(t)$ for $\psi(t)=\log\sum p^{1-t}q^t$, and $\int_0^1\psi''=\psi'(1)-\psi'(0)=\mathbb E_q\log\frac qp-\mathbb E_p\log\frac qp=\mathrm{KL}[q\Vert p]+\mathrm{KL}[p\Vert q]$. Along the m-geodesic $g_m(t)=\sum(q_i-p_i)^2/p_{t,i}$ and $\int_0^1\frac{(q-p)^2}{(1-t)p+tq}dt=(q-p)\log\frac qp$ term by term, the same sum.
**The factor $\tfrac12$ in (3.66) is wrong.** The sum, not half the sum, equals both integrals. Numbers: five random pairs on 4 outcomes give $\int g_e=\int g_m=\mathrm{KL}[p\Vert q]+\mathrm{KL}[q\Vert p]$ to ten digits ($0.2913584789$, $0.1271592108$, $1.4922201743$, $1.0694497183$, $2.7142067396$), with half of each sum visibly different. For the figure pair $p=(.60,.38,.02)$, $q=(.02,.38,.60)$: $\int g_e=\int g_m=3.945389=\mathrm{KL}+\mathrm{KL}^*$, half of it $1.972694$, and $s^2=3.447955$ with $s$ the Fisher–Rao distance. A local sanity check shows why half cannot be right: near $p$, $\mathrm{KL}\approx\tfrac12s^2$ each, so the sum is $s^2$ and half the sum is $s^2/2$, while $s^2\le\int g$ for any curve of unit parameter length (Cauchy–Schwarz): the ratios of $\int g_e/s^2$, $\int g_m/s^2$, $(D+D^*)/s^2$ are all $1.000108,\,1.000008,\,1.000001,\,1.000000$ at distances $s=0.1050,\,0.0313,\,0.0104,\,0.0031$ and $\tfrac12(D+D^*)/s^2$ is $0.500054,\dots,0.500000$. The theorem also holds for densities: $N(0,1)$ and $N(1,1.3^2)$ give $\int g_e=0.93671598$, $\int g_m=0.93671599$ and $\mathrm{KL}+\mathrm{KL}^*=0.93671598$ (for wider pairs the m-geodesic's $g_m$ blows up at the ends, still with a finite integral; I used nodes clustered at $t=0,1$).

<img src="figures/symmetrised-kl.svg" alt="Left: the Fisher information along the e-geodesic (smooth, between 3.3 and 4.2) and along the m-geodesic (large near both ends) between p equal to (0.6, 0.38, 0.02) and q equal to (0.02, 0.38, 0.6); each has area 3.945, which is the sum of the two KL divergences, and the dashed lines mark that mean height and its half. Right: along a path towards q, the ratios of the integrals and of the symmetrised divergence to the squared Fisher-Rao distance stay just above 1, while half the symmetrised divergence stays near one half.">

## 5. Fisher information: the unique invariant metric (§3.5)

### 5.1 Every standard $f$-divergence gives the Fisher metric

**In plain words.** For nearby distributions, $D_f[p:p+dp]=\sum p_i f(1+dp_i/p_i)=\sum p_i\{f'(1)\tfrac{dp_i}{p_i}+\tfrac12f''(1)\tfrac{dp_i^2}{p_i^2}+\dots\}$. The first term is $f'(1)\sum dp_i=0$ and the second is $\tfrac12f''(1)\sum dp_i^2/p_i$: **half $f''(1)$ times the Fisher information** of the displacement (3.67)–(3.68). So all $f$-divergences have the same local geometry up to the scalar $f''(1)$, which the standard normalisation sets to 1; they differ only in higher-order terms.
*Checks.* The Hessian of $D_f[p:p']$ in $p'$ at $p'=p$, on $S_3$ in the $p$ chart ($p=(.4,.3,.2,.1)$), against $f''(1)(\mathrm{diag}(1/p_i)+1/p_0)$: relative error $\le1.3\times10^{-5}$ for all eight smooth $f$ (finite-difference size; $10^{-11}$ for the exactly quadratic $\chi^2$), with $f''(1)=1$ except Jensen–Shannon ($\tfrac14$). For Gaussians at $(\mu,\sigma)=(1,2)$ (quadrature in the continuous case): the Hessian is $\mathrm{diag}(0.25,0.5)=\mathrm{diag}(1/\sigma^2,2/\sigma^2)$ for KL, Hellinger and $\alpha=0.5$ ($\chi^2$: $0.25$, $0.50001$), and $\mathrm{diag}(0.0625,0.125)$ for Jensen–Shannon, a quarter of it. (3.67) also contains a typo: its right argument is printed $p(x:\xi+d\xi)$, a colon for a comma.

### 5.2 Chentsov's theorem

**In plain words.** The Fisher metric is not just *one* invariant metric; it is the only one (up to a constant). Invariance for metrics means: take a distribution $q$ on $m$ outcomes, *split* each outcome $j$ into several sub-outcomes in fixed proportions $r_{ij}$ (this adds randomness that does not depend on which $q$ is true, so loses nothing; it is a **Markov embedding** $h:q\mapsto p=rq$, (3.71)–(3.73)), and demand that lengths of tangent vectors are unchanged, $\langle h_*Z,h_*W\rangle_{hq}=\langle Z,W\rangle_q$. The book follows Campbell's proof, which uses only splittings into equal parts: put $q=(k_1,\dots,k_m)/n$ ($k_j$ integers) and split outcome $j$ into $k_j$ equal sub-outcomes; then $q$ goes to the uniform distribution on $n$ outcomes, whose metric by symmetry is $A(n)\delta_{ij}+B(n)$ (3.75)–(3.77); the $B$ part is invisible on tangent vectors (their components sum to zero) (3.78)–(3.80); the pushed-forward basis vector is $\tilde e_1=\frac1{k_1}(e_1+\dots+e_{k_1})$ (3.84) and its length gives $g_{11}(q)=A(n)/k_1$, the same as $c/q_1$ if $A(n)=cn$ (3.85)–(3.87); continuity extends from rational to all $q$.

*A small case.* $q=(.3,.7)$, split the first outcome into $(.3,.7)$ of it: $hq=(.09,.21,.70)$. The vector $e_1$ (all the change in outcome 0) has squared Fisher length $1/q_0=3.3333$ and its image $(.3,.7,0)$ has $0.3^2/.09+0.7^2/.21=3.3333$ again; the Euclidean squared length goes from $1.0000$ to $0.58$.

**Checks.**

- *Fisher is invariant, others are not.* 400 random embeddings of $S_2$ (3 outcomes split into cells of 1–4 outcomes, random proportions) and random tangent vectors $Z$ ($\sum Z=0$): the largest $\lvert\langle hZ,hZ\rangle/\langle Z,Z\rangle-1\rvert$ is $4.4\times10^{-16}$ for $\sum Z_i^2/p_i$ and $0.739$ for the Euclidean $\sum Z_i^2$, $0.491$ for the weight $p^{-1/2}$, $0.988$ for $p^{-3/2}$, $3.00$ for $p^{-2}$ (the Hessian of the Burg entropy $-\sum\log p_i$); and $4.4\times10^{-16}$ again for $\sum Z_i^2/p_i+0.7(\sum Z_i)^2$, for tangent *and* for arbitrary $Z$ (see the second gap below).
- *Monotone and invariant are separate demands.* Coarse graining a vector (merging cells) gives $\lVert f_*Z\rVert^2/\lVert Z\rVert^2\le1$ for Fisher (largest $1.0000$, and exactly $1$ for "horizontal" $Z$ with $Z_i/p_i$ constant on cells). For the weight $p^{-a}$ with $a<1$ ($a=0.5$) the ratio can exceed $1$ (largest $1.5549$: not monotone) and with $a>1$ it stays below $1$ but at horizontal vectors drops to $0.4750$ ($a=1.5$) and $0.2533$ ($a=2$): the same pattern as for the costs $p^af(q/p)$ in §2.3.
- *The proof's own computation.* $n=12$, $k=(2,3,7)$, $q=(.1667,.25,.5833)$; splitting into equal parts gives the uniform distribution, and the pushed-forward basis vectors have Fisher lengths squared $6.0000,\,4.0000,\,1.7143$ ($=n/k_j=1/q_j$) with zero cross terms. For the tangent vector $e_1-e_2$: $10.0000=1/q_1+1/q_2$.
- *Splitting changes length by $\sum_ir_i^{2-a}$.* For the weight $p^{-a}$ the squared length of the split vector $\tilde e$ is multiplied by $\sum_ir_i^{2-a}$: for $r=(.5,.5)$ at $a=0,0.5,1,1.5,2$ it is $0.5000,\,0.7071,\,1.0000,\,1.4142,\,2.0000$; for $r=(.3,.7)$ it is $0.5800,\,0.7500,\,1.0000,\,1.3844,\,2.0000$; for $r=(.2,.3,.5)$ it is $0.3800,\,0.6073,\,1.0000,\,1.7020,\,3.0000$. Only $a=1$ gives 1 for every split. The picture: the image of $S_1$ under the splitting by $(r,1-r)$ is a segment of $S_2$ whose Euclidean length in the picture of the triangle depends on $r$ ($0.9165,\,0.8660,\,0.9165$ for $r=.2,.5,.8$) while its Fisher length is $\pi=3.1416$ for every $r$ ($S_1$ is a quarter circle of radius 2, §6).
- *The cubic tensor (3.88).* $T_b(p;Z)=\sum Z_i^3/p_i^b$ under the same embeddings: $b=2$ (the expectation of the cube of the score, the cubic tensor) is invariant to $1.7\times10^{-15}$; $b=1,1.5,2.5,3$ fail by $9.68,\,26.5,\,3.64,\,323$. This shows invariance of $T$ and the failure of the obvious competitors, not the uniqueness claimed in the remark.

<img src="figures/chentsov.svg" alt="Left: the triangle of distributions on three outcomes with three segments that are the images of the one-dimensional simplex under the embeddings that split the first outcome into two with proportions 0.2, 0.5 and 0.8; the Euclidean lengths in the picture differ (0.917, 0.866, 0.917) while the Fisher length of each image is pi. Right: the squared length of the split basis vector as a function of the weight exponent a, equal to 1 for every split only at a equal to 1.">

**Two gaps in the proof.**

1. *$A(n)=nc$ is not argued.* (3.85) gives $g_{11}(q)=A(n)/k_1$ for *every* $n$ for which $nq$ is an integer vector; (3.86) writes it as $nc/k_1$ for an unexplained constant $c$. The constant is the same for all $n$ because the same rational $q$ is reached from many $n$ (for $q=(\tfrac13,\tfrac23)$ from $n=3,6,9,\dots$, for $(\tfrac12,\tfrac12)$ from $n=2,4,6,\dots$, and any $n,n'\ge2$ are linked through $nn'$), so $A(n)/n$ cannot depend on $n$. Easy to repair, but it is the heart of the argument and the text does not say it.
2. *$B(n)=0$ is justified only for tangent vectors.* The book works on $\mathbb R^n_+$ and puts $B(n)=0$ because tangent vectors of $S_{n-1}$ have components summing to zero, but (3.84)–(3.85) use $e_1^m\in\mathbb R^m_+$, which is **not tangent** to the simplex. The step is correct if one uses tangent vectors such as $e_1-e_2$ (then $B$ drops out and the proof goes through for $S_n$), but on **positive measures** the metrics $c\,\mathrm{diag}(1/m_i)+B\,\mathbf 1\mathbf 1^{\mathsf T}$ are *all* invariant under the embeddings: with $B=0.7$ the defect is $4.4\times10^{-16}$ for arbitrary $Z$ (first bullet above), and for the split vector of the small case the squared-length ratio is $0.7529$ at $a=0$, $B=0.7$ but exactly $1$ at $a=1$ for every $B$. So uniqueness is a theorem about $S_n$; on $\mathbb R^n_+$ the proof does not give it, and the metric $\mathrm{diag}(1/m)$ of Theorem 3.4 comes from $f$-divergences, not from invariance alone (the constant $B$ may even depend on the total mass, which embeddings preserve).

## 6. $f$-divergence in the manifold of positive measures (§3.6)

### 6.1 Standard $f$ is needed; the sign in (3.95)

**In plain words.** A positive measure is a distribution whose total mass is free ($m=(m_0,\dots,m_n)$, $m_i>0$): a histogram of counts, an image. The same classification works (the proof of Theorem 3.1 goes through unchanged; the book refers to Theorem 4.1 for this, which must be a misprint for Theorem 3.1, since Theorem 4.1 is about flat divergences), but now the choice of $f$ matters, because (3.27) fails: the linear term no longer cancels. **A divergence must be non-negative, so $f$ must be standard.** Example: $m=(1)$, $n=(0.5)$: $f=u\log u$ gives $\sum n\log(n/m)=-0.3466<0$, the standard $u\log u-(u-1)$ gives $+0.1534$. Over 20000 random pairs of positive measures the smallest value is $-6.8621$ for $u\log u$, $-8.5554$ for $-\log u$, $-57.4114$ for the book's (3.38) at $\alpha=0.5$, and $+0.0052$ for the standard functions.

**(3.95) has the wrong sign.** The standard $\alpha$ function is $\frac4{1-\alpha^2}(1-u^{(1+\alpha)/2})+\frac2{1-\alpha}(u-1)$ ($f'(1)=0$); the printed (3.95) has $-\frac2{1-\alpha}(u-1)$, which has slope $-\frac4{1-\alpha}$ at $u=1$: $-1.0000,\,-2.6667,\,-4.0000,\,-8.0000,\,+2.0000$ at $\alpha=-3,-0.5,0,0.5,3$ (the corrected one: $0$ to $10^{-10}$). The printed function is not standard, its divergence is negative in $1207$ of 3000 random cases and differs from (3.96) by a relative $29.6$; the corrected one agrees with (3.96) to $8.4\times10^{-15}$, with the lines for $\alpha=\pm1$ printed in the same equation (they are standard), and with the exact duality $D_\alpha[m:n]=D_{-\alpha}[n:m]$ ($-2.9\times10^{-15}$, $6.7\times10^{-16}$, $1.8\times10^{-15}$ at $\alpha=0.3,2.5,-0.7$ through the functions). (3.96) itself is right: the $\alpha\to1$ limit gives $1.462931,\,1.460061\to1.460032$ at $\alpha=0.99,0.9999$.
The right panel of the $\alpha$-function figure above shows the three versions at $\alpha=0$, and the second widget lets you change the mass of $n$ and watch only the standard one stay non-negative.

### 6.2 Theorem 3.4 and the sphere

**In plain words.** For a standard $f$, $D_f[m:m+dm]=\tfrac12\sum dm_i^2/m_i$ (3.91), so the metric of $\mathbb R^n_+$ is $\mathrm{diag}(1/m_i)$. In the chart $\xi_i=2\sqrt{m_i}$ we have $d\xi_i=dm_i/\sqrt{m_i}$, so $ds^2=\sum dm_i^2/m_i=\sum d\xi_i^2$: the metric becomes the ordinary Euclidean one (3.92)–(3.93). Normalisation $\sum m_i=1$ becomes $\sum\xi_i^2=4$: **the simplex $S_n$ is the positive part of a sphere of radius 2** (3.94), curved even though the ambient space is flat.
*Checks.* $D_f[m:m+\varepsilon d]/\varepsilon^2\to\tfrac12\sum d^2/m=0.111538$ for $m=(1.3,0.7,2.1)$, $d=(.4,-.2,.3)$, for $\alpha=-1,\dots,3$ (e.g. $0.110659,\,0.111447,\,0.111529$ at $\varepsilon=0.1,0.01,0.001$ for KL; exactly $0.111538$ for $\chi^2$); the Jacobian of $m\mapsto2\sqrt m$ pulls $\mathrm{diag}(1/m)$ back to the identity, and $\sum\xi^2=16.4=4\times4.1$. On $S_2$ I computed the Gauss curvature of the Fisher metric *in the $p$ chart* (Christoffel symbols and Riemann tensor by finite differences, the routine of Chapter 5): $0.250000$ at three points, a sphere of radius 2 has $1/4$; $\sqrt{\det G}=1/\sqrt{p_0p_1p_2}$ to $5.3\times10^{-15}$, and with $p=(s_1^2,s_2^2,1-s_1^2-s_2^2)$ the volume element becomes $4\sin t\,dt\,d\phi$, total volume $6.2831853072=2\pi=4\pi\cdot2^2/8$, an octant, matching the Dirichlet integral $\Gamma(\tfrac12)^3/\Gamma(\tfrac32)$.

*Distances.* The Fisher–Rao distance between $p$ and $q$ is the great-circle arc, $s=2\arccos\sum\sqrt{p_iq_i}$. For the figure pair $s=1.856867$; minimising the discrete energy of a polyline in the $p$ chart with Newton's method gives $1.835843,\,1.851448,\,1.855503$ with $8,16,32$ segments (errors shrinking like $1/N^2$). The m-geodesic (straight in $p$) is $1.904921$ long ($2.59\%$ longer) and the e-geodesic $1.985012$ ($6.90\%$ longer): of the three "straight lines" only the Fisher–Rao geodesic is shortest. The chord through the ball is $1.790890$ and **the Hellinger divergence is half its square**: $D_0[p:q]=1.60364391=\text{chord}^2/2=4(1-\cos(s/2))$. That is also why $D_0$ is bounded by $4$, the bound (3.47), reached at the largest distance $s=\pi$ (disjoint supports).

*Scale.* Write $m=Mp$ with $M$ the total mass. Then $\sum dm_i^2/m_i=dM^2/M+M\sum dp_i^2/p_i$ (checked on random vectors: $2.534729183271$ on both sides), so the geometry of $\mathbb R^n_+$ is a cone over the simplex: the Fisher metric of the normalised distribution is multiplied by $M$, the radius of the sphere grows like $2\sqrt M$, distances between $Mp$ and $Mq$ like $\sqrt M$, and $D_f[Mp:Mq]=M\,D_f[p:q]$ (ratio $3.700000000000$ for $M=3.7$, $\alpha=0.5$). The eighth widget turns the octant and changes $M$. On $\mathbb R^3_+$ the Bregman picture is richer than on $S_2$ (Theorem 4.2): the mixed-derivative test gives $3.5\times10^{-9}$ for KL in the $m$ chart, $4.7\times10^{-1}$ for $\alpha=0.5$ in the $m$ chart but $2.4\times10^{-9}$ in the chart $m^{(1-\alpha)/2}=m^{1/4}$, and $8.3\times10^{-9}$ for Hellinger in the chart $\sqrt m$ (I ran the test; the theorem is Chapter 4's).

<img src="figures/sphere.svg" alt="Left: the octant of a sphere of radius 2 that is the simplex of three outcomes in the coordinates 2 times the square root of p, with the great circle through p equal to (0.6, 0.38, 0.02) and q equal to (0.02, 0.38, 0.6) (Fisher-Rao length 1.857), the m-geodesic (1.905), the e-geodesic (1.985) and the chord (1.791). Right: the Hellinger divergence 4(1 minus cos(s/2)) against the Fisher-Rao distance s, starting like s squared over 2 and saturating at 4 for disjoint distributions at s equal to pi.">

## Remarks at the end of the chapter

The closing pages are a history (Rao 1945 and the metric; Jeffreys' prior as the Riemannian volume; Hotelling's unpublished 1929 paper; Chentsov's invariance and his $\alpha$-connections; Efron's statistical curvature and Dawid's remark; the dually flat theory, and the fate of its first submission) and a paragraph on function spaces (Pistone and Sempi's Orlicz charts, Newton's Hilbert chart $\Phi[p]=p+\log p$ with $\mathbb E[p^2]$ and $\mathbb E[\log^2p]$ finite, Fukumizu's kernel exponential family). I only restate the substance and have checked none of the history or the function-space claims. One mathematical sentence is checkable: Hotelling's remark that a location-scale model has constant negative curvature.

**Check.** For a location-scale family $\frac1sq(\frac{x-m}s)$ the Fisher matrix is $\left[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right]/s^2$ with $a=\mathbb E[q'/q]^2$, $b=\mathbb E[\tfrac{q'}q(1+z\tfrac{q'}q)]$, $c=\mathbb E[(1+z\tfrac{q'}q)^2]$ (all in the standardised variable $z$), and the metric $(a\,dm^2+2b\,dm\,ds+c\,ds^2)/s^2$ is a hyperbolic plane of curvature $-a/(ac-b^2)$ (shear $m$ by $\tfrac bas$ to remove $b$). The integrals by Gauss–Legendre, the curvature by finite differences at three points $(m,s)$ for each family:

| family | $a$ | $b$ | $c$ | Gauss curvature at three points | $-a/(ac-b^2)$ |
|---|---|---|---|---|---|
| Gaussian | 1.00000 | 0 | 2.00000 | $-0.50000$ each | $-0.50000$ |
| Laplace | 1.00000 | 0 | 1.00000 | $-1.00000$ each | $-1.00000$ |
| logistic | 0.33333 | 0 | 1.42996 | $-0.69932$ each | $-0.69932$ |
| Cauchy | 0.50000 | 0 | 0.50000 | $-2.00000$ each | $-2.00000$ |
| Student $t$, 3 d.f. | 0.66667 | 0 | 1.00000 | $-1.00000$ each | $-1.00000$ |
| Gumbel (skew) | 1.00000 | 0.42278 | 1.82368 | $-0.60793$ each | $-0.60793=-6/\pi^2$ |

So the remark holds, with a value that depends on the family (the Gaussian's $-\tfrac12$ is Chapter 5's number).

## Checks of the book's statements

| Where | Statement | What I found |
|---|---|---|
| (3.4), (3.11) | $f$-divergences are information-monotone | verified: nine $f$, 20000 random merges, $\max\bar D/D=1.000000$; Markov kernels $\le0.7572$ |
| (3.12)–(3.13) | equality iff $y$ is sufficient | verified for equal conditionals ($\le2\times10^{-12}$); **read for the pair or the model, not $S_n$**: the sum of two Bernoulli trials keeps $1.165371$, on $S_3$ the median loss is $0.0916$ |
| Thm 3.1, (3.16)–(3.19) | an $f$-divergence is invariant | monotone: yes. **"Equality only for sufficient" needs strictly convex $f$**: variation distance (no loss in $0.1877$ of merges) and a $C^1$ Huber-type $f$ ($0.556250$ before and after) |
| (3.14)–(3.15), p. 54 | convex $f$, $f(1)=0$ gives a divergence | **needs $f''(1)>0$**: $f=0$, $(u-1)^4$ give $D/\varepsilon^2\to0$ |
| Thm 3.1, (3.20)–(3.24) | decomposable and invariant means $f$-divergence | consistent: additivity residual $+0.2878,\,0,\,-0.0998$ for $a=0.5,1,1.5$; Euclid fails ($0.326$), Itakura–Saito not invariant ($0.2001$), non-convex $f$ fails ($0.241$); needs regularity; $n=1$ vacuous ($4.7619$ vs $4$) |
| (3.25) | $D+D^2$ invariant, not decomposable | verified (mixed derivative $0.2026$ vs $0$) |
| (3.26)–(3.27) | $f+c(u-1)$ gives the same divergence | true on $S_n$ ($0.0$); **false on $\mathbb R^n_+$** ($+0.910000=c(\sum n-\sum m)$) |
| (3.31)–(3.32) | dual $f^*$ gives $D_f[q:p]$, standard | verified ($8.5\times10^{-14}$) |
| (3.36) | dual KL is the Bregman divergence of $\psi$ | verified ($1.0\times10^{-15}$) |
| (3.37), (3.38)–(3.41) | $\chi^2$, $\alpha$-family, Hellinger | verified ($\le2.3\times10^{-13}$); $\chi^2$ is $\alpha=3$; Hellinger is half the squared chord |
| (3.40), (3.42) | dual of $\alpha$ is $-\alpha$; limit $\alpha\to\pm1$ | true for the divergences; **only up to the linear term (3.26) for the functions**: differ by $16.000$ at $u=4$; $f_\alpha(2)=-19.62,\,-199.61,\,-1999.61$ |
| (3.44)–(3.45) | variation distance | **factor 2**: $f=\lvert1-u\rvert$ gives $\sum\lvert p-q\rvert=0.2000$, printed $\tfrac12\sum=0.1000$ |
| (3.46) | squared Euclid is not invariant | verified: $0.24\to0.32$ with equal conditionals |
| p. 59 (1), (3.47) | joint convexity; upper bound $f(0)+f^*(0)$ | verified (margin $8.8\times10^{-5}$); $4,\,5.3333,\,0.6931,\,2$ reached at disjoint supports |
| **(3.48)** | $D_f[p:q]\le\sum(p_i-q_i)f'(p_i/q_i)$ | **false as printed** ($\chi^2$: $0.8889>0.6400$; violated in $0.263$ of random pairs); the version with $p,q$ exchanged holds ($1.7778$) |
| (3.49)–(3.50) | infinite $\alpha$-divergences | verified (table) |
| (3.52)–(3.53) | zero forcing, zero avoiding | verified at $\alpha\ge1$, $\le-1$; **the switch is earlier**: $\alpha=0$ ($3\times3$ table), between $0$ and $0.5$ (smooth) |
| (3.55) | Sanov's lemma | verified: exponent $0.276608\to0.274887$, prefactor $0.999844$ at $N=5000$ |
| **(3.56)** | $\varepsilon=\frac1{\sqrt N}(\hat p-p)$ | **should be $\sqrt N(\hat p-p)$**: $N\,\mathrm{KL}=0.72752$ vs $0.72667$; printed $\varepsilon$ gives $4.5\times10^{-6}$ |
| (3.57) | covariance $g_{ij}/N$ | verified exactly ($3.6\times10^{-16}$), $g_{ij}=\mathrm{diag}(p)-pp^{\mathsf T}$; the $p$-chart Fisher matrix is its inverse |
| (3.58)–(3.59) | large deviations, e-projection | verified, convex $A$: $0.12313394$ by tilt and Cramér, Bahadur–Rao ratio $0.998078$ at $N=1600$, Fisher-orthogonal; non-convex $A$ not tested |
| **(3.66)** | $\tfrac12\{D+D^*\}=\int g_e=\int g_m$ | **factor 2**: both integrals equal $D+D^*$ ($3.945389$ vs half $1.972694$); true for densities too |
| (3.67)–(3.68) | the metric of any standard $f$-divergence is Fisher | verified; $f''(1)$ in general ($\tfrac14$ for Jensen–Shannon); colon typo in (3.67) |
| (3.70)–(3.73) | coarse graining and Markov embeddings | embeddings preserve every $f$-divergence ($5.7\times10^{-14}$); (3.73) $f\circ h=\mathrm{Id}$ needs $r$ supported on the cells (my reading) |
| Thm 3.3, (3.74)–(3.87) | Fisher is the unique invariant metric | Fisher invariant to $4.4\times10^{-16}$, competitors fail ($0.739$, $0.491$, $0.988$, $3.00$); proof's numbers $6.0000,\,4.0000,\,1.7143$; **$A(n)=nc$ unargued; $B(n)=0$ only on tangent vectors; not unique on $\mathbb R^n_+$** |
| (3.88) | uniqueness of the cubic tensor | invariance verified ($1.7\times10^{-15}$), competitors fail; uniqueness not checked |
| (3.89)–(3.90) | positive measures; standard $f$ needed | verified: $-0.3466$ with $u\log u$; standard $+0.0052$ minimum |
| (3.91)–(3.94) | metric $\mathrm{diag}(1/m)$; $S_n$ is a sphere of radius 2 | verified: $K=0.250000$, volume $2\pi$, $s=1.856867$, Hellinger $=$ chord$^2/2$ |
| **(3.95)** | standard $\alpha$ function | **sign of the linear term is wrong** (slope $-4/(1-\alpha)$, negative divergence in $1207$ of 3000); (3.96) is right ($8.4\times10^{-15}$) |
| §3.6, first lines | "the proof of Theorem 4.1" | should be Theorem 3.1 (Theorem 4.1 is about flat divergences) |
| closing remark | location-scale models have constant negative curvature | verified for six families ($-\tfrac12$ Gaussian, $-6/\pi^2$ Gumbel) |

## Questions and doubts

- **What is "sufficient" in the invariance criterion?** On the full simplex $S_n$ only bijections are sufficient, so "equality iff sufficient" can only be meant for the pair of distributions being compared (conditionals agree, (3.13)) or for a model inside $S_n$. The book moves between the two readings. Settling it would mean stating the criterion for a model $M\subset S_n$ and for pairs, and seeing which of Theorems 3.1 and 3.3 use which.
- **Hypotheses of Theorem 3.1.** Strict convexity (so that equality forces equal conditionals), $f''(1)>0$ (so that there is a metric), regularity of $d$ in the converse (Cauchy's equation), and the domain $p_1+p_2\le1$. I added the first two with explicit counterexamples; I did not examine how little regularity the converse needs, nor the classification the book cites for $n=1$ (Jiao et al.), which I have not read.
- **Chentsov on positive measures.** The proof sets $B(n)=0$ on non-tangent vectors and the family $c\,\mathrm{diag}(1/m)+B\mathbf 1\mathbf 1^{\mathsf T}$ is invariant on $\mathbb R^n_+$, so as written the proof yields uniqueness only on $S_n$; and the constant $c$ could in principle depend on the total mass. What the extension to $\mathbb R^n_+$ needs (extra hypotheses such as scale covariance?) I did not work out, and I did not consult the literature.
- **Uniqueness of the cubic tensor (3.88).** The book says the same argument works; I verified only that $T=\sum Z^3/p^2$ is invariant and that sibling powers are not. The uniqueness of the $\alpha$-connections in Part II rests on it, so a proof along (3.74)–(3.87) with $T$ in place of $g$ would be worth writing out (what plays the role of $A(n)\delta_{ij}+B(n)$ for a symmetric 3-tensor at the uniform point?).
- **The location of the zero-forcing transition.** (3.52)–(3.53) are stated for $\alpha\ge1$ and $\le-1$ because the divergence is then infinite on the wrong support. In both of my examples the best approximation already switches from covering to hiding in $[0,0.5]$, with finite divergences. Is there a general statement (for exponential families $S$ and $p$ with several modes) about where, and does it depend on $p$? I have only examples.
- **Large deviations for non-convex $A$.** The theorem is stated for a closed region $A$, with the minimiser obtained by e-projecting $p$ onto its boundary; for a non-convex $A$ the minimiser of $\mathrm{KL}[q\Vert p]$ is the best of several local projections and need not be unique. I tested a half-plane only.
- **Continuous variables and the function-space remarks.** (3.3) obtains the induced density by integration and (3.67) integrates over $x$; the measure-theoretic statements (what $p(x\mid y)$ means for non-discrete $y$, when KL is finite, what $\Phi[p]=p+\log p$ in (3.98)–(3.99) buys) are asserted, and I checked only Gaussian and discrete cases. Whether a Hilbert-space chart really makes the two flat structures and (3.4) available simultaneously is, as the closing paragraph says, open.
- **Two small notational points.** The letter $f$ is a convex function in §3.2–3.4 and a coarse-graining map in (3.70), (3.73); and $n$ counts outcomes in $S_n$ but names a positive measure in (3.89).

## Takeaways

- **Invariance singles out $f$-divergences.** Information may only be lost by merging outcomes ($\bar D\le D$) and not at all by a sufficient merge; for sums over outcomes this forces $D=\sum p\,f(q/p)$ with $f$ convex, and Jensen's inequality does the rest. All nine $f$-divergences I tried are monotone in 20000 random merges; the squared Euclidean distance is not ($0.326$ of the merges raise it), nor is a non-convex $f$ ($0.241$), and a cost of the wrong degree $p^af(q/p)$ fails at $a=0.5$ and $a=1.5$.
- **Read the hypotheses.** "Invariant" needs $f$ *strictly* convex (variation distance, Huber-type $f$ keep $D$ equal for non-sufficient merges), "divergence" needs $f''(1)>0$, and "equality iff sufficient" is about the pair or the model, never about $S_n$ itself.
- **Locally all $f$-divergences are the Fisher metric**, scaled by $f''(1)$ ($1$ for the standard ones, $\tfrac14$ for Jensen–Shannon); Chentsov's theorem says the Fisher metric is also the only invariant one on $S_n$. The proof is a clean uniform-splitting argument with two unstated steps; on positive measures an extra term $B(\sum Z)^2$ is invariant too.
- **KL is special.** It is the only one of these that is Bregman in the mixture chart (its dual in the exponential chart), the exponent of Sanov and of large deviations (rare events follow the e-projection; exact probabilities approach $e^{-NI}$ with the Bahadur–Rao prefactor), and $\mathrm{KL}[p\Vert q]+\mathrm{KL}[q\Vert p]$ is the integral of the Fisher information along **either** geodesic, with no factor $\tfrac12$.
- **The simplex is a sphere of radius 2** in the coordinates $2\sqrt p$: Gauss curvature $\tfrac14$, area $2\pi$ for three outcomes, Fisher–Rao distance $2\arccos\sum\sqrt{pq}$, Hellinger divergence half the squared chord; the e- and m-geodesics are longer ($6.90\%$, $2.59\%$ in my example). On positive measures it becomes a cone: radius $2\sqrt M$.
- **Printed slips to know about:** (3.48) (arguments exchanged), (3.66) (factor 2), (3.56) ($\sqrt N$ vs $1/\sqrt N$), (3.95) (sign of the linear term), (3.45) (factor 2), the cross-reference to "Theorem 4.1" in §3.6, and the colon in (3.67).

| Term | One line |
|---|---|
| coarse graining | merge outcomes into cells, $\bar p_a=\sum_{i\in X_a}p_i$; may only lower a divergence |
| sufficient | the detail inside the cells has the same conditional law for both distributions; no loss |
| $f$-divergence | $D_f[p:q]=\sum p_if(q_i/p_i)$, $f$ strictly convex, standard: $f(1)=f'(1)=0$, $f''(1)=1$ |
| dual | $f^*(u)=uf(1/u)$, $D_{f^*}[p:q]=D_f[q:p]$ |
| $\alpha$-family | $f_\alpha=\frac4{1-\alpha^2}(1-u^{(1+\alpha)/2})+\frac2{1-\alpha}(u-1)$ (standard); $\alpha=-1$ KL, $0$ Hellinger, $3$ $\chi^2$ |
| local form | $D_f[p:p+dp]=\tfrac12f''(1)\sum dp_i^2/p_i$: Fisher |
| zero forcing / avoiding | $\alpha\ge1$: $\hat q=0$ where $p=0$; $\alpha\le-1$: $\hat q>0$ where $p>0$ (transition earlier in my examples) |
| Sanov, large deviation | $P(\hat p)\approx e^{-N\mathrm{KL}[\hat p\Vert p]}$; $P(\hat p\in A)\approx e^{-N\min_A\mathrm{KL}[q\Vert p]}$, minimiser = e-projection = exponential tilt |
| Theorem 3.2 | $\int_0^1g_e=\int_0^1g_m=\mathrm{KL}[p\Vert q]+\mathrm{KL}[q\Vert p]$ |
| Markov embedding | split outcomes in fixed proportions; invariant metrics keep lengths; only $\sum Z^2/p$ does |
| positive measures | metric $\mathrm{diag}(1/m)$, chart $\xi=2\sqrt m$ Euclidean, $S_n$ a sphere of radius 2, $D_f[Mp:Mq]=MD_f[p:q]$ |

---

*Notes written 2026-10-02.*
