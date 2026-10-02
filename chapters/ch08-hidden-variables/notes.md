---
title: "Chapter 8: estimation in the presence of hidden variables"
short_title: "Ch. 8 — Hidden variables"
chapter: 8
category: "Part III"
book_pages: "179–189"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [hidden-variables, em-algorithm, alternating-projection, data-manifold, missing-information, loss-of-information, data-reduction, misspecified-model, kl-projection, boltzmann-machine, neural-field]
status: read
---

## Links

- **[Interactive companion](figures/interactive.html)**: seven widgets, most of them on the two-component Gaussian mixture. (1) EM one projection at a time on a 200-point sample: E-step and M-step as separate buttons, with the exact bookkeeping of the divergence. (2) One observation: the plane of the responsibility $a$ and the parameter $m$, the sigmoid E-step, the straight M-step, the staircase between them, and the stationary point at $m=0$.
  (3) The speed of EM against the separation of the components, and the sub-linear crawl on data from $N(0,1)$. (4) The identity (8.30) for the three-parameter mixture: $G_X$, $G_Y$, $G_{X\mid Y}$, their eigenvalues and the measured contraction. (5) Grouping and censoring of a Gaussian observation: the lost fractions, the small-bin law, covariance ellipses. (6) Misspecified models of $N(u,u^2)$: leaves, pseudo-true value, efficiency by geometry, sandwich and score correlation, and a simulation. (7) The decoder that ignores correlation: two neurons, and a ring of $n$ neurons.
- **[Runnable checks](https://github.com/msrepo/information_geometry_notes/tree/main/chapters/ch08-hidden-variables/code)**:
  `code/hidden_variables.py` prints every number on this page and regenerates the figures with `python3 code/hidden_variables.py --figures`; `make verify` runs it in about five seconds. Seeds are fixed, and where the exact law of a sufficient statistic is available it is sampled instead of the data.
- The book: Amari, *Information Geometry and Its Applications* (Springer, 2016), DOI [10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8). These notes cover Chapter 8 only; equation numbers such as (8.26) refer to the book.
  No text of the book is reproduced here; everything is restated and re-derived.
- Previous chapters written up: **[Chapter 7: asymptotic theory of inference](../ch07-asymptotic-theory-of-inference/index.html)** (§8.3 reuses its leaves, angles and efficiencies, with the leaves chosen by the wrong model), **[Chapter 1: dually flat structure](../ch01-dually-flat-structure/index.html)** (the alternating projections that §8.1 turns into EM), **[Chapter 2: exponential and mixture families](../ch02-exponential-and-mixture-families/index.html)** and **[Chapter 6: dual connections](../ch06-dual-connections/index.html)** (the mixed coordinates used in §2.3). Next: **[Chapter 9](../ch09-neyman-scott-and-semiparametrics/index.html)**; natural gradients, of which EM turns out to be one (§1.3 below), are in **[Chapter 12](../ch12-natural-gradient-and-singular-regions/index.html)**.
  Background on the annotations site: **[The Fisher information matrix](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/)** and **[EM and Gaussian mixtures](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/)**.

## In one paragraph

Some of the quantities behind a data set are never recorded: which of two sources produced a reading, which units of a network were on. Such a **hidden variable** $h$ makes estimation awkward. The model of everything, $p(y,h;\xi)$, is simple (usually an exponential family), but the model of what you actually see, $p_Y(y;\xi)=\sum_hp(y,h;\xi)$, is a mixture, and maximising its likelihood directly is hard.
The chapter's first idea is geometric. A data set that shows only $y$ is not a point of the big space $S$ of distributions of $(y,h)$; it is a whole m-flat set $D$ of points, one for each way of *guessing* $h$ (a table of guesses, the "responsibilities"). The maximum-likelihood problem becomes the problem of finding the closest pair of points between $D$ and the model $M\subset S$, and the alternating projection of Chapter 1 (the **em algorithm**) solves it: the e-projection onto $D$ is the E-step, the m-projection onto $M$ is the M-step. That is the whole EM algorithm, seen from the side.
The second idea concerns **data reduction**: if only a summary $T$ of the data is kept, the Fisher information splits exactly as $g^X=g^T+g^{X\mid T}$, and the lost part is a conditional information. The third concerns a **misspecified model**: the maximum likelihood estimator of a wrong model goes to the KL-projection of the truth onto it, along the same straight leaves as in Chapter 7, and the loss of information is again the angle between a leaf and the true model.
What I add, by checking: an exact formula for how much each EM half-step gains, a statement that the speed of EM *is* the fraction of information lost (so §8.1 and §8.2 are one subject), examples where the algorithm sits at a saddle, at a bad local maximum, or crawls, a few printed slips, and the exact condition under which the book's neural-field example loses nothing.

## The spine of the argument

1. A hidden-variable model has a big model $M=\{p(y,h;\xi)\}$ inside $S$ and a visible marginal $p_Y$. Observed $y$-data define the data manifold $D=\{\bar q_Y(y)\,q(h\mid y)\}$, which is m-flat (§8.1.1–8.1.2).
2. **Theorem 8.1**: the MLE minimises the divergence from $D$ to $M$. Behind it is a chain rule: the divergence of a table $R$ from $\xi$ is $-\ell(\xi)/N$ plus the mean divergence of the table's rows from the posteriors at $\xi$. **Lemma 8.1**: for fixed $\xi$ the best table is the posterior, the e-projection onto $D$.
3. **EM = em** (§8.1.3): the E-step is that e-projection, the M-step the m-projection onto $M$, which for an exponential family is moment matching, (8.26). **Theorem 8.2**: the divergence decreases; with $M$ e-flat both half-steps lower it by explicit KL divergences.
4. What is reached is a stationary point of the likelihood: a maximum, but possibly a saddle or a poor local maximum, and the speed is the fraction of missing information.
5. **Data reduction** (§8.2): the score of the whole data is the score of $T$ plus the score of the rest given $T$, orthogonal, so $g^X=g^T+g^{X\mid T}$ and the loss is $g^{X\mid T}$ (8.28)–(8.31). With hidden variables, $T=y$ and the loss is the missing information.
6. **Misspecified model** (§8.3): the q-MLE is the m-projection of the observed point onto $M_q$ along q-ancillary leaves; it is consistent iff $E_p[\partial\log q]=0$ (**Theorem 8.3**); the loss of information is $g_{a\kappa}g_{b\lambda}g^{\kappa\lambda}$, the angle between a leaf and $M$ (**Theorem 8.4**, which is (7.58) again).
7. The neural-field example shows how this looks for Gaussians: the decoder that ignores correlation loses nothing exactly when the tuning derivative is an eigenvector of the covariance.

## Setup and notation

| Symbol | Meaning | In the running examples |
|---|---|---|
| $x=(y,h)$ | complete data; $y$ observed, $h$ hidden | $y\in\mathbb R$, $h\in\{1,2\}$ the component |
| $p(y,h;\xi)$, $M$ | the complete model and its manifold; an exponential family in my examples, hence e-flat | $w_h\,N(y;\mu_h,1)$ |
| $p_Y(y;\xi)=\sum_hp(y,h;\xi)$ | the visible model (8.1) | the mixture $wN(\mu_1,1)+(1-w)N(\mu_2,1)$ |
| $S$ | all distributions of $(y,h)$ (8.2) | |
| $\bar q_Y$ | empirical distribution of the observed $y_i$: a spike of weight $1/N$ at each value | |
| $D$ | data manifold (8.6): all $\bar q_Y(y)q(h\mid y)$; m-flat. For a sample it is the set of $N\times k$ tables $R$ of responsibilities, rows adding to 1 | 200 points: a product of 200 intervals |
| $F(R,\xi)$ | divergence of the point $R$ of $D$ from the point $\xi$ of $M$, with the constant $c$ of (8.15) dropped: $\tfrac1N\sum_i\sum_kR_{ik}\log\frac{R_{ik}}{w_k\,N(y_i;\mu_k,1)}$ | |
| $\ell(\xi)$ | log-likelihood $\sum_i\log p_Y(y_i;\xi)$ | |
| $g^X,\ g^T,\ g^{X\mid T}$ | Fisher information of the whole data, of a statistic $T$, and conditional (8.28)–(8.30); with $T=y$ in the mixture the script and the page write $G_X,G_Y,G_{X\mid Y}$ | |
| $M_q=\{q(x;v)\}$ | the misspecified model; $A_q(v)$ its q-ancillary family, the m-flat leaf through $q(v)$ orthogonal to $M_q$ | |
| $v^*(u)$, $f$ | pseudo-true value (limit of the q-MLE when the truth is $p(\cdot;u)$); $f(v)=u$ where the leaf $A_q(v)$ meets $M$ | $v^*=\kappa u$, $f(v)=v/\kappa$ |

The roles of $M$ and $S$ are swapped relative to Chapter 7, as in the book: here $S$ is the big family and $M$ the model (Chapter 7 called the curved model $S$ and the ambient family $M$). Index conventions as in the previous chapters: natural parameters $\theta$ (upper index), expectation parameters $\eta$ (lower index), $\mathrm{KL}[p\Vert q]$ with the *first* argument the truth. Both projections below use $\mathrm{KL}[q\Vert p]$ for $q\in D$, $p\in M$: the **e-projection** of a model point $p$ onto $D$ varies $q$ and keeps $p$ fixed; the **m-projection** of a point $q\in D$ onto $M$ varies $p$ and keeps $q$ fixed. This is the pairing I found in Chapter 1 to be the right one, and the chapter uses it consistently.

**Running examples.** (a) A mixture of two unit-variance Gaussians, $\xi=(w,\mu_1,\mu_2)$: a sample of $N=200$ points from $0.35\,N(-1,1)+0.65\,N(1.5,1)$ (sample mean $0.6497$, variance $2.6667$), and the whole density by quadrature. (b) Its one-parameter symmetric version $\tfrac12N(-m,1)+\tfrac12N(m,1)$. (c) A restricted Boltzmann machine with 4 visible and 2 hidden binary units. (d) Three binary neurons whose third-order correlation is discarded. (e) The curved family $N(u,u^2)$ of Chapter 7 with several wrong models $q(x;v)$. (f) Gaussian neurons with correlated truth $N(r(u),V)$ and model $N(r(u),I)$.

## 1. The EM algorithm (§8.1)

### 1.1 Hidden variables, the data manifold and the model manifold (§8.1.1–8.1.2)

**In plain words.** Take the 200 numbers of example (a). If someone told you, for every point, which of the two Gaussians produced it, estimating $w,\mu_1,\mu_2$ would be trivial: count and average. You are not told. A *guess* for point $i$ is a pair $(r_{i1},r_{i2})$ of non-negative numbers adding to 1: how responsible each component is for it. A whole data set of guesses is a table $R$ with 200 rows. Every such table is equally legitimate as a stand-in for the missing labels, and the set of all of them is the data manifold $D$ of the book (8.5)–(8.7).
It is the set of all joint distributions on $(y,h)$ whose $y$-marginal is the empirical distribution (a spike of weight $1/N$ at each observed value) and whose conditionals $q(h\mid y_i)$ are arbitrary. The model manifold $M$ is the set of joint distributions that the mixture can produce. To estimate, find the closest pair of points between the two sets: start at a point of $M$, go to the nearest point of $D$, then to the nearest point of $M$ from there, and so on. "Nearest" is measured by the KL divergence, and because it is not symmetric the two moves are different projections.

**Formal statement.** $D$ is *m-flat* (8.7): mixing two tables row by row, $q=\alpha q_1+(1-\alpha)q_2$, gives a table, and the $y$-marginal stays the empirical one. I checked this on a toy ($y\in\{0,1,2\}$, $h\in\{0,1\}$, $\bar q_Y=(0.5,0.3,0.2)$): the mixture $0.3q_1+0.7q_2$ of two points of $D$ has marginal $(0.5,0.3,0.2)$ to $2.8\times10^{-17}$. $D$ is *not* e-flat: the normalised geometric mean of the same two points has marginal $(0.4456,0.3762,0.1782)$, a deviation of $0.0762$, so the e-geodesic leaves $D$ at once. For a sample of $N$ points with $k$ components, $D$ is the product of $N$ simplices, a convex set; $M$ is, for the mixture, the full exponential family with statistics $(\mathbf 1(h{=}1),\,y\,\mathbf 1(h{=}1),\,y\,\mathbf 1(h{=}2))$, which is e-flat in $S$.

One gloss. For continuous $y$ the empirical distribution is a sum of spikes and the divergence (8.14) from it to a density is infinite; the constant $c$ of (8.15) is that infinity. Everything below is about differences (or about $F$ with the constant dropped), which are finite, and the book's statements are right in that form.

### 1.2 Theorem 8.1 and Lemma 8.1

**In plain words.** The claim is that the closest pair of points between $D$ and $M$ sits at the maximum likelihood estimate. The reason is a bookkeeping identity that the book does not state. The divergence of a table $R$ from a model point $\xi$ splits into two non-negative pieces: *how badly the model explains the visible data* ($-\ell(\xi)/N$) plus *how far the table's guesses are from the guesses the model itself would make* (the mean KL divergence of the rows of $R$ from the posteriors $p(h\mid y_i;\xi)$). For a fixed $\xi$ the second piece vanishes when the table is the posterior, and then what is left is minus the log-likelihood.

**Formal statement.**
$$F(R,\xi)=-\frac{\ell(\xi)}N+\frac1N\sum_i\mathrm{KL}\big[R_i\Vert p(h\mid y_i;\xi)\big].$$
It is the chain rule for KL, and it is (8.15) with the constant absorbed. Checked for 300 random pairs $(R,\xi)$: it holds to $3.6\times10^{-15}$. So $\min_{R\in D}F(R,\xi)=-\ell(\xi)/N$, and minimising over $\xi$ as well is maximum likelihood (Theorem 8.1). The minimiser over $D$ is the posterior: this is **Lemma 8.1** and (8.21), which the book obtains by a Lagrange multiplier (8.20). I confirmed it with no formula: golden-section search on each of the 200 rows from the point $\xi_0=(w{=}0.5,\ \mu{=}-0.5,\,0.5)$ lands on the posterior to $1.5\times10^{-8}$ (the limit of a search on function values).
The geometry is the Pythagorean theorem of Chapter 1: for *every* table $R\in D$, $F(R,\xi)=F(R^*,\xi)+\frac1N\sum_i\mathrm{KL}[R_i\Vert R^*_i]$ with $R^*$ the posterior; checked for 2000 random $R$, to $1.1\times10^{-15}$. The e-geodesic from the model point to $R^*$ has tangent $\log(\bar q_Y/p_Y)$, a function of $y$ alone, and its pairing with every tangent vector of $D$ (which only moves mass between values of $h$ at a fixed $y$) is $5.6\times10^{-17}$: the projection is orthogonal to $D$, as the word "e-projection" promises.

**Two slips in the derivation (page image of p. 183).** The proof differentiates $\int q(h\mid y)\log p(y,h;\xi)\,dh$ and prints (8.17) with the factor $q/p(h\mid y;\xi)$ in front of $\partial_\xi p(y,h;\xi)$; the derivative of the logarithm carries one more factor $1/p_Y(y;\xi)$. For one observed $y$ that is a harmless constant, and the text does say it works with one $y$, but the conclusion (8.22) is then used for the MLE of a sample, where the factor depends on $i$ and cannot be dropped. At the MLE of the 200-point sample, $\sum_i\partial\log p_Y(y_i)=0$ (largest component $2.9\times10^{-13}$) whereas the printed form summed over the sample, $\sum_i\partial p_Y(y_i)$, equals $(-9.9697,\,3.7851,\,-4.6611)$: not zero, and not an estimating equation at all (it is the gradient of $\sum_ip_Y(y_i)$, which is linear in $w$). Also, (8.23) writes $p(y,R,\xi_0)$ where $p(y,h,\xi_0)$ is meant.

### 1.3 EM is em (§8.1.3–8.1.4)

**In plain words.** *E-step*: freeze the model at $\xi_t$ and take the table of responsibilities that the model itself implies (the posteriors). *M-step*: freeze the table and refit the model to it, as if the guessed labels were real, weighting each point by its responsibility. For the mixture that gives (8.26): the new weight of a component is the average responsibility, its new mean the responsibility-weighted average of the data. The book obtains (8.26) by differentiating the conditional expectation. There is a better reason: $M$ is an exponential family, and the m-projection of any distribution onto an exponential family is found by **matching expectation parameters** ($\eta$); the expectation parameters of the complete model are $(w_1,\,w_1\mu_1,\,w_2\mu_2)$, which are exactly the averages in (8.26).

**Numbers.** After one step from $\xi_0$, $\eta(\xi_1)=(0.388804,\,-0.198106,\,0.847796)$ and the table's expectation of the statistics is the same, to $2.2\times10^{-16}$. The same projection by damped Newton on $F(R,\cdot)$ with numerical derivatives and without (8.26) gives $(w_1,\mu_1,\mu_2)=(0.38880398,\,-0.50952586,\,1.38710928)$, the closed form to $9.1\times10^{-11}$. Because $M$ is e-flat the projection obeys the exact Pythagorean relation $F(R,\xi')-F(R,\xi_1)=\mathrm{KL}[p_{\xi_1}\Vert p_{\xi'}]$ (complete-data KL) for every $\xi'$, checked for 1000 random $\xi'$ to $5.3\times10^{-15}$: the M-step is unique and global, and the book's remark that the m-projection is unique when $M$ is e-flat is right.

**Theorem 8.2 with its proof made quantitative.** "The divergence decreases" is easy; here is how much. The M-step lowers $F$ by exactly $\mathrm{KL}[p_{\xi_{t+1}}\Vert p_{\xi_t}]$ (the KL between the two complete-data distributions, the Pythagorean relation above); the next E-step lowers it by exactly $\frac1N\sum_i\mathrm{KL}[p(h\mid y_i;\xi_t)\Vert p(h\mid y_i;\xi_{t+1})]$, the posterior divergence (the chain rule). Their sum is the gain in log-likelihood per observation, an *identity*, true to $4.3\times10^{-16}$ at every one of the first 40 steps:

| step | $\ell$ before | gain per observation | M part | E part |
|---|---|---|---|---|
| 1 | −455.530334 | 0.32926 | 0.26545 | 0.063809 |
| 2 | −389.678638 | 0.035888 | 0.029010 | 0.0068780 |
| 3 | −382.500990 | 0.0016293 | 0.0013854 | 0.00024395 |
| 4 | −382.175125 | $9.1235\times10^{-5}$ | $6.1334\times10^{-5}$ | $2.9902\times10^{-5}$ |
| 10 | −382.133623 | $7.7540\times10^{-6}$ | $4.1976\times10^{-6}$ | $3.5563\times10^{-6}$ |
| 20 | −382.128336 | $2.7609\times10^{-7}$ | $1.4958\times10^{-7}$ | $1.2650\times10^{-7}$ |
| 40 | −382.128142 | $3.3496\times10^{-10}$ | $1.8151\times10^{-10}$ | $1.5344\times10^{-10}$ |

So the likelihood cannot stop rising unless $\xi_{t+1}=\xi_t$ (both parts vanish together). The explicit accounting needs $M$ e-flat. For a curved complete-data model (weights fixed at $\tfrac12$, means $(0,m)$) the M-step gain is $(\text{mean responsibility})(m'-m)^2/2$ and not $\mathrm{KL}[p_{m'}\Vert p_m]=(m'-m)^2/4$: the ratio is $1.0604,\,0.9769,\,0.9294,\,0.9095$ over the first four steps, above and below 1, so not even a one-sided inequality. What survives is the split of the gain into the M-step's drop of $F$ plus the E part (to $3.9\times10^{-16}$), which is all that monotonicity needs.
The interactive page's first widget lets you press the two half-steps and read these parts.

**EM is a natural-gradient step.** In the $\eta$ chart of the complete-data family the update is $\eta_{t+1}-\eta_t=\tfrac1N\,\partial\ell/\partial\theta$ (Fisher's identity: the log-likelihood gradient in $\theta$ is the posterior expectation of the statistics minus their model expectation). Checked by finite differences of the likelihood in $\theta$ for 8 steps: largest difference $2.9\times10^{-10}$. Since $d\eta=G_Xd\theta$, this is a natural-gradient step of unit length for the complete-data Fisher metric, taken along the m-geodesic (a straight line in $\eta$). It is why EM needs no step size, and why it is slow exactly where the visible data are less informative than the complete data (§1.5).

<img src="figures/em-staircase.svg" alt="Left: for one observation y = 1.8 of the symmetric two-component mixture, contours of the divergence F(a, m) between a point a of the data manifold D and a point m of the model manifold M, the sigmoid curve of the E-step, the straight line of the M-step and the staircase of the em algorithm descending to the minimum near m = 1.79; the point m = 0 is a stationary point that is not a minimum. Right: the EM map of the symmetric mixture for true separations m = 0.5 and 1.5 with cobwebs from m = 2.6; the slope at the fixed point is 0.657 and 0.084.">

*The simplest picture* (left): one observation $y=1.8$ and the symmetric model. $D$ is an interval (the responsibility $a$) and $M$ a line ($m$). The M-step is the straight line $m=(2a-1)y$, the E-step the sigmoid $a=\sigma(2my)$, and EM from $m=0.3$ gives $m_t=0.88738,\,1.65826,\,1.79083,\,1.79430,\,1.79437$, converging to $m^*=1.794374$ with slope $0.02022$. The crossing at $m=0$ is also a fixed point; its slope is $y^2=3.2400>1$, so EM leaves it from any nearby start, and for $\lvert y\rvert<1$ it would be the only one.

**The Boltzmann machine (the book's second example).** With four visible and two hidden binary units and no biases: reading (8.10) with $x=(y,h)$ and no connections inside a layer, $\tfrac12x^{\mathsf T}Wx=y^{\mathsf T}Ah$ exactly (largest difference over the 64 states $4.4\times10^{-16}$), $A$ being the visible–hidden block of the symmetric $W$. So (8.11), which prints $\tfrac12y^{\mathsf T}Wh$, agrees with (8.10) only if its $W$ means twice the block; the exponent is $-y^{\mathsf T}Ah$. The conditional (8.13) factorises over the hidden units, $p(h_j{=}1\mid y)=\sigma(-(A^{\mathsf T}y)_j)$ (deviation from the exact posterior $2.2\times10^{-16}$), which is why the E-step is cheap although $p_Y$ is a mixture of $2^{n_h}$ exponential-family members (8.12). The M-step has no closed form, but it is a convex moment-matching problem: solved by Newton, the residual after each of 400 steps was at most $9.4\times10^{-15}$ and the log-likelihood rose at every step. And the em algorithm run with no closed form anywhere (mirror descent for each E-step, Newton with finite-difference derivatives for the M-step) follows the posterior-plus-moment-matching version to $2.4\times10^{-10}$, $3.8\times10^{-10}$, $7.9\times10^{-10}$ after 1, 3 and 5 steps: EM and em are one algorithm in a discrete model as well.

### 1.4 What the algorithm converges to

**In plain words.** The divergence decreases and is bounded below, so the likelihood converges. That alone says nothing about *where*. The remark after Theorem 8.2 adds that the m-projection is not unique unless $M$ is e-flat, "hence" local minima. I tried to find out what can actually happen.

**Stationary points.** At the end of EM from $(w,\mu)=(0.5,-0.5,0.5)$ ($179$ steps), $(w_1,\mu_1,\mu_2)=(0.381885,-0.914132,1.615855)$, $\ell=-382.128142$, the gradient (by Fisher's identity) is $2.9\times10^{-13}$ and the Hessian eigenvalues are $-594.53,\,-91.47,\,-19.57$: a strict local maximum. So the fixed points of EM are the stationary points of $\ell$, as they must be.

**A stationary point that is not a maximum.** Start with equal means, $\mu_1=\mu_2=$ sample mean $0.6497$, $w=0.5$. The gradient is $4.9\times10^{-15}$ and $\ell=-450.4590$ against $-382.1281$ at the maximum. The Hessian has eigenvalues $-100.0,\,0.0,\,166.67$: a saddle. The Jacobian of the EM map there has eigenvalues $2.6667,\,1.0,\,0$, and the unstable one $2.6667$ is the sample variance (relative to the model's unit variance). EM started exactly there stays: $\lvert\mu_1-\mu_2\rvert$ is $0$ after one step, $2.4\times10^{-13}$ after 10, $4.3\times10^{-9}$ after 20, $7.8\times10^{-5}$ after 30, and then rounding error (about $10^{-16}$) is amplified by $2.67$ per step and the iteration leaves ($1.1\times10^{-2}$ after 35, $1.2$ after 40) for the maximum. Nudged by $10^{-6}$ ($\mu_{1,2}=$ sample mean $\pm10^{-6}$), the gap exceeds $0.5$ after $13$ steps (the linear estimate $2\times10^{-6}\times2.67^{13}$ is $0.69$) and EM ends at $\ell=-382.1281$; in exactly symmetric arithmetic it would stay for ever. An "equilibrium" is a fixed point, not a maximum.

**Local maxima although every M-step is unique.** $N=300$ points from three clusters at $-10,0,10$ with weights $0.4,0.2,0.4$, fitted with two unit-variance components. Started from $\mu=(-10,5)$, EM stops after 14 steps at $(w_1,\mu_1,\mu_2)=(0.3737,-9.7397,6.8289)$ with $\ell=-2604.5849$; started from $(-5,10)$ it stops after 37 steps at $(0.5581,-6.4478,9.5867)$ with $\ell=-2737.4011$. Both have gradient below $10^{-11}$ and all Hessian eigenvalues negative (largest $-82.4$): two strict local maxima, $132.82$ apart. $M$ is e-flat here, so each M-step is unique and global, yet the limit depends on the start: uniqueness of the M-step is not global convergence, because the divergence is not jointly convex in $(R,\xi)$. The remark's reason (non-uniqueness of the projection) does not explain this example; what e-flatness of $M$ buys is that each *step* is well defined. (Whether a Csiszár–Tusnády-type global theorem applies when one set is m-flat and the other e-flat is something I have not checked.)

**A crawl.** Feed the symmetric model $\tfrac12N(-m,1)+\tfrac12N(m,1)$ with data from a single $N(0,1)$ (exact expectations, no sampling). The fixed point is $m=0$, the map is $m\mapsto E[Z\tanh(mZ)]=m-m^3+O(m^5)$ with slope exactly $1$, and the iteration crawls: from $m=1$, $m_t=0.22404,\,0.07092,\,0.02237$ after $10,\,100,\,1000$ steps (the prediction $1/\sqrt{2t+1}$ gives $0.21822,\,0.07053,\,0.02236$); it takes $5002$ steps to get below $0.01$ (the prediction $1/(2\cdot0.01^2)$ is $5000$). The rate of EM is $1$ at a point where the components are not distinguishable (§1.5).

**Unknown variances.** The chapter says the mixture with unknown covariance matrices can be treated "in a similar way". The updates are indeed similar (the M-step also averages squared deviations), but the problem is no longer the same: with free variances the likelihood is unbounded and EM, still monotone, can run into the singularity: a component started on the isolated data point $y=4.687$ with variance $0.001$ has $\ell=-537.080$ at the start, $-374.742$ after one step (variance $2.9\times10^{-9}$, already above the regular maximum $-379.270$), and the next step sets the variance to $0$, where $\ell=+\infty$. With unit variances, as in the example, $\ell\le N\log(1/\sqrt{2\pi})=-183.8$ and there is no such trap.

### 1.5 How fast: the fraction of missing information

**In plain words.** EM moves by the gradient of the visible log-likelihood, but measured in the geometry of the *complete* data (§1.3). Near the answer the visible data say less about the parameter than the complete data would, and the less they say the more slowly the algorithm moves. The rate at which the error shrinks each step is the largest fraction of the complete information that the hidden label withholds. (The matrix form is classical and goes back to Dempster, Laird and Rubin, the authors of the EM paper the chapter cites; I have not re-read their statement. The chapter does not give it, but §8.2 supplies the very quantity.)

**Formal statement.** With $I_c$ the complete-data information (the posterior expectation of the negative Hessian of the complete log-likelihood), $I_o=-\partial^2\ell$ the observed information and $I_m=I_c-I_o=\sum_i\mathrm{Cov}_{h\mid y_i}[\text{complete score}]$ the missing information (Louis' formula), the Jacobian of the EM map at a fixed point is
$$DM=I-I_c^{-1}I_o=I_c^{-1}I_m.$$
For the population version $I_c\to Ng^X$, $I_o\to Ng^Y$, $I_m\to Ng^{X\mid Y}$ and $DM=I-(g^X)^{-1}g^Y=(g^X)^{-1}g^{X\mid Y}$, *with exactly the $g^{X\mid Y}$ of (8.29)–(8.31) for $T=y$*.

**Checks.** *Sample of 200.* The Jacobian of the EM map (finite differences of the closed form (8.26)) has eigenvalues $0.84535,\,0.13958,\,0$; $I-I_c^{-1}I_o$ computed from the finite-difference Hessian and $I_c=N\,\mathrm{diag}(1/w(1-w),w,1-w)$ agrees entrywise to $6.1\times10^{-8}$, and Louis' formula agrees with the Hessian to $5.2\times10^{-5}$ (the finite-difference error). The observed contraction $\lVert\xi_{t+1}-\xi^*\rVert/\lVert\xi_t-\xi^*\rVert$ at $t=10,20,40,60$ is $0.84647,\,0.84556,\,0.84536,\,0.84535$, converging to the largest eigenvalue. The eigenvalue $0$ is the direction of the mixture mean $w_1\mu_1+w_2\mu_2$: after one step it equals the sample mean exactly (largest deviation $5.6\times10^{-16}$ over 5 random starts), so the hidden label carries no missing information about that combination.
*Population*, at the true value $(0.35,-1,1.5)$ by quadrature on a grid of step $0.02$: $G_X=\mathrm{diag}(4.3956,\,0.35,\,0.65)$ (also by quadrature of the score outer product, off-diagonal $3.1\times10^{-16}$), and
$$G_Y=\begin{bmatrix}3.0037&-0.3568&-0.4349\\-0.3568&0.2223&-0.0752\\-0.4349&-0.0752&0.4778\end{bmatrix},\qquad G_{X\mid Y}=\begin{bmatrix}1.3919&0.3568&0.4349\\0.3568&0.1277&0.0752\\0.4349&0.0752&0.1722\end{bmatrix}.$$
(8.30) holds: $\lvert G_X-G_Y-G_{X\mid Y}\rvert\le9.7\times10^{-14}$, and $G_{X\mid Y}$ is positive semidefinite (eigenvalues $0,\,0.0725,\,1.6193$). The information equality (score variance equals minus expected Hessian) holds for the marginal model to $3.0\times10^{-7}$. The eigenvalues of $G_X^{-1}G_{X\mid Y}$ are $0.79203,\,0.15448,\,0$, and so are those of the Jacobian of the EM map run on the whole density (it satisfies $\lvert J-(I-G_X^{-1}G_Y)\rvert=2.2\times10^{-10}$); the observed contraction at steps 30, 40, 50 is $0.79204,\,0.79203,\,0.79203$ (the sample of 200 gave $0.84535$: sampling variation).
*One parameter.* For $\tfrac12N(-m,1)+\tfrac12N(m,1)$ the complete information is $1$ and the rate is $E[y^2\operatorname{sech}^2(my)]=1-I_o(m)$. EM run on the whole density from $m_0=m+0.001$:

| true $m$ | rate (formula) | $I_o(m)$ | measured contraction | steps to error $10^{-10}$ |
|---|---|---|---|---|
| 0.25 | 0.888473 | 0.111527 | 0.888473 | 137 |
| 0.50 | 0.656736 | 0.343264 | 0.656736 | 39 |
| 1.00 | 0.266088 | 0.733912 | 0.266088 | 13 |
| 1.50 | 0.084157 | 0.915843 | 0.084156 | 7 |
| 2.00 | 0.021655 | 0.978345 | 0.021655 | 5 |
| 3.00 | 0.000776 | 0.999224 | 0.000775 | 3 |
| 5.00 | 0.000000 | 1.000000 | (exact in 1 step) | 1 |

Far apart, nothing is missing and EM is exact after a step or two; as the components merge the rate tends to 1, and at $m=0$ it is exactly 1, the crawl of §1.4.
*The Boltzmann machine.* At $W_{\rm true}$ the eigenvalues of $G_X^{-1}G_{X\mid Y}$ are $0.9851,\,0.9589,\,0.9267,\,0.9048,\,0.8047,\,0.407,\,0.3809,\,0.3788$: four of the eight directions have more than 90 percent of their information missing (the visible units identify the weights only weakly), and the observed ratio of consecutive increments at steps 300–400 is $0.9851$, as predicted; reaching an error of $10^{-3}$ along the slowest direction takes about $\ln(10^{-3})/\ln(0.9851)=461$ steps.

<img src="figures/em-rate.svg" alt="Left: the contraction factor of EM per step against the separation of the component means, for the symmetric mixture and for a mixture with weights 0.35 and 0.65. The curves are one minus the Fisher information of the marginal model over the complete-data information, and the largest eigenvalue of the inverse complete information times the missing information; dots are measured by running EM. The factor goes from 0 for far apart components to 1 for merging components. Right: the number of steps (logarithmic scale) needed to reduce the error by seven orders of magnitude, from the formula and from runs.">

## 2. Loss of information by data reduction (§8.2)

### 2.1 The identity (8.30)

**In plain words.** The Fisher information of a model is the variance of the *score*, the sensitivity of the log-probability of the data to the parameter. If you keep only a summary $T$ of the data, you keep the score of $T$ and lose what the data would still tell you about the parameter *given* $T$. The key fact is that these two pieces are uncorrelated: the leftover score has mean zero given $T$, so it cannot be explained by anything that depends on $T$ only. Variances of uncorrelated pieces add.
*A small example.* Three observations of $N(0.3,1)$ and the parameter $(\mu,\sigma)$: the whole data have $g^X=\mathrm{diag}(3,6)$. Keep $T=\bar x$ only: $g^T=\mathrm{diag}(3,2)$ and the loss is $g^{X\mid T}=\mathrm{diag}(0,4)$, the information about $\sigma$ in the spread that $\bar x$ forgets. Keep the sum of squares about the mean only: $g^T=\mathrm{diag}(0,4)$, loss $\mathrm{diag}(3,2)$. Keep both: $g^T=g^X$ and the conditional score is identically zero (largest $\lvert\text{score}_X-\text{score}_T\rvert$ over the quadrature nodes $3.1\times10^{-14}$): the pair is sufficient. In all three cases the cross term $E[\text{score}_T\,\text{score}_{X\mid T}]$ is at most $2.9\times10^{-16}$ and $g^X=g^T+g^{X\mid T}$ to $4.7\times10^{-14}$ or better; $\bar x$ and the sum of squares are independent, so each loses exactly the information of the other.

**Formal statement** (8.28)–(8.31). With $\partial_i\log p(D_X;\xi)=\partial_i\log p(T;\xi)+\partial_i\log p(D_X\mid T;\xi)$,
$$g^X_{ij}=g^T_{ij}+g^{X\mid T}_{ij},\qquad g^{X\mid T}_{ij}=E_T\big[\mathrm{Cov}(\partial_i\log p,\partial_j\log p\mid T)\big]\ \succeq0,$$
and $\Delta g^T=g^{X\mid T}$ is the loss. The book gives no derivation of the equality; the orthogonality above is the proof.

**Grouping and censoring (checks).** Replace one observation of $N(0.3,1)$ by the width-$h$ bin it falls in. $g^T$ comes from the multinomial formula $\sum_k(\partial P_k)(\partial P_k)^{\mathsf T}/P_k$ and $g^{X\mid T}$ independently from the score covariance inside each bin; they add up to $g^X$ to $1.7\times10^{-13}$ in the worst row, and the conditional mean of the score in a bin equals the score of $T$ (to $10^{-9}$ for bins of probability above $10^{-6}$):

| $h$ | $g^T_{\mu\mu}$ | $g^T_{\sigma\sigma}$ | lost fraction, $\mu$ | $h^2/12$ | lost fraction, $\sigma$ | $h^2/6$ |
|---|---|---|---|---|---|---|
| 0.25 | 0.99482 | 1.97936 | 0.00518 | 0.00521 | 0.01032 | 0.01042 |
| 0.50 | 0.97959 | 1.91968 | 0.02041 | 0.02083 | 0.04016 | 0.04167 |
| 1.00 | 0.92308 | 1.71032 | 0.07692 | 0.08333 | 0.14484 | 0.16667 |
| 2.00 | 0.75336 | 1.12176 | 0.24664 | 0.33333 | 0.43912 | 0.66667 |
| 3.00 | 0.63917 | 0.35924 | 0.36083 | 0.75000 | 0.82038 | 1.50000 |

Small bins lose $h^2/12\sigma^2$ of the information about $\mu$ and twice that about $\sigma$ (the score is nearly uniform inside a narrow bin); coarse bins lose less than that law says. *Censoring*, $T=\min(x,c)$: the atom at $c$ carries the score $E[\text{score}\mid x\ge c]$ and the lost part is $P(x\ge c)\,\mathrm{Cov}(\text{score}\mid x\ge c)$. For $x\sim N(0.3,1)$ and $c=-1,\,0,\,1,\,2,\,3$: $P(x\ge c)=0.90320,\,0.61791,\,0.24196,\,0.04457,\,0.00347$, lost fraction for $\mu$ $0.64791,\,0.26809,\,0.05758,\,0.00597,\,0.00028$ and for $\sigma$ $0.57609,\,0.54496,\,0.30608,\,0.06874,\,0.00592$; the identity holds to $1.6\times10^{-15}$.

<img src="figures/info-loss.svg" alt="Left: the fraction of the Fisher information about the mean and about the standard deviation of a Gaussian that is lost when the observation is replaced by the index of a bin of width h; for small h the losses are h squared over 12 and h squared over 6 (dotted), for coarse bins they are smaller. Right: the fraction lost when the observation is replaced by the minimum of x and a censoring point c; it is small for c large and grows as c moves towards the centre of the distribution.">

### 2.2 Hidden variables are a data reduction

Take $X=(y,h)$ per observation and $T=y$. Then $g^{X\mid Y}=E_y\mathrm{Cov}_{h\mid y}[\text{complete score}]$, since $\partial\log p(h\mid y;\xi)$ is the complete score minus its posterior mean (Fisher's identity), and it is the *missing information* of §1.5. The book's remark that $T=y$ is not sufficient in general has a precise form: $T=y$ is sufficient iff the posterior of $h$ does not depend on $\xi$, and then $g^{X\mid Y}=0$ and EM is exact in one step. The population matrices of §1.5 are the numerical example (the loss in the weight direction is $1.3919$ out of $4.3956$; in the mean direction there is none).

### 2.3 Three neurons: dropping the third-order correlation

**In plain words.** Three neurons are recorded as a table of eight firing patterns. If the experimenter keeps only the firing rates and the pairwise correlations, one number is thrown away: the tendency of all three to fire together beyond what the pairs explain. How much does that cost for a parameter that lives in that number?

**The book's paragraph.** The firing patterns of $n$ neurons form the full exponential family (8.32)–(8.33) with statistics $X=(x_i,x_ix_j,\dots,x_1\cdots x_n)$; if the high-order correlations are not recorded, the observed point is not identified and one has a data manifold $D$ instead, and the best estimator is the minimiser of $D_{KL}[D:M]$. The size of the loss is attributed to a paper and not given.

**What it comes to.** Three neurons, $p(x;\xi)=\exp(\theta\cdot X-\psi)$ with $\theta=(-1,-1,-1,0,0,0,\xi)$: independent sparse neurons (firing probability $0.2689$) with a third-order interaction $\xi$, true value $\xi=0$. An experimenter who keeps rates and pairwise correlations has the data manifold $D=\{q+s\cdot\text{parity}\}$, parity$(x)=(-1)^{x_1+x_2+x_3}$, an m-flat *line* (it has zero first and second moments exactly). In the mixed coordinates $(\eta_R,\theta^{123})$ of Chapter 6 (§6.8: the retained moments as expectation parameters, the lost interaction as a natural parameter) the Fisher metric is block diagonal, so
$$\text{loss}=\Big(\frac{d\theta^{123}}{d\xi}\Big)^2\Big(G_{LL}-G_{LR}G_{RR}^{-1}G_{RL}\Big),$$
where the bracket is the *residual variance of $x_1x_2x_3$ after regressing it on the other six statistics* (I computed it that way as well; it is the Schur complement that Chapter 6 found for the $\theta^{12}$ entry of the mixed metric, and it is not the corresponding entry of $G$ or of $G^{-1}$). Numbers: $g^X=\mathrm{Var}(x_1x_2x_3)=0.019074$, retained information $g^T=0.011474$, loss $0.007600=0.3985\,g^X$, and the regression has $R^2=0.6015$. Exact check of (8.30) at finite $N$ with $T$ the six retained counts, by enumerating all count vectors:

| $N$ | count vectors | values of $T$ | $g^X$ | $g^T$ | $g^{X\mid T}$ | residual |
|---|---|---|---|---|---|---|
| 4 | 330 | 329 | 0.076296 | 0.075603 | 0.000693 | $2.6\times10^{-18}$ |
| 8 | 6435 | 6105 | 0.152592 | 0.139804 | 0.012788 | $1.7\times10^{-17}$ |
| 12 | 50388 | 43953 | 0.228888 | 0.192505 | 0.036383 | $6.9\times10^{-18}$ |

(8.30) is exact at every $N$; per observation the loss approaches $0.007600$ slowly, because at small $N$ the six counts almost determine the whole table (the ratio to $N\cdot0.007600$ is $0.023,\,0.210,\,0.399$).

**The optimum estimator.** The estimator that minimises $\mathrm{KL}[D:M]$ is itself an em algorithm: e-project the model point onto $D$ (one-dimensional Newton for $s$), m-project onto $M$ (moment matching for $\xi$). Run on the population point from $\xi_0=1$, its error contracts by $0.397312,\,0.398451,\,0.398463$ per step, and the loss fraction is $0.398463$: the em rate is the fraction of information lost by the reduction (§1.5 in a second setting). Monte Carlo with $N=20000$ patterns, 4000 data sets, true $\xi=0$, $N\cdot\mathrm{Var}$ of the estimator: full-data MLE $52.767$ against $1/g^X=52.427$; minimiser of $\mathrm{KL}[D:M]$ from the retained moments $88.618$ against $1/g^T=87.156$ (relative standard error of a variance $0.022$); *unweighted* least squares on the same retained moments $150.484$. So "the optimum estimator is the minimiser of $D_{KL}[D:M]$" is right for this example and it matters: it attains the information of the reduced data ($R^2=0.6015$ of the full information) while naive least squares does much worse (the ratio of the first two variances is $1.679$ against $g^X/g^T=1.662$).

## 3. Estimation based on a misspecified statistical model (§8.3)

### 3.1 The q-MLE is the KL-projection of the truth

**In plain words.** You believe the data come from a model $M_q$ but they come from something else, $p$. The maximum likelihood estimator of the wrong model still converges to *something*: the point of $M_q$ that is closest to $p$ in the sense $\mathrm{KL}[p\Vert q]$, the **m-projection** of the truth onto $M_q$. Whether this is the "right" parameter value depends on how the parameters of the two models are tied: here both are indexed by $u$, and the q-MLE is consistent only when the point it converges to carries the true label.
**Setup in the plane.** The truth is the curved family $M=\{N(u,u^2)\}$ of Chapter 7 (called $S$ there), a parabola in the plane of the observed point $\bar\eta=(\bar x,\overline{x^2})$; the wrong models are curves in the same plane. The q-MLE sends the observed point to the nearest point of $M_q$ along a straight line (an m-geodesic) orthogonal to $M_q$: the **q-ancillary leaf** $A_q(v)$. I checked this for $q=N(v,1.5v^2)$ (the right shape but the wrong coefficient of variation): for 1500 observed points ($N=20$) the minimiser over $v$ of $\mathrm{KL}[p_{\bar\eta}\Vert q(v)]$ by golden-section search agrees with the closed form $(-\bar x+\sqrt{\bar x^2+4c\,m_2})/2c$ to $1.5\times10^{-8}$, and the leaf equation $\theta_q'(\hat v)\cdot(\bar\eta-\eta_q(\hat v))=0$ holds to $1.3\times10^{-15}$: the leaf through $q(v)$ is the straight line orthogonal to $\theta_q'(v)$, and it is orthogonal to $M_q$ at its foot ($9.0\times10^{-12}$ in the Fisher metric).

### 3.2 Theorem 8.3: consistency

**Formal statement.** The q-MLE is consistent iff $E_{p(x,u)}[\partial_a\log q(x,u)]=0$ (8.37), which holds when the leaf $A_q(u)$ passes through $p(x,u)$. Because $\partial_v\log q=\theta_q'(v)\cdot(X-\eta_q(v))$, (8.37) in expectation reads $\theta_q'(v)\cdot(\eta_p-\eta_q(v))=0$, which is literally "the leaf through $q(v)$ contains $\eta_p$"; checked at 12 pairs $(u,v)$ to $1.5\times10^{-8}$. So the two formulations in the theorem are one.

**A sign slip in the proof (page image of p. 188).** For $r(t)=(1-t)q+tp$ the derivative of $\log r$ at $t=0$ is $(p-q)/q$; the printed (8.39) has $\{q-p\}/q$. I checked it by finite differences at 11 points (agreement to $2.5\times10^{-10}$ with $(p-q)/q$, and it is the negative of the printed one). The consequence is in (8.41): $\langle\dot r,\dot l_q\rangle_q=\int(p-q)\partial_u\log q\,dx=+E_p[\partial_u\log q]$ (for $q=N(0.5,1)$, $p=N(1,1)$ it is $+0.5000$), so the last line should end with $+\int p\,\partial_u\log q\,dx$. The conclusion, that the pairing vanishes iff (8.37) holds, is unaffected.

**Pseudo-true values, three ways.** The limit $v^*(u)$ of the q-MLE when the data come from $N(u,u^2)$: by the root of (8.37), by the minimiser of $\mathrm{KL}[p\Vert q(v)]$, and by the average of the q-MLE over $2\times10^6$ exact data sets of $N=2000$ (the sufficient statistics sampled from their exact law):

| model | $u$ | root of (8.37) | $\arg\min\mathrm{KL}$ | q-MLE, Monte Carlo | $\kappa u$ |
|---|---|---|---|---|---|
| $N(v,1)$ | 0.6 | 0.600000 | 0.600000 | 0.59999 ± 0.00001 | 0.60000 |
| $N(v,1)$ | 1.0 | 1.000000 | 1.000000 | 1.00001 ± 0.00002 | 1.00000 |
| $N(v,1)$ | 1.4 | 1.400000 | 1.400000 | 1.40001 ± 0.00002 | 1.40000 |
| $N(0,v^2)$ | 0.6 | 0.848528 | 0.848528 | 0.84844 ± 0.00001 | 0.84853 |
| $N(0,v^2)$ | 1.0 | 1.414214 | 1.414214 | 1.41410 ± 0.00001 | 1.41421 |
| $N(0,v^2)$ | 1.4 | 1.979899 | 1.979899 | 1.97972 ± 0.00002 | 1.97990 |
| $N(v,1.5v^2)$ | 0.6 | 0.521110 | 0.521110 | 0.52105 ± 0.00000 | 0.52111 |
| $N(v,1.5v^2)$ | 1.0 | 0.868517 | 0.868517 | 0.86842 ± 0.00001 | 0.86852 |
| $N(v,1.5v^2)$ | 1.4 | 1.215924 | 1.215924 | 1.21580 ± 0.00001 | 1.21592 |

(The Monte Carlo averages are within about $10^{-4}$ of the limit; the rest is the $O(1/N)$ bias of the estimators.) Only $N(v,1)$ is consistent; the others converge to $v^*=\kappa u$ with $\kappa=\sqrt2$ for $N(0,v^2)$ and $\kappa=(\sqrt{1+8c}-1)/2c=0.8685$ for $N(v,cv^2)$, $c=1.5$.

**The remark after Theorem 8.4.** If $f(v)$ is the $M$-coordinate where the leaf $A_q(v)$ meets $M$ ($f(v)=v/\kappa$ here), relabelling the points of $M_q$ makes the estimator consistent. The book writes that the new parameter of $M_q$ is "$f^{-1}(u)$", which is ambiguous in direction; the labelling that works is $q_{\rm new}(x;w)=q(x;f^{-1}(w))$, i.e. estimate $w=f(\hat v)$. Checked: the relabelled estimators $\hat v/\kappa$ have mean $1.00000,\,0.99996,\,0.99994$ at $u=1$ for the three models, with standard error $10^{-5}$ ($10^{-4}$-level differences are the $O(1/N)$ bias). So *consistency can always be repaired by relabelling; efficiency cannot*.

### 3.3 Theorem 8.4: the loss of information

**Formal statement.** For a consistent q-MLE (leaf through $p$), in coordinates $(u^a,v^\kappa)$ adapted to the leaf, the loss of Fisher information is $\Delta g_{ab}=g_{a\kappa}g_{b\lambda}g^{\kappa\lambda}$ (8.42), zero iff the leaf is orthogonal to $M$. This is (7.58) with the leaves of the wrong model. As in Chapter 7, $g^{\kappa\lambda}$ must be the inverse of the $\kappa\lambda$ block, not the block of the full inverse; in the five-dimensional example below the two readings give $0.562500$ and $0.878906$ for the same loss. The theorem as printed does not say that the leaf must pass through $p$; it is the remark that supplies it.

**Four routes to the same number.** For the three models, efficiency of the relabelled q-MLE at $u=1$ ($g_{uu}=3$, Cramér–Rao bound $1/3$): (a) geometry: leaf direction orthogonal to $\theta_q'(v^*)$, $\Delta g=g_{uv}^2/g_{vv}$ in the Fisher metric of $S$ at the truth; (b) the sandwich $A^{-1}BA^{-1}$ with $A=-E_p[\partial_v^2\log q]$, $B=E_p[(\partial_v\log q)^2]$; (c) the squared correlation of the true score and the q-score under $p$; (d) Monte Carlo, $N=4000$, $2\times10^6$ data sets.

| model | geometry | sandwich | corr² | asymptotic $N\mathrm{Var}$ | Monte Carlo $N\mathrm{Var}$ |
|---|---|---|---|---|---|
| $N(v,1)$ | 0.33333 | 0.33333 | 0.33333 | 1.00000 | 1.00079 ± 0.00100 |
| $N(0,v^2)$ | 0.88889 | 0.88889 | 0.88889 | 0.37500 | 0.37541 ± 0.00038 |
| $N(v,1.5v^2)$ | 0.99649 | 0.99649 | 0.99649 | 0.33451 | 0.33467 ± 0.00033 |

With a two-dimensional $M$ the same statement holds as a matrix identity: for three neurons, two parameters, $V=\left[\begin{smallmatrix}1&0.8&0.1\\0.8&1.5&0.9\\0.1&0.9&2\end{smallmatrix}\right]$ in the nine-dimensional Gaussian family (leaf of dimension 7), the loss matrix from (8.42) is $\left[\begin{smallmatrix}0.47806&-0.00105\\-0.00105&0\end{smallmatrix}\right]$ and $g-AB^{-1}A$ from the sandwich ($A=J^{\mathsf T}J$, $B=J^{\mathsf T}VJ$) is the same to $1.1\times10^{-14}$; the fractions of information lost in the two eigendirections are $0$ and $0.27537$, and reading $g^{\kappa\lambda}$ as a block of the full inverse gives $0.65974$ in the first entry instead of $0.47806$.

The first two models are the leaf families of Chapter 7: vertical leaves are the sample mean (efficiency $\tfrac13$) and horizontal leaves are $\sqrt{m_2/2}$ ($\tfrac89$ after relabelling). In general the efficiency is the squared correlation between the true score and the model's score, because for a q-MLE with $E_p[\text{q-score}]=0$ identically in $u$ one has $A=E_p[\text{q-score}\cdot\text{true score}]$ (differentiate $E_p[\text{q-score}]=0$ in $u$, with the relabelling absorbed); in the leaf language that is $\sin^2$ of the angle. The wrong coefficient of variation costs bias ($v^*=0.8685u$) but almost no efficiency: the leaves are nearly orthogonal to $M$.
*The geometry of the misspecified likelihood.* When $M_q$ is e-flat ($N(v,1)$ and $N(0,v^2)$ are straight lines in $\theta$) the foot $q^*$ of the m-projection satisfies the exact Pythagorean relation $\mathrm{KL}[p\Vert q(v)]=\mathrm{KL}[p\Vert q^*]+\mathrm{KL}[q^*\Vert q(v)]$ for every $v$ (residual $1.5\times10^{-12}$ and $4.6\times10^{-10}$), so the expected log-likelihood is a bowl whose curvature $A$ is the Fisher information of the *model*, $A=g_q(v^*)=1.00000$ for both. For the curved $N(v,1.5v^2)$ the relation fails (residual $6.9\times10^{-2}$), and $A=3.66898$ differs from $g_q(v^*)=3.53518$ by exactly $-\theta_q''\cdot(\eta_p-\eta_q)=+0.13380$: the e-curvature of $M_q$ times the displacement of $p$ from its foot.

<img src="figures/misspecified.svg" alt="Left: the plane of the observed point (mean of x, mean of x squared) with the true model N(u, u squared) as a black parabola, the misspecified model N(v, 1.5 v squared) in green, its straight leaves, and the three leaves through the true point N(1, 1): the vertical leaf of the model N(v, 1), the horizontal leaf of the model N(0, v squared) and the sloped leaf of N(v, 1.5 v squared), with the dashed leaf of the true maximum likelihood estimator. Right: the efficiency of a consistent estimator as a function of the angle of its leaf, equal to one third for the vertical leaf, 8/9 for the horizontal leaf and 1 for the orthogonal leaf; the three models sit on the curve.">

### 3.4 The neural field

**In plain words.** Neurons that share noise respond together, and a decoder that pretends they are independent throws that structure away. Does it lose anything? The book says no, asymptotically; the answer depends on how the noise correlation lines up with the way the average response changes with the stimulus.

**The book's example.** $n$ neurons respond with a Gaussian vector of mean $r(u)$ and covariance $V$ (8.35); the model ignores the covariance (8.36). The text then reports that Wu et al. showed that asymptotically nothing is lost by using the simple model, and presents this as encouraging for the brain.

**What the geometry says.** Both are Gaussian with the same mean, and $V$ does not depend on $u$. Then $E_p[\partial_u\log q]=r'\cdot(E_px-r)=0$ for every $V$: the q-MLE (least squares) is consistent. Its q-score is $r'\cdot(x-r)$, the true score $r'^{\mathsf T}V^{-1}(x-r)$, their covariance under $p$ is $\lvert r'\rvert^2$, and the variances are $r'^{\mathsf T}Vr'$ and $r'^{\mathsf T}V^{-1}r'$. So by the correlation route and by the sandwich
$$\text{efficiency}=\frac{\lvert r'\rvert^4}{(r'^{\mathsf T}Vr')(r'^{\mathsf T}V^{-1}r')}\le1,$$
with equality iff $r'$ is an eigenvector of $V$ (Cauchy–Schwarz). "No loss" is a property of the pair $(V,r')$, not of the misspecified model. *Two neurons*, $r(u)=(\cos u,\sin u)$, $V=\left[\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right]$, $\rho=0.6$, geometry in the five-dimensional Gaussian family (the leaf is 4-dimensional; $g^{\kappa\lambda}$ the inverse of the block, in an arbitrary basis):

| $u$ | $g_{uu}$ | loss (8.42) | efficiency, geometry | formula | corr² | loss, block of the full inverse |
|---|---|---|---|---|---|---|
| 0 | 1.5625 | 0.562500 | 0.640000 | 0.640000 | 0.640000 | 0.878906 |
| 0.785 ($\pi/4$) | 2.5000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| 0.700 | 2.4864 | 0.039757 | 0.984010 | 0.984010 | 0.984010 | 0.040403 |

At $u=0$ the tuning direction $r'=(0,1)$ is along an axis, not an eigenvector of $V$ (whose eigenvectors are $(1,1)$ and $(1,-1)$), and the efficiency is $1-\rho^2=0.64$; at $u=\pi/4$ it is an eigenvector and nothing is lost.
*A ring of $n$ neurons*, Gaussian bump tuning of width $a=0.1$, correlation $(1-\rho)I+\rho K$ with $\rho=0.5$ and $K$ the wrapped Gaussian of the ring distance with range $b$:

| $n$ | uniform $K$ | $b=0.02$ | $b=0.05$ | $b=0.10$ | $b=0.20$ | $b=0.50$ |
|---|---|---|---|---|---|---|
| 25 | 1.0000 | 0.9999 | 0.9887 | 0.8557 | 0.5696 | 0.9938 |
| 50 | 1.0000 | 0.9997 | 0.9843 | 0.7997 | 0.4191 | 0.9787 |
| 100 | 1.0000 | 0.9996 | 0.9811 | 0.7525 | 0.2966 | 0.9354 |
| 200 | 1.0000 | 0.9995 | 0.9790 | 0.7170 | 0.2017 | 0.8366 |
| 400 | 1.0000 | 0.9995 | 0.9779 | 0.6924 | 0.1316 | 0.6700 |

Uniform correlation loses nothing ($r'$ sums to $-1.9\times10^{-4}$, so it is an eigenvector of $(1-\rho)I+\rho\mathbf 1\mathbf 1^{\mathsf T}$; the loss is $3.9\times10^{-11}$), short-range correlation loses little, correlation as wide as the tuning curve loses most of the information and more as $n$ grows ($b=0.2$: $0.570$ at $n=25$, $0.132$ at $n=400$); for $n=100$ the efficiency is smallest, $0.246$, at $b=0.26$, and returns to $1$ as the correlation becomes uniform ($0.935$ at $b=0.5$). Cosine tuning has $r'$ a single Fourier mode, an eigenvector of every translation-invariant $V$: the loss is $4.4\times10^{-16}$ for all four ranges.

**How to read the cited result.** I did not check the cited paper's own model; I read the conclusion of Wu, Amari and Nakahara (2002, *Neural Computation* 14), whose asymptotics are in the number of neurons for a single presentation of the stimulus, not in repeated observations. By my reading it lists uniform correlation (and a multiplicative variant) as cases where this type of decoder is asymptotically efficient, speaks of quasi-efficiency for the decoder that ignores correlation (the usual rate and a Gaussian limit, but not the Cramér–Rao bound), and says that for correlation of non-local range the estimators are non-Fisherian (the Cramér–Rao paradigm does not apply). That fits the geometry above, and means the book's one-line summary compresses several cases.

<img src="figures/neural-field.svg" alt="Left: efficiency of the least-squares decoder that ignores a covariance V for two neurons, as a function of the direction of the derivative of the tuning curve, for correlations 0.3, 0.6 and 0.9; it is one when the direction is an eigenvector of V and one minus rho squared along an axis; dots are the geometric computation of Theorem 8.4. Right: efficiency for a ring of n neurons with Gaussian bump tuning of width 0.1 and wrapped-Gaussian correlation of range b, for n = 25, 100 and 400: nearly one for short-range correlation, much smaller for correlation as wide as the tuning curve and decreasing with n, and one again for uniform correlation (large b); cosine tuning gives exactly one.">

## Checks of the book's statements

| Where | Statement | What I found |
|---|---|---|
| (8.6)–(8.7) | $D$ is m-flat | verified: a mixture of two tables keeps the $y$-marginal ($2.8\times10^{-17}$); $D$ is not e-flat (marginal of the geometric mean off by $0.0762$) |
| Theorem 8.1, (8.14)–(8.15) | the MLE minimises the divergence from $D$ to $M$ | verified through the exact chain rule $F=-\ell/N+\text{posterior divergence}$ ($3.6\times10^{-15}$, 300 random pairs); **for continuous data the divergence from an empirical distribution is infinite**, the constant $c$ of (8.15) is that infinity and only differences make sense |
| Lemma 8.1, (8.21) | the e-projection onto $D$ is the posterior | verified by brute force ($1.5\times10^{-8}$) and by the exact Pythagorean relation over 2000 random tables ($1.1\times10^{-15}$); orthogonality of the e-geodesic $5.6\times10^{-17}$ |
| (8.17), (8.22) | stationarity of $\int q\log p\,dh$, then "the MLE" | **a factor $1/p_Y(y;\xi)$ is missing** in (8.17); harmless for one $y$ but not summed over a sample: at the MLE the summed printed form is $(-9.9697,3.7851,-4.6611)\neq0$ |
| (8.23), (8.25) | E-step and conditional expectation | **$p(y,R,\xi_0)$ should read $p(y,h,\xi_0)$**; in (8.25) **$y_h$ should be $\mu_h$** (page images) |
| (8.26) | M-step formulas for the mixture | verified; they are moment matching, the m-projection onto the e-flat $M$ (matches to $2.2\times10^{-16}$; Newton $9.1\times10^{-11}$) |
| Theorem 8.2 | the divergence decreases monotonically | verified, with an exact accounting: M part $\mathrm{KL}[p_{t+1}\Vert p_t]$ plus E part (posterior KL) equals the loglik gain to $4.3\times10^{-16}$ for 40 steps; the accounting needs $M$ e-flat (for a curved model the ratio of the M-step's gain to that KL is $1.0604,\,0.9769,\,\dots$) |
| Theorem 8.2 | the conclusion that the algorithm converges to an equilibrium | the likelihood converges; the iterates reach stationary points: a **saddle** ($\ell=-450.4590$, Hessian eigenvalues $-100.0,0.0,166.67$; EM stays for about 35 steps), a **second local maximum** ($-2737.4011$ against $-2604.5849$) and a **crawl** at the null model ($5002$ steps to reach $0.01$) |
| remark after Th. 8.2 | the m-projection is unique iff $M$ is e-flat, "hence" local minima | uniqueness holds (Pythagoras to $5.3\times10^{-15}$) **and** local maxima exist: the cause is the non-convexity of the joint divergence, not non-uniqueness of the M-step |
| §8.1.1 | unknown covariances can be treated "in a similar way" | the updates are similar, but **with free variances the likelihood is unbounded**: EM from a component started tightly on an isolated point reaches $\ell=-374.742$ in one step (above the regular maximum $-379.270$) and then diverges |
| (8.10)–(8.11) | Boltzmann machine and its restricted form | **a factor 2**: $\tfrac12x^{\mathsf T}Wx=y^{\mathsf T}Ah$ exactly, so (8.11)'s $\tfrac12y^{\mathsf T}Wh$ needs $W=2A$; (8.13) factorises over hidden units ($2.2\times10^{-16}$); EM with a Newton M-step and em by brute force agree to $7.9\times10^{-10}$ after 5 steps |
| (8.28)–(8.31) | $g^X=g^T+g^{X\mid T}$, loss $=g^{X\mid T}$ | verified exactly: mixture ($9.7\times10^{-14}$), grouping ($1.7\times10^{-13}$), censoring ($1.6\times10^{-15}$), three Gaussians with $T=\bar x$, sum of squares, both ($\le4.7\times10^{-14}$), enumeration for three neurons ($\le6.9\times10^{-18}$); the derivation is the orthogonality of the two scores, not given |
| §8.2, last paragraph | the optimum estimator is the minimiser of $D_{KL}[D:M]$ | verified on three neurons: $N\mathrm{Var}=88.618$ against $1/g^T=87.156$; the full MLE $52.767$ against $52.427$; unweighted least squares $150.484$; loss $=0.3985\,g^X$ and the em contraction is $0.398463$ |
| (8.33) | the number of patterns | written $N$ where the preceding sentence used $t$ (notation) |
| (8.35)–(8.36) and the sentence after | the no-loss claim for the misspecified model | **only when $V r'\parallel r'$**: efficiency $=\lvert r'\rvert^4/(r'^{\mathsf T}Vr'\,r'^{\mathsf T}V^{-1}r')$, $0.640000$ for two neurons with $\rho=0.6$ along an axis, $0.1316$ for 400 neurons with $b=0.2$, $1$ for uniform correlation or cosine tuning |
| Theorem 8.3, (8.37) | consistent iff $E_p[\partial_a\log q]=0$ | verified (three models, three ways); equivalent to the leaf through $q(v)$ containing $p$ ($1.5\times10^{-8}$); "iff" needs the root to be the global maximiser of the expected log-likelihood (not tested) |
| (8.39), (8.41) | derivative of $\log r$ and the pairing | **sign slip**: $\dot r=(p-q)/q$ and $\langle\dot r,\dot l_q\rangle_q=+E_p[\partial_u\log q]$ ($+0.5000$ for $q=N(0.5,1)$, $p=N(1,1)$); the conclusion stands |
| Theorem 8.4, (8.42) | loss $=g_{a\kappa}g_{b\lambda}g^{\kappa\lambda}$ | verified four ways ($0.33333,\,0.88889,\,0.99649$ for three models) and as a matrix for $\dim M=2$ ($1.1\times10^{-14}$); **$g^{\kappa\lambda}$ is the inverse of the block** ($0.562500$, not $0.878906$); the leaf must pass through $p$, which the theorem leaves to the remark |
| remark after Th. 8.4 | relabel $M_q$ by $f^{-1}$ | **direction ambiguous**; the labelling $q(x;f^{-1}(w))$, estimating $f(\hat v)$, gives means $1.00000,\,0.99996,\,0.99994$ |
| not in the chapter | EM contracts by the fraction of missing information | $DM=I-(g^X)^{-1}g^Y$: eigenvalues $0.79203,\,0.15448,\,0$ both ways; sample $0.84535$; Boltzmann machine $0.9851$ |

## Questions and doubts

- **Which conditions give convergence of the iterates, not only of the likelihood?** Theorem 8.2 concludes convergence to an equilibrium with no hypotheses. Monotonicity gives convergence of $\ell$; I saw iterates reach a saddle and a poor local maximum, and a singularity with unknown variances. What would settle it: a statement of the regularity conditions (compact level sets, isolated stationary points) under which limit points are stationary, and a reference for when they are maxima.
- **How much of the em picture survives when $M$ is not e-flat?** My exact gain identity (§1.3) used the Pythagorean relation for the m-projection onto an e-flat $M$. In one curved example (weights fixed at $\tfrac12$) the KL accounting fails while the split into the M-step's drop of $F$ and the E part still holds; uniqueness of the M-step and global statements for curved models are untested.
- **Is there a global statement for alternating projection between an m-flat and an e-flat set?** The unit-variance mixture has $D$ convex and $M$ e-flat, and still has two strict local maxima 132.82 apart. A Csiszár–Tusnády-type result seems to need both sets convex in the same sense; I have not looked it up, only observed that this case is not covered.
- **How much does the cited neural-field result actually claim?** I read the conclusion of Wu–Amari–Nakahara (2002) but did not reproduce its model (a continuous field, one presentation, asymptotics in the number of neurons). The exact criterion I derived ($Vr'\parallel r'$) applies to Gaussians of the book's form; matching it to the paper's conditions (local-range, uniform, multiplicative correlation) would need the paper's own formulas.
- **Is the loss formula behind the neuron paragraph the same as in the cited paper (Oizumi et al. 2011)?** The chapter gives no formula. Mine is a mixed-coordinates one (squared derivative of the lost $\theta$-coordinate times a Schur complement) and was checked on one family of three neurons, for the third-order interaction only; lost second-order correlations and larger $n$ are untested.
- **Is "iff" in Theorem 8.3 right without identifiability?** $E_p[\partial\log q]=0$ is a stationarity condition. A q-MLE converges to a global maximiser of the expected log-likelihood, so one needs that the stationary point is that maximiser. It can fail when the score has spurious zeros (the model $N(v^2,1)$ has score $2v(x-v^2)$, which vanishes at $v=0$ whatever the truth); I only tested monotone examples.
- **Beyond Gaussians.** (8.42) was tested as a matrix identity for a one- and a two-dimensional $M$ (leaves of dimension 1, 4 and 7), but only for Gaussian models inside the Gaussian family; the m-flat leaves and the Schur-complement structure for other exponential families (where $\theta_q''$ and the curvature term of §3.3 matter) are untested.
- **Higher-order and finite-$N$ behaviour.** Everything in §3 is first order. The exact finite-$N$ identity (8.30) is verified, but the finite-$N$ behaviour of the q-MLE (bias of order $1/N$, visible as $10^{-4}$ in the tables) is not explained by the chapter's theory.

## Takeaways

- **Hidden variables turn estimation into a distance between two sets.** $D$ (tables of guesses for $h$) is m-flat, $M$ (the complete model) is e-flat in my examples; the closest pair is the maximum likelihood estimate, by the chain rule $F=-\ell/N+\text{posterior divergence}$.
- **EM is the em algorithm and nothing else.** E-step: e-projection onto $D$ (posterior). M-step: m-projection onto $M$ (moment matching for an exponential family, the formulas (8.26)). Each half-step lowers the divergence by an explicit KL divergence, and the sum is the gain in log-likelihood.
- **EM is a unit natural-gradient step** along an m-geodesic of the complete-data family, and its speed is the largest **fraction of missing information**: eigenvalues $0.79203,\,0.15448,\,0$ for the mixture, $\to1$ as the components merge, $\to0$ when they separate.
- **An equilibrium is only stationary.** Saddles (EM stays on them until rounding error escapes), bad local maxima although every step is unique, a crawl where the rate is $1$, a singular likelihood when variances are unknown.
- **Data reduction: $g^X=g^T+g^{X\mid T}$**, exactly, from the orthogonality of the score of $T$ and the score of the rest. Binning loses $h^2/12\sigma^2$ and $h^2/6\sigma^2$ for small bins; hidden variables lose the missing information; three neurons that drop their third-order moment lose $0.3985$ of $g^X$ here, and the minimiser of $\mathrm{KL}[D:M]$ attains what is left.
- **A wrong model's MLE goes to the KL-projection of the truth** (consistent iff $E_p[\partial\log q]=0$), along straight leaves; relabelling repairs consistency, not efficiency. The loss is the angle between the leaf and $M$, the squared correlation of the true and the model score, and Chapter 7's (7.58) again.
- **"No loss" for ignoring correlation means $Vr'\parallel r'$,** not a general fact: $0.640000$ for $\rho=0.6$ along an axis, $0.1316$ for 400 neurons with $b=0.2$.
- **Fine print:** (8.17) lacks $1/p_Y(y;\xi)$ for a sample; (8.25) has $y_h$ for $\mu_h$; (8.39)/(8.41) have a sign; (8.11) a factor 2; $g^{\kappa\lambda}$ in (8.42) is a block inverse; the local-minima remark is incomplete (e-flatness of $M$ does not remove local maxima).

| Term | One line |
|---|---|
| hidden variable | $h$ in $x=(y,h)$, not observed; visible model $p_Y=\sum_hp(y,h;\xi)$ |
| data manifold $D$ | all $\bar q_Y(y)q(h\mid y)$: tables of responsibilities; m-flat |
| E-step | e-projection of $p_\xi$ onto $D$: the posterior; $F(R^*,\xi)=-\ell/N$ |
| M-step | m-projection of $R$ onto $M$: moment matching in $\eta$; (8.26) for the mixture |
| Theorem 8.2 | each EM step gains $\mathrm{KL}[p_{t+1}\Vert p_t]+\tfrac1N\sum_i\mathrm{KL}[\text{post}_t\Vert\text{post}_{t+1}]$ in $\ell/N$ ($M$ e-flat) |
| EM rate | $DM=I-(g^X)^{-1}g^Y=(g^X)^{-1}g^{X\mid Y}$: the fraction of missing information |
| $g^X=g^T+g^{X\mid T}$ | the score splits orthogonally; loss $=g^{X\mid T}=E_T\mathrm{Cov}(\text{score}\mid T)$ |
| q-MLE | m-projection of the observed point onto $M_q$ along q-ancillary leaves; limit $v^*$ with $E_p[\partial\log q(v^*)]=0$ |
| (8.42) | loss $=g_{a\kappa}g_{b\lambda}g^{\kappa\lambda}$ ($g^{\kappa\lambda}$ the block inverse) $=g_{uu}(1-\rho^2)$, $\rho$ the correlation of true and model scores |
| ignoring correlation | efficiency $\lvert r'\rvert^4/(r'^{\mathsf T}Vr'\cdot r'^{\mathsf T}V^{-1}r')$; 1 iff $r'$ is an eigenvector of $V$ |

---

*Notes written 2026-10-02.*
