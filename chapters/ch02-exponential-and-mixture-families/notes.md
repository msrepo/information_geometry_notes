---
title: "Chapter 2: exponential families and mixture families"
short_title: "Ch. 2 — Exponential & mixture families"
chapter: 2
category: "Part I"
book_pages: "31–50"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [exponential-family, cumulant-generating-function, gaussian, natural-parameter]
status: reading
---

Notes for Chapter 2 of Amari, *Information Geometry and Its Applications* (Springer, 2016), DOI [10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8). No text of the book is reproduced here. This page keeps only the parts I asked for: the picture of a Gaussian family and its cumulant generating function, and the derivation of that function.

Previous chapter: [Chapter 1: the dually flat structure](../ch01-dually-flat-structure/index.html).

## A Gaussian family and its cumulant generating function

The chapter's opening line says that an exponential family is *associated with* a convex function, the cumulant generating function $\psi$ (the "free energy"). Two pictures show what that means for Gaussians.

**One parameter: only the mean moves.** Fix the width and slide the mean. Each member of the family is a bell curve, and each is one point on the curve of $\psi$.

<figure>
<img src="figures/gaussian-cgf-1d.svg" alt="Left: four bell curves of equal width with different means. Middle: a convex bowl, psi of theta, with four coloured dots matching the four bell curves and a dashed tangent line at one of them. Right: the same bowl for three different widths; a wider Gaussian gives a more sharply curved bowl.">
<figcaption>Left: a one-parameter Gaussian family. Middle: its cumulant generating function $\psi(\theta)$; each dot is the bell curve of the same colour, and the slope at a dot is that curve's mean. Right: the same function for three widths: the wider the Gaussian, the more curved the bowl.</figcaption>
</figure>

The function does the bookkeeping for the whole family. Its **slope** at a point is the mean of the distribution there, and its **curvature** is the variance. So a single convex function holds both the average and the spread of every member, and this is the reason the Hessian of $\psi$ is the Fisher information (the Fisher information).

**Where $\psi$ comes from, for the one-parameter family.** Fix the width $\sigma$ and start from one reference density, the zero-mean Gaussian $h(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-x^2/(2\sigma^2)}$. The family is a *tilt* of $h$ by $e^{\theta x}$, renormalised:

$$p(x;\theta)=h(x)\,e^{\theta x-\psi(\theta)} .$$

*Step 1: normalisation defines $\psi$.* The tilted function $h\,e^{\theta x}$ is generally not a probability density, so we divide it by a constant that depends on $\theta$ but not on $x$; writing that constant as $e^{\psi(\theta)}$ is just naming the normaliser on a log scale. Requiring $\int p\,dx=1$ and pulling the $x$-independent factor $e^{-\psi(\theta)}$ out of the integral gives
$e^{\psi(\theta)}=\int h(x)\,e^{\theta x}dx$, that is

$$\psi(\theta)=\log\mathbb E_h\!\left[e^{\theta x}\right].$$

This is the log of the moment generating function of $h$, hence the name *cumulant generating function* (in physics, the log of the partition function, the "free energy"). Nothing here is Gaussian yet: the form $h\,e^{\theta x-\psi}$ is the *definition* of an exponential family, and $\psi$ is forced by normalisation.

*Step 2: complete the square.* The Gaussian enters only through $h$. Combine the exponents:
$\theta x-\frac{x^2}{2\sigma^2}=-\frac{(x-\sigma^2\theta)^2}{2\sigma^2}+\frac{\sigma^2\theta^2}{2}$.
So $\int h\,e^{\theta x}dx=e^{\sigma^2\theta^2/2}\times\big(\text{a full Gaussian density with mean }\sigma^2\theta\big)$, and the density integrates to 1.

*Step 3: the answer.*

$$\psi(\theta)=\tfrac12\,\sigma^2\theta^2 .$$

For unit width this is the parabola $\theta^2/2$ in the middle panel of the figure. Reading it back:

- **The members are Gaussians.** Tilting a zero-mean Gaussian by $e^{\theta x}$ only shifts it: $p(x;\theta)=\mathcal N(\sigma^2\theta,\ \sigma^2)$.
- **Slope is the mean:** $\psi'(\theta)=\sigma^2\theta=\mu$, so $\theta=\mu/\sigma^2$, the first natural parameter of the full family below.
- **Curvature is the variance:** $\psi''(\theta)=\sigma^2$, a constant, so the Fisher information is the same everywhere along this family, and wider Gaussians give a more curved bowl (right panel).
- **Why the bowl is an exact parabola.** The Taylor coefficients of a cumulant generating function are the cumulants. A Gaussian has only a mean and a variance, and every higher cumulant vanishes, so the series stops at the quadratic term. A non-Gaussian family has higher-order terms and a bowl that is not exactly a parabola.

**Two parameters: the full Gaussian family.** Let the mean and the width both vary. The natural parameters are $\theta_1=\mu/\sigma^2$ and $\theta_2=-1/(2\sigma^2)$, so the allowed region is only the half-plane $\theta_2<0$.

<figure>
<img src="figures/gaussian-cgf-2d.svg" alt="Left: the plane of the two natural parameters with a forbidden edge at the top and four coloured points. Middle: the four Gaussians that the points stand for. Right: the cumulant generating function drawn as a bowl over the parameter plane, rising steeply toward the forbidden edge, with the four coloured points on the bowl.">
<figcaption>Left: the parameter plane; each point is one Gaussian, and the edge $\theta_2=0$ is not allowed. Middle: the Gaussians those points stand for. Right: the cumulant generating function as a bowl over the plane; each Gaussian is a coloured point on it.</figcaption>
</figure>

Three things to see. First, the function is a bowl over a **half-plane**, not the whole plane: the parameters are only meaningful while the variance is positive. Second, the bowl climbs without limit as a point approaches the forbidden edge, which is where the Gaussian becomes infinitely wide. Third, the same recipe works as before: the slope of the bowl (now a two-component gradient) gives the expectation parameters $(\mu,\ \mu^2+\sigma^2)$, and its curvature (now a matrix) gives the covariance of $(x,x^2)$, the Fisher matrix.

## A kernel exponential family, the simplest case

An ordinary exponential family multiplies a few fixed functions of $x$ (such as $x$ and $x^2$) by weights $\theta_i$, adds them up in the exponent and normalises. The book's **kernel exponential family** (§2.6, due to Fukumizu) does the same with *bumps* $k(x,y)$, one around every point $y$, and a whole function $\theta(y)$ of weights, one for each point $y$:

$$p(x;\theta)=\exp\Big\{\int\theta(y)\,k(x,y)\,dy-\psi[\theta]\Big\}\quad\text{with respect to a base measure }d\mu(x).$$

Here $k$ is a positive-definite kernel, for example the Gaussian kernel $k_\sigma(x,y)\propto e^{-(x-y)^2/(2\sigma^2)}$ with width $\sigma$, and the base measure is a suitable fixed one, for instance $d\mu(x)=e^{-x^2/(2\tau^2)}dx$. The natural parameter is the function $\theta(y)$ (instead of a list $\theta_i$), the dual parameter is the function $\eta(y)=\mathbb E[k(x,y)]$, the average of the bump at $y$, and $\psi[\theta]$ is a convex functional of $\theta$. The book stresses that this family does **not** cover every density: there are many such models, one for each choice of $k$ and $d\mu$. The "naive treatment" of the space of all densities in §2.5 is the special case where the kernel is a delta function, $k(x,y)=\delta(x-y)$.

**The simplest picture: two centres.** Replace the continuum of points $y$ by just two centres $c_1,c_2$, so the weight function is concentrated on them, $\theta(y)=\theta_1\delta(y-c_1)+\theta_2\delta(y-c_2)$. The integral becomes a sum and the family is

$$p(x;\theta)\;\propto\;\exp\{\theta_1k(x,c_1)+\theta_2k(x,c_2)\}\quad(\text{times the base measure}),$$

an ordinary two-parameter exponential family whose two "statistics" are the two bumps. The figure draws it with a uniform base measure on a fixed interval, to keep it simple; the book's example base measure is the Gaussian-weighted one above.

<figure>
<img src="figures/kernel-exponential-family.svg" alt="Left: two bump-shaped kernels, one solid and one dashed, centred at two points on a line. Middle: four densities built by weighting the two bumps and exponentiating: a flat grey one for zero weights, a blue one with a peak at the left centre, a green one with a peak at each centre, and an orange one with a tall peak at the right centre and a dip at the left. Right: the plane of the two weights, with four coloured points that are the four densities.">
<figcaption>Left: two kernel bumps. Middle: densities built from them; each curve is one choice of weights. Right: the two weights are the parameters, and each coloured point is the density of the same colour in the middle panel.</figcaption>
</figure>

Read the picture from left to right. The two bumps are fixed once and for all. A choice of the two weights, one point in the right panel, gives one density in the middle panel. Zero weights give the flat grey density. A positive weight on a bump piles probability near its centre (blue, left bump), positive weights on both give two peaks (green), and a negative weight on one bump pushes probability away from its centre (orange has a dip at the left centre). So the weights are the natural parameters, the dual parameters are the averages $\eta_i=\mathbb E[k(x,c_i)]$ of the two bumps, and everything said about exponential families applies: $\psi$ is convex, its slope gives the averages of the bumps and its curvature gives how they co-vary. With many centres the same construction becomes the function-valued $\theta(y)$ of the book.

## The maximum entropy principle (§2.8.1)

**The question.** You do not know a distribution on the outcomes, but you do know the average of some quantities of the outcome, for instance the average face value of a die. Many distributions agree with what you know. Which one should you pick? The **maximum entropy principle** says: with nothing else to go on, pick the one with the largest entropy, the one that assumes the least beyond what you know.

In the book's notation there are $k$ functions $c_1(x),\dots,c_k(x)$ of the outcome, and the known averages are $\mathbb E[c_i(x)]=a_i$.

<figure>
<img src="figures/maxent-projection.svg" alt="Left: the triangle of distributions on three outcomes with the uniform distribution P0 at the centre and grey contour lines of equal entropy around it. A straight orange line M(a) is the sheet of distributions with a prescribed average. It touches one entropy contour at the point P-hat, which lies on a blue curve running from the uniform distribution, the family of maximum-entropy points. Another distribution P is marked on the orange line. Right: the entropy along the orange line, rising to one peak at P-hat, with P lower on the curve.">
<figcaption>Left: all distributions on three outcomes. The orange line holds those that satisfy the prescribed average; the grey contours are lines of equal entropy around the uniform distribution $P_0$. The orange line touches the highest reachable contour at $\hat P$. Right: along the orange line the entropy has one peak, at $\hat P$.</figcaption>
</figure>

**Step 1: the constraints cut out a flat sheet.** All distributions with the prescribed averages form a sheet $M(a)$ inside the set of all distributions, of dimension $n-k$ when there are $n+1$ outcomes and $k$ constraints. It is **m-flat** (straight in the probabilities, the orange line in the picture), because a mixture of two distributions that both have the prescribed averages has them too.

**Step 2: entropy is a distance to the uniform distribution.** The uniform distribution $P_0$ has the largest entropy of all, and its natural parameters are zero, $\theta_0=0$. The book shows that the divergence $D_{\mathrm{KL}}[P:P_0]$ is the negative entropy of $P$ plus a constant. So maximising entropy is the same as getting as close as possible to the uniform distribution in KL divergence: in the picture, reaching the highest entropy contour that the sheet can touch.

**Step 3: Pythagoras picks the point.** Let $\hat P$ be the point of $M(a)$ with the largest entropy. For every other $P$ in the sheet,

$$D_{\mathrm{KL}}[P:P_0]=D_{\mathrm{KL}}[P:\hat P]+D_{\mathrm{KL}}[\hat P:P_0].$$

This is the generalised Pythagorean theorem of Chapter 1. The second term is the same for every $P$, and the first is positive unless $P=\hat P$, so $\hat P$ is the closest point to the uniform distribution. Geometrically, $\hat P$ is the **e-projection of $P_0$ onto $M(a)$**: the e-geodesic from $P_0$ to $\hat P$ meets the m-flat sheet at a right angle.

**Step 4: the answer is an exponential family.** Change the prescribed averages and the sheet moves, and so does its maximum-entropy point. All these points $\hat P(a)$ together form a $k$-dimensional family, the blue curve, and it is an exponential family:

$$\hat p(x;\theta)=\exp\{\theta\cdot c(x)-\psi(\theta)\}.$$

In words: the maximum-entropy distribution for given averages is the uniform distribution tilted by an exponential of the quantities you constrained. The tilts $\theta$ are the Lagrange multipliers of the constraints. The book's variational derivation (maximising entropy under the constraints) gives the same result.

**A note on the book's wording.** The text says the natural coordinates of this family are "specified by $\theta=a$". As I read it, the constraints fix the *expectation* parameters, $\eta=a$, and $\theta$ is the dual coordinate that goes with them, so I take the printed sentence to be a slip.
