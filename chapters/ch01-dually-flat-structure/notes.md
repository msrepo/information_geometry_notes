---
title: "Chapter 1: manifolds, divergences and the dually flat structure"
short_title: "Ch. 1 — Dually flat structure"
chapter: 1
category: "Part I"
book_pages: "3–30"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [information-geometry, manifold, chart, kl-divergence, fisher-metric, bregman-divergence, legendre-transform, geodesic, pythagorean-theorem, projection]
status: read
---

Notes for Chapter 1 of Amari, *Information Geometry and Its Applications* (Springer, 2016). This page keeps only the parts I asked for, in the order of the chapter. Each piece was added when I asked a question, so the sections below are not a summary of the whole chapter.

## Links

- **[Interactive companion](figures/interactive.html)**: eleven widgets. (1) One Gaussian in three coordinate charts, with the grid bent by the change of labels.
  (2) The KL divergence between two Gaussians, both orders, with the integrand that adds up to it. (3) Tangent vectors and the Fisher metric on the Gaussians:
  Fisher ellipses over the half-plane, a tangent step, its Riemannian length and the KL it costs. (4) What a positive-definite Hessian looks like: drag a 2×2 matrix
  and watch the ellipse, eigen-directions and curvature by direction. (5) The Bregman divergence as the gap above a tangent, for five choices of the convex function.
  (6) The Legendre transform: slope as a coordinate and the tangent's intercept as the dual function. (7) Two kinds of straight line on the three-outcome probability
  triangle: drag two distributions and watch the e-geodesic (straight in the logits), the m-geodesic (straight in the probabilities) and the Fisher–Rao geodesic part ways.
  (8) Pythagoras on the Gaussians: the 3-4-5 triangle's cousin, stacking two KL divergences against the direct one, with a tilt away from the right angle.
  (9) Build a right-angled triangle on the probability triangle and tilt it to see the Pythagorean theorem fail by exactly the amount the proof predicts.
  (10) Projection onto the zero-mean Gaussians: both feet, both orders of KL, and the Pythagorean split along the family.
  (11) Project a point onto a straight line or a curved arc on the probability triangle, for either order of the KL divergence, and count the critical points.
- **[Runnable checks](https://github.com/msrepo/information_geometry_notes/tree/main/chapters/ch01-dually-flat-structure/code)**:
  `code/dually_flat.py` prints every number on this page and regenerates the figures with
  `python3 code/dually_flat.py --figures`. `make verify` runs it, in about a second.
- The book: Amari, *Information Geometry and Its Applications* (Springer, 2016), DOI
  [10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8). These notes cover Chapter 1 only. Equation numbers
  such as (1.69) refer to the book. No text of the book is reproduced here; everything is restated and re-derived.

## 1. Manifolds and coordinates (§1.1)

### Two questions about what a manifold is

**How is a manifold different from a vector space?** A vector space has algebra built in: you can add two points, scale a point, and there is an origin. One set of coordinates covers everything, and "straight line" and "distance from the origin" mean the same thing everywhere. A manifold promises only that each small neighbourhood can be labelled by $n$ numbers, with nearby points getting nearby labels. There is no addition, no origin and no global straight line, so "the midpoint of $P$ and $Q$" means nothing until you choose a rule: the arithmetic mixture, the geometric mixture or the shortest path, which in general give different points (§5 computes all three for softmax). One chart may not cover the whole set (next question), and relabelling by any smooth invertible map is allowed, because the set does not change.

This is why the softmax outputs of this chapter form a manifold and not a vector space. The probability triangle is not closed under addition (the sum of two distributions has total mass 2). In logit coordinates the same set looks like a flat plane, where straight lines are e-geodesics, while in the triangle the straight lines are m-geodesics; neither chart is "the" right one. Two caveats. Every vector space is a manifold, the simplest kind, so manifold is the more general notion. And at each point of a manifold there is a tangent space, which *is* a vector space: small steps, gradients and the Fisher metric live there, even though the whole set has no such structure.

**Why does a sphere need at least two charts?** A chart is a continuous, invertible labelling of points by numbers in a flat region, continuous in both directions. A globe cannot be flattened onto one map without cutting it or sending points to infinity. Latitude and longitude show the failure:

- at the poles longitude is undefined, because every meridian meets there (two points $0.00025$ apart near the pole can have longitudes $90^\circ$ apart);
- across the $180^\circ$ meridian the label jumps from $+180$ to $-180$ (two points $0.00344$ apart get longitudes that differ by $359.8$).

Dropping those points leaves a chart that no longer covers the sphere. This is not a flaw of one map. The sphere is compact (closed and bounded, with no edge), while a chart has to make it look like an open piece of the plane, and a closed-up set cannot be matched to an open one without cutting: the labels would have to "run out" somewhere, and there nearby points get badly separated labels.

The fix is two overlapping charts. Project stereographically from the north pole, $(x,y,z)\mapsto(x,y)/(1-z)$: this covers everything except the north pole, which goes to infinity. Project from the south pole, $(x,y)/(1+z)$: this covers everything except the south pole. Every point is in at least one chart and most are in both. In the overlap a smooth map converts one label into the other, here $r\mapsto1/r$ (the two radii multiply to $1$; largest error $4.6\times10^{-14}$ over $2000$ random points). A collection of charts that cover the set and agree smoothly in the overlaps is what "manifold" means. In the picture the overlap is the ring of latitudes between $-60^\circ$ and $60^\circ$, which is the annulus $0.2679<r<3.7321$ in either chart. For the point $P$ at latitude $30^\circ$, longitude $-150^\circ$ the labels are $(-1.5000,-0.8660)$ in the north chart and $(-0.5000,-0.2887)$ in the south chart; round trips back to the sphere agree to $2.7\times10^{-15}$. Across the $180^\circ$ meridian, where longitude jumps, the two charts put the same two points $0.00416$ and $0.00293$ apart, with no jump.

<img src="figures/sphere-charts.svg" alt="Left: a sphere with latitude and longitude lines; the north pole is marked where all meridians meet and longitude is undefined, and the 180 degree meridian is dashed because the label jumps there. Middle and right: stereographic projections from the north and the south pole, each a disc-like plane missing one point (sent to infinity); the ring of latitudes between minus 60 and 60 degrees is shaded as the overlap, where the two labels are related by r to 1/r. The point at latitude 30, longitude minus 150 has radius 1.7321 in the north chart and 0.5774 in the south chart.">

Contrast with the examples below. The $(\mu,\sigma)$ half-plane is covered by one chart, because that set does not close up on itself; the same holds for the open probability simplex. One chart is enough for some manifolds, and the sphere is simply not one of them.

### Worked example: the Gaussians in three charts

The *points* of this manifold are Gaussian curves. A *chart* is a way of naming one such curve with two numbers. Three
namings are in common use.

- **Chart 1, $(\mu,\sigma)$**: where the bell sits and how wide it is. The only restriction is $\sigma>0$, so the chart is
  a half-plane.
- **Chart 2, moments $(m_1,m_2)=(\mu,\mu^2+\sigma^2)$**: the two averages $\mathbb E[x]$ and $\mathbb E[x^2]$. Not every
  pair is a Gaussian: $m_2-m_1^2=\sigma^2>0$ confines the chart to the region above the parabola $m_2=m_1^2$.
- **Chart 3, natural parameters $\theta=(\mu/\sigma^2,\,-1/(2\sigma^2))$**: write the density as
  $\exp\{\theta_1x+\theta_2x^2-\psi(\theta)\}$, so $\theta_1,\theta_2$ are the coefficients of $x$ and $x^2$ in the exponent
  (this is the exponential-family form of §3). The restriction is $\theta_2<0$, which is $\sigma>0$ again.

Numbers (from `code/dually_flat.py`): the Gaussian $(\mu,\sigma)=(1,2)$ is $(m_1,m_2)=(1,5)$ and $\theta=(0.25,-0.125)$;
$(0,1)$ is $(0,1)$ and $(0,-0.5)$; $(-1.5,0.5)$ is $(-1.5,2.5)$ and $(-6,-2)$. Going back, $\mu=m_1$, $\sigma^2=m_2-m_1^2$,
and $\sigma^2=-1/(2\theta_2)$, $\mu=\theta_1\sigma^2$; the script recovers $(\mu,\sigma)$ from both charts at all three points.

<img src="figures/gaussian-charts.svg" alt="The coordinate grid of the (mu, sigma) half-plane (blue: constant mu, orange: constant sigma) redrawn in moment coordinates, where the lines become vertical segments and parabolas, and in natural coordinates, where they fan out and the sigma lines bunch toward theta2 = 0. The Gaussian (1, 2) is the point (1, 5) in moments and (0.25, -0.125) in natural parameters.">

The picture is the point of the example. Nothing about the *set* of Gaussians changes between the panels; only the labels do,
and the grid of chart 1 gets bent. Equal steps of $\sigma$ at $\mu=1$ ($\sigma=0.5,1,2,3$) give $m_2=1.25,\,2,\,5,\,10$ in chart 2
but $\theta_2=-2,\,-0.5,\,-0.125,\,-0.0556$ in chart 3, which squeezes everything toward $\theta_2=0$. So **distances read off a
chart are not real distances**; a chart is only a naming. Making "distance" chart-independent is the job of the metric in
§2, and it is why straightness and convexity, which *do* depend on the chart, must be fixed by choosing one on purpose.

Why chart 2 and chart 3 are called a dual pair: the normaliser of the density in chart 3 is
$\psi(\theta)=-\theta_1^2/(4\theta_2)+\tfrac12\log(-\pi/\theta_2)$, and its gradient is exactly $(m_1,m_2)$. Each chart is the slope
of a convex function of the other (the Legendre duality of §4).

The interactive companion's first widget lets you move $\mu$ and $\sigma$ and watch the same Gaussian, and the bent grid, in all three charts.

### Worked example: the softmax in four charts

The same exercise for the running example. A 3-class softmax turns three real numbers $z=(z_0,z_1,z_2)$, the logits, into probabilities $p_k=e^{z_k}/\sum_je^{z_j}$. The *points* of this manifold are the distributions $p$. The logits are one naming of them, and a redundant one: adding the same number to all three logits changes no probability, so a network can move its logits along a whole line without changing its output. The redundancy can be removed in more than one way, and the probabilities are another naming altogether. Four are worth knowing.

- **Chart 1, logits relative to outcome 0**, $\theta=(z_1-z_0,\;z_2-z_0)=\bigl(\log\tfrac{p_1}{p_0},\,\log\tfrac{p_2}{p_0}\bigr)$: pin the first logit to 0. These are the natural parameters of §3 (the distribution is $\exp\{\theta_1x_1+\theta_2x_2-\psi(\theta)\}$, with $x$ the indicator vector of the outcome), and every pair of real numbers is allowed, so the chart is the whole plane.
- **Chart 2, centred logits** $z-\bar z$, $\bar z=\tfrac13(z_0+z_1+z_2)$: remove the redundancy by subtracting the mean instead, so that no outcome is special. The three numbers add to zero, so they live in a plane, which the picture draws oriented like the probability triangle. It is a linear image of chart 1, so straight lines stay straight; what shows up is the three-fold symmetry that the choice of a reference outcome hid.
- **Chart 3, probabilities** $(p_0,p_1,p_2)$, or $\eta=(p_1,p_2)$ once $p_0=1-p_1-p_2$ is eliminated: the expectation parameters, confined to the triangle $p_k>0$.
- **Chart 4, square roots** $(x_1,x_2)=(2\sqrt{p_1},\,2\sqrt{p_2})$: with $x_0=2\sqrt{p_0}$ the point $(x_0,x_1,x_2)$ lies on the sphere of radius 2, and the chart is the octant of that sphere seen from above, a quarter disc. The Fisher metric becomes the ordinary length on the sphere, which is why §5 uses this chart for the Fisher–Rao geodesic.

Numbers (from `code/dually_flat.py`): $P=(0.7,0.2,0.1)$, the $P$ of the geodesic figure in §5, is $\theta=(-1.2528,-1.9459)$ in chart 1, the centred logits $(1.0662,-0.1865,-0.8797)$ in chart 2, itself in chart 3 and $(0.8944,0.6325)$ in chart 4. The uniform distribution is the origin $\theta=(0,0)$, the centred logits $(0,0,0)$, and $(1.1547,1.1547)$ in chart 4. Going back, every chart returns $p$ to $1.1\times10^{-16}$ for $P$, for $Q=(0.1,0.3,0.6)$, for the uniform distribution and for a saturated output, $p=(0.0179,0.9796,0.0024)$ from the logits $(0,4,-2)$, which is $\theta=(4,-2)$ and centred logits $(-0.6667,3.3333,-2.6667)$. Shifting the logits $(0.3,-1.2,2.0)$ by $7.3$ changes the softmax by at most $1.7\times10^{-16}$.

<img src="figures/softmax-charts.svg" alt="The coordinate grid of the logits (theta1, theta2) = (z1 - z0, z2 - z0) of a 3-class softmax, with equal steps of one logit (blue: constant theta1, orange: constant theta2), redrawn in centred logits, where it stays a straight lattice with 60 degree angles, in probabilities, where the lines become two fans of straight lines through two corners of the triangle, and in square-root coordinates, where they become two fans of ellipse arcs in a quarter disc. The distribution P = (0.7, 0.2, 0.1) is theta = (-1.2528, -1.9459), centred logits (1.0662, -0.1865, -0.8797) and square roots (0.8944, 0.6325).">

The picture is the counterpart of the Gaussian one: nothing about the *set* of distributions changes between the panels, only the labels do, and the grid of chart 1 (equal steps of one logit) is bent. The bending has a concrete meaning.

**Saturation.** The lines $\theta_1=c$ (blue) are, in chart 3, straight lines through the corner of outcome 2, because $\theta_1=c$ means $p_1=e^cp_0$. They meet the opposite edge at $p_1=1/(1+e^{-c})=0.0474,\,0.1192,\,0.2689,\,0.5000,\,0.7311,\,0.8808,\,0.9526$ for $c=-3,\dots,3$: equal steps of a logit crowd towards the corners. Along $\theta_2=0$ the logits $\theta_1=-4,-2,0,2,4$ give $p_1=0.0091,\,0.0634,\,0.3333,\,0.7870,\,0.9647$, so the four steps of two logits move the probability by $0.0543,\,0.2700,\,0.4537$ and $0.1777$. Near the centre a small change of a logit is a large change of the output, near the boundary the output hardly responds: that is the saturation of the softmax. The area scale between charts 1 and 3 is $\det G=p_0p_1p_2$ (the finite-difference Jacobian agrees to the six digits printed): $0.037037$ at the uniform distribution, $0.014000$ at $P$, $0.005781$ at $\theta=(3,3)$ and $0.000043$ at the saturated output, so a unit square of logits covers far less probability where the softmax is saturated.

**What does not bend.** Length measured with the metric. The step $dp=(-0.02,0.05,-0.03)$ at $P$ has squared length $ds^2=0.022071$ whichever chart computes it, each with its own metric: $G_\theta=\operatorname{diag}(\eta)-\eta\eta^\top$ in chart 1, the covariance of the logit change under $p$ in chart 2, $G^*=\operatorname{diag}(1/\eta_i)+\tfrac1{p_0}\mathbf 1\mathbf 1^\top$ in chart 3 and $I+xx^\top/x_0^2$ in chart 4. The four numbers agree with $\sum_kdp_k^2/p_k$ to $10^{-17}$. Cells of equal size in one panel are not equal lengths; the metric of §2, not the picture, says how far apart two distributions are.

**Lines that are straight in two charts.** The blue lines are straight in charts 1 and 3. In chart 4 they are great circles through the corner of outcome 2: $x_1=e^{c/2}x_0$ is a plane through the origin (residual $2.2\times10^{-16}$ along the line $c=1.2$), and the arcs of ellipses that the quarter disc shows are the projection of those circles, $(1+e^c)x_1^2+e^cx_2^2=4e^c$ (residual $3.6\times10^{-15}$). As sets, these particular lines are therefore e-geodesics (straight in $\theta$), m-geodesics (straight in $\eta$: mixtures of a fixed pair of outcomes with a point mass on outcome 2) and Fisher–Rao geodesics all at once; as paths with a speed they differ, the e-geodesic being affine in $\theta_2$ and the m-geodesic in the mixing weight. A generic e-geodesic, like the blue curve of the figure in §5, bends in chart 3. The orange lines do the same with outcome 1, and the third family, $z_1-z_2=$ const, with outcome 0; chart 2 treats the three alike.

## 2. Divergence (§1.2)

**What is asked of $D$.** Three things: $D[P{:}Q]\ge0$; $D[P{:}Q]=0$ only if $P=Q$; and for $Q=P+d\xi$ the function
is, to second order, a positive-definite quadratic form in $d\xi$:

$$
D[\xi:\xi+d\xi]=\tfrac12\,g_{ij}(\xi)\,d\xi^i d\xi^j+O(|d\xi|^3),\qquad G=(g_{ij})\succ0 .
$$

**Why the first-order term is missing.** For fixed $P$, the map $Q\mapsto D[P{:}Q]$ is $\ge0$ and equals $0$ at $Q=P$, so $P$ is a
minimum and the gradient there is zero. What is left at second order is the Hessian, which is a
quadratic form; condition (3) just says it is positive-definite rather than merely positive semi-definite. The
metric is *defined* by $ds^2=2D[\xi:\xi+d\xi]=g_{ij}d\xi^id\xi^j$, with the factor 2 chosen so that
$D=\tfrac12\|\Delta\|^2$ gives the identity matrix.

**A worked example: KL between two Gaussians.** Fix $p=N(0,1)$ and let $q=N(m,\sigma^2)$ move. Then
$\mathrm{KL}[p{:}q]=\int p\log\frac pq\,dx=\log\sigma+\frac{1+m^2}{2\sigma^2}-\frac12$, and the other order is
$\mathrm{KL}[q{:}p]=-\log\sigma+\frac{\sigma^2+m^2}{2}-\frac12$. What you can see by moving $q$ (the interactive page's second widget):

- *The integrand can be negative, the integral cannot.* The integrand $p\log(p/q)$ is positive where $q<p$ and negative where $q>p$.
  For $q=N(1,0.5^2)$ it dips as low as $-0.2896$, yet the total is $2.8069>0$ (the numeric integral agrees to four decimals).
- *Zero only at equality, quadratic nearby.* With equal widths a shift $e$ costs exactly $e^2/2$ in both orders
  ($0.0050$ at $e=0.1$, $1.1250$ at $e=1.5$). That is (1.24) with $g=1$, the Fisher information of the mean at $\sigma=1$.
- *Asymmetric as soon as the widths differ.* $q=N(1,0.5^2)$ gives $\mathrm{KL}[p{:}q]=2.8069$ but $\mathrm{KL}[q{:}p]=0.8181$; $q=N(0,2^2)$ gives
  $0.3181$ and $0.8069$. $\mathrm{KL}[p{:}q]$ punishes a $q$ that is too narrow where $p$ has mass; $\mathrm{KL}[q{:}p]$ punishes the reverse.

**Tangent spaces and the metric, on the Gaussian manifold.** Intuition first. A *tangent vector* at a point is a velocity: pass a smooth curve of Gaussians through $N(\mu,\sigma^2)$ and record how fast the
mean and the width are changing, $v=(\dot\mu,\dot\sigma)$. Collect all such velocities and you get the **tangent space** at that point, a flat plane attached to it. Every point has its own plane; they are
copies of $\mathbb R^2$, but a vector in one plane is not a vector in another until you say how to compare them. A **Riemannian metric** is a rule for measuring the length of the vectors in each plane,
$|v|^2=g_{ij}v^iv^j$, changing smoothly from point to point. Here it comes from the divergence, $ds^2=2D$, and for the Gaussians in the chart $(\mu,\sigma)$ it is

$$
g=\begin{bmatrix}1/\sigma^2&0\\0&2/\sigma^2\end{bmatrix},\qquad ds^2=\frac{d\mu^2+2\,d\sigma^2}{\sigma^2}.
$$

Why this shape: whether a change in the mean matters depends on how wide the bell is. A shift $d\mu$ is hard to notice when $\sigma$ is large and easy when $\sigma$ is small, so the same Euclidean
step is *longer* where the Gaussian is narrow. Numbers from `code/dually_flat.py`: a Euclidean step of length $0.2$ along $\mu$ has Riemannian length $0.4000$ at $\sigma=0.5$, $0.2000$ at $\sigma=1$ and $0.1000$ at $\sigma=2$.
At $(\mu,\sigma)=(1,2)$ the metric is $\operatorname{diag}(0.25,0.5)$ and the set of steps of unit Riemannian length (the *indicatrix*) is an ellipse with half-axes $2.0000$ along $\mu$ and $1.4142$ along $\sigma$;
at $(1,0.5)$ it is $\operatorname{diag}(4,8)$ with half-axes $0.5000$ and $0.3536$. The ellipses shrink as $\sigma\to0$, which is the picture on the interactive page.

The length is what you pay in divergence: for a small step $d$, $\mathrm{KL}\approx\tfrac12 d^\top g\,d$. At $(1,2)$ along the direction $(0.6,-0.8)$ the ratio $\mathrm{KL}/(\tfrac12ds^2)$ is $2.4540$ for a step of length 1,
then $1.0737,\ 1.0070,\ 1.0007$ as the step is scaled by $0.1,\ 0.01,\ 0.001$: it tends to 1, with the error shrinking linearly in the step (the next term in the expansion is cubic). Large steps, especially near small $\sigma$ (at $(1,0.5)$ a step of $(0.12,-0.16)$ gives ratio
$1.9660$), are far from the quadratic regime.

The length does not depend on the chart. The step $(0.12,-0.16)$ at $(1,2)$ has $ds^2=0.0164$ in $(\mu,\sigma)$; in natural coordinates it becomes $d\theta=J\,d=(0.0700,-0.0200)$ and
$d\theta^\top G_\theta d\theta=0.0164$ with $G_\theta=\nabla^2\psi$: the same number. That is the tensor law (1.130) at work: $g$ changes with the chart, but the *length of the same tangent vector* does not.
Euclidean length is chart-dependent and misleading: the step above has Euclidean length $0.2000$ and Riemannian length $0.1281$ at $\sigma=2$, $0.5122$ at $\sigma=0.5$.

**Deriving the KL divergence between two Gaussians.** Let $p=N(\mu,\sigma^2)$ and $q=N(\mu+a,(\sigma+b)^2)$; write $\mu'=\mu+a$ and $\sigma'=\sigma+b$. KL is the average, under $p$, of the log-ratio of the densities, so start by writing the two log-densities:

$$
\log p(x)=-\ln\!\big(\sigma\sqrt{2\pi}\big)-\frac{(x-\mu)^2}{2\sigma^2},\qquad
\log q(x)=-\ln\!\big(\sigma'\sqrt{2\pi}\big)-\frac{(x-\mu')^2}{2\sigma'^2}.
$$

Subtracting, the $\sqrt{2\pi}$ cancels:

$$
\log\frac{p(x)}{q(x)}=\ln\frac{\sigma'}{\sigma}-\frac{(x-\mu)^2}{2\sigma^2}+\frac{(x-\mu')^2}{2\sigma'^2}.
$$

Now take the expectation under $p$, where $x$ has mean $\mu$ and variance $\sigma^2$. The constant $\ln(\sigma'/\sigma)$ stays. The middle term uses $\mathbb E_p[(x-\mu)^2]=\sigma^2$, giving $-\sigma^2/(2\sigma^2)=-\tfrac12$.
For the last term, split $x-\mu'=(x-\mu)+(\mu-\mu')=(x-\mu)-a$; the cross term has mean zero, so
$\mathbb E_p[(x-\mu')^2]=\sigma^2+a^2$. Putting the three pieces together,

$$
\boxed{\ \mathrm{KL}[p{:}q]=\ln\frac{\sigma'}{\sigma}+\frac{\sigma^2+a^2}{2\sigma'^2}-\frac12
=\ln\frac{\sigma+b}{\sigma}+\frac{\sigma^2+a^2}{2(\sigma+b)^2}-\frac12\ }
$$

which is the formula used in the expansion below. Check at $p=N(1,2^2)$, $q=N(1.6,1.2^2)$ ($a=0.6$, $b=-0.8$): the three terms are $-0.5108$, $-0.5000$ and $1.5139$, summing to $0.5031$; numerical integration of $\int p\log(p/q)$ gives $0.5031$. The two expectations
also check: $\mathbb E_p[(x-\mu)^2]=4.0000=\sigma^2$ and $\mathbb E_p[(x-\mu')^2]=4.3600=\sigma^2+a^2$. Swapping the roles gives the other order,
$\mathrm{KL}[q{:}p]=\ln\frac{\sigma}{\sigma'}-\frac12+\frac{\sigma'^2+a^2}{2\sigma^2}$, which is $0.2358$ here (quadrature agrees), against $0.5031$: the asymmetry in numbers.

Sanity checks on the formula. *Identical Gaussians* ($a=b=0$): $0+\tfrac12-\tfrac12=0$. *Equal widths* ($b=0$): $\mathrm{KL}=a^2/(2\sigma^2)$, half the squared shift measured in units of the width ($0.0450$ for $a=0.6$, $\sigma=2$, matching the formula).
*Equal means* ($a=0$), with $r=\sigma'/\sigma$: $\mathrm{KL}=\ln r+\frac1{2r^2}-\frac12$, which is $0.8069,\ 0.0000,\ 0.3181,\ 0.9175$ at $r=0.5,\ 1,\ 2,\ 4$: zero only at $r=1$, and bigger for a too-narrow $q$ ($r<1$) than for a too-wide one of the same ratio
(compare $r=0.5$ with $r=2$). That last point is the asymmetry that §2 describes in words: $\mathrm{KL}[p{:}q]$ punishes a $q$ that is narrower than $p$.

**Deriving $g$, $ds^2$ and the length of a tangent vector.** *Velocity.* A path of Gaussians $t\mapsto(\mu(t),\sigma(t))$ has velocity $v=(\dot\mu,\dot\sigma)$ at $t=0$, with components in the basis
$\partial/\partial\mu,\ \partial/\partial\sigma$. *Metric.* By definition $ds^2=2D[\xi{:}\xi+d\xi]$. For the step $(\mu,\sigma)\to(\mu+a,\sigma+b)$ the KL divergence has the closed form
$\mathrm{KL}=\ln\frac{\sigma+b}{\sigma}+\frac{\sigma^2+a^2}{2(\sigma+b)^2}-\frac12$. Expand each piece to second order:

$$
\ln\!\Big(1+\frac b\sigma\Big)=\frac b\sigma-\frac{b^2}{2\sigma^2}+\cdots,\qquad
\frac{\sigma^2+a^2}{2(\sigma+b)^2}=\frac12+\frac{a^2}{2\sigma^2}-\frac b\sigma+\frac{3b^2}{2\sigma^2}+\cdots
$$

(the term $a^2b$ is cubic and dropped). Adding them and subtracting $\tfrac12$, the $b/\sigma$ terms cancel:

$$
\mathrm{KL}\approx\frac{a^2}{2\sigma^2}+\frac{b^2}{\sigma^2}=\tfrac12\,\frac{a^2+2b^2}{\sigma^2}
\ \Longrightarrow\ g=\begin{bmatrix}1/\sigma^2&0\\0&2/\sigma^2\end{bmatrix},\quad ds^2=\frac{d\mu^2+2\,d\sigma^2}{\sigma^2}.
$$

Check at $(1,2)$: a step $(0.01,0)$ has $\mathrm{KL}=1.250\times10^{-5}$ against the prediction $1.250\times10^{-5}$, and a step $(0,0.01)$ has $2.479\times10^{-5}$ against $2.500\times10^{-5}$ (the $1\%$ gap is the cubic term).
*A second route.* The Fisher information is $g_{ij}=\mathbb E[s_is_j]$ with scores $s_i=\partial_i\log p$. Here $s_\mu=(x-\mu)/\sigma^2$ and $s_\sigma=((x-\mu)^2-\sigma^2)/\sigma^3$, so with
$z=(x-\mu)/\sigma$ standard normal, $\mathbb E[s_\mu^2]=1/\sigma^2$, $\mathbb E[s_\sigma^2]=\mathbb E[(z^2-1)^2]/\sigma^2=2/\sigma^2$ and $\mathbb E[s_\mu s_\sigma]=\mathbb E[z(z^2-1)]/\sigma^2=0$. By quadrature at $(1,2)$:
$0.2500$, $0.5000$ and $-6.6\times10^{-18}$, the same matrix. *Length of a vector.* For $v=(\dot\mu,\dot\sigma)$ at the point $(\mu,\sigma)$,
$|v|^2=v^\top Gv=(\dot\mu^2+2\dot\sigma^2)/\sigma^2$, and the length of a curve is $\int\sqrt{(\dot\mu^2+2\dot\sigma^2)/\sigma^2}\,dt$.

**The Fisher ellipse, and why it shrinks.** The *Fisher ellipse* (indicatrix) at a point is the set of tangent vectors of fixed length $\varepsilon$:
$\dot\mu^2/(\varepsilon\sigma)^2+\dot\sigma^2/(\varepsilon\sigma/\sqrt2)^2=1$, with half-axes $\varepsilon\sigma$ along $\mu$ and $\varepsilon\sigma/\sqrt2$ along $\sigma$. Since $\mathrm{KL}\approx\tfrac12|v|^2$, every step on one ellipse
costs the same divergence $\approx\varepsilon^2/2$: it is the set of changes to the distribution that are equally detectable. For $\varepsilon=0.15$ the half-axes are $(0.0750,0.0530)$ at $\sigma=0.5$, $(0.1500,0.1061)$ at $\sigma=1$
and $(0.3000,0.2121)$ at $\sigma=2$. It shrinks as $\sigma\to0$ for two reasons. In the mean direction a shift $a$ costs $a^2/(2\sigma^2)$: a narrow bell leaves its former self after a small shift, so a fixed cost allows only $a=\varepsilon\sigma$.
In the width direction the cost $b^2/\sigma^2$ depends only on the *relative* change $b/\sigma$, so the allowed $b$ is again proportional to $\sigma$. Said once: in relative units $(d\mu/\sigma,\,d\sigma/\sigma)$, $ds^2=(d\mu/\sigma)^2+2(d\sigma/\sigma)^2$ and the ellipse has half-axes $\varepsilon,\ \varepsilon/\sqrt2$
at *every* $\sigma$ (the check prints $0.1500,\ 0.1061$ for each); the absolute sizes shrink because the unit of measurement, $\sigma$, does. This matches the statistical reading: the mean of $N=100$ samples is known to
$\sigma/\sqrt N$, i.e. $0.050$ at $\sigma=0.5$ and $0.200$ at $\sigma=2$, so shifts smaller than that are invisible. (Writing $ds^2=2\big[(d\mu/\sqrt2)^2+d\sigma^2\big]/\sigma^2$ shows the half-plane is the hyperbolic plane of curvature $-\tfrac12$ in
the coordinates $(\mu/\sqrt2,\sigma)$; the Fisher notes make the same point.)

## 3. Convex functions and the Bregman divergence (§1.3)

**A picture first.** Draw a convex function $\psi$ and its tangent line at a point $\theta_0$. The curve stays above
the line. The vertical gap at another point $\theta$ is the **Bregman divergence**

$$
D_\psi[\theta:\theta_0]=\psi(\theta)-\psi(\theta_0)-\nabla\psi(\theta_0)\cdot(\theta-\theta_0).
$$

It is $\ge0$ by convexity, and it is zero only at $\theta=\theta_0$ if $\psi$ is *strictly* convex. Taylor-expanding $\psi$ about
$\theta_0$ shows $D_\psi[\theta_0+d\theta:\theta_0]=\tfrac12d\theta^\top\nabla^2\psi(\theta_0)\,d\theta+O(|d\theta|^3)$, so the metric of §2 is
the **Hessian**, $g_{ij}=\partial_i\partial_j\psi$ (1.86).

<img src="figures/legendre.svg" alt="Left: the convex function log(1 + e^theta) for one coin with its tangent at theta0 = -0.4 and the gap 0.6508 above the tangent at theta1 = 2. Right: the negative entropy of a coin with its tangent at eta1 = 0.881 and the gap 0.6508 above that tangent at eta0 = 0.401. The two gaps are equal because the Legendre transform swaps the two points.">

**Bregman divergence, step by step, in one dimension.** Pick a convex $\psi$ and two points $x_0$ (the base) and $x$ (the probe).

1. Draw the tangent line of $\psi$ at $x_0$: $\ell_{x_0}(x)=\psi(x_0)+\psi'(x_0)(x-x_0)$.
2. Because $\psi$ is convex it lies on or above its tangent everywhere.
3. The vertical gap at the probe is $D_\psi[x{:}x_0]=\psi(x)-\ell_{x_0}(x)$, "how far above the tangent drawn at $x_0$ is the function at $x$".

Swap the roles (draw the tangent at $x$, measure at $x_0$) and you get $D_\psi[x_0{:}x]$, a different gap unless $\psi$ is a parabola. Numbers
from `code/dually_flat.py`, base $x_0$ and probe $x$:

| $\psi$ | $x_0$ | $x$ | $D_\psi[x{:}x_0]$ | $D_\psi[x_0{:}x]$ | what it is |
|---|---|---|---|---|---|
| $\tfrac12x^2$ | 0.5 | 2 | 1.1250 | 1.1250 | half the squared distance, symmetric |
| $\log(1+e^x)$ | −0.4 | 2 | 0.6508 | 0.5000 | KL between two coins, reversed |
| $-\log x$ | 1 | 3 | 0.9014 | 0.4319 | Itakura–Saito |
| $x\log x$ | 1 | 3 | 1.2958 | 0.9014 | generalised KL |

Near the base the gap is a parabola with curvature $\psi''(x_0)$: at $e=0.1$ the true gap $D_\psi[x_0{+}e{:}x_0]$ against $\tfrac12\psi''(x_0)e^2$ is $0.005000$
vs $0.005000$ for $\tfrac12x^2$, $0.001209$ vs $0.001201$ for $\log(1+e^x)$, $0.004690$ vs $0.005000$ for $-\log x$ and $0.004841$ vs $0.005000$ for $x\log x$
(the last two have a $\psi''$ that changes quickly, so the match is looser at this step). Away from the base the shape is set by the third and higher derivatives, which is where the
asymmetry lives. And if $\psi''(x_0)=0$ there is no parabola at all: for $\psi=x^4$ at $0$, $D=0.0625$ at $x=0.5$ and the gap grows like $x^4$, so it is not a squared length.
The interactive page's fourth widget draws the tangent, both gaps and the parabola for five choices of $\psi$.

**What "positive-definite Hessian" means, concretely.** Near a point, a smooth function is its tangent plane plus a quadratic bowl:
$\psi(\theta+d)\approx\psi(\theta)+\nabla\psi\cdot d+\tfrac12\,d^\top H\,d$ with $H=\nabla^2\psi(\theta)$. For a Bregman divergence the tangent plane
is subtracted off, so only the bowl is left, $D\approx\tfrac12 d^\top Hd$. The matrix $H$ is **positive-definite** when $d^\top Hd>0$ for *every*
direction $d\ne0$, that is, when the bowl curves upwards in every direction. Four equivalent ways to say it:

- *Every direction curves up.* The "curvature along the unit direction $u$" is $u^\top Hu$, and it must be positive for all $u$.
- *All eigenvalues are positive.* The eigenvectors are the directions of extreme curvature, the eigenvalues are the curvatures there, and every other direction
  lies in between. For the Gaussian potential at $(\mu,\sigma)=(1,2)$, $H=\left[\begin{smallmatrix}4&8\\8&48\end{smallmatrix}\right]$ has eigenvalues $2.5906$ and
  $49.4094$; scanning 1800 directions the curvature never leaves $[2.5906,\,49.4094]$.
- *The unit ball is an ellipse.* Define the length $|d|_H=\sqrt{d^\top Hd}$; the set $|d|_H=1$ is an ellipse whose half-axes are $1/\sqrt{\lambda_i}$ along the
  eigenvectors ($0.1423$ and $0.6213$ for the Gaussian). This is why a positive-definite $H$ can serve as a **metric**: it assigns every nonzero step a positive length.
  Where $\lambda$ is large, steps are expensive and the ellipse is thin.
- *Sylvester's test (2×2).* $H=\left[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right]$ is positive-definite exactly when $a>0$ and $\det H=ac-b^2>0$; for the Gaussian, $4>0$ and $\det=128>0$.

What goes wrong otherwise, with the examples the widget offers: if the smallest eigenvalue is **zero** (positive *semi*-definite) there is a flat direction, a trough, along which
steps have length zero. $\psi=x^4+y^2$ at the origin has $H=\left[\begin{smallmatrix}0&0\\0&2\end{smallmatrix}\right]$, eigenvalues $0$ and $2$; a step $e=0.3$ along $x$ has true gap
$D=0.0081$ but the quadratic term predicts $0.0000$, so the Hessian is blind there and is not a metric. If an eigenvalue is **negative** the surface is a saddle or a hill in that direction:
$\psi=x^2-y^2$ has eigenvalues $-2$ and $2$, "length" would be imaginary along $y$, and $\psi$ is not convex. For the softmax at $\theta=(0.4,-0.8)$ the Hessian
$\left[\begin{smallmatrix}0.2499&-0.0775\\-0.0775&0.1294\end{smallmatrix}\right]$ has eigenvalues $0.0915$ and $0.2879$ and determinant $0.0263$, positive-definite, as a covariance matrix must be. The interactive page's third widget lets you drag the three entries of a 2×2 matrix and watch the ellipse, the eigen-directions and the curvature-by-direction curve turn into a trough or a saddle.

## 4. The Legendre dual: slopes as coordinates (§1.4)

**Idea.** Because $\nabla^2\psi\succ0$, the slope map $\theta\mapsto\eta=\nabla\psi(\theta)$ is one-to-one: distinct points of the
graph have distinct tangent planes. So *the slope can serve as a coordinate*. For softmax this says that the logits
and the probabilities are two names for the same point, and the map between them (softmax) is the slope map of log-sum-exp.

**The Legendre transform, three pictures of one thing.** Take a convex $\psi$ and a slope $\eta$.

1. *Slope as a name.* Each point $\theta$ of the graph has a tangent line with slope $\eta=\psi'(\theta)$; because $\psi'$ is increasing, distinct points have distinct slopes, so
   the slope identifies the point just as well as $\theta$ does.
2. *Intercept as the dual function.* Draw the tangent line of slope $\eta$. Where it crosses the vertical axis $\theta=0$ is $\psi(\theta)-\eta\theta$, so
   $\psi^*(\eta)=\eta\theta-\psi(\theta)$ is **minus the intercept**. Tracing how the intercept changes with the slope draws the curve $\psi^*$.
3. *The best a linear function can do.* $\psi^*(\eta)=\max_{\theta'}\{\eta\theta'-\psi(\theta')\}$: the largest amount by which the line $\eta\theta'$ rises above $\psi$. The maximum is at the
   $\theta'$ where the slope of $\psi$ equals $\eta$, which is picture 1 again.

Numbers from `code/dually_flat.py` (each: the intercept, the supremum over a fine grid, and the closed form agree):

| $\psi(\theta)$ | $\theta_0$ | $\eta=\psi'(\theta_0)$ | $\psi^*(\eta)$ | closed form of $\psi^*$ |
|---|---|---|---|---|
| $\tfrac12\theta^2$ | 1.5 | 1.5000 | 1.1250 | $\tfrac12\eta^2$ |
| $\log(1+e^\theta)$ | −0.4 | 0.4013 | −0.6735 | $\eta\log\eta+(1-\eta)\log(1-\eta)$ |
| $e^\theta$ | 1 | 2.7183 | 0.0000 | $\eta\log\eta-\eta$ |
| $\theta^4$ | 1 | 4.0000 | 3.0000 | $\tfrac34\eta(\eta/4)^{1/3}$ |

Two things to see. The slope of $\psi^*$ at $\eta$ is $\theta_0$ in every row ($1.5000$, $-0.4000$, $1.0000$, $1.0000$): the roles of point and slope are exchanged, which is (1.64). And the curvatures are
reciprocal, $\psi''(\theta_0)\,\psi^{*\prime\prime}(\eta)=1$ in every row ($0.2403\times4.1621$ for the coin, $2.7183\times0.3679$ for $e^\theta$): a steep potential has a flat dual and vice versa, which is $G^*=G^{-1}$ of (1.66).
Transforming twice returns $\psi$: starting from $\psi^*=\eta\log\eta-\eta$ and maximising $\eta\cdot1-\psi^*(\eta)$ gives $2.7183=e^1$. The interactive page's fifth widget shows the tangent, its intercept and the dual curve
together for the four potentials above.

## 5. Two flat structures and the shortest path (§1.5)

**Two kinds of straight.** Declare $\theta$ an affine coordinate system: a curve $\theta(t)=at+b$ is straight, an
*e-geodesic*. Declare $\eta$ affine too: $\eta(t)=at+b$ is a *m-geodesic*. Because $\theta\mapsto\eta$ is not linear, the
two families of lines are different. The picture below shows one pair $P=(0.7,0.2,0.1)$, $Q=(0.1,0.3,0.6)$
joined by both, in the logit chart (left, e-geodesic straight) and the probability triangle (right, m-geodesic straight).

<img src="figures/charts.svg" alt="Two charts of the three-outcome probability simplex with P = (0.7, 0.2, 0.1) and Q = (0.1, 0.3, 0.6). In the logit chart the e-geodesic is a straight blue line and the m-geodesic an orange curve; in the probability triangle the roles swap. At t = 0.5 the e-geodesic is at (0.351, 0.325, 0.325) and the m-geodesic at (0.400, 0.250, 0.350).">

At $t=0.5$ the e-geodesic passes through $(0.3507,0.3247,0.3247)$ (a normalised geometric mixture, $p_i\propto\sqrt{p_iq_i}$)
and the m-geodesic through $(0.4,0.25,0.35)$. Neither is the **Fisher–Rao geodesic**, the true shortest path for the
metric $G$ (a great-circle arc after $p\mapsto\sqrt p$), which passes through $(0.3788,0.2821,0.3391)$. In this example it sits
between the other two in eight of the nine coordinates I looked at ($t\in\{0.25,0.5,0.75\}$, three coordinates each); the exception is $p_2$ at $t=0.75$, where the three
values are $0.4692$ (e), $0.4750$ (m) and $0.4754$ (Fisher–Rao). That is the pattern one expects from the standard fact, which this chapter does not prove, that the Riemannian
connection is the average of the two flat ones, but it is only an observation about one pair of points.

**The two straight lines on the Gaussian manifold.** Take the example from §1: a point is $N(\mu,\sigma^2)$, with two affine charts, the natural parameters $\theta=(\mu/\sigma^2,\,-1/(2\sigma^2))$ and the moments $\eta=(\mu,\ \mu^2+\sigma^2)$. Join $P=N(0,1^2)$ and $Q=N(1.5,1.5^2)$.

*The e-geodesic: straight in $\theta$.* Put $\theta(s)=(1-s)\theta_P+s\,\theta_Q$. Translating back to $(\mu,\sigma)$ with $\theta_2=-1/(2\sigma^2)$ and $\theta_1=\mu/\sigma^2$:

$$
\frac1{\sigma_s^2}=\frac{1-s}{\sigma_P^2}+\frac{s}{\sigma_Q^2},\qquad
\frac{\mu_s}{\sigma_s^2}=(1-s)\frac{\mu_P}{\sigma_P^2}+s\,\frac{\mu_Q}{\sigma_Q^2}.
$$

In words: the **precisions** $1/\sigma^2$ average linearly, and the mean is the **precision-weighted** average, $\mu_s=\sigma_s^2\big[(1-s)\mu_P/\sigma_P^2+s\,\mu_Q/\sigma_Q^2\big]$. Equivalently, the density is a normalised geometric mixture,
$\log p_s=(1-s)\log p+s\log q+\text{const}$, which for Gaussians is again a Gaussian. At $s=0.5$: precision $(1/1+1/2.25)/2=0.7222$, so $\sigma_s=1.1767$, and $\mu_s=1.3846\times(1.5/2.25)/2=0.4615$.

*The m-geodesic: straight in $\eta$.* Put $\eta(s)=(1-s)\eta_P+s\,\eta_Q$. The first moment is linear, $\mu_s=(1-s)\mu_P+s\mu_Q$, and the second moment is linear, so the variance is

$$
\sigma_s^2=(1-s)\sigma_P^2+s\,\sigma_Q^2+s(1-s)(\mu_P-\mu_Q)^2 .
$$

This is the variance of the *mixture* $(1-s)p+s\,q$: the mixture's two moments are averaged, and the extra $s(1-s)(\mu_P-\mu_Q)^2$ is the spread between the two component means. At $s=0.5$: $\mu_s=0.75$ and $\sigma_s^2=(1+2.25)/2+0.25\times1.5^2=2.1875$, so $\sigma_s=1.4790$.
**A subtlety.** The mixture itself is *not* a Gaussian (the 50/50 mixture has excess kurtosis $+0.1127$), so it is not a point of the Gaussian manifold. The point of the m-geodesic is the Gaussian with the *same mean and variance as the mixture*
($0.7500$ and $2.1875$ by quadrature, matching). Within the manifold, "straight in $\eta$" means exactly this moment-matched curve.

*The third curve.* The Fisher–Rao geodesic, the true shortest path for the metric $G$, is neither. In the coordinates $(x,y)=(\mu/\sqrt2,\sigma)$ the metric is $ds^2=2(dx^2+dy^2)/y^2$, a hyperbolic half-plane, whose geodesics are semicircles centred on $y=0$; that gives the third curve below.

<img src="figures/gaussian-geodesics.svg" alt="The three curves joining N(0,1) and N(1.5, 1.5^2) drawn in the (mu, sigma) plane, in natural coordinates (theta1, theta2) and in moment coordinates (eta1, eta2). The e-geodesic is straight in natural coordinates, the m-geodesic is straight in moment coordinates, and the Fisher-Rao geodesic, the shortest path, is straight in neither. The m-curve bows up (wider) in the other two charts and the e-curve bows down.">

Midpoints and numbers (from `code/dually_flat.py`; each curve is straight in its own chart to $10^{-16}$):

| $s$ | e-geodesic $(\mu,\sigma)$ | m-geodesic $(\mu,\sigma)$ | Fisher–Rao $(\mu,\sigma)$ |
|---|---|---|---|
| 0.25 | (0.1935, 1.0776) | (0.3750, 1.3170) | (0.2575, 1.1724) |
| 0.50 | (0.4615, 1.1767) | (0.7500, 1.4790) | (0.6000, 1.3304) |
| 0.75 | (0.8571, 1.3093) | (1.1250, 1.5360) | (1.0230, 1.4479) |

The e-curve stays *narrower* than the m-curve at every $s$: the e-path averages precisions (small $\sigma$ dominates), the m-path averages variances and adds the mean-spread term. The Fisher–Rao curve lies between them in all six entries of the table,
and it is the shortest in the metric: the lengths $\int\sqrt{(\dot\mu^2+2\dot\sigma^2)/\sigma^2}\,ds$ are $1.3313$ (e), $1.3250$ (m) and $1.3070$ (Fisher–Rao).
They are also *different paths*, not different speeds on the same path: the KL divergences from the two ends differ. At the e-midpoint $\mathrm{KL}[P{:}E]=0.1007$ and $\mathrm{KL}[E{:}Q]=0.2901$; at the m-midpoint $\mathrm{KL}[P{:}M]=0.2485$ and $\mathrm{KL}[M{:}Q]=0.1252$.

Which is "straight" depends on which coordinates you count as affine, and the chapter's choice of $\psi$ makes $\theta$ affine for one structure and $\eta$ for the other. For the exponential family $\theta$-lines are the *natural* interpolations (average the exponent: geometric mixtures),
and $\eta$-lines are the moment interpolations (average what you can measure: mixtures). Interpolating two trained Gaussian models, say, by averaging logits and averaging probabilities are the same dichotomy.

**The Fisher–Rao geodesic.** The e- and m-geodesics are "straight" in a chart. The Fisher–Rao geodesic is something else: the path that is **shortest for the metric $G$**, locally. Intuition first: a path through the manifold has a length, add up the Riemannian length of each little step, and a *geodesic* is a path
you cannot shorten by wiggling it with the endpoints held fixed. Formally

$$
L[\gamma]=\int_0^1\sqrt{g_{ij}(\gamma)\,\dot\gamma^i\dot\gamma^j}\;dt ,
$$

and a geodesic is a critical point of $L$; it satisfies $\ddot\gamma^k+\Gamma^k_{ij}\dot\gamma^i\dot\gamma^j=0$, where the symbols $\Gamma$ are built from $g$ and its derivatives (the Levi-Civita connection; the book develops it in Part II, and I use it here only through its answer). Parametrised at constant speed, a geodesic does no "sideways steering".
It differs from the e- and m-lines because the metric is not constant in $\theta$ or in $\eta$: in a chart where $g$ varies, a straight line in the chart is not the shortest path.

*On the Gaussians it can be written down.* With $x=\mu/\sqrt2$ and $y=\sigma$ the metric is $ds^2=2(dx^2+dy^2)/y^2$: twice the hyperbolic metric of the upper half-plane. Its geodesics are the **vertical half-lines** and the **semicircles centred on the axis $y=0$**. Why, in three steps:
(i) a vertical line is the fixed set of the reflection $x\mapsto 2x_0-x$, which is an isometry, so a shortest path cannot leave it; (ii) inversion in a circle centred on $y=0$ is an isometry and maps vertical lines to semicircles; (iii) a semicircle $x=c+r\cos\varphi,\ y=r\sin\varphi$ has
$ds=\sqrt2\,d\varphi/\sin\varphi$, so its length is $\sqrt2\,[\ln\tan(\varphi/2)]$ between two angles, and moving uniformly in $\ln\tan(\varphi/2)$ is moving at constant speed (that is how the points in the table above were placed).

<img src="figures/fisher-rao.svg" alt="The Gaussian manifold as the upper half-plane with x = mu/sqrt(2) and y = sigma. Vertical lines and semicircles centred on the axis y = 0 are the geodesics. The geodesic from N(0,1) to N(1.5, 1.5^2) is a semicircle of length 1.3070; the straight chord between the same two points has length 1.3448.">

*The distance.* Two Gaussians $(\mu_1,\sigma_1)$ and $(\mu_2,\sigma_2)$ are at Fisher–Rao distance

$$
d=\sqrt2\;\operatorname{arccosh}\!\Big(1+\frac{(\Delta\mu)^2/2+(\Delta\sigma)^2}{2\sigma_1\sigma_2}\Big).
$$

For $P=N(0,1^2)$ and $Q=N(1.5,1.5^2)$: $\Delta x=1.0607$, $\Delta y=0.5$, $\sigma_1\sigma_2=1.5$, the argument is $1.4583$ and $d=1.3070$; numerically integrating the length along the semicircle gives $1.3070$. The semicircle is centred at $x=1.1196$ with radius $1.5012$.
Checks that it really is the shortest: the straight chord between $P$ and $Q$ in the $(\mu,\sigma)$ plane has length $1.3448$; the e-geodesic has $1.3313$ and the m-geodesic $1.3250$; and in 300 random smooth perturbations with the endpoints fixed, the *smallest* increase in length was $+0.00689$ (every perturbation lengthens the path).
The constant-speed parametrisation splits the length into four equal quarters of $0.3267$. A vertical line is a geodesic too: from $N(0,1)$ to $N(0,3)$ the distance is $\sqrt2\ln3=1.5537$, and the formula gives $1.5537$.

*Relation to KL.* For nearby points $d^2\approx2\,\mathrm{KL}$ in either order, because both share the quadratic part (§2). The table below takes $P=N(0,1)$ and $q=N(\varepsilon,(1+\varepsilon/2)^2)$:

| $\varepsilon$ | $d$ | $\sqrt{2\,\mathrm{KL}[P{:}q]}$ | $\sqrt{2\,\mathrm{KL}[q{:}P]}$ |
|---|---|---|---|
| 0.01 | 0.01222 | 0.01219 | 0.01224 |
| 0.1 | 0.11949 | 0.11696 | 0.12215 |
| 1 | 0.98026 | 0.83655 | 1.19961 |

The three agree as $\varepsilon\to0$ and spread as it grows; in these rows the true distance lies between the two KL-based numbers. At the far ends of our pair, $\sqrt{2\,\mathrm{KL}[P{:}Q]}=1.1204$ and $\sqrt{2\,\mathrm{KL}[Q{:}P]}=1.6398$ against $d=1.3070$. So KL is not a distance, and its two orders over- and under-estimate the shortest-path distance by different amounts.
The Fisher–Rao distance, unlike KL, is symmetric, satisfies the triangle inequality and does not depend on the chart.

*Three curves, three meanings.* The e-geodesic is straight in the natural coordinates and the m-geodesic straight in the moment coordinates: they depend on the *flat structures* built from $\psi$. The Fisher–Rao geodesic depends only on the *metric*, so it is the same for every divergence that induces this $g$
(§2: all of them agree to second order). It is the Riemannian geodesic of the dually flat manifold; that it lies "between" the e- and m-geodesics (as in the table) is what one expects from the standard fact that the Riemannian connection is the average of the two flat ones, which this chapter does not prove and I have only seen confirmed on this pair of points.

## 6. The Pythagorean theorem and projections (§1.6)

Let $P,Q,R$ be three points. Suppose the **m-geodesic** from $P$ to $Q$ is orthogonal, at $Q$, to the **e-geodesic** from $Q$ to $R$. Then $D_\psi(R{:}P)=D_\psi(Q{:}P)+D_\psi(R{:}Q)$ (Theorem 1.2).

### Pythagoras, a picture to hold on to

*Start from the one you know.* Put $P=(0,0)$, $Q=(3,0)$, $R=(3,4)$ in the plane and take $\psi=\tfrac12\|x\|^2$, so that $D=\tfrac12\times$ squared distance. Then $D(Q{:}P)=4.5$, $D(R{:}Q)=8.0$, their sum is $12.5=D(R{:}P)$: the 3-4-5 triangle with every
side squared and halved. The right angle is the statement $(Q-P)\cdot(R-Q)=0$. Two things played a role, and both get generalised.

| In the plane | In a dually flat manifold |
|---|---|
| squared length $\tfrac12\|x-y\|^2$ | the divergence $D_\psi$ |
| the right angle $(Q-P)\cdot(R-Q)=0$ | the pairing $(\eta_Q-\eta_P)\cdot(\theta_R-\theta_Q)=0$ |
| a straight line (one kind) | an $\eta$-straight line for one leg, a $\theta$-straight line for the other |
| $\theta=\eta$ | $\theta\ne\eta$: the two charts differ, so "perpendicular" must pair a difference of one with a difference of the other |

The pairing is the whole trick. When $\psi=\tfrac12\|x\|^2$ the charts coincide ($\eta=\theta$) and the pairing is the ordinary dot product; in general the leg $P\to Q$ is naturally described by the *difference of $\eta$*, the leg $Q\to R$ by the *difference of $\theta$*, and orthogonality
is the vanishing of their pairing.

*The same statement for Gaussians.* Take $P=N(0,1^2)$ and $Q=N(1.5,1.5^2)$. Their dual coordinates are $\eta=(\mu,\mu^2+\sigma^2)$, giving $\eta_P=(0,1)$ and $\eta_Q=(1.5,4.5)$, and the natural parameters $\theta=(\mu/\sigma^2,-1/(2\sigma^2))$, giving $\theta_P=(0,-0.5)$ and
$\theta_Q=(0.6667,-0.2222)$. The m-geodesic from $P$ to $Q$ is the line $\eta(s)=(1-s)\eta_P+s\eta_Q$ (a path of mixtures of moments, drawn as a curve in the $(\mu,\sigma)$ plane). Leave $Q$ along the $\theta$-straight line in the direction orthogonal to it,
$\theta(t)=\theta_Q+t\,v_\perp$ with $v_\perp\perp(\eta_Q-\eta_P)$, and stop at $R$. Since $D_\psi[\theta_X{:}\theta_Y]=\mathrm{KL}[Y{:}X]$, the theorem reads $\mathrm{KL}[P{:}R]=\mathrm{KL}[P{:}Q]+\mathrm{KL}[Q{:}R]$. Numbers:

| stop | $R$ | $\mathrm{KL}[P{:}Q]$ | $\mathrm{KL}[Q{:}R]$ | sum | $\mathrm{KL}[P{:}R]$ |
|---|---|---|---|---|---|
| $t=-1$ | $N(1.2869,\,0.9008^2)$ | 0.6277 | 0.4044 | 1.0321 | 1.0321 |
| $t=+0.3$ | $N(1.8786,\,2.1922^2)$ | 0.6277 | 0.1284 | 0.7561 | 0.7561 |

The two sides agree exactly, for a $R$ narrower than $Q$ and one wider than $Q$: the leg $Q\to R$ may go either way along the $\theta$-line.

*Tilting away from the right angle.* Rotate the direction of the second leg by $20^\circ$ toward the first leg's own direction (at $t=-1$): now $R=N(0.7696,\,0.7426^2)$, $\mathrm{KL}[P{:}Q]+\mathrm{KL}[Q{:}R]=1.9485$ but
$\mathrm{KL}[P{:}R]=0.6462$, a leftover of $+1.3024$. It is not an approximation error: it equals $(\theta_Q-\theta_R)\cdot(\eta_Q-\eta_P)=+1.3024$, the pairing that the right angle sets to zero. In the Euclidean case this is the law of cosines:
the leftover is $-(Q-P)\cdot(R-Q)$ (twice the $-\|a\|\|b\|\cos$ term, halved). So the theorem is the law of cosines with the cosine term switched off; moving away from a right angle switches it on, by exactly the pairing.

*Why you cannot see the right angle in a picture.* Neither chart is orthonormal: in the $\theta$ chart the $\theta$-leg is straight and the $\eta$-leg curved, in the $\eta$ chart the opposite, and the metric $G$ that defines "perpendicular" varies from point to point. What you can check is the pairing, which is chart-free. The next
widget on the interactive page draws both legs on the Gaussians, stacks the two divergences against the direct one, and lets you tilt.

*What it buys.* Information from $P$ to $R$ *splits*: $\mathrm{KL}[P{:}R]=\mathrm{KL}[P{:}Q]+\mathrm{KL}[Q{:}R]$ says the cost of describing $R$ through $P$ is the cost of reaching $Q$ plus the cost of going on to $R$, with no cross term, exactly when
$Q$ is the foot of the perpendicular. That is why the foot is the closest point (see the projection theorem below) and why maximum-likelihood fitting decomposes into a fit term and an error term.

### The projection theorem on the Gaussians

The projection theorem answers: *given a point $P$ and a family $S$ inside the manifold, which member of $S$ is closest to $P$?* In the plane the answer is the foot of the perpendicular, and flatness of $S$ (a line) makes it unique. The same holds here, with "closest" meaning smallest divergence and "perpendicular" meaning the pairing of §6 vanishes.
A Gaussian example where every step can be seen: let $S=\{N(0,\sigma^2)\}$, the **zero-mean Gaussians**. In the $(\mu,\sigma)$ plane it is the vertical axis $\mu=0$. It is a straight line in $\theta$ ($\theta_1=\mu/\sigma^2=0$) and also in $\eta$ ($\eta_1=\mu=0$), so it is flat in both senses.
Take $P=N(1.5,1^2)$ and ask for the member of $S$ that minimises $\mathrm{KL}[P{:}R]$, the direction of model fitting.

*Find the foot by the geometry.* The foot $\hat R$ is where the m-geodesic from $P$ (straight in $\eta$) meets $S$ at a right angle. Write the pairing: $\eta_P=(1.5,\ 3.25)$ and a point $\hat R=N(0,\hat\sigma^2)$ has $\eta=(0,\ \hat\sigma^2)$, so $\eta_P-\eta_{\hat R}=(1.5,\ 3.25-\hat\sigma^2)$. The tangent of $S$ in $\theta$ is $(0,1)$ ($\theta_1$ stays $0$, $\theta_2$ moves).
Orthogonality is $(\eta_P-\eta_{\hat R})\cdot(0,1)=3.25-\hat\sigma^2=0$, so $\hat\sigma^2=3.25$: **the foot keeps the second moment of $P$**, $\hat\sigma^2=\mu_P^2+\sigma_P^2$ and $\hat\sigma=1.8028$.
Along the way the m-geodesic keeps $\mu^2+\sigma^2=3.25$: it is an arc of a circle in the $(\mu,\sigma)$ plane, passing through $(1.500,1.000)$, $(1.125,1.409)$, $(0.750,1.639)$, $(0.375,1.763)$ and ending at $(0,1.803)$. Because the metric is diagonal here, "perpendicular" is the usual perpendicular in the picture: the circle arrives horizontally at the vertical axis.

*Check against brute force.* The minimum of $\mathrm{KL}[P{:}N(0,s^2)]$ over $s\in[0.3,4]$ is $0.5893$ at $s=1.8028$; the closed form is $\mathrm{KL}[P{:}\hat R]=\tfrac12\ln(1+\mu_P^2/\sigma_P^2)=0.5893$.

*Why it is the minimum, not just a critical point (Pythagoras).* For every $R=N(0,s^2)$ in $S$, $\mathrm{KL}[P{:}R]=\mathrm{KL}[P{:}\hat R]+\mathrm{KL}[\hat R{:}R]$: at $s=0.5$, $5.3069=0.5893+4.7175$; at $s=3$, $0.7792=0.5893+0.1898$; over 2000 members the largest gap is $5.3\times10^{-15}$.
The second term is $\ge0$ and vanishes only at $R=\hat R$: that is uniqueness (Theorem 1.5). The right panel of the figure below plots both sides, and they coincide.

<img src="figures/projection-gaussian.svg" alt="Left: the (mu, sigma) plane with the vertical line S of zero-mean Gaussians, the Gaussian P = N(1.5, 1), the m-geodesic from P to S, an arc of the circle mu^2 + sigma^2 = 3.25 that meets S at R-hat = N(0, 3.25), and the horizontal e-geodesic that meets S at N(0, 1). Right: KL[P:R] along S and KL[P:R-hat] + KL[R-hat:R] coincide, with minimum 0.5893 at sigma = 1.8028, while KL[R:P] has its minimum 1.1250 at sigma = 1.">

*This is maximum likelihood.* Fitting a zero-mean Gaussian to data means maximising the likelihood over $\sigma$, which is minimising $\mathrm{KL}[\text{data}{:}R]$; the answer is the **second moment**, $\hat\sigma^2=\overline{x^2}$. With $200\,000$ draws from $P$ the best $\sigma$ by likelihood is $1.8027$, equal to $\sqrt{\overline{x^2}}=1.8027$, and close to the population value $1.8028$.
In the geometry: the m-projection onto an e-flat family matches the dual coordinates ($\eta_2$) that the family can express, and ignores the rest ($\eta_1$, the mean, which $S$ has no freedom to match). Matching moments *is* the m-projection.

*The other order has its own foot.* Minimise $\mathrm{KL}[R{:}P]$ over $S$ instead: the minimiser keeps $P$'s width, $s=\sigma_P=1.0$, with value $\mu_P^2/(2\sigma_P^2)=1.1250$ (grid agrees). Its path from $P$ is the e-geodesic, which is **horizontal** ($\sigma$ fixed, $\mu$ going $1.5\to0$) and also meets $S$ at a right angle, at a different point.
Here the split is $\mathrm{KL}[R{:}P]=\mathrm{KL}[R{:}R_e]+\mathrm{KL}[R_e{:}P]$ (largest gap over 2000 members $1.8\times10^{-15}$), the mirror image. Using the wrong split around the wrong foot fails by up to $11.375$: the theorem pairs each projection with *its* order of $\mathrm{KL}$.
Both versions work because $S$ is flat in both senses; the next example has only one.

*Only one flat structure: the dual case.* Fix the mean at $c=-0.5$: $S_c=\{N(-0.5,s^2)\}$ is a straight line in $\eta$ ($\eta_1=-0.5$) but not in $\theta$, so it is m-flat only. Minimising $\mathrm{KL}[R{:}P]$ (the e-projection) gives $s=\sigma_P=1.0$ with value $(c-\mu_P)^2/(2\sigma_P^2)=2.0000$, and the split $\mathrm{KL}[R{:}P]=\mathrm{KL}[R{:}\hat R]+\mathrm{KL}[\hat R{:}P]$ holds for 2000 members to $1.8\times10^{-15}$.
So for a family that is flat in only one sense, only one of the two projections is guaranteed to be the unique minimiser; the circle counterexample below shows what fails for a curved family.

### Orthogonality is necessary, not sufficient

The projection theorem only says that the foot point is a *critical* point. The book notes this. A concrete case: in
the Euclidean setting ($\psi=\tfrac12\|\xi\|^2$) take $S$ the unit circle and $P=(0.5,0)$. Both $(1,0)$ and $(-1,0)$ have the
segment to $P$ orthogonal to the circle (residuals $0$ and $6\times10^{-17}$), with $D=0.125$ at the first (the closest point) and $D=1.125$ at the
second (the *farthest*). Flatness of $S$ is what excludes this.

<img src="figures/critical-points.svg" alt="Left: a unit circle S in the plane with P = (0.5, 0); the closest point (1, 0) and the farthest point (-1, 0) both have the segment to P orthogonal to the circle. Right: half the squared distance along the circle has a minimum 0.125 and a maximum 1.125.">

The interactive page lets you see the non-Euclidean version: for a curved arc and $\mathrm{KL}[R\Vert P]$, put $P$ near the centre of curvature
and the divergence along the arc has two minima and a maximum.
