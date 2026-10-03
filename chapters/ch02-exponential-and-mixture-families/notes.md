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
