---
title: "Chapter 5: elements of differential geometry"
short_title: "Ch. 5 — Elements of differential geometry"
chapter: 5
category: "Part II"
book_pages: "109–130"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [differential-geometry, tangent-space, riemannian-metric, affine-connection, christoffel-symbols, tensors, covariant-derivative, geodesic, parallel-transport, curvature, levi-civita, submanifold, embedding-curvature]
status: read
---

## Links

- **[Interactive companion](figures/interactive.html)**: six widgets. (1) A tangent vector as an arrow, a derivative operator and a random variable (the score), on the Gaussian manifold with its unit Fisher circle.
  (2) What a Christoffel symbol measures: the polar frame at two nearby points, the exact change of the frame against the first-order prediction. (3) One start and one velocity under the e-, m- and Levi-Civita connections and a dial between e and m, with speeds and lengths against the Fisher–Rao distance.
  (4) Parallel transport round a coordinate rectangle on the sphere, in the plane with polar coordinates and on the Gaussian manifold, with the e-to-m dial. (5) The loop formula for curvature, measured against the derived and the printed right-hand sides. (6) Bending a flat sheet into a cylinder: intrinsic curvature stays $0$ while embedding curvature grows.
- **[Runnable checks](https://github.com/msrepo/information_geometry_notes/tree/main/chapters/ch05-elements-of-differential-geometry/code)**:
  `code/diffgeo.py` prints every number on this page and regenerates the figures with `python3 code/diffgeo.py --figures`. `make verify` runs it, in about three seconds.
- The book: Amari, *Information Geometry and Its Applications* (Springer, 2016), DOI [10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8). These notes cover Chapter 5 only; equation numbers such as (5.54) refer to the book.
  No text of the book is reproduced here; everything is restated and re-derived.
- Previous chapter written up: **[Chapter 2: exponential and mixture families](../ch02-exponential-and-mixture-families/index.html)**. This chapter uses nothing from Chapters 3–4 (still to come) beyond the Gaussian example of Chapters 1–2, and Chapter 6 (dual connections) will build directly on it.
  Background on the annotations site: **[The Fisher information matrix](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/)**.

## In one paragraph

Calculus tells you how a *function* changes. To say how a *vector* changes you need more, because a vector at one point of a curved space lives in a different space from a vector at the next point, so "the same vector" has no meaning until a rule is chosen. The same trouble appears on a perfectly flat plane the moment you use polar coordinates: the basis vectors $e_r,e_\vartheta$ themselves turn from place to place.
The rule is an **affine connection**, written in coordinates as an array $\Gamma_{ij}{}^k$ (the Christoffel symbols). Everything else in the chapter hangs on it: the **covariant derivative** (the change of a vector field with the turning of the frame subtracted), **geodesics** (curves whose velocity does not change, the straight lines of that connection), **parallel transport** (carrying a vector without changing it) and **curvature** (the failure of parallel transport to be path independent). A **metric** is a different, independent structure: lengths and angles in every tangent space. The two meet in the **Levi-Civita connection**, the one symmetric connection that preserves lengths, and for it straight lines are also shortest lines.
A **submanifold** inherits a metric and a connection but has one more kind of curvature, how it bends inside the bigger space, which its own inhabitants cannot see (a cylinder is flat inside).
What matters for information geometry is the point the book ends on: the Fisher metric selects Levi-Civita, but a manifold of distributions carries other natural connections that are *not* metric. On the Gaussian manifold the e-connection and the m-connection are both **flat**, Levi-Civita has constant curvature $-\tfrac12$, and from one point with one velocity the three give three different straight lines. Chapter 6 turns this into a theory by coupling two connections through the metric.

## The spine of the argument

1. Tangent vectors are derivative operators $\partial_i$; for a family of distributions they are scores $\partial_i\log p$. Basis vectors change with the Jacobian, components with its inverse (§1).
2. A metric is an inner product $g_{ij}=\langle e_i,e_j\rangle$ on each tangent space; on a statistical manifold it is the Fisher information (§2).
3. To differentiate a vector field, nearby tangent spaces must be identified. An affine connection does this to first order: $de_i=\Gamma_{ki}{}^j\,d\xi^k\,e_j$. It is **not a tensor** (§3, §4).
4. Covariant derivative $\nabla_iX^k=\partial_iX^k+\Gamma_{ij}{}^kX^j$; geodesics $\nabla_{\dot\xi}\dot\xi=0$, i.e. $\ddot\xi^k+\Gamma_{ij}{}^k\dot\xi^i\dot\xi^j=0$; parallel transport $\nabla_{\dot\xi}A=0$ (§5–§7).
5. Carrying a vector round a tiny loop changes it by the curvature tensor $R_{ijk}{}^l$; $R=0$ exactly when there are affine coordinates with $\Gamma=0$ (§8).
6. Metric compatibility plus symmetry pins $\Gamma$ down uniquely (Levi-Civita), and then geodesics are also the shortest curves (§9).
7. A submanifold has an induced metric, an induced connection and an **embedding curvature** $H$; intrinsic and embedding curvature are different (§10).

## Setup and notation

| Symbol | Meaning | In the examples |
|---|---|---|
| $\xi=(\xi^1,\dots,\xi^n)$ | a chart (coordinates) | $(\mu,\sigma)$, $(r,\vartheta)$, $(\theta,\varphi)$ |
| $e_i=\partial_i$ | basis of the tangent space $T_\xi$ along the coordinate curve $\xi^i$ | $\partial_\mu,\partial_\sigma$ |
| $A=A^ie_i$ | a tangent vector (upper index: contravariant components) | the velocity $\dot\xi^i$ of a curve |
| $g_{ij}=\langle e_i,e_j\rangle$ | the metric; $g^{ij}$ is the inverse matrix | Fisher information |
| $\Gamma_{ij}{}^k$ | connection coefficients, $\nabla_{e_i}e_j=\Gamma_{ij}{}^ke_k$; **the first index is the direction of differentiation** | |
| $\Gamma_{ijk}=\Gamma_{ij}{}^mg_{mk}$ | lowered form, $\langle\nabla_{e_i}e_j,e_k\rangle$ (5.21)–(5.22) | |
| $T_{ijk}=\mathbb E[\partial_il\,\partial_jl\,\partial_kl]$ | the cubic tensor, $l=\log p$ (5.33) | |
| $\nabla_iX^k$ | covariant derivative (5.48) | |
| $R_{ijk}{}^l$ | curvature: $R(e_i,e_j)e_k=R_{ijk}{}^le_l$ (5.66) | |
| $K$ | Gauss curvature of a surface, $K=R_{122}{}^mg_{m1}/\det g$ | |
| $J_\kappa{}^i=\partial\xi^i/\partial\zeta^\kappa$ | Jacobian of a change of chart $\xi\to\zeta$ (the convention of (5.7)) | |
| $B_a{}^i=\partial\xi^i/\partial u^a$, $H_{ab}{}^\kappa$ | embedding of a submanifold with coordinates $u^a$; its embedding curvature (5.90), (5.101) | |

Index convention. The book puts the *direction* index first, $\Gamma_{ij}{}^k$ with $\nabla_{e_i}e_j=\Gamma_{ij}{}^ke_k$; many textbooks put it last. In the code `G[i, j, k]` stores $\Gamma_{ij}{}^k$ and `R[i, j, k, l]` stores $R_{ijk}{}^l$.

**Three running examples.** Every claim below is checked on at least one of them.

| Space | Chart | Metric | Non-zero $\Gamma_{ij}{}^k$ | Curvature |
|---|---|---|---|---|
| plane, polar | $(r,\vartheta)$ | $\operatorname{diag}(1,r^2)$ | $\Gamma_{\vartheta\vartheta}{}^r=-r$, $\Gamma_{r\vartheta}{}^\vartheta=\Gamma_{\vartheta r}{}^\vartheta=1/r$ | $0$ |
| unit sphere | $(\theta,\varphi)$ | $\operatorname{diag}(1,\sin^2\theta)$ | $\Gamma_{\varphi\varphi}{}^\theta=-\sin\theta\cos\theta$, $\Gamma_{\theta\varphi}{}^\varphi=\Gamma_{\varphi\theta}{}^\varphi=\cot\theta$ | $K=+1$ |
| Gaussians, Levi-Civita | $(\mu,\sigma)$ | $\operatorname{diag}(1/\sigma^2,\,2/\sigma^2)$ | $\Gamma_{\mu\sigma}{}^\mu=\Gamma_{\sigma\mu}{}^\mu=-1/\sigma$, $\Gamma_{\mu\mu}{}^\sigma=1/(2\sigma)$, $\Gamma_{\sigma\sigma}{}^\sigma=-1/\sigma$ | $K=-\tfrac12$ |
| Gaussians, e-connection | $(\mu,\sigma)$ | same | $\Gamma_{\mu\sigma}{}^\mu=\Gamma_{\sigma\mu}{}^\mu=-2/\sigma$, $\Gamma_{\sigma\sigma}{}^\sigma=-3/\sigma$ | $0$ (flat) |
| Gaussians, m-connection | $(\mu,\sigma)$ | same | $\Gamma_{\mu\mu}{}^\sigma=1/\sigma$, $\Gamma_{\sigma\sigma}{}^\sigma=1/\sigma$ | $0$ (flat) |

The Gaussian rows are explained in §3: the e-connection is the one for which $\theta=(\mu/\sigma^2,-1/2\sigma^2)$ is an affine chart, the m-connection the one for which $\eta=(\mu,\mu^2+\sigma^2)$ is, and Levi-Civita is the one tied to the Fisher metric. At the point $(\mu,\sigma)=(1,2)$ used throughout the numbers are $\Gamma^e=(-1.0,-1.5)$, $\Gamma^m=(+0.5,+0.5)$ and $\Gamma^{LC}=(-0.5,+0.25,-0.5)$ in the order listed.

## 1. Tangent vectors (§5.1)

**In plain words.** At a point of a surface, the tangent space is the flat plane that best fits the surface there, and a tangent vector is a possible velocity of a curve through the point. In a chart $\xi$ the velocity of a curve $\xi(t)$ has components $\dot\xi^i$, and the basis vector $e_i$ is the velocity of the coordinate curve along which only $\xi^i$ moves.
The mathematicians' version replaces the arrow by the thing an arrow does: it differentiates functions. The vector $A=A^i\partial_i$ is the operator $f\mapsto A^i\partial_if$, the rate of change of $f$ along $A$ (5.2)–(5.5). The reason to prefer the operator is that the chain rule makes it chart independent, which forces the transformation laws.

**The laws.** If $\zeta^\kappa=\zeta^\kappa(\xi)$ is a new chart then $\partial_i=\frac{\partial\zeta^\kappa}{\partial\xi^i}\partial_\kappa$ and $\partial_\kappa=\frac{\partial\xi^i}{\partial\zeta^\kappa}\partial_i=J_\kappa{}^i\partial_i$ (5.6)–(5.9).
Since $A=A^i\partial_i=A^\kappa\partial_\kappa$ must not depend on the chart, the components go the other way, $A^\kappa=\frac{\partial\zeta^\kappa}{\partial\xi^i}A^i$ (5.28): basis vectors transform with $J$, components with the inverse of $J$, and "contravariant" is just the name for this opposite behaviour.

**For distributions: a tangent vector is a random variable.** Take a curve of distributions $p(x;\xi(t))$. How fast does $\log p(x;\xi(t))$ change at a fixed outcome $x$? By the chain rule, $\dot\xi^i\,\partial_i\log p(x;\xi)$: the velocity of the curve, read as a function of $x$. So the basis vector $e_i$ is identified with the **score** $\partial_i\log p$, and a tangent vector $A$ with $f(x)=A^i\partial_i\log p(x;\xi)$ (5.10). Scores have mean zero (the density integrates to $1$), so tangent vectors are zero-mean random variables, and the inner product of two of them will be their covariance (§2).

*Check on the Gaussian at $(\mu,\sigma)=(1,2)$.* The scores are $\partial_\mu\log p=(x-\mu)/\sigma^2$ and $\partial_\sigma\log p=((x-\mu)^2-\sigma^2)/\sigma^3$. The Jacobians to the natural parameters $\theta$ are
$$\frac{\partial\xi}{\partial\theta}=\begin{pmatrix}4&8\\0&8\end{pmatrix},\qquad \frac{\partial\theta}{\partial\xi}=\begin{pmatrix}0.25&-0.25\\0&0.125\end{pmatrix}.$$
The basis law then says $\partial_{\theta_1}=4\,\partial_\mu$ and $\partial_{\theta_2}=8\,\partial_\mu+8\,\partial_\sigma$. As scores: $4\cdot\frac{x-1}{4}=x-1$ and $8\cdot\frac{x-1}{4}+8\cdot\frac{(x-1)^2-4}{8}=x^2-5$, which are exactly $x-\eta_1$ and $x^2-\eta_2$, the $\theta$-scores of an exponential family (largest error over 41 values of $x$: $1.9\times10^{-9}$).
Now take the arrow $A=(1,\tfrac12)$ in $(\mu,\sigma)$. Its components in $\theta$ are $A^\theta=(0.1250,\,0.0625)$, and the random variable $f(x)=A^i\partial_i\log p$ is the same function of $x$ whichever way it is computed ($1.4\times10^{-10}$ apart). Its second moment is $\mathbb E[f^2]=0.375000$; that number is the squared length of $A$ and returns in §2.
The interactive page's first widget lets you move the arrow and watch the two routes to $f(x)$ coincide.

## 2. The Riemannian metric (§5.2)

**In plain words.** A metric says how long a tangent vector is and what angle two of them make. In components, $g_{ij}=\langle e_i,e_j\rangle$ is a positive definite matrix at every point, and $\langle A,B\rangle=g_{ij}A^iB^j$ (5.11), (5.14). Because $\langle A,A\rangle$ cannot depend on the chart, the matrix must transform with two copies of the Jacobian, $g_{\kappa\lambda}=J_\kappa{}^iJ_\lambda{}^jg_{ij}$ (5.12). That law is the definition of a **tensor** (§4).

**For distributions** the inner product is the covariance of the scores, $\langle e_i,e_j\rangle=\mathbb E[\partial_il\,\partial_jl]$ (5.13): the Fisher information. It is invariant because scores are.

*Numbers.* At $(\mu,\sigma)=(1,2)$ quadrature gives $g=\operatorname{diag}(0.25,0.5)=\operatorname{diag}(1/\sigma^2,2/\sigma^2)$. The law (5.12) moves it to the natural parameters: $J^{\mathsf T}gJ=\left[\begin{smallmatrix}4&8\\8&48\end{smallmatrix}\right]$, which is also $\operatorname{Cov}[(x,x^2)]$ and the Hessian of the cumulant function $\psi$ of Chapter 2 (the three agree to $10^{-4}$ or better). The length of $A=(1,\tfrac12)$ is $0.375000$ computed in either chart. For the plane in polar coordinates $J^{\mathsf T}J=\operatorname{diag}(1,r^2)$, $=\operatorname{diag}(1,4)$ at $r=2$.

**Euclidean means: some chart has $g=\delta$** (5.15). The plane is, via Cartesian coordinates. The Gaussian manifold is not, for a reason that only shows up with curvature (§8): its Gauss curvature is $-\tfrac12$.

## 3. The affine connection and Christoffel symbols (§5.3)

**The problem, in plain words.** To say how a vector field $X$ changes from $\xi$ to $\xi+d\xi$ you must compare $X(\xi)\in T_\xi$ with $X(\xi+d\xi)\in T_{\xi+d\xi}$, two different spaces. Equal components do not mean equal vectors, because the basis moved. The plane in polar coordinates shows it at once: the field $e_r$ has constant components $(1,0)$ and yet points in a different direction at every $\vartheta$.
A **connection** is a rule that carries $T_{\xi+d\xi}$ back to $T_\xi$, close to the identity when $d\xi$ is small, hence to first order a linear rule. The book's picture is a surface in ordinary space: slide the nearby tangent plane back, then project it onto the first. Formally, the transported frame differs from the original by

$$de_i=\Gamma_{ki}{}^j\,d\xi^k\,e_j\qquad(5.19\text{–}5.20),$$

so $\Gamma_{ki}{}^j$ says: *moving in direction $k$, the $i$-th basis vector picks up a component $\Gamma_{ki}{}^j$ along the $j$-th.* The lowered form $\Gamma_{kim}=\langle de_i/d\xi^k,e_m\rangle$ is (5.21)–(5.22); the metric is used only for that.

**Reading $\Gamma$ off the polar plane.** Moving by $d\vartheta$ rotates the radial unit vector by the angle $d\vartheta$ towards $\hat e_\vartheta$. The coordinate vector $e_\vartheta=r\hat e_\vartheta$ has length $r$, so $de_r=d\vartheta\,\hat e_\vartheta=\frac{d\vartheta}{r}e_\vartheta$, i.e. $\Gamma_{\vartheta r}{}^\vartheta=1/r$.
For $e_\vartheta=r\hat e_\vartheta$: $de_\vartheta=dr\,\hat e_\vartheta-r\,d\vartheta\,\hat e_r=\frac{dr}{r}e_\vartheta-r\,d\vartheta\,e_r$, i.e. $\Gamma_{r\vartheta}{}^\vartheta=1/r$ and $\Gamma_{\vartheta\vartheta}{}^r=-r$. At $r=2$, $\vartheta=40^\circ$, $d\vartheta=0.4$, $dr=0$, solving for the exact change of the frame in the basis at $P$ gives, for $de_r$, components $(-0.0789,\,0.1947)$ against the first-order prediction $(0,\,0.2)$; for $de_\vartheta$, $(-0.7788,\,-0.0789)$ against $(-0.8,\,0)$.
They agree to second order in $d\xi$, which is all (5.20) claims. The second widget shows the two frames, the difference vectors and these numbers.

**Where connections come from.** The book names two sources. (a) The metric: Levi-Civita (§9). (b) A divergence, giving a pair of connections coupled through the metric (Chapter 6). A third, which the Gaussian examples below need, is to declare a chart affine: *define* the connection by $\Gamma=0$ in some chart $a$. Then in any other chart $\xi$ the transformation law (5.37) leaves only its inhomogeneous term,
$$\Gamma_{ij}{}^m=\frac{\partial^2a^A}{\partial\xi^i\partial\xi^j}\frac{\partial\xi^m}{\partial a^A}.$$
On the Gaussian manifold this gives two flat connections. For the e-connection $a=\theta$; for the m-connection $a=\eta=(\mu,\mu^2+\sigma^2)$. The calculation at $(1,2)$: $\partial^2\theta_1/\partial\mu\partial\sigma=-2/\sigma^3$ and $\partial\mu/\partial\theta_1=\sigma^2$ give $\Gamma^e_{\mu\sigma}{}^\mu=-2/\sigma=-1.0$; $\partial^2\theta_2/\partial\sigma^2=-3/\sigma^4$ and $\partial\sigma/\partial\theta_2=\sigma^3$ give $\Gamma^e_{\sigma\sigma}{}^\sigma=-3/\sigma=-1.5$. Likewise $\partial^2\eta_2/\partial\mu^2=2$ and $\partial\sigma/\partial\eta_2=1/(2\sigma)$ give $\Gamma^m_{\mu\mu}{}^\sigma=1/\sigma=0.5$. The Levi-Civita symbol of the Fisher metric, computed from (5.85) below, is $(-1/\sigma,\ 1/(2\sigma),\ -1/\sigma)$, and it is exactly the **average** of the two: $\Gamma^{LC}=\tfrac12(\Gamma^e+\Gamma^m)$ (difference $2.8\times10^{-9}$).
(The book defines these connections in Chapter 6 as the $\alpha=\pm1$ members of a family; here I use only the characterisation by affine charts.) All three are symmetric in the two lower indices. The unit sphere's induced connection (5.100) from its embedding in $\mathbb R^3$ equals the Levi-Civita symbol of its induced metric, $\Gamma_{\varphi\varphi}{}^\theta=-\sin\theta\cos\theta=-0.4546$ and $\Gamma_{\theta\varphi}{}^\varphi=\cot\theta=0.6421$ at $(\theta,\varphi)=(1,0.4)$.

## 4. Tensors, and why $\Gamma$ is not one (§5.4)

**In plain words.** A tensor is a bunch of numbers attached to a point that transforms with one Jacobian per index, so that an equation between tensors holds in every chart or in none. Vectors have one upper index; the metric two lower; a mixed quantity like $K^{ij}{}_{klm}$ any number of each (5.27)–(5.30). Raising and lowering uses the metric: with the dual basis $e^i=g^{ij}e_j$ (5.24) a vector has covariant components $A_i=g_{ij}A^j$ (5.26). The gradient of a function, $(\partial_if)$, is covariant (5.31)–(5.32).

**Not every array with indices is a tensor.**

- *The Hessian of a function* is not (5.34)–(5.35): $f_{\kappa\lambda}=J_\kappa{}^iJ_\lambda{}^jf_{ij}+(\partial_\kappa J_\lambda{}^j)\partial_jf$, the extra term involving the slope. It is a tensor where the slope vanishes. Check with $f=(\mu-1)^2+(\sigma-2)^2+\tfrac12\mu\sigma$ and the chart $\theta$: at the non-critical point $(1.5,2.5)$ the honest second derivatives in $\theta$ are $\left[\begin{smallmatrix}78.125&458.986\\458.986&3051.79\end{smallmatrix}\right]$, the tensor law gives $\left[\begin{smallmatrix}78.125&283.203\\283.203&1484.375\end{smallmatrix}\right]$, and the difference is the extra term $\left[\begin{smallmatrix}0&175.781\\175.781&1567.383\end{smallmatrix}\right]$; at the critical point $(0.5333,1.8667)$ the extra term vanishes and the two agree.
- *The connection coefficients* $\Gamma$ are not (5.36)–(5.37): $\Gamma_{\kappa\lambda\mu}=J_\kappa{}^iJ_\lambda{}^jJ_\mu{}^k\Gamma_{ijk}+(\partial_\kappa J_\lambda{}^j)J_\mu{}^kg_{jk}$. The second term is what turns the Cartesian $\Gamma=0$ into the polar $\Gamma_{\vartheta\vartheta}{}^r=-r$. Numerical check on the Gaussian Levi-Civita symbol, moved from $(\mu,\sigma)$ to $\theta$: the law gives $\Gamma_{111},\Gamma_{112},\Gamma_{122},\Gamma_{222}=0,\,16,\,64,\,448$, the direct computation from the metric by (5.85) gives the same, while the tensor-like first term alone gives $0,\,16,\,0,\,-192$.
  So "$\Gamma=0$ in one chart" is a statement about the chart, not about the geometry (5.38)–(5.41).

**The cubic tensor is a tensor** (5.33). $T_{ijk}=\mathbb E[\partial_il\,\partial_jl\,\partial_kl]$ is symmetric and transforms with three Jacobians, because the scores do. At $(1,2)$: $T_{\mu\mu\sigma}=2/\sigma^3=0.25$, $T_{\sigma\sigma\sigma}=8/\sigma^3=1.0$, the other entries $0$. Transformed to $\theta$ it becomes $T_{\theta_1\theta_1\theta_1}=0$, $T_{112}=32$, $T_{122}=128$, $T_{222}=896$, which are the third derivatives $\partial_i\partial_j\partial_k\psi$ and the third central moments of $(x,x^2)$ (agreement $10^{-13}$). So for an exponential family $T=\nabla^3\psi$, the next derivative after the metric $\nabla^2\psi$.

**A fact that the book uses without saying so in this chapter: the difference of two connections is a tensor.** The inhomogeneous term in (5.37) involves only the chart and the metric, not $\Gamma$, so it cancels in $\Gamma'-\Gamma$. On the Gaussians $\Gamma^m_{ijk}-\Gamma^e_{ijk}=T_{ijk}$ at $(1,2)$ (difference $10^{-16}$), and in the chart $\theta$, where $\Gamma^e=0$, the m-symbol is $\partial_i\partial_j\partial_k\psi$ and the Levi-Civita symbol is half of it, $\Gamma^{LC}_{ijk}=\tfrac12\partial_i\partial_j\partial_k\psi$ (the standard identity for a Hessian metric). The theory of Chapter 6 is largely about that tensor $T$.

## 5. The covariant derivative (§5.5)

**In plain words.** The ordinary derivative $\partial_iX^k$ of a vector field's components mixes two things: the field really changing, and the frame turning. The covariant derivative keeps only the first. Compare $X(\xi+d\xi)$ mapped back to $T_\xi$ with $X(\xi)$: the components of the mapped vector are $X^k+\partial_iX^k\,d\xi^i+\Gamma_{ij}{}^kX^j\,d\xi^i$, so

$$\nabla_iX^k=\partial_iX^k+\Gamma_{ij}{}^kX^j\qquad(5.48),$$

and along a field $Y$: $\nabla_YX=Y^i\nabla_iX^k\,e_k$ (5.49). Unlike $\partial_iX^k$, $\nabla_iX^k$ is a tensor. For a scalar, $\nabla$ is just $\partial$ (5.51).

<img src="figures/polar-frames.svg" alt="Left: the plane with circles r = 1, 2, 3 and rays. At eight points of the circle r = 2 the polar basis vectors e_r (blue) and the unit vector along e_theta (orange) turn, while the constant field d/dx (black) is the same arrow everywhere. Right: along the circle, the partial derivative of the radial component of that field, minus sin theta, and the connection term plus sin theta, add to zero.">

*The cleanest test.* In polar coordinates the constant field $X=\partial/\partial x$ has components $(X^r,X^\vartheta)=(\cos\vartheta,\,-\sin\vartheta/r)$. At $(r,\vartheta)=(2,0.7)$ the partial derivatives $\partial_iX^k$ are $\left[\begin{smallmatrix}0&0.1611\\-0.6442&-0.3824\end{smallmatrix}\right]$ (rows $i=r,\vartheta$), not zero, yet $\nabla_iX^k=0$ to $10^{-8}$ in every entry: the field does not change. The radial entry shows how, $\nabla_\vartheta X^r=\partial_\vartheta X^r+\Gamma_{\vartheta\vartheta}{}^rX^\vartheta=-\sin\vartheta+(-r)(-\sin\vartheta/r)=0$ (the picture above, right).
The reverse also happens: the field $e_r$ has constant components $(1,0)$, all partial derivatives vanish, and yet $\nabla_\vartheta e_r=\Gamma_{\vartheta r}{}^\vartheta e_\vartheta=\tfrac1r e_\vartheta\neq0$.
*Tensor check.* For $Y=x\,\partial_y$ the covariant derivative computed in polar coordinates, $\left[\begin{smallmatrix}0.49272&0.29249\\-0.83003&-0.49272\end{smallmatrix}\right]$, equals the tensor law applied to the Cartesian $\nabla Y=\partial Y$ ($2\times10^{-10}$ apart), while the bare partial derivatives fail the law by $1.17$.

## 6. Geodesics (§5.6)

**In plain words.** A straight line does not change direction. On a manifold: a curve is a **geodesic** of the connection if its velocity is parallel to itself, $\nabla_{\dot\xi}\dot\xi=0$ (5.53), in components

$$\ddot\xi^k+\Gamma_{ij}{}^k\,\dot\xi^i\dot\xi^j=0\qquad(5.54).$$

The first term is the acceleration in the chart; the second subtracts the part of it that is only the frame turning. The book's alternative $\nabla_{\dot\xi}\dot\xi=c(t)\dot\xi$ (5.55) is the same curve run at a different pace: if $\xi(t)=\gamma(t^3)$ with $\gamma$ a geodesic, then $\ddot\xi+\Gamma\dot\xi\dot\xi=(2/t)\dot\xi$, checked numerically at $t=1.3$ (both components of the left side equal $c\,\dot\xi$ with $c=2/t=1.5385$).

*Examples.* In polar coordinates start at $(r,\vartheta)=(1,0)$ with velocity $(0.5,1)$: (5.54) integrates to the Cartesian straight line $(1+t/2,\,t)$ with deviation $5\times10^{-14}$, reaching $(r,\vartheta)=(2.8284,0.7854)$ at $t=2$, although both $r(t)$ and $\vartheta(t)$ are curved functions of $t$. On the unit sphere the solutions are great circles, with constant speed.

**Straight is not shortest.** The book stresses (§5.6) that the geodesic of a connection and the shortest curve are different notions. Two ways to see it.
(i) On the sphere, the equator-tilted great circle from $(\theta,\varphi)=(\pi/2,0)$ with velocity $(0.6,0.8)$ run for length $3\pi/2$ is a geodesic, but it ends at angular distance $\pi/2$ from where it started: the other way round is three times shorter. A geodesic is only locally shortest.
(ii) **One start, one velocity, three connections.** On the Gaussian manifold start at $(\mu,\sigma)=(1,2)$ with velocity $(1,\tfrac12)$ and run for time $1$. The e-geodesic is a straight line in $\theta$ and ends at $(3.0000,2.8284)$ (in $\theta$: from $(0.25,-0.125)$ by $(0.125,0.0625)$ to $(0.375,-0.0625)$). The m-geodesic is a straight line in $\eta$, from $(1,5)$ by $(1,4)$ to $(2,9)$, ending at $(2.0000,2.2361)$. The Levi-Civita geodesic ends at $(2.2319,2.3885)$. Same manifold, same data, three answers.

<img src="figures/geodesics-gaussian.svg" alt="On the Gaussian half-plane, drawn in coordinates x = mu over root 2 and sigma, three curves leave the point (1, 2) with the same velocity: the e-geodesic (blue) climbs fastest and ends at (3.00, 2.83) at t = 1, the m-geodesic (orange) bends over and ends at (2.00, 2.24), and the Levi-Civita geodesic (green) is between them, ending at (2.23, 2.39). Right: the speed in the Fisher metric along each: it rises from 0.612 to 1.581 along the e-geodesic, falls to 0.447 along the m-geodesic and stays 0.612 along the Levi-Civita geodesic.">

*Speed.* Only Levi-Civita keeps the speed $\sqrt{g(\dot\xi,\dot\xi)}$ constant ($0.6124$ throughout). Along the e-geodesic it grows from $0.6124$ to $1.5811$, along the m-geodesic it falls to $0.4472$ (picture, right). This is the metric-compatibility of §9 in action: $\tfrac{d}{dt}g(\dot\xi,\dot\xi)=2g(\nabla_{\dot\xi}\dot\xi,\dot\xi)$ vanishes along a geodesic only if $\nabla$ preserves $g$.
*Length.* Between $N(0,1)$ and $N(3,4)$ the Fisher–Rao distance is $2.136237$ (closed form of Chapter 1). The e-geodesic between them has length $2.234252$ (**$4.59\%$ longer**), the m-geodesic $2.185904$ ($2.32\%$ longer), the straight line in $(\mu,\sigma)$ $2.298909$ ($7.61\%$ longer). Only the Levi-Civita geodesic achieves the distance (§9). The third widget lets you move the start, the velocity and a dial $s$ between e ($s=0$) and m ($s=1$) and prints the length-to-distance ratio of each.

## 7. Parallel transport (§5.7)

**In plain words.** Carry a vector along a curve while asking the connection to keep it "the same": its covariant derivative along the curve is zero, $\nabla_{\dot\xi}A=0$ (5.57), i.e.

$$\dot A^i+\Gamma_{jk}{}^i\,\dot\xi^j A^k=0\qquad(5.58).$$

A geodesic is a curve that carries its own velocity. The transported vector generally depends on the path (5.59), and that dependence is curvature (§8).

*Sphere.* Take the coordinate rectangle $\theta\in[0.8,1.6]$, $\varphi\in[0.2,1.4]$, area $(\cos0.8-\cos1.6)(1.4-0.2)=0.871087$, and go $P\to Q\to R\to S\to P$ along its four sides starting with a vector along $e_\theta$. The vector comes back turned by $0.871087$ rad counter-clockwise, exactly $K\times\text{area}$ with $K=1$, its length unchanged. The two routes from $P$ to $R$ (via $Q$, via $S$) arrive $0.871087$ rad apart: the transported vector depends on the path.
Round the circle of colatitude $\pi/4$ (it encloses a cap of area $2\pi(1-\cos\theta)=1.840302$) the vector returns rotated by $1.840302$ rad: relative to the moving frame it turned by $-2\pi\cos\theta=-4.442883$, and $-4.442883\equiv1.840302\pmod{2\pi}$.

*Plane in polar coordinates.* Round the circle $r=2$ the *polar components* of a transported vector change continuously (at $\vartheta=\pi$ the vector that started as $(1,0)$ has components $(-1,0)$), but the Cartesian arrow is the same at the start, halfway and the end, net rotation $10^{-12}$. Components are not vectors.

*Gaussians.* Take $\mu\in[0,1.5]$, $\sigma\in[1,2]$, with Fisher area $\sqrt2\,\Delta\mu\,(1/\sigma_1-1/\sigma_2)=1.060660$. The three connections behave differently:

| Connection | Vector after the loop | Rotation | Fisher length of $A$ along the loop |
|---|---|---|---|
| Levi-Civita | $(0.8626,-0.3577)$ | $-0.530330$ rad $=K\times\text{area}$, $K=-\tfrac12$ | constant $1.0000$ |
| e | $(1,0)$, exactly back | $0$ | $1.0000$ rising to $2.0000$, back to $1$ |
| m | $(1,0)$, exactly back | $0$ | $0.5000$ to $2.3452$, back to $1$ |

The sign of the rotation is opposite to the sphere's, because $K<0$. The e- and m-connections return the vector exactly (they are flat) but do not preserve its length along the way (they are not metric). In the flat case parallel transport is path independent: for the e-connection the two routes from $P$ to $R$ give the same vector (difference exactly $0$), and its $\theta$-components are constant, $(1,0)\to(1,0)$.

<img src="figures/holonomy.svg" alt="Left: the unit sphere seen from outside with the coordinate rectangle theta in 0.8 to 1.6, phi in 0.2 to 1.4 and a vector transported round it; the start (blue) and the arrival (orange) differ by 0.871 radians counter-clockwise. Right: the Gaussian half-plane in coordinates x = mu over root 2 and sigma with the rectangle mu in 0 to 1.5, sigma in 1 to 2; the arrival is rotated by 0.530 radians clockwise.">

## 8. Riemann–Christoffel curvature (§5.8)

### 8.1 The round-the-world derivation

**In plain words.** Because transport depends on the path, the amount of dependence over an infinitesimal loop measures the curvature. Take a small parallelogram $P,Q,R,S$ with sides $d_1\xi$ ($P\to Q$, $S\to R$) and $d_2\xi$ ($Q\to R$, $P\to S$).
Carry $A$ along $P\to Q\to R$. On the first leg, $d_1A^i=-\Gamma_{jk}{}^iA^kd_1\xi^j$. On the second leg the symbol must be evaluated at $Q$, i.e. expanded to first order, and the vector is already $A+d_1A$, giving the extra change
$$\delta_{12}A^i=-\Gamma_{jk}{}^iA^k\,d_2\xi^j-\partial_l\Gamma_{jk}{}^i\,A^k\,d_1\xi^ld_2\xi^j+\Gamma_{jk}{}^i\Gamma_{lm}{}^kA^m\,d_1\xi^ld_2\xi^j\qquad(5.63).$$
The other route $P\to S\to R$ is the same with $d_1\xi\leftrightarrow d_2\xi$, **in every term**. The first-order terms of the two routes are equal and cancel in the difference; what is left is, writing $X_{ab}{}^i{}_k=\partial_a\Gamma_{bk}{}^i+\Gamma_{am}{}^i\Gamma_{bk}{}^m$,
$$A_{21}-A_{12}=X_{ab}{}^i{}_k\,A^k\,(d_1\xi^ad_2\xi^b-d_2\xi^ad_1\xi^b)=(X_{ab}{}^i{}_k-X_{ba}{}^i{}_k)\,A^k\,d_1\xi^ad_2\xi^b=R_{abk}{}^i\,A^k\,d_1\xi^a\,d_2\xi^b,$$
with $R_{ijk}{}^l=\partial_i\Gamma_{jk}{}^l-\partial_j\Gamma_{ik}{}^l+\Gamma_{im}{}^l\Gamma_{jk}{}^m-\Gamma_{jm}{}^l\Gamma_{ik}{}^m$, which is (5.66). $R$ is a tensor, with two antisymmetric directions $i,j$ and a "vector slot" $k$.

**Two slips in the printed version** (I checked the page images).

1. *(5.64) is not obtained by exchanging $d_1\xi$ and $d_2\xi$ in (5.63).* The first two terms are exchanged but the last, $\Gamma_{jk}{}^i\Gamma_{lm}{}^kA^m\,d_2\xi^jd_1\xi^l$, is printed unchanged. If it were right, the $\Gamma\Gamma$ terms would cancel in the difference and (5.65) would produce only $\partial_i\Gamma_{jk}{}^l-\partial_j\Gamma_{ik}{}^l$, not (5.66). On the sphere at $\theta=1$ that would give $R_{\theta\varphi\varphi}{}^\theta=\sin^2\theta-\cos^2\theta=0.4161$ instead of $\sin^2\theta=0.7081$ (the right-hand panel below).
2. *(5.65), (5.69) and (5.70) are a factor $2$ too large.* They write $A_{21}-A_{12}=R_{jkl}{}^iA^l(d_1\xi^jd_2\xi^k-d_1\xi^kd_2\xi^j)$, and the bracket equals $d_1\xi^jd_2\xi^k$ doubled once $R$'s antisymmetry is used: $R_{jkl}{}^i(d_1^jd_2^k-d_1^kd_2^j)=2R_{jkl}{}^id_1^jd_2^k$. The truth is $A_{21}-A_{12}=R_{jkl}{}^iA^ld_1\xi^jd_2\xi^k=\tfrac12R_{jkl}{}^iA^ldf^{jk}$ if $df^{jk}$ is the printed (5.67). The same $\tfrac12$ belongs in the surface integral (5.70) (or $df^{jk}$ should be read as half of (5.67)).
 *Orientation.* $A_{21}-A_{12}$ is the arrival difference of the two routes. Going round the loop $P\to Q\to R\to S\to P$ changes $A$ by $A_{12}-A_{21}$, the **negative** (the sphere's rotation above is $+K\,\text{area}$ for that loop, as $-R(d_1\xi,d_2\xi)A$ requires).

*Numerical test of the factor.* Loops of size $\varepsilon$ with $d_1\xi=\varepsilon(1,0.3)$, $d_2\xi=\varepsilon(-0.2,1)$ and $A=(0.3,0.8)$, transported with a fourth-order Runge–Kutta integrator and compared with $R_{jkl}{}^iA^ld_1^jd_2^k$ (not the printed bracket):

| $\varepsilon$ | 0.2 | 0.1 | 0.05 | 0.02 |
|---|---|---|---|---|
| sphere at $(1,0.4)$: $\lvert A_{21}-A_{12}\rvert/\lvert RAd_1d_2\rvert$ | 1.0374 | 1.0214 | 1.0115 | 1.0048 |
| Gaussians, Levi-Civita at $(1,2)$ | 0.9912 | 0.9960 | 0.9981 | 0.9993 |
| Gaussians, e-connection ($R=0$): $\lvert A_{21}-A_{12}\rvert$ | $8\times10^{-14}$ | $2\times10^{-15}$ | $9\times10^{-16}$ | $5\times10^{-16}$ |

The ratio tends to $1$, not to $2$ (left panel below), and the change of $A$ round the whole loop, divided by $-RAd_1d_2$, also tends to $1$ ($1.0051$ and $0.9871$ at $\varepsilon=0.02$). The fifth widget lets you shrink $\varepsilon$ and see the measured vector land on the black arrow, not the red one.

<img src="figures/loop-factor.svg" alt="Left: measured ratio of the difference of the two transports round a small quadrilateral to R A d1 d2, for the sphere and the Gaussian manifold, against the size of the loop: both tend to 1, while the printed formula would give 2. Right: the sphere's curvature component R_theta phi phi^theta as sin squared theta from the full formula, and as sin squared minus cosine squared if the Gamma Gamma terms were missing; at theta = 1 the values are 0.708 and 0.416.">

### 8.2 Curvature as non-commuting derivatives, and the abstract form

Partial derivatives commute, covariant derivatives do not (5.71)–(5.72): $(\nabla_{e_i}\nabla_{e_j}-\nabla_{e_j}\nabla_{e_i})X=R_{ijk}{}^lX^ke_l$ (5.74). The modern definition is $R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z$ (5.75), where $[X,Y]=XY-YX$ is the commutator of the vector fields (5.76); for coordinate fields $[e_i,e_j]=0$ and the last term disappears, which gives (5.66) back: $R(e_i,e_j)e_k=\nabla_{e_i}(\Gamma_{jk}{}^me_m)-\nabla_{e_j}(\Gamma_{ik}{}^me_m)=\left(\partial_i\Gamma_{jk}{}^l-\partial_j\Gamma_{ik}{}^l+\Gamma_{jk}{}^m\Gamma_{im}{}^l-\Gamma_{ik}{}^m\Gamma_{jm}{}^l\right)e_l=R_{ijk}{}^le_l$.
*Check.* For the polynomial field $X=(1+2\theta-\varphi+\theta\varphi,\ 0.5+\theta+\varphi^2)$ and the commutator computed by differencing $\nabla_i(\nabla_jX)$ numerically: on the sphere at $(1,0.4)$ it is $(1.1754,-3.0)$ and $R_{12k}{}^lX^k=(1.1754,-3.0)$; on the Gaussians with Levi-Civita at $(1,2)$ it is $(-1.375,0.375)$ on both sides; for the e-connection and for the polar plane both sides are $0$ (all differences $\le2\times10^{-10}$).

### 8.3 Flat manifolds, and curvature of the examples

$R=0$ means parallel transport does not depend on the path. Transport a basis of one tangent space everywhere; the resulting parallel vector fields have $\nabla e_j=0$, and the coordinate lines tangent to them are geodesics, giving a chart with $\Gamma=0$ everywhere, an **affine chart** (5.77)–(5.78). Conversely $\Gamma=0$ in some chart makes $R=0$ by (5.66).

| Space and connection | $\max\lvert R_{ijk}{}^l\rvert$ | Curvature | Affine chart |
|---|---|---|---|
| plane, polar, Levi-Civita | $10^{-11}$ | flat | Cartesian |
| unit sphere, Levi-Civita | $1$ | $K=+1$ everywhere (3 points tried) | none |
| Gaussians, Levi-Civita | $0.25$ at $(1,2)$ | $K=-\tfrac12$ everywhere (3 points tried) | none |
| Gaussians, e | $2\times10^{-11}$ | flat | $\theta$ |
| Gaussians, m | $10^{-11}$ | flat | $\eta$ |

For the e-connection the frame $\partial\xi/\partial\theta$ at $(1,2)$, carried to $(2.5,1.2)$ along two different routes, equals the $\theta$-coordinate frame there to $1.6\times10^{-9}$, and for m the $\eta$-coordinate frame to $1.8\times10^{-10}$; for Levi-Civita the same vector arrives as $(0.5176,-0.2146)$ or $(0.3805,-0.3280)$ depending on the route, so there is no parallel frame at all. Geodesics of the e-connection are straight lines in $\theta$ to $1.7\times10^{-11}$.
In two dimensions $R$ is determined by one number: $R_{ijkl}=K(g_{jk}g_{il}-g_{ik}g_{jl})$ (verified to $10^{-10}$ for the sphere and the Gaussian Levi-Civita), and for a metric connection $R_{ijkl}$ is also antisymmetric in $(k,l)$. And $R$ really is a tensor: computing it from its own $\Gamma$ in the stereographic chart $(u,v)$ of the sphere agrees with the tensor law applied to $R$ in $(\theta,\varphi)$ to $2.4\times10^{-11}$.

**Between e and m.** Any average $(1-s)\Gamma^e+s\Gamma^m$ is again a connection. Checking the five values $s=0,\tfrac14,\tfrac12,\tfrac34,1$ at $(1,2)$: its curvature is $4s(1-s)$ times the Levi-Civita curvature (differences below $10^{-10}$), flat at the two ends and equal to Levi-Civita's at $s=\tfrac12$; and its failure of the metric condition (5.82) is $\lvert1-2s\rvert$ times that of e, zero only at $s=\tfrac12$. **So on this manifold no connection is both flat and metric**: the metric one is unique (§9) and has $K=-\tfrac12$. The fourth widget has this dial.

## 9. The Levi-Civita connection (§5.9)

**In plain words.** Ask transport to preserve lengths. The book notes that "lengths are preserved" and "inner products are preserved" are equivalent (5.79)–(5.80), and that for the frame it reads (5.81)

$$\partial_kg_{ij}=\Gamma_{kij}+\Gamma_{kji}\qquad(5.82),$$

i.e. $Z\langle X,Y\rangle=\langle\nabla_ZX,Y\rangle+\langle X,\nabla_ZY\rangle$ for all fields (5.83); such a connection is called *metric*.

**Theorem 5.1: a metric connection with $\Gamma_{ijk}=\Gamma_{jik}$ is unique and given by**
$$\Gamma_{ijk}=\tfrac12\left(\partial_ig_{jk}+\partial_jg_{ik}-\partial_kg_{ij}\right)\qquad(5.85).$$
*Proof* (the book leaves it as an exercise). Write (5.82) three times with the indices permuted and combine: $\partial_kg_{ij}+\partial_ig_{jk}-\partial_jg_{ki}=(\Gamma_{kij}+\Gamma_{kji})+(\Gamma_{ijk}+\Gamma_{ikj})-(\Gamma_{jki}+\Gamma_{jik})$. Using the symmetry in the first two indices, $\Gamma_{kij}=\Gamma_{ikj}$, $\Gamma_{kji}=\Gamma_{jki}$, $\Gamma_{ijk}=\Gamma_{jik}$, six terms cancel in pairs and $2\Gamma_{ikj}$ remains. Rename the indices.
*Check by linear algebra* (a random $\partial g$ at a point). In $n=2$: 6 unknowns $\Gamma_{ijk}$ (with the symmetry), 6 equations (5.82), rank $6$, so the solution is unique, and (5.85) satisfies them ($1.4\times10^{-11}$). Without the symmetry: 8 unknowns, still rank 6, so a **2-parameter family** of metric connections; in general $n^2(n-1)/2$ free parameters ($2,9,24$ for $n=2,3,4$, all verified), which is exactly the number of components of an antisymmetric-in-two-indices tensor, the torsion that the book mentions in its closing remarks.

**Theorem 5.2: a shortest curve is a geodesic of Levi-Civita.** *Derivation.* Minimise the energy $\tfrac12\int g_{ij}\dot\xi^i\dot\xi^j\,dt$ (same minimisers as the length (5.86) once the curve is run at constant speed). Euler–Lagrange gives $\frac{d}{dt}(g_{ij}\dot\xi^j)-\tfrac12\partial_ig_{jk}\dot\xi^j\dot\xi^k=0$, i.e. $g_{il}\ddot\xi^l+(\partial_jg_{ik}-\tfrac12\partial_ig_{jk})\dot\xi^j\dot\xi^k=0$. Symmetrising in $(j,k)$, the bracket is $\tfrac12(\partial_jg_{ik}+\partial_kg_{ij}-\partial_ig_{jk})=\Gamma_{jki}$, and raising $i$ gives $\ddot\xi^l+\Gamma_{jk}{}^l\dot\xi^j\dot\xi^k=0$: equation (5.54) with the Levi-Civita symbol (5.85).
*Numerical test, without using that conclusion.* Between $N(0,1)$ and $N(3,4)$ I minimised the discrete energy of a polyline by Newton's method, with 12, 24 and 48 segments: lengths $2.136341$, $2.136263$, $2.136244$ against the closed-form Fisher–Rao distance $2.136237$ (the error shrinks like $1/N^2$). Then I integrated (5.54) with the Levi-Civita symbol, adjusting the initial velocity (shooting) until it hit the endpoint: initial velocity $(1.0518,1.3148)$, constant speed $2.136237$, which equals the distance, and at $t=k/48$ it lies within $1.5\times10^{-4}$ of the $k$-th vertex of the minimiser. The e- and m-geodesics and the straight line between the same endpoints are $4.59\%$, $2.32\%$ and $7.61\%$ longer (§6).
The converse of the theorem is false (the $3\pi/2$ arc on the sphere is a geodesic but not minimal, §6); Theorem 5.2 says minimal $\Rightarrow$ geodesic.

## 10. Submanifolds and embedding curvature (§5.10)

**In plain words.** A surface in space, a curve on a surface, a model inside a bigger model. Let the submanifold have coordinates $u^a$ ($m$ of them) inside an $n$-dimensional manifold with coordinates $\xi^i$, so $\xi=\xi(u)$. Its tangent vectors are $e_a=B_a{}^ie_i$ with $B_a{}^i=\partial\xi^i/\partial u^a$ (5.89)–(5.91), and the ambient metric restricts to
$$g_{ab}=B_a{}^iB_b{}^jg_{ij}\qquad(5.94).$$

**Induced connection and embedding curvature.** Differentiate the submanifold's basis with the ambient connection: $\nabla_{e_a}e_b=\Gamma_{ab}{}^ke_k$ with $\Gamma_{ab}{}^k=B_a{}^i\partial_iB_b{}^k+B_a{}^iB_b{}^j\Gamma_{ij}{}^k$ (5.97). This vector need not be tangent to the submanifold. Split it orthogonally, with respect to the metric, into a tangential part, which defines the **induced connection** $\Gamma_{abc}=B_a{}^iB_b{}^jB_c{}^k\Gamma_{ijk}+B_c{}^j\,\partial_aB_b{}^i\,g_{ij}$ (5.100), and a normal part,
$$H_{ab}{}^\kappa e_\kappa=(\nabla_{e_a}e_b)^\perp,\qquad H_{ab\kappa}=\langle\nabla_{e_a}e_b,e_\kappa\rangle\qquad(5.101\text{–}5.102),$$
the **embedding (Euler–Schouten) curvature**: how the submanifold bends within the ambient manifold. It is the extrinsic curvature; the Riemann–Christoffel curvature of the induced connection is the intrinsic one.

<img src="figures/cylinder.svg" alt="Left: a cylinder in three-dimensional space with a helix; at one point the velocity (blue) is tangent and the acceleration (orange) points at the axis, normal to the surface. Right: the same cylinder unrolled into a flat strip, where the helix is a straight line.">

*The cylinder.* Radius $R=2$, coordinates $(\varphi,z)$, $x=(R\cos\varphi,R\sin\varphi,z)$ in Cartesian $\mathbb R^3$ where the ambient $\Gamma=0$. Then $g=\operatorname{diag}(4,1)$ and the induced connection (5.100) is zero (largest entry $5\times10^{-10}$), so its Riemann–Christoffel curvature is $0$: inside, Euclidean geometry holds. But with the outward normal $n$, $H_{ab}=\langle\partial_a\partial_bx,n\rangle=\operatorname{diag}(-2,0)$, i.e. normal curvature $H_{\varphi\varphi}/g_{\varphi\varphi}=-1/R=-0.5$ in the $\varphi$ direction and $0$ along $z$: **intrinsically flat, extrinsically curved**, as the book says. A geodesic of the sheet is a helix; in space its acceleration, $(-0.9855,-4.3908,0)$ for $(\varphi,z)=(1.5t,0.7t)$ at $t=0.9$, is purely normal (tangential part $4.5\times10^{-8}$) and has length $R\omega^2=4.5$; unrolled it is the straight line $z=(c/R\omega)\,s$.
*The sphere,* by contrast, of radius $\rho=2$: $H_{ab}=-g_{ab}/\rho$ (error $5\times10^{-9}$), and the intrinsic $K=1/\rho^2=0.25$.
*How the two curvatures are related* (not in the book, but it is what the numbers say): for a surface in Euclidean space $K=\det H/\det g$ (Gauss's equation). Cylinder: $\det H=0$, so $K=0$ although $H\neq0$. Sphere: $\det H/\det g=0.250000=K$. A flat sheet can be bent into a cylinder because one principal curvature is $0$; it cannot be bent into a sphere. The sixth widget bends the sheet.
*A check that the induced connection is Levi-Civita:* for the sphere and the cylinder the connection (5.100) induced from Cartesian $\mathbb R^3$ equals the Levi-Civita symbol (5.85) of the induced metric (5.94), to $2\times10^{-9}$ and $5\times10^{-10}$. That is the standard fact that projecting the ambient derivative is the same as the Riemannian connection of the induced metric, when the ambient connection is metric.

**Embedding curvature depends on the ambient connection.** A curve of Gaussians is a one-dimensional submanifold; its embedding curvature is the normal part of $\nabla_{\dot c}\dot c$. At $(1,2)$ with velocity $(1,\tfrac12)$ the Fisher length per unit speed squared of that normal part is

| Curve \\ ambient connection | e | m | Levi-Civita |
|---|---|---|---|
| $\theta$-line (e-geodesic) | $0$ | $0.7698$ | $0.3849$ |
| $\eta$-line (m-geodesic) | $0.7698$ | $0$ | $0.3849$ |
| Fisher–Rao geodesic | $0.3849$ | $0.3849$ | $0$ |

A curve has zero embedding curvature exactly for the connection that defines it as straight. The book returns to this in Part III, where the flat submanifolds that matter statistically are those with zero embedding curvature for e (e-flat) or for m (m-flat); this table is the one-dimensional version.

## The remarks the chapter ends on

Differential geometry is local here: curvature and geodesics, not the global topology. A **torsion** tensor, antisymmetric in two indices, is a further structure on a Riemannian manifold (Einstein's attempted unification; dislocations in continua; non-holonomic constraints in electromechanics and robotics); it is exactly the freedom counted in §9 and the book does not use it. What information geometry needs and textbook Riemannian geometry does not provide is a **pair of connections dual with respect to the metric**, which is Chapter 6; the e/m pair above is the example, $\partial_kg_{ij}=\Gamma^e_{kij}+\Gamma^m_{kji}$ (residual $3\times10^{-9}$), the two-connection analogue of (5.82), which is why the e- and m-connections, neither metric on its own, can be flat while the metric one is not.

## Checks of the book's statements

| Where | Statement | What I found |
|---|---|---|
| (5.6)–(5.9), (5.28) | basis vectors transform with $J$, components with the inverse | verified for the Gaussians at $(1,2)$: $\partial_{\theta_1}=4\partial_\mu$, $\partial_{\theta_2}=8\partial_\mu+8\partial_\sigma$, $A=(1,\tfrac12)\mapsto(0.125,0.0625)$; the score $A^i\partial_i\log p$ is the same function of $x$ ($10^{-10}$) |
| (5.10), (5.13) | scores are tangent vectors, Fisher information is the inner product | verified: $\mathbb E[f^2]=A^{\mathsf T}gA=0.375$, $\mathbb E[f]=-2\times10^{-17}$ |
| (5.12) | metric transforms with two Jacobians | verified: $\left[\begin{smallmatrix}4&8\\8&48\end{smallmatrix}\right]=\operatorname{Cov}[(x,x^2)]=\nabla^2\psi$ |
| (5.33) | cubic tensor is a symmetric tensor | verified; in $\theta$ it is $\nabla^3\psi=(0,32,128,896)$ |
| (5.35) | Hessian is a tensor only at a critical point | verified: extra term $\left[\begin{smallmatrix}0&175.78\\175.78&1567.38\end{smallmatrix}\right]$ off a critical point, $0$ at one |
| (5.37)–(5.41) | $\Gamma$ transforms inhomogeneously | verified: $\Gamma_{\vartheta\vartheta}{}^r=-r$, $\Gamma_{r\vartheta}{}^\vartheta=1/r$ from the Cartesian chart; Gaussian Levi-Civita in $\theta$ is $(0,16,64,448)$ from the law, tensor part alone $(0,16,0,-192)$ |
| (5.48) | $\nabla_iX^k$ is a tensor | verified ($2\times10^{-10}$); partial derivatives alone are not ($1.17$) |
| (5.54), (5.55) | geodesic equation; reparametrisation | verified: straight line in polar coordinates to $5\times10^{-14}$; $\xi(t)=\gamma(t^3)$ gives $c=2/t$ |
| (5.58), (5.59) | transport; path dependence | verified: sphere $0.871087=K\,\text{area}$; Gaussians $-0.530330$; polar plane $10^{-12}$ |
| (5.63) | $\delta_{12}A$ | verified |
| **(5.64)** | "exchange $d_1\xi$ and $d_2\xi$ in (5.63)" | **the $\Gamma\Gamma$ term is printed unexchanged**; used as printed it would give $R_{\theta\varphi\varphi}{}^\theta=0.416$ instead of $0.708$ at $\theta=1$ |
| **(5.65), (5.69), (5.70)** | $A_{21}-A_{12}=R_{jkl}{}^iA^l(d_1^jd_2^k-d_1^kd_2^j)$; loop integral | **a factor $2$ too large**: measured ratio to $R A d_1d_2$ is $1.0048$ (sphere) and $0.9993$ (Gaussians) at $\varepsilon=0.02$; the loop $P\to Q\to R\to S\to P$ changes $A$ by the negative |
| (5.66) | RC tensor, a tensor | verified (stereographic chart $2.4\times10^{-11}$); $K=+1$ sphere, $-\tfrac12$ Gaussians, flat for polar plane, e and m |
| (5.74)–(5.75) | commutator of $\nabla$'s is $R$ | verified on four examples ($\le2\times10^{-10}$) |
| (5.77)–(5.78) | flat $\Rightarrow$ affine coordinates | verified: parallel frame = coordinate frame of $\theta$ ($1.6\times10^{-9}$) and of $\eta$ ($1.8\times10^{-10}$) |
| (5.82) | metric compatibility | Levi-Civita $10^{-11}$; e and m fail by $1.0$ |
| Theorem 5.1 | unique symmetric metric connection (5.85) | verified: rank $=$ unknowns for $n=2,3,4$; without symmetry $n^2(n-1)/2$ free parameters |
| Theorem 5.2 | shortest curve is a Levi-Civita geodesic | verified: minimiser $2.136244$ vs $2.136237$; shooting equals the distance; converse false ($3\pi/2$ arc) |
| (5.94), (5.100) | induced metric and connection | verified; equals Levi-Civita of the induced metric ($2\times10^{-9}$) |
| §5.10 | cylinder: RC curvature $0$, embedding curvature $\neq0$ | verified: $H=\operatorname{diag}(-2,0)$ for $R=2$; $K=\det H/\det g=0$ |

## Questions and doubts

- **Which "flat"?** Section 5.2 says a manifold is locally flat "when and only when" the Riemann–Christoffel tensor vanishes, and links it to the existence of a chart with $g=\delta$. That is true for the Levi-Civita connection. The Gaussian manifold is flat for e and for m ($R=0$, affine charts $\theta$ and $\eta$) but not Euclidean: Levi-Civita has $K=-\tfrac12$ and no chart makes the Fisher metric $\delta$. So "flat" needs the connection named, and the whole of the book's Part I rests on flatness for a connection that is *not* the metric one.
- **The factor 2 and the loop orientation.** I derived $A_{21}-A_{12}=R_{jkl}{}^iA^ld_1^jd_2^k$ and tested it numerically, so I am confident, but I would like to know whether the book's $df^{jk}$ in (5.67) was meant with a $\tfrac12$ that was dropped in typesetting, since (5.70) as printed would then be right. The sign convention matters later: which sign of $K$ and which orientation of the loop the book intends is not stated.
- **Is $R=0$ enough for affine coordinates?** The sketch in §5.8.3 builds the chart from parallel frames and geodesics. For the coordinate lines to close up into a chart the parallel vector fields must commute, $[e_i,e_j]=\nabla_{e_i}e_j-\nabla_{e_j}e_i=0$, and that is the torsion-free condition. So the statement needs $R=0$ **and** symmetry of $\Gamma$ (a connection with torsion can have parallel frames and still no chart with $\Gamma=0$). All connections used in the book are symmetric, so nothing is lost there, but the chapter does not say it.
- **How is "one-to-one map between $T_\xi$ and $T_{\xi'}$" meant?** (§5.3) A connection fixes the map only along a path; different paths give different maps (holonomy). The wording suggests a single map for each pair of nearby points, which is only true to first order.
- **Embedding curvature for non-metric ambient connections.** The orthogonal decomposition (5.98) uses the metric, but for the e- or m-connection the ambient transport is not metric, and the induced connection (5.100) need not be the one a submanifold would get from its own structure. The table in §10 uses the same projection for all three. I expect Part III to define e- and m-embedding curvatures this way and to show that e-flat means $H^{(e)}=0$, which the table is consistent with; I have not checked the general statement.
- **What $\nabla\nabla X$ means in (5.74).** The printed $(\nabla_{e_i}\nabla_{e_j}-\nabla_{e_j}\nabla_{e_i})X$ can be read as the iterated directional derivative or as the second covariant derivative tensor. For coordinate vector fields and a symmetric connection they give the same antisymmetrised result, which is what I computed; for a connection with torsion they would differ by a torsion term.
- **Which parameter?** (5.54) holds for a special "affine" parameter $t$ (§6, the $c(t)$ remark). For the e- and m-connections that parameter is the one in which the curve is a straight line in $\theta$ or $\eta$ at constant speed in those charts; it is not Fisher arc length, and the two coincide only for Levi-Civita (speeds in the table of §6 change along e and m curves).

## Takeaways

- **A connection is extra data, and one manifold can carry many.** Same Gaussian manifold, same point, same velocity: three connections, three different straight lines (endpoints $(3.00,2.83)$, $(2.00,2.24)$, $(2.23,2.39)$). Geodesic means "straight for $\nabla$"; shortest is a statement about the metric.
- **$\Gamma$ is a property of the chart as much as of the geometry.** It is not a tensor; it is $0$ in Cartesian and $-r,1/r$ in polar coordinates for the same flat plane. Differences of connections *are* tensors, and on a statistical manifold the difference of the m- and e-connections is the cubic tensor $T=\nabla^3\psi$.
- **Components are not vectors.** The Cartesian-constant field has non-zero partial derivatives in polar coordinates and zero covariant derivative; $e_r$ has zero partials and non-zero covariant derivative.
- **Curvature is path dependence**, quantitatively: rotation $=K\times$area (sphere $K=1$, Gaussians $K=-\tfrac12$, opposite senses), with $A_{21}-A_{12}=R_{jkl}{}^iA^l\,d_1\xi^jd_2\xi^k$ and no factor 2.
- **Flat and metric are different.** e and m are flat and not metric; Levi-Civita is metric and not flat; no connection is both on the Gaussian manifold, and between e and m the curvature is $4s(1-s)$ times Levi-Civita's.
- **Intrinsic is not extrinsic.** A cylinder is flat inside and has $H\neq0$; $K=\det H/\det g$ for surfaces in Euclidean space. For a curve in a manifold with several connections, the embedding curvature depends on which connection you use.
- **Two printed slips to know about:** the last term of (5.64), and the factor 2 in (5.65), (5.69), (5.70).

| Term | One line |
|---|---|
| tangent vector | $A=A^i\partial_i$, a derivative operator; for distributions the score $A^i\partial_i\log p$ |
| metric | $g_{ij}=\langle e_i,e_j\rangle$; Fisher: $\mathbb E[\partial_il\,\partial_jl]$; $g_{\kappa\lambda}=J_\kappa^iJ_\lambda^jg_{ij}$ |
| connection | $de_i=\Gamma_{ki}{}^je_jd\xi^k$; not a tensor: $\Gamma'=JJJ\Gamma+(\partial J)Jg$ |
| covariant derivative | $\nabla_iX^k=\partial_iX^k+\Gamma_{ij}{}^kX^j$ |
| geodesic | $\ddot\xi^k+\Gamma_{ij}{}^k\dot\xi^i\dot\xi^j=0$ (straight, not necessarily shortest) |
| parallel transport | $\dot A^i+\Gamma_{jk}{}^i\dot\xi^jA^k=0$ |
| curvature | $R_{ijk}{}^l=\partial_i\Gamma_{jk}{}^l-\partial_j\Gamma_{ik}{}^l+\Gamma_{im}{}^l\Gamma_{jk}{}^m-\Gamma_{jm}{}^l\Gamma_{ik}{}^m$; $A_{21}-A_{12}=R_{jkl}{}^iA^ld_1\xi^jd_2\xi^k$ |
| flat | $R=0\iff$ affine chart with $\Gamma=0$; parallel transport is path independent |
| Levi-Civita | $\Gamma_{ijk}=\tfrac12(\partial_ig_{jk}+\partial_jg_{ik}-\partial_kg_{ij})$: metric, symmetric, unique; geodesics are shortest |
| induced geometry | $g_{ab}=B_a^iB_b^jg_{ij}$; $H_{ab}{}^\kappa$ = normal part of $\nabla_{e_a}e_b$ |
| Gauss's equation | surface in Euclidean space: $K=\det H/\det g$ |

---

*Notes written 2026-10-02.*
