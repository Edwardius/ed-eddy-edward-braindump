A subset of [[Manifold]], part of [[Topology]] and [[What is Differential Geometry]]

>[!error] It is a class of manifolds that you can do calculus on.
## Formal Definition: Smooth Manifold

A **smooth manifold** (or $C^\infty$ manifold) is a topological manifold $M$ together with a maximal smooth atlas.

An **atlas** $\mathcal{A} = {(U_\alpha, \phi_\alpha)}_{\alpha \in I}$ is a collection of charts such that:

- $\bigcup_{\alpha \in I} U_\alpha = M$ (the charts cover $M$)
- Each $\phi_\alpha: U_\alpha \to V_\alpha \subseteq \mathbb{R}^n$ is a homeomorphism

An atlas is **smooth** if for all $\alpha, \beta \in I$ with $U_\alpha \cap U_\beta \neq \emptyset$, the **transition map**:

$$\phi_\beta \circ \phi_\alpha^{-1}: \phi_\alpha(U_\alpha \cap U_\beta) \to \phi_\beta(U_\alpha \cap U_\beta)$$

is a smooth ($C^\infty$) map between open subsets of $\mathbb{R}^n$.

A **maximal smooth atlas** is a smooth atlas that contains every chart compatible with it.
