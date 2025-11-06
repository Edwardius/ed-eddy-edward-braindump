It is a branch of pure mathematics that studies math in the context of abstract structures.
# Motivation
For a long time, many people studied math in the context of specific things: numbers, polynomials, geometric transformations. Some similar patterns emerge from all of them, so the field of **Abstract Algebra** emerged as a way to consolidate these similarities as a combined abstract idea.

---
**EXAMPLE (ADDITION)** For ALL mathematical objects, addition follows the same patterns.

- **Integers**: $3 + 5 = 8$
- **Polynomials**: $(x^2 + 2x) + (3x + 5) = x^2 + 5x + 5$
- **Matrices**: $\begin{bmatrix} 1 & 2 \ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \ 10 & 12 \end{bmatrix}$
- **Vectors**: $(1, 2, 3) + (4, 5, 6) = (5, 7, 9)$
- **Complex numbers**: $(3 + 4i) + (1 + 2i) = 4 + 6i$
- **Functions**: $(f + g)(x) = f(x) + g(x)$
- **Residue classes** (clock arithmetic): $9 + 5 = 2 \pmod{12}$ — like adding hours on a clock
- **Rational numbers**: $\frac{1}{3} + \frac{2}{5} = \frac{11}{15}$
- **Real numbers**: $\pi + e \approx 5.859$
- **Modular integers**: $7 + 8 \equiv 0 \pmod{5}$
- **Power series**: $(1 + x + \frac{x^2}{2} + \cdots) + (1 + 2x + 3x^2 + \cdots) = 2 + 3x + (\frac{1}{2} + 3)x^2 + \cdots$
- **Symmetry operations**: "rotate $90°$" + "rotate $90°$" = "rotate $180°$"
- **Translations in space**: "move 3m east" + "move 2m east" = "move 5m east"
- **Permutations**: Swapping positions $(1 \to 2, 2 \to 1)$ + $(2 \to 3, 3 \to 2)$ = combined rearrangement
- **Velocity vectors**: $10 \text{ m/s north} + 5 \text{ m/s north} = 15 \text{ m/s north}$
- **Force vectors**: $3N \text{ right} + 5N \text{ right} = 8N \text{ right}$
- **Electrical currents**: $2A + 3A = 5A$ (in parallel)
- **Probability distributions**: Convolution of distributions
- **Sets** (symmetric difference): ${1,2,3} \triangle {2,3,4} = {1,4}$
- **Logic propositions**: $(A \lor B)$ combined with $(C \lor D)$
- **Angles**: $30° + 45° = 75°$

All of these follow the **same fundamental rules** (commutative, associative, identity, inverse).

- **Commutative**: a + b = b + a (order doesn't matter)
- **Associative**: (a + b) + c = a + (b + c) (grouping doesn't matter)
- **Identity**: There's a "zero" where a + 0 = a
- **Inverse**: For every a, there's a -a where a + (-a) = 0

So these fundamental operations and their rules are interesting, but **how do I prove to people that these rules do in fact hold for this operation in every type of mathematical object?** 

>[!error] That's what the field of Abstract Algebra is about! It studies the patterns and properties that emerge from operations on mathematical objects, **identifying what's common across different contexts**!

---

The way we fundamentally analyze these similar patterns and properties is with [[Groups]]. These are sets of mathematical objects and an operator that, when it is shown to follow four important axioms, can get access to thousands of common patterns and properties that have been deduced from those axioms.

