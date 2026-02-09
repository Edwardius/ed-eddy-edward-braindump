**NP** is when given a candidate, we can **check correctness** in polynomial time.
**NP-Hard** problems that are at least as hard as the hardest NP problems: **no known polynomial-time algorithm**

>[!info] For engineers, we trade a solution quality for time/memory/evaluations

So problem formulation and heuristics become very important!!

These labels are **scaling warnings:**
- combinatorial explosion: the number of candidates grows super fast
- search trees blow up
- evaluation is often the bottleneck
- budgets dominate over reaching the optimal solution

## Engineering Takeaway
**NP** if someone hands you a candidate solution, you can **verify** it efficiently
**NP-hard** no known polynomial-time algorithm
**NP-complete** problems that are both in NP and NP-hard (they are the "frontier")

>[!error] NP-hardness makes problem formulation and [[Heuristics]] design first-class engineering decisions rather than afterthoughts

