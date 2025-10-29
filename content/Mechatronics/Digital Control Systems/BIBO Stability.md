# Boundedness
Let $u(t) / u[k]$ be a real-valued signal defined for $t\geq 0 / k \in \mathbb{Z}_{\geq_{0}}$. Then $u$ is bounded if there exists $\bar{u}\in\mathbb{R}$, $\bar{u}>0$ such that $-\bar{u} < u(t) < \bar{u}$ for all t and k greater than 0.

![[Pasted image 20251027140649.png]]
![[Pasted image 20251027140701.png]]

Both are bounded.

Signals that explode (either by design or unintentionally) are **unbounded**.

# Bounded Input Bounded Output (BIBO)
A LTI system $P(s) / G[z]$ is BIBO stable if every bounded input produces a bounded output.
- **Assuming $P(s) / G[z]$ is real, rational, and proper. Then they are stable if and only if it is BIBO stable.**

>[!info] BIBO Stable <> Stable (IF AND ONLY IF)

