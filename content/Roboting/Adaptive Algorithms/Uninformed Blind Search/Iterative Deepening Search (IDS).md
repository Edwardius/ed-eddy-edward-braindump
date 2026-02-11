Run [[Depth Limited Search (DLS)]] repeatedly with increasing depth: $\ell = 0, 1, 2, 3, \ldots$

**Combines**:
- DFS space efficiency
- BFS completeness

**Tradeoff**: Re-expands nodes — but cost is acceptable
### IDS Properties

- **Complete?** Yes (finite branching)
- **Optimal?** Yes (unit step costs)
- **Time**: $O(b^d)$
- **Space**: $O(bd)$

Same asymptotic time as BFS, space like DFS

### Search Tree Example

Left-to-right expansion:
```
depth 0:     S

depth 1:     A   B   C

depth 2:     D E F G
                   (Goal)

depth 3:     H I
```

Children are expanded strictly left-to-right

### IDS Cost: Re-expansion Analysis

**Example expansion counts per iteration**:
- $N(0) = 1$
- $N(1) = 3$
- $N(2) = 7$
- $N(3) = 5$ (stops when goal reached)

**Total expansions** performed by IDS until goal: $1 + 3 + 7 + 5 = 16$

**Unique nodes** touched (before finding goal): $\{S, A, B, D, E, F, G, H, I\} \Rightarrow 9$ unique nodes

**Why re-expansion is acceptable**:
- Most work is near the top of the tree (small depth), which is cheap
- As depth grows, the number of nodes at depth $d$ dominates anyway
- IDS keeps space like DFS: $O(bd)$, not $O(b^d)$ like BFS