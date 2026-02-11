They use **no information** about the likely "direction" of the goal. It only has the information provided by the problem formulation: initial state, set of actions, goal test, cost function.

Strategies include:
- [[Breadth First Search (BFS)]]
- [[Depth First Search (DFS)]]
- [[Depth Limited Search (DLS)]]
- [[Uniform Cost Search (UCS)]]
- [[Iterative Deepening Search (IDS)]]
- [[Bidirectional Search]]

## Comparing Search Strategies on Same Graph

**Graph structure**:
```
       S
      /|\
   3 / | \ 20
    A  B  C
   /|  |  |\
 3/ |8 |7 | \5
 D  E  F  G (Goal)
```
Edges: S-A(3), S-B(1), S-C(20), A-D(3), A-E(8), B-F(1), C-G(5), A-G(15)

### Breadth-First Search (BFS)
- Expanded: S, A, B, C, D, E, G
- Solution: S→A→G, cost = 3 + 15 = 18

### Depth-First Search (DFS) (left-first)
- Expanded: S, A, D, E, G
- Solution: S→A→G, cost = 18

### Uniform-Cost Search (UCS)
- Expanded: S, A, D, B, C, E, G
- **Optimal solution**: S→C→G, cost = 8 + 5 = 13

### Iterative Deepening Search (IDS)
- Expanded: S | S,A,B,C | S,A,D,E,G
- Solution: S→A→G, cost = 18

**Key point**: Only UCS guarantees optimality when step costs differ
## DFS vs DLS vs IDS Summary

| Aspect | DFS | DLS | IDS |
|--------|-----|-----|-----|
| Complete | No | No | Yes |
| Optimal | No | No | Yes |
| Space | Low | Low | Low |
| Risk | Infinite depth | Cutoff | Re-expansion |

**Key Takeaways**:
- DFS is memory-efficient but dangerous alone
- DLS controls depth but may miss solutions
- IDS is the safest default when depth is unknown
- **Rule of thumb**: If you don't know the depth — use IDS

## Wrap-Up: Choosing a Search Strategy

| Method | Complete? | Optimal? | Key Cost |
|--------|-----------|----------|----------|
| BFS | Yes (finite $b$) | Yes if unit step cost | Time/space $O(b^d)$ |
| DFS | No (infinite depth/loops) | No | Space $O(bm)$, time can be $O(b^m)$ |
| UCS | Yes (if $\epsilon > 0$) | Yes | Time/space $O\left(b^{\left\lceil\frac{C^*}{\epsilon}\right\rceil + 1}\right)$ |
| DLS | Yes if $d \leq \ell$ | No | Time $O(b^\ell)$, space $O(b\ell)$ |

**Notation**:
- $b$ = branching factor
- $d$ = depth of optimal solution
- $m$ = maximum depth
- $\ell$ = depth limit
- $C^*$ = optimal solution cost
- $\epsilon$ = minimum step cost

**Takeaway**: Constraints and cost model decide what is "best" (BFS: depth, UCS: path cost, DFS/DLS: memory)
