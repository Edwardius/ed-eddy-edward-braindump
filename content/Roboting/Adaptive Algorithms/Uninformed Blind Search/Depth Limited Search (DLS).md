DLS is Depth-First Search with a depth bound $\ell$

**Motivation**: DFS can get stuck in:
- Infinitely deep paths, or
- Cycles / looping behavior

**Rule**: Expand like DFS, but do not expand beyond depth $\ell$

**Outcome**:
- Finds a solution only if the goal depth $d \leq \ell$
- Otherwise returns cutoff/failure even if a solution exists

**Summary**: DLS avoids DFS non-termination by enforcing a maximum depth

### Properties and Complexity

**Properties**:
- Terminates (always)
- Complete if a solution exists within the bound: $d \leq \ell$
- If $\ell \ll d$, DLS may waste effort compared to BFS
- If $\ell < d \Rightarrow$ DLS fails despite existing solution
- Not optimal (can return a deeper / higher-cost solution)

**Complexity**:
- Time: $O(b^\ell)$
- Space: $O(b\ell)$ (linear in depth)

### Cutoff Intuition

DLS expands like DFS, but does not expand nodes deeper than $\ell$

**Example**: With depth limit $\ell = 2$:
```
       S (depth 0)
      /|\
     A B C (depth 1)
    /| |\ |\
   D E F G H (depth 2)
   | |
   I J (blocked - depth 3)
   | |
   K (blocked - depth 3)
```

If the goal depth $d > \ell$, it will be missed even if it is "one edge away"

Nodes I and J are cut off, even if they contain a goal