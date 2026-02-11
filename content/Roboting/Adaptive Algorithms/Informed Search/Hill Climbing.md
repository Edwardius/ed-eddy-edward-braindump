1. **Tries to improve the efficiency of depth-first**
   - Informed depth-first algorithm

2. **Iterative improvement from an initial solution**
   - Starts with an arbitrary solution
   - Searches for a better one by incrementally changing a single element

3. **Sort successors by heuristic value**
   - Sorts the successors of a node (according to their heuristic values) before adding them to the list to be expanded

4. **Repeat until no improvement**
   - If the change produces a better solution, accept it and repeat until no further improvements can be found

## Hill Climbing Search: Strategy

### Idea
Looks one step ahead to determine if any successor is better than the current state; if there is, move to the best successor.

### Rule
If there exists a successor $s$ of the current state $n$ such that:
- $h(s) < h(n)$, and
- $h(s) \leq h(t)$ for all successors $t$ of $n$,

then move from $n$ to $s$. Otherwise, halt at $n$.

**Hill climbing uses only local information and does not backtrack.**
### Failure Modes

```
objective f(·)
      |
      |        global
      |        maximum
      |          /\
      |         /  \
      |        /    \
      |   /\  /      \
      | /   \/        \
      |/  local     shoulder/
      |  maximum     plateau
      |    flat local
      |    maximum
      |      current
      |      state
      └─────────────────────> state space
```

**Failure modes**:
1. Local maxima
2. Plateaus (shoulders)
3. Ridges
