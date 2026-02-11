**Strategy**: Expand the deepest unexpanded node first
- Follows one path as far as possible before backtracking

**Fringe discipline**: LIFO stack (explicit or via recursion)

**Search behavior**:
- Explores narrow paths deeply
- Postpones alternative branches

**Mental model**: "Go deep first, worry about breadth later"
### DFS Properties

**Completeness**:
- Not complete in infinite-depth or cyclic spaces
- May follow an infinite branch and never find a solution

**Optimality**:
- Not optimal
- First solution found may be arbitrarily bad

**Cost awareness**:
- Ignores path cost entirely

**Tradeoff**: DFS trades guarantees for speed and memory efficiency

### DFS Tree vs. Expansion Order

Assume children expanded left→right:

```
Tree structure:
       A
      /|\
     B E C
    /|   |\
   D H   F G
     |
     I

DFS expansion order (preorder): A, B, D, H, I, E, C, F, G
```

**Rule**: Go as deep as possible; when stuck, backtrack to the nearest ancestor with an unexpanded child

### DFS in Practice

**Strengths**:
- Very low memory usage: $O(b \cdot d)$
- Simple to implement
- Useful for:
  - Feasibility checking
  - Constraint satisfaction
  - Structural exploration

**Limitations**:
- Poor solution quality
- Sensitive to action ordering

**Key pedagogical role**: Motivates depth limits, heuristics, and backtracking control

## DFS vs BFS: Design Tradeoff

| Aspect | BFS | DFS |
|--------|-----|-----|
| Fringe | Queue (FIFO) | Stack (LIFO) |
| Completeness | Yes (finite) | No (infinite) |
| Optimality | Yes (unit cost) | No |
| Memory | Very high | Very low |
| Typical use | Shortest paths | Feasibility / structure |

**Key lesson**: Search strategy encodes what you care about