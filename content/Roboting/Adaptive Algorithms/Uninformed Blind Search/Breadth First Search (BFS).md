### Strategy
Search through a tree one level at a time:
- Traverse through one entire level of children nodes first, before moving on to traverse through the grandchildren nodes
- Traverse through an entire level of grandchildren nodes before going on to traverse through great-grandchildren nodes

### Mechanics

**Core principle:** Expand the shallowest unexpanded node first

**Data structure:** FIFO queue (First-In, First-Out)
- Fringe is maintained as a FIFO queue
- New successors go to the **end** of the queue

### BFS Algorithm Steps

```
BREADTH-FIRST-SEARCH:
1. Initialize fringe with initial state
2. Loop:
   a. If fringe is empty, return failure
   b. Remove first node from fringe (FIFO)
   c. Test if it is the goal state
   d. If goal, return solution
   e. Otherwise, expand node and add children to end of fringe
```

### Complexity Analysis

#### Upper-bound Case
Goal is the last node at depth $d$

#### Number of Generated Nodes

At each depth level:
- $d=0$: $b^0 = 1$
- $d=1$: $b^1 = b$
- $d=2$: $b^2$
- $d=3$: $b^3$
- ...
- $d=d$: $b^d$

**Total states generated:**
$$\text{Total} = 1 + b + b^2 + b^3 + \ldots + b^d = \frac{b^{d+1} - 1}{b - 1} = O(b^{d+1})$$

#### Alternative Formula (when goal is found at depth d)

We generate all nodes at depths 0 through $d$, plus we would have started generating some nodes at depth $d+1$ before finding the goal.

$$\text{Total} = 1 + b + b^2 + b^3 + \ldots + b^d + (b^{d+1} - b) = O(b^{d+1})$$

### BFS Properties

**Completeness:** Complete, if $b$ is finite

**Optimality:** Optimal, if path cost is equal to depth (i.e., if all operators have the same cost)
- Guaranteed to return the shallowest goal at depth $d$

**Time Complexity:** $O(b^{d+1})$
- Exponential in the depth of the solution

**Space Complexity:** $O(b^{d+1})$
- Must store all nodes at current and previous levels
- Exponential space requirement

Where:
- $d$ = depth of the solution
- $b$ = branching factor (number of children at each node)