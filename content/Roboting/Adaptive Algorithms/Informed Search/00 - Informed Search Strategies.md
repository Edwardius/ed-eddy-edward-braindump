Also known as heuristic search they **use information about the domain** to head in the general direction of the goal.

Algorithms include:
- Hill climbing
- Best-first search
- Greedy search
- Beam search
- A search 
- A* search

All informed search algorithms have a sort of **heuristic function** which provides an estimate of the "distance" to the goal. It guides the search towards promising states.

# Heuristic
For programming, it is a rule of thumb, simplification, or educated guess that reduces or limits the search space.
- They do not guarantee optimality
- may not guarantee feasibility
- Often have no theoretical guarantee

> [!error] This is opposed to an **algorithm** that is clearly defined, finite set of instructions or states that guarantees a solution. Assuming sufficient time and resources.

### Admissible Heuristic Function
It **never overestimates** the true minimal remaining cost to a goal:
$$
\forall n:h(n)\leq h^{*}(n)
$$
Where $h^{*}(n)$ is the optimal cost from n to a goal.

#### How do we design?
- solving a **relaxed** version of the original problem
	- This makes the problem easier and therefor the solution cost is a lower bound
	- Lower bounds never overestimate , **therefore admissible**
	- ie. A* for navigation assumes empty space when computing cost (straight line distance), so cost is lower than with obstacles.

#### Step-by-Step Recipe

1. **Start with the original problem**
   - Full constraints
   - Real cost function

2. **Relax one or more constraints**
   - Allow illegal moves
   - Merge distinctions between states

3. **Solve the relaxed problem**
   - Often easier: BFS, counting, geometry

4. **Use the solution cost as $h(n)$**
   - Cost of relaxed problem $\leq$ true cost

## Classic Problems with Admissible Heuristics

Admissible heuristics arise by solving a relaxed or simplified version of the original problem.

| Problem | Admissible Heuristic |
|---------|---------------------|
| **Route Planning / GPS** | Straight-line (Euclidean) distance |
| **Grid Navigation / Robotics** | Manhattan or Euclidean distance |
| **Blocks World / Planning** | Blocks out of place; Pattern databases |
| **Rubik's Cube** | Corner-only / edge-only abstractions |
| **Traveling Salesperson (TSP)** | Minimum Spanning Tree (MST); Assignment relaxation |
### Perfect Heuristic
$$
h(n)=h^{*}(n)\;\;\forall n
$$
### Null Heuristic
$$
h(n)=0\;\; \forall n
$$
### More vs Less Informed Heuristic
If $\forall n: h_1(n) \leq h_2(n) \leq h^*(n)$, then:
- $h_2$ provides more information than $h_1$
- $h_2$ better discriminates between states
**Key Idea**:
- Heuristics differ in how closely they approximate reality
- Better heuristics give stronger guidance
- Heuristics encode domain knowledge — not search strategy
- 
# Weak vs Strong Methods
**Weak methods** are general strategies applicable to many problems
- use limited domain knowledge
- called weak because they may not fully exploit the domain-specific structure
- examples
	- **means-ends analysis** compare current and goal state, pick an operation that reduces that gap the most
	- **space splitting** enumerate candidates and rule out ones that cant work
	- **subgoaling** break a large goal down into smaller subgoals

**Strong methods** are designed for a specific class of problems
- exploit rich domain structure
- often peforms better for that specific domain


## Failure Modes of Heuristic Search

### Key Observation
Heuristics provide guidance, not guarantees.

### Common Failure Modes

1. **Local Minima**
   - All neighbors have worse $h(n)$
   - Hill climbing stops prematurely

2. **Plateaus**
   - Many neighbors have identical $h(n)$
   - Search wanders randomly or stalls

3. **Ridges**
   - Best path requires temporary increase in $h(n)$
   - Greedy choices miss the correct direction

