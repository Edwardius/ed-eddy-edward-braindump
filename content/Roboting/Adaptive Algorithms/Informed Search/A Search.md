$$
f(n)=g(n)+h(n)
$$
Where $g(n)$ is the cost from the **start node to node n**
Where $h(n)$ heuristic estimate **from n to a goal**

- maintains a priority queue ordered by $f(n)$
- expands the node with the smallest $f(n)$
- combines search depth (g) and goal guidance (h)

>[!error] this is a framework, not an algorithm.

>[!error] if $h(n)$ overestimates the true remaining cost, algorithm A may expand a suboptimal path first

