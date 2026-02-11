Maintains a priority queue of nodes to traverse ordered by total cost from the root node to the node in question.

```pseudo
function UCS(start, goal):
    frontier = priority queue ordered by path cost
    frontier.add(start, cost=0)
    explored = empty set

    while frontier is not empty:
        node, cost = frontier.pop_min()
        if node == goal: return solution
        explored.add(node)
        for each neighbor, edge_cost of node:
            new_cost = cost + edge_cost
            if neighbor not in explored and not in frontier:
                frontier.add(neighbor, new_cost)
            elif neighbor in frontier with higher cost:
                replace with new_cost
    return failure
```

