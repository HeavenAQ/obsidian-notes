---
base: "[[DSA.base]]"
Order: 21
Type: Concept
Source: Lesson-Summary
Topic: DFS
Status: To-Review
Category:
  - Graphs
---
## Overview

Explore as deep as possible along each branch before backtracking. Implement with recursion (call stack) or an explicit stack. Visits every vertex/edge once: O(V + E).

## Recursive implementation

```java
boolean[] visited;
void dfs(int u, List<List<Integer>> adj) {
    visited[u] = true;
    // process u
    for (int v : adj.get(u))
        if (!visited[v]) dfs(v, adj);
}
```

## Classic uses

- Connected components, flood fill on grids.
- Cycle detection, topological sort (post-order), bipartite check.
- Tree traversals (pre/in/post-order), path enumeration, backtracking.

## Grid DFS pattern

Treat each cell as a node with up/down/left/right neighbors; guard bounds and visited.

## Complexity

Time O(V + E); space O(V) for visited + recursion depth (can hit stack limits on huge graphs — use an explicit stack then).

## Pitfalls

- Forgetting `visited` → infinite loops on cycles.
- Deep recursion stack overflow on large/linear graphs.