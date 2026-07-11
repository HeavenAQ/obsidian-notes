---
base: "[[DSA.base]]"
Order: 22
Type: Concept
Source: Lesson-Summary
Topic: BFS
Status: To-Review
Category:
  - Graphs
---
## Overview

Explore level by level using a queue. On an **unweighted** graph, BFS finds the shortest path (fewest edges) from the source. O(V + E).

## Implementation

```java
int[] bfs(int src, List<List<Integer>> adj, int n) {
    int[] dist = new int[n];
    Arrays.fill(dist, -1);
    Queue<Integer> q = new ArrayDeque<>();
    dist[src] = 0; q.offer(src);
    while (!q.isEmpty()) {
        int u = q.poll();
        for (int v : adj.get(u))
            if (dist[v] == -1) {       // mark on enqueue
                dist[v] = dist[u] + 1;
                q.offer(v);
            }
    }
    return dist;
}
```

## Classic uses

- Shortest path in unweighted graphs / grids, level-order tree traversal.
- Multi-source BFS (push all sources first): rotting oranges, nearest 0.
- 0-1 BFS with a deque when edges are weight 0 or 1.

## Complexity

Time O(V + E); space O(V) for the queue + distance array.

## Pitfalls

- Mark visited when **enqueuing**, not dequeuing, or nodes get added multiple times.
- BFS gives shortest path only when all edge weights are equal — otherwise use Dijkstra.