---
base: "[[DSA.base]]"
Order: 24
Type: Concept
Source: Lesson Summary
Topic:
  - Shortest Path
  - Graph
  - Heap
Status: To Review
Category: Graphs
---
## Overview

Single-source shortest paths on a graph with **non-negative** edge weights. Greedily settle the closest unfinished vertex using a min-heap. O((V + E) log V).

## Implementation

```java
int[] dijkstra(int src, List<int[]>[] adj, int n) {  // adj[u] = {v, w}
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
    pq.offer(new int[]{src, 0});
    while (!pq.isEmpty()) {
        int[] cur = pq.poll();
        int u = cur[0], d = cur[1];
        if (d > dist[u]) continue;            // stale entry, skip
        for (int[] e : adj[u]) {
            int v = e[0], nd = d + e[1];
            if (nd < dist[v]) { dist[v] = nd; pq.offer(new int[]{v, nd}); }
        }
    }
    return dist;
}
```

## Why non-negative weights

Dijkstra finalizes a vertex when popped, assuming no later path can be shorter — negative edges break that. For negative edges use Bellman–Ford; for all-pairs use Floyd–Warshall.

## Complexity

O((V + E) log V) with a binary heap. Skip stale heap entries with the `d > dist[u]` check.

## Pitfalls

- Updating priorities: push a new entry and ignore outdated ones.
- Using `int` distances can overflow when summing large weights — use `long` if needed.