---
base: "[[DSA.base]]"
Order: 23
Type: Concept
Source: Lesson Summary
Topic:
  - Union-Find
  - Graph
Status: To Review
Category: Graphs
---
## Overview

Maintains a partition of elements into disjoint sets with two near-O(1) operations: `find(x)` (which set?) and `union(a, b)` (merge sets). With **path compression** + **union by rank/size**, operations are O(α(n)) — effectively constant.

## Implementation

```java
class DSU {
    int[] parent, rank;
    DSU(int n) {
        parent = new int[n]; rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]); // path compression
        return parent[x];
    }
    boolean union(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;                      // already connected
        if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        return true;
    }
}
```

## Classic uses

- Connected components / number of islands (incremental), cycle detection in undirected graphs.
- Kruskal's MST, account merging, redundant connection, equations consistency.

## Complexity

Nearly O(1) amortized per op (inverse Ackermann α). Space O(n).

## Pitfalls

- Always call `find` to get roots before comparing/merging.
- Forgetting path compression / union by rank degrades to O(n).