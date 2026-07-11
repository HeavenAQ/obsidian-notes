---
base: "[[DSA.base]]"
Order: 26
Type: Concept
Source: Advanced Lecture
Topic:
  - Shortest Path
  - Dynamic Programming
  - Graph
Status: To Review
Category: Advanced
---
## Overview

Computes shortest paths between **all pairs** of vertices via DP, allowing more intermediate vertices one at a time. Simple triple loop, O(V³) time, O(V²) space. Handles negative edges (no negative cycles).

## Implementation

```java
for (int k = 0; k < n; k++)          // intermediate vertex
    for (int i = 0; i < n; i++)      // source
        for (int j = 0; j < n; j++)  // destination
            if (dist[i][k] + dist[k][j] < dist[i][j])
                dist[i][j] = dist[i][k] + dist[k][j];
```

## The DP behind it

State `dp[i][j][k]` = shortest i→j path using only vertices 0..k as intermediates. Transition:

`dp[i][j][k] = min(dp[i][j][k-1], dp[i][k][k-1] + dp[k][j][k-1])` — either don't use k, or route through k.

## Why k must be the outermost loop

When we add k as a candidate, `dp[i][k]` and `dp[k][j]` must already be optimal for intermediates < k. Because k is outermost, those were finalized in earlier rounds, so the 2D in-place array (dropping the k dimension) still yields correct answers.

## Classic uses

- Dense graphs needing all-pairs distances, transitive closure (reachability), detecting negative cycles (`dist[i][i] < 0`).

## Complexity

O(V³) time, O(V²) space — fine for V up to a few hundred; use Dijkstra-per-source for sparse large graphs.

## Pitfalls

- Initialize `dist[i][i]=0`, others to a large value (guard against overflow when adding two ∞).
- k must be the outer loop — reordering breaks correctness.