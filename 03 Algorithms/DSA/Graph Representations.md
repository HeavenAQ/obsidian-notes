---
base: "[[DSA.base]]"
Order: 20
Type: Concept
Source: Lesson-Summary
Topic: Graph
Status: To-Review
Category:
  - Graphs
---
## Overview

A graph is a set of vertices connected by edges (directed/undirected, weighted/unweighted). The two standard representations trade memory for lookup speed.

## Adjacency List (default choice)

Each vertex stores a list of its neighbors. Space O(V + E), iterate neighbors in O(deg). Best for sparse graphs (most real graphs).

```java
List<List<Integer>> adj = new ArrayList<>();
for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
adj.get(u).add(v);
adj.get(v).add(u);   // undirected
```

## Adjacency Matrix

`m[i][j]` = 1/weight if edge i→j. Space O(V²), edge lookup O(1). Best for dense graphs or when you frequently test "is there an edge i–j?".

```java
int[][] m = new int[n][n];
m[u][v] = w;   // m[v][u] = w for undirected
```

## Choosing

- Sparse (E ≪ V²) → adjacency list.
- Dense or need O(1) edge tests / all-pairs DP (Floyd–Warshall) → matrix.

## Pitfalls

- For weighted graphs store `int[]{neighbor, weight}` in the list.
- Remember to add both directions for undirected graphs.