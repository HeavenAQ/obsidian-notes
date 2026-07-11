---
base: "[[DSA.base]]"
Order: 25
Type: Concept
Source: Advanced Lecture
Topic:
  - Segment Tree
Status: To Review
Category: Advanced
---
## Overview

A tree that supports **range queries** (sum, max, min, ...) and **point updates** both in O(log n). It fixes prefix sum's two weaknesses: non-invertible queries (max/min) and updates. Idea: split the array into segments; each node stores the merged answer for its segment, and a query combines O(log n) segments.

## Why it works

The tree halves the range at each level, so height is O(log n). A query touches at most two nodes per level (a third middle node would already be merged by a parent), so query/update are O(log n).

## Implementation (max, pointer style from the lecture)

```java
class Node { Node left, right; int lo, hi, maxVal; Node(int lo, int hi){ this.lo=lo; this.hi=hi; } }

void pull(Node n){ n.maxVal = Math.max(n.left.maxVal, n.right.maxVal); }

void build(Node n, int[] v) {
    if (n.lo == n.hi) { n.maxVal = v[n.lo]; return; }
    int mid = (n.lo + n.hi) / 2;
    n.left = new Node(n.lo, mid);
    n.right = new Node(mid + 1, n.hi);
    build(n.left, v); build(n.right, v);
    pull(n);
}

int queryMax(Node n, int l, int r) {
    if (n.lo >= l && n.hi <= r) return n.maxVal;          // fully inside
    if (n.hi < l || n.lo > r) return Integer.MIN_VALUE;   // fully outside
    return Math.max(queryMax(n.left, l, r), queryMax(n.right, l, r));
}

void update(Node n, int idx, int val) {
    if (n.lo == n.hi) { n.maxVal = val; return; }
    int mid = (n.lo + n.hi) / 2;
    if (idx <= mid) update(n.left, idx, val); else update(n.right, idx, val);
    pull(n);
}
```

## Classic uses

- Range max/min/sum with updates, range GCD, count of smaller numbers, with lazy propagation for range updates.

## Complexity

build O(n), query/update O(log n), space O(n).

## Pitfalls

- Return the identity element for the "fully outside" case (MIN_VALUE for max, 0 for sum).
- Range **updates** need lazy propagation — a plain segment tree only does point updates efficiently.