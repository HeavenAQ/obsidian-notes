---
base: "[[DSA.base]]"
Order: 18
Type: Concept
Source: Lesson-Summary
Topic: Prefix-Sum
Status: To-Review
Category:
  - Data-Structures
---
## Overview

Precompute cumulative sums so any range sum is answered in O(1). `pre[i] = a[0] + ... + a[i-1]`, then `sum(l..r) = pre[r+1] - pre[l]`. Build in O(n), then each query is O(1).

## Implementation

```java
int[] pre = new int[a.length + 1];
for (int i = 0; i < a.length; i++)
    pre[i + 1] = pre[i] + a[i];
int rangeSum = pre[r + 1] - pre[l];   // sum of a[l..r] inclusive
```

## Extensions

- **2D prefix sum** for submatrix sums (inclusion-exclusion).
- **Prefix XOR** for range XOR.
- **Hashing prefix sums** for "subarray sum equals k" (count `pre - k` seen so far).
- **Difference array** (inverse idea) for many range updates + one final scan.

## Limitation → Segment Tree

Prefix sum handles invertible operations (sum, XOR). It **cannot** do range max/min (no subtraction) or updates efficiently — that's where a Segment Tree is needed.

## Pitfalls

- Off-by-one with the size-(n+1) array and inclusive/exclusive bounds.
- Doesn't support updates well (rebuild is O(n)).