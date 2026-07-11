---
base: "[[DSA.base]]"
Order: 8
Type: Concept
Source: Lesson Summary
Topic:
  - Divide and Conquer
  - Recursion
Status: To Review
Category: Paradigms
---
## Overview

Solve a problem by (1) **dividing** it into smaller independent subproblems, (2) **conquering** each recursively, and (3) **combining** their results. Merge sort and quick sort are the canonical examples.

## Structure

```java
Result solve(Problem p) {
    if (small(p)) return base(p);     // base case
    Problem[] parts = divide(p);      // split
    Result a = solve(parts[0]);       // conquer
    Result b = solve(parts[1]);
    return combine(a, b);             // merge
}
```

## Complexity via the recurrence

T(n) = a·T(n/b) + f(n). Merge sort: T(n) = 2T(n/2) + O(n) = O(n log n). The depth is O(log n) when the input halves each level.

## When to use

- The problem splits into independent halves whose answers combine cheaply (sorting, closest pair, counting inversions, FFT).
- Contrast with DP, where subproblems **overlap** and must be cached; in pure divide & conquer they are disjoint.

## Pitfalls

- Combine step can dominate cost — analyze f(n).
- Overlapping subproblems mean you actually want DP/memoization, not plain recursion.