---
base: "[[DSA.base]]"
Order: 2
Type: Concept
Source: Lesson Summary
Topic:
  - Recursion
Status: To Review
Category: Fundamentals
---
## Overview

A function that calls itself. Two things must be designed: a **base case** (stops the recursion at the smallest input) and the **recursive step** (combine the result of smaller subproblems). The key mindset: *trust that the recursive call returns the correct answer for the smaller input*, then build your answer from it.

## Example: sum 1..n

```java
int sum(int n) {
    if (n == 1) return 1;        // base case
    return sum(n - 1) + n;       // trust sum(n-1) is correct
}
```

## How to reason about it

1. **Base case** — the smallest input you can answer directly (often a single element; a 1-element array is already sorted).
2. **Recursive case** — reduce to one or more smaller calls, then merge their results (e.g., merge sort splits into two halves, sorts each, then merges).

## Complexity

Model with a recurrence. T(n) = 2T(n/2) + O(n) → O(n log n) (merge sort). T(n) = T(n-1) + O(1) → O(n). Each pending call uses stack space, so depth d costs O(d) auxiliary space.

## Pitfalls

- Missing/incorrect base case → stack overflow.
- Recomputing the same subproblem repeatedly → exponential blow-up (fix with memoization → DP).
- Deep recursion on large n may overflow the call stack; convert to iteration if needed.