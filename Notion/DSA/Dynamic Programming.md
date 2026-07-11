---
base: "[[DSA.base]]"
Order: 6
Type: Concept
Source: Lesson Summary
Topic:
  - Dynamic Programming
Status: To Review
Category: Paradigms
---
## Overview

DP solves problems with **overlapping subproblems** and **optimal substructure** by computing each subproblem once and reusing it. Two equivalent styles: top-down (recursion + memoization) and bottom-up (fill a table).

## The 4-step recipe

1. **State**: what does dp[i] (or dp[i][j]) mean? Define it precisely.
2. **Transition**: how is a state built from smaller states?
3. **Base case**: smallest states' values.
4. **Answer**: which state holds the final result, and in what iteration order.

## Example: 0/1 Knapsack

```java
int knapsack(int[] w, int[] v, int W) {
    int[] dp = new int[W + 1];                 // dp[c] = best value with capacity c
    for (int i = 0; i < w.length; i++)
        for (int c = W; c >= w[i]; c--)        // iterate capacity descending (0/1)
            dp[c] = Math.max(dp[c], dp[c - w[i]] + v[i]);
    return dp[W];
}
```

## Top-down vs bottom-up

- **Top-down**: write the recurrence naturally, cache results in a `memo` array/map. Easy to reason about.
- **Bottom-up**: fill the table in dependency order; lets you do space optimization (rolling array, as above where the 2D table collapses to 1D).

## Complexity

Usually (number of states) × (cost per transition). Knapsack: O(nW) time; O(W) space after rolling.

## Pitfalls

- Wrong iteration order (0/1 knapsack must loop capacity descending; unbounded loops ascending).
- State that isn't self-contained → missing a dimension.
- Forgetting base cases.