---
base: "[[DSA.base]]"
Order: 5
Type: Concept
Source: Lesson Summary
Topic:
  - Greedy
Status: To Review
Category: Paradigms
---
## Overview

Build a solution by always taking the locally optimal choice, never reconsidering. Greedy is fast (often O(n log n) after sorting) but only correct when the problem has the **greedy-choice property** and **optimal substructure**.

## Example: interval scheduling (max non-overlapping)

Sort by end time, always take the earliest-finishing compatible interval.

```java
int maxIntervals(int[][] iv) {
    Arrays.sort(iv, (x, y) -> Integer.compare(x[1], y[1]));
    int count = 0, end = Integer.MIN_VALUE;
    for (int[] i : iv) {
        if (i[0] >= end) { count++; end = i[1]; }
    }
    return count;
}
```

## How to trust a greedy

- **Greedy-choice property**: a globally optimal solution can be reached by local optimal choices.
- **Exchange argument**: prove that swapping the greedy choice into any optimal solution does not make it worse.

## Greedy vs DP

DP explores all combinations of choices (safe but slower); greedy commits immediately (fast but needs proof). If you can't prove the greedy, fall back to DP.

## Pitfalls

- A plausible greedy is often wrong (e.g., coin change with arbitrary denominations needs DP).
- Always test against brute force on small inputs before trusting it.