---
base: "[[DSA.base]]"
Order: 1
Type: Concept
Source: Lesson-Summary
Topic: Complexity
Status: To-Review
Category:
  - Fundamentals
---
## Overview

Complexity describes how an algorithm's resource use **grows** as the input size *n* grows. It ignores constant factors and lower-order terms, so it captures the growth *trend*, not the exact runtime. Two O(n) algorithms can differ 3x in real speed but still share the same growth rate.

## Time Complexity

- Sequential blocks **add**: O(n) + O(n) = O(n).
- Nested loops **multiply**: a loop running n times, each doing O(n) work, is O(n²).
- Repeatedly halving the input gives O(log n): if n/2^k = 1 then k = log₂ n. This is why binary search is O(log n).

Decompose the program, compute each part, then combine:

```java
for (int i = 0; i < n; i++)      // O(n)
    for (int j = 0; j < n; j++)  // O(n) each
        sum += i + j;            // => O(n^2)
```

## Space Complexity

Split into (1) input space and (2) **auxiliary** space (everything except the input). For "sum an array of length n": input is O(n), auxiliary is O(1) (one accumulator). Always state auxiliary space separately — recursion stacks count.

## Common growth rates

O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!).

## Why it matters / pitfalls

- Constants are dropped but matter in practice; profile when n is small.
- Hidden costs: `String` concatenation in a loop, `ArrayList.contains` (O(n)), recursion stack depth.
- Always know the input bound: n ≤ 10^5 usually rules out O(n²).