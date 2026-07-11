---
base: "[[DSA.base]]"
Order: 11
Type: Concept
Source: Lesson-Summary
Topic: Sorting
Status: To-Review
Category:
  - Sorting
---
## Overview

A non-comparison sort for integers in a small known range [0, k]. Count occurrences of each value, then emit values in order. O(n + k) time and O(k) space — beats the O(n log n) comparison lower bound when k is small.

## Implementation

```java
int[] countingSort(int[] a, int k) {   // values in [0, k]
    int[] count = new int[k + 1];
    for (int x : a) count[x]++;
    int[] out = new int[a.length];
    int idx = 0;
    for (int v = 0; v <= k; v++)
        while (count[v]-- > 0) out[idx++] = v;
    return out;
}
```

## Why it beats comparison sorts

Comparison sorts need Ω(n log n). Counting sort sidesteps comparisons entirely by using values as array indices, so it's linear when k = O(n).

## When to use

- Small integer range: ages, lowercase letters, scores, bucketed values.
- As a stable subroutine inside radix sort.

## Pitfalls

- Memory blows up if k is large (e.g., 32-bit ints).
- Only works for integers / things mappable to a bounded index. For stability, build a prefix-sum of counts and place from the back.