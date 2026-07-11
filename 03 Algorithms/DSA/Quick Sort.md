---
base: "[[DSA.base]]"
Order: 10
Type: Concept
Source: Lesson-Summary
Topic: Sorting
Status: To-Review
Category:
  - Sorting
---
## Overview

In-place divide-and-conquer sort: pick a **pivot**, partition elements into < pivot and > pivot, then recurse on each side. Average O(n log n), worst case O(n²) (bad pivots on sorted input).

## Implementation (Lomuto partition)

```java
void quickSort(int[] a, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(a, lo, hi);
    quickSort(a, lo, p - 1);
    quickSort(a, p + 1, hi);
}
int partition(int[] a, int lo, int hi) {
    int pivot = a[hi], i = lo;
    for (int j = lo; j < hi; j++)
        if (a[j] < pivot) swap(a, i++, j);
    swap(a, i, hi);
    return i;
}
void swap(int[] a, int x, int y) { int t = a[x]; a[x] = a[y]; a[y] = t; }
```

## Complexity

- Average: O(n log n) (balanced partitions).
- Worst: O(n²) (already-sorted with last-element pivot) — mitigate with randomized or median-of-three pivot.
- Space: O(log n) average recursion depth, in-place.

## When to use

- General-purpose in-memory sorting where average speed and low memory matter (it's `Arrays.sort` for primitives via dual-pivot quicksort).

## Pitfalls

- Not stable.
- Pick pivots carefully (randomize) to avoid O(n²) on adversarial input.