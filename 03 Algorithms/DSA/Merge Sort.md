---
base: "[[DSA.base]]"
Order: 9
Type: Concept
Source: Lesson-Summary
Topic: Sorting
Status: To-Review
Category:
  - Sorting
---
## Overview

A stable, divide-and-conquer sort: split the array in half, sort each half recursively, then **merge** the two sorted halves. Guaranteed O(n log n) time, O(n) auxiliary space.

## Implementation

```java
void mergeSort(int[] a, int l, int r) {
    if (l >= r) return;                 // base: 0 or 1 element
    int mid = l + (r - l) / 2;
    mergeSort(a, l, mid);
    mergeSort(a, mid + 1, r);
    merge(a, l, mid, r);
}
void merge(int[] a, int l, int mid, int r) {
    int[] tmp = new int[r - l + 1];
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= r)   tmp[k++] = a[j++];
    System.arraycopy(tmp, 0, a, l, tmp.length);
}
```

## Why O(n log n)

T(n) = 2T(n/2) + O(n). log n levels, O(n) merge work per level.

## When to use

- Need **stability** (equal keys keep order) or guaranteed worst case.
- Counting inversions, external sorting, linked-list sorting (O(1) extra space on lists).

## Pitfalls

- Uses O(n) extra memory (unlike in-place quicksort).
- Use `<=` in the merge comparison to stay stable.