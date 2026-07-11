---
base: "[[DSA.base]]"
Order: 3
Type: Concept
Source: Lesson Summary
Topic:
  - Binary Search
Status: To Review
Category: Searching
---
## Overview

Search a **sorted/monotonic** space by repeatedly halving it: compare the middle, then discard the half that cannot contain the answer. O(log n) time, O(1) space.

## Classic (find target index)

```java
int binarySearch(int[] a, int target) {
    int lo = 0, hi = a.length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;   // avoid overflow
        if (a[mid] == target) return mid;
        else if (a[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

## The TTTT...FFFF view (binary search on the answer)

Many problems map to a boolean predicate that is monotonic: true,true,...,true,false,...,false. Binary search finds the **boundary** — the last T or first F. This generalizes to "minimize the largest", "can we finish in x time", etc.

```java
int firstFalse(int lo, int hi) {       // find boundary on [lo, hi]
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (check(mid)) lo = mid + 1;  // mid is T, answer is to the right
        else hi = mid;                 // mid is F, keep it as candidate
    }
    return lo;
}
```

## Implementation tips

- Use `lo + (hi - lo) / 2` to avoid integer overflow.
- Decide invariants up front: is `hi` inclusive (`length-1`) or exclusive (`length`)? Keep it consistent.
- For lower/upper bound, mirror `Arrays.binarySearch`/`Collections` or write the boundary form above.

## Pitfalls

- Off-by-one and infinite loops from wrong `mid` rounding when `lo`/`hi` move by 0.
- Requires a monotonic predicate — verify it before applying.