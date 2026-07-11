---
base: "[[DSA.base]]"
Order: 4
Type: Concept
Source: Lesson Summary
Topic:
  - Sliding Window
  - Two Pointers
Status: To Review
Category: Searching
---
## Overview

Maintain a contiguous window [left, right] over an array/string and slide it to answer subarray/substring questions in O(n) instead of O(n²). Expand `right` to include elements; shrink `left` when the window violates a constraint.

## Template (longest window satisfying a condition)

```java
int longest(int[] a) {
    int left = 0, best = 0;
    // window state, e.g. counts/sum, goes here
    for (int right = 0; right < a.length; right++) {
        // add a[right] to window state
        while (/* window invalid */ false) {
            // remove a[left] from window state
            left++;
        }
        best = Math.max(best, right - left + 1);
    }
    return best;
}
```

## Two variants

- **Variable-size**: grow `right`, shrink `left` while invalid (longest substring without repeats, min subarray ≥ target).
- **Fixed-size k**: slide a window of constant width, updating state in O(1) per step (max average of length k).

## Why O(n)

Each index enters the window once (via `right`) and leaves at most once (via `left`), so total work is O(n) amortized even with the inner `while`.

## Pitfalls

- Works when the metric is monotonic as the window grows/shrinks (e.g., counts). If not, sliding window may be invalid.
- Remember to update the answer at the right moment (after shrinking for "longest", possibly during shrinking for "shortest").