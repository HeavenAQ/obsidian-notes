---
base: "[[DSA.base]]"
Order: 7
Type: Concept
Source: Lesson-Summary
Topic: Backtracking
Status: To-Review
Category:
  - Paradigms
---
## Overview

Systematically explore all candidate solutions by building them incrementally and **undoing** (backtracking) a choice when it can't lead to a valid/complete solution. It is DFS over the space of partial solutions, with pruning.

## Template (subsets/permutations/combinations)

```java
void backtrack(List<Integer> path, int start, int[] nums, List<List<Integer>> res) {
    res.add(new ArrayList<>(path));            // record current partial solution
    for (int i = start; i < nums.length; i++) {
        path.add(nums[i]);                     // choose
        backtrack(path, i + 1, nums, res);     // explore
        path.remove(path.size() - 1);          // un-choose (backtrack)
    }
}
```

## The 3 moves

1. **Choose** a candidate and apply it to the state.
2. **Explore** by recursing.
3. **Un-choose** to restore state before trying the next candidate.

## Pruning

Cut branches that cannot succeed (constraint violated, bound exceeded). Good pruning turns exponential search into something tractable (N-Queens, Sudoku).

## Complexity

Often exponential (e.g., O(2^n) subsets, O(n!) permutations) — the point is correctness + pruning, not polynomial time.

## Pitfalls

- Forgetting to undo the choice corrupts sibling branches.
- Copy the path when recording (`new ArrayList<>(path)`), otherwise all results alias one list.
- Handle duplicates with sorting + skip conditions.