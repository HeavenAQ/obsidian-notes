---
base: "[[DSA.base]]"
Order: 17
Type: Concept
Source: Lesson-Summary
Topic: Tree
Status: To-Review
Category:
  - Data-Structures
---
## Overview

A BST keeps keys ordered (left < node < right) so search/insert/delete are O(height). A **balanced** BST (red-black, AVL) guarantees height O(log n), preventing the O(n) degenerate chain. Java's `TreeMap`/`TreeSet` are red-black trees.

## Java usage (TreeMap/TreeSet)

```java
TreeMap<Integer, String> map = new TreeMap<>();
map.put(5, "a");
map.floorKey(4);    // largest key <= 4
map.ceilingKey(6);  // smallest key >= 6
map.firstKey();     // min

TreeSet<Integer> set = new TreeSet<>();
set.add(10);
set.higher(10);     // strictly greater
```

## Why balance matters

Inserting sorted data into a plain BST yields a linked list (O(n) ops). Rotations on insert/delete keep the tree height ≈ log n, so all operations stay O(log n).

## Classic uses

- Ordered map/set: range queries, floor/ceiling, predecessor/successor.
- "Find nearest value", sweep-line problems, ordered statistics.

## Complexity

search/insert/delete O(log n); in-order traversal yields sorted order in O(n).

## Pitfalls

- Don't hand-roll balancing under time pressure — use `TreeMap`/`TreeSet`.
- For pure ordered iteration without lookups, sorting an array may be simpler.