---
base: "[[DSA.base]]"
Order: 16
Type: Concept
Source: Lesson Summary
Topic:
  - Heap
Status: To Review
Category: Data Structures
---
## Overview

A binary heap is a complete binary tree where each parent is ≤ (min-heap) or ≥ (max-heap) its children. It gives O(log n) insert and extract-min/max, and O(1) peek. Java exposes it as `PriorityQueue`.

## Java usage

```java
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
minHeap.offer(5); minHeap.offer(1);
int smallest = minHeap.peek();  // 1
minHeap.poll();

// custom comparator (e.g., by second field)
PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
```

## Classic uses

- Top-K elements, K closest points, merge K sorted lists.
- Dijkstra / Prim (extract the current best frontier).
- Running median (two heaps), task scheduling.

## Complexity

peek O(1); offer/poll O(log n); building from n items O(n) via heapify. Backed by an array using index math (children of i at 2i+1, 2i+2).

## Pitfalls

- `PriorityQueue` is **not** sorted on iteration — only the head is ordered.
- Updating a key's priority isn't supported directly; push a new entry and skip stale ones, or use an indexed heap.