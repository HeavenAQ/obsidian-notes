---
base: "[[DSA.base]]"
Order: 14
Type: Concept
Source: Lesson Summary
Topic:
  - Queue
Status: To Review
Category: Data Structures
---
## Overview

FIFO (first-in, first-out) container; enqueue at the back, dequeue from the front, both O(1). A **deque** allows O(1) operations at both ends and powers sliding-window-maximum.

## Java usage

```java
Queue<Integer> q = new ArrayDeque<>();
q.offer(1); q.offer(2);
int front = q.peek();   // 1
q.poll();               // removes 1

Deque<Integer> dq = new ArrayDeque<>();  // double-ended
dq.offerFirst(0); dq.offerLast(3);
```

## Classic uses

- **BFS** level-order traversal of graphs/trees.
- **Monotonic deque** for sliding window maximum/minimum in O(n).
- Producer/consumer and scheduling.

## Complexity

All core ops O(1) with `ArrayDeque` (array-backed ring buffer). Avoid `LinkedList` unless you need it as a list too.

## Pitfalls

- `ArrayDeque` forbids null elements.
- For BFS, mark nodes visited when you enqueue (not when you dequeue) to avoid duplicates.