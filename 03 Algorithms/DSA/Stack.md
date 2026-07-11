---
base: "[[DSA.base]]"
Order: 13
Type: Concept
Source: Lesson-Summary
Topic: Stack
Status: To-Review
Category:
  - Data-Structures
---
## Overview

LIFO (last-in, first-out) container. Push/pop/peek are all O(1). Models nesting and "most recent unmatched" relationships.

## Java usage

```java
Deque<Integer> stack = new ArrayDeque<>();  // preferred over legacy Stack
stack.push(1);
stack.push(2);
int top = stack.peek();   // 2
stack.pop();              // removes 2
boolean empty = stack.isEmpty();
```

## Classic uses

- Balanced parentheses / valid brackets.
- Expression evaluation (infix → postfix, RPN).
- **Monotonic stack**: next greater/smaller element, largest rectangle in histogram, daily temperatures.
- Iterative DFS and call-stack simulation.

## Monotonic stack idea

Keep elements in increasing (or decreasing) order; when a new element breaks the order, pop and resolve those elements — each index is pushed/popped once → O(n).

## Pitfalls

- Prefer `ArrayDeque` over `java.util.Stack` (the latter is synchronized and slower).
- Don't `pop` an empty stack — check `isEmpty()`.