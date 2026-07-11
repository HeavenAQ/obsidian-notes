---
base: "[[DSA.base]]"
Order: 12
Type: Concept
Source: Lesson-Summary
Topic: Linked-List
Status: To-Review
Category:
  - Data-Structures
---
## Overview

A chain of nodes, each holding a value and a reference to the next node. O(1) insert/delete given the node, but O(n) random access (no indexing). Great when you insert/remove frequently and don't need indexed access.

## Node + common operations

```java
class ListNode { int val; ListNode next; ListNode(int v){ val = v; } }

ListNode reverse(ListNode head) {     // iterative reversal
    ListNode prev = null;
    while (head != null) {
        ListNode nxt = head.next;
        head.next = prev;
        prev = head;
        head = nxt;
    }
    return prev;
}
```

## Key techniques

- **Dummy head** to simplify insert/delete at the front.
- **Two pointers**: fast/slow to find the middle or detect a cycle (Floyd's).
- **Reversal** of all or part of the list.

## Complexity

Access/search O(n); insert/delete at a known node O(1). Singly vs doubly linked trades memory for O(1) backward moves.

## Pitfalls

- Losing the `next` pointer before reassigning — save it first.
- Null-pointer errors at head/tail; a dummy node removes most edge cases.