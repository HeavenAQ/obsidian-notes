---
base: "[[DSA.base]]"
Order: 15
Type: Concept
Source: Lesson Summary
Topic:
  - Hash Table
Status: To Review
Category: Data Structures
---
## Overview

Stores key→value pairs with **average O(1)** insert/lookup/delete via a hash function that maps keys to buckets. The workhorse for de-duplication, counting, and fast membership tests.

## Java usage

```java
Map<String, Integer> count = new HashMap<>();
for (String w : words)
    count.merge(w, 1, Integer::sum);     // frequency map

Set<Integer> seen = new HashSet<>();
if (!seen.add(x)) { /* x is a duplicate */ }

count.getOrDefault("a", 0);
```

## How it works

A hash maps the key to a bucket index; collisions are handled by chaining (lists/trees) or open addressing. Load factor triggers resizing to keep operations O(1) amortized.

## Classic uses

- Two-sum (value → index), grouping anagrams, frequency counts, caching/memoization, detecting cycles/duplicates.

## Complexity

Average O(1) per op; worst case O(n) if many collisions (Java treeifies large buckets to O(log n)). Unordered — use `TreeMap` for sorted keys or `LinkedHashMap` for insertion order.

## Pitfalls

- Custom key objects must override `equals` and `hashCode` consistently.
- Don't rely on iteration order of `HashMap`.