---
base: "[[DSA.base]]"
Order: 19
Type: Concept
Source: Lesson Summary
Topic:
  - Trie
Status: To Review
Category: Data Structures
---
## Overview

A tree where each path from the root spells a prefix; used for fast string prefix queries. Insert/search/startsWith run in O(L) where L is the word length, independent of how many words are stored.

## Implementation

```java
class Trie {
    private final Trie[] children = new Trie[26];
    private boolean isWord;
    void insert(String w) {
        Trie node = this;
        for (char c : w.toCharArray()) {
            int i = c - 'a';
            if (node.children[i] == null) node.children[i] = new Trie();
            node = node.children[i];
        }
        node.isWord = true;
    }
    boolean search(String w) { Trie n = walk(w); return n != null && n.isWord; }
    boolean startsWith(String p) { return walk(p) != null; }
    private Trie walk(String s) {
        Trie node = this;
        for (char c : s.toCharArray()) {
            node = node.children[c - 'a'];
            if (node == null) return null;
        }
        return node;
    }
}
```

## Classic uses

- Autocomplete / prefix search, spell-check, word dictionaries.
- Word search II (trie + DFS on a grid), maximum XOR pair (bitwise trie), IP routing.

## Complexity

insert/search/startsWith O(L); space O(total characters × alphabet) — can be heavy.

## Pitfalls

- Memory cost with a fixed 26-way (or 256-way) array; use a `HashMap<Character, Node>` for sparse alphabets.
- Remember the `isWord` flag to distinguish a stored word from a mere prefix.