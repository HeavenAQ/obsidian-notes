---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🧩 Algorithms Hub

## Main base

- 🧮 **[[03 Algorithms/DSA/DSA.base|DSA Base]]**

## Practice queue

```dataview
TABLE WITHOUT ID file.link AS "To solve", Difficulty, Topic, Type, Link
FROM "03 Algorithms/DSA"
WHERE Status = "To Solve"
SORT Difficulty DESC, Topic ASC
LIMIT 12
```

```dataview
TABLE WITHOUT ID file.link AS "Review", Difficulty, Topic, Type
FROM "03 Algorithms/DSA"
WHERE Status = "To Review"
SORT Topic ASC
LIMIT 12
```

## Topic index

```dataview
TABLE rows.file.link AS "Notes"
FROM "03 Algorithms/DSA"
WHERE Topic
GROUP BY Topic
SORT key ASC
```
