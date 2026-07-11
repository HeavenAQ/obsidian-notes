---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 📚 Reference Hub

## Main base

- 🗂️ **[[04 Reference/Document Hub/Document Hub.base|Document Hub Base]]**

## In progress

```dataview
TABLE WITHOUT ID file.link AS "Document", Category, dateformat(file.mtime, "MMM dd") AS "Updated"
FROM "04 Reference/Document Hub"
WHERE Status = "In progress"
SORT file.mtime DESC
```

## By category

```dataview
TABLE rows.file.link AS "Notes"
FROM "04 Reference/Document Hub"
WHERE Category
GROUP BY Category
SORT key ASC
```

## Recently touched

```dataview
TABLE WITHOUT ID file.link AS "Document", Status, Category, dateformat(file.mtime, "MMM dd") AS "Updated"
FROM "04 Reference/Document Hub"
SORT file.mtime DESC
LIMIT 12
```
