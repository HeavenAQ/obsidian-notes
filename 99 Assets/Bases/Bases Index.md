---
cssclasses:
  - dashboard
---

# 🗃️ Bases Index

Scratch or untitled Bases are kept here until promoted into a main collection.

```dataview
TABLE WITHOUT ID file.link AS "Base", dateformat(file.mtime, "MMM dd") AS "Modified"
FROM "99 Assets/Bases"
WHERE file.name != "Bases Index"
SORT file.mtime DESC
```
