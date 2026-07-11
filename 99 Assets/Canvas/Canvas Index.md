---
cssclasses:
  - dashboard
---

# 🧩 Canvas Index

Canvas files are kept here to avoid cluttering the root.

```dataview
TABLE WITHOUT ID file.link AS "Canvas", dateformat(file.mtime, "MMM dd") AS "Modified"
FROM "99 Assets/Canvas"
WHERE file.name != "Canvas Index"
SORT file.mtime DESC
```
