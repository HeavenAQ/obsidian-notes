---
cssclasses:
  - dashboard
---

# 🗄️ Archive

Inactive or starter material lives here so the main file tree stays focused.

```dataview
TABLE WITHOUT ID file.link AS "Archived note", dateformat(file.mtime, "MMM dd") AS "Updated"
FROM "90 Archive"
WHERE file.name != "Archive Index"
SORT file.mtime DESC
```
