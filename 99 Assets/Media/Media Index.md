---
cssclasses:
  - dashboard
---

# 🖼️ Media Index

Images, GIFs, videos, and PDFs used by notes.

```dataview
TABLE WITHOUT ID file.link AS "Media", file.ext AS "Type", dateformat(file.mtime, "MMM dd") AS "Modified"
FROM "99 Assets/Media"
WHERE file.name != "Media Index"
SORT file.mtime DESC
LIMIT 50
```
