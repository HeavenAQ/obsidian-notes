---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🖼️ Assets Index

Assets are intentionally separated from notes so the file tree stays readable.

## Folders

- **[[99 Assets/Media/Media Index|99 Assets/Media]]** — images, GIFs, videos, PDFs
- **[[99 Assets/Canvas/Canvas Index|99 Assets/Canvas]]** — canvas files
- **[[99 Assets/Bases/Bases Index|99 Assets/Bases]]** — scratch/untitled base files

## Recent media

```dataview
TABLE WITHOUT ID file.link AS "Asset", file.ext AS "Type", dateformat(file.mtime, "MMM dd, HH:mm") AS "Modified"
FROM "99 Assets"
SORT file.mtime DESC
LIMIT 20
```
