---
cssclasses:
  - dashboard
---

# 📥 Inbox

Drop quick notes here first, then sort them into:

- [[00 Home/Learning Hub|Learning]]
- [[02 Research|Research]]
- [[03 Algorithms/03 Algorithms|Algorithms]]
- [[04 Reference/04 Reference|Reference]]

```dataview
TABLE file.folder AS "Folder", dateformat(file.ctime, "MMM dd") AS "Created"
FROM "00 Home/Inbox"
WHERE file.name != "Inbox"
SORT file.ctime DESC
```
