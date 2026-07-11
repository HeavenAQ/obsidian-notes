---
cssclasses:
  - dashboard
---

# 📥 Inbox

Drop quick notes here first, then sort them into:

- [[01 Learning/Learning Hub|Learning]]
- [[02 Research/Research Hub|Research]]
- [[03 Algorithms/Algorithms Hub|Algorithms]]
- [[04 Reference/Reference Hub|Reference]]

```dataview
TABLE file.folder AS "Folder", dateformat(file.ctime, "MMM dd") AS "Created"
FROM "00 Home/Inbox"
WHERE file.name != "Inbox"
SORT file.ctime DESC
```
