---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 📚 Reference Hub

> [!important] Primary hub
> This hub is now maintained as a Base: [[04 Reference/Reference Hub.base|Reference Hub Base]].

## Main base

- 🗂️ **[[04 Reference/Document Hub/Document Hub.base|Document Hub Base]]**

## 📊 Reference Progress Summary

```dataviewjs
const pages = dv.pages('"04 Reference/Document Hub"').where(p => p.file.ext === "md");
const statuses = ["In progress", "Not started", "Done"];
const rows = statuses.map(s => [s, pages.where(p => p.Status === s).length]);
const total = pages.length;
const done = pages.where(p => p.Status === "Done").length;
const pct = total ? Math.round(done / total * 100) : 0;
rows.push(["Total", total]);
rows.push(["Completion", `${done}/${total} (${pct}%)`]);
dv.table(["Status", "Count"], rows);
```

## 📚 All Reference Documents

```dataviewjs
const rows = dv.pages('"04 Reference/Document Hub"')
  .where(p => p.file.ext === "md")
  .sort(p => p["Last updated time"] ?? p.file.mtime, 'desc')
  .map(p => [
    p.file.link,
    p.Status ?? "—",
    p.Category ?? "—",
    p["Created time"] ?? "—",
    p["Last updated time"] ?? "—",
    dv.date(p.file.mtime).toFormat("MMM dd, HH:mm")
  ]);

dv.table(["Document", "Status", "Category", "Created", "Last updated", "File modified"], rows);
```

## 🗂️ By Category

```dataview
TABLE rows.file.link AS "Documents"
FROM "04 Reference/Document Hub"
WHERE Category
GROUP BY Category
SORT key ASC
```
