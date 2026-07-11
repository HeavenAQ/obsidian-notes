---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🧩 Algorithms Hub

> [!important] Primary hub
> This hub is now maintained as a Base: [[03 Algorithms/Algorithms Hub.base|Algorithms Hub Base]].

## Main base

- 🧮 **[[03 Algorithms/DSA/DSA.base|DSA Base]]**

## 📊 DSA Progress Summary

```dataviewjs
const pages = dv.pages('"03 Algorithms/DSA"').where(p => p.file.ext === "md");
const statuses = ["To-Solve", "To-Review", "Done"];
const rows = statuses.map(s => [s, pages.where(p => p.Status === s).length]);
const total = pages.length;
const done = pages.where(p => p.Status === "Done").length;
const pct = total ? Math.round(done / total * 100) : 0;
rows.push(["Total", total]);
rows.push(["Completion", `${done}/${total} (${pct}%)`]);
dv.table(["Status", "Count"], rows);
```

## 🧮 All DSA Documents / Problems

```dataviewjs
const rows = dv.pages('"03 Algorithms/DSA"')
  .where(p => p.file.ext === "md")
  .sort(p => p.Order ?? 9999, 'asc')
  .sort(p => p["Date Added"] ?? "0000-00-00", 'desc')
  .map(p => [
    p.file.link,
    p.Status ?? "—",
    p.Difficulty ?? "—",
    p.Topic ?? "—",
    p.Type ?? "—",
    p.Category ?? "—",
    p.Source ?? "—",
    p["Due Date"] ?? "—",
    p["Date Added"] ?? "—",
    p.Link ?? "—"
  ]);

dv.table(["Document", "Status", "Difficulty", "Topic", "Type", "Category", "Source", "Due", "Added", "Link"], rows);
```

## 🗂️ Topic Index

```dataview
TABLE rows.file.link AS "Documents"
FROM "03 Algorithms/DSA"
WHERE Topic
GROUP BY Topic
SORT key ASC
```
